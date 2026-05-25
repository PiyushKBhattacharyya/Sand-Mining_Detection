"""
Zone Builder  shared utility for generating the river buffer GeoJSON.
Used by main.py (startup), app.py (API endpoint), and cluster_engine.py (hot-reload).
"""
import logging
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CENTERLINE   = PROJECT_ROOT / "data" / "legal_zones" / "river_centerline.geojson"
BUFFER_OUT   = PROJECT_ROOT / "data" / "legal_zones" / "river_buffer_1km.geojson"


def build_buffer(radius_m = 1000.0, output_path = None):
    """
    Generates a river buffer GeoJSON polygon at the given radius (metres).
    Saves to output_path (default: river_buffer_1km.geojson) and returns True on success.
    Thread-safe: reads centerline, writes output.
    """
    import geopandas as gpd
    warnings.filterwarnings("ignore", category=UserWarning)

    out = Path(output_path) if output_path else BUFFER_OUT

    if not CENTERLINE.exists():
        logger.warning("Centerline GeoJSON not found  cannot build buffer.")
        return False

    try:
        gdf = gpd.read_file(str(CENTERLINE))
        if str(gdf.crs) != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        # Project to UTM 46N (metres)  buffer  back to WGS84
        gdf_m = gdf.to_crs("EPSG:32646")
        buf   = gdf_m.geometry.buffer(radius_m, cap_style=2, join_style=1, resolution=32)
        buf_gdf = gpd.GeoDataFrame(
            geometry=buf,
            crs="EPSG:32646"
        ).to_crs("EPSG:4326")

        # Embed radius in properties so readers know what radius was used
        buf_gdf["radius_m"] = radius_m
        buf_gdf.to_file(str(out), driver="GeoJSON")

        n_pts = len(list(gdf.iloc[0].geometry.coords))
        logger.info(f"Buffer rebuilt: {radius_m:.0f}m | {n_pts} centerline pts  {out.name}")
        return True

    except Exception as e:
        logger.error(f"Buffer build failed: {e}")
        return False
