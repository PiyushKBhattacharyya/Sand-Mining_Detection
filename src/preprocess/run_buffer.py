import geopandas as gpd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

project_root = Path(r"D:\Projects\Sand-Mining_Detection")
INPUT  = project_root / "data" / "legal_zones" / "river_centerline.geojson"
OUTPUT = project_root / "data" / "legal_zones" / "river_buffer_1km.geojson"

print(f"Loading: {INPUT}")
gdf = gpd.read_file(INPUT)
if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326")

print("Projecting to UTM Zone 46N (EPSG:32646)...")
gdf_metric = gdf.to_crs("EPSG:32646")

print("Buffering 1000m...")
buffered = gdf_metric.geometry.buffer(1000, cap_style=2, join_style=1)  # flat ends, mitered joins
buf_gdf = gpd.GeoDataFrame(geometry=buffered, crs="EPSG:32646")

print("Reprojecting to WGS84...")
final = buf_gdf.to_crs("EPSG:4326")

final.to_file(OUTPUT, driver="GeoJSON")
print(f"Saved: {OUTPUT}")
