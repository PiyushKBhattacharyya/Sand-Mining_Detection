import geopandas as gpd
from pathlib import Path
import pyproj
import warnings

# Suppress geometry warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def create_river_buffer(input_path: str, output_path: str, buffer_meters: int = 500):
    """
    Reads a river shapefile/geojson, creates a buffer of N meters, 
    and saves the resulting polygon.
    """
    print(f"Loading river geometry from: {input_path}")
    
    # 1. Load the data
    gdf = gpd.read_file(input_path)
    
    # Ensure it's in WGS84 (Standard GPS Lat/Lon)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
        
    # 2. Project to a Metric CRS to accurately calculate meters
    # EPSG:32646 is WGS 84 / UTM zone 46N, which is extremely precise for Guwahati, Assam.
    # It preserves shape, area, and ellipsoidal distance metrics perfectly without distortions.
    print("Projecting to local UTM Zone 46N (EPSG:32646) metric CRS...")
    gdf_metric = gdf.to_crs("EPSG:32646")
    
    # 3. Create the buffer
    print(f"Applying {buffer_meters}m buffer...")
    # .buffer() applies to the geometry column
    buffered_metric = gdf_metric.geometry.buffer(buffer_meters)
    
    # Create a new GeoDataFrame with the buffered geometry
    buffered_gdf = gpd.GeoDataFrame(geometry=buffered_metric, crs="EPSG:32646")
    
    # 4. Project back to WGS84 (Lat/Lon)
    print("Projecting back to GPS coordinates (WGS84)...")
    final_gdf = buffered_gdf.to_crs("EPSG:4326")
    
    # 5. Save the output
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as GeoJSON for easy web integration (Leaflet) later
    final_gdf.to_file(out_file, driver="GeoJSON")
    print(f"Success! Buffer saved to: {out_file}")

if __name__ == "__main__":
    # Define paths
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    INPUT_RIVER_FILE = os.path.join(project_root, "data", "legal_zones", "river_centerline.geojson")
    OUTPUT_BUFFER_FILE = os.path.join(project_root, "data", "legal_zones", "river_buffer_1km.geojson")
    
    # Check if input exists before running
    if Path(INPUT_RIVER_FILE).exists():
        create_river_buffer(
            input_path=INPUT_RIVER_FILE, 
            output_path=OUTPUT_BUFFER_FILE, 
            buffer_meters=1000
        )
    else:
        print(f"❌ Error: Could not find input file at {INPUT_RIVER_FILE}")
        print("Please place your river boundary file there and update the script if the name differs.")
