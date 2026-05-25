#  Phase 1: Environment Setup & GIS Foundations

This phase sets up the core project structure, installs necessary dependencies, and creates the most critical GIS component: the script to generate a **0.5 km (500 meter) buffer** around the river to define the illegal mining zone.

---

## 1. Project Structure Setup (Completed)

We have created the following directory structure in your workspace:

```text
Sand-Mining_Detection/
 data/
    raw/                  # For raw drone imagery/video
    processed/            # For orthomosaics and extracted frames
    annotations/          # For YOLO label txt files
    legal_zones/          # For river shapefiles and generated buffers
 models/
    weights/              # For trained YOLO .pt and .onnx files
 notebooks/                # For Jupyter notebooks (exploration)
 src/
    preprocess/           # Data preparation and GIS scripts
    detection/            # YOLO inference, tracking, clustering
    reporting/            # Report generation (JSON/PDF)
    dashboard/
        frontend/         # Web UI
 requirements.txt          # Python dependencies
```

---

## 2. Dependencies (`requirements.txt`)

We need a specific set of libraries, primarily focusing on spatial data processing for this phase.

```txt
# requirements.txt
# Geospatial Processing
geopandas==0.14.3
shapely==2.0.3
pyproj==3.6.1
folium==0.15.1      # For quick map visualization

# ML and Vision (for upcoming phases)
ultralytics==8.1.0  # YOLOv8
opencv-python-headless==4.9.0.80
scikit-learn==1.4.1.post1 # For DBSCAN clustering
numpy==1.26.4
```

> **Action:** Install these by running `pip install -r requirements.txt` in your terminal.

---

## 3. The Zone Builder Script (`src/preprocess/zone_builder.py`)

This script takes a river boundary (e.g., a GeoJSON or Shapefile) and creates a 500-meter buffer around it. 

**Crucial Concept**: GPS coordinates (WGS84 / EPSG:4326) are in degrees. To accurately buffer by exactly 500 *meters*, we must temporarily project the map into a local Metric CRS (Coordinate Reference System), apply the buffer, and project it back to GPS coordinates.

### Code Implementation

```python
# src/preprocess/zone_builder.py
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
    # EPSG:3857 is Web Mercator (used by Google Maps), measured in meters.
    # Note: For highly precise local surveying, you would use a specific UTM zone,
    # but 3857 is generally acceptable for a 500m buffer.
    print("Projecting to metric CRS...")
    gdf_metric = gdf.to_crs("EPSG:3857")
    
    # 3. Create the buffer
    print(f"Applying {buffer_meters}m buffer...")
    # .buffer() applies to the geometry column
    buffered_metric = gdf_metric.geometry.buffer(buffer_meters)
    
    # Create a new GeoDataFrame with the buffered geometry
    buffered_gdf = gpd.GeoDataFrame(geometry=buffered_metric, crs="EPSG:3857")
    
    # 4. Project back to WGS84 (Lat/Lon)
    print("Projecting back to GPS coordinates (WGS84)...")
    final_gdf = buffered_gdf.to_crs("EPSG:4326")
    
    # 5. Save the output
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as GeoJSON for easy web integration (Leaflet) later
    final_gdf.to_file(out_file, driver="GeoJSON")
    print(f" Success! Buffer saved to: {out_file}")

if __name__ == "__main__":
    # Define paths
    # IMPORTANT: You need to place a river shapefile or geojson here first!
    INPUT_RIVER_FILE = "../../data/legal_zones/river_centerline.geojson" 
    OUTPUT_BUFFER_FILE = "../../data/legal_zones/river_buffer_500m.geojson"
    
    # Check if input exists before running
    if Path(INPUT_RIVER_FILE).exists():
        create_river_buffer(
            input_path=INPUT_RIVER_FILE, 
            output_path=OUTPUT_BUFFER_FILE, 
            buffer_meters=500
        )
    else:
        print(f" Error: Could not find input file at {INPUT_RIVER_FILE}")
        print("Please place your river boundary file there and update the script if the name differs.")
```

---

## 4. Map Visualization (`notebooks/01_visualize_zones.ipynb` snippet)

Once you generate the buffer, you'll want to verify it looks correct. We'll use `folium`.

```python
import folium
import geopandas as gpd

# Load the generated buffer
buffer_gdf = gpd.read_file("../data/legal_zones/river_buffer_500m.geojson")

# Get center coordinates to center the map
center_lat = buffer_gdf.geometry.centroid.y.mean()
center_lon = buffer_gdf.geometry.centroid.x.mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

# Add the buffer polygon to the map
folium.GeoJson(
    buffer_gdf,
    name="0.5km Illegal Zone",
    style_function=lambda feature: {
        'fillColor': 'red',
        'color': 'red',
        'weight': 2,
        'fillOpacity': 0.2,
    }
).add_to(m)

# Display the map (in a notebook, just type 'm')
m.save("zone_map.html")
```

---

## Next Steps for You

1.  Create a virtual environment: `python -m venv venv` and activate it.
2.  Run `pip install -r requirements.txt`.
3.  **Crucial Task**: Find or draw a `river_centerline.geojson` file for your target testing area and place it in `data/legal_zones/`. (You can draw one quickly using [geojson.io](https://geojson.io/)).
4.  Run `python src/preprocess/zone_builder.py` to generate your 500m buffer.
