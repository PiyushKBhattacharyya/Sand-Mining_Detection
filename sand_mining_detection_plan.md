#  Illegal Sand Mining Detection via Drone  Technical Project Plan

> **Scope**: Detect illegal activity within **0.5 km of riverbed**  specifically clusters of **Trucks, JCBs, and People**  and auto-report their **GPS coordinates**.

---

## 1. Core Detection Targets

| Target | Why It Matters | Detection Method |
|---|---|---|
| **Illegal Zone Violation** | Mining within 0.5 km of river is prohibited | GIS buffer overlay on river shapefile |
| **Trucks (Dumpers)** | Primary sand transport vehicle | YOLOv8 object detection |
| **JCBs / Excavators** | Active excavation machinery | YOLOv8 object detection |
| **People / Workers** | Presence confirms active operation | YOLOv8 object detection |
| **Activity Clusters** | Trucks + JCBs + People co-located = high-confidence illegal site | DBSCAN spatial clustering |

---

## 2. Data Requirements

### A. Drone Imagery (Primary Input)

| Data Type | Spec | Purpose |
|---|---|---|
| **RGB Video / Images** |  4K, GSD  5 cm/px | Object detection (trucks, JCBs, people) |
| **GPS Telemetry Log** | RTK-GPS (2 cm) preferred | Accurate coordinate projection |
| **Gimbal Angles** | Yaw, Pitch, Roll per frame | Correct pixel-to-GPS math |
| **Altitude AGL** | From barometer/rangefinder | GSD calculation |
| **Timestamp** | Per frame (ISO 8601) | Temporal correlation |

### B. GIS Reference Data (Required for Zone Detection)

| Data | Source | Format |
|---|---|---|
| **River Centerline / Boundary** | Survey of India, NRSC Bhuvan, OSM | SHP / GeoJSON |
| **0.5 km Buffer Zone** | Generated from river shapefile | Polygon SHP |
| **Legal Mining Permit Zones** | State Mining Dept. | KML / SHP |
| **Administrative Boundaries** | Census / OSM | SHP |

> **Key step**: Pre-generate a 0.5 km buffer polygon around the river using GeoPandas. Any detection whose coordinates fall inside this buffer is flagged as **ILLEGAL**.

### C. Training Labels (ML Model)

- Annotate aerial images with bounding boxes for: `truck`, `jcb`, `person`
- Tools: **Roboflow** (recommended) or **CVAT**
- Format: **YOLO TXT** (for YOLOv8 training)
- Target: **5001000 labeled instances per class** minimum

### D. Existing Datasets to Bootstrap Training

| Dataset | Source | Use |
|---|---|---|
| DOTA v2.0 | captain-whu.github.io/DOTA | Aerial vehicles (cars, trucks) |
| VisDrone | VisDrone benchmark | People + vehicles from drones |
| Roboflow Universe | universe.roboflow.com | Search "aerial truck", "construction vehicles" |
| OpenAerialMap | openaerialmap.org | Free drone orthophotos for background |

---

## 3. System Architecture

```
[Drone in Flight]
    
     RGB Camera (4K) + RTK-GPS + IMU/Gimbal
     Edge Compute (optional: Jetson Nano for live detection)
            
             Raw footage + telemetry log
[Ground Station / Cloud Processing]
    
     1. Telemetry Parser       GPS + gimbal per frame
     2. YOLOv8 Detector        bboxes: truck / jcb / person
     3. GPS Projector          pixel bbox  lat/lon
     4. Zone Checker           inside 0.5 km buffer?  ILLEGAL
     5. DBSCAN Cluster Engine  group nearby detections into sites
     6. Report Generator       JSON + PDF + KML map pins
            
            
[Dashboard  Leaflet.js Map]
     Red pins = illegal clusters
     Blue boundary = 0.5 km river buffer
     Downloadable PDF incident reports
```

---

## 4. ML Pipeline

### Model: YOLOv8 (Ultralytics)

**Classes**: `truck` | `jcb` | `person`

**Training Config**
```yaml
model: yolov8m.pt
data: data/dataset.yaml
epochs: 100
imgsz: 640
batch: 16
augment: true
degrees: 15.0      # rotation augment for aerial view
flipud: 0.5
mosaic: 1.0
```

**Performance Targets**
- mAP@0.5  0.80 for all three classes
- Inference:  100ms/frame (GPU),  500ms (Jetson Nano)
- Use **SAHI** (Sliced Inference) for small object detection at high altitude

### Cluster Detection (DBSCAN)

```python
from sklearn.cluster import DBSCAN
import numpy as np

def cluster_detections(detections, eps_meters=50, min_samples=2):
    """
    detections: list of {'lat', 'lon', 'class', 'confidence'} dicts
    eps_meters: max distance between points in same cluster
    """
    coords = np.array([[d['lat'], d['lon']] for d in detections])
    eps_deg = eps_meters / 111320  # convert meters to degrees
    labels = DBSCAN(eps=eps_deg, min_samples=min_samples).fit_predict(coords)

    clusters = {}
    for i, label in enumerate(labels):
        if label == -1: continue  # noise / isolated detection
        clusters.setdefault(label, []).append(detections[i])
    return clusters
```

A cluster with **truck + JCB** or **3+ people** near the river = **HIGH SEVERITY**.

---

## 5. Coordinate Reporting

### Step 1  Pixel to GPS Projection

```python
from math import radians, cos

def pixel_to_gps(bbox_center_px, drone_gps, altitude_m,
                 focal_length_mm, img_size_px, sensor_mm=(13.2, 8.8)):
    """Returns (lat, lon) for center of a detected bounding box."""
    GSD_x = (sensor_mm[0] * altitude_m) / (focal_length_mm * img_size_px[0])
    GSD_y = (sensor_mm[1] * altitude_m) / (focal_length_mm * img_size_px[1])

    dx = (bbox_center_px[0] - img_size_px[0] / 2) * GSD_x  # meters east
    dy = (bbox_center_px[1] - img_size_px[1] / 2) * GSD_y  # meters south

    lat = drone_gps[0] - (dy / 111320)
    lon = drone_gps[1] + (dx / (111320 * cos(radians(drone_gps[0]))))
    return round(lat, 7), round(lon, 7)
```

>  **Better accuracy**: Use WebODM to generate a georeferenced orthomosaic (GeoTIFF). Any pixel in the ortho has a direct EPSG coordinate  no math needed, sub-meter accuracy.

### Step 2  Zone Check

```python
import geopandas as gpd
from shapely.geometry import Point

river_buffer = gpd.read_file("data/legal_zones/river_buffer_500m.shp")

def is_in_illegal_zone(lat, lon):
    point = Point(lon, lat)  # Shapely uses (lon, lat)
    return river_buffer.geometry.contains(point).any()
```

### Step 3  Incident Report Schema (JSON)

```json
{
  "incident_id": "SM-2026-001",
  "timestamp": "2026-05-15T09:30:00+05:30",
  "flight_id": "FLIGHT-20260515-001",
  "cluster_id": 3,
  "severity": "HIGH",
  "illegal_zone": true,
  "distance_from_river_m": 210,
  "centroid_coordinates": {
    "latitude": 25.345612,
    "longitude": 83.123456,
    "crs": "WGS84",
    "accuracy_m": 2.0
  },
  "detections": [
    { "class": "truck",  "confidence": 0.92, "lat": 25.345580, "lon": 83.123410 },
    { "class": "jcb",    "confidence": 0.89, "lat": 25.345640, "lon": 83.123500 },
    { "class": "person", "confidence": 0.87, "lat": 25.345600, "lon": 83.123480 }
  ],
  "evidence": {
    "drone_frame": "data/raw/frame_04521.jpg",
    "annotated_image": "reports/SM-2026-001_annotated.jpg"
  }
}
```

---

## 6. Project Structure

```
Sand-Mining_Detection/

 data/
    raw/                    # Raw drone frames + telemetry logs
    processed/              # Orthomosaics (GeoTIFF), tiled frames
    annotations/            # YOLO .txt labels + dataset.yaml
    legal_zones/
        river_centerline.shp
        river_buffer_500m.shp    PRE-GENERATE THIS FIRST

 models/
    weights/
        yolov8m_sandmining.pt    # Fine-tuned weights
        yolov8m_sandmining.onnx  # Edge deployment

 src/
    preprocess/
       telemetry_parser.py   # GPS/gimbal from .srt/.csv logs
       frame_extractor.py    # Video  frames
       zone_builder.py       # Build 0.5km river buffer SHP
   
    detection/
       detector.py           # YOLOv8 inference (+ SAHI tiling)
       gps_projector.py      # Pixel  GPS
       zone_checker.py       # Is point in illegal zone?
       cluster_engine.py     # DBSCAN clustering
   
    reporting/
       report_generator.py   # Build JSON incident reports
       pdf_exporter.py       # PDF with images + map
       kml_exporter.py       # KML for Google Earth
   
    dashboard/
        app.py                # FastAPI backend
        frontend/             # Leaflet.js map UI

 notebooks/
    01_zone_builder.ipynb
    02_model_training.ipynb
    03_evaluation.ipynb

 requirements.txt
```

---

## 7. Phased Roadmap (8 Weeks)

### Phase 1  GIS Setup & Zone Definition (Week 1)
- [ ] Download river shapefile for target area (NRSC Bhuvan / OSM)
- [ ] Generate 0.5 km buffer polygon using GeoPandas
- [ ] Visually verify buffer on Google Maps
- [ ] Map legal mining permit zones if available

### Phase 2  Data Collection & Annotation (Week 23)
- [ ] Plan drone survey grid (Mission Planner / DJI Pilot 2)
- [ ] Fly river area at 5080m AGL, 4K resolution
- [ ] Extract frames at 2 fps
- [ ] Annotate 300500 images per class on Roboflow
- [ ] Export in YOLO format

### Phase 3  Model Training (Week 34)
- [ ] Fine-tune YOLOv8m (start from DOTA/VisDrone pretrain)
- [ ] Target: mAP@0.5  0.80 per class
- [ ] Export ONNX for edge deployment
- [ ] Integrate SAHI for small object handling

### Phase 4  Coordinate & Zone Pipeline (Week 45)
- [ ] Build `gps_projector.py`, test vs. known ground truth
- [ ] Integrate `zone_checker.py` with river buffer SHP
- [ ] Build `cluster_engine.py`  tune eps=50m, min_samples=2
- [ ] End-to-end test: video  detections  GPS  zone flag  JSON

### Phase 5  Reporting & Dashboard (Week 56)
- [ ] Auto-generate PDF incident reports (ReportLab)
- [ ] KML export for Google Earth
- [ ] FastAPI backend serving detection results API
- [ ] Leaflet.js dashboard: buffer overlay + color-coded cluster pins

### Phase 6  Field Test & Validation (Week 78)
- [ ] Live field drone flight over known site
- [ ] Measure coordinate accuracy vs. handheld GPS (target:  5m)
- [ ] Measure false positive rate, tune confidence thresholds
- [ ] Generate final validation report

---

## 8. Severity Classification

| Severity | Trigger |
|---|---|
|  **CRITICAL** | JCB + Truck + People cluster inside 0.5 km zone |
|  **HIGH** | JCB or Truck + People inside 0.5 km zone |
|  **MEDIUM** | Vehicles only (no people) inside 0.5 km zone |
|  **LOW** | People only, no machinery, inside 0.5 km zone |
|  **INFO** | Any detection beyond 0.5 km  log only |

---

## 9. Key Tools & Libraries

| Category | Tool |
|---|---|
| **Object Detection** | Ultralytics YOLOv8 + SAHI |
| **Geospatial** | GeoPandas, Shapely, pyproj, GDAL, Rasterio |
| **Clustering** | scikit-learn (DBSCAN) |
| **Photogrammetry** | WebODM / OpenDroneMap |
| **Dashboard** | Leaflet.js + FastAPI |
| **Report Export** | ReportLab (PDF), simplekml (KML) |
| **Annotation** | Roboflow |
| **Edge Compute** | NVIDIA Jetson + ONNX Runtime |

---

## 10. Key Challenges & Mitigations

| Challenge | Mitigation |
|---|---|
| Small objects at altitude | Fly  80m AGL; use SAHI sliced inference |
| GPS projection error | RTK-GPS + WebODM georeferenced ortho |
| JCB vs. other machinery | Domain-specific training data from Indian riverbeds |
| People detection at altitude | Fine-tune on VisDrone dataset |
| Vegetation occlusion | Fly during low-vegetation season; use NIR if available |
| Regulatory compliance | Operate under DGCA NPNT framework |

---

*Plan v2.0 | 2026-05-15 | Scope: Illegal Zone + Cluster Detection + Coordinate Reporting*
