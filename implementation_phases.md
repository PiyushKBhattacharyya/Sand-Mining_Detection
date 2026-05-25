#  Implementation Phases: Illegal Sand Mining Detection

This document breaks down the high-level roadmap into actionable, code-level implementation steps. Each phase produces a tangible deliverable.

---

## Phase 1: Environment Setup & GIS Foundations (Week 1)

**Objective**: Set up the foundational environment and generate the critical 0.5km legal boundary.

1.  **Project Initialization**:
    *   Create project structure as defined in the plan.
    *   Initialize Git repository.
    *   Create `requirements.txt` (Geopandas, Shapely, Ultralytics, scikit-learn, etc.) and virtual environment.
2.  **GIS Data Acquisition**:
    *   Download or create a sample river centerline/boundary shapefile (e.g., `river_centerline.geojson`).
3.  **Buffer Generation Script (`src/preprocess/zone_builder.py`)**:
    *   **Input**: River centerline/polygon.
    *   **Action**: Use `geopandas` to apply a 500-meter buffer.
    *   **Output**: `river_buffer_500m.geojson`.
4.  **Verification**:
    *   Visualize the generated buffer on a base map (e.g., using QGIS or a simple Folium script) to ensure it correctly surrounds the river at the right distance.

**Deliverable**: A working Python script that generates the legal boundary and the resulting `.geojson` file.

---

## Phase 2: Data Pipeline & Model Training (Weeks 2-4)

**Objective**: Prepare the dataset and train the YOLOv8 object detection model.

1.  **Dataset Preparation**:
    *   Acquire aerial imagery (drone flights or open datasets like DOTA/VisDrone).
    *   Annotate images (Trucks, JCBs, People) using Roboflow or CVAT.
    *   Export dataset in YOLO format (`dataset.yaml` + images/labels folders).
2.  **Model Training Script (`src/detection/train.py`)**:
    *   Initialize `yolov8m.pt`.
    *   Configure training parameters (epochs, imgsz=640, augmentations).
    *   Execute training loop.
3.  **Evaluation & Export**:
    *   Evaluate model performance on the validation set (mAP metrics).
    *   Export the best weights to ONNX format (`yolov8m_sandmining.onnx`) for faster edge inference.
4.  **Inference Script (`src/detection/detector.py`)**:
    *   Write a script to load the trained model and run inference on test images/videos.
    *   Extract bounding boxes, classes, and confidence scores.

**Deliverable**: Trained YOLOv8 model weights (`.pt` and `.onnx`) and a working inference script.

---

## Phase 3: Spatial Analysis & Clustering (Weeks 4-5)

**Objective**: Map detections to real-world coordinates and group them into actionable incident clusters.

1.  **GPS Projection Script (`src/detection/gps_projector.py`)**:
    *   **Input**: Bounding box center pixels, drone telemetry (GPS, altitude, gimbal).
    *   **Action**: Implement the pixel-to-GPS math using camera intrinsics.
    *   **Output**: Estimated Latitude/Longitude for each detection.
    *   *(Note: If using orthomosaics, this script will extract coordinates directly from the GeoTIFF).*
2.  **Zone Checking Script (`src/detection/zone_checker.py`)**:
    *   **Input**: Detection Lat/Lon, `river_buffer_500m.geojson`.
    *   **Action**: Use `shapely.geometry.Point` to check if the point is within the buffer.
3.  **Clustering Engine (`src/detection/cluster_engine.py`)**:
    *   **Input**: List of all detection coordinates in a single frame/area.
    *   **Action**: Apply DBSCAN (eps=50m) to group nearby detections.
    *   **Output**: Cluster IDs assigned to detections. Determine cluster severity based on the types of objects present (e.g., Truck + JCB = High).

**Deliverable**: Python modules for coordinate projection, zone verification, and spatial clustering.

---

## Phase 4: Reporting & Dashboard Integration (Weeks 6-7)

**Objective**: Package the analysis into human-readable reports and a visual dashboard.

1.  **Report Generator (`src/reporting/report_generator.py`)**:
    *   **Input**: Clustered detections, severity, zone status.
    *   **Action**: Generate the structured JSON incident report.
2.  **Backend API (`src/dashboard/app.py`)**:
    *   Set up a FastAPI server.
    *   Create endpoints to receive incident JSON data and serve it to the frontend.
3.  **Frontend Dashboard (`src/dashboard/frontend/`)**:
    *   Create a basic HTML/JS page using Leaflet.js.
    *   Load the `river_buffer_500m.geojson` onto the map.
    *   Fetch incidents from the API and plot them as color-coded pins (Red=Critical, Yellow=Warning).
4.  **PDF/KML Exporter (Optional but recommended)**:
    *   Scripts to convert the JSON report into a downloadable PDF summary or a KML file for Google Earth.

**Deliverable**: A functional FastAPI backend and Leaflet map dashboard visualizing the incidents.

---

## Phase 5: Pipeline Integration & Testing (Week 8)

**Objective**: Connect all components into a seamless, end-to-end automated pipeline.

1.  **Main Pipeline Script (`main.py`)**:
    *   Create a master script that orchestrates the flow:
        `Video Frame -> YOLO Detection -> GPS Projection -> Zone Check -> Clustering -> Report Generation -> API Update`.
2.  **System Testing**:
    *   Run the pipeline on a complete, unseen drone flight video/dataset.
    *   Verify coordinate accuracy against known landmarks.
    *   Verify the dashboard updates correctly.
3.  **Optimization**:
    *   Profile the code. If inference is slow, integrate SAHI (Slicing Aided Hyper Inference) for better small object detection or optimize the ONNX runtime.

**Deliverable**: The finalized, end-to-end working system ready for field deployment.
