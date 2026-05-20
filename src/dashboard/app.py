import os
import json
import base64
import logging
import threading
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import asyncio
import sys

# Import database manager
sys.path.append(str(Path(__file__).resolve().parent.parent / "preprocess"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "reporting"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "detection"))
from db_setup import DatabaseManager
from pdf_generator import generate_incident_report
from zone_builder import build_buffer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Illegal Sand Mining Drone Surveillance Server",
    description="Real-time Edge-Cloud Pipeline with Dual Dashboard feeds and spatial queries"
)

# ── Video source config ───────────────────────────────────────────────────────
# Controls what feeds the two dashboard video windows.
# cv2.VideoCapture() accepts BOTH an integer (webcam) and a string URL (RTSP),
# so switching from webcam to drone requires only changing this env var.
#
# WEBCAM  (default, testing):   VIDEO_SOURCE=0       ← built-in Mac/Windows webcam
#                                VIDEO_SOURCE=1       ← second/external webcam
#
# DRONE   (when hardware arrives):
#   DJI via phone relay (DJI MSDK v5 RTSP relay app on Android):
#                                VIDEO_SOURCE=rtsp://192.168.1.50:8554/live
#   DJI direct RTSP (some models expose this natively):
#                                VIDEO_SOURCE=rtsp://192.168.42.1:554/live
#   Any generic RTSP drone/IP camera:
#                                VIDEO_SOURCE=rtsp://<camera-ip>:<port>/<path>
#
# Set via environment variable before starting the server:
#   export VIDEO_SOURCE="rtsp://192.168.1.50:8554/live"
#   python main.py server
#
_raw_source  = os.getenv("VIDEO_SOURCE", "0")
# Auto-detect: if the value is a plain integer string → webcam index, else RTSP URL
VIDEO_SOURCE: int | str = int(_raw_source) if _raw_source.lstrip("-").isdigit() else _raw_source

CAMERA_FPS     = float(os.getenv("CAMERA_FPS",     "15.0"))
CAMERA_QUALITY = int(os.getenv("CAMERA_QUALITY",   "75"))

# RTSP-specific tuning (only relevant when VIDEO_SOURCE is a URL)
# Prefer TCP transport for reliability over Wi-Fi (default is UDP which can drop frames)
RTSP_TRANSPORT = os.getenv("RTSP_TRANSPORT", "tcp")   # "tcp" | "udp"


def _video_capture_loop():
    """
    Background daemon thread: opens the configured video source and continuously
    pushes JPEG frames into latest_raw_frame / latest_overlay_frame.

    Source is controlled by the VIDEO_SOURCE env var:
      • Integer  → local webcam (testing mode)
      • URL str  → RTSP stream from drone or IP camera (field deployment)

    No restart needed — just change VIDEO_SOURCE and reboot the server.
    """
    global latest_raw_frame, latest_overlay_frame, latest_webcam_detections

    try:
        import cv2
    except ImportError:
        logger.warning("opencv-python not installed — video feed disabled.")
        return

    is_rtsp = isinstance(VIDEO_SOURCE, str)

    if is_rtsp:
        # ── DRONE / RTSP MODE ──────────────────────────────────────────────
        # Force TCP transport for stable Wi-Fi streaming (avoids UDP packet loss).
        # GStreamer pipeline string can be swapped here for Jetson hardware decode.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{RTSP_TRANSPORT}"
        logger.info(f"🛸  Drone RTSP stream opening: {VIDEO_SOURCE}  (transport={RTSP_TRANSPORT})")
        logger.info("    Waiting for drone to broadcast... (this may take a few seconds)")
        cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)
    else:
        # ── WEBCAM MODE (default, no drone yet) ───────────────────────────
        logger.info(f"📷  Webcam capture starting on camera index {VIDEO_SOURCE}...")
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        # Request a reasonable resolution — driver will use nearest supported
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        if is_rtsp:
            logger.warning(
                f"⚠️  Could not connect to RTSP stream: {VIDEO_SOURCE}\n"
                "    Check that the drone is powered on, broadcasting, and on the same network.\n"
                "    Falling back to no feed — dashboard will show placeholder."
            )
        else:
            logger.warning(
                f"⚠️  Could not open camera {VIDEO_SOURCE}. "
                "Try a different index via VIDEO_SOURCE env var."
            )
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_label = f"RTSP drone stream" if is_rtsp else f"webcam {VIDEO_SOURCE}"
    logger.info(f"✅  {source_label} opened at {w}x{h} — feeding both dashboard streams.")

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, CAMERA_QUALITY]
    interval = 1.0 / CAMERA_FPS

    while True:
        t0 = time.time()
        ret, frame = cap.read()

        if not ret:
            if is_rtsp:
                # RTSP can drop temporarily (drone banking, interference) — keep retrying
                logger.warning("⚠️  RTSP frame drop — retrying...")
                time.sleep(0.1)
            else:
                time.sleep(0.05)
            continue

        _, buf = cv2.imencode(".jpg", frame, encode_params)
        jpeg = buf.tobytes()

        # Raw feed: clean, unprocessed frame from the camera
        latest_raw_frame = jpeg

        # ── DETECTION HOOK ────────────────────────────────────────────────
        # Person-only detection (COCO class 0).
        # Replace _yolo_model with your custom model by dropping best.pt
        # into models/weights/ — it auto-loads at startup.
        if _yolo_model is not None:
            try:
                results = _yolo_model(
                    frame,
                    verbose=False,
                    classes=[0],    # 0 = person in COCO; swap for your custom class IDs later
                    conf=0.30,      # confidence threshold — lower = catches more detections
                    iou=0.45,
                )
                overlay = results[0].plot()   # annotated BGR numpy array
                _, obuf = cv2.imencode(".jpg", overlay, encode_params)
                latest_overlay_frame = obuf.tobytes()

                # Extract and store bounding box details globally for hybrid telemetry mapping
                active_dets = []
                if len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        coords = box.xyxy[0].tolist()
                        conf = float(box.conf[0].item())
                        active_dets.append({
                            'class_name': 'person',
                            'confidence': conf,
                            'bbox_x_min': int(coords[0]),
                            'bbox_y_min': int(coords[1]),
                            'bbox_x_max': int(coords[2]),
                            'bbox_y_max': int(coords[3])
                        })
                latest_webcam_detections = active_dets
            except Exception as exc:
                logger.debug(f"YOLO inference error: {exc}")
                latest_overlay_frame = jpeg   # fallback: show raw if inference crashes
                latest_webcam_detections = []
        else:
            latest_overlay_frame = jpeg   # no model loaded — mirror raw feed
            latest_webcam_detections = []

        elapsed = time.time() - t0
        sleep_for = interval - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

# Mount the project's data directory so the frontend can directly load spatial GeoJSON files
app.mount("/data", StaticFiles(directory=str(Path(__file__).resolve().parent.parent.parent / "data")), name="data")

# Evidence directory — also served as static for UI image display
EVIDENCE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "detections"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

# Connect to database
db_manager = DatabaseManager(db_type="sqlite")
# Ensure DB is initialized
db_manager.initialize_database()

# Active buffer radius — starts at 1km, updated via /api/zone/radius
active_buffer_radius_m: float = 1000.0

# Store active websocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                # Handle broken connections silently during broadcast
                pass

manager = ConnectionManager()

# Global frames storage for live multipart streaming
# In a physical deployment, the Jetson Nano continuously POSTs frames here,
# which are then distributed to the dashboard HTML IMG tags.
latest_raw_frame: bytes = b""
latest_overlay_frame: bytes = b""
latest_webcam_detections: List[Dict[str, Any]] = []

# Holds the loaded YOLO model — set once at startup, used in _video_capture_loop
_yolo_model = None

# Global ClusterEngine for runtime buffer size synchronization
global_cluster_engine = None

async def _webcam_telemetry_simulation_loop():
    global latest_webcam_detections, db_manager, active_buffer_radius_m, global_cluster_engine
    
    # Wait for the app to start up fully
    await asyncio.sleep(2.0)
    
    from cluster_engine import ClusterEngine
    import random
    import math
    from datetime import datetime
    
    # Initialize server-side cluster engine and save to global variable
    global_cluster_engine = ClusterEngine(db_manager=db_manager)
    # Ensure buffer is in sync with UI slider on startup
    global_cluster_engine.set_radius(active_buffer_radius_m)
    
    # Load and interpolate centerline
    centerline_path = Path(__file__).resolve().parent.parent.parent / "data" / "legal_zones" / "river_centerline.geojson"
    if not centerline_path.exists():
        logger.error(f"Centerline not found for hybrid simulation: {centerline_path}")
        return
        
    try:
        with open(centerline_path, 'r') as f:
            cl_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading centerline for hybrid simulation: {e}")
        return
        
    raw_coords = cl_data['features'][0]['geometry']['coordinates']
    flight_points = []
    speed_mps = 42.0 / 3.6
    
    for i in range(len(raw_coords) - 1):
        lon1, lat1 = raw_coords[i]
        lon2, lat2 = raw_coords[i+1]
        lat_mid = (lat1 + lat2) / 2.0
        dy = (lat2 - lat1) * 111320
        dx = (lon2 - lon1) * 111320 * math.cos(math.radians(lat_mid))
        distance = math.sqrt(dx**2 + dy**2)
        # Generate steps at 3 Hz
        steps = max(10, int(distance / (speed_mps / 3.0)))
        
        for step in range(steps):
            t = step / steps
            interp_lon = lon1 + (lon2 - lon1) * t
            interp_lat = lat1 + (lat2 - lat1) * t
            
            weave_phase = (i * steps + step) * 0.05
            # Weave up to 1800m laterally so the drone continuously flies IN and OUT of the buffer zone
            lateral_offset_meters = math.sin(weave_phase) * 1800.0
            
            heading = math.atan2(dy, dx)
            perp_angle = heading + math.pi / 2.0
            
            offset_lat = (lateral_offset_meters * math.sin(perp_angle)) / 111320
            offset_lon = (lateral_offset_meters * math.cos(perp_angle)) / (111320 * math.cos(math.radians(interp_lat)))
            
            final_lat = interp_lat + offset_lat
            final_lon = interp_lon + offset_lon
            heading_deg = (90.0 - math.degrees(heading)) % 360.0
            
            flight_points.append({
                'lat': final_lat,
                'lon': final_lon,
                'heading': heading_deg
            })
            
    logger.info(f"🚀 Hybrid simulation generated {len(flight_points)} waypoints.")
    
    step = 0
    battery = 100.0
    
    insert_telemetry_sql = """
    INSERT INTO telemetry_logs (
        timestamp, latitude, longitude, altitude_agl, 
        gimbal_pitch, gimbal_yaw, gimbal_roll, 
        drone_speed, battery_percentage, gps_accuracy_m
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    
    while True:
        try:
            if not flight_points:
                await asyncio.sleep(1.0)
                continue
                
            point_idx = step % len(flight_points)
            point = flight_points[point_idx]
            
            lat, lon = point['lat'], point['lon']
            alt = 70.0 + random.uniform(-1.0, 1.0)
            speed = speed_mps
            heading = point['heading']
            battery = max(0.0, battery - 0.02)
            if battery <= 0:
                battery = 100.0
                
            timestamp = datetime.now().isoformat()
            
            # Save Telemetry Locally to database
            loop = asyncio.get_event_loop()
            
            def save_telemetry_db():
                conn = db_manager.get_connection()
                cursor = conn.cursor()
                cursor.execute(insert_telemetry_sql, (
                    timestamp, lat, lon, alt, -80.0, heading, 0.0, speed, int(battery), 0.15
                ))
                t_id = cursor.lastrowid
                conn.commit()
                cursor.close()
                conn.close()
                return t_id
                
            telemetry_id = await loop.run_in_executor(None, save_telemetry_db)
            
            # Broadcast telemetry to WebSockets
            telemetry_payload = {
                "type": "telemetry",
                "payload": {
                    "timestamp": timestamp,
                    "lat": lat,
                    "lon": lon,
                    "altitude": alt,
                    "speed": speed,
                    "battery": int(battery)
                }
            }
            await manager.broadcast(telemetry_payload)
            
            # ── Process active webcam detections mapped to this GPS location! ──
            active_dets = list(latest_webcam_detections)
            if active_dets:
                mapped_dets = []
                for idx, det in enumerate(active_dets):
                    # Add a tiny random coordinate offset on the ground (e.g. up to 30m)
                    offset_lat = random.uniform(-0.0002, 0.0002)
                    offset_lon = random.uniform(-0.0002, 0.0002)
                    
                    mapped_dets.append({
                        'class_name': det['class_name'],
                        'confidence': det['confidence'],
                        'bbox_x_min': det['bbox_x_min'],
                        'bbox_y_min': det['bbox_y_min'],
                        'bbox_x_max': det['bbox_x_max'],
                        'bbox_y_max': det['bbox_y_max'],
                        'lat': lat + offset_lat,
                        'lon': lon + offset_lon
                    })
                    
                # Run cluster engine on mapped detections
                raw_incidents = global_cluster_engine.cluster_detections(mapped_dets, eps_meters=60.0)
                
                # FILTER: Only keep incidents inside the enforcement boundaries (illegal_zone == 1)
                incidents = [inc for inc in raw_incidents if inc.get('illegal_zone', 0) == 1]
                
                if incidents:
                    # Save crop snapshot of the webcam feed as dynamic evidence!
                    try:
                        import cv2
                        import numpy as np
                        from evidence_engine import save_incident_evidence
                        
                        frame_bytes = latest_overlay_frame
                        if frame_bytes:
                            nparr = np.frombuffer(frame_bytes, np.uint8)
                            frame_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if frame_np is not None:
                                for inc in incidents:
                                    # Mock the incident id for the file naming format
                                    inc['incident_id'] = step + 5000
                                    inc['detections'] = mapped_dets
                                    
                                    evidence_paths = save_incident_evidence(
                                        annotated_frame=frame_np,
                                        incident=inc,
                                        telemetry={'lat': lat, 'lon': lon}
                                    )
                                    if evidence_paths:
                                        inc['evidence_image_path'] = evidence_paths[0]
                    except Exception as e:
                        logger.error(f"Error generating hybrid evidence snapshot: {e}")

                    # Save incidents to DB
                    def save_incidents_db():
                        global_cluster_engine.save_incidents_to_db(incidents, telemetry_log_id=telemetry_id)
                    await loop.run_in_executor(None, save_incidents_db)
                    
                    # Broadcast detection warning alerts immediately
                    for inc in incidents:
                        await manager.broadcast({
                            "type": "detections",
                            "payload": {
                                "incident_id": step + 5000,
                                "severity": inc['severity'],
                                "centroid_latitude": inc['centroid_lat'],
                                "centroid_longitude": inc['centroid_lon'],
                                "detections": inc['detections']
                            }
                        })
            
            step += 1
            # Run at 3 Hz (approx 0.33s per step)
            await asyncio.sleep(0.33)
            
        except Exception as e:
            logger.error(f"Error in hybrid telemetry loop: {e}")
            await asyncio.sleep(1.0)


@app.on_event("startup")
async def startup_event():
    """
    Fires once when uvicorn starts.
    1. Loads the YOLO detection model (custom best.pt if available, else yolov8n.pt).
    2. Launches the video capture thread so both dashboard feed windows go live.
    """
    global _yolo_model

    # ── Load YOLO model ───────────────────────────────────────────────────────
    # Priority: custom trained weights → generic YOLOv8n placeholder
    custom_weights = Path(__file__).resolve().parent.parent.parent / "models" / "weights" / "best.pt"
    try:
        from ultralytics import YOLO

        if custom_weights.exists():
            _yolo_model = YOLO(str(custom_weights))
            logger.info(f"✅  Loaded CUSTOM YOLO model: {custom_weights.name}")
        else:
            # Auto-downloads yolov8n.pt on first run (~6 MB) — already cached
            _yolo_model = YOLO("yolov8n.pt")
            logger.info("🤖  YOLOv8n placeholder loaded — detecting PERSON ONLY (conf≥0.30). Swap best.pt when ready.")
    except Exception as e:
        logger.warning(f"⚠️  YOLO failed to load — overlay will mirror raw feed. Error: {e}")
        _yolo_model = None

    # ── Start video capture thread AFTER model is ready ────────────────
    # Ensures first frames already have a model to run against.
    t = threading.Thread(target=_video_capture_loop, daemon=True, name="video-capture")
    t.start()
    source_desc = f"RTSP: {VIDEO_SOURCE}" if isinstance(VIDEO_SOURCE, str) else f"webcam {VIDEO_SOURCE}"
    logger.info(f"🎥  Video capture thread launched ({source_desc}) — dashboard feeds will populate shortly.")

    # Start the hybrid telemetry simulation loop if we are using the webcam (testing mode)
    if isinstance(VIDEO_SOURCE, int):
        asyncio.create_task(_webcam_telemetry_simulation_loop())
        logger.info("🛰️ Launched dynamic hybrid flight telemetry simulator background task.")

# Frame generator for multipart MJPEG streaming
async def frame_generator(stream_type: str):
    global latest_raw_frame, latest_overlay_frame
    
    # 1. Fallback dummy frame if no feed is active
    # A simple 1x1 black pixel JPEG byte representation
    dummy_pixel = b'\xff\xd8\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9'

    while True:
        frame = latest_raw_frame if stream_type == "raw" else latest_overlay_frame
        
        # If no frame has been sent yet, serve the dummy black pixel
        if not frame:
            frame = dummy_pixel
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        await asyncio.sleep(0.06)  # Stream at approx 15-20 FPS

# Live Video Endpoints (multipart MJPEG)
@app.get("/stream/raw")
async def stream_raw():
    """Serves the raw video feed from the DJI drone camera."""
    return StreamingResponse(
        frame_generator("raw"),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/stream/overlay")
async def stream_overlay():
    """Serves the real-time AI bounding box overlay video feed."""
    return StreamingResponse(
        frame_generator("overlay"),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# Edge Frame Receiver Endpoint
@app.post("/api/edge/frame")
async def receive_edge_frame(stream_type: str, request: Request):
    """Receives compressed JPEG frames uploaded by the Edge Jetson Nano."""
    global latest_raw_frame, latest_overlay_frame
    frame_data = await request.body()
    if stream_type == "raw":
        latest_raw_frame = frame_data
    elif stream_type == "overlay":
        latest_overlay_frame = frame_data
    return {"status": "ok"}

# Edge Telemetry & Event Sync Endpoint
@app.post("/api/edge/sync")
async def receive_edge_sync(data: Dict[str, Any]):
    """
    Receives real-time telemetry logs, detections, and alerts from the Jetson Nano
    and broadcasts them immediately to the operator dashboard via WebSockets.
    Also handles base64-encoded evidence images from the offline sync worker.
    """
    logger.info(f"Sync event received. Type: {data.get('type')}")

    # If payload contains a base64 evidence image, decode and save it cloud-side
    payload = data.get("payload", {})
    img_b64 = payload.pop("evidence_image_b64", None)
    if img_b64:
        try:
            inc_id   = payload.get("incident_id", "unknown")
            img_data = base64.b64decode(img_b64)
            img_path = EVIDENCE_DIR / f"cloud_evidence_{inc_id}.jpg"
            with open(img_path, "wb") as f:
                f.write(img_data)
            payload["evidence_image_path"] = str(img_path.relative_to(
                Path(__file__).resolve().parent.parent.parent
            ))
        except Exception as e:
            logger.warning(f"Could not save evidence image: {e}")

    # Broadcast to all open dashboards
    await manager.broadcast(data)
    return {"status": "ok"}

# REST APIs for historical query & filtering

@app.get("/api/incidents")
def get_incidents(severity: Optional[str] = Query(None, description="Filter by severity: CRITICAL, HIGH, MEDIUM, LOW")):
    """Retrieves list of all historic clusters/incidents."""
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    query = "SELECT id, timestamp, centroid_latitude, centroid_longitude, severity, illegal_zone, distance_to_river_m, evidence_image_path FROM incidents"
    params = []
    
    if severity:
        query += " WHERE severity = ?" if db_manager.db_type == "sqlite" else " WHERE severity = %s"
        params.append(severity.upper())
        
    query += " ORDER BY id DESC"
    
    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        incidents = []
        for r in rows:
            incidents.append({
                "id": r[0],
                "timestamp": r[1],
                "centroid_latitude": r[2],
                "centroid_longitude": r[3],
                "severity": r[4],
                "illegal_zone": bool(r[5]),
                "distance_to_river_m": r[6],
                "evidence_image_path": r[7]
            })
        return incidents
    except Exception as e:
        logger.error(f"Error fetching incidents: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        cursor.close()
        conn.close()

@app.get("/api/detections")
def get_detections(
    incident_id: Optional[int] = Query(None, description="Filter detections by Incident (Cluster) ID"),
    class_name: Optional[str] = Query(None, description="Filter by class type: jcb, truck, person")
):
    """
    Retrieves individual object detections with coordinates.
    Allows powerful class-level filtering (e.g., viewing ONLY workers/humans)!
    """
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    query = "SELECT id, telemetry_log_id, incident_id, timestamp, class_name, confidence, bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max, latitude, longitude, frame_path FROM detections"
    clauses = []
    params = []
    
    is_sqlite = db_manager.db_type == "sqlite"
    ph = "?" if is_sqlite else "%s"
    
    if incident_id is not None:
        clauses.append(f"incident_id = {ph}")
        params.append(incident_id)
        
    if class_name:
        clauses.append(f"class_name = {ph}")
        params.append(class_name.lower())
        
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
        
    query += " ORDER BY id DESC"
    
    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        detections = []
        for r in rows:
            detections.append({
                "id": r[0],
                "telemetry_log_id": r[1],
                "incident_id": r[2],
                "timestamp": r[3],
                "class_name": r[4],
                "confidence": r[5],
                "bbox": [r[6], r[7], r[8], r[9]], # x_min, y_min, x_max, y_max
                "latitude": r[10],
                "longitude": r[11],
                "frame_path": r[12]
            })
        return detections
    except Exception as e:
        logger.error(f"Error fetching detections: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        cursor.close()
        conn.close()

@app.get("/api/stats")
def get_dashboard_stats():
    """Retrieves aggregate telemetry and spatial count metrics for the widgets."""
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    try:
        # Total incidents
        cursor.execute("SELECT COUNT(*), SUM(CASE WHEN severity = 'CRITICAL' THEN 1 ELSE 0 END) FROM incidents")
        total_inc, total_crit = cursor.fetchone()
        total_crit = total_crit or 0
        
        # Detections by class
        cursor.execute("SELECT class_name, COUNT(*) FROM detections GROUP BY class_name")
        rows = cursor.fetchall()
        class_counts = {r[0]: r[1] for r in rows}
        
        return {
            "total_incidents": total_inc,
            "critical_incidents": total_crit,
            "detections_count": {
                "jcb": class_counts.get("jcb", 0),
                "truck": class_counts.get("truck", 0),
                "person": class_counts.get("person", 0)
            }
        }
    except Exception as e:
        logger.error(f"Error fetching database statistics: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        cursor.close()
        conn.close()

@app.get("/api/report/pdf")
def export_pdf_report(severity: Optional[str] = Query(None, description="Filter by severity"),
                      mission_id: str = Query("BRH-01", description="Mission identifier")):
    """
    Generates and streams a PDF incident report.
    Includes incident table, evidence gallery, and GPS coordinate appendix.
    """
    conn   = db_manager.get_connection()
    cursor = conn.cursor()
    try:
        query  = "SELECT id, timestamp, centroid_latitude, centroid_longitude, severity, illegal_zone, distance_to_river_m, evidence_image_path FROM incidents"
        params = []
        if severity:
            ph = "?" if db_manager.db_type == "sqlite" else "%s"
            query += f" WHERE severity = {ph}"
            params.append(severity.upper())
        query += " ORDER BY id DESC"
        cursor.execute(query, params)
        rows = cursor.fetchall()
        incidents = [{
            "id":                  r[0],
            "timestamp":           r[1],
            "centroid_latitude":   r[2],
            "centroid_longitude":  r[3],
            "severity":            r[4],
            "illegal_zone":        bool(r[5]),
            "distance_to_river_m": r[6],
            "evidence_image_path": r[7]
        } for r in rows]
    finally:
        cursor.close()
        conn.close()

    pdf_bytes = generate_incident_report(incidents=incidents, mission_id=mission_id)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"sand_mining_report_{mission_id}_{ts}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.get("/api/evidence/{filename}")
def get_evidence_image(filename: str):
    """Serves a specific evidence JPEG image by filename."""
    img_path = EVIDENCE_DIR / filename
    if not img_path.exists() or not filename.endswith(".jpg"):
        raise HTTPException(status_code=404, detail="Evidence image not found")
    with open(img_path, "rb") as f:
        return Response(content=f.read(), media_type="image/jpeg")


@app.get("/api/zone/radius")
def get_zone_radius():
    """Returns the currently active buffer radius in metres."""
    return {"radius_m": active_buffer_radius_m}


@app.post("/api/zone/radius")
async def set_zone_radius(data: Dict[str, Any]):
    """
    Updates the active zone enforcement radius.
    1. Rebuilds river_buffer_1km.geojson with the new radius (server-side)
    2. Broadcasts the change over WebSocket so:
       - The browser map redraws its Turf.js buffer to match
       - The Jetson sync_worker can detect the change and reload its ClusterEngine
    """
    global active_buffer_radius_m, global_cluster_engine

    radius_m = float(data.get("radius_m", 1000.0))
    # Clamp to reasonable operational range
    radius_m = max(250.0, min(radius_m, 5000.0))

    # Rebuild GeoJSON and reload in-memory via global ClusterEngine (runs in thread pool)
    import asyncio
    if global_cluster_engine is not None:
        loop = asyncio.get_event_loop()
        def reload_engine():
            global_cluster_engine.set_radius(radius_m)
        await loop.run_in_executor(None, reload_engine)
    else:
        # Fallback if engine is not initialized yet
        from preprocess.zone_builder import build_buffer
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, build_buffer, radius_m)
        if not success:
            raise HTTPException(status_code=500, detail="Buffer rebuild failed — check centerline data")

    active_buffer_radius_m = radius_m
    logger.info(f"Zone radius updated to {radius_m:.0f}m by operator")

    # Broadcast to all dashboard clients + Jetson sync_worker
    await manager.broadcast({
        "type": "zone_radius_update",
        "payload": {"radius_m": radius_m}
    })

    return {"status": "ok", "radius_m": radius_m}


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, listen for any client messages if needed
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        manager.disconnect(websocket)

# HTML Server
@app.get("/", response_class=HTMLResponse)
def get_dashboard_page():
    """Serves the unified, premium dark-themed operator control dashboards."""
    dashboard_path = Path(__file__).resolve().parent / "frontend" / "index.html"
    if dashboard_path.exists():
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        # Fallback basic response if html is missing during initial boot
        return HTMLResponse(content="<h1>Dashboard Page Loading...</h1><p>Please implement frontend/index.html first.</p>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
