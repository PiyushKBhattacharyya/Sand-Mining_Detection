import os
import json
import base64
import logging
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
from db_setup import DatabaseManager
from pdf_generator import generate_incident_report
from zone_builder import build_buffer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Illegal Sand Mining Drone Surveillance Server",
    description="Real-time Edge-Cloud Pipeline with Dual Dashboard feeds and spatial queries"
)

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
    global active_buffer_radius_m

    radius_m = float(data.get("radius_m", 1000.0))
    # Clamp to reasonable operational range
    radius_m = max(250.0, min(radius_m, 5000.0))

    # Rebuild GeoJSON with new radius (runs in thread pool so we don't block)
    import asyncio
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
