
import os
import json
import base64
import logging
import threading
import time
import hashlib
import uuid
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import asyncio
import sys
import torch
import numpy as np

# Monkeypatch torch.load to default weights_only=False for PyTorch 2.6+ compatibility with Ultralytics YOLO
try:
    orig_load = torch.load
    def patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return orig_load(*args, **kwargs)
    torch.load = patched_load
except Exception:
    pass

# Import database manager
sys.path.append(str(Path(__file__).resolve().parent.parent / "preprocess"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "reporting"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "detection"))
# pyrefly: ignore [missing-import]
from db_setup import DatabaseManager
# pyrefly: ignore [missing-import]
from pdf_generator import generate_incident_report
# pyrefly: ignore [missing-import]
from zone_builder import build_buffer

# Salting and SHA-256 Hashing helper functions for secure authentication
def hash_password(password, salt=None):
    if not salt:
        salt = uuid.uuid4().hex
    # Use standard encoding and concatenation to bypass hidden character bugs
    combined = (str(salt) + str(password)).encode('utf-8')
    hashed = hashlib.sha256(combined).hexdigest()
    return str(salt) + ":" + str(hashed)

def verify_password(password, stored_password_hash):
    try:
        salt, hashed = stored_password_hash.split(":")
        check_hash = hashlib.sha256((salt + password).encode('utf-8')).hexdigest()
        return check_hash == hashed
    except Exception:
        return False

# In-memory session store
ACTIVE_SESSIONS = {}

def get_session_user(request):
    session_id = request.cookies.get("session_id")
    if session_id and session_id in ACTIVE_SESSIONS:
        return ACTIVE_SESSIONS[session_id]
    return None

# Flight Recording States & Globals
is_recording = False
recording_writer = None
recording_start_time = None
recording_filepath = None
recording_filename = None
recording_lock = threading.Lock()
global_video_w = 1280
global_video_h = 720
local_webcam_mode = False
last_webcam_frame_time = 0.0


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Illegal Sand Mining Drone Surveillance Server",
    description="Real-time Edge-Cloud Pipeline with Dual Dashboard feeds and spatial queries",
    root_path=os.getenv("ROOT_PATH", "")
)

#  Video source config 
# Controls what feeds the two dashboard video windows.
# cv2.VideoCapture() accepts BOTH an integer (webcam) and a string URL (RTSP),
# so switching from webcam to drone requires only changing this env var.
#
# WEBCAM  (default, testing):   VIDEO_SOURCE=0        built-in Mac/Windows webcam
#                                VIDEO_SOURCE=1        second/external webcam
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
_raw_source = os.getenv("VIDEO_SOURCE", "0")
# Auto-detect: if the value is a plain integer string webcam index, else RTSP URL
VIDEO_SOURCE = int(_raw_source) if _raw_source.lstrip("-").isdigit() else _raw_source

CAMERA_FPS = float(os.getenv("CAMERA_FPS", "15.0"))
CAMERA_QUALITY = int(os.getenv("CAMERA_QUALITY", "50"))

# RTSP-specific tuning (only relevant when VIDEO_SOURCE is a URL)
# Prefer TCP transport for reliability over Wi-Fi (default is UDP which can drop frames)
RTSP_TRANSPORT = os.getenv("RTSP_TRANSPORT", "tcp")   # "tcp" | "udp"


latest_bgr_frame = None
frame_lock = threading.Lock()

main_loop = None
raw_frame_event = None
overlay_frame_event = None

def notify_raw_frame():
    global main_loop, raw_frame_event
    if main_loop and raw_frame_event:
        main_loop.call_soon_threadsafe(raw_frame_event.set)

def notify_overlay_frame():
    global main_loop, overlay_frame_event
    if main_loop and overlay_frame_event:
        main_loop.call_soon_threadsafe(overlay_frame_event.set)

class FreshFrameGrabber:
    """
    A lightweight, dedicated thread that continuously drains OpenCV's internal
    FFMPEG buffer queue for live RTMP/RTSP streams, ensuring we always serve
    the absolute freshest real-time frame with zero latency.
    """
    def __init__(self, cap):
        self.cap = cap
        self.latest_frame = None
        self.ret = False
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._grab_loop, daemon=True, name="rtmp-grabber")
        self.thread.start()

    def _grab_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest_frame = frame
                self.ret = ret

    def read(self):
        with self.lock:
            return self.ret, self.latest_frame

    def release(self):
        self.running = False
        try:
            self.cap.release()
        except Exception:
            pass

def _video_capture_loop():
    """
    Background daemon thread: opens the configured video source and continuously
    pushes raw BGR frames to latest_bgr_frame, and raw JPEG frames into latest_raw_frame.
    It does NOT run YOLO inference, keeping frame acquisition extremely fast (30 FPS).
    """
    global latest_raw_frame, latest_overlay_frame, latest_webcam_detections, local_webcam_mode, last_webcam_frame_time, latest_bgr_frame, use_synthetic_video

    try:
        import cv2
    except ImportError:
        logger.warning("opencv-python not installed video feed disabled.")
        return

    cap = None
    grabber = None
    current_source = None
    is_rtsp = False
    is_file = False

    global global_video_w, global_video_h
    global_video_w = 1280
    global_video_h = 720

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, CAMERA_QUALITY]
    interval = 1.0 / CAMERA_FPS

    consecutive_drops = 0
    last_retry_time = 0.0

    while True:
        t0 = time.time()

        # Dynamically hot-swap video source if the operator changed it on the dashboard HUD!
        target_source = flight_config.get("video_source", "0")
        
        # Normalize target source type (integers for cameras, strings for RTMP/RTSP/files)
        if isinstance(target_source, str) and target_source.lstrip("-").isdigit():
            target_source = int(target_source)

        # Force a periodic retry if we are currently in synthetic fallback mode but a real stream source is configured
        if use_synthetic_video and target_source != "dummy":
            if time.time() - last_retry_time > 5.0:
                logger.info(" Currently in synthetic fallback. Retrying to connect to real stream source: {}".format(target_source))
                current_source = None # This will trigger a re-connection attempt below!
                last_retry_time = time.time()

        if current_source != target_source:
            logger.info(" Swapping live video capture source: {} -> {}".format(current_source, target_source))
            if grabber is not None:
                try:
                    grabber.release()
                except Exception as ex_rel:
                    logger.debug("Error releasing previous FreshFrameGrabber: {}".format(ex_rel))
                cap = None
                grabber = None
            elif cap is not None:
                try:
                    cap.release()
                except Exception as ex_rel:
                    logger.debug("Error releasing previous VideoCapture: {}".format(ex_rel))
                cap = None

            current_source = target_source
            use_synthetic_video = False

            if target_source == "dummy":
                logger.info("  New source is dummy. Falling back to synthetic simulation mode.")
                use_synthetic_video = True
            else:
                try:
                    is_rtsp = isinstance(target_source, str) and (
                        target_source.startswith("rtsp://") or 
                        target_source.startswith("rtmp://") or 
                        target_source.startswith("http://") or 
                        target_source.startswith("https://")
                    )
                    is_file = isinstance(target_source, str) and not is_rtsp

                    if is_rtsp:
                        # Build low-latency FFMPEG options: disable buffering, reduce probe size/duration
                        opts = f"rtsp_transport;{RTSP_TRANSPORT}|fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0"
                        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
                        logger.info("  Connecting to dynamic drone stream: {} (transport={} with low-delay)".format(target_source, RTSP_TRANSPORT))
                        cap = cv2.VideoCapture(target_source, cv2.CAP_FFMPEG)
                        if cap is not None and cap.isOpened():
                            # Set internal buffer size to 1 to prevent frame queuing lag
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            grabber = FreshFrameGrabber(cap)
                    elif is_file:
                        logger.info("  Opening dynamic video file: {}...".format(target_source))
                        cap = cv2.VideoCapture(target_source)
                    else:
                        logger.info("  Starting dynamic webcam camera index {}...".format(target_source))
                        cap = cv2.VideoCapture(target_source)
                        if cap is not None and cap.isOpened():
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                    if cap is None or not cap.isOpened():
                        raise Exception("VideoCapture returned an error or is closed")

                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    global_video_w = w if w > 0 else 1280
                    global_video_h = h if h > 0 else 720
                    source_label = "RTSP dynamic stream" if is_rtsp else f"Camera index {target_source}"
                    logger.info("  [SUCCESS] {} dynamic swapped successfully at {}x{}".format(source_label, global_video_w, global_video_h))

                except Exception as e:
                    logger.warning("  Could not open dynamic source ({})  falling back to synthetic simulation.".format(e))
                    use_synthetic_video = True
                    global_video_w = 1280
                    global_video_h = 720

        if local_webcam_mode:
            # If the user is actively streaming their browser webcam, pause drone file playback!
            # If no frame has been uploaded for more than 3 seconds, reset mode back to false!
            if time.time() - last_webcam_frame_time > 3.0:
                local_webcam_mode = False
                logger.info(" Browser webcam stream timed out. Restoring default drone video playback.")
            else:
                time.sleep(0.06)
                continue

        if use_synthetic_video:
            # Create a nice dark-blue grid background (simulating a tactical drone camera screen)
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            # Make it a sleek dark-blue grid background
            frame[:, :] = [18, 12, 8] # Very dark navy blue
            
            # Draw standard 80px gridlines
            for x in range(0, 1280, 80):
                cv2.line(frame, (x, 0), (x, 720), (32, 24, 18), 1)
            for y in range(0, 720, 80):
                cv2.line(frame, (0, y), (1280, y), (32, 24, 18), 1)
                
            # Draw central tactical HUD green crosshair (slightly darker to not clash with warning box)
            cv2.line(frame, (640 - 20, 360), (640 + 20, 360), (0, 160, 0), 1)
            cv2.line(frame, (640, 360 - 20), (640, 360 + 20), (0, 160, 0), 1)
            
            # Print subtle status coordinate indicators
            drone_lat = latest_drone_coords.get("lat", 0.0)
            drone_lon = latest_drone_coords.get("lon", 0.0)
            cv2.putText(frame, "LAT: {:.6f}".format(drone_lat), (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 150, 100), 1)
            cv2.putText(frame, "LON: {:.6f}".format(drone_lon), (40, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 150, 100), 1)
            
            # Add dynamic scan line sweep animation
            scan_y = int((time.time() * 200) % 720)
            cv2.line(frame, (0, scan_y), (1280, scan_y), (0, 80, 0), 1)

            ret = True
        else:
            if grabber is not None:
                ret, frame = grabber.read()
            else:
                ret, frame = cap.read()
                
            if not ret and is_file:
                # Rewind local video file to loop infinitely!
                logger.info("Video file ended. Rewinding back to start...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

        if not ret:
            if is_rtsp:
                consecutive_drops += 1
                if consecutive_drops > 25:
                    logger.warning("  RTMP/RTSP stream disconnected. Resetting stream client for automatic reconnection...")
                    if grabber is not None:
                        try:
                            grabber.release()
                        except Exception:
                            pass
                    elif cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                    cap = None
                    grabber = None
                    current_source = None
                    consecutive_drops = 0
                    time.sleep(2.0)
                else:
                    logger.warning("  RTSP frame drop  retrying...")
                    time.sleep(0.1)
            else:
                time.sleep(0.05)
            continue

        consecutive_drops = 0

        if not use_synthetic_video:
            # Flip the frame horizontally to correct webcam mirroring
            frame = cv2.flip(frame, 1)

        # Store in shared BGR buffer for the AI inference thread
        with frame_lock:
            latest_bgr_frame = frame.copy()

        # OPTIMIZATION: Downscale raw frame for low-latency web streaming to save bandwidth
        h, w = frame.shape[:2]
        if w > 854 or h > 480:
            stream_frame = cv2.resize(frame, (854, 480))
        else:
            stream_frame = frame

        _, buf = cv2.imencode(".jpg", stream_frame, encode_params)
        jpeg = buf.tobytes()

        # Raw feed: clean, unprocessed frame from the camera
        latest_raw_frame = jpeg
        notify_raw_frame()

        if use_synthetic_video:
            latest_overlay_frame = jpeg
            notify_overlay_frame()
            latest_webcam_detections = []

        # Determine the final frame to write to the recording
        recording_frame = frame

        # Write to video recorder if active
        global is_recording, recording_writer, recording_lock
        if is_recording and not local_webcam_mode:
            with recording_lock:
                if recording_writer is not None:
                    try:
                        recording_writer.write(recording_frame)
                    except Exception as e:
                        logger.error("Error writing frame to recording: {}".format(e))

        # Sleep throttling strategy:
        # 1. Synthetic video needs sleep to match target FPS.
        # 2. File playback needs sleep to match target FPS.
        # 3. Live network streams (RTMP/RTSP) or Webcams do NOT sleep, as cap.read() blocks naturally to stream FPS.
        if use_synthetic_video or is_file:
            elapsed = time.time() - t0
            sleep_for = interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
        else:
            # Short sleep to prevent CPU hogging if capture stream drops or returns too fast
            time.sleep(0.001)

def _yolo_inference_loop():
    """
    Background daemon thread: periodically grabs the absolute latest BGR frame from
    latest_bgr_frame, runs YOLO inference on it, and updates latest_overlay_frame.
    This guarantees that the raw stream remains high-FPS (30 FPS), and the AI stream
    never falls behind real-time or plays in slow-motion.
    """
    global latest_bgr_frame, latest_overlay_frame, latest_webcam_detections, _yolo_model, local_webcam_mode
    try:
        import cv2
    except ImportError:
        return

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, CAMERA_QUALITY]
    
    # Track the active loaded model for the local server webcam feed
    custom_weights = Path(__file__).resolve().parent.parent.parent / "models" / "weights" / "best.pt"
    current_loaded_model_path = str(custom_weights) if custom_weights.exists() else "yolov8n.pt"

    while True:
        # Sleep a tiny bit to avoid raw spinning when no frames are available
        time.sleep(0.01)

        # Dynamically hot-swap local YOLO model if operator changed it in the dropdown!
        target_model_name = flight_config.get("active_model", "yolov8n.pt")
        weights_dir = Path(__file__).resolve().parent.parent.parent / "models" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        if target_model_name == "best.pt":
            target_path = str(weights_dir / "best.pt")
            if not Path(target_path).exists():
                target_path = str(weights_dir / "yolov8n.pt")
        else:
            target_path = str(weights_dir / target_model_name)

        if current_loaded_model_path != target_path:
            logger.info(" Swapping local server YOLO model: {} -> {}".format(current_loaded_model_path, target_path))
            try:
                from ultralytics import YOLO
                _yolo_model = YOLO(target_path)
                current_loaded_model_path = target_path
                logger.info(" Local server YOLO model successfully swapped to: {}".format(target_model_name))
            except Exception as e:
                logger.error(" Failed to dynamic swap local YOLO model: {}".format(e))
                # Prevent CPU-burning infinite retry loops on model loading failures:
                current_loaded_model_path = target_path

        frame = None
        with frame_lock:
            if latest_bgr_frame is not None:
                frame = latest_bgr_frame.copy()

        if frame is None:
            continue

        overlay = None
        if _yolo_model is not None:
            try:
                # Detect person (0), car (2), motorcycle (3), bus (5), truck (7)
                # OPTIMIZATION: Run YOLO at imgsz=1280 to capture tiny targets/humans from aerial heights!
                results = _yolo_model(
                    frame,
                    verbose=False,
                    classes=[0, 2, 3, 5, 7],
                    conf=0.30,
                    iou=0.45,
                    imgsz=1280
                )
                overlay = results[0].plot()   # annotated BGR numpy array
                
                # Downscale the annotated overlay for web streaming to save bandwidth
                h_ov, w_ov = overlay.shape[:2]
                if w_ov > 854 or h_ov > 480:
                    overlay_stream = cv2.resize(overlay, (854, 480))
                else:
                    overlay_stream = overlay
                    
                _, obuf = cv2.imencode(".jpg", overlay_stream, encode_params)
                latest_overlay_frame = obuf.tobytes()
                notify_overlay_frame()

                # Extract and store bounding box details globally for hybrid telemetry mapping
                # But ONLY trigger incidents when inside geofence start area!
                active_dets = []
                if is_at_starting_spot():
                    if len(results[0].boxes) > 0:
                        for box in results[0].boxes:
                            coords = box.xyxy[0].tolist()
                            conf = float(box.conf[0].item())
                            cls_id = int(box.cls[0].item())
                            
                            # Map YOLO class IDs to standard names
                            cls_map = {0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
                            class_name = cls_map.get(cls_id, 'jcb')
                            
                            active_dets.append({
                                'class_name': class_name,
                                'confidence': conf,
                                'bbox_x_min': int(coords[0]),
                                'bbox_y_min': int(coords[1]),
                                'bbox_x_max': int(coords[2]),
                                'bbox_y_max': int(coords[3])
                            })
                latest_webcam_detections = active_dets
            except Exception as exc:
                logger.debug("YOLO inference error: {}".format(exc))
                # Fallback to raw frame on inference error
                h_ov, w_ov = frame.shape[:2]
                if w_ov > 854 or h_ov > 480:
                    overlay_stream = cv2.resize(frame, (854, 480))
                else:
                    overlay_stream = frame
                _, obuf = cv2.imencode(".jpg", overlay_stream, encode_params)
                latest_overlay_frame = obuf.tobytes()
                notify_overlay_frame()
                latest_webcam_detections = []
                overlay = frame
        else:
            # No model loaded, mirror raw frame
            h_ov, w_ov = frame.shape[:2]
            if w_ov > 854 or h_ov > 480:
                overlay_stream = cv2.resize(frame, (854, 480))
            else:
                overlay_stream = frame
            _, obuf = cv2.imencode(".jpg", overlay_stream, encode_params)
            latest_overlay_frame = obuf.tobytes()
            notify_overlay_frame()
            latest_webcam_detections = []
            overlay = frame

        # Write to video recorder if active and local_webcam_mode is True
        global is_recording, recording_writer, recording_lock
        if is_recording and local_webcam_mode:
            with recording_lock:
                if recording_writer is not None:
                    try:
                        # Pick annotated overlay if inside geofence, else raw frame
                        if is_at_starting_spot() and overlay is not None:
                            recording_frame = overlay
                        else:
                            recording_frame = frame
                        # ALWAYS resize to match the VideoWriter's initialized dimensions
                        rw, rh = global_video_w, global_video_h
                        fh, fw = recording_frame.shape[:2]
                        if fw != rw or fh != rh:
                            recording_frame = cv2.resize(recording_frame, (rw, rh))
                        recording_writer.write(recording_frame)
                    except Exception as e:
                        logger.error("Error writing frame to recording: {}".format(e))


# Mount the project's data directory so the frontend can directly load spatial GeoJSON files
app.mount("/data", StaticFiles(directory=str(Path(__file__).resolve().parent.parent.parent / "data")), name="data")

# Evidence directory  also served as static for UI image display
EVIDENCE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "detections"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

# Connect to database
db_manager = DatabaseManager(db_type="sqlite")
# Ensure DB is initialized
db_manager.initialize_database()

# Active buffer radius  starts at 1km, updated via /api/zone/radius
active_buffer_radius_m = 1000.0

#  Global dictionary holding the active flight mission control configuration.
#  Allows the operator dashboard to set parameters that are fetched by the drone
# mid-flight (target detection model and starting coordinates).

flight_config = {
    "active_model": "yolov8n.pt",  # Default model weights filename
    "start_lat": 0.0,              # Dynamic start coordinate latitude
    "start_lng": 0.0,              # Dynamic start coordinate longitude
    "start_radius_meters": 500.0,  # Dynamic start radius in meters
    "video_source": os.getenv("VIDEO_SOURCE", "0"), # Dynamic RTMP/RTSP stream source or camera index
    "detection_enabled": False,
}

latest_drone_coords = {"lat": 0.0, "lon": 0.0}
has_reached_starting_spot = False

# Calculate distance to see if drone has reached the Detection Starting Spot
def is_at_starting_spot():
    global has_reached_starting_spot

    target_lat = flight_config.get("start_lat", 0.0)
    target_lng = flight_config.get("start_lng", 0.0)
    radius = flight_config.get("start_radius_meters", 500.0)
    enabled = flight_config.get("detection_enabled", False)

    if not enabled or target_lat == 0.0 or target_lng == 0.0:
        return False

    drone_lat = latest_drone_coords.get("lat", 0.0)
    drone_lon = latest_drone_coords.get("lon", 0.0)
    if drone_lat == 0.0 or drone_lon == 0.0:
        return False

    import math
    lat1, lon1 = math.radians(drone_lat), math.radians(drone_lon)
    lat2, lon2 = math.radians(target_lat), math.radians(target_lng)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance_meters = 6371000.0 * c
    if distance_meters <= radius:
        if not has_reached_starting_spot:
            logger.info(" Drone has entered the Detection Starting Spot! AI Detection System is now ACTIVE.")
            has_reached_starting_spot = True

    return has_reached_starting_spot

def is_inside_fence():
    global global_cluster_engine
    if global_cluster_engine is None:
        return True  # Fallback if engine is initializing
    drone_lat = latest_drone_coords.get("lat", 0.0)
    drone_lon = latest_drone_coords.get("lon", 0.0)
    if drone_lat == 0.0 or drone_lon == 0.0:
        return False
    return global_cluster_engine.is_in_illegal_zone(drone_lat, drone_lon)


# Store active websocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("New client connected. Total clients: {}".format(len(self.active_connections)))

    def disconnect(self, websocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("Client disconnected. Total clients: {}".format(len(self.active_connections)))

    async def broadcast(self, message):
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
latest_raw_frame = b""
latest_overlay_frame = b""
latest_webcam_detections = []
use_synthetic_video = True

# Holds the loaded YOLO model  set once at startup, used in _video_capture_loop
_yolo_model = None
yolo_lock = None

# Global ClusterEngine for runtime buffer size synchronization
global_cluster_engine = None

async def _telemetry_fallback_broadcast_loop():
    """
    WHAT: Fallback task that periodically broadcasts the drone's position to WebSockets
    if the operator has manually set geofence start coordinates but no real telemetry
    relayer is active.
    WHY: Allows operators using standard DJI Pilot/Fly apps (without custom API relay subs)
    to set the drone location by double-clicking on the map!
    """
    global latest_drone_coords, flight_config, manager, use_synthetic_video
    await asyncio.sleep(4.0)  # Wait for startup and uvicorn bind
    
    # We initialize the server-side cluster engine here as well to ensure
    # geofence and enforcement logic functions seamlessly if webcam/real streams trigger incidents
    global global_cluster_engine, db_manager, active_buffer_radius_m
    if global_cluster_engine is None:
        try:
            # pyrefly: ignore [missing-import]
            from cluster_engine import ClusterEngine
            global_cluster_engine = ClusterEngine(db_manager=db_manager)
            global_cluster_engine.set_radius(active_buffer_radius_m)
        except Exception as e:
            logger.debug(f"Failed to initialize cluster engine: {e}")

    import datetime
    while True:
        try:
            start_lat = float(flight_config.get("start_lat", 0.0))
            start_lng = float(flight_config.get("start_lng", 0.0))
            
            # If a geofence location is applied, and we have received NO active telemetry packets
            # from a real physical relayer/companion app yet (latest_drone_coords is 0.0)
            if start_lat != 0.0 and start_lng != 0.0 and latest_drone_coords.get("lat", 0.0) == 0.0:
                # Mirror the geofence starting spot as the active drone coordinate!
                latest_drone_coords["lat"] = start_lat
                latest_drone_coords["lon"] = start_lng
                
                # Broadcast this mock telemetry packet to place the blue marker and update status widgets!
                tele_payload = {
                    "type": "telemetry",
                    "payload": {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "lat": start_lat,
                        "lon": start_lng,
                        "altitude": 70.0,
                        "speed": 0.0,
                        "battery": 95
                    }
                }
                await manager.broadcast(tele_payload)
            
            # Periodically broadcast stream standby status to the frontend!
            await manager.broadcast({
                "type": "stream_status",
                "payload": {
                    "use_synthetic_video": use_synthetic_video,
                    "video_source": str(flight_config.get("video_source", "rtmp://187.127.142.58:1935/live/drone"))
                }
            })
        except Exception as e:
            logger.debug(f"Telemetry fallback loop: {e}")
        await asyncio.sleep(1.0)

async def _mavsdk_telemetry_loop():
    """
    Asynchronous task that listens for incoming MAVLink packets on UDP port 14550.
    Extracts exact live GPS, altitude, speed, and battery telemetry from MAVSDK,
    saves them to the local sqlite database, and broadcasts them via websockets.
    """
    global latest_drone_coords, manager, db_manager
    await asyncio.sleep(5.0)  # Wait for startup and uvicorn bind
    
    try:
        from mavsdk import System
        import datetime
        
        drone = System()
        
        logger.info("[MAVSDK] Attempting to bind MAVLink UDP port 14550...")
        await drone.connect(system_address="udp://:14550")
        
        # Monitor connection state
        logger.info("[MAVSDK] MAVLink listener bound! Monitoring connection state...")
        async for state in drone.core.connection_state():
            if state.is_connected:
                logger.info("[MAVSDK] Telemetry link successfully established!")
                break
    except Exception as e:
        logger.error(f"[MAVSDK] Telemetry initialization failed (graceful bypass): {e}")
        return

    # Define SQL command for database telemetry inserts
    insert_telemetry_sql = """
    INSERT INTO telemetry_logs (
        timestamp, latitude, longitude, altitude_agl, 
        gimbal_pitch, gimbal_yaw, gimbal_roll, 
        drone_speed, battery_percentage, gps_accuracy_m
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    # Helper function to save telemetry into SQLite
    def save_telemetry_db(timestamp, lat, lon, alt, speed, battery):
        try:
            conn = db_manager.get_connection()
            cursor = conn.cursor()
            cursor.execute(insert_telemetry_sql, (
                timestamp, lat, lon, alt, -80.0, 0.0, 0.0, speed, int(battery), 0.15
            ))
            t_id = cursor.lastrowid
            conn.commit()
            cursor.close()
            conn.close()
            return t_id
        except Exception as db_err:
            logger.error(f"[MAVSDK] Database logging error: {db_err}")
            return None

    # Async sub-listeners to fetch coordinates and telemetry from MAVSDK in real-time
    async def position_listener():
        global latest_drone_coords
        try:
            async for pos in drone.telemetry.position():
                latest_drone_coords["lat"] = pos.latitude_deg
                latest_drone_coords["lon"] = pos.longitude_deg
                latest_drone_coords["alt"] = pos.relative_altitude_m
        except Exception as pos_ex:
            logger.debug(f"[MAVSDK] Position stream interrupted: {pos_ex}")

    async def battery_listener():
        try:
            async for bat in drone.telemetry.battery():
                flight_config["battery"] = int(bat.remaining_percent * 100)
        except Exception as bat_ex:
            logger.debug(f"[MAVSDK] Battery stream interrupted: {bat_ex}")

    async def speed_listener():
        try:
            async for speed_info in drone.telemetry.velocity_ned():
                v_north = speed_info.north_m_s
                v_east = speed_info.east_m_s
                abs_speed = (v_north**2 + v_east**2)**0.5
                flight_config["speed"] = abs_speed
        except Exception as sp_ex:
            logger.debug(f"[MAVSDK] Speed stream interrupted: {sp_ex}")

    # Launch MAVSDK telemetry sub-listeners as parallel tasks
    asyncio.ensure_future(position_listener())
    asyncio.ensure_future(battery_listener())
    asyncio.ensure_future(speed_listener())

    loop = asyncio.get_event_loop()
    
    # Standard broadcast rate loop (approx 1 Hz)
    while True:
        try:
            lat = latest_drone_coords.get("lat", 0.0)
            lon = latest_drone_coords.get("lon", 0.0)
            alt = latest_drone_coords.get("alt", 70.0)
            battery = flight_config.get("battery", 95)
            speed = flight_config.get("speed", 0.0)

            if lat != 0.0 and lon != 0.0:
                timestamp = datetime.datetime.now().isoformat()
                
                # Save to database
                await loop.run_in_executor(None, save_telemetry_db, timestamp, lat, lon, alt, speed, battery)
                
                # Broadcast MAVLink telemetry packet to all WebSockets
                await manager.broadcast({
                    "type": "telemetry",
                    "payload": {
                        "timestamp": timestamp,
                        "lat": lat,
                        "lon": lon,
                        "altitude": alt,
                        "speed": speed,
                        "battery": battery
                    }
                })
        except Exception as loop_ex:
            logger.debug(f"[MAVSDK] Telemetry broadcast loop error: {loop_ex}")
        await asyncio.sleep(1.0)

async def _webcam_telemetry_simulation_loop():
    global latest_webcam_detections, db_manager, active_buffer_radius_m, global_cluster_engine
    
    # Wait for the app to start up fully
    await asyncio.sleep(2.0)
    
    # pyrefly: ignore [missing-import]
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
        logger.error("Centerline not found for hybrid simulation: {}".format(centerline_path))
        return
        
    try:
        with open(centerline_path, 'r') as f:
            cl_data = json.load(f)
    except Exception as e:
        logger.error("Error loading centerline for hybrid simulation: {}".format(e))
        return
        
    raw_coords = cl_data['features'][0]['geometry']['coordinates']
    # Filter: only keep coordinates in Assam, India (lat > 26.0)
    # The southern portion of the centerline crosses into Bangladesh
    # where Google Maps has no road data and directions fail.
    raw_coords = [c for c in raw_coords if c[1] > 26.0]
    if len(raw_coords) < 10:
        logger.error("Not enough centerline points in India after filtering.")
        return
    logger.info("Centerline filtered to {} points in Assam, India.".format(len(raw_coords)))
    # Store raw centerline waypoint data WITHOUT baking in the lateral offset.
    # The actual lat/lon is computed dynamically each step so the weave amplitude
    # instantly reflects the live active_buffer_radius_m when the slider moves.
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

        heading = math.atan2(dy, dx)
        perp_angle = heading + math.pi / 2.0
        heading_deg = (90.0 - math.degrees(heading)) % 360.0

        for step in range(steps):
            t = step / steps
            interp_lon = lon1 + (lon2 - lon1) * t
            interp_lat = lat1 + (lat2 - lat1) * t
            weave_phase = (i * steps + step) * 0.05

            flight_points.append({
                # Base centerline position (no offset applied yet)
                'base_lat':   interp_lat,
                'base_lon':   interp_lon,
                # Perpendicular direction components (unit vector on ground plane)
                'perp_sin':   math.sin(perp_angle),
                'perp_cos':   math.cos(perp_angle),
                # Sine-wave phase so the drone weaves in/out of the buffer
                'weave_phase': weave_phase,
                'heading':    heading_deg,
            })

    logger.info(" Hybrid simulation generated {} waypoints (dynamic-radius weave).".format(len(flight_points)))
    
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

            # Dynamically compute lat/lon using the LIVE buffer radius so the
            # drone's weave amplitude instantly matches the slider value.
            # Weave amplitude = 1.8 * buffer radius (flies well inside AND outside)
            lateral_offset_meters = math.sin(point['weave_phase']) * (active_buffer_radius_m * 1.8)
            base_lat = point['base_lat']
            base_lon = point['base_lon']
            lat = base_lat + (lateral_offset_meters * point['perp_sin']) / 111320
            lon = base_lon + (lateral_offset_meters * point['perp_cos']) / (
                111320 * math.cos(math.radians(base_lat))
            )

            # Update global latest_drone_coords for local webcam YOLO gating!
            global latest_drone_coords
            latest_drone_coords["lat"] = lat
            latest_drone_coords["lon"] = lon

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
            
            #  Process active webcam detections mapped to this GPS location! 
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
                        # pyrefly: ignore [missing-import]
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
                        logger.error("Error generating hybrid evidence snapshot: {}".format(e))

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
            logger.error("Error in hybrid telemetry loop: {}".format(e))
            await asyncio.sleep(1.0)


@app.on_event("startup")
async def startup_event():
    """
    Fires once when uvicorn starts.
    1. Loads the YOLO detection model (custom best.pt if available, else yolov8n.pt).
    2. Launches the video capture thread so both dashboard feed windows go live.
    """
    global _yolo_model, main_loop, raw_frame_event, overlay_frame_event
    main_loop = asyncio.get_running_loop()
    raw_frame_event = asyncio.Event()
    overlay_frame_event = asyncio.Event()

    #  Load YOLO model 
    # Priority: custom trained weights  generic YOLOv8n placeholder
    custom_weights = Path(__file__).resolve().parent.parent.parent / "models" / "weights" / "best.pt"
    try:
        from ultralytics import YOLO

        if custom_weights.exists():
            _yolo_model = YOLO(str(custom_weights))
            logger.info("  Loaded CUSTOM YOLO model: {}".format(custom_weights.name))
        else:
            # Auto-downloads yolov8n.pt on first run (~6 MB) directly to models/weights/
            placeholder_path = custom_weights.parent / "yolov8n.pt"
            custom_weights.parent.mkdir(parents=True, exist_ok=True)
            _yolo_model = YOLO(str(placeholder_path))
            logger.info("  YOLOv8n placeholder loaded at {}  detecting PERSON ONLY (conf0.30). Swap best.pt when ready.".format(placeholder_path.name))
    except Exception as e:
        logger.warning("  YOLO failed to load  overlay will mirror raw feed. Error: {}".format(e))
        _yolo_model = None

    #  Start video capture and YOLO inference threads AFTER model is ready 
    # Ensures first frames already have a model to run against.
    t = threading.Thread(target=_video_capture_loop, daemon=True, name="video-capture")
    t.start()
    
    t_yolo = threading.Thread(target=_yolo_inference_loop, daemon=True, name="yolo-inference")
    t_yolo.start()
    
    source_desc = "RTSP: {}".format(VIDEO_SOURCE) if isinstance(VIDEO_SOURCE, str) else f"webcam {VIDEO_SOURCE}"
    logger.info("  Video capture and YOLO inference threads launched ({})  dashboard feeds will populate shortly.".format(source_desc))

    # Start the fallback telemetry loop so manual map clicks can place the drone!
    asyncio.ensure_future(_telemetry_fallback_broadcast_loop())
    
    # Start the live MAVSDK MAVLink telemetry listener on UDP port 14550!
    asyncio.ensure_future(_mavsdk_telemetry_loop())



# Frame generator for multipart MJPEG streaming
async def frame_generator(stream_type: str):
    global latest_raw_frame, latest_overlay_frame, raw_frame_event, overlay_frame_event
    event = raw_frame_event if stream_type == "raw" else overlay_frame_event
    
    # 1. Fallback dummy frame if no feed is active
    # A simple 1x1 black pixel JPEG byte representation
    dummy_pixel = b'\xff\xd8\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9'

    while True:
        if event:
            try:
                await asyncio.wait_for(event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            event.clear()

        frame = latest_raw_frame if stream_type == "raw" else latest_overlay_frame
        
        # If no frame has been sent yet, serve the dummy black pixel
        if not frame:
            frame = dummy_pixel
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Live Video Endpoints (multipart MJPEG)
@app.get("/stream/raw")
async def stream_raw(request: Request):
    """Serves the raw video feed from the DJI drone camera."""
    user = get_session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return StreamingResponse(
        frame_generator("raw"),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/stream/overlay")
async def stream_overlay(request: Request):
    """Serves the real-time AI bounding box overlay video feed."""
    user = get_session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return StreamingResponse(
        frame_generator("overlay"),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# Edge Frame Receiver Endpoint
@app.post("/api/edge/frame")
async def receive_edge_frame(stream_type: str, request: Request):
    """Receives compressed JPEG frames uploaded by the Edge Jetson Nano or local Webcam pipeline."""
    global latest_raw_frame, latest_overlay_frame, latest_webcam_detections, local_webcam_mode, last_webcam_frame_time, latest_bgr_frame
    frame_data = await request.body()
    if not frame_data:
        return {"status": "ok"}

    # Browser is streaming frames to us! Activate local webcam mode so recording / detection pipelines fall back to this stream
    local_webcam_mode = True
    last_webcam_frame_time = time.time()

    if stream_type == "raw":
        # Decode and process browser webcam frame
        try:
            import cv2
            import numpy as np
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                # Flip the frame horizontally to correct webcam mirroring
                frame = cv2.flip(frame, 1)
                
                # OPTIMIZATION: Downscale raw frame for low-latency web streaming, but keep 'frame' high-resolution for YOLO
                h, w = frame.shape[:2]
                if w > 854 or h > 480:
                    stream_frame = cv2.resize(frame, (854, 480))
                else:
                    stream_frame = frame
                
                # Re-encode flipped frame to update the raw stream
                _, raw_buf = cv2.imencode(".jpg", stream_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                latest_raw_frame = raw_buf.tobytes()
                notify_raw_frame()
                
                with frame_lock:
                    latest_bgr_frame = frame.copy()
        except Exception as exc:
            logger.debug("VPS received-frame decode error: {}".format(exc))
            
    elif stream_type == "overlay":
        # Save received overlay JPEG bytes to latest_overlay_frame and trigger notify_overlay_frame()
        latest_overlay_frame = frame_data
        notify_overlay_frame()
            
    return {"status": "ok"}

# Edge Telemetry & Event Sync Endpoint
@app.post("/api/edge/sync")
async def receive_edge_sync(data: dict):
    """
    Receives real-time telemetry logs, detections, and alerts from the Jetson Nano/Mobile App
    and broadcasts them immediately to the operator dashboard via WebSockets.
    Also handles base64-encoded evidence images from the offline sync worker.
    """
    # Auto-wrap flat telemetry payloads from the mobile companion app
    if "lat" in data and "lon" in data and data.get("type") is None:
        data = {
            "type": "telemetry",
            "payload": data
        }

    logger.info("Sync event received. Type: {}".format(data.get('type')))

    # If payload contains a base64 evidence image, decode and save it cloud-side
    payload = data.get("payload", {})
    if data.get("type") == "telemetry":
        global latest_drone_coords
        latest_drone_coords["lat"] = float(payload.get("lat", 0.0))
        latest_drone_coords["lon"] = float(payload.get("lon", 0.0))
    img_b64 = payload.pop("evidence_image_b64", None)
    if img_b64:
        try:
            inc_id   = payload.get("incident_id", "unknown")
            # 1. Decode the base64 string sent by the Jetson Sync Worker
            img_data = base64.b64decode(img_b64)
            # 2. Save the image in the cloud evidence directory
            img_path = EVIDENCE_DIR / "cloud_evidence_{}.jpg".format(inc_id)
            with open(img_path, "wb") as f:
                f.write(img_data)
            payload["evidence_image_path"] = str(img_path.relative_to(
                Path(__file__).resolve().parent.parent.parent
            ))

            # Save the raw image bytes directly inside PostgreSQL on the VPS!
            # This fulfills the user requirement to extract images and save them directly in Postgres VPS!

            conn = db_manager.get_connection()
            cursor = conn.cursor()
            ph = "%s" if db_manager.db_type == "postgresql" else "?"

            cursor.execute(
                "UPDATE incidents SET evidence_image_blob = {} WHERE id = {}".format(ph, ph),
                (img_data, inc_id)
            )

            conn.commit()
            cursor.close()
            conn.close()
            logger.info("Saved binary evidence blob to DB for incident #{}".format(inc_id))

        except Exception as e:
            logger.error("Could not save evidence image: {}".format(e))

    # Broadcast to all open dashboards
    await manager.broadcast(data)
    return {"status": "ok"}

# REST APIs for historical query & filtering

def parse_date_to_utc(dt_str, is_end = False):
    """
    Converts a local browser datetime string to UTC in YYYY-MM-DD HH:MM:SS format.
    If date-only (10 chars), appends start-of-day or end-of-day time.
    """
    from datetime import datetime, timezone
    try:
        val = dt_str.strip()
        if len(val) == 10:
            val += "T23:59:59" if is_end else "T00:00:00"
        elif len(val) == 16:
            val += ":59" if is_end else ":00"
            
        dt = datetime.fromisoformat(val.replace(' ', 'T'))
        if dt.tzinfo is None:
            dt = dt.astimezone()
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.warning("Error parsing date {}: {}".format(dt_str, e))
        return dt_str.replace('T', ' ')

@app.get("/api/incidents")
def get_incidents(
    request: Request,
    severity                = Query(None, description="Filter by severity: EXTREME, SEVERE, MEDIUM, LOW"),
    start_date                = Query(None, description="Filter by start date/time (local timezone)"),
    end_date                = Query(None, description="Filter by end date/time (local timezone)")
):
    """Retrieves list of all historic clusters/incidents with optional filtering."""
    user = get_session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    query = "SELECT id, timestamp, centroid_latitude, centroid_longitude, severity, illegal_zone, distance_to_river_m, evidence_image_path FROM incidents"
    clauses = []
    params = []
    
    is_sqlite = db_manager.db_type == "sqlite"
    ph = "?" if is_sqlite else "%s"
    
    if severity:
        clauses.append("severity = {}".format(ph))
        params.append(severity.upper())
        
    if start_date:
        start_utc = parse_date_to_utc(start_date, is_end=False)
        clauses.append("timestamp >= {}".format(ph))
        params.append(start_utc)
        
    if end_date:
        end_utc = parse_date_to_utc(end_date, is_end=True)
        clauses.append("timestamp <= {}".format(ph))
        params.append(end_utc)
        
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
        
    query += " ORDER BY id DESC LIMIT 100"
    
    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        incidents = []
        for r in rows:
            db_ts = r[1]
            if db_ts and "Z" not in db_ts and "+" not in db_ts:
                ts_formatted = db_ts.replace(" ", "T") + "Z"
            else:
                ts_formatted = db_ts
                
            incidents.append({
                "id": r[0],
                "timestamp": ts_formatted,
                "centroid_latitude": r[2],
                "centroid_longitude": r[3],
                "severity": r[4],
                "illegal_zone": bool(r[5]),
                "distance_to_river_m": r[6],
                "evidence_image_path": r[7]
            })
        return incidents
    except Exception as e:
        logger.error("Error fetching incidents: {}".format(e))
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        cursor.close()
        conn.close()

@app.get("/api/detections")
def get_detections(
    request: Request,
    incident_id = Query(None, description="Filter detections by Incident (Cluster) ID"),
    class_name = Query(None, description="Filter by class type: jcb, truck, person")
):
    """
    Retrieves individual object detections with coordinates.
    Allows powerful class-level filtering (e.g., viewing ONLY workers/humans)!
    """
    user = get_session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    query = "SELECT id, telemetry_log_id, incident_id, timestamp, class_name, confidence, bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max, latitude, longitude, frame_path FROM detections"
    clauses = []
    params = []
    
    is_sqlite = db_manager.db_type == "sqlite"
    ph = "?" if is_sqlite else "%s"
    
    if incident_id is not None:
        clauses.append("incident_id = {}".format(ph))
        params.append(incident_id)
        
    if class_name:
        clauses.append("class_name = {}".format(ph))
        params.append(class_name.lower())
        
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
        
    query += " ORDER BY id DESC"
    
    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        detections = []
        for r in rows:
            db_ts = r[3]
            if db_ts and "Z" not in db_ts and "+" not in db_ts:
                ts_formatted = db_ts.replace(" ", "T") + "Z"
            else:
                ts_formatted = db_ts
                
            detections.append({
                "id": r[0],
                "telemetry_log_id": r[1],
                "incident_id": r[2],
                "timestamp": ts_formatted,
                "class_name": r[4],
                "confidence": r[5],
                "bbox": [r[6], r[7], r[8], r[9]], # x_min, y_min, x_max, y_max
                "latitude": r[10],
                "longitude": r[11],
                "frame_path": r[12]
            })
        return detections
    except Exception as e:
        logger.error("Error fetching detections: {}".format(e))
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        cursor.close()
        conn.close()

@app.get("/api/stats")
def get_dashboard_stats(request: Request):
    """Retrieves aggregate telemetry and spatial count metrics for the widgets."""
    user = get_session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    try:
        # Total incidents and high-severity count (EXTREME + SEVERE)
        cursor.execute("SELECT COUNT(*), SUM(CASE WHEN severity IN ('EXTREME', 'SEVERE') THEN 1 ELSE 0 END) FROM incidents")
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
        logger.error("Error fetching database statistics: {}".format(e))
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        cursor.close()
        conn.close()

@app.get("/api/report/pdf")
def export_pdf_report(request: Request,
                      severity                = Query(None, description="Filter by severity"),
                      mission_id      = Query("BRH-01", description="Mission identifier")):
    """
    Generates and streams a PDF incident report.
    Includes incident table, evidence gallery, and GPS coordinate appendix.
    """
    user = get_session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    conn   = db_manager.get_connection()
    cursor = conn.cursor()
    try:
        query  = "SELECT id, timestamp, centroid_latitude, centroid_longitude, severity, illegal_zone, distance_to_river_m, evidence_image_path FROM incidents"
        params = []
        if severity:
            ph = "?" if db_manager.db_type == "sqlite" else "%s"
            query += " WHERE severity = {}".format(ph)
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
    filename = "sand_mining_report_{}_{}.pdf".format(mission_id, ts)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="{}"'.format(filename)}
    )


#  FLIGHT CONTROL APIS (MID-FLIGHT SWITCHING & DYNAMIC GEOFENCING) 
@app.get("/api/flight/config")
def get_flight_config(request: Request):
    """
    WHAT: REST endpoint returning active flight mission control config.
    WHY: Checked periodically by the drone edge pipeline to load the correct
    model mid-flight and monitor geofenced start coordinates!
    """
    user = get_session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return flight_config


# ============================================================
# MODEL MANAGEMENT — Upload & List Custom YOLO Weights
# ============================================================

from fastapi import UploadFile, File, Form

@app.post("/api/model/upload")
async def upload_model(request: Request, file: UploadFile = File(...)):
    """
    WHAT: Accepts a multipart/form-data file upload of a .pt YOLO weight file.
    WHY:  Operators can upload their custom-trained model (e.g., trained on
          specific sand-mining imagery) without needing SSH access to the server.
    HOW:
        - FastAPI's UploadFile is a thin async wrapper over Python's SpooledTemporaryFile.
          It reads the incoming HTTP body in chunks using Starlette's form-data parser,
          which streams directly from the socket rather than loading the whole file
          into memory at once. This is critical because a YOLO .pt file is typically
          100MB–300MB.
        - We use aiofiles-compatible reads: `await file.read()` reads the entire
          spooled buffer. For very large files you could do chunked reads, but for
          model files this is fine.
        - The file is saved into the same `models/weights/` directory that
          `_yolo_inference_loop` already resolves from. This means the newly
          uploaded file is immediately available for hot-swap with zero code
          path changes elsewhere.
    """
    user = get_session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Security: ONLY allow .pt files. Strip any path traversal attack.
    # os.path.basename eliminates directory components like ../../etc/passwd
        filename = Path(file.filename).name
    ALLOWED_EXTENSIONS = {".pt", ".onnx", ".engine", ".torchscript"}
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported format '{}'. Allowed: {}".format(
                Path(filename).suffix, ", ".join(ALLOWED_EXTENSIONS)
            )
        )

    weights_dir = Path(__file__).resolve().parent.parent.parent / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    dest_path = weights_dir / filename

    # Read the uploaded bytes from the socket buffer into memory, then flush to disk.
    # `await file.read()` is non-blocking. The OS write below IS blocking, but it's
    # fast because we're just writing to local disk.
    contents = await file.read()
    with open(dest_path, "wb") as f:
        f.write(contents)

    file_size_mb = round(len(contents) / (1024 * 1024), 2)
    logger.info("Model uploaded: {} ({} MB) -> {}".format(filename, file_size_mb, dest_path))

    return {
        "status": "ok",
        "filename": filename,
        "size_mb": file_size_mb,
        "message": "Model saved. Select '{}' from the dropdown to activate it.".format(filename)
    }


@app.get("/api/model/list")
def list_models(request: Request):
    """
    WHAT: Returns a JSON list of all .pt weight files available on disk.
    WHY:  The frontend fetches this on page-load to dynamically populate the
          <select> dropdown. This means the dropdown is always in sync with
          what files are actually present on the server — no hardcoding.
    HOW:
        - Path.glob("*.pt") walks the OS directory inode table for files matching
          the pattern. This is an O(n) syscall on the directory, not a recursive
          tree walk, so it's extremely fast regardless of how many model files exist.
        - We also return file sizes so the operator knows if they uploaded a
          real model or accidentally uploaded something empty/corrupt.
    """
    user = get_session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    weights_dir = Path(__file__).resolve().parent.parent.parent / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    models = []
    for pt_file in sorted(
        f for ext in ("*.pt", "*.onnx", "*.engine", "*.torchscript")
        for f in weights_dir.glob(ext)
    ):
        models.append({
            "filename": pt_file.name,
            "size_mb": round(pt_file.stat().st_size / (1024 * 1024), 2)
        })

    # Always ensure the baseline yolov8n.pt is listed even if not yet downloaded
    base_names = [m["filename"] for m in models]
    if "yolov8n.pt" not in base_names:
        models.insert(0, {"filename": "yolov8n.pt", "size_mb": 0, "note": "auto-download on first use"})

    return {"models": models}

@app.post("/api/flight/config")
async def update_flight_config(request: Request, data: dict):
    """
    WHAT: Endpoint to update active model and geofencing coordinates.
    WHY: Operators can switch YOLOv8 vs YOLOv10 mid-flight or adjust the trigger geofence!
    """
    user = get_session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    global flight_config, has_reached_starting_spot
    
    # Save old coordinates to check if they actually changed
    old_lat = flight_config["start_lat"]
    old_lng = flight_config["start_lng"]
    old_rad = flight_config["start_radius_meters"]
    new_lat = float(data.get("start_lat", flight_config["start_lat"]))
    new_lng = float(data.get("start_lng", data.get("start_lon", flight_config["start_lng"])))
    new_rad = float(data.get("start_radius_meters", flight_config["start_radius_meters"]))

    flight_config["active_model"]        = data.get("active_model", flight_config["active_model"])
    flight_config["start_lat"]           = new_lat
    flight_config["start_lng"]           = new_lng
    flight_config["start_radius_meters"] = new_rad
    flight_config["video_source"]        = data.get("video_source", flight_config.get("video_source", ""))
    flight_config["detection_enabled"]   = bool(data.get("detection_enabled", flight_config["detection_enabled"]))
    
    # Reset starting spot trigger only if starting geofence coordinates/radius actually changed!
    if old_lat != new_lat or old_lng != new_lng or old_rad != new_rad:
        has_reached_starting_spot = False
        logger.info(" Geofence start coordinates updated  resetting starting spot trigger.")
    
    logger.info(" Updated Flight Configuration: {}".format(flight_config))
    
    # Broadcast to all connected WebSocket dashboards so map and parameters update instantly!
    await manager.broadcast({
        "type": "flight_config_update",
        "payload": flight_config
    })
    return {"status": "ok", "config": flight_config}


#  POSTGRES VPS DIRECT IMAGE RETRIEVAL API 
@app.get("/api/evidence/db/{incident_id}")
def get_evidence_image_from_db(request: Request, incident_id):
    """
    WHAT: Retrieves binary JPEG data directly from PostgreSQL / SQLite blob storage.
    WHY: Allows serving evidence snapshots to the frontend HTML direct from the DB
    without any dependency on the server file system! Includes seamless local
    filesystem fallback for offline/local development.
    """
    user = get_session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    ph = "%s" if db_manager.db_type == "postgresql" else "?"
    
    try:
        # 1. Try reading the blob and filesystem path from database
        cursor.execute("SELECT evidence_image_blob, evidence_image_path FROM incidents WHERE id = {}".format(ph), (incident_id,))
        row = cursor.fetchone()
        
        if row:
            blob = row[0]
            path_str = row[1]
            
            # If binary blob exists, serve it immediately (production/VPS mode)
            if blob:
                blob_bytes = bytes(blob) if isinstance(blob, (memoryview, bytes)) else blob
                return Response(content=blob_bytes, media_type="image/jpeg")
            
            # Fallback: If blob is empty but path exists in DB, load local file (local development mode)
            if path_str:
                project_root = Path(__file__).resolve().parent.parent.parent
                file_path = project_root / path_str
                if file_path.exists():
                    with open(file_path, "rb") as f:
                        return Response(content=f.read(), media_type="image/jpeg")
                        
        # 2. Hard fallback: Search local detections folder directly by incident ID
        project_root = Path(__file__).resolve().parent.parent.parent
        detections_dir = project_root / "data" / "detections"
        # Search for files matching evidence_{incident_id}*.jpg or cloud_evidence_{incident_id}*.jpg
        matches = list(detections_dir.glob("evidence_{}*.jpg".format(incident_id)))
        if not matches:
            matches = list(detections_dir.glob("cloud_evidence_{}*.jpg".format(incident_id)))
        if matches:
            with open(matches[0], "rb") as f:
                return Response(content=f.read(), media_type="image/jpeg")
                
        raise HTTPException(status_code=404, detail="Incident evidence image not found")
    except Exception as e:
        logger.error("Error serving image from DB: {}".format(e))
        # Try search fallback directly in case of schema/SQL failures
        try:
            project_root = Path(__file__).resolve().parent.parent.parent
            detections_dir = project_root / "data" / "detections"
            matches = list(detections_dir.glob("evidence_{}*.jpg".format(incident_id)))
            if not matches:
                matches = list(detections_dir.glob("cloud_evidence_{}*.jpg".format(incident_id)))
            if matches:
                with open(matches[0], "rb") as f:
                    return Response(content=f.read(), media_type="image/jpeg")
        except Exception:
            pass
            
        raise HTTPException(status_code=500, detail="Database failure fetching image blob")
    finally:
        cursor.close()
        conn.close()

@app.get("/api/evidence/{filename}")
def get_evidence_image(request: Request, filename):
    """Serves a specific evidence JPEG image by filename."""
    user = get_session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    img_path = EVIDENCE_DIR / filename
    if not img_path.exists() or not filename.endswith(".jpg"):
        raise HTTPException(status_code=404, detail="Evidence image not found")
    with open(img_path, "rb") as f:
        return Response(content=f.read(), media_type="image/jpeg")


@app.get("/api/zone/radius")
def get_zone_radius(request: Request):
    """Returns the currently active buffer radius in metres."""
    user = get_session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"radius_m": active_buffer_radius_m}


@app.post("/api/zone/radius")
async def set_zone_radius(request: Request, data: dict):
    """
    Updates the active zone enforcement radius.
    1. Rebuilds river_buffer_1km.geojson with the new radius (server-side)
    2. Broadcasts the change over WebSocket so:
       - The browser map redraws its Turf.js buffer to match
       - The Jetson sync_worker can detect the change and reload its ClusterEngine
    """
    global active_buffer_radius_m, global_cluster_engine

    user = get_session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
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
            raise HTTPException(status_code=500, detail="Buffer rebuild failed  check centerline data")

    active_buffer_radius_m = radius_m
    logger.info("Zone radius updated to {}m by operator".format(radius_m))

    # Broadcast to all dashboard clients + Jetson sync_worker
    await manager.broadcast({
        "type": "zone_radius_update",
        "payload": {"radius_m": radius_m}
    })

    return {"status": "ok", "radius_m": radius_m}


# WebSocket endpoint
# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Extract session_id from cookie to verify connection
    session_id = websocket.cookies.get("session_id")
    if not session_id or session_id not in ACTIVE_SESSIONS:
        await websocket.close(code=1008) # Policy Violation
        return

    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, listen for any client messages if needed
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        manager.disconnect(websocket)

# Auth API Endpoints
@app.post("/api/auth/login")
async def login(request: Request, response: Response, payload: dict):
    username = payload.get("username")
    password = payload.get("password")
    
    if not username or not password:
        raise HTTPException(status_code=400, detail="Missing username or password")
        
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    is_pg = db_manager.db_type == "postgresql"
    query = "SELECT password_hash, role FROM users WHERE username = %s;" if is_pg else "SELECT password_hash, role FROM users WHERE username = ?;"
    
    try:
        cursor.execute(query, (username,))
        row = cursor.fetchone()
    except Exception as e:
        logger.error("Error querying user: {}".format(e))
        raise HTTPException(status_code=500, detail="Database lookup failure")
    finally:
        cursor.close()
        conn.close()
        
    if not row or not verify_password(password, row[0]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
        
    role = row[1]
    
    # Create session
    session_id = uuid.uuid4().hex
    ACTIVE_SESSIONS[session_id] = {
        "username": username,
        "role": role
    }
    
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        max_age=3600 * 24, # 24 hours
        samesite="lax",
        secure=False
    )
    
    return {"status": "success", "username": username, "role": role}

ALLOWED_EMAIL_DOMAINS = [".gov", ".gov.in", ".edu", ".edu.in", ".ac.in", ".org", "gmail.com"]

@app.post("/api/auth/register")
async def register(request: Request, response: Response, payload: dict):
    username = payload.get("username", "").strip()
    email = payload.get("email", "").strip().lower()
    password = payload.get("password", "")
    
    if not username or not email or not password:
        raise HTTPException(status_code=400, detail="Missing required registration fields")
        
    # Domain whitelist check
    is_valid_domain = False
    for domain in ALLOWED_EMAIL_DOMAINS:
        if email.endswith(domain):
            is_valid_domain = True
            break
            
    if not is_valid_domain:
        raise HTTPException(
            status_code=400,
            detail="This is not an authorized email address. Access is restricted to trusted domains (.gov, .edu, .org, gmail.com)."
        )
        
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    is_pg = db_manager.db_type == "postgresql"
    
    # Check if username or email already exists
    query = "SELECT id FROM users WHERE username = %s OR email = %s;" if is_pg else "SELECT id FROM users WHERE username = ? OR email = ?;"
    try:
        cursor.execute(query, (username, email))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Username or email is already registered")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error("Error checking user registration: {}".format(e))
        raise HTTPException(status_code=500, detail="Database lookup error")
    finally:
        cursor.close()
        conn.close()
        
    # Hash password
    salt = uuid.uuid4().hex
    hashed = hashlib.sha256((salt + password).encode('utf-8')).hexdigest()
    password_hash = "{}:{}".format(salt, hashed)
    
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    insert_query = (
        "INSERT INTO users (username, email, password_hash, role) VALUES (%s, %s, %s, %s);"
        if is_pg else
        "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?);"
    )
    
    try:
        cursor.execute(insert_query, (username, email, password_hash, "operator"))
        conn.commit()
    except Exception as e:
        logger.error("Error inserting registered user: {}".format(e))
        raise HTTPException(status_code=500, detail="Registration save failure")
    finally:
        cursor.close()
        conn.close()
        
    # Create active session
    session_id = uuid.uuid4().hex
    ACTIVE_SESSIONS[session_id] = {
        "username": username,
        "role": "operator"
    }
    
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        max_age=3600 * 24, # 24 hours
        samesite="lax",
        secure=False
    )
    
    return {"status": "success", "username": username, "role": "operator"}

@app.post("/api/auth/logout")
async def logout(request: Request, response: Response):
    session_id = request.cookies.get("session_id")
    if session_id in ACTIVE_SESSIONS:
        del ACTIVE_SESSIONS[session_id]
    response.delete_cookie("session_id")
    return {"status": "success"}

@app.get("/api/auth/status")
async def get_auth_status(request: Request):
    user = get_session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

@app.get("/login", response_class=HTMLResponse)
def get_login_page(request: Request):
    """Serves the login page, redirects to dashboard if already authenticated."""
    user = get_session_user(request)
    if user:
        return RedirectResponse(url=request.scope.get("root_path", "") + "/", status_code=303)
        
    login_path = Path(__file__).resolve().parent / "frontend" / "login.html"
    if login_path.exists():
        with open(login_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Login Page Not Found</h1>", status_code=404)

# HTML Server
@app.get("/", response_class=HTMLResponse)
def get_dashboard_page(request: Request):
    """Serves the unified, premium dark-themed operator control dashboards."""
    user = get_session_user(request)
    if not user:
        return RedirectResponse(url=request.scope.get("root_path", "") + "/login", status_code=303)
        
    dashboard_path = Path(__file__).resolve().parent / "frontend" / "index.html"
    if dashboard_path.exists():
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        # Fallback basic response if html is missing during initial boot
        return HTMLResponse(content="<h1>Dashboard Page Loading...</h1><p>Please implement frontend/index.html first.</p>")

# Admin Flight Recording Endpoints
@app.post("/api/admin/record/start")
async def start_recording(request: Request):
    user = get_session_user(request)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden: Admin privilege required.")
        
    global is_recording, recording_writer, recording_start_time, recording_filepath, recording_filename, recording_lock, use_synthetic_video
    import cv2
    with recording_lock:
        if use_synthetic_video:
            raise HTTPException(
                status_code=400,
                detail="Cannot start recording: No active drone camera feed is connected (currently in standby)."
            )
            
        if is_recording:
            return {"status": "already_recording", "filename": recording_filename}
            
        import datetime
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_filename = "flight_rec_{}.mp4".format(timestamp_str)
        recordings_dir = Path(__file__).resolve().parent.parent.parent / "data" / "recordings"
        recordings_dir.mkdir(parents=True, exist_ok=True)
        recording_filepath = recordings_dir / recording_filename
        
        # Try to use 'avc1' (H.264) for direct native HTML5 browser playback support.
        # Fall back to standard 'mp4v' if the system's OpenCV has no H.264 encoder.
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            recording_writer = cv2.VideoWriter(str(recording_filepath), fourcc, 15.0, (global_video_w, global_video_h))
            if not recording_writer.isOpened():
                raise RuntimeError("avc1 writer failed to open")
        except Exception:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            recording_writer = cv2.VideoWriter(str(recording_filepath), fourcc, 15.0, (global_video_w, global_video_h))
        
        if not recording_writer.isOpened():
            recording_writer = None
            raise HTTPException(status_code=500, detail="Failed to initialize video writer.")
            
        recording_start_time = time.time()
        is_recording = True
        logger.info(" Admin Flight Recording Started: {} ({}x{})".format(recording_filepath, global_video_w, global_video_h))
        return {"status": "started", "filename": recording_filename}

@app.post("/api/admin/record/stop")
async def stop_recording(request: Request):
    user = get_session_user(request)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden: Admin privilege required.")
        
    global is_recording, recording_writer, recording_start_time, recording_filepath, recording_filename, recording_lock
    with recording_lock:
        if not is_recording or recording_writer is None:
            raise HTTPException(status_code=400, detail="No active flight recording to stop.")
            
        is_recording = False
        recording_writer.release()
        recording_writer = None
        
        duration = round(time.time() - recording_start_time, 1)
        
        # Transcode raw mp4v video to standard H.264 (libx264) using FFmpeg
        # so that it is natively playable inside HTML5 <video> tags in all modern web browsers.
        if recording_filepath.exists():
            try:
                import subprocess
                temp_filepath = recording_filepath.parent / "temp_{}".format(recording_filename)
                logger.info("Transcoding raw recorded video to browser-compatible H.264...")
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(recording_filepath),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-an",
                    str(temp_filepath)
                ]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20.0)
                if res.returncode == 0 and temp_filepath.exists():
                    recording_filepath.unlink(missing_ok=True)
                    temp_filepath.rename(recording_filepath)
                    logger.info("Transcoding completed successfully! Video file converted to H.264 AVC.")
                else:
                    logger.warning("FFmpeg transcoding failed (returncode {}): {}".format(res.returncode, res.stderr))
            except Exception as ex:
                logger.error("Failed to transcode recorded video to H.264: {}".format(ex))
                
        file_size = 0
        if recording_filepath.exists():
            file_size = recording_filepath.stat().st_size
            
        # Save to DB
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        is_pg = db_manager.db_type == "postgresql"
        ph = "%s" if is_pg else "?"
        
        cursor.execute(
            "INSERT INTO recordings (filename, filepath, duration_seconds, size_bytes) VALUES ({}, {}, {}, {})".format(ph, ph, ph, ph),
            (recording_filename, "data/recordings/{}".format(recording_filename), duration, file_size)
        )
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(" Admin Flight Recording Saved: {} ({}s, {} bytes)".format(recording_filename, duration, file_size))
        return {
            "status": "stopped",
            "filename": recording_filename,
            "duration_seconds": duration,
            "size_bytes": file_size
        }

@app.get("/api/admin/recordings")
async def list_recordings(request: Request):
    user = get_session_user(request)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden: Admin privilege required.")
        
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, filename, filepath, duration_seconds, size_bytes FROM recordings ORDER BY id DESC")
    rows = cursor.fetchall()
    
    recordings = []
    for r in rows:
        ts_val = r[1]
        ts_formatted = ts_val.replace(" ", "T") + "Z" if ts_val and "Z" not in ts_val and "+" not in ts_val else ts_val
        recordings.append({
            "id": r[0],
            "timestamp": ts_formatted,
            "filename": r[2],
            "filepath": r[3],
            "duration_seconds": r[4],
            "size_bytes": r[5]
        })
    cursor.close()
    conn.close()
    return recordings

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
