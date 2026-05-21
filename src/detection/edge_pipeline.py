import os
import sys
import json
import time
import math
import random
import cv2
import numpy as np
import requests
from datetime import datetime
from pathlib import Path
import logging
from threading import Thread
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add directories to system path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / "src" / "preprocess"))
sys.path.append(str(project_root / "src" / "detection"))

# pyrefly: ignore [missing-import]
from db_setup import DatabaseManager
# pyrefly: ignore [missing-import]
from drone_simulator import DroneSimulator
from gps_projector import pixel_to_gps
from cluster_engine import ClusterEngine
from evidence_engine import save_incident_evidence
from sync_worker import SyncWorker

class EdgePipeline:
    """
    Simulates the entire Jetson Nano Edge compute flow running on the drone:
    Telemetry -> Simulated AI Inference -> Coordinate Projection -> Spatial DBSCAN -> Local DB Logging -> Cloud Upload.
    """
    def __init__(self, cloud_url="http://localhost:8000"):
        self.cloud_url = cloud_url
        self.db_manager = DatabaseManager(db_type="sqlite")
        self.db_manager.initialize_database()
        
        # Initialize cluster engine
        self.cluster_engine = ClusterEngine(db_manager=self.db_manager)
        
        # Load drone simulator flight path
        self.drone_sim = DroneSimulator(
            db_manager=self.db_manager,
            speed_kmh=42.0,
            altitude_m=70.0
        )
        
        self.running = False
        self.frame_w, self.frame_h = 1920, 1080  # 1080p canvas for performance

        # Offline-first sync worker — starts as daemon, retries with backoff
        self.sync_worker = SyncWorker(
            db_manager=self.db_manager,
            cloud_url=self.cloud_url,
            sync_interval_s=5.0
        )

        # ── WHAT: NEW ACTIVE PARAMETERS FOR PRODUCT LEVEL DEPLOYMENT ──────
        # ── WHY: Tracks dynamically loaded models and geofence states.
        self.yolo_model = None
        self.active_model_name = None
        
        # Dynamic Geofence starting coordinates - default to 0.0 (idle geofence)
        self.target_model = "yolov8n.pt"
        self.start_lat = 0.0
        self.start_lon = 0.0
        self.start_radius = 500.0
        self.detection_enabled = False
        
        # Track if we have already generated our dynamic test path for takeoff simulation
        self.dynamic_path_generated = False
        self.current_flight_idx = 0

    def load_yolo_model(self, model_name: str):
        """Dynamically loads/swaps the active YOLO model weights mid-flight."""
        if self.active_model_name == model_name and self.yolo_model is not None:
            return  # Already loaded
            
        logger.info(f"🔄 Switching model mid-flight: {self.active_model_name} -> {model_name}")
        try:
            from ultralytics import YOLO
            weights_path = Path(__file__).resolve().parent.parent.parent / "models" / "weights" / model_name
            
            # If standard weight is missing locally, YOLO automatically downloads it
            if weights_path.exists():
                self.yolo_model = YOLO(str(weights_path))
            else:
                self.yolo_model = YOLO(model_name)
                
            self.active_model_name = model_name
            logger.info(f"✅ Active model successfully switched to: {model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load model weights {model_name}: {e}")

    def check_geofence_trigger(self, drone_lat: float, drone_lon: float) -> bool:
        """Calculates distance to starting point and returns True if inside start geofence."""
        if self.start_lat == 0.0 or self.start_lon == 0.0:
            return False

        lat1, lon1 = math.radians(drone_lat), math.radians(drone_lon)
        lat2, lon2 = math.radians(self.start_lat), math.radians(self.start_lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance_meters = 6371000.0 * c  # Earth radius ~6,371,000 meters
        return distance_meters <= self.start_radius

    def generate_simulated_detections(self, drone_lat: float, drone_lon: float, step: int) -> List[Dict[str, Any]]:
        """
        Periodically generates mock illegal sand mining target clusters in the field of view:
        e.g., Trucks, Excavators (JCBs), and Workers.
        """
        detections = []
        
        # We spawn a sand mining site cluster with a 12% probability at any given step,
        # but only if step is not near the start to let the flight stabilize.
        # Once spawned, it stays active for a few seconds to simulate the drone flying over it.
        # Let's check step intervals to create 3 separate distinct mining clusters along the flight path.
        is_cluster_active = False
        cluster_type = "MEDIUM"
        
        if 20 <= step <= 35:
            is_cluster_active = True
            cluster_type = "CRITICAL"  # Trucks, JCBs, and Workers
        elif 60 <= step <= 75:
            is_cluster_active = True
            cluster_type = "HIGH"      # JCB + Workers
        elif 95 <= step <= 110:
            is_cluster_active = True
            cluster_type = "LOW"       # People only (recreational or small scale)

        if not is_cluster_active:
            return []

        # Define cluster centers relative to drone lat/lon
        random.seed(step // 4) # Group points together across consecutive frames
        
        # Let's spawn 2-5 elements inside the cluster
        num_items = 5 if cluster_type == "CRITICAL" else 3 if cluster_type == "HIGH" else 2
        
        classes = []
        if cluster_type == "CRITICAL":
            classes = ["jcb", "truck", "person", "person", "truck"]
        elif cluster_type == "HIGH":
            classes = ["jcb", "person", "person"]
        else:
            classes = ["person", "person"]

        for idx in range(num_items):
            cls_name = classes[idx]
            
            # Place bounding box pixels within the camera frame
            # Center of the frame is (960, 540)
            px_x = int(960 + random.uniform(-400, 400))
            px_y = int(540 + random.uniform(-300, 300))
            
            # Box width and height
            box_w = int(random.uniform(80, 160)) if cls_name != "person" else int(random.uniform(30, 60))
            box_h = int(random.uniform(80, 160)) if cls_name != "person" else int(random.uniform(60, 100))
            
            bbox_x_min = max(0, px_x - box_w // 2)
            bbox_y_min = max(0, px_y - box_h // 2)
            bbox_x_max = min(self.frame_w, px_x + box_w // 2)
            bbox_y_max = min(self.frame_h, px_y + box_h // 2)
            
            # Project this bounding box pixel to lat/lon using the active drone state
            # Drone is looking straight down (-90) or slightly forward (-70)
            lat, lon = pixel_to_gps(
                bbox_center_px=(px_x, px_y),
                drone_gps=(drone_lat, drone_lon),
                altitude_m=70.0,
                gimbal_pitch=-80.0,
                gimbal_yaw=self.drone_sim.flight_points[step % len(self.drone_sim.flight_points)]['heading'],
                img_size_px=(self.frame_w, self.frame_h)
            )
            
            detections.append({
                'class_name': cls_name,
                'confidence': float(round(random.uniform(0.78, 0.96), 2)),
                'bbox_x_min': bbox_x_min,
                'bbox_y_min': bbox_y_min,
                'bbox_x_max': bbox_x_max,
                'bbox_y_max': bbox_y_max,
                'lat': lat,
                'lon': lon
            })
            
        # Reset seed for normal random drift
        random.seed()
        return detections

    def draw_edge_overlay_canvas(self, telemetry: Dict[str, Any], detections: List[Dict[str, Any]], step: int) -> Tuple[bytes, bytes]:
        """
        Creates two beautiful simulated video feeds on the Jetson Nano:
        1. Raw Video: Simulated high-altitude orthophoto ground background with altimeter overlays.
        2. Annotated Video: Raw background with YOLO bounding box layers.
        """
        # 1. Create a tactical synthetic background (dark grid to simulate camera feed)
        bg = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        bg[:, :] = [10, 15, 25]  # Very dark indigo base
        
        # Draw nice spatial mapping grids
        for x in range(0, self.frame_w, 80):
            cv2.line(bg, (x, 0), (x, self.frame_h), (255, 255, 255, 10), 1)
        for y in range(0, self.frame_h, 80):
            cv2.line(bg, (0, y), (self.frame_w, y), (255, 255, 255, 10), 1)

        # Draw a synthetic river representation scrolling across the screen
        # Brahmaputra water body (deep cyan)
        river_pts = np.array([
            [0, 800], [500, 650], [1000, 600], [1500, 480], [1920, 400],
            [1920, 800], [1500, 880], [1000, 920], [500, 950], [0, 1000]
        ], dtype=np.int32)
        cv2.fillPoly(bg, [river_pts], (40, 60, 25))

        # Copy background for raw stream
        raw_canvas = bg.copy()
        
        # Add basic HUD details to raw canvas (Crosshairs, Altimeter tape)
        cv2.circle(raw_canvas, (960, 540), 100, (0, 240, 255, 40), 1)
        cv2.line(raw_canvas, (960, 400), (960, 680), (0, 240, 255, 20), 1)
        cv2.line(raw_canvas, (800, 540), (1120, 540), (0, 240, 255, 20), 1)
        
        # Draw tech readout on raw feed
        cv2.putText(raw_canvas, f"DJI M300 | 4K CAM01 | FOCAL: 24MM", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 240, 255), 1)
        cv2.putText(raw_canvas, f"LAT: {telemetry['lat']:.6f} LON: {telemetry['lon']:.6f}", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(raw_canvas, f"ALT AGL: {telemetry['altitude']:.1f} M", (1600, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 240, 255), 1)
        cv2.putText(raw_canvas, f"SPEED: {telemetry['speed']*3.6:.1f} KM/H", (1600, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 240, 255), 1)

        # 2. Draw Bounding boxes on annotated canvas
        annotated_canvas = raw_canvas.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox_x_min'], det['bbox_y_min'], det['bbox_x_max'], det['bbox_y_max']
            cls = det['class_name']
            conf = det['confidence']
            
            # Select target bounding box color: Green (person), Amber/Yellow (JCB), Red (Critical)
            color = (0, 240, 255) # Cyan default for truck
            if cls == "person":
                color = (0, 230, 100) # Green for personnel
            elif cls == "jcb":
                color = (0, 180, 245) # Amber for JCB

            # Draw glowing double rectangle
            cv2.rectangle(annotated_canvas, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(annotated_canvas, (x1-2, y1-2), (x2+2, y2+2), (255, 255, 255, 20), 1)
            
            # Bounding box tag details (filtering indicators mapped out)
            label = f"{cls.upper()} {conf*100:.0f}%"
            cv2.putText(annotated_canvas, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            
            # Draw tiny coordinate text below box
            coord_str = f"{det['lat']:.5f}, {det['lon']:.5f}"
            cv2.putText(annotated_canvas, coord_str, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

        # Encode canvases as JPEG buffers
        _, raw_encoded = cv2.imencode('.jpg', raw_canvas)
        _, overlay_encoded = cv2.imencode('.jpg', annotated_canvas)
        
        return raw_encoded.tobytes(), overlay_encoded.tobytes()

    def run_pipeline(self, steps=130):
        """Runs the entire edge-cloud streaming simulation loop."""
        self.running = True
        logger.info(f"Edge Computing Pipeline started on Jetson Nano. Cloud Sync: {self.cloud_url}")

        # Start the resilient offline-first background sync worker
        self.sync_worker.start()
        
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        is_pg = self.db_manager.db_type == "postgresql"
        insert_telemetry_sql = """
        INSERT INTO telemetry_logs (
            timestamp, latitude, longitude, altitude_agl, 
            gimbal_pitch, gimbal_yaw, gimbal_roll, 
            drone_speed, battery_percentage, gps_accuracy_m
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
        """ if is_pg else """
        INSERT INTO telemetry_logs (
            timestamp, latitude, longitude, altitude_agl, 
            gimbal_pitch, gimbal_yaw, gimbal_roll, 
            drone_speed, battery_percentage, gps_accuracy_m
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        try:
            battery = 100.0
            
            for step in range(steps):
                if not self.running:
                    break
                    
                # ── A. GET CONFIG FROM VPS (MID-FLIGHT UPDATE) ──
                if step % 5 == 0:
                    try:
                        r = requests.get(f"{self.cloud_url}/api/flight/config", timeout=0.5)
                        if r.status_code == 200:
                            cfg = r.json()
                            self.target_model      = cfg.get("active_model", self.target_model)
                            self.start_lat         = cfg.get("start_lat", self.start_lat)
                            self.start_lon         = cfg.get("start_lng", self.start_lon)
                            self.start_radius      = cfg.get("start_radius_meters", self.start_radius)
                            self.detection_enabled = cfg.get("detection_enabled", self.detection_enabled)
                            
                            # ── DYNAMIC TEST PATH TRIGGER ──
                            # WHAT: If a start geofence coordinate is received and we haven't generated our 
                            # launch-to-target path yet, generate it now!
                            # WHY: Triggers takeoff simulation from random starting coordinates.
                            if self.start_lat != 0.0 and self.start_lon != 0.0 and not self.dynamic_path_generated:
                                self.drone_sim.generate_dynamic_test_path(
                                    target_lat=self.start_lat,
                                    target_lon=self.start_lon,
                                    start_radius_meters=self.start_radius
                                )
                                self.dynamic_path_generated = True
                                # Reset loop index to start dynamic flight from Takeoff home base!
                                self.current_flight_idx = 0
                    except Exception:
                        pass # Fallback to current settings if VPS link is down

                # 1. Telemetry Step
                point_idx = self.current_flight_idx % len(self.drone_sim.flight_points)
                point = self.drone_sim.flight_points[point_idx]
                
                lat, lon = point['lat'], point['lon']
                alt = 70.0 + random.uniform(-1.0, 1.0)
                speed = self.drone_sim.speed_mps
                heading = point['heading']
                gimbal_pitch = -80.0
                battery = max(0.0, battery - 0.04)
                
                timestamp = datetime.now().isoformat()
                
                # Save Telemetry Locally to edge DB first! (Resilient Offline DB logging)
                tele_params = (
                    timestamp, lat, lon, alt,
                    gimbal_pitch, heading, 0.0,
                    speed, int(battery), 0.15
                )
                cursor.execute(insert_telemetry_sql, tele_params)
                
                if is_pg:
                    telemetry_id = cursor.fetchone()[0]
                else:
                    telemetry_id = cursor.lastrowid
                conn.commit()
                
                # Telemetry dictionary for frame overlays
                telemetry_dict = {
                    'lat': lat, 'lon': lon, 'altitude': alt, 
                    'speed': speed, 'heading': heading, 'timestamp': timestamp, 'battery': int(battery)
                }

                # ── B. GEOFENCED INFERENCE FILTER (NO HARDCODING) ──
                is_active = self.check_geofence_trigger(lat, lon) and self.detection_enabled
                
                raw_detections = []
                incidents = []
                if is_active:
                    # Hot-load active YOLO model (YOLOv8 vs YOLOv10) mid-flight if needed
                    self.load_yolo_model(self.target_model)
                    
                    # 2. Simulated YOLO Object Detection & GPS Projection (User requirement #2)
                    raw_detections = self.generate_simulated_detections(lat, lon, step)
                    
                    # 3. Spatial Aggregation & DBSCAN Clustering
                    # Groups trucks/workers/excavators within 60m into single unified incident zones
                    incidents = self.cluster_engine.cluster_detections(raw_detections, eps_meters=60.0)
                    
                    # Save Detections and Incidents locally to edge DB! (User requirement #1)
                    self.cluster_engine.save_incidents_to_db(incidents, telemetry_log_id=telemetry_id)

                # 4. Generate Video Feeds (Raw vs Overlay)
                raw_jpeg, overlay_jpeg = self.draw_edge_overlay_canvas(telemetry_dict, raw_detections, step)

                # 5. Save Evidence Snapshots to Jetson SSD (offline-first, always runs)
                if incidents:
                    # Decode overlay jpeg back to numpy for cropping
                    overlay_np = cv2.imdecode(np.frombuffer(overlay_jpeg, np.uint8), cv2.IMREAD_COLOR)
                    for inc in incidents:
                        evidence_paths = save_incident_evidence(
                            annotated_frame=overlay_np,
                            incident=inc,
                            telemetry=telemetry_dict
                        )
                        # Update incident record with first evidence image path
                        if evidence_paths:
                            inc['evidence_image_path'] = evidence_paths[0]
                            try:
                                ev_conn = self.db_manager.get_connection()
                                ev_cur  = ev_conn.cursor()
                                ph = '?' if self.db_manager.db_type == 'sqlite' else '%s'
                                ev_cur.execute(
                                    f"UPDATE incidents SET evidence_image_path = {ph} "
                                    f"WHERE id = (SELECT MAX(id) FROM incidents WHERE "
                                    f"ABS(centroid_latitude - {inc['centroid_lat']}) < 0.0001)",
                                    (evidence_paths[0],)
                                )
                                ev_conn.commit()
                                ev_cur.close()
                                ev_conn.close()
                            except Exception as e:
                                logger.debug(f"Evidence path DB update: {e}")

                # 6. Cloud Streaming (best-effort — sync_worker handles reliable retry)
                # Try to POST real-time frames & telemetry updates to FastAPI server
                try:
                    # Upload Raw Video Frame
                    requests.post(
                        f"{self.cloud_url}/api/edge/frame?stream_type=raw",
                        data=raw_jpeg,
                        headers={"Content-Type": "image/jpeg"},
                        timeout=0.1
                    )
                    # Upload Overlay Video Frame
                    requests.post(
                        f"{self.cloud_url}/api/edge/frame?stream_type=overlay",
                        data=overlay_jpeg,
                        headers={"Content-Type": "image/jpeg"},
                        timeout=0.1
                    )
                    
                    # Send Telemetry log sync update via API
                    requests.post(
                        f"{self.cloud_url}/api/edge/sync",
                        json={
                            "type": "telemetry",
                            "payload": {
                                "timestamp": timestamp,
                                "lat": lat,
                                "lon": lon,
                                "altitude": alt,
                                "speed": speed,
                                "battery": int(battery)
                            }
                        },
                        timeout=0.1
                    )

                    # Send detection warning sync alerts immediately if any cluster forms
                    for inc in incidents:
                        requests.post(
                            f"{self.cloud_url}/api/edge/sync",
                            json={
                                "type": "detections",
                                "payload": {
                                    "incident_id": step + 1000, # Mock synced index
                                    "severity": inc['severity'],
                                    "centroid_latitude": inc['centroid_lat'],
                                    "centroid_longitude": inc['centroid_lon'],
                                    "detections": inc['detections']
                                }
                            },
                            timeout=0.1
                        )

                except requests.RequestException:
                    # Silent ignore if cloud server is offline - pipeline remains resiliently logging locally!
                    pass

                if step % 20 == 0:
                    logger.info(f"Jetson Nano Status - Frame: {step} | Battery: {int(battery)}% | Detections in frame: {len(raw_detections)}")

                self.current_flight_idx += 1
                time.sleep(0.3)  # Loop at approx 3 FPS for simulation visual clarity
                
        except KeyboardInterrupt:
            logger.info("Pipeline terminated by operator.")
        finally:
            cursor.close()
            conn.close()
            self.running = False
            logger.info("Edge Pipeline shut down successfully.")

if __name__ == "__main__":
    pipeline = EdgePipeline(cloud_url="http://localhost:8000")
    pipeline.run_pipeline(steps=120)
