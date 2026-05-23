"""
Evidence Snapshot Engine — Jetson Nano Edge Component
Saves cropped JPEG evidence images for each detection to local disk.
Works offline-first: all saves go to data/detections/ regardless of cloud connectivity.
Images are named deterministically: evidence_{incident_id}_{class}_{timestamp}.jpg
"""
import cv2
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVIDENCE_DIR = PROJECT_ROOT / "data" / "detections"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


def save_evidence_crop(
    frame: np.ndarray,
    detection: Dict[str, Any],
    incident_id: int,
    telemetry: Dict[str, Any]
) -> Optional[str]:
    """
    Crops the bounding box region from the annotated frame and saves it as JPEG.
    Adds a forensic header band with GPS coords, class, confidence, and timestamp.

    Returns the relative path string saved to DB (e.g. 'data/detections/evidence_42_jcb_....jpg'),
    or None if save failed.
    """
    try:
        x1 = max(0, detection["bbox_x_min"] - 10)   # 10px padding
        y1 = max(0, detection["bbox_y_min"] - 10)
        x2 = min(frame.shape[1], detection["bbox_x_max"] + 10)
        y2 = min(frame.shape[0], detection["bbox_y_max"] + 10)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2].copy()

        # Scale up small crops for readability (min 200px wide)
        h, w = crop.shape[:2]
        if w < 200:
            scale = 200 / w
            crop = cv2.resize(crop, (200, int(h * scale)), interpolation=cv2.INTER_LANCZOS4)

        # Forensic header band (dark bar at bottom with metadata)
        header_h = 36
        canvas = np.zeros((crop.shape[0] + header_h, crop.shape[1], 3), dtype=np.uint8)
        canvas[:crop.shape[0], :] = crop
        canvas[crop.shape[0]:, :] = [12, 18, 32]   # Dark indigo band

        cls   = detection["class_name"].upper()
        conf  = detection["confidence"] * 100
        lat   = detection.get("lat", telemetry.get("lat", 0))
        lon   = detection.get("lon", telemetry.get("lon", 0))
        ts    = datetime.now().strftime("%H:%M:%S")

        label = f"{cls} {conf:.0f}%  |  {lat:.5f}, {lon:.5f}  |  {ts}"
        cv2.putText(canvas, label, (4, crop.shape[0] + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)

        # Draw colored box border on crop area
        color_map = {"jcb": (0, 180, 245), "truck": (0, 200, 255), "person": (0, 200, 100)}
        border_color = color_map.get(detection["class_name"], (255, 255, 255))
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1]-1, crop.shape[0]-1), border_color, 2)

        # Save
        ts_file = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        fname   = f"evidence_{incident_id}_{detection['class_name']}_{ts_file}.jpg"
        fpath   = EVIDENCE_DIR / fname

        cv2.imwrite(str(fpath), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        rel_path = str(fpath.relative_to(PROJECT_ROOT))
        logger.debug(f"Evidence saved: {rel_path}")
        return rel_path

    except Exception as e:
        logger.error(f"Evidence save failed: {e}")
        return None


def save_incident_evidence(
    annotated_frame: np.ndarray,
    incident: Dict[str, Any],
    telemetry: Dict[str, Any]
) -> List[str]:
    """
    Saves evidence crops for ALL detections in an incident cluster.
    Returns list of saved relative paths.
    """
    paths = []
    for det in incident.get("detections", []):
        path = save_evidence_crop(
            annotated_frame, det,
            incident_id=incident.get("incident_id", 0),
            telemetry=telemetry
        )
        if path:
            paths.append(path)
    return paths
