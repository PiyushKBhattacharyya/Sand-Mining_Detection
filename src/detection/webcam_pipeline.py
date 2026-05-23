"""
webcam_pipeline.py — Webcam-based feed streamer (drone substitute for testing).
Streams two MJPEG feeds to the FastAPI dashboard server:
  • raw    → CAM 01: Raw Live Feed window
  • overlay → CAM 01: YOLOv8 Annotated Feed window  (same frame for now;
              plug your detection code into process_overlay_frame() later)

Launched via:
  python main.py webcam                         (local webcam, local server)
  python main.py webcam --server http://<ip>:8000  (stream to remote server)
  python main.py webcam --camera 1              (second webcam)
"""
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import requests

logger = logging.getLogger(__name__)


class WebcamPipeline:
    """
    Reads frames from a webcam and streams them to the FastAPI dashboard server
    as JPEG POSTs, feeding both the raw and overlay video windows.

    Works on macOS and Windows — OpenCV's VideoCapture handles both.
    """

    RAW_ENDPOINT     = "/api/edge/frame?stream_type=raw"
    OVERLAY_ENDPOINT = "/api/edge/frame?stream_type=overlay"

    def __init__(
        self,
        cloud_url: str = "http://localhost:8000",
        camera_index: int = 0,
        target_fps: float = 15.0,
        jpeg_quality: int = 75,
    ):
        self.cloud_url     = cloud_url.rstrip("/")
        self.camera_index  = camera_index
        self.target_fps    = target_fps
        self.jpeg_quality  = jpeg_quality
        self.running       = False
        self._session: Optional[requests.Session] = None

    # ------------------------------------------------------------------
    # Override this method when you add your detection code.
    # `frame` is a raw BGR numpy array from the webcam.
    # Return a BGR numpy array with bounding boxes / overlays drawn on it.
    # ------------------------------------------------------------------
    def process_overlay_frame(self, frame):
        """
        Detection hook — currently returns the raw frame unchanged.
        Replace the body of this method with your YOLO inference when ready.
        """
        return frame

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _open_camera(self) -> cv2.VideoCapture:
        logger.info(f"Opening camera index {self.camera_index}...")
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error(
                f"❌  Could not open camera {self.camera_index}. "
                "Check --camera index or grant camera permissions."
            )
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"✅  Camera opened at {w}x{h}")
        return cap

    def _encode(self, frame) -> bytes:
        params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        ok, buf = cv2.imencode(".jpg", frame, params)
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return buf.tobytes()

    def _post(self, endpoint: str, jpeg_bytes: bytes):
        """
        Best-effort POST — silently drops the frame if the server is
        unreachable (same resilient pattern as the edge pipeline).
        """
        try:
            self._session.post(
                self.cloud_url + endpoint,
                data=jpeg_bytes,
                headers={"Content-Type": "image/jpeg"},
                timeout=0.3,
            )
        except requests.RequestException:
            pass   # server offline / slow — skip frame, keep streaming

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self):
        """Starts the webcam streaming loop. Blocks until Ctrl-C."""
        cap = self._open_camera()
        self._session = requests.Session()   # keep-alive across frames
        self.running = True

        frame_interval = 1.0 / self.target_fps
        frame_count    = 0
        fps_timer      = time.time()

        logger.info(
            f"🎥  Webcam pipeline active → {self.cloud_url} | "
            f"FPS target: {self.target_fps} | JPEG quality: {self.jpeg_quality}"
        )
        logger.info("Press Ctrl+C to stop streaming.")

        try:
            while self.running:
                t0 = time.time()

                ret, frame = cap.read()
                if not ret:
                    logger.warning("⚠️  Missed frame from camera — retrying...")
                    time.sleep(0.05)
                    continue

                # --- Raw feed: clean, unmodified webcam frame ---
                raw_jpeg = self._encode(frame)
                self._post(self.RAW_ENDPOINT, raw_jpeg)

                # --- Overlay feed: processed frame (detection hook above) ---
                overlay_frame = self.process_overlay_frame(frame)
                overlay_jpeg  = self._encode(overlay_frame)
                self._post(self.OVERLAY_ENDPOINT, overlay_jpeg)

                frame_count += 1

                # Log actual FPS every 5 seconds
                elapsed = time.time() - fps_timer
                if elapsed >= 5.0:
                    logger.info(f"📡  Streaming — actual FPS: {frame_count / elapsed:.1f}")
                    frame_count = 0
                    fps_timer   = time.time()

                # Throttle to target FPS
                sleep_for = frame_interval - (time.time() - t0)
                if sleep_for > 0:
                    time.sleep(sleep_for)

        except KeyboardInterrupt:
            logger.info("🛑  Webcam pipeline stopped by operator.")
        finally:
            self.running = False
            cap.release()
            self._session.close()
            logger.info("Camera and HTTP session released cleanly.")


if __name__ == "__main__":
    # Allow running directly for quick testing:
    # python src/detection/webcam_pipeline.py
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--server",  default="http://localhost:8000")
    p.add_argument("--camera",  type=int, default=0)
    p.add_argument("--fps",     type=float, default=15.0)
    p.add_argument("--quality", type=int, default=75)
    a = p.parse_args()
    WebcamPipeline(a.server, a.camera, a.fps, a.quality).run()
