"""
webcam_streamer.py — Drone Feed Simulator using Webcam
=======================================================
Use this when you don't have the drone yet.
It reads from your laptop webcam and streams it to the FastAPI dashboard server
in place of the drone's RTSP/video feed.

Works on macOS and Windows — OpenCV handles webcam access on both.

Usage:
  # Stream local webcam to local server (default)
  python webcam_streamer.py

  # Stream to a different server (e.g. VPS or another machine on LAN)
  python webcam_streamer.py --server http://192.168.1.10:8000

  # Use a specific camera index if you have multiple webcams
  python webcam_streamer.py --camera 1

  # Limit FPS to reduce CPU/network load
  python webcam_streamer.py --fps 10
"""

import argparse
import logging
import sys
import time

import cv2
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Webcam → Dashboard Streamer (drone feed substitute)"
    )
    parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="Base URL of the FastAPI dashboard server (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index to open (default: 0 = built-in webcam)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Target streaming FPS (default: 15)"
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=75,
        help="JPEG compression quality 1-100 (default: 75). Lower = smaller payload."
    )
    return parser.parse_args()


def open_camera(index: int) -> cv2.VideoCapture:
    """Opens the webcam and configures it. Exits on failure."""
    logger.info(f"Opening camera index {index}...")
    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        logger.error(
            f"❌  Could not open camera {index}. "
            "Try --camera 1 if you have multiple cameras, or check permissions."
        )
        sys.exit(1)

    # Request a reasonable resolution — webcam driver will use nearest supported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"✅  Camera opened: {actual_w}x{actual_h}")
    return cap


def encode_frame(frame, quality: int) -> bytes:
    """Encodes a numpy BGR frame to JPEG bytes."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        raise RuntimeError("Failed to encode frame as JPEG")
    return buffer.tobytes()


def post_frame(server: str, stream_type: str, jpeg_bytes: bytes, session: requests.Session):
    """
    POSTs a JPEG frame to the FastAPI endpoint.
    Silently drops frames if the server is temporarily unreachable
    (same offline-resilient pattern as the edge pipeline).
    """
    try:
        session.post(
            f"{server}/api/edge/frame",
            params={"stream_type": stream_type},
            data=jpeg_bytes,
            headers={"Content-Type": "image/jpeg"},
            timeout=0.3   # short timeout — drop frame rather than block pipeline
        )
    except requests.RequestException:
        # Server offline or slow — just skip this frame, keep looping
        pass


def stream(args):
    cap = open_camera(args.camera)
    frame_interval = 1.0 / args.fps

    # Reuse a single HTTP session for keep-alive (much faster than new connections)
    session = requests.Session()

    logger.info(
        f"🎥  Streaming webcam → {args.server} | "
        f"FPS: {args.fps} | JPEG quality: {args.quality}"
    )
    logger.info("Press Ctrl+C to stop.")

    frame_count = 0
    fps_timer = time.time()

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                logger.warning("⚠️  Failed to read frame from camera. Retrying...")
                time.sleep(0.1)
                continue

            # Encode once
            jpeg_bytes = encode_frame(frame, args.quality)

            # --- Raw feed ---
            # Send the clean, unmodified webcam frame as the "raw drone camera" feed.
            post_frame(args.server, "raw", jpeg_bytes, session)

            # --- Overlay / annotated feed ---
            # For now this is the same raw frame.
            # When you add your detection code, process `frame` here first,
            # draw bounding boxes on it, then encode and send separately.
            post_frame(args.server, "overlay", jpeg_bytes, session)

            frame_count += 1

            # Log actual FPS every 5 seconds
            elapsed = time.time() - fps_timer
            if elapsed >= 5.0:
                actual_fps = frame_count / elapsed
                logger.info(f"📡  Streaming — actual FPS: {actual_fps:.1f}")
                frame_count = 0
                fps_timer = time.time()

            # Throttle to target FPS
            processing_time = time.time() - loop_start
            sleep_time = frame_interval - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("🛑  Stream stopped by operator.")
    finally:
        cap.release()
        session.close()
        logger.info("Camera and session released cleanly.")


if __name__ == "__main__":
    args = parse_args()
    stream(args)
