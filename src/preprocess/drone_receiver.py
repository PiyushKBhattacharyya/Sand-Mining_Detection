"""
Drone Receiver — Jetson Nano Component
Receives live telemetry from the Android phone relay app (DJI MSDK v5)
over UDP, and provides a thread-safe accessor for the edge pipeline.

Phone relay app sends JSON packets every 200ms to UDP port 9000:
{
  "lat": 26.12345, "lon": 91.67890, "altitude": 70.5,
  "speed": 11.2, "heading": 182.4, "battery": 87,
  "gimbal_pitch": -80.0, "gimbal_yaw": 0.0,
  "timestamp": "2026-05-18T09:00:00.123"
}

Also provides an OpenCV VideoCapture wrapper for the RTSP video stream
from the phone relay app, with automatic reconnection on drop.
"""
import socket
import json
import time
import logging
import threading
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class TelemetryReceiver:
    """
    UDP server that listens for telemetry JSON packets from the phone relay.
    Stores the latest packet and provides thread-safe access.
    Runs in a background daemon thread.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9000,
                 stale_timeout_s: float = 5.0):
        self.host    = host
        self.port    = port
        self.stale_timeout = stale_timeout_s

        self._lock           = threading.Lock()
        self._latest: Dict[str, Any] = {}
        self._last_received  = 0.0
        self._thread: Optional[threading.Thread] = None
        self._running        = False

    def start(self):
        """Start the UDP listener as a daemon thread."""
        self._running = True
        self._thread  = threading.Thread(target=self._listen, daemon=True,
                                          name="TelemetryReceiver")
        self._thread.start()
        logger.info(f"Telemetry receiver listening on UDP {self.host}:{self.port}")

    def stop(self):
        self._running = False

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """
        Returns the latest telemetry dict, or None if no packet received yet
        or the last packet is older than stale_timeout_s.
        """
        with self._lock:
            if not self._latest:
                return None
            age = time.time() - self._last_received
            if age > self.stale_timeout:
                logger.warning(f"Telemetry stale ({age:.1f}s). Check phone relay.")
                return None
            return dict(self._latest)

    def is_online(self) -> bool:
        return (time.time() - self._last_received) < self.stale_timeout

    def _listen(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(2.0)  # Allow clean shutdown
        sock.bind((self.host, self.port))

        logger.info("UDP telemetry socket bound.")

        while self._running:
            try:
                data, addr = sock.recvfrom(4096)
                packet = json.loads(data.decode("utf-8"))

                # Validate required fields
                required = {"lat", "lon", "altitude", "speed", "heading", "battery"}
                if not required.issubset(packet.keys()):
                    logger.warning(f"Incomplete telemetry packet from {addr}: {packet}")
                    continue

                # Add defaults for optional fields
                packet.setdefault("gimbal_pitch", -80.0)
                packet.setdefault("gimbal_yaw",   0.0)
                packet.setdefault("timestamp",    datetime.now().isoformat())

                with self._lock:
                    self._latest       = packet
                    self._last_received = time.time()

            except socket.timeout:
                continue
            except json.JSONDecodeError as e:
                logger.warning(f"Bad telemetry JSON: {e}")
            except Exception as e:
                logger.error(f"Telemetry receiver error: {e}")

        sock.close()
        logger.info("Telemetry receiver stopped.")


class DroneVideoCapture:
    """
    Wraps OpenCV VideoCapture to read from the RTSP stream of the phone relay
    (or a USB camera as fallback). Automatically reconnects on stream drop.
    """

    def __init__(self, source: str, frame_w: int = 1920, frame_h: int = 1080,
                 reconnect_delay_s: float = 3.0):
        self.source          = source
        self.frame_w         = frame_w
        self.frame_h         = frame_h
        self.reconnect_delay = reconnect_delay_s
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._connect()

    def _connect(self):
        """Open the capture source. Supports RTSP URL or integer camera index."""
        src = int(self.source) if str(self.source).isdigit() else self.source
        logger.info(f"Opening video source: {src}")
        cap = cv2.VideoCapture(src)

        if isinstance(src, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.frame_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)
            cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            # RTSP — reduce latency buffer
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        with self._lock:
            self._cap = cap

        if cap.isOpened():
            logger.info(f"Video source opened: {src}")
        else:
            logger.warning(f"Could not open video source: {src}. Will retry.")

    def read(self) -> np.ndarray:
        """
        Read one frame. Returns the frame if successful.
        On failure, returns a dark fallback frame and schedules reconnect.
        """
        with self._lock:
            cap = self._cap

        if cap and cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                return frame
            else:
                logger.warning("Frame read failed — stream dropped. Reconnecting...")

        # Fallback frame while reconnecting
        fallback = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        fallback[:, :] = [10, 15, 25]
        cv2.putText(fallback, "RECONNECTING TO DRONE CAMERA...",
                    (400, 540), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 100, 200), 2, cv2.LINE_AA)

        # Reconnect in background
        t = threading.Thread(target=self._reconnect, daemon=True)
        t.start()
        return fallback

    def _reconnect(self):
        time.sleep(self.reconnect_delay)
        try:
            with self._lock:
                if self._cap:
                    self._cap.release()
            self._connect()
        except Exception as e:
            logger.error(f"Reconnect failed: {e}")

    def release(self):
        with self._lock:
            if self._cap:
                self._cap.release()
                self._cap = None
