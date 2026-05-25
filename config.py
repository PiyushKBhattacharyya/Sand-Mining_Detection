"""
Environment-driven configuration for Sand Mining Detection System.
Controls whether we run in simulation (dev), edge (Jetson Nano), or cloud (VPS) mode.

Usage:
  On PC (dev):    python main.py all       (DEPLOY_MODE=simulation)
  On Jetson Nano: python main.py edge      (DEPLOY_MODE=jetson)
  On VPS:         python main.py cloud     (DEPLOY_MODE=cloud)

Set via environment variables or a .env file on each machine.
"""
import os
from pathlib import Path

# Load .env file manually if it exists at the project root
project_root = Path(__file__).resolve().parent
_env_path = project_root / ".env"
if _env_path.exists():
    with open(_env_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                # Strip quotes if present
                _v_str = _v.strip().strip("'\"")
                os.environ[_k.strip()] = _v_str

#  Deployment mode 
# "simulation" | "jetson" | "cloud"
DEPLOY_MODE = os.getenv("DEPLOY_MODE", "simulation")

IS_JETSON     = DEPLOY_MODE == "jetson"
IS_CLOUD      = DEPLOY_MODE == "cloud"
IS_SIMULATION = DEPLOY_MODE == "simulation"

#  Cloud / VPS settings 
# Set CLOUD_URL to your Hostinger VPS domain/IP on Jetson
# e.g. "https://sandmining.yourdomain.com" or "http://YOUR_VPS_IP:8000"
CLOUD_URL = os.getenv("CLOUD_URL", "http://localhost:8000")

# VPS server bind settings
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

#  Phone relay settings (Jetson  Android phone running DJI MSDK v5 relay) 
# Phone's LAN IP when both are on the same Wi-Fi hotspot
PHONE_IP         = os.getenv("PHONE_IP",         "192.168.1.50")
PHONE_TELE_PORT  = int(os.getenv("PHONE_TELE_PORT", "9000"))    # UDP telemetry
PHONE_RTSP_URL   = os.getenv("PHONE_RTSP",
                              f"rtsp://{os.getenv('PHONE_IP','192.168.1.50')}:8554/live")

# Fallback: USB camera index if RTSP is unavailable
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")   # "0" = /dev/video0, or rtsp://...

#  Database settings 
# "sqlite" for dev/Jetson offline, "postgresql" for production VPS
DB_TYPE    = os.getenv("DB_TYPE", "sqlite")
PG_CONN_STR = os.getenv("PG_CONN_STR", "")       # e.g. "postgresql://user:pass@localhost/sandmining"

#  Drone / Inference settings 
# Drone flight parameters (used for GPS projection math)
DRONE_ALTITUDE_M   = float(os.getenv("DRONE_ALTITUDE_M", "70.0"))
DRONE_SPEED_KMH    = float(os.getenv("DRONE_SPEED_KMH",  "42.0"))
GIMBAL_PITCH_DEG   = float(os.getenv("GIMBAL_PITCH_DEG", "-80.0"))

# YOLOv8 model weights path
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_WEIGHTS = os.getenv(
    "MODEL_WEIGHTS",
    str(PROJECT_ROOT / "models" / "weights" / "best.pt")
)

#  Telemetry timeouts 
# How long (seconds) to wait for a telemetry packet before using last-known position
TELE_STALE_TIMEOUT_S = float(os.getenv("TELE_STALE_TIMEOUT_S", "5.0"))
