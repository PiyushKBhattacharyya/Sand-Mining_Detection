import os
import sys
import argparse
import time
import subprocess
import logging
import warnings
from threading import Thread
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent
venv_python = project_root / ".venv" / "Scripts" / "python.exe"

if not venv_python.exists():
    # Fallback to standard python if venv isn't in expected location
    venv_python = "python"

def rebuild_zones(radius_m: float = 1000.0):
    """Regenerates the river buffer GeoJSON from the latest centerline."""
    sys.path.insert(0, str(project_root / "src" / "preprocess"))
    from zone_builder import build_buffer
    build_buffer(radius_m=radius_m)


def run_server():
    """Runs the FastAPI server."""
    logger.info("🚀 Launching Cloud Backend Dashboard Server (FastAPI)...")
    app_path = project_root / "src" / "dashboard" / "app.py"
    subprocess.run([str(venv_python), str(app_path)])

def run_edge_pipeline():
    """Runs the simulated Jetson Nano edge computing loop on the drone."""
    logger.info("🛸 Powering up Edge Computing Pipeline on DJI Jetson Nano...")
    pipeline_path = project_root / "src" / "detection" / "edge_pipeline.py"
    subprocess.run([str(venv_python), str(pipeline_path)])


def run_webcam_pipeline(server_url: str = "http://localhost:8000",
                        camera: int = 0, fps: float = 15.0, quality: int = 75):
    """Streams webcam feeds to the dashboard (drone substitute for local testing)."""
    logger.info("📷  Starting Webcam Pipeline (drone substitute)...")
    sys.path.insert(0, str(project_root / "src" / "detection"))
    from webcam_pipeline import WebcamPipeline
    WebcamPipeline(
        cloud_url=server_url,
        camera_index=camera,
        target_fps=fps,
        jpeg_quality=quality,
    ).run()

def main():
    parser = argparse.ArgumentParser(
        description="Brahmaputra Illegal Sand Mining Drone Surveillance Command Center Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "mode",
        choices=["server", "edge", "webcam", "all"],
        default="all",
        nargs="?",
        help=(
            "Execution Mode: "
            "'server' = web dashboard only | "
            "'edge' = simulated drone pipeline | "
            "'webcam' = server + webcam feed (no drone needed) | "
            "'all' = server + simulated edge pipeline"
        )
    )
    # Webcam-mode options (only used when mode == 'webcam')
    parser.add_argument("--server-url", default="http://localhost:8000",
                        help="Dashboard server URL for webcam mode (default: http://localhost:8000)")
    parser.add_argument("--camera",     type=int,   default=0,
                        help="Webcam index to use (default: 0)")
    parser.add_argument("--fps",        type=float, default=15.0,
                        help="Target streaming FPS for webcam mode (default: 15)")
    parser.add_argument("--quality",    type=int,   default=75,
                        help="JPEG quality 1-100 for webcam mode (default: 75)")
    
    args = parser.parse_args()

    if args.mode == "server":
        run_server()

    elif args.mode == "edge":
        run_edge_pipeline()

    elif args.mode == "webcam":
        logger.info("🔥 BOOTING WEBCAM SURVEILLANCE MODE...")

        # 0. Rebuild GIS zones from latest centerline
        logger.info("Rebuilding river buffer zone from latest centerline...")
        rebuild_zones()

        # 1. Start server in a background thread
        server_thread = Thread(target=run_server, daemon=True)
        server_thread.start()

        # Give the server 3 seconds to spin up
        time.sleep(3)

        # 2. Start webcam pipeline in the foreground
        try:
            run_webcam_pipeline(
                server_url=args.server_url,
                camera=args.camera,
                fps=args.fps,
                quality=args.quality,
            )
        except KeyboardInterrupt:
            logger.info("Webcam surveillance shutdown by operator.")

    elif args.mode == "all":
        logger.info("🔥 BOOTING SURVEILLANCE ECOSYSTEM END-TO-END...")

        # 0. Rebuild GIS zones from latest centerline
        logger.info("Rebuilding river buffer zone from latest centerline...")
        rebuild_zones()

        # 1. Start Server thread
        server_thread = Thread(target=run_server, daemon=True)
        server_thread.start()

        # Give the server 3 seconds to spin up and bind port 8000
        time.sleep(3)

        # 2. Start Simulated Drone Edge Pipeline
        logger.info("🛰️ Initializing drone takeoff sequence...")
        try:
            run_edge_pipeline()
        except KeyboardInterrupt:
            logger.info("Ecosystem shutdown by operator.")
            
if __name__ == "__main__":
    main()
