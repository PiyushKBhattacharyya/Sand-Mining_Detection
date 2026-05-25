#  Jetson Nano + Real Drone Deployment Guide

## Architecture Split

```
      Wi-Fi / 4G LTE
           JETSON NANO  (on drone)              Cloud Server (your PC / VPS)
                                                                      
                                 FastAPI + Dashboard     
   Real Camera   edge_pipeline.py       POST /api/edge/sync      app.py  (port 8000)    
   (USB/MIPI)       (YOLOv8 inference)                           
        POST /api/edge/frame     WebSocket  Browser    
                                                                     
                             
   DJI GPS/                                If network DOWN:
   Telemetry    SQLite DB (local SSD)     All data saved here
   (UART/USB)     sync_worker retries     Uploaded when network returns
                             

```

---

## Phase A: Jetson Nano OS & Environment Setup

### 1. Flash JetPack (do once)
- Flash **JetPack 5.1.2** (Ubuntu 20.04 + CUDA 11.4) onto a 64GB microSD
- Boot, run `sudo apt update && sudo apt upgrade -y`

### 2. Install system dependencies
```bash
sudo apt install -y python3-pip python3-venv git libopencv-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    sqlite3 libsqlite3-dev pkg-config curl
```

### 3. Clone project onto Jetson
```bash
git clone https://github.com/YOUR_USER/Sand-Mining_Detection.git ~/sand_mining
cd ~/sand_mining
```

> [!IMPORTANT]
> Do **NOT** run `pip install` directly. Jetson uses ARM64  some wheels need special handling.

### 4. Create venv + install Jetson-compatible packages
```bash
python3 -m venv .venv
source .venv/bin/activate

# OpenCV: use system-linked version (already has CUDA support)
pip install --no-deps opencv-python  # skip, use system cv2

# PyTorch for Jetson (JetPack 5.x  torch 2.1)
# Download from: https://developer.nvidia.com/embedded/pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu114

# YOLOv8 (ultralytics)
pip install ultralytics

# Everything else
pip install fastapi uvicorn requests websockets fpdf2 Pillow \
    scikit-learn numpy geopandas shapely pyproj psycopg2-binary
```

---

## Phase B: Connecting the Drone

### Option 1: DJI Enterprise (M300 RTK / M600)  DJI OSDK via UART

The Jetson connects to DJI's flight controller via **UART serial** at 921600 baud.

**Wiring:**
```
Jetson GPIO Header         DJI A3/N3/M300 OSDK Port
          
Pin 8  (UART_TX)      RX
Pin 10 (UART_RX)      TX
Pin 6  (GND)          GND
3.3V logic  use level shifter if DJI port is 5V TTL
```

**Install DJI OSDK:**
```bash
git clone https://github.com/dji-sdk/Onboard-SDK.git
cd Onboard-SDK && mkdir build && cd build
cmake .. && make -j4 && sudo make install
pip install djiosdk-python  # Python bindings
```

**Replace drone_simulator.py with real telemetry reader:**
```python
# src/preprocess/drone_telemetry.py  (NEW  replaces simulator)
import djiosdk
vehicle = djiosdk.Vehicle(port="/dev/ttyTHS1", baud=921600)

def get_live_telemetry():
    gps   = vehicle.broadcast.getLatestStatusGPS()
    imu   = vehicle.broadcast.getLatestStatusIMU()
    return {
        "lat":      gps.latitude,
        "lon":      gps.longitude,
        "altitude": vehicle.broadcast.getLatestStatusVelocity().z,
        "speed":    vehicle.broadcast.getLatestStatusVelocity().x,
        "heading":  imu.q.toEulerAngle()[2],
        "battery":  vehicle.broadcast.getLatestStatusBattery().percentage
    }
```

---

### Option 2: DJI Consumer (Mavic 3E / Air 2S)  DJI Mobile SDK relay

Consumer drones don't have OSDK. Use a **Mobile SDK relay** via phone  Jetson:

```
DJI RC  Android Phone (DJI SDK App) Wi-Fi Jetson  Cloud
```

- Write a small Android app using DJI Mobile SDK that broadcasts telemetry over UDP
- Or use **DJI Assistant 2** + a custom relay script

---

### Option 3: Any Drone  MAVLink (ArduPilot / PX4)
If your drone runs ArduPilot or PX4:
```bash
pip install dronekit

# src/preprocess/drone_telemetry.py
from dronekit import connect

vehicle = connect('/dev/ttyUSB0', baud=57600, wait_ready=True)

def get_live_telemetry():
    return {
        "lat":      vehicle.location.global_frame.lat,
        "lon":      vehicle.location.global_frame.lon,
        "altitude": vehicle.location.global_relative_frame.alt,
        "speed":    vehicle.groundspeed,
        "heading":  vehicle.heading,
        "battery":  vehicle.battery.level
    }
```

---

## Phase C: Real Camera Feed

### USB Camera (easiest)
```python
# In edge_pipeline.py  replace simulated canvas with:
import cv2
cap = cv2.VideoCapture(0)  # /dev/video0 = USB camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

ret, frame = cap.read()
if not ret:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)  # fallback
```

### DJI Payload Camera (HDMI capture card)
```python
cap = cv2.VideoCapture("hdmi_capture_device_index")
# Or via GStreamer for MIPI CSI:
cap = cv2.VideoCapture(
    "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1920,height=1080 ! "
    "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! appsink",
    cv2.CAP_GSTREAMER
)
```

---

## Phase D: Code Changes for Jetson

### 1. Create `config.py` for environment-aware settings
```python
# config.py (new file at project root)
import os

# Set DEPLOY_MODE=jetson on Jetson, leave empty on PC
DEPLOY_MODE = os.getenv("DEPLOY_MODE", "simulation")

# Cloud server address (your PC's LAN IP or public VPS IP)
CLOUD_URL = os.getenv("CLOUD_URL", "http://192.168.1.100:8000")

# Drone connection
DRONE_PORT    = os.getenv("DRONE_PORT", "/dev/ttyTHS1")
DRONE_BAUD    = int(os.getenv("DRONE_BAUD", "921600"))
CAMERA_SOURCE = int(os.getenv("CAMERA_SOURCE", "0"))

# DB: use PostgreSQL on Jetson for prod, SQLite for dev
DB_TYPE       = os.getenv("DB_TYPE", "sqlite")
PG_CONN_STR   = os.getenv("PG_CONN_STR", "")
```

### 2. Modify `main.py` to run only the edge pipeline on Jetson
```python
# On Jetson, you DON'T run the dashboard (that stays on cloud PC)
# Run ONLY:
#   python main.py edge
```

**Split boot modes in `main.py`:**
```python
mode = sys.argv[1] if len(sys.argv) > 1 else "all"

if mode == "edge":
    # Jetson-only: run detection + sync, no dashboard
    run_edge_pipeline_only()
elif mode == "cloud":
    # PC/VPS-only: run FastAPI dashboard
    run_cloud_server_only()
elif mode == "all":
    # Dev simulation: run everything locally
    run_all_simulation()
```

---

## Phase E: Network Configuration

### On your Cloud PC / VPS
```bash
# Open port 8000 for Jetson to reach dashboard
# Windows: allow in Windows Firewall
netsh advfirewall firewall add rule name="SandMining" dir=in action=allow protocol=TCP localport=8000

# Find your PC's IP (Jetson will use this)
ipconfig  # note the LAN IP, e.g. 192.168.1.100
```

### On Jetson (set cloud URL)
```bash
export CLOUD_URL="http://192.168.1.100:8000"
export DEPLOY_MODE="jetson"
export CAMERA_SOURCE=0
```

### 4G LTE Fallback (for remote field ops)
```bash
# Install ModemManager for USB 4G dongle
sudo apt install modemmanager
# Jetson will route through LTE when field Wi-Fi is unavailable
# sync_worker handles reconnection automatically
```

---

## Phase F: Auto-Start on Boot (systemd)

Create a service so the edge pipeline starts automatically when Jetson powers on:

```bash
sudo nano /etc/systemd/system/sand-mining-edge.service
```

```ini
[Unit]
Description=Sand Mining Detection Edge Pipeline
After=network.target

[Service]
Type=simple
User=nvidia
WorkingDirectory=/home/nvidia/sand_mining
Environment="DEPLOY_MODE=jetson"
Environment="CLOUD_URL=http://192.168.1.100:8000"
Environment="CAMERA_SOURCE=0"
ExecStart=/home/nvidia/sand_mining/.venv/bin/python main.py edge
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable sand-mining-edge
sudo systemctl start sand-mining-edge
sudo systemctl status sand-mining-edge
```

---

## Phase G: What to implement in code next

| Task | File to modify | Priority |
|---|---|---|
| Real telemetry reader (OSDK / MAVLink / UDP) | `src/preprocess/drone_telemetry.py` (new) |  HIGH |
| Real camera capture replacing simulator canvas | `src/detection/edge_pipeline.py` |  HIGH |
| Config-driven boot (edge/cloud/sim modes) | `main.py` + `config.py` |  HIGH |
| Real YOLOv8 weights (trained on mining data) | `models/weights/best.pt` |  MEDIUM |
| PostgreSQL setup on Jetson | `db_setup.py` + env vars |  LOW |
| HTTPS/auth for cloud endpoint | `app.py` |  LOW |

---

##  Tell me your drone model

The implementation will differ significantly based on your drone:

| Model | Connection Method | SDK |
|---|---|---|
| **DJI M300 RTK** | UART (OSDK port) | DJI OSDK 4.x |
| **DJI M600 Pro** | UART (A3 controller) | DJI OSDK 3.x |
| **DJI Mavic 3E / Air** | Mobile SDK relay or Wi-Fi | DJI MSDK |
| **Custom / ArduPilot** | UART/USB serial | DroneKit / MAVLink |
| **DJI FPV / Mini** | USB/Wi-Fi + DJI Assistant | Manual video capture |
