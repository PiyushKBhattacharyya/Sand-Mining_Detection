import os
import json
import time
import math
import random
from datetime import datetime
from pathlib import Path
import logging
from db_setup import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DroneSimulator:
    """
    Simulates a DJI drone flight path along the river centerline.
    Generates telemetry packets and logs them to the local PostgreSQL/SQLite database.
    """
    def __init__(self, db_manager, centerline_path=None, speed_kmh=40.0, altitude_m=60.0):
        self.db_manager = db_manager
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.centerline_path = centerline_path or (
            self.project_root / "data" / "legal_zones" / "river_centerline.geojson"
        )
        
        self.speed_mps = speed_kmh / 3.6  # Convert km/h to m/s
        self.altitude_m = altitude_m
        self.battery = 100.0
        self.flight_points = []
        
        # Load and parse path
        self._load_flight_path()

    def _load_flight_path(self):
        """Loads and interpolates the centerline geojson into high-resolution coordinate steps."""
        if not Path(self.centerline_path).exists():
            logger.error(f"Centerline file not found: {self.centerline_path}")
            return
        
        with open(self.centerline_path, 'r') as f:
            data = json.load(f)
            
        # Extract original coordinates (lon, lat)
        raw_coords = data['features'][0]['geometry']['coordinates']
        logger.info(f"Loaded {len(raw_coords)} primary waypoints from centerline.")
        
        # Interpolate between waypoints to create high-resolution coordinates
        self.flight_points = []
        for i in range(len(raw_coords) - 1):
            lon1, lat1 = raw_coords[i]
            lon2, lat2 = raw_coords[i+1]
            
            # Estimate distance in meters (approx: 1 deg lat = 111320m, 1 deg lon = 111320 * cos(lat) )
            lat_mid = (lat1 + lat2) / 2.0
            dy = (lat2 - lat1) * 111320
            dx = (lon2 - lon1) * 111320 * math.cos(math.radians(lat_mid))
            distance = math.sqrt(dx**2 + dy**2)
            
            # Number of steps (at 5 Hz simulation frequency)
            steps = max(10, int(distance / (self.speed_mps / 5.0)))
            
            for step in range(steps):
                t = step / steps
                # Linear interpolation along centerline
                interp_lon = lon1 + (lon2 - lon1) * t
                interp_lat = lat1 + (lat2 - lat1) * t
                
                # Add a lateral sinusoidal weave to simulate panning/sweeping the riverbed
                # Toggles between left and right of the river up to 120 meters
                weave_phase = (i * steps + step) * 0.05
                lateral_offset_meters = math.sin(weave_phase) * 120.0
                
                # Calculate heading angle
                heading = math.atan2(dy, dx)
                # Perpendicular angle to heading (for lateral offset)
                perp_angle = heading + math.pi / 2.0
                
                # Convert lateral offset back to degrees
                offset_lat = (lateral_offset_meters * math.sin(perp_angle)) / 111320
                offset_lon = (lateral_offset_meters * math.cos(perp_angle)) / (111320 * math.cos(math.radians(interp_lat)))
                
                final_lat = interp_lat + offset_lat
                final_lon = interp_lon + offset_lon
                
                # Calculate heading in degrees (0-360 clockwise from North)
                heading_deg = (90.0 - math.degrees(heading)) % 360.0
                
                self.flight_points.append({
                    'lat': final_lat,
                    'lon': final_lon,
                    'heading': heading_deg
                })
                
        logger.info(f"Generated {len(self.flight_points)} high-resolution flight points.")

    def run_simulation(self, duration_seconds=120, frequency_hz=5):
        """Runs the telemetry loop, logging database coordinates at the specified frequency."""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        logger.info(f"Starting Drone Telemetry Simulator at {frequency_hz} Hz...")
        
        sleep_interval = 1.0 / frequency_hz
        total_steps = int(duration_seconds * frequency_hz)
        
        is_pg = self.db_manager.db_type == "postgresql"
        insert_query = """
        INSERT INTO telemetry_logs (
            timestamp, latitude, longitude, altitude_agl, 
            gimbal_pitch, gimbal_yaw, gimbal_roll, 
            drone_speed, battery_percentage, gps_accuracy_m
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """ if is_pg else """
        INSERT INTO telemetry_logs (
            timestamp, latitude, longitude, altitude_agl, 
            gimbal_pitch, gimbal_yaw, gimbal_roll, 
            drone_speed, battery_percentage, gps_accuracy_m
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        try:
            for step in range(total_steps):
                point_idx = step % len(self.flight_points)
                point = self.flight_points[point_idx]
                
                # Dynamic simulated variables
                timestamp = datetime.now().isoformat()
                latitude = point['lat']
                longitude = point['lon']
                
                # Small random variations on altitude and speed
                altitude = self.altitude_m + random.uniform(-2.0, 2.0)
                speed = self.speed_mps + random.uniform(-0.5, 0.5)
                
                # Gimbal pitch between -45 (looking ahead) and -90 (looking straight down)
                gimbal_pitch = -60.0 + math.sin(step * 0.1) * 20.0
                # Gimbal yaw matches flight heading with slight adjustments
                gimbal_yaw = (point['heading'] + random.uniform(-3.0, 3.0)) % 360.0
                gimbal_roll = random.uniform(-1.0, 1.0)
                
                # Drain battery slowly (approx 0.05% per step at 5 Hz => 15 mins total flight time)
                self.battery = max(0.0, self.battery - 0.02)
                gps_accuracy = random.uniform(0.05, 1.2) # High accuracy
                
                # Log entry parameters
                params = (
                    timestamp, latitude, longitude, altitude,
                    gimbal_pitch, gimbal_yaw, gimbal_roll,
                    speed, int(self.battery), gps_accuracy
                )
                
                cursor.execute(insert_query, params)
                conn.commit()
                
                if step % 25 == 0:
                    logger.info(
                        f"Telemetry Logged [Step {step}/{total_steps}]: "
                        f"Lat: {latitude:.6f}, Lon: {longitude:.6f}, "
                        f"Alt: {altitude:.1f}m, Batt: {int(self.battery)}%"
                    )
                
                time.sleep(sleep_interval)
                
        except KeyboardInterrupt:
            logger.info("Simulation stopped by user.")
        finally:
            cursor.close()
            conn.close()
            logger.info("Database connection closed. Telemetry Simulator Finished.")

if __name__ == "__main__":
    db = DatabaseManager(db_type="sqlite")
    # Make sure tables exist
    db.initialize_database()
    
    sim = DroneSimulator(db_manager=db, speed_kmh=45.0, altitude_m=65.0)
    # Run for 20 seconds (100 steps) for testing
    sim.run_simulation(duration_seconds=20, frequency_hz=5)
