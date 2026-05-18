import math
from typing import Tuple

def pixel_to_gps(
    bbox_center_px: Tuple[float, float],
    drone_gps: Tuple[float, float],
    altitude_m: float,
    gimbal_pitch: float,
    gimbal_yaw: float,
    img_size_px: Tuple[int, int] = (3840, 2160),  # 4K by default
    focal_length_mm: float = 24.0,                 # DJI Mavic standard focal length
    sensor_mm: Tuple[float, float] = (13.2, 8.8)   # DJI 1-inch sensor dimensions
) -> Tuple[float, float]:
    """
    Projects a bounding box center pixel to real-world WGS84 GPS Coordinates (latitude, longitude).
    Takes into account the drone GPS, altitude, camera focal length, sensor size, and camera gimbal.
    
    Args:
        bbox_center_px: (x, y) coordinates of the detection center in pixels
        drone_gps: (latitude, longitude) of the drone
        altitude_m: Altitude above ground level (AGL) in meters
        gimbal_pitch: Gimbal pitch angle in degrees (e.g. -90 is nadir, straight down)
        gimbal_yaw: Gimbal yaw/heading in degrees (0-360 clockwise from North)
        img_size_px: (width, height) of the image/frame in pixels
        focal_length_mm: Camera focal length in mm
        sensor_mm: (width, height) of the camera sensor in mm
        
    Returns:
        (latitude, longitude) of the projected detection point
    """
    drone_lat, drone_lon = drone_gps
    px_x, px_y = bbox_center_px
    w_px, h_px = img_size_px
    sensor_w, sensor_h = sensor_mm
    
    # 1. Calculate Ground Sampling Distance (GSD) - meters per pixel
    # At nadir (straight down), GSD is linear
    gsd_x = (sensor_w * altitude_m) / (focal_length_mm * w_px)
    gsd_y = (sensor_h * altitude_m) / (focal_length_mm * h_px)
    
    # Calculate pixel offsets from the center of the image
    offset_x_px = px_x - (w_px / 2.0)
    offset_y_px = (h_px / 2.0) - px_y  # Invert Y so up is positive
    
    # Calculate offset in meters relative to drone (nadir model)
    dx_meters = offset_x_px * gsd_x
    dy_meters = offset_y_px * gsd_y
    
    # 2. Account for gimbal pitch tilt if it is not straight down (-90 degrees)
    # Pitch ranges from 0 (horizontal) to -90 (vertical nadir)
    pitch_rad = math.radians(abs(gimbal_pitch))
    if pitch_rad < math.radians(88.0):  # If camera is tilted up
        # Correct Y distance due to oblique projection angle
        # dy_meters is scaled by the perspective elongation
        sec_pitch = 1.0 / math.sin(pitch_rad)
        dy_meters = dy_meters * sec_pitch
    
    # 3. Rotate offsets (dx, dy) by the gimbal yaw heading to align with True North
    # Gimbal yaw is clockwise from North. In trigonometry, 0 is East counter-clockwise.
    # Convert clockwise heading to standard Cartesian rotation angle (radians)
    rot_rad = math.radians((90.0 - gimbal_yaw) % 360)
    
    # Apply rotation matrix
    dn_meters = dx_meters * math.sin(rot_rad) + dy_meters * math.cos(rot_rad)  # Delta North
    de_meters = dx_meters * math.cos(rot_rad) - dy_meters * math.sin(rot_rad)  # Delta East
    
    # 4. Project meters to WGS84 degrees
    # Earth radius approximations
    meters_per_degree_lat = 111320.0
    meters_per_degree_lon = 111320.0 * math.cos(math.radians(drone_lat))
    
    target_lat = drone_lat + (dn_meters / meters_per_degree_lat)
    target_lon = drone_lon + (de_meters / meters_per_degree_lon)
    
    return round(target_lat, 7), round(target_lon, 7)

if __name__ == "__main__":
    # Test project: Drone at lat: 26.1264, lon: 91.6025, altitude: 60m, looking straight down
    # Object is located exactly at the center of the screen
    center_gps = pixel_to_gps(
        bbox_center_px=(1920, 1080),
        drone_gps=(26.1264, 91.6025),
        altitude_m=60.0,
        gimbal_pitch=-90.0,
        gimbal_yaw=0.0
    )
    print(f"Center Projection (expected: 26.1264, 91.6025): {center_gps}")
    
    # Object is offset to the right (east)
    right_gps = pixel_to_gps(
        bbox_center_px=(2500, 1080),
        drone_gps=(26.1264, 91.6025),
        altitude_m=60.0,
        gimbal_pitch=-90.0,
        gimbal_yaw=0.0
    )
    print(f"Right Offset Projection (expected Lon > 91.6025): {right_gps}")
