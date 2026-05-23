"""
Fills gaps in the Brahmaputra centerline by linear interpolation.
Inserts intermediate points at every ~200m in gaps > 500m.
This eliminates the visual "straight line jumps" in the buffer.
"""
import json
import math
from pathlib import Path

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    a = math.sin((lat2-lat1)*math.pi/360)**2 + \
        math.cos(math.radians(lat1))*math.cos(math.radians(lat2)) * \
        math.sin((lon2-lon1)*math.pi/360)**2
    return 2 * R * math.asin(math.sqrt(a))

def interpolate(p1, p2, step_m=200):
    """Return intermediate points between p1 and p2 spaced ~step_m meters apart."""
    lon1, lat1 = p1
    lon2, lat2 = p2
    total = haversine_m(lat1, lon1, lat2, lon2)
    if total <= step_m:
        return []
    n = int(total / step_m)
    pts = []
    for i in range(1, n):
        t = i / n
        pts.append([lon1 + (lon2-lon1)*t, lat1 + (lat2-lat1)*t])
    return pts

project_root = Path(__file__).resolve().parent.parent.parent
cl_path = project_root / "data" / "legal_zones" / "river_centerline.geojson"

with open(cl_path) as f:
    gj = json.load(f)

coords = gj["features"][0]["geometry"]["coordinates"]
print(f"Input: {len(coords)} points")

# Fill gaps > 400m with interpolated points every ~200m
filled = [coords[0]]
inserted = 0
for i in range(1, len(coords)):
    p1, p2 = coords[i-1], coords[i]
    gap = haversine_m(p1[1], p1[0], p2[1], p2[0])
    if gap > 400:
        interp = interpolate(p1, p2, step_m=200)
        filled.extend(interp)
        inserted += len(interp)
        print(f"  Gap {gap:.0f}m at idx {i}: inserted {len(interp)} interpolated points")
    filled.append(p2)

print(f"\nOutput: {len(filled)} points ({inserted} inserted to fill gaps)")

# Validate
max_gap = max(
    haversine_m(filled[i-1][1], filled[i-1][0], filled[i][1], filled[i][0])
    for i in range(1, len(filled))
)
print(f"Largest remaining gap: {max_gap:.0f}m")

gj["features"][0]["geometry"]["coordinates"] = filled
gj["features"][0]["properties"]["point_count"] = len(filled)
gj["features"][0]["properties"]["gap_filled"] = True

with open(cl_path, "w") as f:
    json.dump(gj, f, indent=2)

print(f"Saved to: {cl_path}")
