"""
Re-chains the existing 68 OSM points into proper river flow order
using nearest-neighbor path finding from the westernmost start point.
No network needed  works entirely on the cached geojson.
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

project_root = Path(__file__).resolve().parent.parent.parent
cl_path = project_root / "data" / "legal_zones" / "river_centerline.geojson"

with open(cl_path) as f:
    gj = json.load(f)

coords = gj["features"][0]["geometry"]["coordinates"]
print(f"Loaded {len(coords)} existing points")

# The existing points were lon-sorted, which is roughly correct for a W->E river
# but loses N-S meander detail where the river curves back
# Re-chain with nearest-neighbor starting from the westernmost point
# This reconstructs the actual river path order

unvisited = list(coords)

# Start from westernmost point
start = min(unvisited, key=lambda p: p[0])
unvisited.remove(start)
chain = [start]

while unvisited:
    last = chain[-1]
    # Find nearest unvisited point
    nearest = min(unvisited, key=lambda p: haversine_m(last[1], last[0], p[1], p[0]))
    # Safety: if nearest is more than 8km away, something is wrong  stop
    dist = haversine_m(last[1], last[0], nearest[1], nearest[0])
    if dist > 8000:
        print(f"  Gap of {dist:.0f}m detected at {last} -> {nearest}, stopping chain")
        break
    unvisited.remove(nearest)
    chain.append(nearest)

print(f"Re-chained to {len(chain)} ordered points")

# Validate the chain  check for any large jumps
max_gap = 0
for i in range(1, len(chain)):
    d = haversine_m(chain[i-1][1], chain[i-1][0], chain[i][1], chain[i][0])
    if d > max_gap:
        max_gap = d

print(f"Largest gap between consecutive points: {max_gap:.0f}m")
print(f"\nFirst 5 points: {chain[:5]}")
print(f"Last  5 points: {chain[-5:]}")

# Save back
gj["features"][0]["geometry"]["coordinates"] = chain
gj["features"][0]["properties"]["ordered"] = True
gj["features"][0]["properties"]["point_count"] = len(chain)

with open(cl_path, "w") as f:
    json.dump(gj, f, indent=2)

print(f"\nSaved re-chained centerline to: {cl_path}")
