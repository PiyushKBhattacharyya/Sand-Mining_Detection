"""
Fix the centerline by discarding the broken nearest-neighbor re-chain
and restoring a clean longitude-sorted order.

The nearest-neighbor algorithm created wrong cross-connections, producing
diagonal lines across the river. For the Brahmaputra (which flows primarily
W->E through Guwahati), longitude-sorting gives the correct sequential order.
We then re-fill gaps at 200m intervals to smooth out any sparse sections.
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

def interpolate_gap(p1, p2, step_m=200):
    """Insert intermediate points every step_m meters between p1 and p2."""
    lon1, lat1 = p1
    lon2, lat2 = p2
    total = haversine_m(lat1, lon1, lat2, lon2)
    if total <= step_m:
        return []
    n = max(1, int(total / step_m))
    return [[lon1 + (lon2-lon1)*i/n, lat1 + (lat2-lat1)*i/n] for i in range(1, n)]

project_root = Path(__file__).resolve().parent.parent.parent
cl_path = project_root / "data" / "legal_zones" / "river_centerline.geojson"

with open(cl_path) as f:
    gj = json.load(f)

coords = gj["features"][0]["geometry"]["coordinates"]
print(f"Input: {len(coords)} points")

# Step 1: Sort strictly W->E by longitude
#   For a river flowing primarily west-to-east (like Brahmaputra through Guwahati)
#   this correctly reconstructs the sequential flow order and eliminates
#   the diagonal cross-connections the nearest-neighbor algorithm created.
coords_sorted = sorted(coords, key=lambda p: p[0])
print(f"After lon-sort: {len(coords_sorted)} points")

# Validate: check how many "backward" jumps exist
backward = sum(1 for i in range(1, len(coords_sorted))
               if coords_sorted[i][0] < coords_sorted[i-1][0])
print(f"Backward longitude jumps after sort: {backward}  (should be 0)")

# Step 2: Find and report gaps
gaps_before = [(haversine_m(coords_sorted[i-1][1], coords_sorted[i-1][0],
                             coords_sorted[i][1],   coords_sorted[i][0]), i)
               for i in range(1, len(coords_sorted))]
gaps_before.sort(reverse=True)
print("\nTop 5 gaps before fill:")
for d, i in gaps_before[:5]:
    print(f"  {d:.0f}m between idx {i-1} [{coords_sorted[i-1][0]:.5f}] "
          f"and idx {i} [{coords_sorted[i][0]:.5f}]")

# Step 3: Fill all gaps > 300m at 200m intervals
filled = [coords_sorted[0]]
inserted = 0
for i in range(1, len(coords_sorted)):
    p1, p2 = coords_sorted[i-1], coords_sorted[i]
    gap = haversine_m(p1[1], p1[0], p2[1], p2[0])
    if gap > 300:
        interp = interpolate_gap(p1, p2, step_m=200)
        filled.extend(interp)
        inserted += len(interp)
    filled.append(p2)

print(f"\nOutput: {len(filled)} points ({inserted} inserted)")
max_gap = max(haversine_m(filled[i-1][1], filled[i-1][0],
                           filled[i][1],   filled[i][0])
              for i in range(1, len(filled)))
print(f"Largest remaining gap: {max_gap:.0f}m")

# Step 4: Save
gj["features"][0]["geometry"]["coordinates"] = filled
gj["features"][0]["properties"]["point_count"] = len(filled)
gj["features"][0]["properties"]["fix"] = "lon_sorted_gap_filled"

with open(cl_path, "w") as f:
    json.dump(gj, f, indent=2)

print(f"\nSaved to: {cl_path}")
