import json
import numpy as np

# ============================================================
# Load rich Fusion export
# ============================================================
JSON_PATH = "c:/Users/charl/Desktop/fusion_kinematics_rich_export.json"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

joints = data["joints"]
occurrences = data["occurrences"]

# ============================================================
# User-defined mechanism structure
# ============================================================
LEG_CHAINS = {
    "leg_A": [
        "fuckass v2:2",
        "tripteron link v1:3",
        "tripteron link v1:4",
        "tripteron ee v6:1",
    ],
    "leg_B": [
        "fuckass mirrored v1:1",
        "tripteron link v1:2",
        "tripteron link v1:1",
        "tripteron ee v6:1",
    ],
    "leg_C": [
        "fuckass2 v3:1",
        "tripteron link v1:5",
        "tripteron link v1:9",
        "tripteron ee v6:1",
        "tripteron link v1:8",
        "tripteron link v1:6",
        "fuckass2 v3:1",
    ],
}

EE_NAME = "tripteron ee v6:1"

# ============================================================
# Helpers
# ============================================================
def arr(x):
    return np.array(x, dtype=float)

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Near-zero vector cannot be normalized.")
    return v / n

def point_key(p, digits=6):
    return tuple(np.round(np.asarray(p, dtype=float), digits))

def fit_plane(points):
    """
    Fit a plane to 3D points using SVD.
    Returns:
        centroid, normal, basis_u, basis_v
    """
    pts = np.asarray(points, dtype=float)
    centroid = pts.mean(axis=0)
    X = pts - centroid
    _, _, vh = np.linalg.svd(X, full_matrices=False)

    # Plane basis = first two singular directions
    u = normalize(vh[0])
    v = normalize(vh[1])
    n = normalize(np.cross(u, v))

    # Re-orthogonalize
    v = normalize(np.cross(n, u))
    return centroid, n, u, v

def project_to_plane_2d(points, origin, u, v):
    pts = np.asarray(points, dtype=float)
    rel = pts - origin
    coords = np.column_stack([rel @ u, rel @ v])
    return coords

def joint_center(j):
    """
    Prefer geometry_or_origin_one origin_world.
    Fall back to geometry_or_origin_two.
    """
    g1 = j.get("geometry_or_origin_one")
    g2 = j.get("geometry_or_origin_two")

    if g1 and g1.get("origin_world") is not None:
        return arr(g1["origin_world"])
    if g2 and g2.get("origin_world") is not None:
        return arr(g2["origin_world"])
    return None

def joint_axis(j):
    motion = j.get("motion", {})
    jt = motion.get("joint_type")
    if jt == "slider" and motion.get("slide_direction_world") is not None:
        return normalize(arr(motion["slide_direction_world"]))
    if jt == "revolute" and motion.get("rotation_axis_world") is not None:
        return normalize(arr(motion["rotation_axis_world"]))
    return None

def connected_to_leg(j, leg_occ_set):
    a = j.get("occurrence_one")
    b = j.get("occurrence_two")
    return (a in leg_occ_set) or (b in leg_occ_set)

def edge_in_leg(j, leg_occ_set):
    a = j.get("occurrence_one")
    b = j.get("occurrence_two")
    return (a in leg_occ_set) and ((b in leg_occ_set) or (b is None))

# ============================================================
# Extract global slider direction / actuator anchors
# ============================================================
slider_joints = [j for j in joints if j.get("motion", {}).get("joint_type") == "slider"]
slider_dirs = [joint_axis(j) for j in slider_joints if joint_axis(j) is not None]
global_x = normalize(np.mean(slider_dirs, axis=0))

slider_info = []
for j in slider_joints:
    p = joint_center(j)
    slider_info.append({
        "name": j["name"],
        "occurrence": j["occurrence_one"],
        "point_world": p,
        "axis_world": joint_axis(j),
        "slide_value": j["motion"].get("slide_value"),
    })

# ============================================================
# Per-leg extraction
# ============================================================
model = {
    "global": {
        "slider_direction_world": global_x.tolist(),
        "sliders": [],
    },
    "legs": {},
    "ee_attachment_points_world": [],
}

for s in slider_info:
    model["global"]["sliders"].append({
        "name": s["name"],
        "occurrence": s["occurrence"],
        "point_world": None if s["point_world"] is None else s["point_world"].tolist(),
        "axis_world": None if s["axis_world"] is None else s["axis_world"].tolist(),
        "slide_value": s["slide_value"],
    })

for leg_name, chain in LEG_CHAINS.items():
    leg_occ_set = set(chain)

    # Joints touching this leg
    leg_joints = [j for j in joints if edge_in_leg(j, leg_occ_set)]

    # Pull unique joint centers
    centers = []
    centers_map = {}
    for j in leg_joints:
        p = joint_center(j)
        if p is None:
            continue
        k = point_key(p)
        centers_map[k] = p

    centers = list(centers_map.values())

    if len(centers) < 3:
        raise RuntimeError(f"{leg_name}: not enough distinct joint centers to fit a plane.")

    plane_origin, plane_normal, plane_u, plane_v = fit_plane(centers)

    # Enforce plane_u to be as aligned with global slider x as possible
    proj_x = global_x - np.dot(global_x, plane_normal) * plane_normal
    if np.linalg.norm(proj_x) > 1e-8:
        plane_u = normalize(proj_x)
        plane_v = normalize(np.cross(plane_normal, plane_u))

    # Store joints with projected 2D coords
    leg_joint_records = []
    for j in leg_joints:
        p = joint_center(j)
        if p is None:
            continue
        uv = project_to_plane_2d([p], plane_origin, plane_u, plane_v)[0]
        leg_joint_records.append({
            "name": j["name"],
            "joint_type": j["motion"]["joint_type"],
            "occurrence_one": j.get("occurrence_one"),
            "occurrence_two": j.get("occurrence_two"),
            "center_world": p.tolist(),
            "center_2d": uv.tolist(),
            "axis_world": None if joint_axis(j) is None else joint_axis(j).tolist(),
        })

    # Estimate link lengths from consecutive joints along the chain
    # We use body-to-body adjacency through shared occurrences.
    # This is just a first pass / consistency check.
    pairwise_dists = []
    for i in range(len(centers)):
        for k in range(i + 1, len(centers)):
            d = np.linalg.norm(centers[i] - centers[k])
            pairwise_dists.append(d)

    pairwise_dists = sorted(pairwise_dists)

    model["legs"][leg_name] = {
        "chain": chain,
        "plane_origin_world": plane_origin.tolist(),
        "plane_normal_world": plane_normal.tolist(),
        "plane_u_world": plane_u.tolist(),
        "plane_v_world": plane_v.tolist(),
        "joint_centers_world": [p.tolist() for p in centers],
        "joints": leg_joint_records,
        "pairwise_center_distances": pairwise_dists,
    }

# ============================================================
# EE attachment extraction
# ============================================================
ee_points = []
for j in joints:
    a = j.get("occurrence_one")
    b = j.get("occurrence_two")
    if a == EE_NAME or b == EE_NAME:
        p = joint_center(j)
        if p is not None:
            ee_points.append(p)

# Unique EE points
ee_unique = {}
for p in ee_points:
    ee_unique[point_key(p)] = p
ee_points = list(ee_unique.values())

model["ee_attachment_points_world"] = [p.tolist() for p in ee_points]

# ============================================================
# Print useful summary
# ============================================================
print("\n=== Global actuator direction ===")
print(global_x)

print("\n=== Slider anchors ===")
for s in model["global"]["sliders"]:
    print(s)

print("\n=== EE attachment points ===")
for p in model["ee_attachment_points_world"]:
    print(p)

for leg_name, leg in model["legs"].items():
    print(f"\n=== {leg_name} ===")
    print("plane origin:", leg["plane_origin_world"])
    print("plane normal:", leg["plane_normal_world"])
    print("num joint centers:", len(leg["joint_centers_world"]))
    print("first few pairwise distances:", leg["pairwise_center_distances"][:10])

# ============================================================
# Optional: save extracted model
# ============================================================
OUT_PATH = "tripteron_extracted_model.json"
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(model, f, indent=2)

print(f"\nSaved extracted model to {OUT_PATH}")