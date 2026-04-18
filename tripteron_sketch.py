import json
import numpy as np
import matplotlib.pyplot as plt


# ============================================
# Load Fusion export JSON
# ============================================
json_path = "c:/Users/charl/fusion_kinematics_export.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

occurrences = data["occurrences"]
joints = data["joints"]


# ============================================
# Helpers
# ============================================
def arr(v):
    return np.array(v, dtype=float)

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Tried to normalize a near-zero vector.")
    return v / n

def translation_from_tf(tf):
    tf = np.array(tf, dtype=float)
    return tf[:3, 3]

def project_perp(v, axis_unit):
    return v - np.dot(v, axis_unit) * axis_unit


# ============================================
# Build lookup tables
# ============================================
occ_by_name = {o["full_path_name"]: o for o in occurrences}


# ============================================
# Extract slider joints and define custom frame
# x = slider direction
# y = direction between two slider origins, projected perp to x
# z = x cross y
# ============================================
slider_joints = [j for j in joints if j.get("joint_type") == "slider"]

if len(slider_joints) < 2:
    raise RuntimeError("Need at least two slider joints to define the custom frame.")

slider_occ_names = []
for j in slider_joints:
    name = j["occurrence_one"]
    if name not in slider_occ_names:
        slider_occ_names.append(name)

slider_origins_world = {}
for name in slider_occ_names:
    occ = occ_by_name[name]
    slider_origins_world[name] = translation_from_tf(occ["transform_world"])

# x-axis from common slider direction
x_candidates = []
for j in slider_joints:
    v = j["motion"].get("slide_direction_world")
    if v is not None:
        x_candidates.append(arr(v))

if not x_candidates:
    raise RuntimeError("No slider direction vectors found.")

x_axis = normalize(np.mean(x_candidates, axis=0))

# y-axis from line between two slider origins, projected perpendicular to x
name0 = slider_occ_names[0]
name1 = slider_occ_names[1]
p0 = slider_origins_world[name0]
p1 = slider_origins_world[name1]

y_seed = p1 - p0
y_proj = project_perp(y_seed, x_axis)

if np.linalg.norm(y_proj) < 1e-9:
    if len(slider_occ_names) >= 3:
        p2 = slider_origins_world[slider_occ_names[2]]
        y_seed = p2 - p0
        y_proj = project_perp(y_seed, x_axis)
    if np.linalg.norm(y_proj) < 1e-9:
        raise RuntimeError("Could not define y-axis from slider origins.")

y_axis = normalize(y_proj)
z_axis = normalize(np.cross(x_axis, y_axis))

# Re-orthogonalize y for numerical cleanliness
y_axis = normalize(np.cross(z_axis, x_axis))

# Use centroid of slider origins as frame origin
origin_world = np.mean(np.vstack(list(slider_origins_world.values())), axis=0)

# Rotation matrix: columns are custom basis vectors expressed in world coords
R = np.column_stack([x_axis, y_axis, z_axis])


def world_to_custom_point(p_world):
    return R.T @ (p_world - origin_world)

def world_to_custom_vec(v_world):
    return R.T @ v_world


# ============================================
# Transform occurrence origins into custom frame
# ============================================
occ_points_custom = {}

for occ in occurrences:
    name = occ["full_path_name"]
    p_world = translation_from_tf(occ["transform_world"])
    occ_points_custom[name] = world_to_custom_point(p_world)


# ============================================
# Build connectivity from joints
# ============================================
edges = []
for j in joints:
    a = j.get("occurrence_one")
    b = j.get("occurrence_two")
    jt = j.get("joint_type")
    jname = j.get("name")

    if a is not None and b is not None and a in occ_points_custom and b in occ_points_custom:
        edges.append((a, b, jt, jname))


# ============================================
# Collect joint axis directions in custom frame
# ============================================
joint_axes_custom = []

for j in joints:
    jt = j.get("joint_type")
    base_occ = j.get("occurrence_one")

    if base_occ not in occ_points_custom:
        continue

    base = occ_points_custom[base_occ]

    if jt == "slider":
        v_world = j["motion"].get("slide_direction_world")
        if v_world is not None:
            v_custom = normalize(world_to_custom_vec(arr(v_world)))
            joint_axes_custom.append((base, v_custom, "slider", j["name"]))

    elif jt == "revolute":
        v_world = j["motion"].get("rotation_axis_world")
        if v_world is not None:
            v_custom = normalize(world_to_custom_vec(arr(v_world)))
            joint_axes_custom.append((base, v_custom, "revolute", j["name"]))


# ============================================
# Plot
# ============================================
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot occurrence points
all_names = list(occ_points_custom.keys())
all_pts = np.vstack([occ_points_custom[n] for n in all_names])

ax.scatter(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2], s=45)

# Labels
for name, p in occ_points_custom.items():
    ax.text(p[0], p[1], p[2], name, fontsize=8)

# Joint connectivity edges
for a, b, jt, jname in edges:
    pa = occ_points_custom[a]
    pb = occ_points_custom[b]
    ax.plot(
        [pa[0], pb[0]],
        [pa[1], pb[1]],
        [pa[2], pb[2]],
        linewidth=1.5
    )

# Highlight slider carriage occurrences
slider_pts = np.vstack([occ_points_custom[n] for n in slider_occ_names])
ax.scatter(slider_pts[:, 0], slider_pts[:, 1], slider_pts[:, 2], s=100, marker="^")

# Draw custom frame axes at custom origin
frame_len = max(5.0, 0.18 * np.max(np.ptp(all_pts, axis=0)))
O = np.zeros(3)

ax.quiver(O[0], O[1], O[2], frame_len, 0, 0, arrow_length_ratio=0.12)
ax.quiver(O[0], O[1], O[2], 0, frame_len, 0, arrow_length_ratio=0.12)
ax.quiver(O[0], O[1], O[2], 0, 0, frame_len, arrow_length_ratio=0.12)

ax.text(frame_len, 0, 0, "x")
ax.text(0, frame_len, 0, "y")
ax.text(0, 0, frame_len, "z")

# Draw joint axes as short line segments
axis_len = 0.12 * max(5.0, np.max(np.ptp(all_pts, axis=0)))

for base, v, jt, name in joint_axes_custom:
    p2 = base + axis_len * v
    ax.plot(
        [base[0], p2[0]],
        [base[1], p2[1]],
        [base[2], p2[2]],
        linewidth=1.0
    )

ax.set_title("Mechanism configuration in custom kinematic frame")
ax.set_xlabel("x (slider direction)")
ax.set_ylabel("y (between-slider direction)")
ax.set_zlabel("z (plane normal)")

# Set roughly equal axes
ranges = np.ptp(all_pts, axis=0)
max_range = np.max(ranges) if np.max(ranges) > 0 else 1.0
mid = np.mean(all_pts, axis=0)

ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)

plt.tight_layout()
plt.show()


# ============================================
# Print frame info
# ============================================
print("Custom frame basis in world coordinates:")
print("x =", x_axis)
print("y =", y_axis)
print("z =", z_axis)
print("origin_world =", origin_world)