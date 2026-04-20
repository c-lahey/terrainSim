import json
import numpy as np
import matplotlib.pyplot as plt


JSON_PATH = "tripteron_extracted_model.json"


def arr(x):
    return np.array(x, dtype=float)


with open(JSON_PATH, "r", encoding="utf-8") as f:
    model = json.load(f)


def get_leg_joint(leg_key, joint_name):
    for j in model["legs"][leg_key]["joints"]:
        if j["name"] == joint_name:
            return arr(j["center_world"])
    raise KeyError((leg_key, joint_name))


def get_slider(slider_name):
    for s in model["global"]["sliders"]:
        if s["name"] == slider_name:
            return arr(s["point_world"])
    raise KeyError(slider_name)


# ------------------------------------------------------------------
# Reference geometry from extracted JSON
# ------------------------------------------------------------------

# Leg A
A_slider = get_slider("Slider 8")
A_base = get_leg_joint("leg_A", "Revolute 9")
A_elbow = get_leg_joint("leg_A", "Revolute 10")
A_ee = get_leg_joint("leg_A", "Revolute 12")

# Leg B
B_slider = get_slider("Slider 21")
B_base = get_leg_joint("leg_B", "Revolute 22")
B_elbow = get_leg_joint("leg_B", "Revolute 2")
B_ee = get_leg_joint("leg_B", "Revolute 24")

# Leg C
C_slider = get_slider("Slider 14")
C_base1 = get_leg_joint("leg_C", "Revolute 29")
C_elbow1 = get_leg_joint("leg_C", "Revolute 16")
C_ee1 = get_leg_joint("leg_C", "Revolute 30")

C_base2 = get_leg_joint("leg_C", "Revolute 31")
C_elbow2 = get_leg_joint("leg_C", "Revolute 19")
C_ee2 = get_leg_joint("leg_C", "Revolute 32")

# EE center for display
ee_pts = np.array(model["ee_attachment_points_world"], dtype=float)
EE_center = ee_pts.mean(axis=0)

# Ground anchors (from the rich export convention you were using)
A_ground = np.array([12.7, 0.0, 0.0])
B_ground = np.array([12.7, 0.0, 12.7])
C_ground = np.array([12.7, 0.0, 12.7])


# ------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------
fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection="3d")


def seg(p, q, **kwargs):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], **kwargs)


def pt(p, **kwargs):
    p = np.asarray(p, dtype=float)
    ax.scatter([p[0]], [p[1]], [p[2]], **kwargs)


# ------------------------------------------------------------------
# Draw fixed ground anchors to moving carriages
# ------------------------------------------------------------------
seg(A_ground, A_slider, color="#7e57c2", linewidth=3)
seg(B_ground, B_slider, color="#7e57c2", linewidth=3)
seg(C_ground, C_slider, color="#7e57c2", linewidth=3)

pt(A_ground, color="#555555", s=55, marker="s")
pt(B_ground, color="#555555", s=55, marker="s")
pt(C_ground, color="#555555", s=55, marker="s")

# ------------------------------------------------------------------
# Leg A
# ------------------------------------------------------------------
seg(A_slider, A_base, color="#2979ff", linewidth=2)
seg(A_base, A_elbow, color="#2979ff", linewidth=2)
seg(A_elbow, A_ee, color="#2979ff", linewidth=2)

# ------------------------------------------------------------------
# Leg B
# ------------------------------------------------------------------
seg(B_slider, B_base, color="#00acc1", linewidth=2)
seg(B_base, B_elbow, color="#00acc1", linewidth=2)
seg(B_elbow, B_ee, color="#00acc1", linewidth=2)

# ------------------------------------------------------------------
# Leg C -- IMPORTANT CLOSED-CHAIN TOPOLOGY
#
# Slider body carries TWO distinct pivots: C_base1 and C_base2.
# EE body carries TWO distinct pivots: C_ee1 and C_ee2.
# The loop is:
#   C_base1 -> C_elbow1 -> C_ee1 -> C_ee2 -> C_elbow2 -> C_base2 -> C_base1
# ------------------------------------------------------------------

# carriage/body side
seg(C_slider, C_base1, color="#ff9100", linewidth=2)
seg(C_slider, C_base2, color="#ff9100", linewidth=2)
seg(C_base1, C_base2, color="#ff9100", linewidth=2, linestyle="--")

# upper branch
seg(C_base1, C_elbow1, color="#ff9100", linewidth=2)
seg(C_elbow1, C_ee1, color="#ff9100", linewidth=2)

# lower branch
seg(C_base2, C_elbow2, color="#ff9100", linewidth=2)
seg(C_elbow2, C_ee2, color="#ff9100", linewidth=2)

# rigid EE-side bracket
seg(C_ee1, C_ee2, color="#00e676", linewidth=2)

# EE polygon / overall platform connections
seg(A_ee, B_ee, color="#00e676", linewidth=1.5, linestyle="--")
seg(B_ee, C_ee1, color="#00e676", linewidth=1.5, linestyle="--")
seg(C_ee2, A_ee, color="#00e676", linewidth=1.5, linestyle="--")

# points
for p in [
    A_slider, A_base, A_elbow, A_ee,
    B_slider, B_base, B_elbow, B_ee,
    C_slider, C_base1, C_elbow1, C_ee1,
    C_base2, C_elbow2, C_ee2,
]:
    pt(p, color="black", s=28)

pt(EE_center, color="red", s=90, marker="x")

# ------------------------------------------------------------------
# Axes / aspect
# ------------------------------------------------------------------
all_pts = np.vstack([
    A_ground, B_ground, C_ground,
    A_slider, A_base, A_elbow, A_ee,
    B_slider, B_base, B_elbow, B_ee,
    C_slider, C_base1, C_elbow1, C_ee1,
    C_base2, C_elbow2, C_ee2,
    EE_center
])

mins = all_pts.min(axis=0)
maxs = all_pts.max(axis=0)
ctr = 0.5 * (mins + maxs)
rng = max((maxs - mins).max(), 1.0) * 0.65

ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
ax.set_zlim(ctr[2] - rng, ctr[2] + rng)

ax.set_title("Tripteron reference pose from CAD-extracted model")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.tight_layout()
plt.show()