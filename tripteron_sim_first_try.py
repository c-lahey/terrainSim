import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.optimize import least_squares


# ============================================================
# Load extracted model
# ============================================================
JSON_PATH = "tripteron_extracted_model.json"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    model = json.load(f)


# ============================================================
# Helpers
# ============================================================
def arr(x):
    return np.array(x, dtype=float)

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Near-zero vector")
    return v / n

def norm(v):
    return np.linalg.norm(v)

def to2d(p_world, leg):
    origin = arr(leg["plane_origin_world"])
    u = arr(leg["plane_u_world"])
    v = arr(leg["plane_v_world"])
    r = arr(p_world) - origin
    return np.array([np.dot(r, u), np.dot(r, v)])

def to3d(p_2d, leg):
    origin = arr(leg["plane_origin_world"])
    u = arr(leg["plane_u_world"])
    v = arr(leg["plane_v_world"])
    return origin + p_2d[0] * u + p_2d[1] * v

def circle_intersections(B, E, L1, L2):
    dvec = E - B
    d = norm(dvec)
    if d < 1e-12:
        return []
    if d > L1 + L2 + 1e-9:
        return []
    if d < abs(L1 - L2) - 1e-9:
        return []

    ex = dvec / d
    a = (L1**2 - L2**2 + d**2) / (2*d)
    h2 = max(L1**2 - a**2, 0.0)
    h = np.sqrt(h2)
    Pm = B + a * ex
    ey = np.array([-ex[1], ex[0]])
    return [Pm + h * ey, Pm - h * ey]

def choose_closest(cands, ref):
    if not cands:
        return None
    d = [norm(c - ref) for c in cands]
    return cands[int(np.argmin(d))]


# ============================================================
# Global geometry from extracted model
# ============================================================
global_x = arr(model["global"]["slider_direction_world"])

legA = model["legs"]["leg_A"]
legB = model["legs"]["leg_B"]
legC = model["legs"]["leg_C"]

ee_pts_ref = [arr(p) for p in model["ee_attachment_points_world"]]
ee_center_ref = np.mean(np.vstack(ee_pts_ref), axis=0)

# EE offsets from CAD reference
A_ee_ref = arr(next(j for j in legA["joints"] if j["name"] == "Revolute 12")["center_world"])
B_ee_ref = arr(next(j for j in legB["joints"] if j["name"] == "Revolute 24")["center_world"])
C_ee1_ref = arr(next(j for j in legC["joints"] if j["name"] == "Revolute 30")["center_world"])
C_ee2_ref = arr(next(j for j in legC["joints"] if j["name"] == "Revolute 32")["center_world"])

A_ee_offset = A_ee_ref - ee_center_ref
B_ee_offset = B_ee_ref - ee_center_ref
C_ee1_offset = C_ee1_ref - ee_center_ref
C_ee2_offset = C_ee2_ref - ee_center_ref

# Slider carriage reference points
A_slider_ref = arr(next(s for s in model["global"]["sliders"] if s["name"] == "Slider 8")["point_world"])
B_slider_ref = arr(next(s for s in model["global"]["sliders"] if s["name"] == "Slider 21")["point_world"])
C_slider_ref = arr(next(s for s in model["global"]["sliders"] if s["name"] == "Slider 14")["point_world"])

# Ground anchors for display only, from rich export convention:
# These are the nominal fixed slider origins you used earlier.
A_slider_ground = arr([12.7, 0.0, 0.0])
B_slider_ground = arr([12.7, 0.0, 12.7])
C_slider_ground = arr([12.7, 0.0, 12.7])

# Leg A joints
A_base_ref = arr(next(j for j in legA["joints"] if j["name"] == "Revolute 9")["center_world"])
A_elbow_ref = arr(next(j for j in legA["joints"] if j["name"] == "Revolute 10")["center_world"])
A_L1 = norm(A_elbow_ref - A_base_ref)
A_L2 = norm(A_ee_ref - A_elbow_ref)
A_base_offset = A_base_ref - A_slider_ref

# Leg B joints
B_base_ref = arr(next(j for j in legB["joints"] if j["name"] == "Revolute 22")["center_world"])
B_elbow_ref = arr(next(j for j in legB["joints"] if j["name"] == "Revolute 2")["center_world"])
B_L1 = norm(B_elbow_ref - B_base_ref)
B_L2 = norm(B_ee_ref - B_elbow_ref)
B_base_offset = B_base_ref - B_slider_ref

# Leg C joints
C_base1_ref = arr(next(j for j in legC["joints"] if j["name"] == "Revolute 29")["center_world"])
C_elbow1_ref = arr(next(j for j in legC["joints"] if j["name"] == "Revolute 16")["center_world"])
C_base2_ref = arr(next(j for j in legC["joints"] if j["name"] == "Revolute 31")["center_world"])
C_elbow2_ref = arr(next(j for j in legC["joints"] if j["name"] == "Revolute 19")["center_world"])

C_L1a = norm(C_elbow1_ref - C_base1_ref)
C_L2a = norm(C_ee1_ref - C_elbow1_ref)
C_L1b = norm(C_elbow2_ref - C_base2_ref)
C_L2b = norm(C_ee2_ref - C_elbow2_ref)

C_base1_offset = C_base1_ref - C_slider_ref
C_base2_offset = C_base2_ref - C_slider_ref

# 2D reference elbows for branch selection
A_elbow_ref_2d = to2d(A_elbow_ref, legA)
B_elbow_ref_2d = to2d(B_elbow_ref, legB)
C_elbow1_ref_2d = to2d(C_elbow1_ref, legC)
C_elbow2_ref_2d = to2d(C_elbow2_ref, legC)

# Reference actuator coordinates
sA_ref = float(next(s for s in model["global"]["sliders"] if s["name"] == "Slider 8")["slide_value"])
sB_ref = float(next(s for s in model["global"]["sliders"] if s["name"] == "Slider 21")["slide_value"])
sC_ref = float(next(s for s in model["global"]["sliders"] if s["name"] == "Slider 14")["slide_value"])


# ============================================================
# Forward solve: actuator inputs -> EE center
# ============================================================
def actuator_positions(sA, sB, sC):
    qA = sA - sA_ref
    qB = sB - sB_ref
    qC = sC - sC_ref

    A_slider = A_slider_ref + qA * global_x
    B_slider = B_slider_ref + qB * global_x
    C_slider = C_slider_ref + qC * global_x

    A_base = A_base_ref + qA * global_x
    B_base = B_base_ref + qB * global_x
    C_base1 = C_base1_ref + qC * global_x
    C_base2 = C_base2_ref + qC * global_x

    return A_slider, B_slider, C_slider, A_base, B_base, C_base1, C_base2


def solve_ee_center_from_actuators(sA, sB, sC, x0=None):
    """
    Solve for EE center so all leg distance constraints are satisfied.
    """
    A_slider, B_slider, C_slider, A_base, B_base, C_base1, C_base2 = actuator_positions(sA, sB, sC)

    if x0 is None:
        x0 = ee_center_ref.copy()

    def residual(X):
        ee = np.array(X, dtype=float)

        A_ee = ee + A_ee_offset
        B_ee = ee + B_ee_offset
        C_ee1 = ee + C_ee1_offset
        C_ee2 = ee + C_ee2_offset

        return np.array([
            norm(A_ee - A_base) - (A_L1 + A_L2),
            norm(B_ee - B_base) - (B_L1 + B_L2),
            norm(C_ee1 - C_base1) - (C_L1a + C_L2a),
            norm(C_ee2 - C_base2) - (C_L1b + C_L2b),
        ])

    sol = least_squares(residual, x0, method="lm")
    return sol.x, sol.success, sol.cost


def reconstruct_configuration(sA, sB, sC, ee_center):
    A_slider, B_slider, C_slider, A_base, B_base, C_base1, C_base2 = actuator_positions(sA, sB, sC)

    A_ee = ee_center + A_ee_offset
    B_ee = ee_center + B_ee_offset
    C_ee1 = ee_center + C_ee1_offset
    C_ee2 = ee_center + C_ee2_offset

    # Leg A elbow
    A_B2 = to2d(A_base, legA)
    A_E2 = to2d(A_ee, legA)
    A_cands = circle_intersections(A_B2, A_E2, A_L1, A_L2)
    A_elbow2 = choose_closest(A_cands, A_elbow_ref_2d)
    A_elbow = to3d(A_elbow2, legA) if A_elbow2 is not None else None

    # Leg B elbow
    B_B2 = to2d(B_base, legB)
    B_E2 = to2d(B_ee, legB)
    B_cands = circle_intersections(B_B2, B_E2, B_L1, B_L2)
    B_elbow2 = choose_closest(B_cands, B_elbow_ref_2d)
    B_elbow = to3d(B_elbow2, legB) if B_elbow2 is not None else None

    # Leg C elbows
    C_B1_2 = to2d(C_base1, legC)
    C_E1_2 = to2d(C_ee1, legC)
    C1_cands = circle_intersections(C_B1_2, C_E1_2, C_L1a, C_L2a)
    C_elbow1_2 = choose_closest(C1_cands, C_elbow1_ref_2d)
    C_elbow1 = to3d(C_elbow1_2, legC) if C_elbow1_2 is not None else None

    C_B2_2 = to2d(C_base2, legC)
    C_E2_2 = to2d(C_ee2, legC)
    C2_cands = circle_intersections(C_B2_2, C_E2_2, C_L1b, C_L2b)
    C_elbow2_2 = choose_closest(C2_cands, C_elbow2_ref_2d)
    C_elbow2 = to3d(C_elbow2_2, legC) if C_elbow2_2 is not None else None

    return {
        "A_slider": A_slider, "B_slider": B_slider, "C_slider": C_slider,
        "A_base": A_base, "B_base": B_base, "C_base1": C_base1, "C_base2": C_base2,
        "A_elbow": A_elbow, "B_elbow": B_elbow, "C_elbow1": C_elbow1, "C_elbow2": C_elbow2,
        "A_ee": A_ee, "B_ee": B_ee, "C_ee1": C_ee1, "C_ee2": C_ee2,
        "ee_center": ee_center,
    }


# ============================================================
# GUI
# ============================================================
fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(left=0.08, bottom=0.25)

ax_sA = plt.axes([0.18, 0.15, 0.65, 0.03])
ax_sB = plt.axes([0.18, 0.10, 0.65, 0.03])
ax_sC = plt.axes([0.18, 0.05, 0.65, 0.03])

sl_sA = Slider(ax_sA, "Actuator A", sA_ref - 20.0, sA_ref + 20.0, valinit=sA_ref)
sl_sB = Slider(ax_sB, "Actuator B", sB_ref - 20.0, sB_ref + 20.0, valinit=sB_ref)
sl_sC = Slider(ax_sC, "Actuator C", sC_ref - 20.0, sC_ref + 20.0, valinit=sC_ref)

ax_reset = plt.axes([0.85, 0.90, 0.10, 0.05])
btn_reset = Button(ax_reset, "Reset")

last_ee = ee_center_ref.copy()


def draw_segment(p1, p2, **kwargs):
    p1 = arr(p1); p2 = arr(p2)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], **kwargs)

def draw_point(p, **kwargs):
    p = arr(p)
    ax.scatter([p[0]], [p[1]], [p[2]], **kwargs)

def update(_=None):
    global last_ee
    ax.cla()

    sA = sl_sA.val
    sB = sl_sB.val
    sC = sl_sC.val

    ee_center, success, cost = solve_ee_center_from_actuators(sA, sB, sC, x0=last_ee)
    last_ee = ee_center.copy()

    cfg = reconstruct_configuration(sA, sB, sC, ee_center)

    # Draw fixed slider bases
    draw_point(A_slider_ground, s=55, marker="s")
    draw_point(B_slider_ground, s=55, marker="s")
    draw_point(C_slider_ground, s=55, marker="s")

    # Draw prismatic actuators explicitly
    draw_segment(A_slider_ground, cfg["A_slider"], linewidth=3)
    draw_segment(B_slider_ground, cfg["B_slider"], linewidth=3)
    draw_segment(C_slider_ground, cfg["C_slider"], linewidth=3)

    # Leg A
    draw_segment(cfg["A_slider"], cfg["A_base"], linewidth=2)
    if cfg["A_elbow"] is not None:
        draw_segment(cfg["A_base"], cfg["A_elbow"], linewidth=2)
        draw_segment(cfg["A_elbow"], cfg["A_ee"], linewidth=2)
        draw_point(cfg["A_elbow"], s=35)
    draw_point(cfg["A_slider"], s=45)
    draw_point(cfg["A_base"], s=35)
    draw_point(cfg["A_ee"], s=50)

    # Leg B
    draw_segment(cfg["B_slider"], cfg["B_base"], linewidth=2)
    if cfg["B_elbow"] is not None:
        draw_segment(cfg["B_base"], cfg["B_elbow"], linewidth=2)
        draw_segment(cfg["B_elbow"], cfg["B_ee"], linewidth=2)
        draw_point(cfg["B_elbow"], s=35)
    draw_point(cfg["B_slider"], s=45)
    draw_point(cfg["B_base"], s=35)
    draw_point(cfg["B_ee"], s=50)

    # Leg C
    draw_segment(cfg["C_slider"], cfg["C_base1"], linewidth=2)
    draw_segment(cfg["C_slider"], cfg["C_base2"], linewidth=2)
    if cfg["C_elbow1"] is not None:
        draw_segment(cfg["C_base1"], cfg["C_elbow1"], linewidth=2)
        draw_segment(cfg["C_elbow1"], cfg["C_ee1"], linewidth=2)
        draw_point(cfg["C_elbow1"], s=35)
    if cfg["C_elbow2"] is not None:
        draw_segment(cfg["C_base2"], cfg["C_elbow2"], linewidth=2)
        draw_segment(cfg["C_elbow2"], cfg["C_ee2"], linewidth=2)
        draw_point(cfg["C_elbow2"], s=35)
    draw_point(cfg["C_slider"], s=45)
    draw_point(cfg["C_base1"], s=35)
    draw_point(cfg["C_base2"], s=35)
    draw_point(cfg["C_ee1"], s=50)
    draw_point(cfg["C_ee2"], s=50)

    # EE polygon
    draw_segment(cfg["A_ee"], cfg["B_ee"], linewidth=1.5, linestyle="--")
    draw_segment(cfg["B_ee"], cfg["C_ee1"], linewidth=1.5, linestyle="--")
    draw_segment(cfg["C_ee1"], cfg["C_ee2"], linewidth=1.5, linestyle="--")
    draw_segment(cfg["C_ee2"], cfg["A_ee"], linewidth=1.5, linestyle="--")
    draw_point(cfg["ee_center"], s=80, marker="x")

    ax.set_title("Actuator-driven Tripteron visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    all_pts = np.vstack([
        A_slider_ground, B_slider_ground, C_slider_ground,
        cfg["A_slider"], cfg["A_base"], cfg["A_ee"],
        cfg["B_slider"], cfg["B_base"], cfg["B_ee"],
        cfg["C_slider"], cfg["C_base1"], cfg["C_base2"], cfg["C_ee1"], cfg["C_ee2"],
        cfg["ee_center"]
    ] + [p for p in [cfg["A_elbow"], cfg["B_elbow"], cfg["C_elbow1"], cfg["C_elbow2"]] if p is not None])

    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    ctr = 0.5 * (mins + maxs)
    rng = max((maxs - mins).max(), 1.0) * 0.65

    ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
    ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
    ax.set_zlim(ctr[2] - rng, ctr[2] + rng)

    txt = (
        f"sA={sA:.2f}, sB={sB:.2f}, sC={sC:.2f}\n"
        f"EE = [{cfg['ee_center'][0]:.2f}, {cfg['ee_center'][1]:.2f}, {cfg['ee_center'][2]:.2f}]\n"
        f"solve success={success}, cost={cost:.3e}"
    )
    ax.text2D(0.02, 0.98, txt, transform=ax.transAxes, va="top")

    fig.canvas.draw_idle()

def on_reset(event):
    sl_sA.reset()
    sl_sB.reset()
    sl_sC.reset()

btn_reset.on_clicked(on_reset)
sl_sA.on_changed(update)
sl_sB.on_changed(update)
sl_sC.on_changed(update)

update()
plt.show()