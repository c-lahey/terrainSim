"""
fig3b_geometry_optimizer.py

Geometry optimizer for Fig. 3(b)-style planar 2T1R mechanism.

Requires:
    scipy
    numpy
    fig3b_2t1r_validation.py

Searches over:
    A2x
    A3x
    active_link = l1a = l2a
    passive_link = l1b = l2b
    l3a
    l3b
    platform_length

Scores candidate geometries at workspace extrema using:
    - reachability
    - worst-case actuator torque under representative forces
    - Jacobian condition number
"""

from __future__ import annotations

import math
import numpy as np
from scipy.optimize import differential_evolution

from fig3b_2t1r_validation import Fig3B2T1R, Fig3BParams


# ---------------------------------------------------------------------
# Desired workspace
# ---------------------------------------------------------------------

# If treating sketch units as inches:
# 500 mm = 19.685 in
# 300 mm = 11.811 in
WORKSPACE_WIDTH = 500.0 / 25.4
WORKSPACE_HEIGHT = 300.0 / 25.4

# Workspace center in sketch units.
CX = 10.0
CY = 10.0

X_MIN = CX - WORKSPACE_WIDTH / 2.0
X_MAX = CX + WORKSPACE_WIDTH / 2.0
Y_MIN = CY - WORKSPACE_HEIGHT / 2.0
Y_MAX = CY + WORKSPACE_HEIGHT / 2.0

THETA_MIN = math.radians(-20.0)
THETA_MAX = math.radians(20.0)


# Representative forces to resist at the platform center.
# Units are arbitrary but should be consistent.
# If length is inches and force is Newtons, torques are N*in.
TEST_WRENCHES = [
    np.array([10.0, 0.0, 0.0]),    # +Fx
    np.array([-10.0, 0.0, 0.0]),   # -Fx
    np.array([0.0, 10.0, 0.0]),    # +Fy
    np.array([0.0, -10.0, 0.0]),   # -Fy
]


def make_test_poses() -> list[np.ndarray]:
    """
    Test workspace extrema.

    Includes:
        - 4 xy corners
        - center
        - theta = min, 0, max

    This is not a full workspace proof, but it is a useful optimizer target.
    """
    xy_points = [
        (X_MIN, Y_MIN),
        (X_MIN, Y_MAX),
        (X_MAX, Y_MIN),
        (X_MAX, Y_MAX),
        (CX, CY),
        (X_MIN, CY),
        (X_MAX, CY),
        (CX, Y_MIN),
        (CX, Y_MAX),
    ]

    theta_points = [THETA_MIN, 0.0, THETA_MAX]

    poses = []
    for x, y in xy_points:
        for theta in theta_points:
            poses.append(np.array([x, y, theta], dtype=float))

    return poses


TEST_POSES = make_test_poses()


# ---------------------------------------------------------------------
# Candidate geometry
# ---------------------------------------------------------------------

def params_from_vector(v: np.ndarray) -> Fig3BParams:
    """
    v = [
        A2x,
        A3x,
        active_link,
        passive_link,
        l3a,
        l3b,
        platform_length,
    ]
    """
    A2x, A3x, active_link, passive_link, l3a, l3b, platform_length = v

    return Fig3BParams(
        A1=(0.0, 0.0),
        A2=(A2x, 0.0),
        A3=(A3x, 0.0),

        # symmetric five-bar
        l1a=active_link,
        l2a=active_link,
        l1b=passive_link,
        l2b=passive_link,

        # orientation limb
        l3a=l3a,
        l3b=l3b,

        platform_length=platform_length,
    )


def choose_reasonable_solution(kin: Fig3B2T1R, pose: np.ndarray):
    """
    Pick a branch for evaluation.

    For now, choose the branch with:
        - valid IK
        - lowest condition number

    Later, you can enforce a specific physical branch.
    """
    sols = [s for s in kin.ik_all(pose) if s.valid]

    if not sols:
        return None

    best_sol = None
    best_score = float("inf")

    for sol in sols:
        try:
            J = kin.numerical_jacobian_dq_dp(pose, q_prev=sol.q)
            cond = np.linalg.cond(J)
        except Exception:
            continue

        if not np.isfinite(cond):
            continue

        # Penalize configurations where the actuated links point below ground.
        # This is optional, but helps avoid silly branches.
        pts = kin.joint_points(sol.q, pose)
        below_penalty = 0.0
        for name in ["B1", "B2", "B3"]:
            if pts[name][1] < -1.0:
                below_penalty += 100.0 * abs(pts[name][1])

        score = cond + below_penalty

        if score < best_score:
            best_score = score
            best_sol = sol

    return best_sol


# ---------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------

def evaluate_geometry(v: np.ndarray, verbose: bool = False) -> float:
    """
    Lower is better.

    Objective terms:
        - large penalty if unreachable
        - worst-case actuator torque
        - average actuator torque
        - worst condition number
        - mild size penalty
    """
    params = params_from_vector(v)
    kin = Fig3B2T1R(params)

    worst_tau = 0.0
    mean_tau_accum = 0.0
    tau_count = 0

    worst_cond = 0.0
    unreachable_count = 0

    for pose in TEST_POSES:
        sol = choose_reasonable_solution(kin, pose)

        if sol is None:
            unreachable_count += 1
            continue

        q = sol.q

        try:
            J_qp = kin.numerical_jacobian_dq_dp(pose, q_prev=q)
            cond = float(np.linalg.cond(J_qp))
        except Exception:
            unreachable_count += 1
            continue

        if not np.isfinite(cond):
            unreachable_count += 1
            continue

        worst_cond = max(worst_cond, cond)

        for wrench in TEST_WRENCHES:
            try:
                tau = np.linalg.solve(J_qp.T, wrench)
            except Exception:
                unreachable_count += 1
                continue

            tau_norm_inf = float(np.max(np.abs(tau)))
            worst_tau = max(worst_tau, tau_norm_inf)
            mean_tau_accum += tau_norm_inf
            tau_count += 1

    # Hard penalty for unreachable extrema.
    if unreachable_count > 0:
        return 1e6 + 1e5 * unreachable_count

    if tau_count == 0:
        return 1e9

    mean_tau = mean_tau_accum / tau_count

    # Mild geometry-size penalty. Avoids making every link huge.
    A2x, A3x, active_link, passive_link, l3a, l3b, platform_length = v
    size_sum = active_link + passive_link + l3a + l3b + platform_length

    # Optional spacing sanity penalties.
    spacing_penalty = 0.0

    # A3 should generally be to the right of A2.
    if A3x <= A2x:
        spacing_penalty += 1e5 * (A2x - A3x + 1.0)

    # Avoid platform longer than the whole base spacing.
    if platform_length > A3x:
        spacing_penalty += 1e3 * (platform_length - A3x)

    # Objective weights.
    # You can tune these depending on whether you care more about torque
    # or conditioning.
    score = (
        10.0 * worst_tau
        + 1.0 * mean_tau
        + 0.2 * worst_cond
        + 0.02 * size_sum
        + spacing_penalty
    )

    if verbose:
        print("v =", v)
        print("score =", score)
        print("worst_tau =", worst_tau)
        print("mean_tau =", mean_tau)
        print("worst_cond =", worst_cond)
        print("size_sum =", size_sum)

    return float(score)


def run_optimization():
    """
    Run global optimization.

    Bounds are in sketch units, currently treated like inches.
    """
    bounds = [
        (6.0, 18.0),   # A2x
        (14.0, 40.0),  # A3x
        (8.0, 30.0),   # active_link
        (8.0, 35.0),   # passive_link
        (8.0, 30.0),   # l3a
        (8.0, 35.0),   # l3b
        (6.0, 25.0),   # platform_length
    ]

    result = differential_evolution(
        evaluate_geometry,
        bounds=bounds,
        strategy="best1bin",
        maxiter=80,
        popsize=12,
        tol=1e-3,
        polish=True,
        workers=1,  # set to -1 for parallel if safe on your machine
        updating="immediate",
        seed=3,
        disp=True,
    )

    print("\n=== OPTIMIZATION RESULT ===")
    print("success:", result.success)
    print("message:", result.message)
    print("score:", result.fun)
    print("v:", result.x)

    A2x, A3x, active_link, passive_link, l3a, l3b, platform_length = result.x

    print("\nRecommended geometry:")
    print(f"A2x             = {A2x:.3f}")
    print(f"A3x             = {A3x:.3f}")
    print(f"active_link     = {active_link:.3f}")
    print(f"passive_link    = {passive_link:.3f}")
    print(f"l3a             = {l3a:.3f}")
    print(f"l3b             = {l3b:.3f}")
    print(f"platform_length = {platform_length:.3f}")

    print("\nSame geometry in mm, assuming sketch units are inches:")
    print(f"A2x             = {25.4 * A2x:.1f} mm")
    print(f"A3x             = {25.4 * A3x:.1f} mm")
    print(f"active_link     = {25.4 * active_link:.1f} mm")
    print(f"passive_link    = {25.4 * passive_link:.1f} mm")
    print(f"l3a             = {25.4 * l3a:.1f} mm")
    print(f"l3b             = {25.4 * l3b:.1f} mm")
    print(f"platform_length = {25.4 * platform_length:.1f} mm")

    print("\nDetailed evaluation:")
    evaluate_geometry(result.x, verbose=True)

    return result


if __name__ == "__main__":
    run_optimization()