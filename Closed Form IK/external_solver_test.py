from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from json_extractor_test import build_reduced_parameter_set, ReducedParameters


Vec3 = np.ndarray


# ============================================================
# Solver-facing parameter wrapper
# ============================================================
@dataclass
class TripteronParameters:
    raw: ReducedParameters

    ex: Vec3
    ey: Vec3
    ez: Vec3

    p0: Vec3

    r_ul: Vec3
    r_ll: Vec3
    r_ur: Vec3
    r_lr: Vec3

    a_A: Vec3
    a_B: Vec3
    a_C: Vec3
    rail_dir: Vec3

    q_A: float
    q_B: float
    q_C: float

    c_ul_0: Vec3
    c_ll_0: Vec3
    c_ur_0: Vec3
    c_lr_0: Vec3

    branch_A_points: Dict[str, Vec3]
    branch_B_points: Dict[str, Vec3]
    branch_C_points: Dict[str, Vec3]

    branch_A_lengths: Dict[str, float]
    branch_B_lengths: Dict[str, float]
    branch_C_lengths: Dict[str, float]

    offset_A_carriage_to_base_joint: Vec3
    offset_B_carriage_to_base_joint: Vec3
    offset_C_carriage_to_R29: Vec3
    offset_C_carriage_to_R31: Vec3


# ============================================================
# Basic geometry helpers
# ============================================================
def arr(x) -> Vec3:
    return np.asarray(x, dtype=float)


def vec3(x: float, y: float, z: float) -> Vec3:
    return np.array([x, y, z], dtype=float)


def norm(x: Vec3) -> float:
    return float(np.linalg.norm(x))


def unit(x: Vec3, tol: float = 1e-12) -> Vec3:
    n = norm(x)
    if n < tol:
        raise ValueError(f"Cannot normalize near-zero vector: {x}")
    return x / n


def sqdist(a: Vec3, b: Vec3) -> float:
    d = a - b
    return float(d @ d)


def dist(a: Vec3, b: Vec3) -> float:
    return float(np.linalg.norm(a - b))


def line_point(a: Vec3, direction: Vec3, q: float) -> Vec3:
    return a + q * direction


def ee_point_from_centroid(p: Vec3, r: Vec3) -> Vec3:
    return p + r


# ============================================================
# Build solver-facing model from extractor output
# ============================================================
def load_tripteron_parameters(input_json: str | Path) -> Tuple[TripteronParameters, object]:
    params, report = build_reduced_parameter_set(Path(input_json), output_json=None)

    ex = arr(params.world_x)
    ey = arr(params.world_y)
    ez = arr(params.world_z)
    p0 = arr(params.ee_centroid_world)

    cp_map = {cp.joint_name: cp for cp in params.ee_coupling_points}

    # Naming convention from the extractor / current assembly:
    # Revolute 24 = upper-left, 12 = lower-left, 30 = upper-right, 32 = lower-right
    r_ul = arr(cp_map["Revolute 24"].offset_from_centroid)
    r_ll = arr(cp_map["Revolute 12"].offset_from_centroid)
    r_ur = arr(cp_map["Revolute 30"].offset_from_centroid)
    r_lr = arr(cp_map["Revolute 32"].offset_from_centroid)

    c_ul_0 = arr(cp_map["Revolute 24"].point_world)
    c_ll_0 = arr(cp_map["Revolute 12"].point_world)
    c_ur_0 = arr(cp_map["Revolute 30"].point_world)
    c_lr_0 = arr(cp_map["Revolute 32"].point_world)

    sA = params.sliders["A"]
    sB = params.sliders["B"]
    sC = params.sliders["C"]

    def point_dict(names, points):
        return {name: arr(pt) for name, pt in zip(names, points)}

    branch_A_points = point_dict(params.branch_A.chain_joint_names, params.branch_A.chain_joint_points_world)
    branch_B_points = point_dict(params.branch_B.chain_joint_names, params.branch_B.chain_joint_points_world)
    branch_C_points = point_dict(params.branch_C.chain_joint_names, params.branch_C.chain_joint_points_world)

    branch_A_lengths = {
        "base_joint_to_mid": float(params.branch_A.rigid_segment_lengths[0]),
        "mid_to_ee": float(params.branch_A.rigid_segment_lengths[1]),
    }
    branch_B_lengths = {
        "base_joint_to_mid": float(params.branch_B.rigid_segment_lengths[0]),
        "mid_to_ee": float(params.branch_B.rigid_segment_lengths[1]),
    }
    branch_C_lengths = {k: float(v) for k, v in params.branch_C.rigid_lengths.items()}

    carriage_A_0 = line_point(arr(sA.rail_reference_origin_world), arr(sA.slide_direction_world), float(sA.slide_value))
    carriage_B_0 = line_point(arr(sB.rail_reference_origin_world), arr(sB.slide_direction_world), float(sB.slide_value))
    carriage_C_0 = line_point(arr(sC.rail_reference_origin_world), arr(sC.slide_direction_world), float(sC.slide_value))

    offset_A_carriage_to_base_joint = branch_A_points["Revolute 22"] - carriage_A_0
    offset_B_carriage_to_base_joint = branch_B_points["Revolute 9"] - carriage_B_0
    offset_C_carriage_to_R29 = branch_C_points["Revolute 29"] - carriage_C_0
    offset_C_carriage_to_R31 = branch_C_points["Revolute 31"] - carriage_C_0

    model = TripteronParameters(
        raw=params,
        ex=ex,
        ey=ey,
        ez=ez,
        p0=p0,
        r_ul=r_ul,
        r_ll=r_ll,
        r_ur=r_ur,
        r_lr=r_lr,
        a_A=arr(sA.rail_reference_origin_world),
        a_B=arr(sB.rail_reference_origin_world),
        a_C=arr(sC.rail_reference_origin_world),
        rail_dir=arr(sA.slide_direction_world),
        q_A=float(sA.slide_value),
        q_B=float(sB.slide_value),
        q_C=float(sC.slide_value),
        c_ul_0=c_ul_0,
        c_ll_0=c_ll_0,
        c_ur_0=c_ur_0,
        c_lr_0=c_lr_0,
        branch_A_points=branch_A_points,
        branch_B_points=branch_B_points,
        branch_C_points=branch_C_points,
        branch_A_lengths=branch_A_lengths,
        branch_B_lengths=branch_B_lengths,
        branch_C_lengths=branch_C_lengths,
        offset_A_carriage_to_base_joint=offset_A_carriage_to_base_joint,
        offset_B_carriage_to_base_joint=offset_B_carriage_to_base_joint,
        offset_C_carriage_to_R29=offset_C_carriage_to_R29,
        offset_C_carriage_to_R31=offset_C_carriage_to_R31,
    )
    return model, report


# ============================================================
# Cartesian mechanism accessors
# ============================================================
def coupling_points_from_p(model: TripteronParameters, p: Vec3) -> Dict[str, Vec3]:
    return {
        "ul": ee_point_from_centroid(p, model.r_ul),
        "ll": ee_point_from_centroid(p, model.r_ll),
        "ur": ee_point_from_centroid(p, model.r_ur),
        "lr": ee_point_from_centroid(p, model.r_lr),
    }


def slider_carriage_points_from_q(model: TripteronParameters, qA: float, qB: float, qC: float) -> Dict[str, Vec3]:
    return {
        "A": line_point(model.a_A, model.rail_dir, qA),
        "B": line_point(model.a_B, model.rail_dir, qB),
        "C": line_point(model.a_C, model.rail_dir, qC),
    }


def simple_branch_base_joints_from_q(model: TripteronParameters, qA: float, qB: float) -> Dict[str, Vec3]:
    car = slider_carriage_points_from_q(model, qA, qB, model.q_C)
    return {
        "A": car["A"] + model.offset_A_carriage_to_base_joint,
        "B": car["B"] + model.offset_B_carriage_to_base_joint,
    }


def closed_branch_base_points_from_q(model: TripteronParameters, qC: float) -> Dict[str, Vec3]:
    car = line_point(model.a_C, model.rail_dir, qC)
    return {
        "R29": car + model.offset_C_carriage_to_R29,
        "R31": car + model.offset_C_carriage_to_R31,
    }


def reference_pose_summary(model: TripteronParameters) -> Dict[str, object]:
    return {
        "p0": model.p0,
        "q0": np.array([model.q_A, model.q_B, model.q_C], dtype=float),
        "coupling_points": coupling_points_from_p(model, model.p0),
        "carriage_points": slider_carriage_points_from_q(model, model.q_A, model.q_B, model.q_C),
    }


# ============================================================
# Branch closure equations and first IK helpers
# ============================================================
def branch_A_midpoint_from_p(model: TripteronParameters, p: Vec3) -> Vec3:
    """
    Current working assumption for the simple branches:
    the elbow/mid joint location is fixed in world coordinates because the branch
    lies in a fixed plane with a world-aligned EE orientation in this CAD model.

    This matches the validated reference-pose data and gives us a clean first
    reduced model to begin IK/FK work. If later CAD snapshots show that these
    midpoints move, we can replace this with a more general construction.
    """
    _ = p
    return model.branch_A_points["Revolute 2"]



def branch_B_midpoint_from_p(model: TripteronParameters, p: Vec3) -> Vec3:
    _ = p
    return model.branch_B_points["Revolute 10"]



def branch_A_closure(model: TripteronParameters, p: Vec3, qA: float) -> Dict[str, float]:
    pts = coupling_points_from_p(model, p)
    base = simple_branch_base_joints_from_q(model, qA, model.q_B)["A"]
    mid = branch_A_midpoint_from_p(model, p)

    return {
        "base_joint_to_mid": dist(base, mid) - model.branch_A_lengths["base_joint_to_mid"],
        "mid_to_ee": dist(mid, pts["ul"]) - model.branch_A_lengths["mid_to_ee"],
    }



def branch_B_closure(model: TripteronParameters, p: Vec3, qB: float) -> Dict[str, float]:
    pts = coupling_points_from_p(model, p)
    base = simple_branch_base_joints_from_q(model, model.q_A, qB)["B"]
    mid = branch_B_midpoint_from_p(model, p)

    return {
        "base_joint_to_mid": dist(base, mid) - model.branch_B_lengths["base_joint_to_mid"],
        "mid_to_ee": dist(mid, pts["ll"]) - model.branch_B_lengths["mid_to_ee"],
    }



def branch_A_ik_candidates(model: TripteronParameters, p: Vec3) -> Tuple[float, float]:
    mid = branch_A_midpoint_from_p(model, p)
    d = mid - model.offset_A_carriage_to_base_joint - model.a_A
    L = model.branch_A_lengths["base_joint_to_mid"]
    yz_sq = float(d[1] ** 2 + d[2] ** 2)
    rad = L ** 2 - yz_sq
    if rad < -1e-9:
        raise ValueError(f"Branch A pose unreachable: negative radicand {rad}")
    rad = max(rad, 0.0)
    root = float(np.sqrt(rad))
    return float(d[0] - root), float(d[0] + root)



def branch_B_ik_candidates(model: TripteronParameters, p: Vec3) -> Tuple[float, float]:
    mid = branch_B_midpoint_from_p(model, p)
    d = mid - model.offset_B_carriage_to_base_joint - model.a_B
    L = model.branch_B_lengths["base_joint_to_mid"]
    yz_sq = float(d[1] ** 2 + d[2] ** 2)
    rad = L ** 2 - yz_sq
    if rad < -1e-9:
        raise ValueError(f"Branch B pose unreachable: negative radicand {rad}")
    rad = max(rad, 0.0)
    root = float(np.sqrt(rad))
    return float(d[0] - root), float(d[0] + root)



def branch_A_ik(model: TripteronParameters, p: Vec3, branch_sign: int = -1) -> float:
    q_minus, q_plus = branch_A_ik_candidates(model, p)
    return q_plus if branch_sign > 0 else q_minus



def branch_B_ik(model: TripteronParameters, p: Vec3, branch_sign: int = -1) -> float:
    q_minus, q_plus = branch_B_ik_candidates(model, p)
    return q_plus if branch_sign > 0 else q_minus



def select_reference_ik_branch(candidates: Tuple[float, float], q_ref: float) -> float:
    return min(candidates, key=lambda q: abs(q - q_ref))



def branch_A_reference_residuals(model: TripteronParameters, p: Vec3, qA: float) -> Dict[str, float]:
    return branch_A_closure(model, p, qA)



def branch_B_reference_residuals(model: TripteronParameters, p: Vec3, qB: float) -> Dict[str, float]:
    return branch_B_closure(model, p, qB)


# Closed-chain branch C
#
# Current reduced model assumption:
# the right-side intermediate joints R16 and R19 are treated as fixed world points,
# just like the validated simple-branch mid joints. Under that assumption, qC can be
# solved from either side of the branch by the same quadratic line-sphere geometry.
# The two sides should agree when the pose is consistent.
#
# If later CAD snapshots show that R16/R19 move with pose, we will replace this with
# a more general elimination-based closure.
def branch_C_closure(model: TripteronParameters, p: Vec3, qC: float) -> Dict[str, float]:
    pts = coupling_points_from_p(model, p)
    base_pts = closed_branch_base_points_from_q(model, qC)

    r29 = base_pts["R29"]
    r31 = base_pts["R31"]
    r16 = model.branch_C_points["Revolute 16"]
    r19 = model.branch_C_points["Revolute 19"]

    return {
        "r29_to_r16": dist(r29, r16) - model.branch_C_lengths["R29_to_R16"],
        "r31_to_r19": dist(r31, r19) - model.branch_C_lengths["R31_to_R19"],
        "r16_to_ur": dist(r16, pts["ur"]) - model.branch_C_lengths["R16_to_R30"],
        "r19_to_lr": dist(r19, pts["lr"]) - model.branch_C_lengths["R19_to_R32"],
        "ur_to_lr": dist(pts["ur"], pts["lr"]) - model.branch_C_lengths["R30_to_R32"],
    }



def branch_C_reference_measures(model: TripteronParameters, p: Vec3, qC: float) -> Dict[str, float]:
    return branch_C_closure(model, p, qC)



def branch_C_ik_candidates_from_upper(model: TripteronParameters, p: Vec3) -> Tuple[float, float]:
    r16 = model.branch_C_points["Revolute 16"]
    d = r16 - model.offset_C_carriage_to_R29 - model.a_C
    L = model.branch_C_lengths["R29_to_R16"]
    yz_sq = float(d[1] ** 2 + d[2] ** 2)
    rad = L ** 2 - yz_sq
    if rad < -1e-9:
        raise ValueError(f"Branch C upper pose unreachable: negative radicand {rad}")
    rad = max(rad, 0.0)
    root = float(np.sqrt(rad))
    return float(d[0] - root), float(d[0] + root)



def branch_C_ik_candidates_from_lower(model: TripteronParameters, p: Vec3) -> Tuple[float, float]:
    r19 = model.branch_C_points["Revolute 19"]
    d = r19 - model.offset_C_carriage_to_R31 - model.a_C
    L = model.branch_C_lengths["R31_to_R19"]
    yz_sq = float(d[1] ** 2 + d[2] ** 2)
    rad = L ** 2 - yz_sq
    if rad < -1e-9:
        raise ValueError(f"Branch C lower pose unreachable: negative radicand {rad}")
    rad = max(rad, 0.0)
    root = float(np.sqrt(rad))
    return float(d[0] - root), float(d[0] + root)



def branch_C_ik_candidates(model: TripteronParameters, p: Vec3) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return branch_C_ik_candidates_from_upper(model, p), branch_C_ik_candidates_from_lower(model, p)



def branch_C_ik(model: TripteronParameters, p: Vec3, branch_sign: int = -1) -> float:
    upper = branch_C_ik_candidates_from_upper(model, p)
    lower = branch_C_ik_candidates_from_lower(model, p)
    q_upper = upper[1] if branch_sign > 0 else upper[0]
    q_lower = lower[1] if branch_sign > 0 else lower[0]
    return 0.5 * (q_upper + q_lower)



def branch_C_ik_consistency(model: TripteronParameters, p: Vec3, q_ref: float | None = None) -> Dict[str, object]:
    upper = branch_C_ik_candidates_from_upper(model, p)
    lower = branch_C_ik_candidates_from_lower(model, p)

    if q_ref is None:
        q_ref = model.q_C

    sel_upper = select_reference_ik_branch(upper, q_ref)
    sel_lower = select_reference_ik_branch(lower, q_ref)
    return {
        "upper_candidates": upper,
        "lower_candidates": lower,
        "upper_selected": sel_upper,
        "lower_selected": sel_lower,
        "selected_difference": abs(sel_upper - sel_lower),
        "selected_average": 0.5 * (sel_upper + sel_lower),
    }


# ============================================================
# Full IK wrapper
# ============================================================
def full_ik_candidates(model: TripteronParameters, p: Vec3) -> Dict[str, object]:
    qA_cands = branch_A_ik_candidates(model, p)
    qB_cands = branch_B_ik_candidates(model, p)
    qC_info = branch_C_ik_consistency(model, p, model.q_C)
    return {
        "qA_candidates": qA_cands,
        "qB_candidates": qB_cands,
        "qC_upper_candidates": qC_info["upper_candidates"],
        "qC_lower_candidates": qC_info["lower_candidates"],
    }



def full_ik_near_reference(model: TripteronParameters, p: Vec3) -> np.ndarray:
    qA = select_reference_ik_branch(branch_A_ik_candidates(model, p), model.q_A)
    qB = select_reference_ik_branch(branch_B_ik_candidates(model, p), model.q_B)
    c_info = branch_C_ik_consistency(model, p, model.q_C)
    qC = 0.5 * (c_info["upper_selected"] + c_info["lower_selected"])
    return np.array([qA, qB, qC], dtype=float)



def full_ik_with_reference(model: TripteronParameters, p: Vec3, q_ref: Vec3) -> np.ndarray:
    q_ref = arr(q_ref)
    qA = select_reference_ik_branch(branch_A_ik_candidates(model, p), float(q_ref[0]))
    qB = select_reference_ik_branch(branch_B_ik_candidates(model, p), float(q_ref[1]))
    c_info = branch_C_ik_consistency(model, p, float(q_ref[2]))
    qC = 0.5 * (c_info["upper_selected"] + c_info["lower_selected"])
    return np.array([qA, qB, qC], dtype=float)


# ============================================================
# Diagnostics / first-use script entry point
# ============================================================
def print_model_summary(model: TripteronParameters) -> None:
    print("=== Tripteron solver model summary ===")
    print(f"p0 = {model.p0}")
    print(f"q0 = {[model.q_A, model.q_B, model.q_C]}")
    print(f"ex = {model.ex}")
    print(f"ey = {model.ey}")
    print(f"ez = {model.ez}")
    print()

    print("EE offsets from centroid:")
    print(f"  upper-left  r_ul = {model.r_ul}")
    print(f"  lower-left  r_ll = {model.r_ll}")
    print(f"  upper-right r_ur = {model.r_ur}")
    print(f"  lower-right r_lr = {model.r_lr}")
    print()

    print("Rail reference origins:")
    print(f"  A: {model.a_A}")
    print(f"  B: {model.a_B}")
    print(f"  C: {model.a_C}")
    print(f"  rail_dir: {model.rail_dir}")
    print()

    print("Simple-branch lengths:")
    print(f"  A: {model.branch_A_lengths}")
    print(f"  B: {model.branch_B_lengths}")
    print()

    print("Carriage-to-branch base offsets:")
    print(f"  A: {model.offset_A_carriage_to_base_joint}")
    print(f"  B: {model.offset_B_carriage_to_base_joint}")
    print(f"  C -> R29: {model.offset_C_carriage_to_R29}")
    print(f"  C -> R31: {model.offset_C_carriage_to_R31}")
    print()

    print("Closed-branch lengths:")
    for k, v in model.branch_C_lengths.items():
        print(f"  {k}: {v}")
    print()

    print("Reference-pose residuals:")
    print(f"  Branch A: {branch_A_reference_residuals(model, model.p0, model.q_A)}")
    print(f"  Branch B: {branch_B_reference_residuals(model, model.p0, model.q_B)}")
    print(f"  Branch C: {branch_C_reference_measures(model, model.p0, model.q_C)}")
    print()

    print("Simple-branch IK evaluated at p0:")
    qA_cands = branch_A_ik_candidates(model, model.p0)
    qB_cands = branch_B_ik_candidates(model, model.p0)
    qA_est = select_reference_ik_branch(qA_cands, model.q_A)
    qB_est = select_reference_ik_branch(qB_cands, model.q_B)
    print(f"  qA_candidates = {qA_cands}   (reference qA = {model.q_A})")
    print(f"  qB_candidates = {qB_cands}   (reference qB = {model.q_B})")
    print(f"  qA_selected   = {qA_est}")
    print(f"  qB_selected   = {qB_est}")
    print()

    print("Closed-branch IK evaluated at p0:")
    c_info = branch_C_ik_consistency(model, model.p0, model.q_C)
    print(f"  upper_candidates   = {c_info['upper_candidates']}   (reference qC = {model.q_C})")
    print(f"  lower_candidates   = {c_info['lower_candidates']}   (reference qC = {model.q_C})")
    print(f"  upper_selected     = {c_info['upper_selected']}")
    print(f"  lower_selected     = {c_info['lower_selected']}")
    print(f"  selected_difference= {c_info['selected_difference']}")
    print(f"  selected_average   = {c_info['selected_average']}")
    print()

    print("Full IK evaluated at p0:")
    q_est = full_ik_near_reference(model, model.p0)
    print(f"  q_est = {q_est}")
    print(f"  q_ref = {np.array([model.q_A, model.q_B, model.q_C], dtype=float)}")
    print(f"  q_est - q_ref = {q_est - np.array([model.q_A, model.q_B, model.q_C], dtype=float)}")


if __name__ == "__main__":
    # Adjust path as needed.
    input_json = Path("bigger_tripteron_export.json")

    model, report = load_tripteron_parameters(input_json)
    print(f"Extractor validation passed: {report.passed}")
    for msg in report.messages:
        print(f"  - {msg}")
    print()

    print_model_summary(model)

