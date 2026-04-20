from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from json_extractor_test import build_reduced_parameter_set, ReducedParameters


Vec3 = np.ndarray
Vec2 = np.ndarray


def arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def unit(x: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    n = norm(x)
    if n < tol:
        raise ValueError(f"Cannot normalize near-zero vector: {x}")
    return x / n


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def circle_circle_intersections(c0: Vec2, r0: float, c1: Vec2, r1: float, tol: float = 1e-10) -> Tuple[Vec2, Vec2] | None:
    dvec = c1 - c0
    d = norm(dvec)

    if d < tol and abs(r0 - r1) < tol:
        return None
    if d > r0 + r1 + tol:
        return None
    if d < abs(r0 - r1) - tol:
        return None
    if d < tol:
        return None

    ex = dvec / d
    a = (r0 * r0 - r1 * r1 + d * d) / (2.0 * d)
    h2 = r0 * r0 - a * a
    if h2 < -tol:
        return None
    h2 = max(h2, 0.0)
    h = np.sqrt(h2)

    p2 = c0 + a * ex
    ey = np.array([-ex[1], ex[0]], dtype=float)
    return p2 + h * ey, p2 - h * ey


@dataclass
class PlaneBasis:
    normal: Vec3
    e1: Vec3
    e2: Vec3

    def coords(self, origin: Vec3, p: Vec3) -> Vec2:
        d = arr(p) - arr(origin)
        return np.array([dot(d, self.e1), dot(d, self.e2)], dtype=float)

    def world(self, origin: Vec3, uv: Vec2) -> Vec3:
        return arr(origin) + uv[0] * self.e1 + uv[1] * self.e2


@dataclass
class SimpleBranchMovingPlane:
    name: str
    plane_basis: PlaneBasis
    a_base: Vec3
    rail_dir: Vec3
    ee_offset: Vec3
    L1: float
    L2: float
    q_ref: float
    reference_base_joint: Vec3
    reference_mid_joint: Vec3
    reference_ee_joint: Vec3

    def ee_point(self, p: Vec3) -> Vec3:
        return arr(p) + self.ee_offset

    def base_joint(self, q: float) -> Vec3:
        return self.a_base + q * self.rail_dir

    def plane_residual(self, p: Vec3, q: float) -> float:
        return dot(self.plane_basis.normal, self.base_joint(q) - self.ee_point(p))

    def q_from_plane_intersection(self, p: Vec3) -> float:
        c = self.ee_point(p)
        denom = dot(self.plane_basis.normal, self.rail_dir)
        if abs(denom) < 1e-10:
            raise ValueError(f"Branch {self.name}: slider line is parallel to moving branch plane")
        return float(dot(self.plane_basis.normal, c - self.a_base) / denom)

    def distance_base_to_ee(self, p: Vec3, q: float) -> float:
        return norm(self.ee_point(p) - self.base_joint(q))

    def closure_margin(self, p: Vec3, q: float | None = None) -> dict:
        if q is None:
            q = self.q_from_plane_intersection(p)
        d = self.distance_base_to_ee(p, q)
        return {
            "plane_residual": self.plane_residual(p, q),
            "base_to_ee": d,
            "min_reach_margin": d - abs(self.L1 - self.L2),
            "max_reach_margin": (self.L1 + self.L2) - d,
        }

    def elbow_candidates(self, p: Vec3, q: float | None = None) -> Tuple[Vec3, Vec3] | None:
        if q is None:
            q = self.q_from_plane_intersection(p)
        b = self.base_joint(q)
        c = self.ee_point(p)
        b2 = self.plane_basis.coords(c, b)
        c2 = np.array([0.0, 0.0], dtype=float)
        sols2 = circle_circle_intersections(b2, self.L1, c2, self.L2)
        if sols2 is None:
            return None
        return self.plane_basis.world(c, sols2[0]), self.plane_basis.world(c, sols2[1])


@dataclass
class ClosedBranchMovingPlane:
    name: str
    plane_basis: PlaneBasis
    a_upper: Vec3
    a_lower: Vec3
    rail_dir: Vec3
    ee_offset_upper: Vec3
    ee_offset_lower: Vec3
    L_upper_1: float
    L_upper_2: float
    L_lower_1: float
    L_lower_2: float
    q_ref: float
    reference_upper_base: Vec3
    reference_lower_base: Vec3
    reference_upper_mid: Vec3
    reference_lower_mid: Vec3
    reference_upper_ee: Vec3
    reference_lower_ee: Vec3

    def ee_upper(self, p: Vec3) -> Vec3:
        return arr(p) + self.ee_offset_upper

    def ee_lower(self, p: Vec3) -> Vec3:
        return arr(p) + self.ee_offset_lower

    def upper_base(self, q: float) -> Vec3:
        return self.a_upper + q * self.rail_dir

    def lower_base(self, q: float) -> Vec3:
        return self.a_lower + q * self.rail_dir

    def q_from_upper_plane_intersection(self, p: Vec3) -> float:
        c = self.ee_upper(p)
        denom = dot(self.plane_basis.normal, self.rail_dir)
        if abs(denom) < 1e-10:
            raise ValueError(f"Branch {self.name}: upper slider line is parallel to moving branch plane")
        return float(dot(self.plane_basis.normal, c - self.a_upper) / denom)

    def q_from_lower_plane_intersection(self, p: Vec3) -> float:
        c = self.ee_lower(p)
        denom = dot(self.plane_basis.normal, self.rail_dir)
        if abs(denom) < 1e-10:
            raise ValueError(f"Branch {self.name}: lower slider line is parallel to moving branch plane")
        return float(dot(self.plane_basis.normal, c - self.a_lower) / denom)

    def q_from_plane_intersection(self, p: Vec3) -> float:
        qu = self.q_from_upper_plane_intersection(p)
        ql = self.q_from_lower_plane_intersection(p)
        return 0.5 * (qu + ql)

    def plane_residuals(self, p: Vec3, q: float) -> dict:
        return {
            "upper": dot(self.plane_basis.normal, self.upper_base(q) - self.ee_upper(p)),
            "lower": dot(self.plane_basis.normal, self.lower_base(q) - self.ee_lower(p)),
            "q_upper_minus_lower": self.q_from_upper_plane_intersection(p) - self.q_from_lower_plane_intersection(p),
        }

    def closure_margin(self, p: Vec3, q: float | None = None) -> dict:
        if q is None:
            q = self.q_from_plane_intersection(p)
        d_upper = norm(self.ee_upper(p) - self.upper_base(q))
        d_lower = norm(self.ee_lower(p) - self.lower_base(q))
        return {
            "plane_residuals": self.plane_residuals(p, q),
            "upper_base_to_ee": d_upper,
            "lower_base_to_ee": d_lower,
            "upper_min_reach_margin": d_upper - abs(self.L_upper_1 - self.L_upper_2),
            "upper_max_reach_margin": (self.L_upper_1 + self.L_upper_2) - d_upper,
            "lower_min_reach_margin": d_lower - abs(self.L_lower_1 - self.L_lower_2),
            "lower_max_reach_margin": (self.L_lower_1 + self.L_lower_2) - d_lower,
            "ee_spacing": norm(self.ee_upper(p) - self.ee_lower(p)),
        }

    def elbow_candidates(self, p: Vec3, q: float | None = None) -> Tuple[Tuple[Vec3, Vec3] | None, Tuple[Vec3, Vec3] | None]:
        if q is None:
            q = self.q_from_plane_intersection(p)

        bu = self.upper_base(q)
        cu = self.ee_upper(p)
        bu2 = self.plane_basis.coords(cu, bu)
        cu2 = np.array([0.0, 0.0], dtype=float)
        sols_u = circle_circle_intersections(bu2, self.L_upper_1, cu2, self.L_upper_2)
        upper = None if sols_u is None else (
            self.plane_basis.world(cu, sols_u[0]),
            self.plane_basis.world(cu, sols_u[1]),
        )

        bl = self.lower_base(q)
        cl = self.ee_lower(p)
        bl2 = self.plane_basis.coords(cl, bl)
        cl2 = np.array([0.0, 0.0], dtype=float)
        sols_l = circle_circle_intersections(bl2, self.L_lower_1, cl2, self.L_lower_2)
        lower = None if sols_l is None else (
            self.plane_basis.world(cl, sols_l[0]),
            self.plane_basis.world(cl, sols_l[1]),
        )
        return upper, lower


@dataclass
class TripteronMovingPlaneKinematics:
    raw: ReducedParameters
    p0: Vec3
    q0: Vec3
    branch_A: SimpleBranchMovingPlane
    branch_B: SimpleBranchMovingPlane
    branch_C: ClosedBranchMovingPlane


@dataclass
class SimpleIKResult:
    q: float
    base_joint: Vec3
    ee_joint: Vec3
    elbow_candidates: Tuple[Vec3, Vec3] | None
    closure: dict


@dataclass
class ClosedIKResult:
    q: float
    upper_base_joint: Vec3
    lower_base_joint: Vec3
    upper_ee_joint: Vec3
    lower_ee_joint: Vec3
    elbow_candidates: Tuple[Tuple[Vec3, Vec3] | None, Tuple[Vec3, Vec3] | None]
    closure: dict


def _plane_basis_from_reference(base_joint: Vec3, mid_joint: Vec3, ee_joint: Vec3, rail_dir: Vec3) -> PlaneBasis:
    v1 = arr(mid_joint) - arr(base_joint)
    v2 = arr(ee_joint) - arr(base_joint)
    normal = unit(np.cross(v1, v2))

    e1_raw = arr(rail_dir) - dot(arr(rail_dir), normal) * normal
    if norm(e1_raw) < 1e-10:
        e1_raw = v1 - dot(v1, normal) * normal
    e1 = unit(e1_raw)
    e2 = unit(np.cross(normal, e1))
    return PlaneBasis(normal=normal, e1=e1, e2=e2)


def _build_simple_branch(params: ReducedParameters, branch_name: str, ee_joint_name: str) -> SimpleBranchMovingPlane:
    cp_map = {cp.joint_name: cp for cp in params.ee_coupling_points}

    if branch_name == "A":
        slider = params.sliders["A"]
        names = params.branch_A.chain_joint_names
        pts = params.branch_A.chain_joint_points_world
        lookup = dict(zip(names, pts))
        base = arr(lookup["Revolute 22"])
        mid = arr(lookup["Revolute 2"])
        ee = arr(params.branch_A.ee_point_world)
        L1 = float(params.branch_A.rigid_segment_lengths[0])
        L2 = float(params.branch_A.rigid_segment_lengths[1])
        q_ref = float(slider.slide_value)
    elif branch_name == "B":
        slider = params.sliders["B"]
        names = params.branch_B.chain_joint_names
        pts = params.branch_B.chain_joint_points_world
        lookup = dict(zip(names, pts))
        base = arr(lookup["Revolute 9"])
        mid = arr(lookup["Revolute 10"])
        ee = arr(params.branch_B.ee_point_world)
        L1 = float(params.branch_B.rigid_segment_lengths[0])
        L2 = float(params.branch_B.rigid_segment_lengths[1])
        q_ref = float(slider.slide_value)
    else:
        raise ValueError(f"Unsupported simple branch name: {branch_name}")

    rail_dir = arr(slider.slide_direction_world)
    a_base = base - q_ref * rail_dir
    basis = _plane_basis_from_reference(base, mid, ee, rail_dir)

    return SimpleBranchMovingPlane(
        name=branch_name,
        plane_basis=basis,
        a_base=a_base,
        rail_dir=rail_dir,
        ee_offset=arr(cp_map[ee_joint_name].offset_from_centroid),
        L1=L1,
        L2=L2,
        q_ref=q_ref,
        reference_base_joint=base,
        reference_mid_joint=mid,
        reference_ee_joint=ee,
    )


def _build_closed_branch(params: ReducedParameters) -> ClosedBranchMovingPlane:
    cp_map = {cp.joint_name: cp for cp in params.ee_coupling_points}
    slider = params.sliders["C"]
    names = params.branch_C.chain_joint_names
    pts = params.branch_C.chain_joint_points_world
    lookup = dict(zip(names, pts))

    base_u = arr(lookup["Revolute 29"])
    base_l = arr(lookup["Revolute 31"])
    mid_u = arr(lookup["Revolute 16"])
    mid_l = arr(lookup["Revolute 19"])
    ee_u = arr(params.branch_C.ee_points_world[0])
    ee_l = arr(params.branch_C.ee_points_world[1])

    rail_dir = arr(slider.slide_direction_world)
    q_ref = float(slider.slide_value)
    a_upper = base_u - q_ref * rail_dir
    a_lower = base_l - q_ref * rail_dir
    basis = _plane_basis_from_reference(base_u, mid_u, ee_u, rail_dir)

    return ClosedBranchMovingPlane(
        name="C",
        plane_basis=basis,
        a_upper=a_upper,
        a_lower=a_lower,
        rail_dir=rail_dir,
        ee_offset_upper=arr(cp_map["Revolute 30"].offset_from_centroid),
        ee_offset_lower=arr(cp_map["Revolute 32"].offset_from_centroid),
        L_upper_1=float(params.branch_C.rigid_lengths["R29_to_R16"]),
        L_upper_2=float(params.branch_C.rigid_lengths["R16_to_R30"]),
        L_lower_1=float(params.branch_C.rigid_lengths["R31_to_R19"]),
        L_lower_2=float(params.branch_C.rigid_lengths["R19_to_R32"]),
        q_ref=q_ref,
        reference_upper_base=base_u,
        reference_lower_base=base_l,
        reference_upper_mid=mid_u,
        reference_lower_mid=mid_l,
        reference_upper_ee=ee_u,
        reference_lower_ee=ee_l,
    )


def load_moving_plane_kinematics(input_json: str | Path) -> TripteronMovingPlaneKinematics:
    params, report = build_reduced_parameter_set(Path(input_json), output_json=None)
    if not report.passed:
        raise ValueError(f"Extractor validation failed: {report.messages}")

    p0 = arr(params.ee_centroid_world)
    q0 = np.array([
        params.sliders["A"].slide_value,
        params.sliders["B"].slide_value,
        params.sliders["C"].slide_value,
    ], dtype=float)

    branch_A = _build_simple_branch(params, "A", "Revolute 24")
    branch_B = _build_simple_branch(params, "B", "Revolute 12")
    branch_C = _build_closed_branch(params)

    return TripteronMovingPlaneKinematics(
        raw=params,
        p0=p0,
        q0=q0,
        branch_A=branch_A,
        branch_B=branch_B,
        branch_C=branch_C,
    )


def solve_simple_branch(branch: SimpleBranchMovingPlane, p: Vec3) -> SimpleIKResult:
    q = branch.q_from_plane_intersection(p)
    b = branch.base_joint(q)
    c = branch.ee_point(p)
    elbows = branch.elbow_candidates(p, q)
    closure = branch.closure_margin(p, q)
    return SimpleIKResult(
        q=q,
        base_joint=b,
        ee_joint=c,
        elbow_candidates=elbows,
        closure=closure,
    )


def solve_closed_branch(branch: ClosedBranchMovingPlane, p: Vec3) -> ClosedIKResult:
    q = branch.q_from_plane_intersection(p)
    bu = branch.upper_base(q)
    bl = branch.lower_base(q)
    cu = branch.ee_upper(p)
    cl = branch.ee_lower(p)
    elbows = branch.elbow_candidates(p, q)
    closure = branch.closure_margin(p, q)
    return ClosedIKResult(
        q=q,
        upper_base_joint=bu,
        lower_base_joint=bl,
        upper_ee_joint=cu,
        lower_ee_joint=cl,
        elbow_candidates=elbows,
        closure=closure,
    )


if __name__ == "__main__":
    kin = load_moving_plane_kinematics("fusion_kinematics_rich_export.json")
    print("Reference pose p0:", kin.p0)
    print("Reference q0:", kin.q0)

    for name, branch, qref in [
        ("A", kin.branch_A, kin.q0[0]),
        ("B", kin.branch_B, kin.q0[1]),
    ]:
        print(f"Branch {name}")
        print("  q_from_plane_intersection at p0:", branch.q_from_plane_intersection(kin.p0))
        print("  reference q:", qref)
        print("  closure at p0:", branch.closure_margin(kin.p0, qref))
        print("  elbow candidates at reference q:", branch.elbow_candidates(kin.p0, qref))

    print("Branch C")
    print("  q_from_upper_plane_intersection at p0:", kin.branch_C.q_from_upper_plane_intersection(kin.p0))
    print("  q_from_lower_plane_intersection at p0:", kin.branch_C.q_from_lower_plane_intersection(kin.p0))
    print("  q_from_plane_intersection at p0:", kin.branch_C.q_from_plane_intersection(kin.p0))
    print("  reference q:", kin.q0[2])
    print("  closure at p0:", kin.branch_C.closure_margin(kin.p0, kin.q0[2]))
    print("  elbow candidates at reference q:", kin.branch_C.elbow_candidates(kin.p0, kin.q0[2]))
