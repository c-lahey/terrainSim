"""
Tripteron inverse kinematics + geometry reconstruction.

This class mirrors the role of DeltaRobot in kinematics.py, but for the
CAD-derived Tripteron extracted from tripteron_extracted_model.json.

Primary use for this first phase:
- User specifies EE centre position (x, y, z)
- inverse(...) returns actuator positions
- geometry(...) returns all points needed for visualization

This is intentionally IK-driven to match the current visualizer style.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from scipy.optimize import least_squares, minimize_scalar

import numpy as np
from scipy.optimize import least_squares


class TripteronPositionError(Exception):
    """Raised when a requested EE position is outside the modeled workspace."""


def _arr(x) -> np.ndarray:
    return np.array(x, dtype=float)


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Near-zero vector cannot be normalized.")
    return v / n


@dataclass
class LegPlane:
    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    normal: np.ndarray


@dataclass
class TwoLinkLeg:
    slider_ref: np.ndarray
    slider_name: str
    base_ref: np.ndarray
    elbow_ref: np.ndarray
    ee_ref: np.ndarray
    ee_offset: np.ndarray
    L1: float
    L2: float
    plane: LegPlane
    slide_value_ref: float


@dataclass
class ClosedTwoBranchLeg:
    slider_ref: np.ndarray
    slider_name: str
    base1_ref: np.ndarray
    elbow1_ref: np.ndarray
    ee1_ref: np.ndarray
    ee1_offset: np.ndarray
    base2_ref: np.ndarray
    elbow2_ref: np.ndarray
    ee2_ref: np.ndarray
    ee2_offset: np.ndarray
    L1a: float
    L2a: float
    L1b: float
    L2b: float
    plane: LegPlane
    slide_value_ref: float


class TripteronRobot:
    """
    IK-first Tripteron model built from tripteron_extracted_model.json.

    Public API mirrors the delta robot style:
      - inverse(x, y, z) -> (sA, sB, sC)
      - geometry(x, y, z) -> dict of points for visualization
    """

    def __init__(self, model_path: str | Path = "tripteron_extracted_model.json"):
        model_path = Path(model_path)
        with model_path.open("r", encoding="utf-8") as f:
            self.model: dict[str, Any] = json.load(f)

        self.global_x = _normalize(_arr(self.model["global"]["slider_direction_world"]))

        # Initialize EE reference BEFORE building legs
        ee_pts = [_arr(p) for p in self.model["ee_attachment_points_world"]]
        self.ee_center_ref = np.mean(np.vstack(ee_pts), axis=0)

        self.legA = self._build_leg_A()
        self.legB = self._build_leg_B()
        self.legC = self._build_leg_C()

        # For continuity in repeated IK solves.
        self._last_solution = np.array([
            self.legA.slide_value_ref,
            self.legB.slide_value_ref,
            self.legC.slide_value_ref,
        ], dtype=float)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    
    def _solve_outer_leg_actuator(self, ee_center, leg, q_ref):
        """
        Solve 1D actuator displacement for a 2R outer leg using true reachability,
        not the collapsed stretched-link approximation.
        """
        ee_world = ee_center + leg.ee_offset

        def objective(s):
            base = leg.base_ref + (s - leg.slide_value_ref) * self.global_x
            err = self._leg_distance_to_range_error(base, ee_world, leg.L1, leg.L2)
            # regularize toward previous/reference solution for continuity
            reg = 1e-4 * (s - q_ref) ** 2
            return err**2 + reg

        res = minimize_scalar(
            objective,
            bounds=(leg.slide_value_ref - 80.0, leg.slide_value_ref + 80.0),
            method="bounded"
        )

        if not res.success or objective(res.x) > 1e-6:
            raise TripteronPositionError("Outer leg actuator solve failed.")

        return float(res.x)
    
    def _leg_distance_to_range_error(self, base_world, ee_world, L1, L2):
        d = _norm(ee_world - base_world)
        dmin = abs(L1 - L2)
        dmax = L1 + L2
        if d < dmin:
            return dmin - d
        if d > dmax:
            return d - dmax
        return 0.0

    def _plane_from_leg(self, leg_key: str) -> LegPlane:
        leg = self.model["legs"][leg_key]
        return LegPlane(
            origin=_arr(leg["plane_origin_world"]),
            u=_arr(leg["plane_u_world"]),
            v=_arr(leg["plane_v_world"]),
            normal=_arr(leg["plane_normal_world"]),
        )

    def _slider_by_name(self, name: str) -> dict[str, Any]:
        for s in self.model["global"]["sliders"]:
            if s["name"] == name:
                return s
        raise KeyError(f"Slider {name} not found")

    def _joint_by_name(self, leg_key: str, joint_name: str) -> dict[str, Any]:
        for j in self.model["legs"][leg_key]["joints"]:
            if j["name"] == joint_name:
                return j
        raise KeyError(f"{leg_key}: joint {joint_name} not found")

    def _build_leg_A(self) -> TwoLinkLeg:
        plane = self._plane_from_leg("leg_A")
        slider = self._slider_by_name("Slider 8")

        base = _arr(self._joint_by_name("leg_A", "Revolute 9")["center_world"])
        elbow = _arr(self._joint_by_name("leg_A", "Revolute 10")["center_world"])
        ee = _arr(self._joint_by_name("leg_A", "Revolute 12")["center_world"])

        slider_ref = _arr(slider["point_world"])
        L1 = _norm(elbow - base)
        L2 = _norm(ee - elbow)

        return TwoLinkLeg(
            slider_ref=slider_ref,
            slider_name="Slider 8",
            base_ref=base,
            elbow_ref=elbow,
            ee_ref=ee,
            ee_offset=ee - self.ee_center_ref,
            L1=L1,
            L2=L2,
            plane=plane,
            slide_value_ref=float(slider["slide_value"]),
        )

    def _build_leg_B(self) -> TwoLinkLeg:
        plane = self._plane_from_leg("leg_B")
        slider = self._slider_by_name("Slider 21")

        base = _arr(self._joint_by_name("leg_B", "Revolute 22")["center_world"])
        elbow = _arr(self._joint_by_name("leg_B", "Revolute 2")["center_world"])
        ee = _arr(self._joint_by_name("leg_B", "Revolute 24")["center_world"])

        slider_ref = _arr(slider["point_world"])
        L1 = _norm(elbow - base)
        L2 = _norm(ee - elbow)

        return TwoLinkLeg(
            slider_ref=slider_ref,
            slider_name="Slider 21",
            base_ref=base,
            elbow_ref=elbow,
            ee_ref=ee,
            ee_offset=ee - self.ee_center_ref,
            L1=L1,
            L2=L2,
            plane=plane,
            slide_value_ref=float(slider["slide_value"]),
        )

    def _build_leg_C(self) -> ClosedTwoBranchLeg:
        plane = self._plane_from_leg("leg_C")
        slider = self._slider_by_name("Slider 14")

        base1 = _arr(self._joint_by_name("leg_C", "Revolute 29")["center_world"])
        elbow1 = _arr(self._joint_by_name("leg_C", "Revolute 16")["center_world"])
        ee1 = _arr(self._joint_by_name("leg_C", "Revolute 30")["center_world"])

        base2 = _arr(self._joint_by_name("leg_C", "Revolute 31")["center_world"])
        elbow2 = _arr(self._joint_by_name("leg_C", "Revolute 19")["center_world"])
        ee2 = _arr(self._joint_by_name("leg_C", "Revolute 32")["center_world"])

        slider_ref = _arr(slider["point_world"])

        return ClosedTwoBranchLeg(
            slider_ref=slider_ref,
            slider_name="Slider 14",
            base1_ref=base1,
            elbow1_ref=elbow1,
            ee1_ref=ee1,
            ee1_offset=ee1 - self.ee_center_ref,
            base2_ref=base2,
            elbow2_ref=elbow2,
            ee2_ref=ee2,
            ee2_offset=ee2 - self.ee_center_ref,
            L1a=_norm(elbow1 - base1),
            L2a=_norm(ee1 - elbow1),
            L1b=_norm(elbow2 - base2),
            L2b=_norm(ee2 - elbow2),
            plane=plane,
            slide_value_ref=float(slider["slide_value"]),
        )

    # ------------------------------------------------------------------
    # Plane transforms
    # ------------------------------------------------------------------

    @staticmethod
    def _to2d(p_world: np.ndarray, plane: LegPlane) -> np.ndarray:
        r = p_world - plane.origin
        return np.array([np.dot(r, plane.u), np.dot(r, plane.v)])

    @staticmethod
    def _to3d(p_2d: np.ndarray, plane: LegPlane) -> np.ndarray:
        return plane.origin + p_2d[0] * plane.u + p_2d[1] * plane.v

    @staticmethod
    def _circle_intersections(B: np.ndarray, E: np.ndarray, L1: float, L2: float) -> list[np.ndarray]:
        dvec = E - B
        d = _norm(dvec)
        if d < 1e-12:
            return []
        if d > L1 + L2 + 1e-9:
            return []
        if d < abs(L1 - L2) - 1e-9:
            return []

        ex = dvec / d
        a = (L1**2 - L2**2 + d**2) / (2.0 * d)
        h2 = max(L1**2 - a**2, 0.0)
        h = np.sqrt(h2)
        Pm = B + a * ex
        ey = np.array([-ex[1], ex[0]])
        return [Pm + h * ey, Pm - h * ey]

    @staticmethod
    def _choose_closest(cands: list[np.ndarray], ref: np.ndarray) -> np.ndarray | None:
        if not cands:
            return None
        idx = int(np.argmin([_norm(c - ref) for c in cands]))
        return cands[idx]

    # ------------------------------------------------------------------
    # Inverse kinematics
    # ------------------------------------------------------------------

    def inverse(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        ee = np.array([x, y, z], dtype=float)

        # Solve outer legs independently first
        sA = self._solve_outer_leg_actuator(ee, self.legA, self._last_solution[0])
        sB = self._solve_outer_leg_actuator(ee, self.legB, self._last_solution[1])

        # Solve closed-chain leg C with least squares
        def residual_c(s):
            sC = s[0]
            C_base1 = self.legC.base1_ref + (sC - self.legC.slide_value_ref) * self.global_x
            C_base2 = self.legC.base2_ref + (sC - self.legC.slide_value_ref) * self.global_x
            C_ee1 = ee + self.legC.ee1_offset
            C_ee2 = ee + self.legC.ee2_offset

            e1 = self._leg_distance_to_range_error(C_base1, C_ee1, self.legC.L1a, self.legC.L2a)
            e2 = self._leg_distance_to_range_error(C_base2, C_ee2, self.legC.L1b, self.legC.L2b)
            reg = 1e-3 * (sC - self._last_solution[2])
            return np.array([e1, e2, reg])

        sol = least_squares(residual_c, np.array([self._last_solution[2]]), method="trf")

        if not sol.success or np.linalg.norm(sol.fun[:2]) > 1e-5:
            raise TripteronPositionError(
                f"Requested EE position ({x:.3f}, {y:.3f}, {z:.3f}) is unreachable."
            )

        sC = float(sol.x[0])

        self._last_solution = np.array([sA, sB, sC], dtype=float)
        return sA, sB, sC
    # ------------------------------------------------------------------
    # Geometry reconstruction
    # ------------------------------------------------------------------

    def geometry(self, x: float, y: float, z: float) -> dict[str, Any]:
        """
        Return all main points needed for visualization.
        """

        ee = np.array([x, y, z], dtype=float)
        sA, sB, sC = self.inverse(x, y, z)

        # Moving slider carriage points
        A_slider = self.legA.slider_ref + (sA - self.legA.slide_value_ref) * self.global_x
        B_slider = self.legB.slider_ref + (sB - self.legB.slide_value_ref) * self.global_x
        C_slider = self.legC.slider_ref + (sC - self.legC.slide_value_ref) * self.global_x

        # First revolute after slider
        A_base = self.legA.base_ref + (sA - self.legA.slide_value_ref) * self.global_x
        B_base = self.legB.base_ref + (sB - self.legB.slide_value_ref) * self.global_x
        C_base1 = self.legC.base1_ref + (sC - self.legC.slide_value_ref) * self.global_x
        C_base2 = self.legC.base2_ref + (sC - self.legC.slide_value_ref) * self.global_x

        # EE attachment points
        A_ee = ee + self.legA.ee_offset
        B_ee = ee + self.legB.ee_offset
        C_ee1 = ee + self.legC.ee1_offset
        C_ee2 = ee + self.legC.ee2_offset

        # Elbows by circle intersection inside each fitted plane
        A_B2 = self._to2d(A_base, self.legA.plane)
        A_E2 = self._to2d(A_ee, self.legA.plane)
        A_ref2 = self._to2d(self.legA.elbow_ref, self.legA.plane)
        A_elbow2 = self._choose_closest(
            self._circle_intersections(A_B2, A_E2, self.legA.L1, self.legA.L2),
            A_ref2,
        )
        if A_elbow2 is None:
            raise TripteronPositionError("Leg A elbow reconstruction failed.")
        A_elbow = self._to3d(A_elbow2, self.legA.plane)

        B_B2 = self._to2d(B_base, self.legB.plane)
        B_E2 = self._to2d(B_ee, self.legB.plane)
        B_ref2 = self._to2d(self.legB.elbow_ref, self.legB.plane)
        B_elbow2 = self._choose_closest(
            self._circle_intersections(B_B2, B_E2, self.legB.L1, self.legB.L2),
            B_ref2,
        )
        if B_elbow2 is None:
            raise TripteronPositionError("Leg B elbow reconstruction failed.")
        B_elbow = self._to3d(B_elbow2, self.legB.plane)

        C_B1_2 = self._to2d(C_base1, self.legC.plane)
        C_E1_2 = self._to2d(C_ee1, self.legC.plane)
        C_ref1_2 = self._to2d(self.legC.elbow1_ref, self.legC.plane)
        C_elbow1_2 = self._choose_closest(
            self._circle_intersections(C_B1_2, C_E1_2, self.legC.L1a, self.legC.L2a),
            C_ref1_2,
        )
        if C_elbow1_2 is None:
            raise TripteronPositionError("Leg C upper branch elbow reconstruction failed.")
        C_elbow1 = self._to3d(C_elbow1_2, self.legC.plane)

        C_B2_2 = self._to2d(C_base2, self.legC.plane)
        C_E2_2 = self._to2d(C_ee2, self.legC.plane)
        C_ref2_2 = self._to2d(self.legC.elbow2_ref, self.legC.plane)
        C_elbow2_2 = self._choose_closest(
            self._circle_intersections(C_B2_2, C_E2_2, self.legC.L1b, self.legC.L2b),
            C_ref2_2,
        )
        if C_elbow2_2 is None:
            raise TripteronPositionError("Leg C lower branch elbow reconstruction failed.")
        C_elbow2 = self._to3d(C_elbow2_2, self.legC.plane)

        return {
            "actuators": np.array([sA, sB, sC], dtype=float),
            "ee_center": ee,
            "ee_points": np.vstack([A_ee, B_ee, C_ee1, C_ee2]),
            "slider_points": np.vstack([A_slider, B_slider, C_slider]),
            "A": {
                "slider": A_slider,
                "base": A_base,
                "elbow": A_elbow,
                "ee": A_ee,
            },
            "B": {
                "slider": B_slider,
                "base": B_base,
                "elbow": B_elbow,
                "ee": B_ee,
            },
            "C": {
                "slider": C_slider,
                "base1": C_base1,
                "elbow1": C_elbow1,
                "ee1": C_ee1,
                "base2": C_base2,
                "elbow2": C_elbow2,
                "ee2": C_ee2,
            },
        }