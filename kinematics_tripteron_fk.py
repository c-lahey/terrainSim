from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares


class TripteronKinematicsError(Exception):
    pass


def arr(x) -> np.ndarray:
    return np.array(x, dtype=float)


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Near-zero vector")
    return v / n


@dataclass
class Plane:
    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    normal: np.ndarray


@dataclass
class OuterLeg:
    slider_name: str
    slider_ref: np.ndarray
    slider_value_ref: float
    base_ref: np.ndarray
    elbow_ref: np.ndarray
    ee_ref: np.ndarray
    ee_offset: np.ndarray
    L1: float
    L2: float
    plane: Plane


@dataclass
class ClosedLeg:
    slider_name: str
    slider_ref: np.ndarray
    slider_value_ref: float

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
    plane: Plane


class TripteronFK:
    """
    Forward-kinematics-first constrained Tripteron model.

    Inputs:
        sA, sB, sC  -> prismatic actuator positions

    Outputs:
        geometry dict containing:
            - ee_center
            - actuator values
            - EE attachment points
            - all leg points
    """

    def __init__(self, model_path: str | Path = "tripteron_extracted_model.json"):
        with Path(model_path).open("r", encoding="utf-8") as f:
            self.model: dict[str, Any] = json.load(f)

        self.global_x = normalize(arr(self.model["global"]["slider_direction_world"]))

        ee_pts = [arr(p) for p in self.model["ee_attachment_points_world"]]
        self.ee_center_ref = np.mean(np.vstack(ee_pts), axis=0)

        self.legA = self._build_leg_A()
        self.legB = self._build_leg_B()
        self.legC = self._build_leg_C()

        # Ground anchors for drawing
        self.ground_A = np.array([12.7, 0.0, 0.0], dtype=float)
        self.ground_B = np.array([12.7, 0.0, 12.7], dtype=float)
        self.ground_C = np.array([12.7, 0.0, 12.7], dtype=float)

        # State for continuation
        self._last_state = np.hstack([
            self.ee_center_ref,                                  # 0:3
            self._to2d(self.legA.elbow_ref, self.legA.plane),    # 3:5
            self._to2d(self.legB.elbow_ref, self.legB.plane),    # 5:7
            self._to2d(self.legC.elbow1_ref, self.legC.plane),   # 7:9
            self._to2d(self.legC.elbow2_ref, self.legC.plane),   # 9:11
        ])

    # ------------------------------------------------------------------
    # Build model from JSON
    # ------------------------------------------------------------------
    def _plane(self, leg_key: str) -> Plane:
        leg = self.model["legs"][leg_key]
        return Plane(
            origin=arr(leg["plane_origin_world"]),
            u=arr(leg["plane_u_world"]),
            v=arr(leg["plane_v_world"]),
            normal=arr(leg["plane_normal_world"]),
        )

    def _slider(self, name: str) -> dict[str, Any]:
        for s in self.model["global"]["sliders"]:
            if s["name"] == name:
                return s
        raise KeyError(name)

    def _joint(self, leg_key: str, joint_name: str) -> dict[str, Any]:
        for j in self.model["legs"][leg_key]["joints"]:
            if j["name"] == joint_name:
                return j
        raise KeyError((leg_key, joint_name))

    def _build_leg_A(self) -> OuterLeg:
        s = self._slider("Slider 8")
        plane = self._plane("leg_A")

        base = arr(self._joint("leg_A", "Revolute 9")["center_world"])
        elbow = arr(self._joint("leg_A", "Revolute 10")["center_world"])
        ee = arr(self._joint("leg_A", "Revolute 12")["center_world"])

        return OuterLeg(
            slider_name="Slider 8",
            slider_ref=arr(s["point_world"]),
            slider_value_ref=float(s["slide_value"]),
            base_ref=base,
            elbow_ref=elbow,
            ee_ref=ee,
            ee_offset=ee - self.ee_center_ref,
            L1=norm(elbow - base),
            L2=norm(ee - elbow),
            plane=plane,
        )

    def _build_leg_B(self) -> OuterLeg:
        s = self._slider("Slider 21")
        plane = self._plane("leg_B")

        base = arr(self._joint("leg_B", "Revolute 22")["center_world"])
        elbow = arr(self._joint("leg_B", "Revolute 2")["center_world"])
        ee = arr(self._joint("leg_B", "Revolute 24")["center_world"])

        return OuterLeg(
            slider_name="Slider 21",
            slider_ref=arr(s["point_world"]),
            slider_value_ref=float(s["slide_value"]),
            base_ref=base,
            elbow_ref=elbow,
            ee_ref=ee,
            ee_offset=ee - self.ee_center_ref,
            L1=norm(elbow - base),
            L2=norm(ee - elbow),
            plane=plane,
        )

    def _build_leg_C(self) -> ClosedLeg:
        s = self._slider("Slider 14")
        plane = self._plane("leg_C")

        base1 = arr(self._joint("leg_C", "Revolute 29")["center_world"])
        elbow1 = arr(self._joint("leg_C", "Revolute 16")["center_world"])
        ee1 = arr(self._joint("leg_C", "Revolute 30")["center_world"])

        base2 = arr(self._joint("leg_C", "Revolute 31")["center_world"])
        elbow2 = arr(self._joint("leg_C", "Revolute 19")["center_world"])
        ee2 = arr(self._joint("leg_C", "Revolute 32")["center_world"])

        return ClosedLeg(
            slider_name="Slider 14",
            slider_ref=arr(s["point_world"]),
            slider_value_ref=float(s["slide_value"]),

            base1_ref=base1,
            elbow1_ref=elbow1,
            ee1_ref=ee1,
            ee1_offset=ee1 - self.ee_center_ref,

            base2_ref=base2,
            elbow2_ref=elbow2,
            ee2_ref=ee2,
            ee2_offset=ee2 - self.ee_center_ref,

            L1a=norm(elbow1 - base1),
            L2a=norm(ee1 - elbow1),
            L1b=norm(elbow2 - base2),
            L2b=norm(ee2 - elbow2),
            plane=plane,
        )

    # ------------------------------------------------------------------
    # Plane transforms
    # ------------------------------------------------------------------
    @staticmethod
    def _to2d(p_world: np.ndarray, plane: Plane) -> np.ndarray:
        r = p_world - plane.origin
        return np.array([np.dot(r, plane.u), np.dot(r, plane.v)])

    @staticmethod
    def _to3d(p_2d: np.ndarray, plane: Plane) -> np.ndarray:
        return plane.origin + p_2d[0] * plane.u + p_2d[1] * plane.v

    # ------------------------------------------------------------------
    # State packing
    # ------------------------------------------------------------------
    @staticmethod
    def _unpack_state(x: np.ndarray):
        ee = x[0:3]
        Ael = x[3:5]
        Bel = x[5:7]
        C1el = x[7:9]
        C2el = x[9:11]
        return ee, Ael, Bel, C1el, C2el

    # ------------------------------------------------------------------
    # Forward solve residuals
    # ------------------------------------------------------------------
    def _residuals_forward(self, x: np.ndarray, sA: float, sB: float, sC: float) -> np.ndarray:
        ee, Ael2, Bel2, C1el2, C2el2 = self._unpack_state(x)

        # Moving carriage points
        A_slider = self.legA.slider_ref + (sA - self.legA.slider_value_ref) * self.global_x
        B_slider = self.legB.slider_ref + (sB - self.legB.slider_value_ref) * self.global_x
        C_slider = self.legC.slider_ref + (sC - self.legC.slider_value_ref) * self.global_x

        # Base pivots ride with carriage
        A_base = self.legA.base_ref + (sA - self.legA.slider_value_ref) * self.global_x
        B_base = self.legB.base_ref + (sB - self.legB.slider_value_ref) * self.global_x

        # Closed-chain carriage bracket: both pivots move together with one actuator
        C_base1 = self.legC.base1_ref + (sC - self.legC.slider_value_ref) * self.global_x
        C_base2 = self.legC.base2_ref + (sC - self.legC.slider_value_ref) * self.global_x

        # EE bracket points
        A_ee = ee + self.legA.ee_offset
        B_ee = ee + self.legB.ee_offset
        C_ee1 = ee + self.legC.ee1_offset
        C_ee2 = ee + self.legC.ee2_offset

        # Elbows
        A_el = self._to3d(Ael2, self.legA.plane)
        B_el = self._to3d(Bel2, self.legB.plane)
        C1_el = self._to3d(C1el2, self.legC.plane)
        C2_el = self._to3d(C2el2, self.legC.plane)

        r = []

        # ---- Leg A rigid 2R constraints
        r.append(norm(A_el - A_base) - self.legA.L1)
        r.append(norm(A_ee - A_el) - self.legA.L2)

        # ---- Leg B rigid 2R constraints
        r.append(norm(B_el - B_base) - self.legB.L1)
        r.append(norm(B_ee - B_el) - self.legB.L2)

        # ---- Leg C upper branch
        r.append(norm(C1_el - C_base1) - self.legC.L1a)
        r.append(norm(C_ee1 - C1_el) - self.legC.L2a)

        # ---- Leg C lower branch
        r.append(norm(C2_el - C_base2) - self.legC.L1b)
        r.append(norm(C_ee2 - C2_el) - self.legC.L2b)

        # ---- Mild branch regularization toward CAD reference
        Aref = self._to2d(self.legA.elbow_ref, self.legA.plane)
        Bref = self._to2d(self.legB.elbow_ref, self.legB.plane)
        C1ref = self._to2d(self.legC.elbow1_ref, self.legC.plane)
        C2ref = self._to2d(self.legC.elbow2_ref, self.legC.plane)

        r.extend(1e-3 * (Ael2 - Aref))
        r.extend(1e-3 * (Bel2 - Bref))
        r.extend(1e-3 * (C1el2 - C1ref))
        r.extend(1e-3 * (C2el2 - C2ref))

        # ---- Keep EE near previous/reference for continuation stability only
        r.extend(1e-5 * (ee - self.ee_center_ref))

        return np.array(r, dtype=float)

    # ------------------------------------------------------------------
    # Geometry build from solved state
    # ------------------------------------------------------------------
    def _geometry_from_state(self, x: np.ndarray, sA: float, sB: float, sC: float) -> dict[str, Any]:
        ee, Ael2, Bel2, C1el2, C2el2 = self._unpack_state(x)

        A_slider = self.legA.slider_ref + (sA - self.legA.slider_value_ref) * self.global_x
        B_slider = self.legB.slider_ref + (sB - self.legB.slider_value_ref) * self.global_x
        C_slider = self.legC.slider_ref + (sC - self.legC.slider_value_ref) * self.global_x

        A_base = self.legA.base_ref + (sA - self.legA.slider_value_ref) * self.global_x
        B_base = self.legB.base_ref + (sB - self.legB.slider_value_ref) * self.global_x
        C_base1 = self.legC.base1_ref + (sC - self.legC.slider_value_ref) * self.global_x
        C_base2 = self.legC.base2_ref + (sC - self.legC.slider_value_ref) * self.global_x

        A_ee = ee + self.legA.ee_offset
        B_ee = ee + self.legB.ee_offset
        C_ee1 = ee + self.legC.ee1_offset
        C_ee2 = ee + self.legC.ee2_offset

        A_el = self._to3d(Ael2, self.legA.plane)
        B_el = self._to3d(Bel2, self.legB.plane)
        C1_el = self._to3d(C1el2, self.legC.plane)
        C2_el = self._to3d(C2el2, self.legC.plane)

        return {
            "actuators": np.array([sA, sB, sC], dtype=float),
            "ee_center": np.array(ee, dtype=float),
            "ee_points": np.vstack([A_ee, B_ee, C_ee1, C_ee2]),

            "ground_points": np.vstack([self.ground_A, self.ground_B, self.ground_C]),

            "A": {
                "slider": A_slider,
                "base": A_base,
                "elbow": A_el,
                "ee": A_ee,
            },
            "B": {
                "slider": B_slider,
                "base": B_base,
                "elbow": B_el,
                "ee": B_ee,
            },
            "C": {
                "slider": C_slider,
                "base1": C_base1,
                "elbow1": C1_el,
                "ee1": C_ee1,
                "base2": C_base2,
                "elbow2": C2_el,
                "ee2": C_ee2,
            },
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(self, sA: float, sB: float, sC: float) -> dict[str, Any]:
        """
        Actuator-driven forward solve.
        """
        x0 = self._last_state.copy()

        sol = least_squares(
            lambda x: self._residuals_forward(x, sA, sB, sC),
            x0,
            method="trf",
            max_nfev=1000,
        )

        if not sol.success:
            raise TripteronKinematicsError(f"Forward solve failed: {sol.message}")

        resnorm = np.linalg.norm(sol.fun)
        if resnorm > 1e-1:
            raise TripteronKinematicsError(f"Forward residual too large: {resnorm:.3e}")

        self._last_state = sol.x.copy()
        return self._geometry_from_state(sol.x, sA, sB, sC)

    def reference_geometry(self) -> dict[str, Any]:
        """
        Convenience helper: return geometry at CAD reference actuator values.
        """
        return self.forward(
            self.legA.slider_value_ref,
            self.legB.slider_value_ref,
            self.legC.slider_value_ref,
        )