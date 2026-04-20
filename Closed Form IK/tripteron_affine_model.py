from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from tripteron_ik import TripteronIKMovingPlane, numerical_jacobian_q_wrt_p


ArrayLike = np.ndarray | list[float] | tuple[float, float, float]


@dataclass
class AffineIKResult:
    ee_center: np.ndarray
    actuators: np.ndarray
    success: bool
    metadata: dict


@dataclass
class ForceMapResult:
    ee_force: np.ndarray
    actuator_forces: np.ndarray
    reconstruction_residual: np.ndarray
    used_least_squares: bool


class TripteronAffineModel:
    """
    Production-facing affine model for the moving-plane tripteron IK.

    The verified local behavior around the reference pose suggests an affine map

        q = A p + b

    where:
        p = EE centroid position
        q = prismatic actuator coordinates

    This class builds A and b from the moving-plane model and exposes:
    - fast IK evaluation
    - exact constant Jacobian access
    - actuator-force mapping via virtual work
    """

    def __init__(self, input_json: str | Path, jacobian_step: float = 1e-4):
        self.input_json = Path(input_json)
        self.base_model = TripteronIKMovingPlane(self.input_json)

        self.p0 = self.base_model.model.p0.copy()
        self.q0 = self.base_model.model.q0.copy()

        self.A = numerical_jacobian_q_wrt_p(self.base_model, self.p0, h=jacobian_step)
        self.b = self.q0 - self.A @ self.p0

    @staticmethod
    def _arr(x: ArrayLike) -> np.ndarray:
        return np.asarray(x, dtype=float).reshape(3)

    def ik(self, p: ArrayLike) -> AffineIKResult:
        p = self._arr(p)
        q = self.A @ p + self.b
        return AffineIKResult(
            ee_center=p,
            actuators=q,
            success=True,
            metadata={
                "A": self.A.copy(),
                "b": self.b.copy(),
            },
        )

    def jacobian(self) -> np.ndarray:
        return self.A.copy()

    def forward_from_reference_delta(self, dp: ArrayLike) -> np.ndarray:
        dp = self._arr(dp)
        return self.q0 + self.A @ dp

    def actuator_forces_from_ee_force(self, ee_force: ArrayLike) -> ForceMapResult:
        F = self._arr(ee_force)
        JT = self.A.T

        used_least_squares = False
        try:
            f_act = np.linalg.solve(JT, F)
        except np.linalg.LinAlgError:
            f_act, *_ = np.linalg.lstsq(JT, F, rcond=None)
            used_least_squares = True

        residual = JT @ f_act - F
        return ForceMapResult(
            ee_force=F,
            actuator_forces=f_act,
            reconstruction_residual=residual,
            used_least_squares=used_least_squares,
        )

    def ee_force_from_actuator_forces(self, actuator_forces: ArrayLike) -> np.ndarray:
        f_act = self._arr(actuator_forces)
        return self.A.T @ f_act

    def compare_against_moving_plane(self, p: ArrayLike) -> dict:
        p = self._arr(p)
        q_aff = self.ik(p).actuators
        q_mp = self.base_model.inverse(p).actuators
        return {
            "p": p,
            "q_affine": q_aff,
            "q_moving_plane": q_mp,
            "error": q_aff - q_mp,
        }


if __name__ == "__main__":
    model = TripteronAffineModel("fusion_kinematics_rich_export.json")

    print("Reference pose:")
    print("  p0 =", model.p0)
    print("  q0 =", model.q0)
    print()

    print("Affine model:")
    print("  A =")
    print(model.A)
    print("  b =", model.b)
    print()

    print("Reference reconstruction:")
    q_ref_rebuilt = model.ik(model.p0).actuators
    print("  q(p0) =", q_ref_rebuilt)
    print("  error =", q_ref_rebuilt - model.q0)
    print()

    tests = [
        np.array([+1.0, 0.0, 0.0]),
        np.array([0.0, +1.0, 0.0]),
        np.array([0.0, 0.0, +1.0]),
        np.array([+1.0, +1.0, +1.0]),
    ]

    print("Affine vs moving-plane comparison:")
    for dp in tests:
        p = model.p0 + dp
        rep = model.compare_against_moving_plane(p)
        print(f"\n  dp = {dp}")
        print("    q_affine      =", rep["q_affine"])
        print("    q_movingplane =", rep["q_moving_plane"])
        print("    error         =", rep["error"])

    print("\nForce mapping demo:")
    F = np.array([10.0, 0.0, 0.0])
    fmap = model.actuator_forces_from_ee_force(F)
    print("  EE force              =", fmap.ee_force)
    print("  actuator forces       =", fmap.actuator_forces)
    print("  reconstruction resid. =", fmap.reconstruction_residual)
    print("  used least squares    =", fmap.used_least_squares)
