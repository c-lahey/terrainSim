from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from tripteron_kinematics import (
    TripteronMovingPlaneKinematics,
    load_moving_plane_kinematics,
    solve_simple_branch,
    solve_closed_branch,
)


ArrayLike = np.ndarray | list[float] | tuple[float, float, float]


@dataclass
class IKResult:
    ee_center: np.ndarray
    actuators: np.ndarray
    success: bool
    metadata: dict


class TripteronIKMovingPlane:
    """
    Lean IK interface built on the moving-plane branch model.

    This is the first full input/output kinematic layer that does not depend on
    freezing elbow joints in the world frame. Elbows remain optional internal
    variables and are returned only in metadata when helpful.
    """

    def __init__(self, input_json: str | Path):
        self.input_json = Path(input_json)
        self.model: TripteronMovingPlaneKinematics = load_moving_plane_kinematics(self.input_json)
        self.q_ref = self.model.q0.copy()
        self._last_q = self.q_ref.copy()

    @staticmethod
    def _arr(x: ArrayLike) -> np.ndarray:
        return np.asarray(x, dtype=float).reshape(3)

    def inverse(self, p: ArrayLike) -> IKResult:
        p = self._arr(p)

        res_A = solve_simple_branch(self.model.branch_A, p)
        res_B = solve_simple_branch(self.model.branch_B, p)
        res_C = solve_closed_branch(self.model.branch_C, p)

        q = np.array([res_A.q, res_B.q, res_C.q], dtype=float)
        self._last_q = q.copy()

        return IKResult(
            ee_center=p,
            actuators=q,
            success=True,
            metadata={
                "branch_A": {
                    "closure": res_A.closure,
                    "base_joint": res_A.base_joint.copy(),
                    "ee_joint": res_A.ee_joint.copy(),
                    "elbow_candidates": res_A.elbow_candidates,
                },
                "branch_B": {
                    "closure": res_B.closure,
                    "base_joint": res_B.base_joint.copy(),
                    "ee_joint": res_B.ee_joint.copy(),
                    "elbow_candidates": res_B.elbow_candidates,
                },
                "branch_C": {
                    "closure": res_C.closure,
                    "upper_base_joint": res_C.upper_base_joint.copy(),
                    "lower_base_joint": res_C.lower_base_joint.copy(),
                    "upper_ee_joint": res_C.upper_ee_joint.copy(),
                    "lower_ee_joint": res_C.lower_ee_joint.copy(),
                    "elbow_candidates": res_C.elbow_candidates,
                },
            },
        )

    def inverse_near_reference(self, p: ArrayLike) -> IKResult:
        # For the moving-plane model there is currently a direct branchwise solve,
        # so near-reference and generic inverse are the same API.
        return self.inverse(p)

    def reset_reference(self, q_ref: Optional[ArrayLike] = None) -> None:
        if q_ref is None:
            self._last_q = self.q_ref.copy()
        else:
            self._last_q = self._arr(q_ref)


# -----------------------------------------------------------------------------
# Small numerical test helpers
# -----------------------------------------------------------------------------
def perturbation_report(ik: TripteronIKMovingPlane, dp: ArrayLike) -> dict:
    dp = np.asarray(dp, dtype=float).reshape(3)
    p_ref = ik.model.p0.copy()
    p_new = p_ref + dp

    ref = ik.inverse(p_ref)
    new = ik.inverse(p_new)

    return {
        "p_ref": p_ref,
        "p_new": p_new,
        "dp": dp,
        "q_ref": ref.actuators,
        "q_new": new.actuators,
        "dq": new.actuators - ref.actuators,
        "branch_A_closure": new.metadata["branch_A"]["closure"],
        "branch_B_closure": new.metadata["branch_B"]["closure"],
        "branch_C_closure": new.metadata["branch_C"]["closure"],
    }


def numerical_jacobian_q_wrt_p(ik: TripteronIKMovingPlane, p: ArrayLike, h: float = 1e-4) -> np.ndarray:
    p = np.asarray(p, dtype=float).reshape(3)
    J = np.zeros((3, 3), dtype=float)
    q0 = ik.inverse(p).actuators

    for i in range(3):
        dp = np.zeros(3, dtype=float)
        dp[i] = h
        q_plus = ik.inverse(p + dp).actuators
        q_minus = ik.inverse(p - dp).actuators
        J[:, i] = (q_plus - q_minus) / (2.0 * h)

    return J


def jacobian_linearity_report(ik: TripteronIKMovingPlane, p: ArrayLike, h: float = 1e-4) -> dict:
    p = np.asarray(p, dtype=float).reshape(3)
    J0 = numerical_jacobian_q_wrt_p(ik, p, h=h)

    probes = {
        "dx": np.array([1.0, 0.0, 0.0]),
        "dy": np.array([0.0, 1.0, 0.0]),
        "dz": np.array([0.0, 0.0, 1.0]),
    }

    q0 = ik.inverse(p).actuators
    results = {}
    for name, dp in probes.items():
        q_exact = ik.inverse(p + dp).actuators
        q_linear = q0 + J0 @ dp
        results[name] = {
            "dp": dp,
            "q_exact": q_exact,
            "q_linear": q_linear,
            "linearization_error": q_exact - q_linear,
        }

    return {
        "p": p,
        "J_qp": J0,
        "probes": results,
    }


if __name__ == "__main__":
    ik = TripteronIKMovingPlane("fusion_kinematics_rich_export.json")

    print("Reference pose p0:", ik.model.p0)
    print("Reference q0:", ik.model.q0)

    res0 = ik.inverse(ik.model.p0)
    print("IK at p0:")
    print("  q_est:", res0.actuators)
    print("  q_ref:", ik.model.q0)
    print("  error:", res0.actuators - ik.model.q0)

    print("Small perturbation tests:")
    tests = [
        np.array([+1.0, 0.0, 0.0]),
        np.array([0.0, +1.0, 0.0]),
        np.array([0.0, 0.0, +1.0]),
        np.array([+1.0, +1.0, +1.0]),
    ]

    for dp in tests:
        rep = perturbation_report(ik, dp)
        print(f"dp = {rep['dp']}")
        print(f"    q_new = {rep['q_new']}")
        print(f"    dq    = {rep['dq']}")
        print(f"    A closure: {rep['branch_A_closure']}")
        print(f"    B closure: {rep['branch_B_closure']}")
        print(f"    C closure: {rep['branch_C_closure']}")

    print("Numerical Jacobian at p0:")
    J = numerical_jacobian_q_wrt_p(ik, ik.model.p0, h=1e-4)
    print(J)

    print("Local linearity report about p0:")
    lin = jacobian_linearity_report(ik, ik.model.p0, h=1e-4)
    for name, info in lin["probes"].items():
        print(f"Probe {name}, dp = {info['dp']}")
        print(f"    q_exact  = {info['q_exact']}")
        print(f"    q_linear = {info['q_linear']}")
        print(f"    error    = {info['linearization_error']}")