"""
fig3b_2t1r_validation.py

Validation scaffold for a Fig. 3(b)-style planar 2T1R mechanism.

Model:
    Desired platform pose:
        pose = [x, y, theta]

    Platform endpoints:
        CL = center - 0.5 * Lp * [cos(theta), sin(theta)]
        CR = center + 0.5 * Lp * [cos(theta), sin(theta)]

    Left five-bar side:
        A1 -> B1 -> CL
        A2 -> B2 -> CL

    Right orientation side:
        A3 -> B3 -> CR

    Actuated joints:
        q = [theta1, theta2, theta3]
        theta1 at A1
        theta2 at A2
        theta3 at A3

Units:
    Default dimensions are in inches, based approximately on your sketch.
    You can scale to mm later by multiplying all lengths by 25.4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math
import itertools

import numpy as np


ArrayLike = Sequence[float] | np.ndarray


@dataclass
class Fig3BParams:
    """
    Geometry parameters.

    A1, A2, A3 are ground pivots.

    Link lengths:
        l1a: A1 -> B1
        l1b: B1 -> CL

        l2a: A2 -> B2
        l2b: B2 -> CL

        l3a: A3 -> B3
        l3b: B3 -> CR

        platform_length: CL -> CR
    """

    # Ground pivots from your sketch:
    # left at x=0, middle at x=8, right at x=20.
    A1: tuple[float, float] = (0.0, 0.0)
    A2: tuple[float, float] = (8.0, 0.0)
    A3: tuple[float, float] = (20.0, 0.0)

    # Approximate sketch lengths.
    l1a: float = 10.0
    l1b: float = 14.0

    l2a: float = 10.0
    l2b: float = 12.0

    l3a: float = 12.0
    l3b: float = 12.0

    platform_length: float = 15.236


@dataclass
class IKSolution:
    q: np.ndarray
    branches: tuple[int, int, int]
    residuals: np.ndarray

    @property
    def residual_norm(self) -> float:
        return float(np.linalg.norm(self.residuals))

    @property
    def valid(self) -> bool:
        return self.residual_norm < 1e-8


class Fig3B2T1R:
    def __init__(self, params: Fig3BParams | None = None):
        self.params = params or Fig3BParams()

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def unit(theta: float) -> np.ndarray:
        return np.array([math.cos(theta), math.sin(theta)], dtype=float)

    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def wrap_vec_to_pi(q: ArrayLike) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return (q + np.pi) % (2.0 * np.pi) - np.pi

    @property
    def A1(self) -> np.ndarray:
        return np.array(self.params.A1, dtype=float)

    @property
    def A2(self) -> np.ndarray:
        return np.array(self.params.A2, dtype=float)

    @property
    def A3(self) -> np.ndarray:
        return np.array(self.params.A3, dtype=float)

    # ------------------------------------------------------------------
    # Platform geometry
    # ------------------------------------------------------------------

    def actuator_torques_for_wrench(
        self,
        pose: ArrayLike,
        wrench: ArrayLike,
        q_prev: ArrayLike | None = None,
    ) -> np.ndarray:
        """
        Compute actuator torques required to hold static equilibrium
        against a platform wrench.

        pose:
            [x, y, theta]

        wrench:
            [Fx, Fy, Mtheta]

        Returns:
            tau = [tau1, tau2, tau3]

        Sign convention:
            tau is the generalized actuator torque that balances the given
            end-effector wrench according to virtual work.

        Important:
            numerical_jacobian_dq_dp returns J_qp = dq/dp.
            Virtual work gives:
                tau^T dq = wrench^T dp
                dq = J_qp dp
                wrench = J_qp.T tau
            Therefore:
                tau = solve(J_qp.T, wrench)
        """
        pose = np.asarray(pose, dtype=float)
        wrench = np.asarray(wrench, dtype=float)

        if wrench.shape != (3,):
            raise ValueError("wrench must be [Fx, Fy, Mtheta]")

        q = self.ik(pose, q_prev=q_prev)
        J_qp = self.numerical_jacobian_dq_dp(pose, q_prev=q)

        return np.linalg.solve(J_qp.T, wrench)

    def platform_points(self, pose: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """
        Return left and right platform attachment points CL, CR.
        """
        x, y, theta = np.asarray(pose, dtype=float)
        center = np.array([x, y], dtype=float)
        axis = self.unit(theta)

        half = 0.5 * self.params.platform_length
        CL = center - half * axis
        CR = center + half * axis

        return CL, CR

    # ------------------------------------------------------------------
    # 2-link IK
    # ------------------------------------------------------------------

    def two_link_ik(
        self,
        A: ArrayLike,
        target: ArrayLike,
        l_a: float,
        l_b: float,
        tol: float = 1e-9,
    ) -> list[tuple[float, int]]:
        """
        Solve for actuator angle q of a two-link chain:

            A -> B -> target

        where:
            B = A + l_a * [cos(q), sin(q)]
            ||target - B|| = l_b

        Returns:
            [(q_elbow_up_or_down, branch_sign), ...]

        branch_sign:
            +1 or -1; use this only as a branch label.
        """
        A = np.asarray(A, dtype=float)
        target = np.asarray(target, dtype=float)

        d = target - A
        r = float(np.linalg.norm(d))

        if r < tol:
            return []

        # Reachability test
        if r > l_a + l_b + tol:
            return []

        if r < abs(l_a - l_b) - tol:
            return []

        phi = math.atan2(d[1], d[0])

        # Law of cosines:
        # angle between vector A->target and active link A->B
        cos_alpha = (l_a**2 + r**2 - l_b**2) / (2.0 * l_a * r)
        cos_alpha = float(np.clip(cos_alpha, -1.0, 1.0))

        alpha = math.acos(cos_alpha)

        q_plus = self.wrap_to_pi(phi + alpha)
        q_minus = self.wrap_to_pi(phi - alpha)

        # If tangent, avoid duplicate roots.
        if abs(alpha) < 1e-8:
            return [(q_plus, 0)]

        return [(q_plus, +1), (q_minus, -1)]

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------

    def ik_all(self, pose: ArrayLike) -> list[IKSolution]:
        """
        Return all IK branch combinations for pose = [x, y, theta].

        There can be up to 8 branches:
            2 for limb 1
            2 for limb 2
            2 for limb 3
        """
        pose = np.asarray(pose, dtype=float)
        p = self.params

        CL, CR = self.platform_points(pose)

        roots1 = self.two_link_ik(self.A1, CL, p.l1a, p.l1b)
        roots2 = self.two_link_ik(self.A2, CL, p.l2a, p.l2b)
        roots3 = self.two_link_ik(self.A3, CR, p.l3a, p.l3b)

        sols: list[IKSolution] = []

        for (q1, s1), (q2, s2), (q3, s3) in itertools.product(
            roots1, roots2, roots3
        ):
            q = np.array([q1, q2, q3], dtype=float)
            residuals = self.constraint_residuals(q, pose)
            sols.append(IKSolution(q=q, branches=(s1, s2, s3), residuals=residuals))

        # Deterministic ordering.
        sols.sort(key=lambda sol: sol.branches)
        return sols

    def ik(
        self,
        pose: ArrayLike,
        q_prev: ArrayLike | None = None,
        branch_preference: tuple[int, int, int] | None = None,
    ) -> np.ndarray:
        """
        Pick a single IK solution.

        Priority:
            1. If q_prev is provided, choose nearest joint-space solution.
            2. Else if branch_preference is provided, choose that branch.
            3. Else choose the first deterministic branch.
        """
        sols = [s for s in self.ik_all(pose) if s.valid]

        if not sols:
            raise ValueError(f"No valid IK solution for pose: {pose}")

        if q_prev is not None:
            q_prev = np.asarray(q_prev, dtype=float)

            def dist(sol: IKSolution) -> float:
                dq = self.wrap_vec_to_pi(sol.q - q_prev)
                return float(np.linalg.norm(dq))

            return min(sols, key=dist).q

        if branch_preference is not None:
            matches = [s for s in sols if s.branches == branch_preference]
            if matches:
                return matches[0].q

        return sols[0].q

    # ------------------------------------------------------------------
    # FK-ish reconstruction / residuals
    # ------------------------------------------------------------------

    def joint_points(self, q: ArrayLike, pose: ArrayLike) -> dict[str, np.ndarray]:
        """
        Return all points for drawing and validation.

        Note:
            This uses pose to define CL and CR. It is not solving FK.
        """
        q = np.asarray(q, dtype=float)
        q1, q2, q3 = q
        p = self.params

        CL, CR = self.platform_points(pose)

        B1 = self.A1 + p.l1a * self.unit(q1)
        B2 = self.A2 + p.l2a * self.unit(q2)
        B3 = self.A3 + p.l3a * self.unit(q3)

        return {
            "A1": self.A1,
            "A2": self.A2,
            "A3": self.A3,
            "B1": B1,
            "B2": B2,
            "B3": B3,
            "CL": CL,
            "CR": CR,
            "C": np.asarray(pose[:2], dtype=float),
        }

    def constraint_residuals(self, q: ArrayLike, pose: ArrayLike) -> np.ndarray:
        """
        Residuals for the three passive links:

            ||B1 - CL|| - l1b
            ||B2 - CL|| - l2b
            ||B3 - CR|| - l3b
        """
        q = np.asarray(q, dtype=float)
        p = self.params

        pts = self.joint_points(q, pose)

        r1 = np.linalg.norm(pts["B1"] - pts["CL"]) - p.l1b
        r2 = np.linalg.norm(pts["B2"] - pts["CL"]) - p.l2b
        r3 = np.linalg.norm(pts["B3"] - pts["CR"]) - p.l3b

        return np.array([r1, r2, r3], dtype=float)

    def numerical_jacobian_dq_dp(
        self,
        pose: ArrayLike,
        q_prev: ArrayLike | None = None,
        eps_xy: float = 1e-5,
        eps_theta: float = 1e-6,
    ) -> np.ndarray:
        """
        Numerical J = dq / d[x, y, theta].

        Useful for checking conditioning.
        """
        pose = np.asarray(pose, dtype=float)
        q0 = self.ik(pose, q_prev=q_prev)

        J = np.zeros((3, 3), dtype=float)
        steps = np.array([eps_xy, eps_xy, eps_theta], dtype=float)

        for j in range(3):
            dp = np.zeros(3)
            dp[j] = steps[j]

            q_plus = self.ik(pose + dp, q_prev=q0)
            q_minus = self.ik(pose - dp, q_prev=q0)

            dq = self.wrap_vec_to_pi(q_plus - q_minus)
            J[:, j] = dq / (2.0 * steps[j])

        return J


def validate_one_pose():
    # Pick a test pose that should be reachable with the approximate sketch dimensions.
    # Since your sketch is in inch-like units, this is also in those units.
    kin = Fig3B2T1R()

    pose = np.array([10.0, 8.0, math.radians(5.0)], dtype=float)

    print("Pose [x, y, theta_rad]:", pose)
    print("Pose theta [deg]:", math.degrees(pose[2]))

    CL, CR = kin.platform_points(pose)
    print("CL:", CL)
    print("CR:", CR)
    print("Platform length check:", np.linalg.norm(CR - CL))

    sols = kin.ik_all(pose)

    print(f"\nFound {len(sols)} IK candidate(s)\n")

    for i, sol in enumerate(sols):
        pts = kin.joint_points(sol.q, pose)

        print(f"solution {i}")
        print("  branches:", sol.branches)
        print("  q [deg]:", np.degrees(sol.q))
        print("  residuals:", sol.residuals)
        print("  residual_norm:", sol.residual_norm)

        print("  B1:", pts["B1"])
        print("  B2:", pts["B2"])
        print("  B3:", pts["B3"])

        print(
            "  link checks:",
            "B1-CL =", np.linalg.norm(pts["B1"] - pts["CL"]),
            "B2-CL =", np.linalg.norm(pts["B2"] - pts["CL"]),
            "B3-CR =", np.linalg.norm(pts["B3"] - pts["CR"]),
        )
        print()

    if sols:
        q = kin.ik(pose)
        J = kin.numerical_jacobian_dq_dp(pose, q_prev=q)

        print("Selected q [deg]:", np.degrees(q))
        print("Numerical dq/dp:")
        print(J)
        print("cond(J):", np.linalg.cond(J))


if __name__ == "__main__":
    validate_one_pose()