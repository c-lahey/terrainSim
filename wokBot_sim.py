"""
kinematics_2t1r_cooking.py

Draft kinematics module for the Fig. 7 2T1R + 1T cooking robot mechanism
from Li et al., "Conceptual design and analysis of the 2T1R mechanism
for a cooking robot", Robotics and Autonomous Systems, 2011.

This implements:
    - closed-form inverse kinematics
    - all IK branches
    - preferred IK branch selection
    - numerical forward kinematics
    - numerical Jacobian dq/dp
    - simple grid workspace feasibility checks

Coordinate convention:
    Pose p = [x, y, beta]
    C1 = [x, y]
    beta is the orientation of vector C3 -> C1
    C3 = C1 - c3 * [cos(beta), sin(beta)]

The mechanism dimensions default to the paper's values, in mm.
Angles are in radians internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import math
import numpy as np


ArrayLike = Sequence[float] | np.ndarray


@dataclass
class Cooking2T1RParams:
    # Active-link lengths
    a1: float = 110.0
    a2: float = 19.0
    a3: float = 80.0

    # Passive-link lengths
    b1: float = 140.0
    b2: float = 96.0
    b3: float = 230.0

    # Moving-platform offsets
    c1: float = 0.0
    c2: float = 80.0
    c3: float = 70.0

    # Base joint locations:
    # A1 = [e1, ep1]
    # A2 = [0, 0]
    # A3 = [e3, ep3]
    e1: float = 80.0
    ep1: float = 20.0
    e3: float = 80.0
    ep3: float = 70.0


@dataclass
class IKSolution:
    q: np.ndarray
    branch: tuple[int, int, int]
    residual_norm: float
    valid: bool


class Cooking2T1RKinematics:
    """
    Kinematics model for the Fig. 7 2T1R mechanism.

    This is intentionally written to look more like a useful engineering
    kinematics wrapper than a pure symbolic transcription of the paper.
    """

    def __init__(self, params: Cooking2T1RParams | None = None):
        self.p = params or Cooking2T1RParams()

    # ------------------------------------------------------------------
    # Basic geometry
    # ------------------------------------------------------------------

    @property
    def A1(self) -> np.ndarray:
        return np.array([self.p.e1, self.p.ep1], dtype=float)

    @property
    def A2(self) -> np.ndarray:
        return np.array([0.0, 0.0], dtype=float)

    @property
    def A3(self) -> np.ndarray:
        return np.array([self.p.e3, self.p.ep3], dtype=float)

    @staticmethod
    def unit(theta: float) -> np.ndarray:
        return np.array([math.cos(theta), math.sin(theta)], dtype=float)

    def B1(self, theta1: float) -> np.ndarray:
        return self.A1 + self.p.a1 * self.unit(theta1)

    def B2(self, theta2: float) -> np.ndarray:
        return self.A2 + self.p.a2 * self.unit(theta2)

    def B3(self, theta3: float) -> np.ndarray:
        return self.A3 + self.p.a3 * self.unit(theta3)

    @staticmethod
    def C1_from_pose(pose: ArrayLike) -> np.ndarray:
        x, y, _beta = pose
        return np.array([x, y], dtype=float)

    def C3_from_pose(self, pose: ArrayLike) -> np.ndarray:
        x, y, beta = pose
        return np.array(
            [
                x - self.p.c3 * math.cos(beta),
                y - self.p.c3 * math.sin(beta),
            ],
            dtype=float,
        )

    def C2_from_C1_B1(self, C1: np.ndarray, B1: np.ndarray) -> np.ndarray:
        """
        Interpreted five-bar geometry for C2.

        The paper gives limb 2 through the five-bar A1-B1-C2-B2-A2,
        and Box I contains terms equivalent to:

            h = b1*B2 + c2*C1 - (b1 + c2)*B1

        This is consistent with:

            C2 = B1 - (c2 / b1) * (C1 - B1)

        so that:

            b1 * (B2 - C2) = b1*B2 + c2*C1 - (b1+c2)*B1

        This is the main assumption in this draft.
        """
        return B1 - (self.p.c2 / self.p.b1) * (C1 - B1)

    # ------------------------------------------------------------------
    # Generic two-link / circle intersection inverse solve
    # ------------------------------------------------------------------

    @staticmethod
    def solve_revolute_for_target(
        A: np.ndarray,
        C: np.ndarray,
        a: float,
        b: float,
        tol: float = 1e-9,
    ) -> list[tuple[float, int, float]]:
        """
        Solve ||A + a*[cos(theta), sin(theta)] - C|| = b.

        Returns:
            [(theta, branch_sign, discriminant), ...]

        branch_sign is +1 or -1 corresponding to the quadratic root.
        """
        dx = A[0] - C[0]
        dy = A[1] - C[1]

        # Constraint:
        # (dx + a cos theta)^2 + (dy + a sin theta)^2 = b^2
        #
        # With t = tan(theta / 2):
        # cos = (1 - t^2)/(1 + t^2)
        # sin = 2t/(1 + t^2)
        K = dx * dx + dy * dy + a * a - b * b

        qa = K - 2.0 * a * dx
        qb = 4.0 * a * dy
        qc = K + 2.0 * a * dx

        roots: list[tuple[float, int, float]] = []

        # Handle near-linear case in t.
        if abs(qa) < tol:
            if abs(qb) < tol:
                return roots
            t = -qc / qb
            theta = 2.0 * math.atan(t)
            roots.append((Cooking2T1RKinematics.wrap_to_pi(theta), 0, 0.0))
            return roots

        disc = qb * qb - 4.0 * qa * qc
        if disc < -tol:
            return roots

        disc_clamped = max(0.0, disc)
        sqrt_disc = math.sqrt(disc_clamped)

        for sign in (+1, -1):
            t = (-qb + sign * sqrt_disc) / (2.0 * qa)
            theta = 2.0 * math.atan(t)
            roots.append((Cooking2T1RKinematics.wrap_to_pi(theta), sign, disc))

        return roots

    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def wrap_vec_to_pi(q: ArrayLike) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return (q + np.pi) % (2.0 * np.pi) - np.pi

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------

    def ik_all(self, pose: ArrayLike, tol: float = 1e-7) -> list[IKSolution]:
        """
        Return all reachable IK branches for a desired pose [x, y, beta].

        This generally returns up to 8 solutions:
            2 choices for theta1
            2 choices for theta2
            2 choices for theta3
        """
        pose = np.asarray(pose, dtype=float)
        C1 = self.C1_from_pose(pose)
        C3 = self.C3_from_pose(pose)

        theta1_roots = self.solve_revolute_for_target(
            self.A1, C1, self.p.a1, self.p.b1
        )

        theta3_roots = self.solve_revolute_for_target(
            self.A3, C3, self.p.a3, self.p.b3
        )

        sols: list[IKSolution] = []

        for theta1, s1, _disc1 in theta1_roots:
            B1 = self.B1(theta1)
            C2 = self.C2_from_C1_B1(C1, B1)

            theta2_roots = self.solve_revolute_for_target(
                self.A2, C2, self.p.a2, self.p.b2
            )

            for theta2, s2, _disc2 in theta2_roots:
                for theta3, s3, _disc3 in theta3_roots:
                    q = np.array([theta1, theta2, theta3], dtype=float)
                    r = self.constraint_residual(q, pose)
                    rnorm = float(np.linalg.norm(r))
                    sols.append(
                        IKSolution(
                            q=q,
                            branch=(s1, s2, s3),
                            residual_norm=rnorm,
                            valid=rnorm <= tol,
                        )
                    )

        sols.sort(key=lambda sol: sol.residual_norm)
        return sols

    def ik(
        self,
        pose: ArrayLike,
        q_prev: Optional[ArrayLike] = None,
        preferred_branch: Optional[tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        Return one IK solution.

        Selection order:
            1. If q_prev is supplied, choose the nearest valid solution.
            2. Else if preferred_branch is supplied, choose that branch.
            3. Else use the paper's stated work mode: +, +, -.
               Depending on the generic quadratic sign convention, this may
               need to be flipped after comparison to a real assembly.
        """
        sols = [s for s in self.ik_all(pose) if s.valid]
        if not sols:
            raise ValueError(f"Pose is unreachable or no IK branch found: {pose}")

        if q_prev is not None:
            q_prev = np.asarray(q_prev, dtype=float)

            def angle_dist(sol: IKSolution) -> float:
                dq = self.wrap_vec_to_pi(sol.q - q_prev)
                return float(np.linalg.norm(dq))

            return min(sols, key=angle_dist).q

        if preferred_branch is None:
            preferred_branch = (+1, +1, -1)

        matching = [s for s in sols if s.branch == preferred_branch]
        if matching:
            return matching[0].q

        # Fallback: best residual.
        return sols[0].q

    # ------------------------------------------------------------------
    # Constraint residuals
    # ------------------------------------------------------------------

    def constraint_residual(self, q: ArrayLike, pose: ArrayLike) -> np.ndarray:
        """
        Residual vector for the three loop-closure constraints.

        residual[i] = actual_length_i - nominal_length_i
        """
        q = np.asarray(q, dtype=float)
        pose = np.asarray(pose, dtype=float)

        theta1, theta2, theta3 = q
        C1 = self.C1_from_pose(pose)
        C3 = self.C3_from_pose(pose)

        B1 = self.B1(theta1)
        B2 = self.B2(theta2)
        B3 = self.B3(theta3)
        C2 = self.C2_from_C1_B1(C1, B1)

        return np.array(
            [
                np.linalg.norm(B1 - C1) - self.p.b1,
                np.linalg.norm(B2 - C2) - self.p.b2,
                np.linalg.norm(B3 - C3) - self.p.b3,
            ],
            dtype=float,
        )

    def squared_constraint_residual(self, q: ArrayLike, pose: ArrayLike) -> np.ndarray:
        """
        Same constraints, but squared-length form. Useful for numerical FK.
        """
        q = np.asarray(q, dtype=float)
        pose = np.asarray(pose, dtype=float)

        theta1, theta2, theta3 = q
        C1 = self.C1_from_pose(pose)
        C3 = self.C3_from_pose(pose)

        B1 = self.B1(theta1)
        B2 = self.B2(theta2)
        B3 = self.B3(theta3)
        C2 = self.C2_from_C1_B1(C1, B1)

        return np.array(
            [
                np.dot(B1 - C1, B1 - C1) - self.p.b1**2,
                np.dot(B2 - C2, B2 - C2) - self.p.b2**2,
                np.dot(B3 - C3, B3 - C3) - self.p.b3**2,
            ],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Numerical FK
    # ------------------------------------------------------------------

    def fk(
        self,
        q: ArrayLike,
        pose_guess: ArrayLike,
        tol: float = 1e-8,
        max_nfev: int = 100,
    ) -> np.ndarray:
        """
        Numerical forward kinematics.

        The paper notes that FK can have multiple solutions.
        This routine returns the solution near pose_guess.

        Requires scipy.
        """
        try:
            from scipy.optimize import least_squares
        except ImportError as exc:
            raise ImportError("fk() requires scipy: pip install scipy") from exc

        q = np.asarray(q, dtype=float)
        pose_guess = np.asarray(pose_guess, dtype=float)

        def fun(pose_var: np.ndarray) -> np.ndarray:
            # Scale residuals to reduce magnitude issues from mm^2.
            return self.squared_constraint_residual(q, pose_var) / 1e4

        result = least_squares(
            fun,
            pose_guess,
            xtol=tol,
            ftol=tol,
            gtol=tol,
            max_nfev=max_nfev,
        )

        if not result.success:
            raise RuntimeError(f"FK solve failed: {result.message}")

        return result.x

    # ------------------------------------------------------------------
    # Jacobian dq/dp
    # ------------------------------------------------------------------

    def numerical_jacobian_dq_dp(
        self,
        pose: ArrayLike,
        q_prev: Optional[ArrayLike] = None,
        eps_xy: float = 1e-4,
        eps_beta: float = 1e-6,
    ) -> np.ndarray:
        """
        Numerically estimate J = dq/dp.

        p = [x, y, beta]
        q = [theta1, theta2, theta3]

        Units:
            theta rows are rad
            x,y columns are rad/mm
            beta column is rad/rad
        """
        pose = np.asarray(pose, dtype=float)
        q0 = self.ik(pose, q_prev=q_prev)

        J = np.zeros((3, 3), dtype=float)
        steps = np.array([eps_xy, eps_xy, eps_beta], dtype=float)

        for j in range(3):
            dp = np.zeros(3)
            dp[j] = steps[j]

            q_plus = self.ik(pose + dp, q_prev=q0)
            q_minus = self.ik(pose - dp, q_prev=q0)

            dq = self.wrap_vec_to_pi(q_plus - q_minus)
            J[:, j] = dq / (2.0 * steps[j])

        return J

    # ------------------------------------------------------------------
    # Workspace / feasibility
    # ------------------------------------------------------------------

    def is_pose_reachable(self, pose: ArrayLike) -> bool:
        return any(sol.valid for sol in self.ik_all(pose))

    def constant_orientation_workspace_grid(
        self,
        beta: float,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        nx: int = 100,
        ny: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute a boolean grid for constant-orientation workspace.

        Returns:
            X, Y, reachable
        """
        xs = np.linspace(x_range[0], x_range[1], nx)
        ys = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(xs, ys)
        reachable = np.zeros_like(X, dtype=bool)

        for row in range(ny):
            for col in range(nx):
                pose = np.array([X[row, col], Y[row, col], beta])
                reachable[row, col] = self.is_pose_reachable(pose)

        return X, Y, reachable

    def total_orientation_workspace_grid(
        self,
        beta_range: tuple[float, float],
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        nx: int = 100,
        ny: int = 100,
        n_beta: int = 21,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Approximate TOW: points C1=(x,y) reachable for every beta in beta_range.

        The paper defines TOW as the region reachable with every orientation
        in a specified orientation interval.
        """
        betas = np.linspace(beta_range[0], beta_range[1], n_beta)

        xs = np.linspace(x_range[0], x_range[1], nx)
        ys = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(xs, ys)
        reachable_all = np.ones_like(X, dtype=bool)

        for beta in betas:
            _X, _Y, reachable_beta = self.constant_orientation_workspace_grid(
                beta=beta,
                x_range=x_range,
                y_range=y_range,
                nx=nx,
                ny=ny,
            )
            reachable_all &= reachable_beta

        return X, Y, reachable_all


if __name__ == "__main__":
    kin = Cooking2T1RKinematics()

    # Example from the paper's prescribed workspace region:
    # C1 in x=[100, 280], y=[-100, -50], beta roughly 30-80 deg.
    pose = np.array([150.0, -75.0, math.radians(50.0)])

    sols = kin.ik_all(pose)

    print(f"Pose: {pose}")
    print(f"Found {len(sols)} IK candidates")
    for sol in sols:
        print(
            "branch=",
            sol.branch,
            "q_deg=",
            np.degrees(sol.q),
            "residual_norm=",
            sol.residual_norm,
            "valid=",
            sol.valid,
        )

    q = kin.ik(pose)
    print("\nSelected q [deg]:", np.degrees(q))

    J = kin.numerical_jacobian_dq_dp(pose, q_prev=q)
    print("\nNumerical dq/dp:")
    print(J)

    # Numerical FK check from multiple initial guesses.
    # This tests whether the same actuator angles q can converge to
    # multiple valid platform poses, as expected for this mechanism.
    try:
        print("\nFK poses from multiple initial guesses:")

        for guess in [
            [100.0, -100.0, math.radians(30.0)],
            [100.0, -50.0, math.radians(80.0)],
            [280.0, -100.0, math.radians(80.0)],
            [280.0, -50.0, math.radians(30.0)],
            [150.0, -75.0, math.radians(50.0)],
        ]:
            guess = np.array(guess, dtype=float)

            try:
                pose_fk = kin.fk(q, pose_guess=guess, max_nfev=1000)

                print(
                    "guess =",
                    guess,
                    "-> FK pose =",
                    pose_fk,
                    "beta_deg =",
                    math.degrees(pose_fk[2]),
                )

            except RuntimeError as err:
                print("guess =", guess, "-> FK failed:", err)

    except ImportError:
        print("\nInstall scipy to run FK.")