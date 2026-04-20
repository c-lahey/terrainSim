from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from kinematics_tripteron_fk import TripteronFK, TripteronKinematicsError


class TripteronIKError(Exception):
    pass


def arr(x) -> np.ndarray:
    return np.array(x, dtype=float)


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


@dataclass
class IKResult:
    actuators: np.ndarray
    ee_center: np.ndarray
    residual: float
    success: bool
    geometry: dict[str, Any]


class TripteronIKFKBacked:
    """
    Numerical IK wrapper around the working FK solver.

    This is the right next step because:
    - FK is currently the most trustworthy mechanism model we have
    - IK only searches actuator space
    - EE is compared against the FK output, not imposed directly
    """

    def __init__(self, fk: TripteronFK | None = None):
        self.fk = fk if fk is not None else TripteronFK("tripteron_extracted_model.json")

        self.s_ref = np.array([
            self.fk.legA.slider_value_ref,
            self.fk.legB.slider_value_ref,
            self.fk.legC.slider_value_ref,
        ], dtype=float)

        self._last_solution = self.s_ref.copy()

    # ------------------------------------------------------------------
    def _safe_forward(self, s: np.ndarray):
        try:
            geo = self.fk.forward(float(s[0]), float(s[1]), float(s[2]))
            ee = arr(geo["ee_center"])
            return geo, ee, True
        except TripteronKinematicsError:
            return None, None, False

    # ------------------------------------------------------------------
    def inverse(
        self,
        x: float,
        y: float,
        z: float,
        s0: np.ndarray | None = None,
        max_delta: float = 60.0,
        regularization: float = 1e-3,
        tol: float = 1e-1,
        n_restarts: int = 5,
    ) -> IKResult:
        """
        Solve actuator values for a desired EE center.

        Parameters
        ----------
        x, y, z:
            Desired EE center.
        s0:
            Initial guess. If None, uses previous solution.
        max_delta:
            Search box half-width around the initial guess.
        regularization:
            Small penalty toward initial guess for continuity.
        tol:
            Acceptable Cartesian residual norm.
        n_restarts:
            Number of local restarts in the search box.
        """
        target = np.array([x, y, z], dtype=float)

        if s0 is None:
            s0 = self._last_solution.copy()
        else:
            s0 = np.array(s0, dtype=float)

        lb = s0 - max_delta
        ub = s0 + max_delta

        seeds = [s0.copy()]

        # Add a few extra seeds to avoid getting trapped in a poor local minimum
        rng = np.random.default_rng(0)
        for _ in range(n_restarts - 1):
            alpha = rng.uniform(-1.0, 1.0, size=3)
            seeds.append(np.clip(s0 + 0.5 * max_delta * alpha, lb, ub))

        best_sol = None
        best_geo = None
        best_ee = None
        best_cost = np.inf

        def residuals(s):
            geo, ee, ok = self._safe_forward(s)
            if not ok:
                return np.array([1e3, 1e3, 1e3, 0.0, 0.0, 0.0], dtype=float)

            cart_err = ee - target
            reg_err = np.sqrt(regularization) * (s - s0)
            return np.hstack([cart_err, reg_err])

        for seed in seeds:
            sol = least_squares(
                residuals,
                x0=seed,
                bounds=(lb, ub),
                method="trf",
                max_nfev=500,
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8,
            )

            if not sol.success:
                continue

            geo, ee, ok = self._safe_forward(sol.x)
            if not ok:
                continue

            cost = norm(ee - target)
            if cost < best_cost:
                best_cost = cost
                best_sol = sol
                best_geo = geo
                best_ee = ee

        if best_sol is None or best_geo is None or best_ee is None:
            raise TripteronIKError("IK failed: no valid FK-backed solution found.")

        if best_cost > tol:
            raise TripteronIKError(
                f"IK residual too large: {best_cost:.4f} > tol={tol:.4f}"
            )

        self._last_solution = best_sol.x.copy()

        return IKResult(
            actuators=best_sol.x.copy(),
            ee_center=best_ee.copy(),
            residual=best_cost,
            success=True,
            geometry=best_geo,
        )

    # ------------------------------------------------------------------
    def inverse_from_target(self, target: np.ndarray, **kwargs) -> IKResult:
        target = np.array(target, dtype=float)
        return self.inverse(float(target[0]), float(target[1]), float(target[2]), **kwargs)

    # ------------------------------------------------------------------
    def reset(self):
        self._last_solution = self.s_ref.copy()


if __name__ == "__main__":
    ik = TripteronIKFKBacked()

    targets = [
        ik.fk.ee_center_ref.copy(),
        ik.fk.ee_center_ref + np.array([1.0, 0.0, 0.0]),
        ik.fk.ee_center_ref + np.array([0.0, 1.0, 0.0]),
        ik.fk.ee_center_ref + np.array([0.0, 0.0, 1.0]),
    ]

    for i, tgt in enumerate(targets, start=1):
        try:
            res = ik.inverse_from_target(tgt, tol=0.2, max_delta=80.0, n_restarts=7)
            print(f"\nCase {i}")
            print("  target   :", tgt)
            print("  actuators:", res.actuators)
            print("  ee out   :", res.ee_center)
            print("  residual :", res.residual)
        except TripteronIKError as e:
            print(f"\nCase {i} failed: {e}")