"""
Closed-form IK and FK for a planar 2T1R 3-DOF parallel mechanism (Fig. 3b topology).

Kinematic structure
-------------------
Three limbs from ground revolutes:

  Limb A : A1 --[lA1]--> eA --[lA2]--> P_AB
  Limb B : B1 --[lB1]--> eB --[lB2]--> P_AB   <-- shares attachment point with A
  Limb C : C1 --[lC1]--> eC --[lC2]--> P_C

Platform : rigid bar from P_AB to P_C, length L
EE       : midpoint of the platform bar
Actuated : ground revolutes at A1, B1, C1 (angles theta_A, theta_B, theta_C)

Sign convention
---------------
theta_i  : angle of limb i's proximal link from the +x axis, CCW positive (rad)
phi      : angle of the platform bar (P_AB -> P_C direction) from +x axis (rad)
elbow=+1 : proximal link rotates -gamma from base-to-target angle  (one geometric branch)
elbow=-1 : proximal link rotates +gamma from base-to-target angle  (other branch)

Default geometry (units match your sketch -- cm)
-------------------------------------------------
  A1 = (0, 0)   B1 = (8, 0)   C1 = (20, 0)
  lA = (10, 14)   lB = (10, 14)   lC = (10, 12)
  L  = 15.25
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Params:
    """Fixed geometric parameters of the mechanism."""
    A1:  tuple[float, float] = (0.0,  0.0)
    B1:  tuple[float, float] = (8.0,  0.0)
    C1:  tuple[float, float] = (20.0, 0.0)
    lA1: float = 10.0
    lA2: float = 14.0
    lB1: float = 10.0
    lB2: float = 14.0
    lC1: float = 10.0
    lC2: float = 12.0
    L:   float = 15.25  # platform bar length (P_AB to P_C)


@dataclass
class IKSolution:
    theta_A: float  # rad, ground revolute A
    theta_B: float  # rad, ground revolute B
    theta_C: float  # rad, ground revolute C
    elbow_A: int    # +1 or -1
    elbow_B: int
    elbow_C: int

    @property
    def degrees(self) -> tuple[float, float, float]:
        return (math.degrees(self.theta_A),
                math.degrees(self.theta_B),
                math.degrees(self.theta_C))


class IKError(Exception):
    pass


# ---------------------------------------------------------------------------
# Core geometry helpers
# ---------------------------------------------------------------------------

def _two_R(base: tuple[float, float], l1: float, l2: float,
           target: tuple[float, float], elbow: int) -> float:
    """
    Single 2R planar limb IK.  Returns the ground-joint angle (rad).

    Derivation:
      d      = ||target - base||
      alpha  = atan2(dy, dx)          -- direction from base to target
      gamma  = acos((l1^2 + d^2 - l2^2) / (2 l1 d))  -- law of cosines
      theta  = alpha - elbow * gamma
    """
    dx, dy = target[0] - base[0], target[1] - base[1]
    d = math.hypot(dx, dy)

    lo, hi = abs(l1 - l2), l1 + l2
    if d > hi + 1e-6:
        raise IKError(f"Unreachable: dist {d:.4f} > max reach {hi:.4f}")
    if d < lo - 1e-6:
        raise IKError(f"Unreachable: dist {d:.4f} < min reach {lo:.4f}")

    d = min(hi, max(lo, d))  # numerical clamp at limits
    alpha = math.atan2(dy, dx)
    cos_g = (l1*l1 + d*d - l2*l2) / (2.0 * l1 * d)
    gamma = math.acos(max(-1.0, min(1.0, cos_g)))
    return alpha - elbow * gamma


def _attachments(x: float, y: float, phi: float, L: float
                 ) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    EE pose -> platform attachment points.

      P_AB = EE - (L/2) * [cos phi, sin phi]
      P_C  = EE + (L/2) * [cos phi, sin phi]
    """
    h = 0.5 * L
    c, s = math.cos(phi), math.sin(phi)
    return (x - h*c, y - h*s), (x + h*c, y + h*s)


def _circle_intersect(c1: tuple[float, float], r1: float,
                      c2: tuple[float, float], r2: float,
                      side: int) -> tuple[float, float]:
    """
    Intersect two circles.  side = +1 or -1 picks one of the two solutions.
    Raises IKError if they don't intersect.
    """
    dx, dy = c2[0] - c1[0], c2[1] - c1[1]
    d = math.hypot(dx, dy)
    if d > r1 + r2 + 1e-6 or d < abs(r1 - r2) - 1e-6 or d < 1e-9:
        raise IKError("Circles do not intersect")
    a = (r1*r1 - r2*r2 + d*d) / (2.0 * d)
    h = math.sqrt(max(0.0, r1*r1 - a*a))
    mx, my = c1[0] + a*dx/d, c1[1] + a*dy/d
    return mx + side*h*dy/d, my - side*h*dx/d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve(x: float, y: float, phi: float,
          params: Params | None = None,
          elbow_A: int = -1,
          elbow_B: int =  1,
          elbow_C: int =  1) -> IKSolution:
    """
    Closed-form IK for one elbow configuration.

    Parameters
    ----------
    x, y   : EE position (midpoint of platform bar), in sketch units
    phi    : platform orientation (rad), angle of P_AB->P_C from +x axis
    elbow_* : +1 or -1, selects the two geometric branches for each limb.
              Default (-1, +1, +1) gives elbows-out for the reference pose.

    Returns IKSolution.  Raises IKError if the pose is unreachable.
    """
    p = params or Params()
    P_AB, P_C = _attachments(x, y, phi, p.L)
    tA = _two_R(p.A1, p.lA1, p.lA2, P_AB, elbow_A)
    tB = _two_R(p.B1, p.lB1, p.lB2, P_AB, elbow_B)
    tC = _two_R(p.C1, p.lC1, p.lC2, P_C,  elbow_C)
    return IKSolution(tA, tB, tC, elbow_A, elbow_B, elbow_C)


def solve_all(x: float, y: float, phi: float,
              params: Params | None = None) -> list[IKSolution]:
    """Return all reachable elbow configurations (up to 2^3 = 8)."""
    results = []
    for eA in (1, -1):
        for eB in (1, -1):
            for eC in (1, -1):
                try:
                    results.append(solve(x, y, phi, params, eA, eB, eC))
                except IKError:
                    pass
    return results


def forward(theta_A: float, theta_B: float, theta_C: float,
            params: Params | None = None,
            side_AB: int = 1,
            side_C:  int = 1) -> dict:
    """
    Forward kinematics (FK) given actuated joint angles.

    Algorithm
    ---------
    1. Compute elbow positions from ground angles.
    2. P_AB = intersection of circle(eA, lA2) and circle(eB, lB2).
    3. P_C  = intersection of circle(eC, lC2) and circle(P_AB, L).
    4. EE   = midpoint(P_AB, P_C).

    Parameters
    ----------
    theta_A/B/C : ground revolute angles (rad)
    side_AB, side_C : +1 or -1, select the geometric branch at each intersection.
                      Must be consistent with the elbow_* used during IK.

    Returns dict with keys: 'ee', 'phi', 'P_AB', 'P_C',
                            'elbow_A', 'elbow_B', 'elbow_C'
    """
    p = params or Params()

    eA = (p.A1[0] + p.lA1 * math.cos(theta_A),
          p.A1[1] + p.lA1 * math.sin(theta_A))
    eB = (p.B1[0] + p.lB1 * math.cos(theta_B),
          p.B1[1] + p.lB1 * math.sin(theta_B))
    eC = (p.C1[0] + p.lC1 * math.cos(theta_C),
          p.C1[1] + p.lC1 * math.sin(theta_C))

    P_AB = _circle_intersect(eA, p.lA2, eB, p.lB2, side_AB)
    P_C  = _circle_intersect(eC, p.lC2, P_AB, p.L, side_C)

    ee  = (0.5*(P_AB[0] + P_C[0]), 0.5*(P_AB[1] + P_C[1]))
    phi = math.atan2(P_C[1] - P_AB[1], P_C[0] - P_AB[0])
    return {
        "ee": ee, "phi": phi,
        "P_AB": P_AB, "P_C": P_C,
        "elbow_A": eA, "elbow_B": eB, "elbow_C": eC,
    }


# ---------------------------------------------------------------------------
# Jacobian and static force analysis
# ---------------------------------------------------------------------------

def _mat33_inv(m: list) -> list:
    """
    Gauss-Jordan inversion of a 3×3 matrix represented as a list of 3 rows.
    Raises IKError if the matrix is singular (mechanism at a singularity).
    """
    a = [r[:] for r in m]
    b = [[float(i == j) for j in range(3)] for i in range(3)]
    for col in range(3):
        pivot_row = max(range(col, 3), key=lambda r: abs(a[r][col]))
        a[col], a[pivot_row] = a[pivot_row], a[col]
        b[col], b[pivot_row] = b[pivot_row], b[col]
        p = a[col][col]
        if abs(p) < 1e-12:
            raise IKError("Jacobian singular – mechanism is at or near a singularity")
        inv_p = 1.0 / p
        for j in range(3):
            a[col][j] *= inv_p
            b[col][j] *= inv_p
        for row in range(3):
            if row == col:
                continue
            f = a[row][col]
            for j in range(3):
                a[row][j] -= f * a[col][j]
                b[row][j] -= f * b[col][j]
    return b


def jacobian_inv_fd(x: float, y: float, phi: float,
                    params: Params | None = None,
                    elbow_A: int = -1, elbow_B: int = 1, elbow_C: int = 1,
                    eps: float = 1e-5) -> list:
    """
    Numerical inverse Jacobian J⁻¹ [3×3] via forward differences on the IK.

    J⁻¹[i][j] = ∂θᵢ / ∂(ee_j)   where ee = [x, y, φ]
                                  and  θ  = [θ_A, θ_B, θ_C]

    Returned as a list of 3 rows, each a list of 3 floats.
    Call _mat33_inv(result) to get the forward Jacobian J.
    """
    p = params or Params()
    sol0 = solve(x, y, phi, p, elbow_A, elbow_B, elbow_C)
    t0 = (sol0.theta_A, sol0.theta_B, sol0.theta_C)

    cols = []   # cols[j] = ∂θ/∂(ee_j), a 3-vector
    for j, (dx, dy, dp) in enumerate([(eps, 0, 0), (0, eps, 0), (0, 0, eps)]):
        sol1 = solve(x + dx, y + dy, phi + dp, p, elbow_A, elbow_B, elbow_C)
        t1 = (sol1.theta_A, sol1.theta_B, sol1.theta_C)
        cols.append([(t1[i] - t0[i]) / eps for i in range(3)])

    # Assemble row-major: Jinv[i][j] = cols[j][i]
    return [[cols[j][i] for j in range(3)] for i in range(3)]


def actuator_torques(x: float, y: float, phi: float,
                     Fx: float, Fy: float, Mz: float,
                     params: Params | None = None,
                     elbow_A: int = -1, elbow_B: int = 1, elbow_C: int = 1
                     ) -> tuple[float, float, float]:
    """
    Required actuator torques for a generalised EE force, via Jacobian transpose.

      τ = Jᵀ · [Fx, Fy, Mz]

    where J = (J⁻¹)⁻¹ is the 3×3 forward Jacobian mapping actuator velocities
    to EE velocities.

    Units: if lengths are in cm and forces in N, torques are in N·cm.
    Mz should be in N·cm (moment around z).
    """
    Jinv = jacobian_inv_fd(x, y, phi, params, elbow_A, elbow_B, elbow_C)
    J    = _mat33_inv(Jinv)
    F    = [Fx, Fy, Mz]
    # τ[i] = Σⱼ J[j][i] · F[j]   (column i of J dotted with F)
    tau  = tuple(sum(J[j][i] * F[j] for j in range(3)) for i in range(3))
    return tau


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    p = Params()
    test_poses = [
        (10.0, 12.0,  0.0),
        (10.0, 12.0,  0.15),
        ( 8.0, 11.0, -0.10),
        (12.0, 13.0,  0.20),
    ]

    all_ok = True
    print(f"{'EE (x,y,phi°)':<28}  {'elbows':>10}  {'θA°':>7} {'θB°':>7} {'θC°':>7}  "
          f"{'pos_err':>9}  {'phi_err':>9}")
    print("-" * 90)

    for x, y, phi in test_poses:
        sols = solve_all(x, y, phi, p)
        if not sols:
            print(f"({x},{y},{math.degrees(phi):.1f}°)  NO SOLUTION")
            all_ok = False
            continue

        for sol in sols:
            # FK round-trip: try all 4 side combinations, keep best
            best_err = float("inf")
            for sAB in (1, -1):
                for sC in (1, -1):
                    try:
                        fk = forward(sol.theta_A, sol.theta_B, sol.theta_C, p, sAB, sC)
                        pos_err = math.hypot(fk["ee"][0]-x, fk["ee"][1]-y)
                        phi_err = abs(fk["phi"] - phi)
                        # normalise phi_err to [-pi, pi]
                        phi_err = abs((phi_err + math.pi) % (2*math.pi) - math.pi)
                        total = pos_err + phi_err
                        if total < best_err:
                            best_err = total
                            best_pos_err = pos_err
                            best_phi_err = phi_err
                    except IKError:
                        pass

            tA, tB, tC = sol.degrees
            label = f"({x},{y},{math.degrees(phi):.1f}°)"
            elbows = f"({sol.elbow_A:+},{sol.elbow_B:+},{sol.elbow_C:+})"
            ok = best_pos_err < 1e-4 and best_phi_err < 1e-4
            if not ok:
                all_ok = False
            flag = "" if ok else "  FAIL"
            print(f"{label:<28}  {elbows:>10}  {tA:>7.2f} {tB:>7.2f} {tC:>7.2f}  "
                  f"{best_pos_err:>9.2e}  {best_phi_err:>9.2e}{flag}")

    print()
    print("All round-trips passed." if all_ok else "SOME ROUND-TRIPS FAILED.")

    # Force analysis spot check
    print("\n--- Force analysis (default elbow config) ---")
    print(f"{'EE pose':<30}  {'Applied force':<22}  {'τ_A':>9} {'τ_B':>9} {'τ_C':>9}")
    print("-" * 90)
    force_cases = [
        (10.0, 12.0, 0.0,    1.0, 0.0, 0.0),   # 1 N in X
        (10.0, 12.0, 0.0,    0.0, 1.0, 0.0),   # 1 N in Y
        (10.0, 12.0, 0.0,    0.0, 0.0, 1.0),   # 1 N·cm torque
        (10.0, 12.0, 0.15,   0.0, 1.0, 0.0),   # 1 N in Y, rotated platform
    ]
    for x, y, phi, Fx, Fy, Mz in force_cases:
        try:
            tA, tB, tC = actuator_torques(x, y, phi, Fx, Fy, Mz, p)
            label  = f"({x},{y},{math.degrees(phi):.1f}°)"
            forces = f"Fx={Fx} Fy={Fy} Mz={Mz}"
            print(f"{label:<30}  {forces:<22}  {tA:>9.4f} {tB:>9.4f} {tC:>9.4f}")
        except IKError as e:
            print(f"  Error: {e}")

    sys.exit(0 if all_ok else 1)
