"""
Delta Robot Kinematics

Forward and inverse kinematics for a 3-DOF delta robot with configurable
triangle geometry (equilateral or isosceles base / EE platforms).

Adapted from: https://github.com/mhp/delta-bot/blob/main/kinematics.py
Original algorithm from: http://forums.trossenrobotics.com/tutorials/introduction-129/delta-robot-kinematics-3276/

Geometry conventions
--------------------
  f          : servo displacement — circumradius of the base triangle
  e          : effector displacement — circumradius of the EE triangle
  rf         : active (servo) link length
  re         : passive (parallel) link length
  apex_angle : interior angle at the apex vertex of both triangles (degrees).
               60° = equilateral; smaller = narrower/more pointed triangle.

The apex vertex always points in the −Y direction. The two base vertices are
symmetric about the Y axis, each rotated by δ = 180° − apex_angle from the
apex direction.

Coordinate system
-----------------
  Origin at the centre of the base triangle.
  Z points downward (toward the effector).
  X / Y follow a right-hand rule in the horizontal plane.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Module-level equilateral constants (kept for backward compatibility)
# ---------------------------------------------------------------------------

LEG_DIRS = np.array([
    [ 0.0,              -1.0,             0.0],   # leg 1  (apex, −Y)
    [ math.sqrt(3)/2,    0.5,             0.0],   # leg 2  (+120°)
    [-math.sqrt(3)/2,    0.5,             0.0],   # leg 3  (−120°)
])

LEG_TANGS = np.array([[-d[1], d[0], 0.0] for d in LEG_DIRS])

_DOWN = np.array([0.0, 0.0, -1.0])


# ---------------------------------------------------------------------------
# Triangle geometry helper
# ---------------------------------------------------------------------------

def _compute_leg_dirs(apex_angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute leg direction and tangent arrays for a given apex angle.

    Parameters
    ----------
    apex_angle_deg : float
        Interior angle at the apex vertex (degrees). 60 = equilateral.

    Returns
    -------
    leg_dirs  : (3, 3) array — outward unit vectors in XY plane.
    leg_tangs : (3, 3) array — CCW-90° tangents to leg_dirs.

    Derivation
    ----------
    The apex leg always points in −Y (angle −90°).  Each base leg is
    rotated by δ = 180° − apex_angle from the apex direction:

        d₁ = (0, −1, 0)
        d₂ = (cos(−90° + δ), sin(−90° + δ), 0)
        d₃ = (cos(−90° − δ), sin(−90° − δ), 0)

    At apex_angle = 60°, δ = 120° and this recovers the standard equilateral
    layout.  The interior angle of the triangle at the apex vertex equals
    apex_angle_deg (verified via the inscribed-angle theorem).
    """
    delta = math.radians(180.0 - apex_angle_deg)
    phi0  = -math.pi / 2  # apex direction

    dirs = np.array([
        [math.cos(phi0),         math.sin(phi0),         0.0],
        [math.cos(phi0 + delta), math.sin(phi0 + delta), 0.0],
        [math.cos(phi0 - delta), math.sin(phi0 - delta), 0.0],
    ])
    tangs = np.array([[-d[1], d[0], 0.0] for d in dirs])
    return dirs, tangs


# ---------------------------------------------------------------------------

class DeltaPositionError(Exception):
    """Raised when a requested position is outside the robot's workspace."""


class DeltaRobot:
    """3-DOF delta robot kinematics with configurable triangle geometry."""

    def __init__(
        self,
        servo_link_length: float,
        parallel_link_length: float,
        servo_displacement: float,
        effector_displacement: float,
        apex_angle: float = 60.0,
    ):
        """
        Parameters
        ----------
        servo_link_length : float
            Length of the active (upper) arm attached to each servo (rf).
        parallel_link_length : float
            Length of the passive (lower) parallel link (re).
        servo_displacement : float
            Circumradius of the base triangle — distance from base centre to
            each servo output shaft (f).
        effector_displacement : float
            Circumradius of the EE triangle — distance from EE centre to each
            parallel-link anchor (e).
        apex_angle : float
            Interior angle at the apex vertex of both triangles (degrees).
            60° gives the standard equilateral delta geometry.
        """
        self.rf = servo_link_length
        self.re = parallel_link_length
        self.f  = servo_displacement
        self.e  = effector_displacement
        self.apex_angle = apex_angle
        self._leg_dirs, self._leg_tangs = _compute_leg_dirs(apex_angle)

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------

    def forward(
        self, theta1: float, theta2: float, theta3: float
    ) -> tuple[float, float, float] | None:
        """
        Compute end-effector position from three servo angles (degrees).

        Returns (x, y, z), or None if geometrically impossible.

        Algorithm
        ---------
        Each servo + arm defines one sphere of radius re centred at the
        "adjusted elbow":

            Q_i = (t + rf·cos θᵢ)·dᵢ  +  (0, 0, −rf·sin θᵢ)

        where t = f − e and dᵢ is the leg's outward unit vector.

        The EE centre is the intersection of the three spheres.  Subtracting
        sphere 1 from spheres 2 and 3 gives two linear equations:

            e₁ⱼ · P = h₁ⱼ   (j = 2, 3)

        Solving for x and y as affine functions of z, then substituting into
        sphere 1 yields a quadratic in z whose downward root is the solution.
        """
        t = self.f - self.e
        d = self._leg_dirs

        # Sphere centres Q_i
        thetas = [math.radians(a) for a in (theta1, theta2, theta3)]
        Q = np.empty((3, 3))
        for i, th in enumerate(thetas):
            r = t + self.rf * math.cos(th)
            Q[i] = [r * d[i, 0], r * d[i, 1], -self.rf * math.sin(th)]

        # Pairwise differences: e₁₂ = Q₂ − Q₁, e₁₃ = Q₃ − Q₁
        e12 = Q[1] - Q[0]
        e13 = Q[2] - Q[0]
        h12 = 0.5 * (np.dot(Q[1], Q[1]) - np.dot(Q[0], Q[0]))
        h13 = 0.5 * (np.dot(Q[2], Q[2]) - np.dot(Q[0], Q[0]))

        # Solve 2×2 for x, y in terms of z:
        #   e12_x·x + e12_y·y = h12 − e12_z·z
        #   e13_x·x + e13_y·y = h13 − e13_z·z
        det = e12[0] * e13[1] - e13[0] * e12[1]
        if abs(det) < 1e-12:
            return None

        Ax = ( h12 * e13[1] - h13 * e12[1]) / det
        Bx = (-e12[2] * e13[1] + e13[2] * e12[1]) / det
        Ay = ( e12[0] * h13 - e13[0] * h12) / det
        By = (-e12[0] * e13[2] + e13[0] * e12[2]) / det

        # Substitute into sphere 0: |P − Q₀|² = re²
        Cx = Ax - Q[0, 0]
        Cy = Ay - Q[0, 1]

        a_q = Bx ** 2 + By ** 2 + 1.0
        b_q = 2.0 * (Cx * Bx + Cy * By - Q[0, 2])
        c_q = Cx ** 2 + Cy ** 2 + Q[0, 2] ** 2 - self.re ** 2

        disc = b_q ** 2 - 4.0 * a_q * c_q
        if disc < 0.0:
            return None

        z0 = -0.5 * (b_q + math.sqrt(disc)) / a_q  # downward root
        return (Ax + Bx * z0, Ay + By * z0, z0)

    # ------------------------------------------------------------------
    # Inverse kinematics
    # ------------------------------------------------------------------

    def _angle_yz(self, x0: float, y0: float, z0: float) -> float:
        """
        Servo angle for one leg after rotating the world frame so that leg
        lies in the YZ plane.  x0 is the residual off-plane component (handled
        analytically — the passive link need not be coplanar with the arm).

        Returns angle in degrees.  Raises DeltaPositionError if unreachable.
        """
        y1 = -self.f
        y0 -= self.e

        a = (x0 ** 2 + y0 ** 2 + z0 ** 2 + self.rf ** 2
             - self.re ** 2 - y1 ** 2) / (2.0 * z0)
        b = (y1 - y0) / z0

        d = -(a + b * y1) ** 2 + self.rf * (b * b * self.rf + self.rf)
        if d < 0:
            raise DeltaPositionError(
                f"Position ({x0:.2f}, {y0:.2f}, {z0:.2f}) is unreachable."
            )

        yj = (y1 - a * b - math.sqrt(d)) / (b * b + 1)
        zj = a + b * yj
        theta = math.degrees(math.atan(-zj / (y1 - yj)))
        if yj > y1:
            theta += 180.0
        return theta

    def inverse(
        self, x: float, y: float, z: float
    ) -> tuple[float, float, float]:
        """
        Compute servo angles from an end-effector position.

        Returns (theta1, theta2, theta3) in degrees.
        Raises DeltaPositionError if unreachable.

        For each leg i with outward direction dᵢ at azimuth φᵢ, the world
        frame is rotated by αᵢ = −π/2 − φᵢ so that dᵢ aligns with −Y, then
        _angle_yz is called in the rotated frame.  At apex_angle = 60° this
        recovers the standard ±120° rotations exactly.
        """
        thetas = []
        for d_i in self._leg_dirs:
            phi_i   = math.atan2(d_i[1], d_i[0])
            alpha_i = -math.pi / 2 - phi_i
            ca, sa  = math.cos(alpha_i), math.sin(alpha_i)
            thetas.append(self._angle_yz(
                x * ca - y * sa,
                x * sa + y * ca,
                z,
            ))
        return tuple(thetas)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def geometry(self, x: float, y: float, z: float) -> dict:
        """
        Compute all 3-D positions needed to render the robot configuration.

        Returns a dict with:
            thetas    : (3,) servo angles in degrees
            servos    : (3, 3) servo output positions
            elbows    : (3, 3) elbow positions (end of active arm)
            anchors   : (3, 3) EE platform anchor points
            ee_center : (3,)  end-effector centre

        Raises DeltaPositionError if (x, y, z) is unreachable.
        """
        thetas = np.array(self.inverse(x, y, z))
        ee     = np.array([x, y, z])
        d      = self._leg_dirs

        servos  = self.f * d           # (3, 3)

        elbows = np.empty((3, 3))
        for i in range(3):
            th = math.radians(thetas[i])
            elbows[i] = servos[i] + self.rf * (d[i] * math.cos(th)
                                                + _DOWN * math.sin(th))

        anchors = ee[None, :] + self.e * d   # (3, 3)

        return {
            "thetas":    thetas,
            "servos":    servos,
            "elbows":    elbows,
            "anchors":   anchors,
            "ee_center": ee,
        }

    # ------------------------------------------------------------------
    # Static torques
    # ------------------------------------------------------------------

    def static_torques(
        self,
        x: float,
        y: float,
        z: float,
        force: np.ndarray,
    ) -> np.ndarray:
        """
        Joint torques for static equilibrium under an external force at the EE.

        Parameters
        ----------
        x, y, z : float
        force    : (3,) force vector [Fx, Fy, Fz] in Newtons (lengths in mm
                   ⇒ torques in N·mm).

        Returns
        -------
        torques : (3,) array [τ₁, τ₂, τ₃].

        Derivation
        ----------
        Differentiating |Eᵢ − Aᵢ|² = re² gives nᵢ · (Ėᵢ − ẋ_ee) = 0
        where nᵢ = Eᵢ − Aᵢ.  Virtual work + constraint elimination yields:

            τᵢ = bᵢ · (N⁻ᵀ F)ᵢ    where  bᵢ = nᵢ · (∂Eᵢ/∂θᵢ),  N rows = nᵢ
        """
        geo     = self.geometry(x, y, z)
        thetas  = geo["thetas"]
        elbows  = geo["elbows"]
        anchors = geo["anchors"]
        d       = self._leg_dirs
        F       = np.asarray(force, dtype=float)

        n_vecs = np.empty((3, 3))
        b_vals = np.empty(3)
        for i in range(3):
            th    = math.radians(thetas[i])
            n_i   = elbows[i] - anchors[i]
            dE_i  = self.rf * (-d[i] * math.sin(th) + _DOWN * math.cos(th))
            n_vecs[i] = n_i
            b_vals[i] = float(np.dot(n_i, dE_i))

        y_vec = np.linalg.solve(n_vecs.T, F)
        return b_vals * y_vec

    # ------------------------------------------------------------------
    # Workspace sampling
    # ------------------------------------------------------------------

    def sample_workspace(
        self,
        angle_min: float = -30.0,
        angle_max: float =  60.0,
        steps:     int   =  20,
    ) -> list[tuple[float, float, float]]:
        """Return reachable (x, y, z) points by sweeping joint space."""
        angles = [
            angle_min + (angle_max - angle_min) * i / (steps - 1)
            for i in range(steps)
        ]
        points: list[tuple[float, float, float]] = []
        for t1 in angles:
            for t2 in angles:
                for t3 in angles:
                    result = self.forward(t1, t2, t3)
                    if result is not None:
                        points.append(result)
        return points
