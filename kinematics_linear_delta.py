"""
Horizontal Linear Delta (Tripteron-style) Kinematics
=====================================================

Three prismatic carriages slide along rails arranged symmetrically at 120°
around the Y axis.  Each carriage is connected to the end-effector platform
by a rigid arm of length L.

Geometry conventions
--------------------
  R  : rail radius — XZ distance from the robot centre to each rail axis.
  r  : platform radius — XZ distance from the EE centre to each arm anchor.
  L  : arm (passive link) length.

Coordinate system
-----------------
  Origin at the centre of the workspace.
  Y points forward (along the rail direction shared by all three carriages).
  X / Z are the horizontal cross-section plane.

Each rail is at an XZ position:

    rail_xz[i] = R · [cos(φ_i), sin(φ_i)]

where φ₁ = 90°, φ₂ = 210°, φ₃ = 330°  (i.e. top, lower-left, lower-right
in a front-view cross-section).

Each EE anchor is at:

    eff_xz[i] = r · [cos(φ_i), sin(φ_i)]

Inverse kinematics
------------------
Given EE position (x, y, z), the carriage position along the rail is:

    ρ_i = y + √(L² − (x − eff_xz[i,0])² − (z − eff_xz[i,1])²)

(The + sign places carriages "behind" the EE along the Y axis.)

Forward kinematics
------------------
Each carriage sits at (rail_xz[i,0], ρ_i, rail_xz[i,1]).  The EE lies on
a sphere of radius L centred at (eff_xz[i,0], ρ_i, eff_xz[i,1]) for each leg.
Pairwise subtraction of sphere equations eliminates the quadratic terms and
yields two linear equations:

    x = ax + bx · y
    z = az + bz · y

Substituting back into sphere 0 gives a quadratic in y; the physically
correct root is the one where the EE is closest to the carriage plane (min y).

Static equilibrium forces
-------------------------
Differentiating the arm-length constraint:

    |P − B_i|² = L²   (P = EE, B_i = arm base = EE anchor on carriage)

gives  n_i · (Ṗ − Ḃ_i) = 0  where  n_i = P − B_i.
Since the carriage moves only in Y:  Ḃ_i = ρ̇_i · ê_y

    n_i · ê_y · ρ̇_i = n_i · ẋ_ee   →   k_i · ρ̇_i = n_i · ẋ_ee

Virtual work: Σ f_i · ρ̇_i = F · ẋ_ee, substituting constraint → f_i = k_i · (N⁻ᵀF)_i
where N has rows n_i and k_i = n_i · ê_y.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Module-level geometry constants
# ---------------------------------------------------------------------------

# Angular positions of the three rails/legs (radians) in the XZ cross-section
# φ₁ = 90° (top), φ₂ = 210° (lower-left), φ₃ = 330° (lower-right)
_ANGLES_DEG = [90.0, 210.0, 330.0]
_ANGLES = [math.radians(a) for a in _ANGLES_DEG]

# Outward unit vectors in XZ plane for each leg
LEG_DIRS_XZ = np.array([[math.cos(a), math.sin(a)] for a in _ANGLES])  # (3, 2)

# Y unit vector
E_Y = np.array([0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------

class LinearDeltaPositionError(Exception):
    """Raised when a requested position is outside the robot's workspace."""


class LinearDeltaRobot:
    """3-DOF horizontal linear delta robot kinematics."""

    def __init__(
        self,
        rail_radius: float,
        platform_radius: float,
        link_length: float,
    ):
        """
        Parameters
        ----------
        rail_radius : float
            XZ distance from robot centre to each rail axis (R).
        platform_radius : float
            XZ distance from EE centre to each arm anchor on the platform (r).
        link_length : float
            Length of each passive arm connecting carriage to EE platform (L).
        """
        self.R = rail_radius
        self.r = platform_radius
        self.L = link_length
        self._update_derived()

    def _update_derived(self) -> None:
        """Precompute geometry that depends only on R, r, L."""
        # Rail XZ positions: R · [cos φ, sin φ]
        self.rail_xz = self.R * LEG_DIRS_XZ          # (3, 2)
        # EE platform anchor offsets (local, relative to EE centre): r · [cos φ, sin φ]
        self.ee_anchor_xz = self.r * LEG_DIRS_XZ     # (3, 2)
        # Sphere-centre XZ offsets used in IK/FK: (R-r) · [cos φ, sin φ]
        # Derived from arm-length constraint:
        #   |carriage - anchor|² = L²
        #   carriage XZ = R*d_i, anchor XZ = (x,z) + r*d_i
        #   → arm XZ component = (R-r)*d_i - (x,z)
        self.eff_xz = (self.R - self.r) * LEG_DIRS_XZ  # (3, 2)

    # ------------------------------------------------------------------
    # Inverse kinematics
    # ------------------------------------------------------------------

    def inverse(
        self, x: float, y: float, z: float
    ) -> tuple[float, float, float]:
        """
        Compute carriage positions from an EE position.

        Parameters
        ----------
        x, y, z : float
            Desired EE position.

        Returns
        -------
        (rho1, rho2, rho3) : carriage Y-positions.

        Raises
        ------
        LinearDeltaPositionError
            If the position is outside the reachable workspace.
        """
        rhos = []
        for i in range(3):
            dx = x - self.eff_xz[i, 0]
            dz = z - self.eff_xz[i, 1]
            disc = self.L ** 2 - dx ** 2 - dz ** 2
            if disc < 0.0:
                raise LinearDeltaPositionError(
                    f"Position ({x:.2f}, {y:.2f}, {z:.2f}) unreachable for leg {i+1}: "
                    f"XZ offset² = {dx**2 + dz**2:.2f} > L² = {self.L**2:.2f}"
                )
            rhos.append(y + math.sqrt(disc))
        return tuple(rhos)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------

    def forward(
        self, rho1: float, rho2: float, rho3: float
    ) -> tuple[float, float, float] | None:
        """
        Compute EE position from carriage positions.

        Each carriage i provides a sphere of radius L centred at
        (eff_xz[i,0], rho_i, eff_xz[i,1]).  Subtracting sphere equations
        pairwise gives two linear equations in (x, y, z).  We express
        x = ax + bx*y and z = az + bz*y, then substitute into sphere 0 to
        get a quadratic in y.

        Returns
        -------
        (x, y, z) tuple, or None if geometrically impossible.
        """
        rhos = [rho1, rho2, rho3]
        # Sphere centres: (cx_i, cy_i, cz_i) = (eff_xz[i,0], rho_i, eff_xz[i,1])
        cx = self.eff_xz[:, 0]  # (3,)
        cz = self.eff_xz[:, 1]  # (3,)
        cy = np.array(rhos, dtype=float)  # (3,)

        # |P - C_i|² = L²  →  x²+y²+z² - 2*cx_i*x - 2*cy_i*y - 2*cz_i*z + |C_i|² = L²
        # Subtract sphere 0 from sphere 1 and from sphere 2:
        #   2*(cx_0 - cx_j)*x + 2*(cy_0 - cy_j)*y + 2*(cz_0 - cz_j)*z = |C_j|² - |C_0|² + L² - L²
        #   = |C_j|² - |C_0|²

        # Subtracting sphere 0 from sphere j eliminates x²+y²+z²:
        #   2*(cx_j - cx_0)*x + 2*(cy_j - cy_0)*y + 2*(cz_j - cz_0)*z
        #   = |C_j|² - |C_0|²
        def sphere_rhs(j):
            return (
                cx[j]**2 + cy[j]**2 + cz[j]**2
                - cx[0]**2 - cy[0]**2 - cz[0]**2
            )

        # Equation 1 (j=1):  a1*x + b1*y + c1*z = d1
        a1 = 2.0 * (cx[1] - cx[0])
        b1 = 2.0 * (cy[1] - cy[0])
        c1 = 2.0 * (cz[1] - cz[0])
        d1 = sphere_rhs(1)

        # Equation 2 (j=2):  a2*x + b2*y + c2*z = d2
        a2 = 2.0 * (cx[2] - cx[0])
        b2 = 2.0 * (cy[2] - cy[0])
        c2 = 2.0 * (cz[2] - cz[0])
        d2 = sphere_rhs(2)

        # Solve for x and z in terms of y (x = Ax + Bx*y, z = Az + Bz*y)
        # [ a1  c1 ] [ x ]   [ d1 - b1*y ]
        # [ a2  c2 ] [ z ] = [ d2 - b2*y ]
        det_xz = a1 * c2 - a2 * c1
        if abs(det_xz) < 1e-12:
            return None  # degenerate configuration

        # x = Ax + Bx*y,  z = Az + Bz*y
        Ax = ( d1 * c2 - d2 * c1) / det_xz
        Bx = (-b1 * c2 + b2 * c1) / det_xz
        Az = ( a1 * d2 - a2 * d1) / det_xz
        Bz = (-a1 * b2 + a2 * b1) / det_xz

        # Substitute into sphere 0:
        # (Ax + Bx*y - cx[0])² + (y - cy[0])² + (Az + Bz*y - cz[0])² = L²
        Px = Ax - cx[0]
        Pz = Az - cz[0]

        a_q = Bx**2 + 1.0 + Bz**2
        b_q = 2.0 * (Px * Bx - cy[0] + Pz * Bz)
        c_q = Px**2 + cy[0]**2 + Pz**2 - self.L**2

        disc = b_q**2 - 4.0 * a_q * c_q
        if disc < 0.0:
            return None

        sqrt_d = math.sqrt(disc)
        y1 = (-b_q + sqrt_d) / (2.0 * a_q)
        y2 = (-b_q - sqrt_d) / (2.0 * a_q)

        # The + sign IK convention means EE is "ahead" (smaller y) of carriages.
        # Pick the root where EE y is smallest (closest to carriage front face).
        y0 = min(y1, y2)
        x0 = Ax + Bx * y0
        z0 = Az + Bz * y0
        return (x0, y0, z0)

    # ------------------------------------------------------------------
    # Full geometry for rendering
    # ------------------------------------------------------------------

    def geometry(
        self, x: float, y: float, z: float
    ) -> dict:
        """
        Compute all 3-D positions needed to render the robot.

        Returns a dict with keys:
            rhos         : (3,) carriage Y-positions
            carriages    : (3, 3) carriage positions [rail_xz[i,0], rho_i, rail_xz[i,1]]
            platform_pts : (3, 3) EE anchor points [eff_xz[i,0]+x, y, eff_xz[i,1]+z]
            ee_center    : (3,)  EE centre position

        Raises LinearDeltaPositionError if (x, y, z) is unreachable.
        """
        rhos = self.inverse(x, y, z)
        ee = np.array([x, y, z], dtype=float)

        carriages = np.column_stack([
            self.rail_xz[:, 0],
            np.array(rhos),
            self.rail_xz[:, 1],
        ])  # (3, 3) — X, Y, Z of each carriage

        # EE anchor points (absolute): EE centre + r·d_i in XZ
        platform_pts = np.column_stack([
            self.ee_anchor_xz[:, 0] + x,
            np.full(3, y),
            self.ee_anchor_xz[:, 1] + z,
        ])  # (3, 3) — EE anchor points

        return {
            "rhos":         np.array(rhos),
            "carriages":    carriages,
            "platform_pts": platform_pts,
            "ee_center":    ee,
        }

    # ------------------------------------------------------------------
    # Static equilibrium actuator forces
    # ------------------------------------------------------------------

    def static_forces(
        self,
        x: float,
        y: float,
        z: float,
        force: np.ndarray,
    ) -> np.ndarray:
        """
        Compute actuator forces for static equilibrium under an applied force.

        Parameters
        ----------
        x, y, z : float
            Current EE position.
        force : array-like, shape (3,)
            Force vector [Fx, Fy, Fz] applied at the EE centre (Newtons).

        Returns
        -------
        forces : ndarray, shape (3,)
            Actuator forces [f₁, f₂, f₃] (Newtons, positive = tension).

        Derivation
        ----------
        n_i = platform_pt_i − carriage_i  (arm vector, length L)
        k_i = n_i · ê_y                   (Y projection)

        The constraint differentiates to: k_i · ρ̇_i = n_i · ẋ_ee
        Virtual work: f_i = k_i · (N⁻ᵀ F)_i  where N has rows n_i.
        """
        geo = self.geometry(x, y, z)
        platform_pts = geo["platform_pts"]
        carriages    = geo["carriages"]
        F = np.asarray(force, dtype=float)

        n_vecs = platform_pts - carriages  # (3, 3) arm vectors
        k_vals = n_vecs @ E_Y              # (3,) Y-projections

        y_vec = np.linalg.solve(n_vecs.T, F)
        return k_vals * y_vec

    # ------------------------------------------------------------------
    # Workspace sampling
    # ------------------------------------------------------------------

    def sample_workspace(
        self,
        rho_min: float | None = None,
        rho_max: float | None = None,
        steps: int = 20,
    ) -> list[tuple[float, float, float]]:
        """
        Return reachable (x, y, z) points by sampling carriage space.

        Parameters
        ----------
        rho_min, rho_max : float, optional
            Range of each carriage Y-position to sweep.
            Defaults to [-L/2, L/2] relative to origin.
        steps : int
            Number of samples per axis.
        """
        if rho_min is None:
            rho_min = -self.L / 2.0
        if rho_max is None:
            rho_max = self.L / 2.0

        rho_vals = [
            rho_min + (rho_max - rho_min) * i / (steps - 1)
            for i in range(steps)
        ]
        points: list[tuple[float, float, float]] = []
        for r1 in rho_vals:
            for r2 in rho_vals:
                for r3 in rho_vals:
                    result = self.forward(r1, r2, r3)
                    if result is not None:
                        points.append(result)
        return points
