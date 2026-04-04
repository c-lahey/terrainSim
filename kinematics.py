"""
Delta Robot Kinematics

Forward and inverse kinematics for a 3-DOF delta robot.

Adapted from: https://github.com/mhp/delta-bot/blob/main/kinematics.py
Original algorithm from: http://forums.trossenrobotics.com/tutorials/introduction-129/delta-robot-kinematics-3276/

Geometry conventions:
    f  : servo displacement — distance from centre of the base triangle to each
         servo output shaft (measured along the triangle's median)
    e  : effector displacement — distance from centre of the end-effector
         platform to each parallel-link anchor point
    rf : active (servo) link length
    re : passive (parallel) link length

Angles are in degrees; zero is horizontal. Negative angles tilt the arm
downward (toward the workpiece).

Coordinate system:
    Origin at the centre of the base triangle.
    Z points downward (toward the effector).
    X/Y follow a right-hand rule in the horizontal plane.
"""

import math


class DeltaPositionError(Exception):
    """Raised when a requested position is outside the robot's workspace."""


class DeltaRobot:
    """3-DOF delta robot kinematics."""

    def __init__(
        self,
        servo_link_length: float,
        parallel_link_length: float,
        servo_displacement: float,
        effector_displacement: float,
    ):
        """
        Parameters
        ----------
        servo_link_length : float
            Length of the active (upper) arm attached to each servo (rf).
        parallel_link_length : float
            Length of the passive (lower) parallel link (re).
        servo_displacement : float
            Distance from base centre to each servo output shaft (f).
        effector_displacement : float
            Distance from effector centre to each parallel-link anchor (e).
        """
        self.rf = servo_link_length
        self.re = parallel_link_length
        self.f = servo_displacement
        self.e = effector_displacement

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------

    def forward(
        self, theta1: float, theta2: float, theta3: float
    ) -> tuple[float, float, float] | None:
        """
        Compute end-effector position from three servo angles (degrees).

        Parameters
        ----------
        theta1, theta2, theta3 : float
            Servo angles in degrees. Zero is horizontal; negative tilts down.

        Returns
        -------
        (x, y, z) tuple, or None if the position is geometrically impossible.
        """
        t = self.f - self.e

        t1 = math.radians(theta1)
        t2 = math.radians(theta2)
        t3 = math.radians(theta3)

        # Joint positions of each leg's lower-link anchor
        y1 = -(t + self.rf * math.cos(t1))
        z1 = -self.rf * math.sin(t1)

        y2 = (t + self.rf * math.cos(t2)) * math.sin(math.pi / 6)
        x2 = y2 * math.tan(math.pi / 3)
        z2 = -self.rf * math.sin(t2)

        y3 = (t + self.rf * math.cos(t3)) * math.sin(math.pi / 6)
        x3 = -y3 * math.tan(math.pi / 3)
        z3 = -self.rf * math.sin(t3)

        # Intersection of three spheres of radius re centred at the joint positions
        dnm = (y2 - y1) * x3 - (y3 - y1) * x2

        w1 = y1 * y1 + z1 * z1
        w2 = x2 * x2 + y2 * y2 + z2 * z2
        w3 = x3 * x3 + y3 * y3 + z3 * z3

        a1 = (z2 - z1) * (y3 - y1) - (z3 - z1) * (y2 - y1)
        b1 = -((w2 - w1) * (y3 - y1) - (w3 - w1) * (y2 - y1)) / 2.0

        a2 = -(z2 - z1) * x3 + (z3 - z1) * x2
        b2 = ((w2 - w1) * x3 - (w3 - w1) * x2) / 2.0

        a = a1 * a1 + a2 * a2 + dnm * dnm
        b = 2.0 * (a1 * b1 + a2 * (b2 - y1 * dnm) - z1 * dnm * dnm)
        c = (
            (b2 - y1 * dnm) ** 2
            + b1 * b1
            + dnm * dnm * (z1 * z1 - self.re * self.re)
        )

        discriminant = b * b - 4.0 * a * c
        if discriminant < 0:
            return None  # position outside workspace

        z0 = -0.5 * (b + math.sqrt(discriminant)) / a
        x0 = (a1 * z0 + b1) / dnm
        y0 = (a2 * z0 + b2) / dnm
        return (x0, y0, z0)

    # ------------------------------------------------------------------
    # Inverse kinematics
    # ------------------------------------------------------------------

    def _angle_yz(self, x0: float, y0: float, z0: float) -> float:
        """
        Compute servo angle for one leg when the coordinate frame has been
        rotated so that leg lies in the YZ plane.

        Returns angle in degrees.
        Raises DeltaPositionError if position is unreachable.
        """
        y1 = -self.f
        y0 -= self.e

        a = (x0 ** 2 + y0 ** 2 + z0 ** 2 + self.rf ** 2 - self.re ** 2 - y1 ** 2) / (2.0 * z0)
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

        Parameters
        ----------
        x, y, z : float
            Desired end-effector position in the robot's coordinate frame.

        Returns
        -------
        (theta1, theta2, theta3) in degrees.

        Raises
        ------
        DeltaPositionError
            If the position is outside the robot's reachable workspace.
        """
        cos120 = math.cos(2.0 * math.pi / 3.0)
        sin120 = math.sin(2.0 * math.pi / 3.0)

        theta1 = self._angle_yz(x, y, z)
        theta2 = self._angle_yz(
            x * cos120 + y * sin120,
            y * cos120 - x * sin120,
            z,
        )  # rotate frame +120°
        theta3 = self._angle_yz(
            x * cos120 - y * sin120,
            y * cos120 + x * sin120,
            z,
        )  # rotate frame −120°

        return (theta1, theta2, theta3)

    # ------------------------------------------------------------------
    # Workspace sampling
    # ------------------------------------------------------------------

    def sample_workspace(
        self,
        angle_min: float = -30.0,
        angle_max: float = 60.0,
        steps: int = 20,
    ) -> list[tuple[float, float, float]]:
        """
        Return a list of reachable (x, y, z) points by sampling joint space.

        Parameters
        ----------
        angle_min, angle_max : float
            Range of each servo angle to sweep (degrees).
        steps : int
            Number of evenly-spaced samples per axis.
        """
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
