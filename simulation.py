"""
Delta Robot Kinematics Simulation

Demonstrates forward kinematics, inverse kinematics, workspace visualisation,
and animated path following for a delta robot.

Usage
-----
    python simulation.py                    # interactive menu
    python simulation.py workspace          # workspace scatter plot
    python simulation.py path              # animated path demo
    python simulation.py roundtrip         # FK/IK round-trip accuracy test
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3-D projection

from kinematics import DeltaRobot, DeltaPositionError

# ---------------------------------------------------------------------------
# Default robot geometry (units: mm)
# Matches the example in mhp/delta-bot:
#   servo_link_length=85, parallel_link_length=210,
#   servo_displacement=72, effector_displacement=20
# ---------------------------------------------------------------------------
DEFAULT_ROBOT = DeltaRobot(
    servo_link_length=85.0,
    parallel_link_length=210.0,
    servo_displacement=72.0,
    effector_displacement=20.0,
)


# ---------------------------------------------------------------------------
# Workspace visualisation
# ---------------------------------------------------------------------------

def plot_workspace(
    robot: DeltaRobot = DEFAULT_ROBOT,
    angle_min: float = -10.0,
    angle_max: float = 50.0,
    steps: int = 15,
) -> None:
    """
    Scatter-plot the robot's reachable workspace by sweeping joint space.
    Points are colour-coded by Z depth.
    """
    print("Sampling workspace…")
    points = robot.sample_workspace(angle_min, angle_max, steps)
    if not points:
        print("No reachable points found — check angle limits.")
        return

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xs, ys, zs, c=zs, cmap="viridis", s=8, alpha=0.6)
    fig.colorbar(sc, ax=ax, label="Z (mm)")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(
        f"Delta Robot Reachable Workspace\n"
        f"rf={robot.rf} mm  re={robot.re} mm  "
        f"f={robot.f} mm  e={robot.e} mm\n"
        f"angles [{angle_min}°, {angle_max}°]  —  {len(points):,} points"
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Forward-kinematics round-trip accuracy test
# ---------------------------------------------------------------------------

def roundtrip_test(
    robot: DeltaRobot = DEFAULT_ROBOT,
    angle_min: float = -10.0,
    angle_max: float = 50.0,
    step: float = 5.0,
) -> None:
    """
    For each angle triple in the sweep, compute FK then IK and report the
    maximum angular error.
    """
    angles = list(np.arange(angle_min, angle_max + step / 2, step))
    max_err = 0.0
    count = 0
    failures = 0

    for t1 in angles:
        for t2 in angles:
            for t3 in angles:
                pos = robot.forward(t1, t2, t3)
                if pos is None:
                    continue
                try:
                    r1, r2, r3 = robot.inverse(*pos)
                except DeltaPositionError:
                    failures += 1
                    continue
                err = max(abs(t1 - r1), abs(t2 - r2), abs(t3 - r3))
                max_err = max(max_err, err)
                count += 1

    print(f"Round-trip test over {count} valid angle triples:")
    print(f"  Max angular error : {max_err:.2e} degrees")
    print(f"  IK failures       : {failures}")


# ---------------------------------------------------------------------------
# Path following animation
# ---------------------------------------------------------------------------

def _build_path(robot: DeltaRobot) -> list[tuple[float, float, float]]:
    """
    Generate a smooth circular path in the XY plane at a fixed Z depth.
    Returns a list of (x, y, z) waypoints, only including reachable points.
    """
    # Pick a Z depth roughly in the middle of the workspace
    # by probing the centre with equal angles
    centre = robot.forward(20.0, 20.0, 20.0)
    if centre is None:
        centre = (0.0, 0.0, -150.0)
    z = centre[2]

    radius = 20.0  # mm
    n_points = 120
    waypoints = []
    for i in range(n_points + 1):
        angle = 2.0 * math.pi * i / n_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        # Verify reachability via IK
        try:
            robot.inverse(x, y, z)
            waypoints.append((x, y, z))
        except DeltaPositionError:
            pass
    return waypoints


def animate_path(robot: DeltaRobot = DEFAULT_ROBOT) -> None:
    """
    Animate the end-effector following a circular path, showing servo angles.
    """
    waypoints = _build_path(robot)
    if len(waypoints) < 2:
        print("Could not build a reachable circular path with default parameters.")
        return

    xs = [p[0] for p in waypoints]
    ys = [p[1] for p in waypoints]
    zs = [p[2] for p in waypoints]

    angles_over_time = [robot.inverse(*p) for p in waypoints]
    t1s = [a[0] for a in angles_over_time]
    t2s = [a[1] for a in angles_over_time]
    t3s = [a[2] for a in angles_over_time]
    frames = list(range(len(waypoints)))

    fig = plt.figure(figsize=(12, 5))
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)

    # Static path trace
    ax3d.plot(xs, ys, zs, "b--", alpha=0.3, linewidth=1)
    (point3d,) = ax3d.plot([], [], [], "ro", markersize=8)
    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")
    ax3d.set_title("End-effector path")

    # Angle histories
    ax2d.set_xlim(0, len(waypoints))
    y_lo = min(min(t1s), min(t2s), min(t3s)) - 2
    y_hi = max(max(t1s), max(t2s), max(t3s)) + 2
    ax2d.set_ylim(y_lo, y_hi)
    ax2d.set_xlabel("Frame")
    ax2d.set_ylabel("Angle (degrees)")
    ax2d.set_title("Servo angles")
    ax2d.grid(True, alpha=0.3)
    (line1,) = ax2d.plot([], [], label="θ₁", color="tab:blue")
    (line2,) = ax2d.plot([], [], label="θ₂", color="tab:orange")
    (line3,) = ax2d.plot([], [], label="θ₃", color="tab:green")
    ax2d.legend(loc="upper right")

    def init():
        point3d.set_data([], [])
        point3d.set_3d_properties([])
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        return point3d, line1, line2, line3

    def update(frame):
        i = frame + 1
        point3d.set_data([xs[frame]], [ys[frame]])
        point3d.set_3d_properties([zs[frame]])
        line1.set_data(frames[:i], t1s[:i])
        line2.set_data(frames[:i], t2s[:i])
        line3.set_data(frames[:i], t3s[:i])
        return point3d, line1, line2, line3

    ani = animation.FuncAnimation(
        fig, update, frames=len(waypoints), init_func=init,
        interval=30, blit=True
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Interactive demo
# ---------------------------------------------------------------------------

def demo_fk(robot: DeltaRobot = DEFAULT_ROBOT) -> None:
    """Prompt for servo angles and print the end-effector position."""
    try:
        t1 = float(input("  θ₁ (deg): "))
        t2 = float(input("  θ₂ (deg): "))
        t3 = float(input("  θ₃ (deg): "))
    except ValueError:
        print("Invalid input.")
        return
    result = robot.forward(t1, t2, t3)
    if result is None:
        print("  → Position unreachable (outside workspace).")
    else:
        x, y, z = result
        print(f"  → End-effector: x={x:.3f} mm  y={y:.3f} mm  z={z:.3f} mm")


def demo_ik(robot: DeltaRobot = DEFAULT_ROBOT) -> None:
    """Prompt for an end-effector position and print servo angles."""
    try:
        x = float(input("  x (mm): "))
        y = float(input("  y (mm): "))
        z = float(input("  z (mm): "))
    except ValueError:
        print("Invalid input.")
        return
    try:
        t1, t2, t3 = robot.inverse(x, y, z)
        print(f"  → θ₁={t1:.3f}°  θ₂={t2:.3f}°  θ₃={t3:.3f}°")
    except DeltaPositionError as exc:
        print(f"  → Unreachable: {exc}")


MENU = """
Delta Robot Kinematics Simulation
==================================
  1) Forward kinematics (angles → position)
  2) Inverse kinematics (position → angles)
  3) Plot reachable workspace
  4) Animate circular path
  5) FK/IK round-trip accuracy test
  q) Quit
"""


def interactive_menu(robot: DeltaRobot = DEFAULT_ROBOT) -> None:
    while True:
        print(MENU)
        choice = input("Select: ").strip().lower()
        if choice in ("q", "quit"):
            break
        elif choice == "1":
            demo_fk(robot)
        elif choice == "2":
            demo_ik(robot)
        elif choice == "3":
            plot_workspace(robot)
        elif choice == "4":
            animate_path(robot)
        elif choice == "5":
            roundtrip_test(robot)
        else:
            print("Unknown option.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    robot = DEFAULT_ROBOT

    if len(sys.argv) < 2:
        interactive_menu(robot)
    else:
        cmd = sys.argv[1].lower()
        if cmd == "workspace":
            plot_workspace(robot)
        elif cmd == "path":
            animate_path(robot)
        elif cmd == "roundtrip":
            roundtrip_test(robot)
        else:
            print(f"Unknown command '{cmd}'. Options: workspace, path, roundtrip")
            sys.exit(1)
