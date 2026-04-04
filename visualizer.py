"""
Interactive Delta Robot Kinematics Visualizer

Layout
------
  Left  : 3-D robot configuration — base triangle, active arms,
           parallel-link parallelograms, EE platform, force arrow.
  Right  : top-down XY drag panel (click and drag the EE),
           Z-height slider, force-vector sliders (Fx Fy Fz),
           live readout of joint angles and equilibrium torques.

Interaction
-----------
  • Click anywhere in the "XY Control" panel and drag to move the EE
    in the horizontal plane.
  • Use the Z slider to change EE height.
  • Use the Fx / Fy / Fz sliders to set the applied force vector.

Run
---
  python visualizer.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3-D projection

from kinematics import DeltaRobot, DeltaPositionError, LEG_DIRS, LEG_TANGS

# ---------------------------------------------------------------------------
# Default robot (matches mhp/delta-bot example, units: mm)
# ---------------------------------------------------------------------------
DEFAULT_ROBOT = DeltaRobot(
    servo_link_length=85.0,
    parallel_link_length=210.0,
    servo_displacement=72.0,
    effector_displacement=20.0,
)

# Visual half-width of each passive-link parallelogram (mm)
_PLINK_W = 5.0

# Colours
_C_BASE  = "#888888"
_C_ARM   = "#2979ff"   # active arms  — blue
_C_LINK  = "#ff9100"   # passive links — amber
_C_EE    = "#00e676"   # EE platform  — green
_C_EE_PT = "#ff1744"   # EE centre dot — red
_C_FORCE = "#ff1744"   # force arrow

_DOWN = np.array([0.0, 0.0, -1.0])


# ---------------------------------------------------------------------------
# Simulator class
# ---------------------------------------------------------------------------

class DeltaRobotSimulator:
    """Interactive matplotlib visualizer for a delta robot."""

    def __init__(self, robot: DeltaRobot = DEFAULT_ROBOT):
        self.robot = robot

        # State
        pos = robot.forward(20.0, 20.0, 20.0)
        self.ee = np.array(pos, dtype=float)
        self._last_valid = self.ee.copy()
        self.force = np.zeros(3)
        self._dragging = False
        self._trail_x: list[float] = []
        self._trail_y: list[float] = []

        self._build_figure()
        self._init_artists()
        self._connect_events()
        self._refresh()

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(15, 8))
        self.fig.suptitle("Delta Robot Kinematics Simulator", fontsize=13, y=0.99)

        # 3-D robot view (left ~57 %)
        self.ax3 = self.fig.add_axes([0.01, 0.06, 0.56, 0.90], projection="3d")
        self.ax3.set_title("Robot Configuration", fontsize=10, pad=6)
        self.ax3.set_xlabel("X (mm)", labelpad=2)
        self.ax3.set_ylabel("Y (mm)", labelpad=2)
        self.ax3.set_zlabel("Z (mm)", labelpad=2)

        # 2-D XY drag panel (top-right)
        self.ax2 = self.fig.add_axes([0.60, 0.56, 0.36, 0.36])
        self.ax2.set_title("XY Control — click & drag to move EE", fontsize=9)
        self.ax2.set_xlim(-65, 65)
        self.ax2.set_ylim(-65, 65)
        self.ax2.set_aspect("equal")
        self.ax2.set_xlabel("X (mm)", fontsize=8)
        self.ax2.set_ylabel("Y (mm)", fontsize=8)
        self.ax2.grid(True, alpha=0.25)

        # Sliders — laid out below the 2-D panel
        def _mkslider(y: float, label: str, lo: float, hi: float, v0: float) -> Slider:
            ax = self.fig.add_axes([0.64, y, 0.28, 0.026])
            return Slider(ax, label, lo, hi, valinit=v0, color="#607d8b")

        self.sl_z  = _mkslider(0.455, "Z (mm)",  -260, -80, float(self.ee[2]))
        self.sl_fx = _mkslider(0.365, "Fx (N)",   -30,  30, 0.0)
        self.sl_fy = _mkslider(0.295, "Fy (N)",   -30,  30, 0.0)
        self.sl_fz = _mkslider(0.225, "Fz (N)",   -30,  30, 0.0)
        for sl in (self.sl_z, self.sl_fx, self.sl_fy, self.sl_fz):
            sl.on_changed(self._on_slider)

        # Info text panel (bottom-right)
        self.ax_info = self.fig.add_axes([0.59, 0.01, 0.40, 0.19])
        self.ax_info.axis("off")
        self._info_txt = self.ax_info.text(
            0.0, 1.0, "",
            transform=self.ax_info.transAxes,
            va="top", fontsize=8.5, fontfamily="monospace",
        )

    # ------------------------------------------------------------------
    # Artist initialisation
    # ------------------------------------------------------------------

    def _init_artists(self) -> None:
        a = self.ax3
        kw2 = dict(lw=2)
        kw3 = dict(lw=3)

        # Base triangle: 3 edges
        self._base = [a.plot([], [], [], color=_C_BASE, **kw2)[0] for _ in range(3)]

        # Active arms: 1 line per leg
        self._arms = [a.plot([], [], [], color=_C_ARM, **kw3)[0] for _ in range(3)]

        # Passive links: 2 rods per leg (parallelogram), 6 lines total
        self._links = [a.plot([], [], [], color=_C_LINK, **kw2)[0] for _ in range(6)]

        # EE platform: 3 edges + centre point
        self._ee_edges = [a.plot([], [], [], color=_C_EE, **kw2)[0] for _ in range(3)]
        self._ee_pt,   = a.plot([], [], [], "o", color=_C_EE_PT, ms=9, zorder=6)

        # Force arrow — placeholder, created/removed in _refresh
        self._quiver = None

        # 2-D panel: workspace circle, crosshairs, EE dot, drag trail
        self.ax2.add_patch(
            plt.Circle((0, 0), 50, fill=False, ls="--", lw=1, color="gray", alpha=0.45)
        )
        self.ax2.axhline(0, color="gray", lw=0.5, alpha=0.3)
        self.ax2.axvline(0, color="gray", lw=0.5, alpha=0.3)
        self._ee2d,   = self.ax2.plot([], [], "o", color=_C_EE_PT, ms=11, zorder=5)
        self._trail2d, = self.ax2.plot([], [], "-", color=_C_EE_PT, lw=1, alpha=0.25)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _seg3(line, p: np.ndarray, q: np.ndarray) -> None:
        """Update a 3-D line segment from p to q in-place."""
        line.set_data([p[0], q[0]], [p[1], q[1]])
        line.set_3d_properties([p[2], q[2]])

    # ------------------------------------------------------------------
    # Main refresh
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        """Recompute geometry, update every artist, redraw."""
        try:
            geo = self.robot.geometry(*self.ee)
        except DeltaPositionError:
            self.ee[:] = self._last_valid
            return

        self._last_valid[:] = self.ee
        thetas  = geo["thetas"]
        servos  = geo["servos"]
        elbows  = geo["elbows"]
        anchors = geo["anchors"]

        # ---- torques -------------------------------------------------
        try:
            torques = self.robot.static_torques(*self.ee, self.force)
        except (DeltaPositionError, np.linalg.LinAlgError):
            torques = np.full(3, np.nan)

        # ---- 3-D artists ---------------------------------------------

        # Base
        for i in range(3):
            self._seg3(self._base[i], servos[i], servos[(i + 1) % 3])

        # Active arms
        for i in range(3):
            self._seg3(self._arms[i], servos[i], elbows[i])

        # Passive links: two parallel rods per leg, offset by ±_PLINK_W
        for i in range(3):
            t = LEG_TANGS[i]
            self._seg3(self._links[2 * i],     elbows[i] + _PLINK_W * t, anchors[i] + _PLINK_W * t)
            self._seg3(self._links[2 * i + 1], elbows[i] - _PLINK_W * t, anchors[i] - _PLINK_W * t)

        # EE platform
        for i in range(3):
            self._seg3(self._ee_edges[i], anchors[i], anchors[(i + 1) % 3])
        self._ee_pt.set_data([self.ee[0]], [self.ee[1]])
        self._ee_pt.set_3d_properties([self.ee[2]])

        # Force arrow — rebuild each frame (no clean set_data API for quiver3D)
        if self._quiver is not None:
            self._quiver.remove()
            self._quiver = None
        if np.linalg.norm(self.force) > 0.01:
            self._quiver = self.ax3.quiver(
                *self.ee, *(self.force * 4.0),  # 4 mm / N — display scale
                color=_C_FORCE, lw=2, arrow_length_ratio=0.25,
            )

        # Auto-scale 3-D axes
        pts = np.vstack([servos, elbows, anchors, self.ee[None, :]])
        hw = max(float(np.max(np.abs(pts[:, :2]))) + 20, 110)
        self.ax3.set_xlim(-hw, hw)
        self.ax3.set_ylim(-hw, hw)
        # Swap min/max so Z axis is inverted: base (Z=0) at bottom, EE (Z negative) above
        z_lo = float(np.min(pts[:, 2])) - 20
        z_hi = float(np.max(pts[:, 2])) + 30
        self.ax3.set_zlim(z_hi, z_lo)

        # ---- 2-D XY panel --------------------------------------------
        self._ee2d.set_data([self.ee[0]], [self.ee[1]])
        if self._dragging:
            self._trail_x.append(float(self.ee[0]))
            self._trail_y.append(float(self.ee[1]))
            if len(self._trail_x) > 120:
                self._trail_x.pop(0)
                self._trail_y.pop(0)
        self._trail2d.set_data(self._trail_x, self._trail_y)

        # ---- info text -----------------------------------------------
        def _ts(v: float) -> str:
            return f"{v:+9.2f}" if not np.isnan(v) else "       N/A"

        txt = (
            f"End-effector             Joint angles\n"
            f"  x = {self.ee[0]:+7.2f} mm        θ₁ = {thetas[0]:+7.3f}°\n"
            f"  y = {self.ee[1]:+7.2f} mm        θ₂ = {thetas[1]:+7.3f}°\n"
            f"  z = {self.ee[2]:+7.2f} mm        θ₃ = {thetas[2]:+7.3f}°\n"
            f"\n"
            f"Applied force (N)        Equilibrium torques (N·mm)\n"
            f"  Fx = {self.force[0]:+6.1f}             τ₁ = {_ts(torques[0])}\n"
            f"  Fy = {self.force[1]:+6.1f}             τ₂ = {_ts(torques[1])}\n"
            f"  Fz = {self.force[2]:+6.1f}             τ₃ = {_ts(torques[2])}\n"
        )
        self._info_txt.set_text(txt)

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_slider(self, _val) -> None:
        self.ee[2] = self.sl_z.val
        self.force[:] = [self.sl_fx.val, self.sl_fy.val, self.sl_fz.val]
        self._refresh()

    def _on_press(self, event) -> None:
        if event.inaxes is self.ax2 and event.button == 1:
            self._dragging = True
            self._trail_x.clear()
            self._trail_y.clear()
            self._move_xy(event)

    def _on_motion(self, event) -> None:
        if self._dragging and event.inaxes is self.ax2:
            self._move_xy(event)

    def _on_release(self, _event) -> None:
        self._dragging = False

    def _move_xy(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        candidate = self.ee.copy()
        candidate[0] = event.xdata
        candidate[1] = event.ydata
        try:
            self.robot.inverse(*candidate)   # reachability check
            self.ee[:] = candidate
        except DeltaPositionError:
            pass  # silently clamp to last valid position
        self._refresh()

    def _connect_events(self) -> None:
        c = self.fig.canvas.mpl_connect
        c("button_press_event",   self._on_press)
        c("motion_notify_event",  self._on_motion)
        c("button_release_event", self._on_release)

    # ------------------------------------------------------------------

    def run(self) -> None:
        plt.show()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sim = DeltaRobotSimulator()
    sim.run()
