"""
Interactive Delta Robot Kinematics Visualizer

Layout
------
  Left  : 3-D robot configuration — base triangle, active arms,
           parallel-link parallelograms, EE platform, force arrow.
  Right  : top-down XY drag panel (click and drag the EE),
           robot geometry sliders (rf, re, f, e, apex angle),
           pose + force sliders (Z, Fx, Fy, Fz),
           live readout of joint angles and equilibrium torques.

Run
---
  python visualizer.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from kinematics import DeltaRobot, DeltaPositionError

# ---------------------------------------------------------------------------
# Default robot (matches mhp/delta-bot example, units: mm)
# ---------------------------------------------------------------------------
DEFAULT_ROBOT = DeltaRobot(
    servo_link_length=85.0,
    parallel_link_length=210.0,
    servo_displacement=72.0,
    effector_displacement=20.0,
    apex_angle=60.0,
)

_PLINK_W = 5.0   # visual half-width of passive-link parallelogram (mm)
_DOWN    = np.array([0.0, 0.0, -1.0])

_C_BASE  = "#888888"
_C_ARM   = "#2979ff"
_C_LINK  = "#ff9100"
_C_EE    = "#00e676"
_C_EE_PT = "#ff1744"
_C_FORCE = "#ff1744"


# ---------------------------------------------------------------------------

class DeltaRobotSimulator:
    """Interactive matplotlib visualizer for a delta robot."""

    def __init__(self, robot: DeltaRobot = DEFAULT_ROBOT):
        self.robot = robot

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
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle("Delta Robot Kinematics Simulator", fontsize=13, y=0.99)

        # 3-D robot view (left ~57 %)
        self.ax3 = self.fig.add_axes([0.01, 0.05, 0.56, 0.92], projection="3d")
        self.ax3.set_title("Robot Configuration", fontsize=10, pad=6)
        self.ax3.set_xlabel("X (mm)", labelpad=2)
        self.ax3.set_ylabel("Y (mm)", labelpad=2)
        self.ax3.set_zlabel("Z (mm)", labelpad=2)
        self.ax3.set_xlim(-200, 200)
        self.ax3.set_ylim(-200, 200)
        self.ax3.set_zlim(50, -280)   # inverted: base at visual bottom

        # 2-D XY drag panel (top-right)
        self.ax2 = self.fig.add_axes([0.60, 0.67, 0.36, 0.28])
        self.ax2.set_title("XY Control — click & drag to move EE", fontsize=9)
        self.ax2.set_xlim(-65, 65)
        self.ax2.set_ylim(-65, 65)
        self.ax2.set_aspect("equal")
        self.ax2.set_xlabel("X (mm)", fontsize=8)
        self.ax2.set_ylabel("Y (mm)", fontsize=8)
        self.ax2.grid(True, alpha=0.25)

        # ------------------------------------------------------------------
        # Sliders
        # ------------------------------------------------------------------
        def _mkslider(y: float, label: str, lo: float, hi: float, v0: float,
                      color: str = "#607d8b") -> Slider:
            ax = self.fig.add_axes([0.64, y, 0.28, 0.022])
            return Slider(ax, label, lo, hi, valinit=v0, color=color)

        self.fig.text(0.605, 0.650, "Robot Geometry", fontsize=8,
                      fontweight="bold", color="#444444")
        self.fig.text(0.605, 0.430, "Pose & Applied Force", fontsize=8,
                      fontweight="bold", color="#444444")

        r = self.robot
        self.sl_rf   = _mkslider(0.617, "rf — active arm (mm)",   20, 200, r.rf,         "#5c6bc0")
        self.sl_re   = _mkslider(0.577, "re — passive link (mm)",  50, 400, r.re,         "#5c6bc0")
        self.sl_f    = _mkslider(0.537, "f  — base radius (mm)",   10, 200, r.f,          "#5c6bc0")
        self.sl_e    = _mkslider(0.497, "e  — EE radius (mm)",      5,  80, r.e,          "#5c6bc0")
        self.sl_apex = _mkslider(0.457, "apex angle (°)",          20, 160, r.apex_angle, "#5c6bc0")

        self.sl_z  = _mkslider(0.395, "Z (mm)",  -350, -50, float(self.ee[2]))
        self.sl_fx = _mkslider(0.353, "Fx (N)",   -30,  30, 0.0)
        self.sl_fy = _mkslider(0.311, "Fy (N)",   -30,  30, 0.0)
        self.sl_fz = _mkslider(0.269, "Fz (N)",   -30,  30, 0.0)

        for sl in (self.sl_rf, self.sl_re, self.sl_f, self.sl_e, self.sl_apex):
            sl.on_changed(self._on_geometry_slider)
        for sl in (self.sl_z, self.sl_fx, self.sl_fy, self.sl_fz):
            sl.on_changed(self._on_slider)

        # Info text panel
        self.ax_info = self.fig.add_axes([0.59, 0.01, 0.40, 0.24])
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

        self._base  = [a.plot([], [], [], color=_C_BASE,  **kw2)[0] for _ in range(3)]
        self._arms  = [a.plot([], [], [], color=_C_ARM,   **kw3)[0] for _ in range(3)]
        self._links = [a.plot([], [], [], color=_C_LINK,  **kw2)[0] for _ in range(6)]
        self._ee_edges = [a.plot([], [], [], color=_C_EE, **kw2)[0] for _ in range(3)]
        self._ee_pt,   = a.plot([], [], [], "o", color=_C_EE_PT, ms=9, zorder=6)
        self._quiver   = None

        self.ax2.add_patch(
            plt.Circle((0, 0), 50, fill=False, ls="--", lw=1, color="gray", alpha=0.45)
        )
        self.ax2.axhline(0, color="gray", lw=0.5, alpha=0.3)
        self.ax2.axvline(0, color="gray", lw=0.5, alpha=0.3)
        self._ee2d,    = self.ax2.plot([], [], "o", color=_C_EE_PT, ms=11, zorder=5)
        self._trail2d, = self.ax2.plot([], [], "-", color=_C_EE_PT, lw=1, alpha=0.25)

    # ------------------------------------------------------------------

    @staticmethod
    def _seg3(line, p: np.ndarray, q: np.ndarray) -> None:
        line.set_data([p[0], q[0]], [p[1], q[1]])
        line.set_3d_properties([p[2], q[2]])

    # ------------------------------------------------------------------
    # Main refresh
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
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
        tangs   = self.robot._leg_tangs   # use instance attribute, not global

        try:
            torques = self.robot.static_torques(*self.ee, self.force)
        except (DeltaPositionError, np.linalg.LinAlgError):
            torques = np.full(3, np.nan)

        # Base triangle
        for i in range(3):
            self._seg3(self._base[i], servos[i], servos[(i + 1) % 3])

        # Active arms
        for i in range(3):
            self._seg3(self._arms[i], servos[i], elbows[i])

        # Passive links (2 rods per leg)
        for i in range(3):
            t = tangs[i]
            self._seg3(self._links[2 * i],     elbows[i] + _PLINK_W * t, anchors[i] + _PLINK_W * t)
            self._seg3(self._links[2 * i + 1], elbows[i] - _PLINK_W * t, anchors[i] - _PLINK_W * t)

        # EE platform
        for i in range(3):
            self._seg3(self._ee_edges[i], anchors[i], anchors[(i + 1) % 3])
        self._ee_pt.set_data([self.ee[0]], [self.ee[1]])
        self._ee_pt.set_3d_properties([self.ee[2]])

        # Force arrow
        if self._quiver is not None:
            self._quiver.remove()
            self._quiver = None
        if np.linalg.norm(self.force) > 0.01:
            self._quiver = self.ax3.quiver(
                *self.ee, *(self.force * 4.0),
                color=_C_FORCE, lw=2, arrow_length_ratio=0.25,
            )

        # 2-D panel
        self._ee2d.set_data([self.ee[0]], [self.ee[1]])
        if self._dragging:
            self._trail_x.append(float(self.ee[0]))
            self._trail_y.append(float(self.ee[1]))
            if len(self._trail_x) > 120:
                self._trail_x.pop(0)
                self._trail_y.pop(0)
        self._trail2d.set_data(self._trail_x, self._trail_y)

        # Info text
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

    def _on_geometry_slider(self, _val) -> None:
        self.robot = DeltaRobot(
            servo_link_length=self.sl_rf.val,
            parallel_link_length=self.sl_re.val,
            servo_displacement=self.sl_f.val,
            effector_displacement=self.sl_e.val,
            apex_angle=self.sl_apex.val,
        )
        try:
            self.robot.inverse(*self.ee)
        except DeltaPositionError:
            fallback = self.robot.forward(20.0, 20.0, 20.0)
            if fallback is None:
                fallback = self.robot.forward(10.0, 10.0, 10.0)
            if fallback is not None:
                self.ee[:] = fallback
                self._last_valid[:] = fallback
                self.sl_z.eventson = False
                self.sl_z.set_val(float(self.ee[2]))
                self.sl_z.eventson = True
        self._refresh()

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
            self.robot.inverse(*candidate)
            self.ee[:] = candidate
        except DeltaPositionError:
            pass
        self._refresh()

    def _connect_events(self) -> None:
        c = self.fig.canvas.mpl_connect
        c("button_press_event",   self._on_press)
        c("motion_notify_event",  self._on_motion)
        c("button_release_event", self._on_release)

    def run(self) -> None:
        plt.show()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sim = DeltaRobotSimulator()
    sim.run()
