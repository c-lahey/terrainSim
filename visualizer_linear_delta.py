"""
Interactive Linear Delta Robot Kinematics Visualizer
=====================================================

Layout
------
  Left  : 3-D robot configuration — rails, carriages, passive arms,
           EE platform triangle, force arrow.
  Right  : front-view XZ drag panel (click and drag the EE in cross-section),
           robot geometry sliders (R, r, L),
           pose + force sliders (Y, Fx, Fy, Fz),
           live readout of carriage positions and equilibrium actuator forces.

Interaction
-----------
  • Click anywhere in the "XZ Control" panel and drag to move the EE
    in the cross-section plane (X and Z).
  • Use the Y slider to change EE depth along the rail axis.
  • Use the Fx / Fy / Fz sliders to set the applied force vector.

Run
---
  python visualizer_linear_delta.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from kinematics_linear_delta import (
    LinearDeltaRobot, LinearDeltaPositionError, LEG_DIRS_XZ, E_Y
)

# ---------------------------------------------------------------------------
# Default robot (mm)
# ---------------------------------------------------------------------------
DEFAULT_ROBOT = LinearDeltaRobot(
    rail_radius=150.0,
    platform_radius=30.0,
    link_length=280.0,
)

# Visual half-width of each passive-arm parallelogram (mm)
_ARM_W = 4.0

# Colours
_C_RAIL    = "#888888"   # rails
_C_CARR    = "#ff9100"   # carriages (amber)
_C_ARM     = "#2979ff"   # passive arms (blue)
_C_EE      = "#00e676"   # EE platform (green)
_C_EE_PT   = "#ff1744"   # EE centre (red)
_C_FORCE   = "#ff1744"   # force arrow


# ---------------------------------------------------------------------------
# Simulator class
# ---------------------------------------------------------------------------

class LinearDeltaSimulator:
    """Interactive matplotlib visualizer for a horizontal linear delta robot."""

    def __init__(self, robot: LinearDeltaRobot = DEFAULT_ROBOT):
        self.robot = robot

        # State — start at centre
        self.ee = np.zeros(3, dtype=float)
        # Find a valid starting position
        for y0 in [0.0, -50.0, 50.0]:
            try:
                self.robot.inverse(0.0, y0, 0.0)
                self.ee[:] = [0.0, y0, 0.0]
                break
            except LinearDeltaPositionError:
                pass

        self._last_valid = self.ee.copy()
        self.force = np.zeros(3)
        self._dragging = False
        self._trail_x: list[float] = []
        self._trail_z: list[float] = []

        self._build_figure()
        self._init_artists()
        self._connect_events()
        self._refresh()

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle("Linear Delta Robot Kinematics Simulator", fontsize=13, y=0.99)

        # 3-D robot view (left ~57 %)
        self.ax3 = self.fig.add_axes([0.01, 0.05, 0.56, 0.92], projection="3d")
        self.ax3.set_title("Robot Configuration", fontsize=10, pad=6)
        self.ax3.set_xlabel("X (mm)", labelpad=2)
        self.ax3.set_ylabel("Y (mm)", labelpad=2)
        self.ax3.set_zlabel("Z (mm)", labelpad=2)

        R, L = self.robot.R, self.robot.L
        lim_xz = R + 40
        lim_y  = L * 0.7
        self.ax3.set_xlim(-lim_xz, lim_xz)
        self.ax3.set_ylim(-lim_y,  lim_y)
        self.ax3.set_zlim(-lim_xz, lim_xz)
        self.ax3.view_init(elev=20, azim=-60)

        # 2-D XZ cross-section drag panel (top-right)
        self.ax2 = self.fig.add_axes([0.60, 0.67, 0.36, 0.28])
        self.ax2.set_title("XZ Control — click & drag to move EE", fontsize=9)
        _drag_lim = max(self.robot.R - self.robot.r - 10, 30)
        self.ax2.set_xlim(-_drag_lim, _drag_lim)
        self.ax2.set_ylim(-_drag_lim, _drag_lim)
        self.ax2.set_aspect("equal")
        self.ax2.set_xlabel("X (mm)", fontsize=8)
        self.ax2.set_ylabel("Z (mm)", fontsize=8)
        self.ax2.grid(True, alpha=0.25)

        # ------------------------------------------------------------------
        # Sliders
        # ------------------------------------------------------------------
        def _mkslider(y: float, label: str, lo: float, hi: float, v0: float,
                      color: str = "#607d8b") -> Slider:
            ax = self.fig.add_axes([0.64, y, 0.28, 0.022])
            return Slider(ax, label, lo, hi, valinit=v0, color=color)

        self.fig.text(0.605, 0.645, "Robot Geometry", fontsize=8,
                      fontweight="bold", color="#444444")
        self.fig.text(0.605, 0.455, "Pose & Applied Force", fontsize=8,
                      fontweight="bold", color="#444444")

        r = self.robot
        self.sl_R = _mkslider(0.610, "R — rail radius (mm)",  30, 300, r.R, "#5c6bc0")
        self.sl_r = _mkslider(0.568, "r — EE radius (mm)",     5, 100, r.r, "#5c6bc0")
        self.sl_L = _mkslider(0.526, "L — arm length (mm)",   50, 500, r.L, "#5c6bc0")

        self.sl_y  = _mkslider(0.420, "Y (mm)",   -200, 200, float(self.ee[1]))
        self.sl_fx = _mkslider(0.375, "Fx (N)",    -30,  30, 0.0)
        self.sl_fy = _mkslider(0.330, "Fy (N)",    -30,  30, 0.0)
        self.sl_fz = _mkslider(0.285, "Fz (N)",    -30,  30, 0.0)

        for sl in (self.sl_R, self.sl_r, self.sl_L):
            sl.on_changed(self._on_geometry_slider)
        for sl in (self.sl_y, self.sl_fx, self.sl_fy, self.sl_fz):
            sl.on_changed(self._on_slider)

        # Info text panel (bottom-right)
        self.ax_info = self.fig.add_axes([0.59, 0.01, 0.40, 0.25])
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

        # Rails: 1 dashed line per leg (along Y)
        self._rails = [
            a.plot([], [], [], color=_C_RAIL, lw=1.5, ls="--", alpha=0.6)[0]
            for _ in range(3)
        ]

        # Carriage markers: 1 per leg
        self._carrs = [
            a.plot([], [], [], "s", color=_C_CARR, ms=10, zorder=5)[0]
            for _ in range(3)
        ]

        # Passive arms: 2 rods per leg (offset ±_ARM_W in tangent direction)
        self._arms = [a.plot([], [], [], color=_C_ARM, **kw2)[0] for _ in range(6)]

        # EE platform: 3 edges + centre point
        self._ee_edges = [a.plot([], [], [], color=_C_EE, **kw2)[0] for _ in range(3)]
        self._ee_pt, = a.plot([], [], [], "o", color=_C_EE_PT, ms=9, zorder=6)

        # Force arrow
        self._quiver = None

        # 2-D panel: workspace circle, crosshairs, EE dot, trail
        _drag_lim = float(self.ax2.get_xlim()[1])
        self.ax2.add_patch(
            plt.Circle((0, 0), _drag_lim * 0.85, fill=False, ls="--",
                       lw=1, color="gray", alpha=0.45)
        )
        self.ax2.axhline(0, color="gray", lw=0.5, alpha=0.3)
        self.ax2.axvline(0, color="gray", lw=0.5, alpha=0.3)
        self._ee2d,    = self.ax2.plot([], [], "o", color=_C_EE_PT, ms=11, zorder=5)
        self._trail2d, = self.ax2.plot([], [], "-", color=_C_EE_PT, lw=1, alpha=0.25)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _seg3(line, p: np.ndarray, q: np.ndarray) -> None:
        line.set_data([p[0], q[0]], [p[1], q[1]])
        line.set_3d_properties([p[2], q[2]])

    def _arm_tangent(self, i: int) -> np.ndarray:
        """Unit tangent perpendicular to the arm vector, in XZ plane."""
        dx = LEG_DIRS_XZ[i, 0]
        dz = LEG_DIRS_XZ[i, 1]
        # Tangent = CCW 90° of outward XZ direction, as 3D vector
        return np.array([-dz, 0.0, dx])

    # ------------------------------------------------------------------
    # Main refresh
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        """Recompute geometry, update every artist, redraw."""
        try:
            geo = self.robot.geometry(*self.ee)
        except LinearDeltaPositionError:
            self.ee[:] = self._last_valid
            return

        self._last_valid[:] = self.ee
        rhos        = geo["rhos"]
        carriages   = geo["carriages"]
        platform_pts = geo["platform_pts"]

        # ---- actuator forces -----------------------------------------
        try:
            forces = self.robot.static_forces(*self.ee, self.force)
        except (LinearDeltaPositionError, np.linalg.LinAlgError):
            forces = np.full(3, np.nan)

        # ---- 3-D axis limits (update when geometry changes) ----------
        R, L = self.robot.R, self.robot.L
        lim_xz = R + 40
        lim_y  = L * 0.7
        self.ax3.set_xlim(-lim_xz, lim_xz)
        self.ax3.set_ylim(-lim_y,  lim_y)
        self.ax3.set_zlim(-lim_xz, lim_xz)

        # ---- Rails ---------------------------------------------------
        for i in range(3):
            rx = self.robot.rail_xz[i, 0]
            rz = self.robot.rail_xz[i, 1]
            p = np.array([rx, -lim_y, rz])
            q = np.array([rx,  lim_y, rz])
            self._seg3(self._rails[i], p, q)

        # ---- Carriages -----------------------------------------------
        for i in range(3):
            c = carriages[i]
            self._carrs[i].set_data([c[0]], [c[1]])
            self._carrs[i].set_3d_properties([c[2]])

        # ---- Passive arms (2 rods per leg) ---------------------------
        for i in range(3):
            t = self._arm_tangent(i)
            p = carriages[i]
            q = platform_pts[i]
            self._seg3(self._arms[2 * i],     p + _ARM_W * t, q + _ARM_W * t)
            self._seg3(self._arms[2 * i + 1], p - _ARM_W * t, q - _ARM_W * t)

        # ---- EE platform ---------------------------------------------
        for i in range(3):
            self._seg3(self._ee_edges[i], platform_pts[i], platform_pts[(i + 1) % 3])
        self._ee_pt.set_data([self.ee[0]], [self.ee[1]])
        self._ee_pt.set_3d_properties([self.ee[2]])

        # ---- Force arrow ---------------------------------------------
        if self._quiver is not None:
            self._quiver.remove()
            self._quiver = None
        if np.linalg.norm(self.force) > 0.01:
            self._quiver = self.ax3.quiver(
                *self.ee, *(self.force * 4.0),
                color=_C_FORCE, lw=2, arrow_length_ratio=0.25,
            )

        # ---- 2-D XZ panel -------------------------------------------
        self._ee2d.set_data([self.ee[0]], [self.ee[2]])
        if self._dragging:
            self._trail_x.append(float(self.ee[0]))
            self._trail_z.append(float(self.ee[2]))
            if len(self._trail_x) > 120:
                self._trail_x.pop(0)
                self._trail_z.pop(0)
        self._trail2d.set_data(self._trail_x, self._trail_z)

        # ---- Info text ----------------------------------------------
        def _fs(v: float) -> str:
            return f"{v:+9.2f}" if not np.isnan(v) else "       N/A"

        txt = (
            f"End-effector             Carriage positions\n"
            f"  x = {self.ee[0]:+7.2f} mm        ρ₁ = {rhos[0]:+9.3f} mm\n"
            f"  y = {self.ee[1]:+7.2f} mm        ρ₂ = {rhos[1]:+9.3f} mm\n"
            f"  z = {self.ee[2]:+7.2f} mm        ρ₃ = {rhos[2]:+9.3f} mm\n"
            f"\n"
            f"Applied force (N)        Actuator forces (N)\n"
            f"  Fx = {self.force[0]:+6.1f}             f₁ = {_fs(forces[0])}\n"
            f"  Fy = {self.force[1]:+6.1f}             f₂ = {_fs(forces[1])}\n"
            f"  Fz = {self.force[2]:+6.1f}             f₃ = {_fs(forces[2])}\n"
        )
        self._info_txt.set_text(txt)

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_geometry_slider(self, _val) -> None:
        """Rebuild robot with new parameters, re-validate EE position."""
        self.robot = LinearDeltaRobot(
            rail_radius=self.sl_R.val,
            platform_radius=self.sl_r.val,
            link_length=self.sl_L.val,
        )
        # Update XZ drag panel limits
        _drag_lim = max(self.robot.R - self.robot.r - 10, 20)
        self.ax2.set_xlim(-_drag_lim, _drag_lim)
        self.ax2.set_ylim(-_drag_lim, _drag_lim)

        try:
            self.robot.inverse(*self.ee)
        except LinearDeltaPositionError:
            # Snap to origin at current Y
            for y0 in [self.ee[1], 0.0, -50.0, 50.0]:
                try:
                    self.robot.inverse(0.0, y0, 0.0)
                    self.ee[:] = [0.0, y0, 0.0]
                    self._last_valid[:] = self.ee
                    self.sl_y.eventson = False
                    self.sl_y.set_val(float(self.ee[1]))
                    self.sl_y.eventson = True
                    break
                except LinearDeltaPositionError:
                    pass
        self._refresh()

    def _on_slider(self, _val) -> None:
        self.ee[1] = self.sl_y.val
        self.force[:] = [self.sl_fx.val, self.sl_fy.val, self.sl_fz.val]
        self._refresh()

    def _on_press(self, event) -> None:
        if event.inaxes is self.ax2 and event.button == 1:
            self._dragging = True
            self._trail_x.clear()
            self._trail_z.clear()
            self._move_xz(event)

    def _on_motion(self, event) -> None:
        if self._dragging and event.inaxes is self.ax2:
            self._move_xz(event)

    def _on_release(self, _event) -> None:
        self._dragging = False

    def _move_xz(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        candidate = self.ee.copy()
        candidate[0] = event.xdata   # X from horizontal axis
        candidate[2] = event.ydata   # Z from vertical axis
        try:
            self.robot.inverse(*candidate)
            self.ee[:] = candidate
        except LinearDeltaPositionError:
            pass
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
    sim = LinearDeltaSimulator()
    sim.run()
