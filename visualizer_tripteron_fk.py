from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from kinematics_tripteron_fk import TripteronFK, TripteronKinematicsError


DEFAULT_ROBOT = TripteronFK("tripteron_extracted_model.json")

_C_ACT = "#7e57c2"
_C_ARM_A = "#2979ff"
_C_ARM_B = "#00acc1"
_C_ARM_C = "#ff9100"
_C_EE = "#00e676"
_C_EE_PT = "#ff1744"
_C_GROUND = "#666666"


class TripteronFKSimulator:
    def __init__(self, robot: TripteronFK = DEFAULT_ROBOT):
        self.robot = robot

        self.sA = self.robot.legA.slider_value_ref
        self.sB = self.robot.legB.slider_value_ref
        self.sC = self.robot.legC.slider_value_ref

        self._build_figure()
        self._init_artists()
        self._refresh()

    # ------------------------------------------------------------------
    def _build_figure(self):
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle("Tripteron Forward Kinematics Simulator", fontsize=13, y=0.99)

        self.ax3 = self.fig.add_axes([0.01, 0.05, 0.56, 0.92], projection="3d")
        self.ax3.set_title("Robot Configuration", fontsize=10, pad=6)
        self.ax3.set_xlabel("X")
        self.ax3.set_ylabel("Y")
        self.ax3.set_zlabel("Z")

        self.ax2 = self.fig.add_axes([0.60, 0.67, 0.36, 0.28])
        self.ax2.set_title("EE Output (XY projection)", fontsize=9)
        self.ax2.set_xlim(-20, 180)
        self.ax2.set_ylim(-60, 100)
        self.ax2.set_aspect("equal")
        self.ax2.set_xlabel("X", fontsize=8)
        self.ax2.set_ylabel("Y", fontsize=8)
        self.ax2.grid(True, alpha=0.25)

        def _mkslider(y, label, lo, hi, v0, color="#607d8b"):
            ax = self.fig.add_axes([0.64, y, 0.28, 0.022])
            return Slider(ax, label, lo, hi, valinit=v0, color=color)

        self.fig.text(0.605, 0.61, "Actuator Inputs", fontsize=8,
                      fontweight="bold", color="#444444")

        self.sl_sA = _mkslider(0.565, "Actuator A", self.sA - 40.0, self.sA + 40.0, self.sA, "#7e57c2")
        self.sl_sB = _mkslider(0.525, "Actuator B", self.sB - 40.0, self.sB + 40.0, self.sB, "#7e57c2")
        self.sl_sC = _mkslider(0.485, "Actuator C", self.sC - 40.0, self.sC + 40.0, self.sC, "#7e57c2")

        self.sl_sA.on_changed(self._on_slider)
        self.sl_sB.on_changed(self._on_slider)
        self.sl_sC.on_changed(self._on_slider)

        ax_reset = self.fig.add_axes([0.72, 0.42, 0.12, 0.045])
        self.btn_reset = Button(ax_reset, "Reset")
        self.btn_reset.on_clicked(self._on_reset)

        self.ax_info = self.fig.add_axes([0.59, 0.01, 0.40, 0.36])
        self.ax_info.axis("off")
        self._info_txt = self.ax_info.text(
            0.0,
            1.0,
            "",
            transform=self.ax_info.transAxes,
            va="top",
            fontsize=8.5,
            fontfamily="monospace",
        )

    # ------------------------------------------------------------------
    def _init_artists(self):
        a = self.ax3
        kw2 = dict(lw=2)
        kw3 = dict(lw=3)

        self._actuators = [a.plot([], [], [], color=_C_ACT, **kw3)[0] for _ in range(3)]

        self._A = [a.plot([], [], [], color=_C_ARM_A, **kw2)[0] for _ in range(3)]
        self._B = [a.plot([], [], [], color=_C_ARM_B, **kw2)[0] for _ in range(3)]
        self._C = [a.plot([], [], [], color=_C_ARM_C, **kw2)[0] for _ in range(6)]

        self._ee_edges = [a.plot([], [], [], color=_C_EE, **kw2)[0] for _ in range(4)]
        self._ee_pt, = a.plot([], [], [], "o", color=_C_EE_PT, ms=9, zorder=6)

        self._ground_pts = a.scatter([], [], [], color=_C_GROUND, s=60, marker="s")

        self.ax2.axhline(0, color="gray", lw=0.5, alpha=0.3)
        self.ax2.axvline(0, color="gray", lw=0.5, alpha=0.3)
        self._ee2d, = self.ax2.plot([], [], "o", color=_C_EE_PT, ms=11, zorder=5)
        self._trail2d, = self.ax2.plot([], [], "-", color=_C_EE_PT, lw=1, alpha=0.25)

        self._trail_x = []
        self._trail_y = []

    # ------------------------------------------------------------------
    @staticmethod
    def _seg3(line, p, q):
        line.set_data([p[0], q[0]], [p[1], q[1]])
        line.set_3d_properties([p[2], q[2]])

    # ------------------------------------------------------------------
    def _refresh(self):
        try:
            geo = self.robot.forward(self.sA, self.sB, self.sC)
        except TripteronKinematicsError as e:
            print("FK solve failed:", e)
            return

        ee = np.array(geo["ee_center"], dtype=float)
        ee_pts = np.array(geo["ee_points"], dtype=float)

        A = geo["A"]
        B = geo["B"]
        C = geo["C"]
        anchors = np.array(geo["ground_points"], dtype=float)

        self._ground_pts._offsets3d = (anchors[:, 0], anchors[:, 1], anchors[:, 2])

        self._seg3(self._actuators[0], anchors[0], A["slider"])
        self._seg3(self._actuators[1], anchors[1], B["slider"])
        self._seg3(self._actuators[2], anchors[2], C["slider"])

        self._seg3(self._A[0], A["slider"], A["base"])
        self._seg3(self._A[1], A["base"], A["elbow"])
        self._seg3(self._A[2], A["elbow"], A["ee"])

        self._seg3(self._B[0], B["slider"], B["base"])
        self._seg3(self._B[1], B["base"], B["elbow"])
        self._seg3(self._B[2], B["elbow"], B["ee"])

        self._seg3(self._C[0], C["slider"], C["base1"])
        self._seg3(self._C[1], C["base1"], C["elbow1"])
        self._seg3(self._C[2], C["elbow1"], C["ee1"])
        self._seg3(self._C[3], C["slider"], C["base2"])
        self._seg3(self._C[4], C["base2"], C["elbow2"])
        self._seg3(self._C[5], C["elbow2"], C["ee2"])

        self._seg3(self._ee_edges[0], ee_pts[0], ee_pts[1])
        self._seg3(self._ee_edges[1], ee_pts[1], ee_pts[2])
        self._seg3(self._ee_edges[2], ee_pts[2], ee_pts[3])
        self._seg3(self._ee_edges[3], ee_pts[3], ee_pts[0])

        self._ee_pt.set_data([ee[0]], [ee[1]])
        self._ee_pt.set_3d_properties([ee[2]])

        self._ee2d.set_data([ee[0]], [ee[1]])
        self._trail_x.append(float(ee[0]))
        self._trail_y.append(float(ee[1]))
        if len(self._trail_x) > 200:
            self._trail_x.pop(0)
            self._trail_y.pop(0)
        self._trail2d.set_data(self._trail_x, self._trail_y)

        txt = (
            f"Forward kinematics mode\n\n"
            f"Actuator inputs\n"
            f" s1 = {self.sA:+9.3f}\n"
            f" s2 = {self.sB:+9.3f}\n"
            f" s3 = {self.sC:+9.3f}\n\n"
            f"EE output\n"
            f" x = {ee[0]:+9.3f}\n"
            f" y = {ee[1]:+9.3f}\n"
            f" z = {ee[2]:+9.3f}\n"
        )
        self._info_txt.set_text(txt)

        all_pts = np.vstack([
            anchors,
            A["slider"], A["base"], A["elbow"], A["ee"],
            B["slider"], B["base"], B["elbow"], B["ee"],
            C["slider"], C["base1"], C["elbow1"], C["ee1"],
            C["base2"], C["elbow2"], C["ee2"],
            ee
        ])

        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        ctr = 0.5 * (mins + maxs)
        rng = max((maxs - mins).max(), 1.0) * 0.65

        self.ax3.set_xlim(ctr[0] - rng, ctr[0] + rng)
        self.ax3.set_ylim(ctr[1] - rng, ctr[1] + rng)
        self.ax3.set_zlim(ctr[2] - rng, ctr[2] + rng)

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _on_slider(self, _val):
        self.sA = self.sl_sA.val
        self.sB = self.sl_sB.val
        self.sC = self.sl_sC.val
        self._refresh()

    def _on_reset(self, _event):
        self.sA = self.robot.legA.slider_value_ref
        self.sB = self.robot.legB.slider_value_ref
        self.sC = self.robot.legC.slider_value_ref

        self.sl_sA.set_val(self.sA)
        self.sl_sB.set_val(self.sB)
        self.sl_sC.set_val(self.sC)

        self._trail_x.clear()
        self._trail_y.clear()
        self._refresh()

    def run(self):
        plt.show()


if __name__ == "__main__":
    sim = TripteronFKSimulator()
    sim.run()