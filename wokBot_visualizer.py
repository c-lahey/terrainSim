"""
wokbot_visualizer.py

Interactive visualizer for the Fig. 7 2T1R cooking-robot mechanism.

Adds:
    - branch-continuous IK selection
    - deterministic branch ordering
    - condition-number display
    - sliders for pose
    - sliders for link lengths / geometry parameters
    - reset-pose and reset-geometry buttons

Expected local module:
    from wokBot_sim import Cooking2T1RKinematics, Cooking2T1RParams

If your class lives in another file, change the import below.
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons

# Change this import if your kinematics file has a different name.
from wokBot_sim import Cooking2T1RKinematics, Cooking2T1RParams


class WokBotVisualizer:
    def __init__(self):
        self.default_params = Cooking2T1RParams()
        self.kin = Cooking2T1RKinematics(self.default_params)

        self.pose = np.array([150.0, -75.0, math.radians(50.0)], dtype=float)

        self.solutions = []
        self.branch_idx = 0
        self.show_all = False
        self.q_current = None

        self._refresh_solutions()

        if self.solutions:
            self.q_current = self.solutions[self.branch_idx].q.copy()

        self.fig = plt.figure(figsize=(15, 9))

        # Main mechanism plot
        self.ax = self.fig.add_axes([0.06, 0.30, 0.58, 0.65])

        # Pose sliders
        self.ax_x = self.fig.add_axes([0.08, 0.21, 0.50, 0.025])
        self.ax_y = self.fig.add_axes([0.08, 0.17, 0.50, 0.025])
        self.ax_beta = self.fig.add_axes([0.08, 0.13, 0.50, 0.025])
        self.ax_branch = self.fig.add_axes([0.08, 0.09, 0.50, 0.025])

        self.slider_x = Slider(self.ax_x, "C1 x [mm]", 100.0, 280.0, valinit=self.pose[0])
        self.slider_y = Slider(self.ax_y, "C1 y [mm]", -100.0, -50.0, valinit=self.pose[1])
        self.slider_beta = Slider(
            self.ax_beta,
            "platform beta [deg]",
            30.0,
            80.0,
            valinit=math.degrees(self.pose[2]),
        )
        self.slider_branch = Slider(
            self.ax_branch,
            "IK branch idx",
            0,
            max(0, len(self.solutions) - 1),
            valinit=self.branch_idx,
            valstep=1,
        )

        # Buttons / checkboxes
        self.ax_prev = self.fig.add_axes([0.68, 0.88, 0.06, 0.04])
        self.ax_next = self.fig.add_axes([0.75, 0.88, 0.06, 0.04])
        self.ax_reset_pose = self.fig.add_axes([0.82, 0.88, 0.13, 0.04])
        self.ax_reset_geom = self.fig.add_axes([0.82, 0.82, 0.13, 0.04])
        self.ax_check = self.fig.add_axes([0.68, 0.78, 0.20, 0.08])

        self.btn_prev = Button(self.ax_prev, "Prev")
        self.btn_next = Button(self.ax_next, "Next")
        self.btn_reset_pose = Button(self.ax_reset_pose, "Reset Pose")
        self.btn_reset_geom = Button(self.ax_reset_geom, "Reset Geometry")
        self.check = CheckButtons(self.ax_check, ["Show all branches"], [self.show_all])

        # Geometry sliders, right side
        self.geom_sliders = {}
        self._create_geometry_sliders()

        # Pose callbacks
        self.slider_x.on_changed(self._on_pose_change)
        self.slider_y.on_changed(self._on_pose_change)
        self.slider_beta.on_changed(self._on_pose_change)
        self.slider_branch.on_changed(self._on_branch_change)

        # Button callbacks
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.btn_reset_pose.on_clicked(self._on_reset_pose)
        self.btn_reset_geom.on_clicked(self._on_reset_geometry)
        self.check.on_clicked(self._on_check)

        self._draw()

    # ------------------------------------------------------------------
    # Geometry slider setup
    # ------------------------------------------------------------------

    def _create_geometry_sliders(self):
        """
        Add link/base geometry sliders.

        Labels are intentionally descriptive:
            a1, b1: mostly set C1 reach from A1 side
            a2, b2, c2: five-bar closure / C2 behavior
            a3, b3, c3: orientation limb / C3 and beta behavior
            e*, ep*: base pivot layout
        """
        slider_specs = [
            # name, label, min, max, initial
            ("a1", "a1 active link A1-B1 | C1 reach", 40.0, 220.0, self.kin.p.a1),
            ("b1", "b1 passive B1-C1 | C1 reach", 60.0, 280.0, self.kin.p.b1),

            ("a2", "a2 active link A2-B2 | five-bar", 5.0, 100.0, self.kin.p.a2),
            ("b2", "b2 passive B2-C2 | five-bar", 30.0, 200.0, self.kin.p.b2),
            ("c2", "c2 C1-to-C2 offset | five-bar", 0.0, 160.0, self.kin.p.c2),

            ("a3", "a3 active link A3-B3 | beta limb", 30.0, 180.0, self.kin.p.a3),
            ("b3", "b3 passive B3-C3 | beta limb", 80.0, 360.0, self.kin.p.b3),
            ("c3", "c3 platform C1-C3 length | beta lever", 10.0, 160.0, self.kin.p.c3),

            ("e1", "e1 A1 x base location", -80.0, 180.0, self.kin.p.e1),
            ("ep1", "ep1 A1 y base location", -80.0, 140.0, self.kin.p.ep1),
            ("e3", "e3 A3 x base location", -80.0, 180.0, self.kin.p.e3),
            ("ep3", "ep3 A3 y base location", -80.0, 180.0, self.kin.p.ep3),
        ]

        x0 = 0.69
        w = 0.27
        h = 0.022
        y_top = 0.70
        dy = 0.045

        for i, (name, label, vmin, vmax, v0) in enumerate(slider_specs):
            ax_slider = self.fig.add_axes([x0, y_top - i * dy, w, h])
            slider = Slider(ax_slider, label, vmin, vmax, valinit=v0)
            slider.on_changed(self._on_geometry_change)
            self.geom_sliders[name] = slider

    def _read_params_from_sliders(self) -> Cooking2T1RParams:
        return Cooking2T1RParams(
            a1=self.geom_sliders["a1"].val,
            a2=self.geom_sliders["a2"].val,
            a3=self.geom_sliders["a3"].val,
            b1=self.geom_sliders["b1"].val,
            b2=self.geom_sliders["b2"].val,
            b3=self.geom_sliders["b3"].val,
            c1=0.0,
            c2=self.geom_sliders["c2"].val,
            c3=self.geom_sliders["c3"].val,
            e1=self.geom_sliders["e1"].val,
            ep1=self.geom_sliders["ep1"].val,
            e3=self.geom_sliders["e3"].val,
            ep3=self.geom_sliders["ep3"].val,
        )

    # ------------------------------------------------------------------
    # Branch continuity
    # ------------------------------------------------------------------

    def _angle_distance(self, q1, q2) -> float:
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)
        dq = self.kin.wrap_vec_to_pi(q1 - q2)
        return float(np.linalg.norm(dq))

    def _refresh_solutions(self):
        sols = [s for s in self.kin.ik_all(self.pose) if s.valid]

        # Deterministic ordering by branch sign.
        sols.sort(key=lambda s: (s.branch[0], s.branch[1], s.branch[2]))

        self.solutions = sols

        if len(self.solutions) == 0:
            self.branch_idx = 0
            self.q_current = None
            return

        if self.q_current is not None:
            distances = [self._angle_distance(sol.q, self.q_current) for sol in self.solutions]
            self.branch_idx = int(np.argmin(distances))
        else:
            self.branch_idx = int(np.clip(self.branch_idx, 0, len(self.solutions) - 1))

        self.q_current = self.solutions[self.branch_idx].q.copy()

    # ------------------------------------------------------------------
    # UI callbacks
    # ------------------------------------------------------------------

    def _on_pose_change(self, _val):
        self.pose = np.array(
            [
                self.slider_x.val,
                self.slider_y.val,
                math.radians(self.slider_beta.val),
            ],
            dtype=float,
        )

        self._refresh_solutions()
        self._sync_branch_slider()
        self._draw()

    def _on_geometry_change(self, _val):
        """
        Rebuild the kinematics object using current slider values.

        Keep q_current if possible, so the visualizer continues on the
        nearest assembly mode after a small geometry adjustment.
        """
        self.kin = Cooking2T1RKinematics(self._read_params_from_sliders())
        self._refresh_solutions()
        self._sync_branch_slider()
        self._draw()

    def _sync_branch_slider(self):
        self.slider_branch.valmax = max(0, len(self.solutions) - 1)
        self.slider_branch.ax.set_xlim(
            self.slider_branch.valmin,
            self.slider_branch.valmax + 1e-9,
        )

        self.slider_branch.eventson = False
        self.slider_branch.set_val(self.branch_idx)
        self.slider_branch.eventson = True

    def _on_branch_change(self, val):
        if len(self.solutions) == 0:
            self.branch_idx = 0
            self.q_current = None
            self._draw()
            return

        self.branch_idx = int(np.clip(int(val), 0, len(self.solutions) - 1))
        self.q_current = self.solutions[self.branch_idx].q.copy()
        self._draw()

    def _on_prev(self, _event):
        if len(self.solutions) == 0:
            return

        self.branch_idx = (self.branch_idx - 1) % len(self.solutions)
        self.q_current = self.solutions[self.branch_idx].q.copy()
        self._sync_branch_slider()
        self._draw()

    def _on_next(self, _event):
        if len(self.solutions) == 0:
            return

        self.branch_idx = (self.branch_idx + 1) % len(self.solutions)
        self.q_current = self.solutions[self.branch_idx].q.copy()
        self._sync_branch_slider()
        self._draw()

    def _on_reset_pose(self, _event):
        self.q_current = None
        self.slider_x.set_val(150.0)
        self.slider_y.set_val(-75.0)
        self.slider_beta.set_val(50.0)

    def _on_reset_geometry(self, _event):
        self.q_current = None

        defaults = self.default_params
        default_values = {
            "a1": defaults.a1,
            "a2": defaults.a2,
            "a3": defaults.a3,
            "b1": defaults.b1,
            "b2": defaults.b2,
            "b3": defaults.b3,
            "c2": defaults.c2,
            "c3": defaults.c3,
            "e1": defaults.e1,
            "ep1": defaults.ep1,
            "e3": defaults.e3,
            "ep3": defaults.ep3,
        }

        # Suppress redraws until all sliders are reset.
        for name, slider in self.geom_sliders.items():
            slider.eventson = False
            slider.set_val(default_values[name])
            slider.eventson = True

        self.kin = Cooking2T1RKinematics(self._read_params_from_sliders())
        self._refresh_solutions()
        self._sync_branch_slider()
        self._draw()

    def _on_check(self, _label):
        self.show_all = not self.show_all
        self._draw()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _mechanism_points(self, q: np.ndarray):
        kin = self.kin
        theta1, theta2, theta3 = q

        A1 = kin.A1
        A2 = kin.A2
        A3 = kin.A3

        B1 = kin.B1(theta1)
        B2 = kin.B2(theta2)
        B3 = kin.B3(theta3)

        C1 = kin.C1_from_pose(self.pose)
        C3 = kin.C3_from_pose(self.pose)
        C2 = kin.C2_from_C1_B1(C1, B1)

        return {
            "A1": A1,
            "A2": A2,
            "A3": A3,
            "B1": B1,
            "B2": B2,
            "B3": B3,
            "C1": C1,
            "C2": C2,
            "C3": C3,
        }

    @staticmethod
    def _plot_segment(ax, P, Q, **kwargs):
        ax.plot([P[0], Q[0]], [P[1], Q[1]], **kwargs)

    def _condition_number(self, q: np.ndarray) -> float:
        try:
            J = self.kin.numerical_jacobian_dq_dp(self.pose, q_prev=q)
            return float(np.linalg.cond(J))
        except Exception:
            return float("nan")

    def _draw_one_solution(
        self,
        ax,
        q: np.ndarray,
        alpha: float = 1.0,
        linewidth: float = 2.0,
        show_labels: bool = True,
    ):
        pts = self._mechanism_points(q)

        A1, A2, A3 = pts["A1"], pts["A2"], pts["A3"]
        B1, B2, B3 = pts["B1"], pts["B2"], pts["B3"]
        C1, C2, C3 = pts["C1"], pts["C2"], pts["C3"]

        # Base anchors
        ax.scatter(
            [A1[0], A2[0], A3[0]],
            [A1[1], A2[1], A3[1]],
            s=70,
            marker="s",
            alpha=alpha,
        )

        # Active links
        self._plot_segment(ax, A1, B1, linewidth=linewidth, alpha=alpha)
        self._plot_segment(ax, A2, B2, linewidth=linewidth, alpha=alpha)
        self._plot_segment(ax, A3, B3, linewidth=linewidth, alpha=alpha)

        # Passive / loop-closure links
        self._plot_segment(ax, B1, C1, linewidth=linewidth, alpha=alpha)
        self._plot_segment(ax, B1, C2, linewidth=linewidth, alpha=alpha, linestyle="--")
        self._plot_segment(ax, B2, C2, linewidth=linewidth, alpha=alpha)
        self._plot_segment(ax, B3, C3, linewidth=linewidth, alpha=alpha)

        # Moving-platform/coupler lines
        self._plot_segment(ax, C3, C1, linewidth=linewidth + 0.7, alpha=alpha)
        self._plot_segment(ax, C2, C1, linewidth=1.2, alpha=alpha, linestyle=":")

        # Points
        ax.scatter(
            [B1[0], B2[0], B3[0]],
            [B1[1], B2[1], B3[1]],
            s=45,
            alpha=alpha,
        )

        ax.scatter(
            [C1[0], C2[0], C3[0]],
            [C1[1], C2[1], C3[1]],
            s=55,
            alpha=alpha,
        )

        # Platform orientation arrow
        beta = self.pose[2]
        arrow_len = 30.0
        ax.arrow(
            C1[0],
            C1[1],
            arrow_len * math.cos(beta),
            arrow_len * math.sin(beta),
            head_width=6.0,
            length_includes_head=True,
            alpha=alpha,
        )

        if show_labels:
            text_alpha = max(0.35, alpha)

            ax.text(A1[0], A1[1], "  A1", alpha=text_alpha)
            ax.text(A2[0], A2[1], "  A2", alpha=text_alpha)
            ax.text(A3[0], A3[1], "  A3", alpha=text_alpha)

            ax.text(B1[0], B1[1], "  B1", alpha=text_alpha)
            ax.text(B2[0], B2[1], "  B2", alpha=text_alpha)
            ax.text(B3[0], B3[1], "  B3", alpha=text_alpha)

            ax.text(C1[0], C1[1], "  C1", alpha=text_alpha)
            ax.text(C2[0], C2[1], "  C2", alpha=text_alpha)
            ax.text(C3[0], C3[1], "  C3", alpha=text_alpha)

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def _draw(self):
        self.ax.clear()

        ax = self.ax
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)

        # Prescribed workspace rectangle from paper.
        rect_x = [100, 280, 280, 100, 100]
        rect_y = [-100, -100, -50, -50, -100]
        ax.plot(rect_x, rect_y, linestyle=":", linewidth=1.5)
        ax.text(102, -48, "prescribed workspace", fontsize=9)

        if self.show_all and len(self.solutions) > 0:
            for sol in self.solutions:
                self._draw_one_solution(
                    ax,
                    sol.q,
                    alpha=0.25,
                    linewidth=1.3,
                    show_labels=False,
                )

        if len(self.solutions) > 0:
            sol = self.solutions[self.branch_idx]
            self._draw_one_solution(
                ax,
                sol.q,
                alpha=1.0,
                linewidth=2.5,
                show_labels=True,
            )

            q_deg = np.degrees(sol.q)
            condJ = self._condition_number(sol.q)

            title = (
                f"Fig. 7 2T1R visualizer | "
                f"pose = [x={self.pose[0]:.1f}, y={self.pose[1]:.1f}, "
                f"beta={math.degrees(self.pose[2]):.1f} deg]\n"
                f"branch {self.branch_idx + 1}/{len(self.solutions)} | "
                f"signs = {sol.branch} | "
                f"q = [{q_deg[0]:.2f}, {q_deg[1]:.2f}, {q_deg[2]:.2f}] deg | "
                f"residual = {sol.residual_norm:.2e} | "
                f"cond(J) = {condJ:.2e}"
            )
        else:
            C1 = self.kin.C1_from_pose(self.pose)
            C3 = self.kin.C3_from_pose(self.pose)

            ax.scatter([C1[0], C3[0]], [C1[1], C3[1]], s=60)
            self._plot_segment(ax, C3, C1, linewidth=2.0)

            title = (
                f"UNREACHABLE pose | "
                f"pose = [x={self.pose[0]:.1f}, y={self.pose[1]:.1f}, "
                f"beta={math.degrees(self.pose[2]):.1f} deg]"
            )

        ax.set_title(title)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

        ax.set_xlim(-100, 360)
        ax.set_ylim(-220, 180)

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


if __name__ == "__main__":
    viz = WokBotVisualizer()
    viz.show()