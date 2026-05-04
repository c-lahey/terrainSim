"""
fig3b_2t1r_force_visualizer.py

Interactive visualizer for Fig. 3(b)-style planar 2T1R mechanism.

Features:
    - x, y, theta pose sliders
    - symmetric five-bar geometry sliders
    - B1/C1 y-offset sliders for staggered base pivots
    - branch-stable IK selection
    - force sliders Fx, Fy at the platform center
    - actuator torque display for static equilibrium

Requires:
    fig3b_2t1r_validation.py

Expected classes from validation file:
    Fig3B2T1R
    Fig3BParams
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons

from fig3b_2t1r_validation import Fig3B2T1R, Fig3BParams


class Fig3BForceVisualizer:
    def __init__(self):
        self.default_params = Fig3BParams()
        self.kin = Fig3B2T1R(self.default_params)

        # Pose = [platform center x, platform center y, platform theta]
        self.pose = np.array([10.0, 8.0, math.radians(5.0)], dtype=float)

        # Applied force at platform center.
        # Units are arbitrary, but if geometry is inch-like and force is N,
        # torque output is N*in.
        self.force = np.array([0.0, -10.0], dtype=float)

        self.solutions = []
        self.branch_idx = 0
        self.show_all = False

        # Persistent selected joint vector for branch continuity.
        self.q_current = None

        self._refresh_solutions()
        if self.solutions:
            self.q_current = self.solutions[self.branch_idx].q.copy()

        self.fig = plt.figure(figsize=(15, 9))

        # Main plot
        self.ax = self.fig.add_axes([0.06, 0.33, 0.58, 0.62])

        # Pose sliders
        self.ax_x = self.fig.add_axes([0.08, 0.24, 0.50, 0.025])
        self.ax_y = self.fig.add_axes([0.08, 0.20, 0.50, 0.025])
        self.ax_theta = self.fig.add_axes([0.08, 0.16, 0.50, 0.025])
        self.ax_branch = self.fig.add_axes([0.08, 0.12, 0.50, 0.025])

        # Force sliders
        self.ax_fx = self.fig.add_axes([0.08, 0.07, 0.50, 0.025])
        self.ax_fy = self.fig.add_axes([0.08, 0.03, 0.50, 0.025])

        self.slider_x = Slider(
            self.ax_x,
            "platform center x",
            -12.0,
            36.0,
            valinit=self.pose[0],
        )
        self.slider_y = Slider(
            self.ax_y,
            "platform center y",
            -5.0,
            20.0,
            valinit=self.pose[1],
        )
        self.slider_theta = Slider(
            self.ax_theta,
            "platform theta [deg]",
            -70.0,
            70.0,
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

        self.slider_fx = Slider(
            self.ax_fx,
            "Fx at EE",
            -1000.0,
            1000.0,
            valinit=self.force[0],
        )
        self.slider_fy = Slider(
            self.ax_fy,
            "Fy at EE",
            -1000.0,
            1000.0,
            valinit=self.force[1],
        )

        # Buttons / checkboxes
        self.ax_prev = self.fig.add_axes([0.68, 0.88, 0.06, 0.04])
        self.ax_next = self.fig.add_axes([0.75, 0.88, 0.06, 0.04])
        self.ax_reset_pose = self.fig.add_axes([0.82, 0.88, 0.13, 0.04])
        self.ax_reset_geom = self.fig.add_axes([0.82, 0.82, 0.13, 0.04])
        self.ax_reset_force = self.fig.add_axes([0.82, 0.76, 0.13, 0.04])
        self.ax_check = self.fig.add_axes([0.68, 0.76, 0.13, 0.08])

        self.btn_prev = Button(self.ax_prev, "Prev")
        self.btn_next = Button(self.ax_next, "Next")
        self.btn_reset_pose = Button(self.ax_reset_pose, "Reset Pose")
        self.btn_reset_geom = Button(self.ax_reset_geom, "Reset Geometry")
        self.btn_reset_force = Button(self.ax_reset_force, "Reset Force")
        self.check = CheckButtons(self.ax_check, ["Show all"], [self.show_all])

        # Geometry sliders
        self.geom_sliders = {}
        self._create_geometry_sliders()

        # Pose callbacks
        self.slider_x.on_changed(self._on_pose_change)
        self.slider_y.on_changed(self._on_pose_change)
        self.slider_theta.on_changed(self._on_pose_change)
        self.slider_branch.on_changed(self._on_branch_change)

        # Force callbacks
        self.slider_fx.on_changed(self._on_force_change)
        self.slider_fy.on_changed(self._on_force_change)

        # Button callbacks
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.btn_reset_pose.on_clicked(self._on_reset_pose)
        self.btn_reset_geom.on_clicked(self._on_reset_geometry)
        self.btn_reset_force.on_clicked(self._on_reset_force)
        self.check.on_clicked(self._on_check)

        self._draw()

    # ------------------------------------------------------------------
    # Geometry sliders
    # ------------------------------------------------------------------

    def _create_geometry_sliders(self):
        """
        Geometry sliders.

        Naming note:
            In our Python model:
                A1 = fixed left base pivot
                A2 = middle base pivot
                A3 = right base pivot

            In the Fig. 3(b)-style sketch, you asked for B1/C1 y offsets.
            Here:
                B1 y-offset slider controls A2[1]
                C1 y-offset slider controls A3[1]
        """
        specs = [
            (
                "A2x",
                "B1 x: five-bar base spacing",
                2.0,
                25.0,
                self.kin.params.A2[0],
            ),
            (
                "A2y",
                "B1 y-offset: middle base height",
                -10.0,
                10.0,
                self.kin.params.A2[1],
            ),
            (
                "A3x",
                "C1 x: right/orientation base spacing",
                8.0,
                45.0,
                self.kin.params.A3[0],
            ),
            (
                "A3y",
                "C1 y-offset: right base height",
                -10.0,
                10.0,
                self.kin.params.A3[1],
            ),
            (
                "active_link",
                "5-bar active links: A1-B1 = A2-B2",
                2.0,
                35.0,
                self.kin.params.l1a,
            ),
            (
                "passive_link",
                "5-bar passive links: B1-CL = B2-CL",
                2.0,
                40.0,
                self.kin.params.l1b,
            ),
            (
                "l3a",
                "right active link A3-B3: angle limb",
                2.0,
                35.0,
                self.kin.params.l3a,
            ),
            (
                "l3b",
                "right passive link B3-CR: angle limb",
                2.0,
                40.0,
                self.kin.params.l3b,
            ),
            (
                "platform_length",
                "platform length CL-CR: theta leverage",
                3.0,
                40.0,
                self.kin.params.platform_length,
            ),
        ]

        x0 = 0.69
        w = 0.28
        h = 0.023
        y_top = 0.68
        dy = 0.050

        for i, (name, label, vmin, vmax, v0) in enumerate(specs):
            ax_slider = self.fig.add_axes([x0, y_top - i * dy, w, h])
            slider = Slider(ax_slider, label, vmin, vmax, valinit=v0)
            slider.on_changed(self._on_geometry_change)
            self.geom_sliders[name] = slider

    def _read_params_from_sliders(self) -> Fig3BParams:
        A2x = self.geom_sliders["A2x"].val
        A2y = self.geom_sliders["A2y"].val

        A3x = self.geom_sliders["A3x"].val
        A3y = self.geom_sliders["A3y"].val

        active_link = self.geom_sliders["active_link"].val
        passive_link = self.geom_sliders["passive_link"].val

        return Fig3BParams(
            A1=(0.0, 0.0),

            # Staggered base pivots
            A2=(A2x, A2y),
            A3=(A3x, A3y),

            # Symmetric five-bar side
            l1a=active_link,
            l2a=active_link,
            l1b=passive_link,
            l2b=passive_link,

            # Right orientation side
            l3a=self.geom_sliders["l3a"].val,
            l3b=self.geom_sliders["l3b"].val,

            platform_length=self.geom_sliders["platform_length"].val,
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

        # Deterministic ordering.
        sols.sort(key=lambda s: s.branches)

        self.solutions = sols

        if len(self.solutions) == 0:
            self.branch_idx = 0
            self.q_current = None
            return

        if self.q_current is not None:
            distances = [
                self._angle_distance(sol.q, self.q_current)
                for sol in self.solutions
            ]
            self.branch_idx = int(np.argmin(distances))
        else:
            self.branch_idx = int(np.clip(self.branch_idx, 0, len(self.solutions) - 1))

        self.q_current = self.solutions[self.branch_idx].q.copy()

    def _sync_branch_slider(self):
        self.slider_branch.valmax = max(0, len(self.solutions) - 1)
        self.slider_branch.ax.set_xlim(
            self.slider_branch.valmin,
            self.slider_branch.valmax + 1e-9,
        )

        self.slider_branch.eventson = False
        self.slider_branch.set_val(self.branch_idx)
        self.slider_branch.eventson = True

    # ------------------------------------------------------------------
    # UI callbacks
    # ------------------------------------------------------------------

    def _on_pose_change(self, _val):
        self.pose = np.array(
            [
                self.slider_x.val,
                self.slider_y.val,
                math.radians(self.slider_theta.val),
            ],
            dtype=float,
        )

        self._refresh_solutions()
        self._sync_branch_slider()
        self._draw()

    def _on_force_change(self, _val):
        self.force = np.array(
            [
                self.slider_fx.val,
                self.slider_fy.val,
            ],
            dtype=float,
        )

        self._draw()

    def _on_geometry_change(self, _val):
        self.kin = Fig3B2T1R(self._read_params_from_sliders())

        self._refresh_solutions()
        self._sync_branch_slider()
        self._draw()

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
        self.slider_x.set_val(10.0)
        self.slider_y.set_val(8.0)
        self.slider_theta.set_val(5.0)

    def _on_reset_force(self, _event):
        self.slider_fx.set_val(0.0)
        self.slider_fy.set_val(-10.0)

    def _on_reset_geometry(self, _event):
        self.q_current = None
        defaults = self.default_params

        values = {
            "A2x": defaults.A2[0],
            "A2y": defaults.A2[1],
            "A3x": defaults.A3[0],
            "A3y": defaults.A3[1],
            "active_link": defaults.l1a,
            "passive_link": defaults.l1b,
            "l3a": defaults.l3a,
            "l3b": defaults.l3b,
            "platform_length": defaults.platform_length,
        }

        for name, slider in self.geom_sliders.items():
            slider.eventson = False
            slider.set_val(values[name])
            slider.eventson = True

        self.kin = Fig3B2T1R(self._read_params_from_sliders())
        self._refresh_solutions()
        self._sync_branch_slider()
        self._draw()

    def _on_check(self, _label):
        self.show_all = not self.show_all
        self._draw()

    # ------------------------------------------------------------------
    # Mechanics helpers
    # ------------------------------------------------------------------

    def _condition_number(self, q: np.ndarray) -> float:
        try:
            J = self.kin.numerical_jacobian_dq_dp(self.pose, q_prev=q)
            return float(np.linalg.cond(J))
        except Exception:
            return float("nan")

    def _actuator_torques(self, q: np.ndarray) -> np.ndarray:
        """
        Return actuator torques required to resist the selected EE force.

        numerical_jacobian_dq_dp returns J_qp = dq/dp.

        Virtual work:
            tau^T dq = wrench^T dp
            dq = J_qp dp

        Therefore:
            wrench = J_qp.T @ tau
            tau = solve(J_qp.T, wrench)
        """
        try:
            wrench = np.array([self.force[0], self.force[1], 0.0], dtype=float)
            J_qp = self.kin.numerical_jacobian_dq_dp(self.pose, q_prev=q)
            tau = np.linalg.solve(J_qp.T, wrench) * .0254
            return tau
        except Exception:
            return np.array([np.nan, np.nan, np.nan], dtype=float)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _plot_segment(ax, P, Q, **kwargs):
        ax.plot([P[0], Q[0]], [P[1], Q[1]], **kwargs)

    def _draw_force_arrow(self, ax):
        C = self.pose[:2]
        Fx, Fy = self.force

        mag = math.hypot(Fx, Fy)
        if mag < 1e-9:
            return

        # Drawing scale only.
        arrow_scale = 0.08

        ax.arrow(
            C[0],
            C[1],
            arrow_scale * Fx,
            arrow_scale * Fy,
            head_width=0.35,
            length_includes_head=True,
            linewidth=2.5,
        )

        ax.text(
            C[0] + arrow_scale * Fx,
            C[1] + arrow_scale * Fy,
            f"  F=({Fx:.1f}, {Fy:.1f})",
            fontsize=9,
        )

    def _draw_one_solution(
        self,
        ax,
        q: np.ndarray,
        alpha: float = 1.0,
        linewidth: float = 2.0,
        show_labels: bool = True,
    ):
        pts = self.kin.joint_points(q, self.pose)

        A1, A2, A3 = pts["A1"], pts["A2"], pts["A3"]
        B1, B2, B3 = pts["B1"], pts["B2"], pts["B3"]
        CL, CR, C = pts["CL"], pts["CR"], pts["C"]

        # Ground pivots
        ax.scatter(
            [A1[0], A2[0], A3[0]],
            [A1[1], A2[1], A3[1]],
            s=80,
            marker="s",
            alpha=alpha,
        )

        # Base layout line
        self._plot_segment(
            ax,
            A1,
            A2,
            linewidth=1.0,
            linestyle=":",
            alpha=max(alpha, 0.4),
        )
        self._plot_segment(
            ax,
            A2,
            A3,
            linewidth=1.0,
            linestyle=":",
            alpha=max(alpha, 0.4),
        )

        # Active links
        self._plot_segment(ax, A1, B1, linewidth=linewidth, alpha=alpha)
        self._plot_segment(ax, A2, B2, linewidth=linewidth, alpha=alpha)
        self._plot_segment(ax, A3, B3, linewidth=linewidth, alpha=alpha)

        # Passive links
        self._plot_segment(ax, B1, CL, linewidth=linewidth, alpha=alpha)
        self._plot_segment(ax, B2, CL, linewidth=linewidth, alpha=alpha)
        self._plot_segment(ax, B3, CR, linewidth=linewidth, alpha=alpha)

        # Platform
        self._plot_segment(ax, CL, CR, linewidth=linewidth + 1.0, alpha=alpha)

        # Points
        ax.scatter(
            [CL[0], CR[0], C[0]],
            [CL[1], CR[1], C[1]],
            s=60,
            alpha=alpha,
        )
        ax.scatter(
            [B1[0], B2[0], B3[0]],
            [B1[1], B2[1], B3[1]],
            s=40,
            alpha=alpha,
        )

        # Orientation arrow
        theta = self.pose[2]
        arrow_len = 2.0
        ax.arrow(
            C[0],
            C[1],
            arrow_len * math.cos(theta),
            arrow_len * math.sin(theta),
            head_width=0.35,
            length_includes_head=True,
            alpha=alpha,
        )

        if show_labels:
            text_alpha = max(0.35, alpha)

            ax.text(A1[0], A1[1], "  A1", alpha=text_alpha)
            ax.text(A2[0], A2[1], "  B1 base", alpha=text_alpha)
            ax.text(A3[0], A3[1], "  C1 base", alpha=text_alpha)

            ax.text(B1[0], B1[1], "  B1", alpha=text_alpha)
            ax.text(B2[0], B2[1], "  B2", alpha=text_alpha)
            ax.text(B3[0], B3[1], "  B3", alpha=text_alpha)

            ax.text(CL[0], CL[1], "  CL", alpha=text_alpha)
            ax.text(CR[0], CR[1], "  CR", alpha=text_alpha)
            ax.text(C[0], C[1], "  C", alpha=text_alpha)

    def _draw_workspace_box(self, ax):
        """
        Approximate 500 mm x 300 mm target box, using sketch units as inches.
        """
        width = 500.0 / 25.4
        height = 300.0 / 25.4

        cx = 10.0
        cy = 8.0

        x0 = cx - width / 2.0
        x1 = cx + width / 2.0
        y0 = cy - height / 2.0
        y1 = cy + height / 2.0

        ax.plot(
            [x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0],
            linestyle=":",
            linewidth=1.5,
        )
        ax.text(x0, y1 + 0.3, "target ~500 x 300 mm", fontsize=9)

    # ------------------------------------------------------------------
    # Main draw
    # ------------------------------------------------------------------

    def _draw(self):
        self.ax.clear()
        ax = self.ax

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)

        self._draw_workspace_box(ax)

        if self.show_all and len(self.solutions) > 0:
            for sol in self.solutions:
                self._draw_one_solution(
                    ax,
                    sol.q,
                    alpha=0.22,
                    linewidth=1.2,
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

            self._draw_force_arrow(ax)

            q_deg = np.degrees(sol.q)
            condJ = self._condition_number(sol.q)
            tau = self._actuator_torques(sol.q)

            title = (
                f"Fig. 3(b) 2T1R force visualizer | "
                f"pose = [x={self.pose[0]:.2f}, y={self.pose[1]:.2f}, "
                f"theta={math.degrees(self.pose[2]):.2f} deg]\n"
                f"branch {self.branch_idx + 1}/{len(self.solutions)} | "
                f"signs = {sol.branches} | "
                f"q = [{q_deg[0]:.1f}, {q_deg[1]:.1f}, {q_deg[2]:.1f}] deg | "
                f"cond(J) = {condJ:.2e}\n"
                f"force = [Fx={self.force[0]:.1f}, Fy={self.force[1]:.1f}] | "
                f"required tau = [{tau[0]:.2f}, {tau[1]:.2f}, {tau[2]:.2f}]"
            )

        else:
            CL, CR = self.kin.platform_points(self.pose)
            C = self.pose[:2]

            ax.scatter([CL[0], CR[0], C[0]], [CL[1], CR[1], C[1]], s=60)
            self._plot_segment(ax, CL, CR, linewidth=2.5)

            title = (
                f"UNREACHABLE pose | "
                f"pose = [x={self.pose[0]:.2f}, y={self.pose[1]:.2f}, "
                f"theta={math.degrees(self.pose[2]):.2f} deg]"
            )

        ax.set_title(title)
        ax.set_xlabel("x [sketch units]")
        ax.set_ylabel("y [sketch units]")

        ax.set_xlim(-12, 38)
        ax.set_ylim(-14, 26)

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


if __name__ == "__main__":
    viz = Fig3BForceVisualizer()
    viz.show()