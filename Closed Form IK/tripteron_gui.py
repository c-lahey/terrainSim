from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from tripteron_affine_model import TripteronAffineModel
from tripteron_kinematics import load_moving_plane_kinematics, solve_simple_branch, solve_closed_branch


Vec3 = np.ndarray


@dataclass
class GUIState:
    p: Vec3
    q: Vec3
    F_ee: Vec3
    actuator_forces: Vec3
    reconstruction_residual: Vec3
    used_least_squares: bool
    branch_A: object
    branch_B: object
    branch_C: object


class TripteronAffineGUI:
    def __init__(self, input_json: str | Path):
        self.info_artist = None
        self.input_json = Path(input_json)
        self.model = TripteronAffineModel(input_json)
        self.kin = load_moving_plane_kinematics(input_json)

        self.p0 = self.model.p0.copy()
        self.q0 = self.model.q0.copy()

        self.state = self._compute_state(self.p0, np.zeros(3, dtype=float))

        self.fig = None
        self.ax_pose = None
        self.ax_info = None
        self.text_info = None
        self.sliders = {}
        self.reset_button = None

    @staticmethod
    def _arr(x) -> np.ndarray:
        return np.asarray(x, dtype=float).reshape(3)

    def _compute_state(self, p: Vec3, F_ee: Vec3) -> GUIState:
        p = self._arr(p)
        F_ee = self._arr(F_ee)

        ik_res = self.model.ik(p)
        fmap = self.model.actuator_forces_from_ee_force(F_ee)

        q = ik_res.actuators
        branch_A = solve_simple_branch(self.kin.branch_A, p)
        branch_B = solve_simple_branch(self.kin.branch_B, p)
        branch_C = solve_closed_branch(self.kin.branch_C, p)

        # Keep geometry and affine q synchronized in the display.
        branch_A.q = q[0]
        branch_A.base_joint = self.kin.branch_A.base_joint(q[0])
        branch_A.elbow_candidates = self.kin.branch_A.elbow_candidates(p, q[0])
        branch_A.closure = self.kin.branch_A.closure_margin(p, q[0])

        branch_B.q = q[1]
        branch_B.base_joint = self.kin.branch_B.base_joint(q[1])
        branch_B.elbow_candidates = self.kin.branch_B.elbow_candidates(p, q[1])
        branch_B.closure = self.kin.branch_B.closure_margin(p, q[1])

        branch_C.q = q[2]
        branch_C.upper_base_joint = self.kin.branch_C.upper_base(q[2])
        branch_C.lower_base_joint = self.kin.branch_C.lower_base(q[2])
        branch_C.elbow_candidates = self.kin.branch_C.elbow_candidates(p, q[2])
        branch_C.closure = self.kin.branch_C.closure_margin(p, q[2])

        return GUIState(
            p=p,
            q=q,
            F_ee=F_ee,
            actuator_forces=fmap.actuator_forces,
            reconstruction_residual=fmap.reconstruction_residual,
            used_least_squares=fmap.used_least_squares,
            branch_A=branch_A,
            branch_B=branch_B,
            branch_C=branch_C,
        )

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.suptitle("Tripteron Affine IK + Actuator Force Explorer")

        self.ax_pose = self.fig.add_axes([0.05, 0.38, 0.36, 0.54], projection="3d")

        # Use figure-level text instead of an axes so it doesn't get clipped.
        self.info_artist = self.fig.text(
            0.45,
            0.90,
            "",
            va="top",
            ha="left",
            family="monospace",
            fontsize=10,
            linespacing=1.15,
        )

        slider_specs = [
            ("px", 0.14, self.p0[0] - 100.0, self.p0[0] + 100.0, self.p0[0]),
            ("py", 0.10, self.p0[1] - 100.0, self.p0[1] + 100.0, self.p0[1]),
            ("pz", 0.06, self.p0[2] - 100.0, self.p0[2] + 100.0, self.p0[2]),
            ("Fx", 0.26, -1500.0, 1500.0, 0.0),
            ("Fy", 0.22, -1500.0, 1500.0, 0.0),
            ("Fz", 0.18, -1500.0, 1500.0, 0.0),
        ]

        for name, y, vmin, vmax, vinit in slider_specs:
            ax = self.fig.add_axes([0.12, y, 0.62, 0.03])
            self.sliders[name] = Slider(ax, name, vmin, vmax, valinit=vinit)
            self.sliders[name].on_changed(self._on_slider_change)

        reset_ax = self.fig.add_axes([0.76, 0.05, 0.08, 0.045])
        self.reset_button = Button(reset_ax, "Reset")
        self.reset_button.on_clicked(self._on_reset)

        self._refresh_all()

    def _current_slider_state(self) -> tuple[Vec3, Vec3]:
        p = np.array([
            self.sliders["px"].val,
            self.sliders["py"].val,
            self.sliders["pz"].val,
        ], dtype=float)
        F_ee = np.array([
            self.sliders["Fx"].val,
            self.sliders["Fy"].val,
            self.sliders["Fz"].val,
        ], dtype=float)
        return p, F_ee

    def _format_matrix(self, A: np.ndarray) -> str:
        rows = []
        for row in A:
            rows.append(f"[{row[0]:+8.3f} {row[1]:+8.3f} {row[2]:+8.3f}]")
        return "".join(rows)

    def _refresh_text(self) -> None:
        p = self.state.p
        q = self.state.q
        F = self.state.F_ee
        fa = self.state.actuator_forces
        residual = self.state.reconstruction_residual
        A = self.model.jacobian()
        b = self.model.b
        source_name = self.input_json.name

        sections = [
            f"Source file: {self.input_json.name}",
            "",
            "EE position p",
            f"  x  {p[0]:+9.3f}",
            f"  y  {p[1]:+9.3f}",
            f"  z  {p[2]:+9.3f}",
            "",
            "Affine IK: q = A p + b",
            f"  qA {q[0]:+9.4f}",
            f"  qB {q[1]:+9.4f}",
            f"  qC {q[2]:+9.4f}",
            "",
            "Applied EE force",
            f"  Fx {F[0]:+9.3f}",
            f"  Fy {F[1]:+9.3f}",
            f"  Fz {F[2]:+9.3f}",
            "",
            "Estimated actuator forces",
            f"  fA {fa[0]:+9.4f}",
            f"  fB {fa[1]:+9.4f}",
            f"  fC {fa[2]:+9.4f}",
            "",
            "Jacobian A = dq/dp",
            self._format_matrix(A),
            "",
            f"Offset b [{b[0]:+8.2f} {b[1]:+8.2f} {b[2]:+8.2f}]",
            "",
            "Force residual",
            f"  rx {residual[0]:+9.2e}",
            f"  ry {residual[1]:+9.2e}",
            f"  rz {residual[2]:+9.2e}",
            f"  least-squares {self.state.used_least_squares}",
        ]
        self.info_artist.set_text("\n".join(sections))

    def _plot_segment(self, p0: Vec3, p1: Vec3, **kwargs) -> None:
        self.ax_pose.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], **kwargs)

    def _refresh_pose(self) -> None:
        ax = self.ax_pose
        ax.cla()
        ax.set_title("Pose")

        s = self.state
        p = s.p

        # EE points
        cA = s.branch_A.ee_joint
        cB = s.branch_B.ee_joint
        cU = s.branch_C.upper_ee_joint
        cL = s.branch_C.lower_ee_joint

        # Bases
        bA = s.branch_A.base_joint
        bB = s.branch_B.base_joint
        bU = s.branch_C.upper_base_joint
        bL = s.branch_C.lower_base_joint

        # Choose the elbow candidate nearest the reference elbow for each branch.
        def pick(cands, ref):
            if cands is None:
                return None
            return min(cands, key=lambda m: np.linalg.norm(m - ref))

        mA = pick(s.branch_A.elbow_candidates, self.kin.branch_A.reference_mid_joint)
        mB = pick(s.branch_B.elbow_candidates, self.kin.branch_B.reference_mid_joint)
        upper_cands, lower_cands = s.branch_C.elbow_candidates
        mU = pick(upper_cands, self.kin.branch_C.reference_upper_mid)
        mL = pick(lower_cands, self.kin.branch_C.reference_lower_mid)

        # Plot rails.
        rail_half = 35.0
        for a in [self.kin.branch_A.a_base, self.kin.branch_B.a_base, self.kin.branch_C.a_upper, self.kin.branch_C.a_lower]:
            p0 = a - rail_half * self.kin.branch_A.rail_dir
            p1 = a + rail_half * self.kin.branch_A.rail_dir
            self._plot_segment(p0, p1, linewidth=1.0, alpha=0.45)

        # Plot branch linkages.
        if mA is not None:
            self._plot_segment(bA, mA, linewidth=2.0)
            self._plot_segment(mA, cA, linewidth=2.0)
        if mB is not None:
            self._plot_segment(bB, mB, linewidth=2.0)
            self._plot_segment(mB, cB, linewidth=2.0)
        if mU is not None:
            self._plot_segment(bU, mU, linewidth=2.0)
            self._plot_segment(mU, cU, linewidth=2.0)
        if mL is not None:
            self._plot_segment(bL, mL, linewidth=2.0)
            self._plot_segment(mL, cL, linewidth=2.0)

        # EE rectangle and centroid.
        ee_pts = np.array([cA, cU, cL, cB, cA])
        ax.plot(ee_pts[:, 0], ee_pts[:, 1], ee_pts[:, 2], linewidth=2.0)
        ax.scatter([p[0]], [p[1]], [p[2]], s=50)

        # Base and elbow markers.
        pts = [bA, bB, bU, bL, cA, cB, cU, cL]
        if mA is not None:
            pts.append(mA)
        if mB is not None:
            pts.append(mB)
        if mU is not None:
            pts.append(mU)
        if mL is not None:
            pts.append(mL)
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=18)

        # Force vector at EE centroid.
        F = s.F_ee
        Fmag = np.linalg.norm(F)
        if Fmag > 1e-9:
            scale = 0.10
            ax.quiver(p[0], p[1], p[2], F[0], F[1], F[2], length=scale * Fmag, normalize=True)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = 0.5 * (mins + maxs)
        radius = 0.6 * np.max(maxs - mins + 1e-9)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect([1, 1, 1])

    def _refresh_all(self) -> None:
        self._refresh_text()
        self._refresh_pose()
        self.fig.canvas.draw_idle()

    def _on_slider_change(self, _val) -> None:
        try:
            p, F_ee = self._current_slider_state()
            self.state = self._compute_state(p, F_ee)
            self._refresh_all()
        except Exception as exc:
            self.text_info.set_text(f"Computation failed:{exc}")
            self.fig.canvas.draw_idle()

    def _on_reset(self, _event) -> None:
        reset_values = {
            "px": self.p0[0],
            "py": self.p0[1],
            "pz": self.p0[2],
            "Fx": 0.0,
            "Fy": 0.0,
            "Fz": 0.0,
        }
        for key, value in reset_values.items():
            self.sliders[key].reset()
            self.sliders[key].set_val(value)
        self.state = self._compute_state(self.p0, np.zeros(3, dtype=float))
        self._refresh_all()

    def show(self) -> None:
        self._build_figure()
        plt.show()

if __name__ == "__main__":
    import sys
    from pathlib import Path

    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fusion_kinematics_rich_export.json")
    print("Loading JSON from:", json_path.resolve())

    gui = TripteronAffineGUI(json_path)
    print("Loaded p0:", gui.p0)
    print("Loaded q0:", gui.q0)
    gui.show()