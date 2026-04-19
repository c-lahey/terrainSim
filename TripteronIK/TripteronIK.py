"""
Tripteron IK Command Add-in
===========================
Adds a command under ADD-INS → Scripts and Add-Ins that places a triad at the
end-effector occurrence and drives the three slider joints to match the dragged
target position.  A finite-difference Jacobian built from Fusion's own FK engine
(transform2) gives actuator force estimates via τ = Jᵀ F.

Assumptions
-----------
* The open design contains exactly three slider-type joints.
* At least one occurrence has "ee" anywhere in its name (case-insensitive) —
  that occurrence is treated as the end-effector platform.
* numpy is available (bundled with Fusion 360).
"""

import adsk.core
import adsk.fusion
import traceback
import numpy as np

# ---------------------------------------------------------------------------
# Module-level globals (kept alive to prevent GC of event handlers)
# ---------------------------------------------------------------------------
_app: "adsk.core.Application | None" = None
_ui:  "adsk.core.UserInterface | None" = None
_handlers: list = []
_robot:    "RobotModel | None" = None
_cmd_def:  "adsk.core.CommandDefinition | None" = None

CMD_ID       = "TripteronIK_Cmd"
CMD_NAME     = "Tripteron IK"
CMD_DESC     = "Drive slider joints via EE triad; estimate actuator forces."
WORKSPACE_ID = "FusionSolidEnvironment"
PANEL_ID     = "SolidScriptsAddinsPanel"

# Numerical parameters
FD_EPS   = 5e-4   # finite-difference step for Jacobian (cm; ~5 µm)
IK_TOL   = 1e-4   # Newton IK position tolerance (cm; ~1 µm)
IK_ITER  = 10     # max Newton iterations (converges in 2-3 for small moves)


# ---------------------------------------------------------------------------
# Robot model: wraps the three slider joints and the EE occurrence
# ---------------------------------------------------------------------------
class RobotModel:
    """
    Stateful wrapper around Fusion's joints and EE occurrence.

    All positions are in Fusion internal units (cm).

    The FK oracle is Fusion itself: we set jointMotion.slideValue on each
    slider joint, Fusion re-solves the constrained assembly, and we read the
    EE position back via occurrence.transform2.
    """

    def __init__(self, slider_joints: list, ee_occ: adsk.fusion.Occurrence):
        if len(slider_joints) != 3:
            raise ValueError(f"Need exactly 3 slider joints; got {len(slider_joints)}.")
        self.joints  = slider_joints
        self.ee_occ  = ee_occ

        # Snapshot of the reference configuration
        self.s_ref  = np.array([j.jointMotion.slideValue for j in self.joints], dtype=float)
        self.ee_ref = self._read_ee()

        # Warm-start cache: last solved configuration
        self._last_s = self.s_ref.copy()

    # ------------------------------------------------------------------
    def _read_ee(self) -> np.ndarray:
        """Read EE world position from transform2 (cm)."""
        t = self.ee_occ.transform2.translation
        return np.array([t.x, t.y, t.z])

    def fk(self, s: np.ndarray) -> np.ndarray:
        """Set slider values, let Fusion solve, return EE position (cm)."""
        for i, jt in enumerate(self.joints):
            jt.jointMotion.slideValue = float(s[i])
        self._last_s = np.array(s, dtype=float)
        return self._read_ee()

    # ------------------------------------------------------------------
    def jacobian_fd(self, s: np.ndarray) -> tuple:
        """
        Forward-difference 3×3 Jacobian J where J[:,i] = ∂x_ee/∂s_i.

        Returns (J, p0) where p0 is the EE position at s.
        Restores joint configuration to s on exit.
        """
        p0 = self.fk(s)
        J  = np.zeros((3, 3))
        for i in range(3):
            sp       = s.copy()
            sp[i]   += FD_EPS
            J[:, i]  = (self.fk(sp) - p0) / FD_EPS
        self.fk(s)   # restore
        return J, p0

    # ------------------------------------------------------------------
    def solve_ik(
        self,
        target: np.ndarray,
        s0:     "np.ndarray | None" = None,
    ) -> tuple:
        """
        Newton-Raphson IK: find slider values so FK(s) ≈ target.

        Returns (s_best, residual_cm, J_at_best) where J is the 3×3
        Jacobian at the returned configuration (used for force estimation
        without an extra round of finite differences).
        """
        s = (s0 if s0 is not None else self._last_s).copy()
        best_s, best_err, best_J = s.copy(), np.inf, np.eye(3)

        for _ in range(IK_ITER):
            J, p   = self.jacobian_fd(s)   # also evaluates fk(s)
            err_v  = target - p
            err    = float(np.linalg.norm(err_v))
            if err < best_err:
                best_s, best_err, best_J = s.copy(), err, J.copy()
            if err < IK_TOL:
                break
            try:
                s = s + np.linalg.solve(J, err_v)
            except np.linalg.LinAlgError:
                break   # singular Jacobian — stop here

        # Leave Fusion in the best-found configuration
        self.fk(best_s)
        return best_s, best_err, best_J

    # ------------------------------------------------------------------
    def static_forces(self, J: np.ndarray, force: np.ndarray) -> np.ndarray:
        """
        Actuator force estimates from virtual work: τ = Jᵀ F.

        J     : 3×3 Jacobian ∂x_ee/∂s (already computed by solve_ik).
        force : (3,) EE force vector [Fx, Fy, Fz] in Newtons.
        Returns (3,) actuator forces in N (positive = push in slide direction).
        """
        return J.T @ np.asarray(force, dtype=float)

    def restore_ref(self):
        """Reset joints to the configuration recorded at startup."""
        self.fk(self.s_ref)


# ---------------------------------------------------------------------------
# Joint / occurrence discovery
# ---------------------------------------------------------------------------
def _is_slider_joint(j) -> bool:
    try:
        return "sliderjointmotion" in j.jointMotion.objectType.lower()
    except Exception:
        return False


def find_robot_components(design: adsk.fusion.Design):
    """
    Scan root component for 3 slider joints and an EE occurrence.
    Returns (slider_joints, ee_occ) or raises RuntimeError with a hint.
    """
    root     = design.rootComponent
    sliders  = []
    seen_tok: set = set()

    def _collect(j):
        try:
            tok = j.entityToken
        except Exception:
            tok = str(id(j))
        if tok in seen_tok:
            return
        seen_tok.add(tok)
        if _is_slider_joint(j):
            sliders.append(j)

    for src in [root.joints, root.asBuiltJoints]:
        try:
            for i in range(src.count):
                _collect(src.item(i))
        except Exception:
            pass

    if len(sliders) < 3:
        raise RuntimeError(
            f"Found {len(sliders)} slider joint(s); need exactly 3.\n"
            "Make sure all prismatic joints are defined in the root component."
        )

    # Stable ordering by joint name
    sliders.sort(key=lambda j: getattr(j, "name", ""))
    sliders = sliders[:3]

    # EE occurrence: first one whose name contains "ee"
    ee_occ = None
    try:
        for i in range(root.allOccurrences.count):
            occ = root.allOccurrences.item(i)
            if "ee" in occ.name.lower():
                ee_occ = occ
                break
    except Exception:
        pass

    if ee_occ is None:
        raise RuntimeError(
            "No occurrence with 'ee' in its name found.\n"
            "Rename the end-effector platform occurrence so its name contains 'ee' "
            "(e.g. 'tripteron ee v6:1')."
        )

    return sliders, ee_occ


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------
def _format_results(robot, s_sol, err_cm, tau):
    """Build a human-readable summary string for the text-box input."""
    if s_sol is None:
        return (
            "Drag the triad to a target EE position.\n"
            "Set Fx / Fy / Fz to apply a static load."
        )

    mm   = 10.0      # cm → mm
    s_mm = s_sol * mm
    ref  = robot.s_ref * mm

    lines = [
        f"Slider A: {s_mm[0]:7.3f} mm  ({s_mm[0]-ref[0]:+.3f})",
        f"Slider B: {s_mm[1]:7.3f} mm  ({s_mm[1]-ref[1]:+.3f})",
        f"Slider C: {s_mm[2]:7.3f} mm  ({s_mm[2]-ref[2]:+.3f})",
        f"IK residual: {err_cm*mm:.4f} mm",
    ]

    if tau is not None:
        lines += [
            "────────────────────────────",
            f"  τ_A: {tau[0]:+8.3f} N",
            f"  τ_B: {tau[1]:+8.3f} N",
            f"  τ_C: {tau[2]:+8.3f} N",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command event handlers
# ---------------------------------------------------------------------------
class _CreatedHandler(adsk.core.CommandCreatedEventHandler):
    def notify(self, args: adsk.core.CommandCreatedEventArgs):
        global _robot
        try:
            cmd    = args.command
            inputs = cmd.commandInputs
            cmd.isRepeatable = False

            # Discover robot kinematics
            app    = adsk.core.Application.get()
            design = adsk.fusion.Design.cast(app.activeProduct)
            sliders, ee_occ = find_robot_components(design)
            _robot = RobotModel(sliders, ee_occ)

            # ---- Triad at current EE position ----
            ee  = _robot.ee_ref
            mat = adsk.core.Matrix3D.create()
            mat.translation = adsk.core.Vector3D.create(
                float(ee[0]), float(ee[1]), float(ee[2])
            )
            triad = inputs.addTriadCommandInput("eeTriad", "EE Target")
            triad.transform = mat
            # This is a 3T robot — disable rotation handles
            try:
                triad.isXRotationEnabled = False
                triad.isYRotationEnabled = False
                triad.isZRotationEnabled = False
            except Exception:
                pass  # older API versions may not expose these flags

            # ---- Applied-force spinners (values are raw Newtons) ----
            inputs.addFloatSpinnerCommandInput(
                "Fx", "Fx  (N)", "", -500.0, 500.0, 1.0, 0.0
            )
            inputs.addFloatSpinnerCommandInput(
                "Fy", "Fy  (N)", "", -500.0, 500.0, 1.0, 0.0
            )
            inputs.addFloatSpinnerCommandInput(
                "Fz", "Fz  (N)", "", -500.0, 500.0, 1.0, 0.0
            )

            # ---- Read-only results panel ----
            inputs.addTextBoxCommandInput(
                "results", "Results",
                _format_results(_robot, None, None, None),
                9, True
            )

            # Wire up sub-handlers
            for evname, handler_cls in [
                ("executePreview", _PreviewHandler),
                ("execute",        _ExecuteHandler),
                ("destroy",        _DestroyHandler),
            ]:
                h = handler_cls()
                getattr(cmd, evname).add(h)
                _handlers.append(h)

        except Exception:
            if _ui:
                _ui.messageBox(
                    f"TripteronIK — setup error:\n{traceback.format_exc()}"
                )


class _PreviewHandler(adsk.core.CommandEventHandler):
    """
    Called whenever any command input changes.

    Workflow:
      1. Read desired EE position from the triad transform.
      2. Solve IK via Newton iteration (Fusion is the FK oracle).
      3. The FK oracle leaves Fusion in the solved configuration → live preview.
      4. Compute Jᵀ F for the applied force.
      5. Write results to the text box.
    """

    def notify(self, args: adsk.core.CommandEventArgs):
        if _robot is None:
            return
        try:
            inputs = args.command.commandInputs

            # --- Target position from triad ---
            triad  = inputs.itemById("eeTriad")
            t      = triad.transform.translation
            target = np.array([t.x, t.y, t.z])

            # --- IK ---
            s_sol, err_cm, J = _robot.solve_ik(target)
            # fk(s_sol) was already called inside solve_ik; Fusion now shows the pose.

            # --- Static force estimation ---
            Fx  = inputs.itemById("Fx").value
            Fy  = inputs.itemById("Fy").value
            Fz  = inputs.itemById("Fz").value
            F   = np.array([Fx, Fy, Fz])
            tau = _robot.static_forces(J, F) if np.linalg.norm(F) > 1e-9 else None

            # --- Update display ---
            inputs.itemById("results").text = _format_results(
                _robot, s_sol, err_cm, tau
            )

            # Signal that the preview is a valid model state
            args.isValidResult = True

        except Exception:
            # Swallow errors silently so Fusion isn't disrupted by bad drags.
            try:
                inputs.itemById("results").text = (
                    "IK error — position may be outside workspace.\n"
                    + traceback.format_exc(limit=3)
                )
            except Exception:
                pass


class _ExecuteHandler(adsk.core.CommandEventHandler):
    """Commit the IK solution when the user clicks OK."""

    def notify(self, args: adsk.core.CommandEventArgs):
        if _robot is None:
            return
        try:
            inputs = args.command.commandInputs
            triad  = inputs.itemById("eeTriad")
            t      = triad.transform.translation
            target = np.array([t.x, t.y, t.z])
            s_sol, _, _ = _robot.solve_ik(target)
            _robot.fk(s_sol)   # apply final configuration
        except Exception:
            pass


class _DestroyHandler(adsk.core.CommandEventHandler):
    def notify(self, args: adsk.core.CommandEventArgs):
        global _robot
        _robot = None


# ---------------------------------------------------------------------------
# Add-in entry points
# ---------------------------------------------------------------------------
def run(context):
    global _app, _ui, _cmd_def, _handlers

    try:
        _app = adsk.core.Application.get()
        _ui  = _app.userInterface

        # Clean up any stale definition from a previous load
        stale = _ui.commandDefinitions.itemById(CMD_ID)
        if stale:
            stale.deleteMe()

        _cmd_def = _ui.commandDefinitions.addButtonDefinition(
            CMD_ID, CMD_NAME, CMD_DESC
        )

        on_created = _CreatedHandler()
        _cmd_def.commandCreated.add(on_created)
        _handlers.append(on_created)

        # Place the button in the ADD-INS panel of the Design workspace
        workspace    = _ui.workspaces.itemById(WORKSPACE_ID)
        addins_panel = workspace.toolbarPanels.itemById(PANEL_ID)
        ctrl = addins_panel.controls.addCommand(_cmd_def)
        ctrl.isPromoted = False

    except Exception:
        if _ui:
            _ui.messageBox(
                f"TripteronIK — failed to start:\n{traceback.format_exc()}"
            )


def stop(context):
    global _app, _ui, _cmd_def, _handlers, _robot

    try:
        if _ui:
            workspace    = _ui.workspaces.itemById(WORKSPACE_ID)
            addins_panel = workspace.toolbarPanels.itemById(PANEL_ID)
            ctrl = addins_panel.controls.itemById(CMD_ID)
            if ctrl:
                ctrl.deleteMe()

        if _cmd_def:
            _cmd_def.deleteMe()

    except Exception:
        pass
    finally:
        _handlers.clear()
        _robot    = None
        _cmd_def  = None
