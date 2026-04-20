"""
Tripteron IK Command Add-in
===========================
Adds a command under ADD-INS → Scripts and Add-Ins that places a triad at the
end-effector occurrence and drives the three slider joints to match the dragged
target position.  A finite-difference Jacobian built from Fusion's own FK engine
(transform2) gives actuator force estimates via τ = Jᵀ F.

No third-party packages required — all math uses the Python standard library.

Assumptions
-----------
* The open design contains exactly three slider-type joints.
* At least one occurrence has "ee" anywhere in its name (case-insensitive) —
  that occurrence is treated as the end-effector platform.
"""

import adsk.core
import adsk.fusion
import traceback
import math

# ---------------------------------------------------------------------------
# Module-level globals (kept alive to prevent GC of event handlers)
# ---------------------------------------------------------------------------
_app      = None
_ui       = None
_handlers = []
_robot    = None
_cmd_def  = None

CMD_ID       = "TripteronIK_Cmd"
CMD_NAME     = "Tripteron IK"
CMD_DESC     = "Drive slider joints via EE triad; estimate actuator forces."
WORKSPACE_ID = "FusionSolidEnvironment"
PANEL_ID     = "SolidScriptsAddinsPanel"

FD_EPS  = 5e-4   # finite-difference step (cm ≈ 5 µm)
IK_TOL  = 1e-4   # Newton convergence tolerance (cm ≈ 1 µm)
IK_ITER = 10     # max Newton iterations


# ---------------------------------------------------------------------------
# Pure-Python 3-D vector / 3×3 matrix helpers
# Vectors are plain [x, y, z] lists.
# Matrices are lists of 3 rows: [[r00,r01,r02], [r10,...], [r20,...]].
# ---------------------------------------------------------------------------

def _norm(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def _sub(a, b):
    return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]

def _add(a, b):
    return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]

def _scale(v, s):
    return [v[0]*s, v[1]*s, v[2]*s]

def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def _zeros33():
    return [[0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]]

def _eye3():
    return [[1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]

def _set_col(M, col, v):
    """Write vector v into column col of matrix M in-place."""
    for row in range(3):
        M[row][col] = v[row]

def _col_div(v, s):
    return [v[0]/s, v[1]/s, v[2]/s]

def _matvec(M, v):
    """3×3 M times 3-vector v."""
    return [_dot(M[i], v) for i in range(3)]

def _transpose_matvec(M, v):
    """Mᵀ times v — columns of M dotted with v."""
    return [
        M[0][0]*v[0] + M[1][0]*v[1] + M[2][0]*v[2],
        M[0][1]*v[0] + M[1][1]*v[1] + M[2][1]*v[2],
        M[0][2]*v[0] + M[1][2]*v[1] + M[2][2]*v[2],
    ]

def _solve3(M, b):
    """
    Solve 3×3 linear system M·x = b.
    Uses Gaussian elimination with partial pivoting.
    Raises ValueError if M is singular.
    """
    # Build augmented matrix [M | b], working on copies
    A = [list(M[i]) + [b[i]] for i in range(3)]

    for col in range(3):
        # Partial pivot
        pivot = max(range(col, 3), key=lambda r: abs(A[r][col]))
        A[col], A[pivot] = A[pivot], A[col]
        if abs(A[col][col]) < 1e-14:
            raise ValueError("Singular Jacobian — configuration may be at a singularity.")
        inv = 1.0 / A[col][col]
        for row in range(col + 1, 3):
            f = A[row][col] * inv
            A[row] = [A[row][j] - f * A[col][j] for j in range(4)]


    x = [0.0, 0.0, 0.0]
    for i in range(2, -1, -1):
        s = A[i][3]
        for j in range(i + 1, 3):
            s -= A[i][j] * x[j]
        x[i] = s / A[i][i]
    return x


# ---------------------------------------------------------------------------
# Robot model
# ---------------------------------------------------------------------------
class RobotModel:
    """
    Wraps the three slider joints and the EE occurrence.
    All positions in Fusion internal units (cm).

    FK oracle: set jointMotion.slideValue → Fusion re-solves → read transform2.
    """

    def __init__(self, slider_joints, ee_occ):
        if len(slider_joints) != 3:
            raise ValueError(f"Need exactly 3 slider joints; got {len(slider_joints)}.")
        self.joints  = slider_joints
        self.ee_occ  = ee_occ

        self.s_ref   = [j.jointMotion.slideValue for j in self.joints]
        self.ee_ref  = self._read_ee()
        self._last_s = list(self.s_ref)

    # ------------------------------------------------------------------
    def _read_ee(self):
        """Return EE world position as [x, y, z] (cm)."""
        t = self.ee_occ.transform2.translation
        return [t.x, t.y, t.z]

    def fk(self, s):
        """Set slider values, Fusion solves, return EE position [x,y,z]."""
        for i, jt in enumerate(self.joints):
            jt.jointMotion.slideValue = float(s[i])
        self._last_s = list(s)
        return self._read_ee()

    # ------------------------------------------------------------------
    def jacobian_fd(self, s):
        """
        Forward-difference Jacobian: J[:,i] = (FK(s+ε·eᵢ) − FK(s)) / ε.
        Returns (J, p0).  Restores joints to s on exit.
        """
        p0 = self.fk(s)
        J  = _zeros33()
        for i in range(3):
            sp    = list(s)
            sp[i] += FD_EPS
            dp    = _sub(self.fk(sp), p0)
            _set_col(J, i, _col_div(dp, FD_EPS))
        self.fk(s)   # restore
        return J, p0

    # ------------------------------------------------------------------
    def solve_ik(self, target, s0=None):
        """
        Newton-Raphson IK.  Returns (s_best, residual_cm, J_at_best).
        Leaves Fusion in the best-found configuration.
        """
        s = list(s0 if s0 is not None else self._last_s)
        best_s, best_err, best_J = list(s), float("inf"), _eye3()

        for _ in range(IK_ITER):
            J, p  = self.jacobian_fd(s)
            err_v = _sub(target, p)
            err   = _norm(err_v)
            if err < best_err:
                best_s, best_err, best_J = list(s), err, [list(r) for r in J]
            if err < IK_TOL:
                break
            try:
                ds = _solve3(J, err_v)
                s  = _add(s, ds)
            except ValueError:
                break   # singular — stop at best so far

        self.fk(best_s)
        return best_s, best_err, best_J

    # ------------------------------------------------------------------
    def static_forces(self, J, force):
        """τ = Jᵀ F.  Returns [τ_A, τ_B, τ_C] in Newtons."""
        return _transpose_matvec(J, force)

    def restore_ref(self):
        self.fk(self.s_ref)


# ---------------------------------------------------------------------------
# Joint / occurrence discovery
# ---------------------------------------------------------------------------
def _is_slider_joint(j):
    try:
        return "sliderjointmotion" in j.jointMotion.objectType.lower()
    except Exception:
        return False


def find_robot_components(design):
    root    = design.rootComponent
    sliders = []
    seen:   set = set()

    def _collect(j):
        try:
            tok = j.entityToken
        except Exception:
            tok = str(id(j))
        if tok in seen:
            return
        seen.add(tok)
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

    sliders.sort(key=lambda j: getattr(j, "name", ""))
    sliders = sliders[:3]

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
            "Rename the end-effector occurrence so its name contains 'ee' "
            "(e.g. 'tripteron ee v6:1')."
        )

    return sliders, ee_occ


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------
def _format_results(robot, s_sol, err_cm, tau):
    if s_sol is None:
        return (
            "Drag the triad to a target EE position.\n"
            "Set Fx / Fy / Fz to apply a static load."
        )

    mm  = 10.0
    ref = robot.s_ref

    lines = [
        f"Slider A: {s_sol[0]*mm:7.3f} mm  ({(s_sol[0]-ref[0])*mm:+.3f})",
        f"Slider B: {s_sol[1]*mm:7.3f} mm  ({(s_sol[1]-ref[1])*mm:+.3f})",
        f"Slider C: {s_sol[2]*mm:7.3f} mm  ({(s_sol[2]-ref[2])*mm:+.3f})",
        f"IK residual: {err_cm*mm:.4f} mm",
    ]

    if tau is not None:
        lines += [
            "────────────────────────────",
            f"  \u03c4_A: {tau[0]:+8.3f} N",
            f"  \u03c4_B: {tau[1]:+8.3f} N",
            f"  \u03c4_C: {tau[2]:+8.3f} N",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------
class _CreatedHandler(adsk.core.CommandCreatedEventHandler):
    def notify(self, args):
        global _robot
        try:
            cmd    = args.command
            inputs = cmd.commandInputs
            cmd.isRepeatable = False

            app    = adsk.core.Application.get()
            design = adsk.fusion.Design.cast(app.activeProduct)
            sliders, ee_occ = find_robot_components(design)
            _robot = RobotModel(sliders, ee_occ)

            # Triad at current EE position
            # Triad at world origin
            triad = inputs.addTriadCommandInput("eeTriad", adsk.core.Matrix3D.create())



            try:
                triad.isXRotationEnabled = False
                triad.isYRotationEnabled = False
                triad.isZRotationEnabled = False
            except Exception:
                pass

            # Force spinners
            inputs.addFloatSpinnerCommandInput("Fx", "Fx  (N)", "", -500.0, 500.0, 1.0, 0.0)
            inputs.addFloatSpinnerCommandInput("Fy", "Fy  (N)", "", -500.0, 500.0, 1.0, 0.0)
            inputs.addFloatSpinnerCommandInput("Fz", "Fz  (N)", "", -500.0, 500.0, 1.0, 0.0)

            # Read-only results box
            inputs.addTextBoxCommandInput(
                "results", "Results",
                _format_results(_robot, None, None, None),
                9, True
            )

            for evname, cls in [
                ("executePreview", _PreviewHandler),
                ("execute",        _ExecuteHandler),
                ("destroy",        _DestroyHandler),
            ]:
                h = cls()
                getattr(cmd, evname).add(h)
                _handlers.append(h)

        except Exception:
            if _ui:
                _ui.messageBox(f"TripteronIK — setup error:\n{traceback.format_exc()}")


class _PreviewHandler(adsk.core.CommandEventHandler):
    def notify(self, args):
        if _robot is None:
            return
        try:
            inputs = args.command.commandInputs

            triad  = inputs.itemById("eeTriad")
            t      = triad.transform.translation
            target = [t.x, t.y, t.z]

            s_sol, err_cm, J = _robot.solve_ik(target)

            Fx  = inputs.itemById("Fx").value
            Fy  = inputs.itemById("Fy").value
            Fz  = inputs.itemById("Fz").value
            F   = [Fx, Fy, Fz]
            tau = _robot.static_forces(J, F) if _norm(F) > 1e-9 else None

            inputs.itemById("results").text = _format_results(_robot, s_sol, err_cm, tau)
            args.isValidResult = True

        except Exception:
            try:
                inputs.itemById("results").text = (
                    "IK error — position may be outside workspace.\n"
                    + traceback.format_exc(limit=3)
                )
            except Exception:
                pass


class _ExecuteHandler(adsk.core.CommandEventHandler):
    def notify(self, args):
        if _robot is None:
            return
        try:
            inputs = args.command.commandInputs
            triad  = inputs.itemById("eeTriad")
            t      = triad.transform.translation
            target = [t.x, t.y, t.z]
            s_sol, _, _ = _robot.solve_ik(target)
            _robot.fk(s_sol)
        except Exception:
            pass


class _DestroyHandler(adsk.core.CommandEventHandler):
    def notify(self, args):
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

        stale = _ui.commandDefinitions.itemById(CMD_ID)
        if stale:
            stale.deleteMe()

        _cmd_def = _ui.commandDefinitions.addButtonDefinition(
            CMD_ID, CMD_NAME, CMD_DESC
        )

        on_created = _CreatedHandler()
        _cmd_def.commandCreated.add(on_created)
        _handlers.append(on_created)

        workspace    = _ui.workspaces.itemById(WORKSPACE_ID)
        addins_panel = workspace.toolbarPanels.itemById(PANEL_ID)
        ctrl = addins_panel.controls.addCommand(_cmd_def)
        ctrl.isPromoted = False

    except Exception:
        if _ui:
            _ui.messageBox(f"TripteronIK — failed to start:\n{traceback.format_exc()}")


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
        _robot   = None
        _cmd_def = None
