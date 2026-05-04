"""
Planar 2T1R Parallel Mechanism – Live IK + Actuator Force Analyser
===================================================================
Fusion 360 add-in.  Adds a toolbar button that activates a live command:

  • Drag the on-screen Triad handle to move the EE position (X/Y arrows).
  • Drag the blue arc on the Triad to rotate the platform angle φ (Z rotation).
  • Enter Fx, Fy (N) and Mz (N·cm) to apply a generalised force at the EE.
  • The dialog shows θ_A/B/C and actuator torques τ_A/B/C continuously.
  • The sketch updates in real-time as you drag.

One-time setup
--------------
1.  Name the three ground-revolute angle dimensions in your sketch:
        theta_A  – angle of limb A proximal link from +x axis
        theta_B  – angle of limb B proximal link from +x axis
        theta_C  – angle of limb C proximal link from +x axis
    Right-click each dimension → Change Parameter Name.
2.  Copy the Planar_2T1R/ folder into Fusion's AddIns directory.
3.  Load it once from Scripts & Add-Ins → Add-Ins tab → Run.
    A "2T1R Live IK" button appears in the Solid > Scripts panel.
4.  Click the button any time to activate the live mode.
    Click OK or press Enter to commit; the sketch retains the final pose.

Geometry constants  (edit the block below to match your sketch)
---------------------------------------------------------------
All lengths in the same unit as your Fusion sketch (cm here).
"""

import adsk.core
import adsk.fusion
import math
import traceback

# ── Mechanism geometry ───────────────────────────────────────────────────────

A1  = (0.0,  0.0);  LA1, LA2 = 10.0, 14.0   # limb A: base, proximal, distal
B1  = (8.0,  0.0);  LB1, LB2 = 10.0, 14.0   # limb B
C1  = (20.0, 0.0);  LC1, LC2 = 10.0, 12.0   # limb C
L_PLATFORM = 15.25                             # platform bar length (P_AB → P_C)

# Elbow configuration: +1 or -1 per limb.
# (-1, +1, +1)  →  A elbow left, B and C elbows right ("elbows-out" default).
# If the sketch snaps to a folded / wrong branch, flip one constant and reload.
ELBOW_A, ELBOW_B, ELBOW_C = -1, 1, 1

# Sketch parameter names (must match the names you gave the angle dimensions)
PARAM_A = "theta_A"
PARAM_B = "theta_B"
PARAM_C = "theta_C"

# Initial EE pose when the command opens (cm, cm, rad)
INIT_X, INIT_Y, INIT_PHI = 10.0, 12.0, 0.0

# Finite-difference step for Jacobian (rad / cm)
FD_EPS = 1e-5


# ── IK core (pure Python, embedded for self-containment) ─────────────────────

class IKError(Exception):
    pass


def _two_R(base, l1, l2, target, elbow):
    """
    Ground-joint angle for a single 2R planar limb (law of cosines).

      theta = atan2(dy, dx)  -  elbow * acos((l1^2 + d^2 - l2^2) / (2 l1 d))
    """
    dx, dy = target[0] - base[0], target[1] - base[1]
    d = math.hypot(dx, dy)
    lo, hi = abs(l1 - l2), l1 + l2
    if d > hi + 1e-6:
        raise IKError(f"Unreachable: dist {d:.3f} > max reach {hi:.3f}")
    if d < lo - 1e-6:
        raise IKError(f"Unreachable: dist {d:.3f} < min reach {lo:.3f}")
    d = min(hi, max(lo, d))
    alpha = math.atan2(dy, dx)
    cos_g = (l1*l1 + d*d - l2*l2) / (2.0 * l1 * d)
    gamma = math.acos(max(-1.0, min(1.0, cos_g)))
    return alpha - elbow * gamma


def _ik(x, y, phi):
    """Full 3-limb IK. Returns (theta_A, theta_B, theta_C) in radians."""
    h = 0.5 * L_PLATFORM
    c, s = math.cos(phi), math.sin(phi)
    PAB = (x - h*c, y - h*s)   # shared attachment (limbs A and B)
    PC  = (x + h*c, y + h*s)   # limb C attachment
    return (
        _two_R(A1, LA1, LA2, PAB, ELBOW_A),
        _two_R(B1, LB1, LB2, PAB, ELBOW_B),
        _two_R(C1, LC1, LC2, PC,  ELBOW_C),
    )


# ── 3×3 linear algebra ────────────────────────────────────────────────────────

def _inv33(m):
    """
    Gauss-Jordan 3×3 matrix inversion.
    m  : list of 3 rows, each a list of 3 floats.
    Raises IKError if the matrix is singular.
    """
    a = [r[:] for r in m]
    b = [[float(i == j) for j in range(3)] for i in range(3)]
    for col in range(3):
        pr = max(range(col, 3), key=lambda r: abs(a[r][col]))
        a[col], a[pr] = a[pr], a[col]
        b[col], b[pr] = b[pr], b[col]
        p = a[col][col]
        if abs(p) < 1e-12:
            raise IKError("Jacobian singular – mechanism is near a singularity")
        ip = 1.0 / p
        for j in range(3):
            a[col][j] *= ip
            b[col][j] *= ip
        for row in range(3):
            if row == col:
                continue
            f = a[row][col]
            for j in range(3):
                a[row][j] -= f * a[col][j]
                b[row][j] -= f * b[col][j]
    return b


def _actuator_torques(x, y, phi, Fx, Fy, Mz):
    """
    τ = Jᵀ · [Fx, Fy, Mz]

    Algorithm
    ---------
    1. Compute J⁻¹ numerically via forward differences on the IK.
       J⁻¹[i][j] = ∂θᵢ/∂(ee_j)  where ee = [x, y, φ].
    2. Invert to get J (forward Jacobian).
    3. τ[i] = Σⱼ J[j][i] · F[j]   (column i of J dotted with F).

    Units: lengths in cm, forces in N  →  torques in N·cm.
    """
    t0 = _ik(x, y, phi)
    cols = []
    for dx, dy, dp in [(FD_EPS, 0, 0), (0, FD_EPS, 0), (0, 0, FD_EPS)]:
        t1 = _ik(x + dx, y + dy, phi + dp)
        cols.append([(t1[i] - t0[i]) / FD_EPS for i in range(3)])

    # Jinv[i][j] = ∂θᵢ/∂(ee_j) = cols[j][i]
    Jinv = [[cols[j][i] for j in range(3)] for i in range(3)]
    J    = _inv33(Jinv)

    F   = [Fx, Fy, Mz]
    tau = [sum(J[j][i] * F[j] for j in range(3)) for i in range(3)]
    return tau


# ── Fusion add-in state (prevent GC of event handlers) ───────────────────────

_handlers = []
_cmd_def  = None
_btn_ctrl = None


# ── Add-in lifecycle ──────────────────────────────────────────────────────────

def run(context):
    global _cmd_def, _btn_ctrl
    app = adsk.core.Application.get()
    ui  = app.userInterface
    try:
        _cleanup(ui)

        _cmd_def = ui.commandDefinitions.addButtonDefinition(
            "Planar2T1RIK_Cmd",
            "2T1R Live IK",
            "Drag the Triad to pose the mechanism.\n"
            "Enter Fx, Fy, Mz to see required actuator torques.\n"
            "Sketch angle dimensions theta_A / theta_B / theta_C update live."
        )
        h = _CreatedHandler()
        _cmd_def.commandCreated.add(h)
        _handlers.append(h)

        panel = ui.allToolbarPanels.itemById("SolidScriptsAddinsPanel")
        _btn_ctrl = panel.controls.addCommand(_cmd_def)
        _btn_ctrl.isPromotedByDefault = True

    except Exception:
        ui.messageBox(traceback.format_exc())


def stop(context):
    global _handlers, _cmd_def, _btn_ctrl
    _cleanup(adsk.core.Application.get().userInterface)
    _handlers.clear()
    _cmd_def  = None
    _btn_ctrl = None


def _cleanup(ui):
    old = ui.commandDefinitions.itemById("Planar2T1RIK_Cmd")
    if old:
        old.deleteMe()
    panel = ui.allToolbarPanels.itemById("SolidScriptsAddinsPanel")
    if panel:
        ctrl = panel.controls.itemById("Planar2T1RIK_Cmd")
        if ctrl:
            ctrl.deleteMe()


# ── Event handlers ────────────────────────────────────────────────────────────

class _CreatedHandler(adsk.core.CommandCreatedEventHandler):
    def notify(self, args):
        try:
            cmd    = args.command
            cmd.isRepeatable = False
            inputs = cmd.commandInputs

            # ── Triad at initial EE position ──────────────────────────────────
            # INIT_X / INIT_Y are in sketch display units (e.g. inches).
            # Matrix3D.translation must be in cm (Fusion API internal unit),
            # so we divide by the cm→display-unit scale factor.
            scale, _ = _sketch_unit()
            mat = adsk.core.Matrix3D.create()
            mat.translation = adsk.core.Vector3D.create(
                INIT_X / scale, INIT_Y / scale, 0.0
            )
            inputs.addTriadCommandInput("triad", mat)

            # ── Applied generalised force inputs ──────────────────────────────
            vr = adsk.core.ValueInput.createByReal
            inputs.addValueInput("Fx", "Applied Fx  (N)",    "", vr(0.0))
            inputs.addValueInput("Fy", "Applied Fy  (N)",    "", vr(0.0))
            inputs.addValueInput("Mz", "Applied Mz  (N·cm)", "", vr(0.0))

            # ── Read-only results panel ───────────────────────────────────────
            inputs.addTextBoxCommandInput(
                "out", "Kinematics / Forces",
                "Drag the Triad to begin.", 7, True
            )

            # ── Wire up preview and change handlers ───────────────────────────
            for cls, evt in (
                (_PreviewHandler,      cmd.executePreview),
                (_InputChangedHandler, cmd.inputChanged),
                (_DestroyHandler,      cmd.destroy),
            ):
                h = cls()
                evt.add(h)
                _handlers.append(h)

        except Exception:
            adsk.core.Application.get().userInterface.messageBox(
                traceback.format_exc()
            )


def _sketch_unit():
    """
    Return (scale, label) where scale converts Fusion API values (always cm)
    to the document's current display length unit.

    Example: document in inches → scale ≈ 0.3937, label = "in"
    """
    try:
        design = adsk.fusion.Design.cast(
            adsk.core.Application.get().activeProduct
        )
        um    = design.unitsManager
        label = um.defaultLengthUnits          # e.g. "in", "cm", "mm"
        scale = um.convert(1.0, "cm", label)   # e.g. 0.3937 for inches
        return scale, label
    except Exception:
        return 1.0, "cm"


def _read_triad(inputs):
    """
    Extract (x, y, phi) from the Triad transform.

    Fusion's Matrix3D always uses cm internally.  We convert x, y to the
    document's display unit so they match the geometry constants (INIT_X,
    link lengths, base positions) which the user specifies in sketch units.

    phi is recovered by transforming the unit-X vector through the rotation
    only (Vector3D.transformBy ignores translation), then taking atan2.
    """
    xf         = inputs.itemById("triad").transform
    t          = xf.translation
    scale, _   = _sketch_unit()
    x, y       = t.x * scale, t.y * scale

    xv = adsk.core.Vector3D.create(1.0, 0.0, 0.0)
    xv.transformBy(xf)
    phi = math.atan2(xv.y, xv.x)
    return x, y, phi


def _do_update(inputs):
    """Core update: read Triad → IK → sketch params → torques → display."""
    out = inputs.itemById("out")
    try:
        x, y, phi = _read_triad(inputs)
        Fx = inputs.itemById("Fx").value
        Fy = inputs.itemById("Fy").value
        Mz = inputs.itemById("Mz").value

        # ── IK ───────────────────────────────────────────────────────────────
        tA, tB, tC = _ik(x, y, phi)

        # ── Drive sketch angle parameters ─────────────────────────────────────
        design = adsk.fusion.Design.cast(
            adsk.core.Application.get().activeProduct
        )
        missing = []
        if design:
            ap = design.allParameters
            for name, angle in ((PARAM_A, tA), (PARAM_B, tB), (PARAM_C, tC)):
                p = ap.itemByName(name)
                if p:
                    p.expression = f"{math.degrees(angle):.8f} deg"
                else:
                    missing.append(name)

        # ── Actuator torques ──────────────────────────────────────────────────
        tau = _actuator_torques(x, y, phi, Fx, Fy, Mz)

        # ── Results display ───────────────────────────────────────────────────
        warn = (f"\n⚠  Parameters not found: {', '.join(missing)}"
                if missing else "")
        _, ulabel = _sketch_unit()
        out.text = (
            f"EE  =  ({x:.3f},  {y:.3f}) {ulabel}     φ = {math.degrees(phi):.2f}°\n"
            f"\n"
            f"θ_A = {math.degrees(tA):>8.3f}°     τ_A = {tau[0]:>9.4f} N·{ulabel}\n"
            f"θ_B = {math.degrees(tB):>8.3f}°     τ_B = {tau[1]:>9.4f} N·{ulabel}\n"
            f"θ_C = {math.degrees(tC):>8.3f}°     τ_C = {tau[2]:>9.4f} N·{ulabel}"
            + warn
        )

    except IKError as e:
        if out:
            out.text = f"IK Error – pose unreachable:\n{e}"
    except Exception:
        if out:
            out.text = traceback.format_exc()[:400]


class _PreviewHandler(adsk.core.CommandEventHandler):
    def notify(self, args):
        _do_update(args.command.commandInputs)
        args.isValidResult = True


class _InputChangedHandler(adsk.core.InputChangedEventHandler):
    def notify(self, args):
        _do_update(args.inputs)


class _DestroyHandler(adsk.core.CommandEventHandler):
    def notify(self, args):
        pass  # add-in stays loaded; button remains until stop() is called
