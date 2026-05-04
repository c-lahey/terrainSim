"""
Fusion 360 script – Planar 2T1R IK solver
==========================================
Solves IK for a planar 2T1R 3-DOF parallel mechanism (Fig. 3b topology) and
drives the three ground-revolute angle dimensions in the active sketch directly.

How to use
----------
1.  Open your sketch in Fusion 360 (it must be the active edit target).
2.  Name the three ground-revolute angle dimensions in your sketch:
        theta_A  – angle of limb A's proximal link from the +x axis
        theta_B  – angle of limb B's proximal link from the +x axis
        theta_C  – angle of limb C's proximal link from the +x axis
    To name a dimension: right-click it → Edit → click the name field at top.
    These become model parameters accessible via design.allParameters.
3.  Run this script from the Fusion 360 Scripts & Add-Ins panel.
4.  Enter the target EE pose in the dialog: x, y (cm) and phi (degrees).
5.  The sketch updates live.

Elbow configuration
-------------------
The constants ELBOW_A, ELBOW_B, ELBOW_C select which of the two geometric
branches each limb uses (+1 or -1). The defaults match the "elbows-out"
reference pose visible in the sketch. If the sketch snaps to a folded or
flipped configuration, flip one of these constants and re-run.

Geometry (must match your sketch)
----------------------------------
All lengths in the same unit as your Fusion sketch (cm here).
"""

import adsk.core
import adsk.fusion
import math
import traceback


# ---------------------------------------------------------------------------
# Mechanism geometry – edit these to match your sketch
# ---------------------------------------------------------------------------

A1  = (0.0,  0.0)   # ground revolute A position
B1  = (8.0,  0.0)   # ground revolute B position
C1  = (20.0, 0.0)   # ground revolute C position

LA1, LA2 = 10.0, 14.0   # limb A link lengths (proximal, distal)
LB1, LB2 = 10.0, 14.0   # limb B link lengths
LC1, LC2 = 10.0, 12.0   # limb C link lengths

L_PLATFORM = 15.25       # platform bar length (P_AB to P_C)

# Elbow configuration: +1 or -1 for each limb.
# Default: A elbow left, B and C elbows right (open / elbows-out).
ELBOW_A = -1
ELBOW_B =  1
ELBOW_C =  1

# Names of the sketch angle dimensions to drive
PARAM_THETA_A = "theta_A"
PARAM_THETA_B = "theta_B"
PARAM_THETA_C = "theta_C"


# ---------------------------------------------------------------------------
# IK core (self-contained, no external imports)
# ---------------------------------------------------------------------------

class IKError(Exception):
    pass


def _two_R(base, l1, l2, target, elbow):
    """
    2R planar IK – returns ground-joint angle (rad).

      theta = atan2(dy,dx)  -  elbow * acos((l1^2 + d^2 - l2^2) / (2 l1 d))
    """
    dx, dy = target[0] - base[0], target[1] - base[1]
    d = math.hypot(dx, dy)
    lo, hi = abs(l1 - l2), l1 + l2
    if d > hi + 1e-6:
        raise IKError(f"Target too far: dist {d:.4f} > {hi:.4f}")
    if d < lo - 1e-6:
        raise IKError(f"Target too close: dist {d:.4f} < {lo:.4f}")
    d = min(hi, max(lo, d))
    alpha = math.atan2(dy, dx)
    cos_g = (l1*l1 + d*d - l2*l2) / (2.0 * l1 * d)
    gamma = math.acos(max(-1.0, min(1.0, cos_g)))
    return alpha - elbow * gamma


def _solve_ik(x, y, phi):
    """
    Closed-form IK.  Returns (theta_A, theta_B, theta_C) in radians.
    Raises IKError if any limb cannot reach its target.

    Platform attachments:
      P_AB = EE - (L/2)[cos phi, sin phi]   <- shared by limbs A and B
      P_C  = EE + (L/2)[cos phi, sin phi]   <- limb C only
    """
    h = 0.5 * L_PLATFORM
    c, s = math.cos(phi), math.sin(phi)
    P_AB = (x - h*c, y - h*s)
    P_C  = (x + h*c, y + h*s)

    theta_A = _two_R(A1, LA1, LA2, P_AB, ELBOW_A)
    theta_B = _two_R(B1, LB1, LB2, P_AB, ELBOW_B)
    theta_C = _two_R(C1, LC1, LC2, P_C,  ELBOW_C)
    return theta_A, theta_B, theta_C


# ---------------------------------------------------------------------------
# Fusion entry point
# ---------------------------------------------------------------------------

def run(context):
    app = adsk.core.Application.get()
    ui  = app.userInterface

    try:
        # --- collect target pose from user -----------------------------------
        result, cancelled = ui.inputBox(
            "Enter target EE pose:   x,  y,  phi\n"
            "\n"
            "  x, y  = EE position (cm)\n"
            "  phi   = platform angle from horizontal (degrees)\n"
            "          positive = CCW\n"
            "\n"
            "Example:  10, 12, 0",
            "2T1R IK Solver",
            "10, 12, 0"
        )
        if cancelled:
            return

        try:
            parts = [s.strip() for s in result.split(",")]
            if len(parts) != 3:
                raise ValueError("Need exactly three comma-separated values.")
            x, y, phi_deg = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError as e:
            ui.messageBox(f"Bad input: {e}\n\nExpected format:  x, y, phi_deg")
            return

        phi = math.radians(phi_deg)

        # --- solve IK --------------------------------------------------------
        try:
            theta_A, theta_B, theta_C = _solve_ik(x, y, phi)
        except IKError as e:
            ui.messageBox(f"IK Error – pose unreachable:\n{e}")
            return

        deg_A = math.degrees(theta_A)
        deg_B = math.degrees(theta_B)
        deg_C = math.degrees(theta_C)

        # --- drive sketch parameters -----------------------------------------
        design = adsk.fusion.Design.cast(app.activeProduct)
        if design is None:
            ui.messageBox("No active Fusion design found.")
            return

        all_params = design.allParameters
        param_map = {
            PARAM_THETA_A: (theta_A, deg_A),
            PARAM_THETA_B: (theta_B, deg_B),
            PARAM_THETA_C: (theta_C, deg_C),
        }

        missing = []
        set_ok  = []
        for name, (rad, deg) in param_map.items():
            param = all_params.itemByName(name)
            if param is None:
                missing.append(name)
            else:
                # Fusion sketch angle dimensions use degrees as internal unit.
                # Negative angles are valid and flip direction.
                param.expression = f"{deg:.8f} deg"
                set_ok.append(f"  {name} = {deg:.3f}°")

        # --- report result ---------------------------------------------------
        lines = [
            f"IK solved for EE = ({x}, {y}) cm,  φ = {phi_deg}°",
            "",
            f"  θ_A = {deg_A:.4f}°",
            f"  θ_B = {deg_B:.4f}°",
            f"  θ_C = {deg_C:.4f}°",
        ]

        if set_ok:
            lines += ["", "Sketch parameters updated:"] + set_ok

        if missing:
            lines += [
                "",
                "⚠️  The following parameters were not found in the design:",
            ] + [f"  • {n}" for n in missing] + [
                "",
                "Name your sketch angle dimensions as shown above,",
                "then re-run the script.",
            ]

        ui.messageBox("\n".join(lines), "2T1R IK Result")

    except Exception:
        ui.messageBox(f"Unexpected error:\n{traceback.format_exc()}")
