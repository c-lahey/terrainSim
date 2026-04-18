import adsk.core
import adsk.fusion
import adsk.cam
import traceback
import json
import os
from math import isnan

# -----------------------------
# Small geometry / serialization helpers
# -----------------------------

def matrix_to_list(m: adsk.core.Matrix3D):
    arr = m.asArray()  # 16 values, row-major in Fusion
    return [
        [arr[0],  arr[1],  arr[2],  arr[3]],
        [arr[4],  arr[5],  arr[6],  arr[7]],
        [arr[8],  arr[9],  arr[10], arr[11]],
        [arr[12], arr[13], arr[14], arr[15]],
    ]

def safe_float(x):
    try:
        v = float(x)
        if isnan(v):
            return None
        return v
    except:
        return None

def point_to_list(p: adsk.core.Point3D):
    if not p:
        return None
    return [safe_float(p.x), safe_float(p.y), safe_float(p.z)]

def vector_to_list(v: adsk.core.Vector3D):
    if not v:
        return None
    return [safe_float(v.x), safe_float(v.y), safe_float(v.z)]

def transform_point(m: adsk.core.Matrix3D, p: adsk.core.Point3D):
    if not m or not p:
        return None
    q = p.copy()
    q.transformBy(m)
    return q

def transform_vector(m: adsk.core.Matrix3D, v: adsk.core.Vector3D):
    if not m or not v:
        return None
    q = v.copy()
    q.transformBy(m)
    return q

def origin_of_matrix(m: adsk.core.Matrix3D):
    if not m:
        return None
    return adsk.core.Point3D.create(m.translation.x, m.translation.y, m.translation.z)

def axis_x_of_matrix(m: adsk.core.Matrix3D):
    arr = m.asArray()
    return adsk.core.Vector3D.create(arr[0], arr[4], arr[8])

def axis_y_of_matrix(m: adsk.core.Matrix3D):
    arr = m.asArray()
    return adsk.core.Vector3D.create(arr[1], arr[5], arr[9])

def axis_z_of_matrix(m: adsk.core.Matrix3D):
    arr = m.asArray()
    return adsk.core.Vector3D.create(arr[2], arr[6], arr[10])

def get_entity_token(obj):
    try:
        return obj.entityToken
    except:
        return None

def get_occ_full_path_name(occ):
    try:
        return occ.fullPathName
    except:
        try:
            return occ.name
        except:
            return None

def get_occurrence_world_transform(occ):
    # Use transform2, not transform.
    try:
        return occ.transform2
    except:
        return None

def try_getattr(obj, name, default=None):
    try:
        return getattr(obj, name)
    except:
        return default

def safe_value_from_parameter(param):
    if not param:
        return None
    try:
        return safe_float(param.value)
    except:
        try:
            return str(param.expression)
        except:
            return None

# -----------------------------
# Joint typing helpers
# -----------------------------

def classify_joint_motion(joint_motion):
    if not joint_motion:
        return "unknown"

    ot = ""
    try:
        ot = joint_motion.objectType
    except:
        pass

    ot = ot.lower()

    if "revolutejointmotion" in ot:
        return "revolute"
    if "sliderjointmotion" in ot:
        return "slider"
    if "cylindricaljointmotion" in ot:
        return "cylindrical"
    if "pinslotjointmotion" in ot:
        return "pin_slot"
    if "planarjointmotion" in ot:
        return "planar"
    if "balljointmotion" in ot:
        return "ball"

    return "unknown"

def extract_motion_data(joint_motion, occ_two_transform=None):
    """
    Pull motion info from the joint motion object when available.
    Many direction vectors are typically expressed in the second geometry's context,
    so using occurrenceTwo world transform is a decent first-pass way to move them
    to world coordinates.
    """
    data = {
        "motion_class": None,
        "joint_type": classify_joint_motion(joint_motion),
        "rotation_axis_world": None,
        "slide_direction_world": None,
        "rotation_value": None,
        "slide_value": None,
        "rotation_limits": None,
        "slide_limits": None,
    }

    if not joint_motion:
        return data

    try:
        data["motion_class"] = joint_motion.objectType
    except:
        pass

    jt = data["joint_type"]

    def world_vec(v):
        if not v:
            return None
        if occ_two_transform:
            try:
                vv = transform_vector(occ_two_transform, v)
                return vector_to_list(vv)
            except:
                pass
        return vector_to_list(v)

    try:
        if jt == "revolute":
            data["rotation_axis_world"] = world_vec(joint_motion.rotationAxisVector)
            data["rotation_value"] = safe_float(joint_motion.rotationValue)
            lim = joint_motion.rotationLimits
            if lim:
                data["rotation_limits"] = {
                    "is_minimum_value_enabled": try_getattr(lim, "isMinimumValueEnabled"),
                    "is_maximum_value_enabled": try_getattr(lim, "isMaximumValueEnabled"),
                    "minimum_value": safe_float(try_getattr(lim, "minimumValue")),
                    "maximum_value": safe_float(try_getattr(lim, "maximumValue")),
                }

        elif jt == "slider":
            data["slide_direction_world"] = world_vec(joint_motion.slideDirectionVector)
            data["slide_value"] = safe_float(joint_motion.slideValue)
            lim = joint_motion.slideLimits
            if lim:
                data["slide_limits"] = {
                    "is_minimum_value_enabled": try_getattr(lim, "isMinimumValueEnabled"),
                    "is_maximum_value_enabled": try_getattr(lim, "isMaximumValueEnabled"),
                    "minimum_value": safe_float(try_getattr(lim, "minimumValue")),
                    "maximum_value": safe_float(try_getattr(lim, "maximumValue")),
                }

        elif jt == "cylindrical":
            data["rotation_axis_world"] = world_vec(joint_motion.rotationAxisVector)
            data["slide_direction_world"] = world_vec(joint_motion.slideDirectionVector)
            data["rotation_value"] = safe_float(joint_motion.rotationValue)
            data["slide_value"] = safe_float(joint_motion.slideValue)

        elif jt == "pin_slot":
            data["rotation_axis_world"] = world_vec(joint_motion.rotationAxisVector)
            data["slide_direction_world"] = world_vec(joint_motion.slideDirectionVector)
            data["rotation_value"] = safe_float(joint_motion.rotationValue)
            data["slide_value"] = safe_float(joint_motion.slideValue)

        elif jt == "planar":
            # Planar is more complex; keep what we can.
            data["rotation_value"] = safe_float(try_getattr(joint_motion, "rotationValue"))
            data["slide_value"] = None

        elif jt == "ball":
            # Ball joints don't give a single axis in the same way.
            pass
    except:
        # Don't kill export if one motion property fails.
        pass

    return data

# -----------------------------
# Joint geometry extraction
# -----------------------------

def joint_occurrence_info(j):
    occ1 = None
    occ2 = None

    for name in ["occurrenceOne", "occurrence1", "entityOne", "occurrence"]:
        try:
            occ1 = getattr(j, name)
            if occ1:
                break
        except:
            pass

    try:
        occ2 = j.occurrenceTwo
    except:
        pass

    return occ1, occ2

def extract_joint_frame_guess(j, occ2_transform=None):
    """
    Best-effort origin/basis extraction.

    For many joint objects, the geometry/origin on side two is the natural frame
    used by the motion direction vectors. We try a few likely properties. If we
    can't get a true joint frame, we fall back to origin only.
    """
    out = {
        "origin_world": None,
        "x_axis_world": None,
        "y_axis_world": None,
        "z_axis_world": None,
    }

    candidate = None

    # Try common properties found on Joint / AsBuiltJoint / JointOrigin-style entities.
    prop_names = [
        "geometryOrOriginTwo",
        "geometryOrOriginOne",
        "jointOrigin",
        "jointGeometry",
        "origin",
    ]

    for pn in prop_names:
        try:
            candidate = getattr(j, pn)
            if candidate:
                break
        except:
            pass

    if not candidate:
        return out

    # Candidate may itself expose an origin / geometry / transform.
    try:
        if hasattr(candidate, "origin"):
            p = candidate.origin
            if p:
                out["origin_world"] = point_to_list(transform_point(occ2_transform, p) if occ2_transform else p)
    except:
        pass

    # If the candidate has a full transform, use it.
    try:
        if hasattr(candidate, "transform"):
            m = candidate.transform
            if m:
                mw = m.copy()
                if occ2_transform:
                    mw.transformBy(occ2_transform)
                out["origin_world"] = point_to_list(origin_of_matrix(mw))
                out["x_axis_world"] = vector_to_list(axis_x_of_matrix(mw))
                out["y_axis_world"] = vector_to_list(axis_y_of_matrix(mw))
                out["z_axis_world"] = vector_to_list(axis_z_of_matrix(mw))
                return out
    except:
        pass

    return out

def extract_joint_record(j, source_kind):
    name = try_getattr(j, "name")
    token = get_entity_token(j)

    occ1, occ2 = joint_occurrence_info(j)
    occ1_name = get_occ_full_path_name(occ1) if occ1 else None
    occ2_name = get_occ_full_path_name(occ2) if occ2 else None
    occ2_tf = get_occurrence_world_transform(occ2) if occ2 else None

    joint_motion = try_getattr(j, "jointMotion")
    motion_data = extract_motion_data(joint_motion, occ2_tf)
    frame_data = extract_joint_frame_guess(j, occ2_tf)

    rec = {
        "name": name,
        "entity_token": token,
        "source_kind": source_kind,
        "object_type": try_getattr(j, "objectType"),
        "occurrence_one": occ1_name,
        "occurrence_two": occ2_name,
        "is_flipped": try_getattr(j, "isFlipped"),
        "is_suppressed": try_getattr(j, "isSuppressed"),
        "joint_type": motion_data["joint_type"],
        "motion": motion_data,
        "frame_guess": frame_data,
        "angle_parameter_value": safe_value_from_parameter(try_getattr(j, "angle")),
        "offset_parameter_value": safe_value_from_parameter(try_getattr(j, "offset")),
    }

    return rec

# -----------------------------
# Occurrence extraction
# -----------------------------

def extract_occurrence_record(occ):
    tf = get_occurrence_world_transform(occ)

    rec = {
        "name": try_getattr(occ, "name"),
        "full_path_name": get_occ_full_path_name(occ),
        "entity_token": get_entity_token(occ),
        "component_name": try_getattr(try_getattr(occ, "component"), "name"),
        "is_grounded": try_getattr(occ, "isGrounded"),
        "is_visible": try_getattr(occ, "isVisible"),
        "is_referenced_component": try_getattr(occ, "isReferencedComponent"),
        "transform_world": matrix_to_list(tf) if tf else None,
    }
    return rec

# -----------------------------
# Joint collection
# -----------------------------

def add_joint_if_new(lst, seen_tokens, j, source_kind):
    if not j:
        return
    tok = get_entity_token(j)
    key = tok if tok else f"{source_kind}:{id(j)}"
    if key in seen_tokens:
        return
    seen_tokens.add(key)
    lst.append(extract_joint_record(j, source_kind))

def collect_joints(root_comp, all_occurrences):
    joints_out = []
    seen = set()

    # Root component joints
    try:
        for i in range(root_comp.joints.count):
            add_joint_if_new(joints_out, seen, root_comp.joints.item(i), "Joint")
    except:
        pass

    # Root component as-built joints
    try:
        abj = root_comp.asBuiltJoints
        for i in range(abj.count):
            add_joint_if_new(joints_out, seen, abj.item(i), "AsBuiltJoint")
    except:
        pass

    # Occurrence-level joint proxies
    for occ in all_occurrences:
        try:
            occ_joints = occ.joints
            for i in range(occ_joints.count):
                add_joint_if_new(joints_out, seen, occ_joints.item(i), "OccurrenceJointProxy")
        except:
            pass

        # Try occurrence component as-built joints too.
        try:
            comp = occ.component
            abj = comp.asBuiltJoints
            for i in range(abj.count):
                j = abj.item(i)

                # Try to create a proxy in this occurrence context if available.
                try:
                    if hasattr(j, "createForAssemblyContext"):
                        proxy = j.createForAssemblyContext(occ)
                        add_joint_if_new(joints_out, seen, proxy, "AsBuiltJointProxy")
                    else:
                        add_joint_if_new(joints_out, seen, j, "AsBuiltJoint")
                except:
                    add_joint_if_new(joints_out, seen, j, "AsBuiltJoint")
        except:
            pass

    return joints_out

# -----------------------------
# Main export logic
# -----------------------------

def build_export_data(design: adsk.fusion.Design):
    root = design.rootComponent

    occs = []
    try:
        for i in range(root.allOccurrences.count):
            occs.append(root.allOccurrences.item(i))
    except:
        # fallback
        pass

    occurrence_records = [extract_occurrence_record(o) for o in occs]
    joint_records = collect_joints(root, occs)

    out = {
        "document_name": try_getattr(try_getattr(adsk.core.Application.get(), "activeDocument"), "name"),
        "root_component": try_getattr(root, "name"),
        "unit_note": "Fusion geometric values are typically in cm and rad unless otherwise documented by property.",
        "occurrence_count": len(occurrence_records),
        "joint_count": len(joint_records),
        "occurrences": occurrence_records,
        "joints": joint_records,
    }

    return out

def choose_output_path(ui):
    try:
        dlg = ui.createFolderDialog()
        dlg.title = "Choose export folder for kinematics JSON"
        if dlg.showDialog() != adsk.core.DialogResults.DialogOK:
            return None
        folder = dlg.folder
        return os.path.join(folder, "fusion_kinematics_export.json")
    except:
        # Fallback to home directory if folder dialog is unavailable
        return os.path.join(os.path.expanduser("~"), "fusion_kinematics_export.json")

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        product = app.activeProduct
        design = adsk.fusion.Design.cast(product)

        if not design:
            ui.messageBox("No active Fusion design.")
            return

        data = build_export_data(design)
        out_path = choose_output_path(ui)
        if not out_path:
            ui.messageBox("Export cancelled.")
            return

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        ui.messageBox(
            f"Kinematics export complete.\n\n"
            f"Occurrences: {data['occurrence_count']}\n"
            f"Joints: {data['joint_count']}\n\n"
            f"Saved to:\n{out_path}"
        )

    except:
        if ui:
            ui.messageBox("Failed:\n{}".format(traceback.format_exc()))