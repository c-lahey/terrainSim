from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]


# =========================
# Basic vector utilities
# =========================
def vec(x: Sequence[float]) -> Vec3:
    if len(x) != 3:
        raise ValueError(f"Expected 3-vector, got {x}")
    return (float(x[0]), float(x[1]), float(x[2]))


def v_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_scale(s: float, a: Vec3) -> Vec3:
    return (s * a[0], s * a[1], s * a[2])


def v_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def v_norm(a: Vec3) -> float:
    return math.sqrt(v_dot(a, a))


def v_unit(a: Vec3, tol: float = 1e-12) -> Vec3:
    n = v_norm(a)
    if n < tol:
        raise ValueError(f"Cannot normalize near-zero vector {a}")
    return v_scale(1.0 / n, a)


def v_mean(points: Sequence[Vec3]) -> Vec3:
    if not points:
        raise ValueError("Cannot average empty point set")
    inv = 1.0 / len(points)
    return (
        inv * sum(p[0] for p in points),
        inv * sum(p[1] for p in points),
        inv * sum(p[2] for p in points),
    )


def dist(a: Vec3, b: Vec3) -> float:
    return v_norm(v_sub(a, b))


def close_vec(a: Vec3, b: Vec3, tol: float = 1e-8) -> bool:
    return dist(a, b) <= tol


def rvec(a: Vec3, ndigits: int = 12) -> Vec3:
    return (round(a[0], ndigits), round(a[1], ndigits), round(a[2], ndigits))


# =========================
# Reduced parameter schema
# =========================
@dataclass
class JointRecord:
    name: str
    joint_type: str
    occurrence_one: Optional[str]
    occurrence_two: Optional[str]
    origin_world_one: Vec3
    origin_world_two: Optional[Vec3]
    rotation_axis_world: Optional[Vec3]
    slide_direction_world: Optional[Vec3]
    slide_value: Optional[float]


@dataclass
class SliderRecord:
    name: str
    carriage_point_world: Vec3
    rail_reference_origin_world: Vec3
    slide_direction_world: Vec3
    slide_value: float


@dataclass
class CouplingPoint:
    joint_name: str
    point_world: Vec3
    offset_from_centroid: Vec3


@dataclass
class SimpleBranch:
    name: str
    slider_joint_name: str
    rail_origin_world: Vec3
    rail_direction_world: Vec3
    slider_value: float
    ee_joint_name: str
    ee_point_world: Vec3
    ee_offset_from_centroid: Vec3
    chain_joint_names: List[str]
    chain_joint_points_world: List[Vec3]
    rigid_segment_lengths: List[float]


@dataclass
class ClosedBranch:
    name: str
    slider_joint_name: str
    rail_origin_world: Vec3
    rail_direction_world: Vec3
    slider_value: float
    ee_joint_names: List[str]
    ee_points_world: List[Vec3]
    ee_offsets_from_centroid: List[Vec3]
    chain_joint_names: List[str]
    chain_joint_points_world: List[Vec3]
    rigid_lengths: Dict[str, float]


@dataclass
class ReducedParameters:
    source_file: str
    root_component: Optional[str]
    units_note: Optional[str]
    world_x: Vec3
    world_y: Vec3
    world_z: Vec3
    ee_centroid_world: Vec3
    ee_coupling_points: List[CouplingPoint]
    sliders: Dict[str, SliderRecord]
    branch_A: SimpleBranch
    branch_B: SimpleBranch
    branch_C: ClosedBranch


@dataclass
class ValidationReport:
    passed: bool
    messages: List[str] = field(default_factory=list)


# =========================
# Extractor
# =========================
class TripteronJSONExtractor:
    """
    Extract a reduced symbolic-model parameter set directly from the Fusion JSON export.

    This version is intentionally explicit for your current mechanism naming scheme.
    It uses world-frame joint geometry as the source of truth.
    """

    EE_OCCURRENCE = "tripteron ee v10:1"

    EE_COUPLING_JOINTS = [
        "Revolute 24",  # upper-left
        "Revolute 12",  # lower-left
        "Revolute 30",  # upper-right
        "Revolute 32",  # lower-right
    ]

    SLIDERS = {
        "A": "Slider 21",
        "B": "Slider 8",
        "C": "Slider 14",
    }

    SIMPLE_BRANCH_JOINTS = {
        "A": ["Revolute 22", "Revolute 2", "Revolute 24"],
        "B": ["Revolute 9", "Revolute 10", "Revolute 12"],
    }

    CLOSED_BRANCH_JOINTS = {
        "C": ["Revolute 29", "Revolute 31", "Revolute 16", "Revolute 19", "Revolute 30", "Revolute 32"],
    }

    def __init__(self, fusion_json: Dict[str, Any], tol: float = 1e-8):
        self.data = fusion_json
        self.tol = tol
        self.joints: Dict[str, JointRecord] = {}
        self._index_joints()

    def _index_joints(self) -> None:
        for raw in self.data.get("joints", []):
            motion = raw.get("motion", {})
            g1 = raw.get("geometry_or_origin_one") or {}
            g2 = raw.get("geometry_or_origin_two") or {}
            rec = JointRecord(
                name=raw["name"],
                joint_type=motion.get("joint_type", "unknown"),
                occurrence_one=raw.get("occurrence_one"),
                occurrence_two=raw.get("occurrence_two"),
                origin_world_one=vec(g1["origin_world"]),
                origin_world_two=vec(g2["origin_world"]) if g2.get("origin_world") is not None else None,
                rotation_axis_world=vec(motion["rotation_axis_world"]) if motion.get("rotation_axis_world") is not None else None,
                slide_direction_world=vec(motion["slide_direction_world"]) if motion.get("slide_direction_world") is not None else None,
                slide_value=float(motion["slide_value"]) if motion.get("slide_value") is not None else None,
            )
            self.joints[rec.name] = rec

    def joint(self, name: str) -> JointRecord:
        if name not in self.joints:
            raise KeyError(f"Joint '{name}' not found in JSON")
        return self.joints[name]

    def joint_point(self, name: str) -> Vec3:
        j = self.joint(name)
        p1 = j.origin_world_one
        p2 = j.origin_world_two

        # Revolute joints: both sides should represent the same physical joint center.
        if j.joint_type == "revolute":
            if p2 is not None and not close_vec(p1, p2, self.tol):
                raise ValueError(f"Joint '{name}' has mismatched revolute world origins: {p1} vs {p2}")
            return p1

        # Slider joints: side one is the current carriage location, side two is the rail/reference origin.
        if j.joint_type == "slider":
            return p1

        return p1

    def slider(self, joint_name: str) -> SliderRecord:
        j = self.joint(joint_name)
        if j.joint_type != "slider":
            raise ValueError(f"Expected slider joint, got {joint_name} of type {j.joint_type}")
        if j.slide_direction_world is None or j.slide_value is None:
            raise ValueError(f"Slider joint '{joint_name}' missing direction/value")
        if j.origin_world_two is None:
            raise ValueError(f"Slider joint '{joint_name}' missing rail reference origin on side two")

        return SliderRecord(
            name=joint_name,
            carriage_point_world=j.origin_world_one,
            rail_reference_origin_world=j.origin_world_two,
            slide_direction_world=v_unit(j.slide_direction_world),
            slide_value=j.slide_value,
        )

    def extract(self, source_file: str) -> ReducedParameters:
        ee_points_raw = [self.joint_point(name) for name in self.EE_COUPLING_JOINTS]
        centroid = v_mean(ee_points_raw)
        coupling_points = [
            CouplingPoint(
                joint_name=name,
                point_world=rvec(self.joint_point(name)),
                offset_from_centroid=rvec(v_sub(self.joint_point(name), centroid)),
            )
            for name in self.EE_COUPLING_JOINTS
        ]
        coupling_map = {cp.joint_name: cp for cp in coupling_points}

        sliders = {label: self.slider(name) for label, name in self.SLIDERS.items()}
        x_hat = sliders["A"].slide_direction_world
        for label in ("B", "C"):
            if not close_vec(x_hat, sliders[label].slide_direction_world, self.tol):
                raise ValueError(f"Slider direction mismatch between A and {label}")

        y_seed = v_sub(
            sliders["A"].rail_reference_origin_world,
            sliders["B"].rail_reference_origin_world,
        )
        y_hat = v_unit(y_seed)
        z_hat = v_unit(v_cross(x_hat, y_hat))

        branch_A = self._extract_simple_branch(
            branch_name="A",
            slider=sliders["A"],
            ee_joint_name="Revolute 24",
            chain_joint_names=self.SIMPLE_BRANCH_JOINTS["A"],
            coupling_map=coupling_map,
        )
        branch_B = self._extract_simple_branch(
            branch_name="B",
            slider=sliders["B"],
            ee_joint_name="Revolute 12",
            chain_joint_names=self.SIMPLE_BRANCH_JOINTS["B"],
            coupling_map=coupling_map,
        )
        branch_C = self._extract_closed_branch(
            branch_name="C",
            slider=sliders["C"],
            ee_joint_names=["Revolute 30", "Revolute 32"],
            chain_joint_names=self.CLOSED_BRANCH_JOINTS["C"],
            coupling_map=coupling_map,
        )

        return ReducedParameters(
            source_file=source_file,
            root_component=self.data.get("root_component"),
            units_note=self.data.get("unit_note"),
            world_x=rvec(x_hat),
            world_y=rvec(y_hat),
            world_z=rvec(z_hat),
            ee_centroid_world=rvec(centroid),
            ee_coupling_points=coupling_points,
            sliders=sliders,
            branch_A=branch_A,
            branch_B=branch_B,
            branch_C=branch_C,
        )

    def _extract_simple_branch(
        self,
        branch_name: str,
        slider: SliderRecord,
        ee_joint_name: str,
        chain_joint_names: List[str],
        coupling_map: Dict[str, CouplingPoint],
    ) -> SimpleBranch:
        pts = [self.joint_point(name) for name in chain_joint_names]
        seg_lengths = [round(dist(pts[i], pts[i + 1]), 12) for i in range(len(pts) - 1)]
        cp = coupling_map[ee_joint_name]
        return SimpleBranch(
            name=branch_name,
            slider_joint_name=slider.name,
            rail_origin_world=rvec(slider.rail_reference_origin_world),
            rail_direction_world=rvec(slider.slide_direction_world),
            slider_value=slider.slide_value,
            ee_joint_name=ee_joint_name,
            ee_point_world=cp.point_world,
            ee_offset_from_centroid=cp.offset_from_centroid,
            chain_joint_names=chain_joint_names,
            chain_joint_points_world=[rvec(p) for p in pts],
            rigid_segment_lengths=seg_lengths,
        )

    def _extract_closed_branch(
        self,
        branch_name: str,
        slider: SliderRecord,
        ee_joint_names: List[str],
        chain_joint_names: List[str],
        coupling_map: Dict[str, CouplingPoint],
    ) -> ClosedBranch:
        pts = [self.joint_point(name) for name in chain_joint_names]
        cps = [coupling_map[name] for name in ee_joint_names]
        rigid_lengths = {
            "R29_to_R16": round(dist(self.joint_point("Revolute 29"), self.joint_point("Revolute 16")), 12),
            "R31_to_R19": round(dist(self.joint_point("Revolute 31"), self.joint_point("Revolute 19")), 12),
            "R16_to_R30": round(dist(self.joint_point("Revolute 16"), self.joint_point("Revolute 30")), 12),
            "R19_to_R32": round(dist(self.joint_point("Revolute 19"), self.joint_point("Revolute 32")), 12),
            "R30_to_R32": round(dist(self.joint_point("Revolute 30"), self.joint_point("Revolute 32")), 12),
            "R29_to_R31": round(dist(self.joint_point("Revolute 29"), self.joint_point("Revolute 31")), 12),
            "R16_to_R19": round(dist(self.joint_point("Revolute 16"), self.joint_point("Revolute 19")), 12),
        }
        return ClosedBranch(
            name=branch_name,
            slider_joint_name=slider.name,
            rail_origin_world=rvec(slider.rail_reference_origin_world),
            rail_direction_world=rvec(slider.slide_direction_world),
            slider_value=slider.slide_value,
            ee_joint_names=ee_joint_names,
            ee_points_world=[cp.point_world for cp in cps],
            ee_offsets_from_centroid=[cp.offset_from_centroid for cp in cps],
            chain_joint_names=chain_joint_names,
            chain_joint_points_world=[rvec(p) for p in pts],
            rigid_lengths=rigid_lengths,
        )


# =========================
# Validation
# =========================
def validate(params: ReducedParameters, tol: float = 1e-8) -> ValidationReport:
    messages: List[str] = []
    passed = True

    centroid = v_mean([cp.point_world for cp in params.ee_coupling_points])
    if not close_vec(centroid, params.ee_centroid_world, tol):
        passed = False
        messages.append(f"Centroid mismatch: recomputed {centroid}, stored {params.ee_centroid_world}")
    else:
        messages.append("EE centroid check passed.")

    offsets_ok = True
    for cp in params.ee_coupling_points:
        recomputed = v_sub(cp.point_world, params.ee_centroid_world)
        if not close_vec(recomputed, cp.offset_from_centroid, tol):
            passed = False
            offsets_ok = False
            messages.append(
                f"Offset mismatch for {cp.joint_name}: recomputed {recomputed}, stored {cp.offset_from_centroid}"
            )
    if offsets_ok:
        messages.append("EE attachment offset checks passed.")

    x_hat = v_unit(params.world_x)
    directions_ok = True
    for label, slider in params.sliders.items():
        if not close_vec(v_unit(slider.slide_direction_world), x_hat, tol):
            passed = False
            directions_ok = False
            messages.append(f"Slider {label} does not match extracted world x.")
    if directions_ok:
        messages.append("Slider-direction consistency checks passed.")

    carriage_ok = True
    for label, slider in params.sliders.items():
        predicted = v_add(
            slider.rail_reference_origin_world,
            v_scale(slider.slide_value, slider.slide_direction_world),
        )
        if not close_vec(predicted, slider.carriage_point_world, tol):
            passed = False
            carriage_ok = False
            messages.append(
                f"Slider {label} carriage mismatch: predicted {predicted}, stored {slider.carriage_point_world}"
            )
    if carriage_ok:
        messages.append("Slider carriage-position checks passed.")

    return ValidationReport(passed=passed, messages=messages)


# =========================
# File/CLI helpers
# =========================
def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_reduced_parameter_set(input_json: Path, output_json: Optional[Path] = None) -> Tuple[ReducedParameters, ValidationReport]:
    raw = load_json(input_json)
    extractor = TripteronJSONExtractor(raw)
    params = extractor.extract(str(input_json))
    report = validate(params)
    if output_json is not None:
        save_json(output_json, asdict(params))
    return params, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract reduced tripteron parameters from Fusion JSON export")
    parser.add_argument("input_json", type=Path, help="Path to Fusion-export JSON")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output path for reduced parameter JSON")
    args = parser.parse_args()

    params, report = build_reduced_parameter_set(args.input_json, args.output_json)

    print("=== Reduced parameter extraction ===")
    print(f"Source: {params.source_file}")
    print(f"EE centroid: {params.ee_centroid_world}")
    print(f"World x: {params.world_x}")
    print(f"World y: {params.world_y}")
    print(f"World z: {params.world_z}")
    print()
    print("Coupling points:")
    for cp in params.ee_coupling_points:
        print(f"  {cp.joint_name}: point={cp.point_world}, offset={cp.offset_from_centroid}")
    print()
    print("Validation:")
    print(f"  passed = {report.passed}")
    for msg in report.messages:
        print(f"  - {msg}")
    if args.output_json is not None:
        print(f"\nWrote reduced parameter file to: {args.output_json}")

if __name__ == "__main__":
    input_json = Path(r"fusion_kinematics_rich_export.json")
    output_json = None  # keep this None if you do not want to write a second JSON

    params, report = build_reduced_parameter_set(input_json, output_json)

    print("=== Reduced parameter extraction ===")
    print(f"Source: {params.source_file}")
    print(f"EE centroid: {params.ee_centroid_world}")
    print(f"World x: {params.world_x}")
    print(f"World y: {params.world_y}")
    print(f"World z: {params.world_z}")
    print()

    print("Coupling points:")
    for cp in params.ee_coupling_points:
        print(f"  {cp.joint_name}: point={cp.point_world}, offset={cp.offset_from_centroid}")
    print()

    print("Sliders:")
    for label, slider in params.sliders.items():
        print(
            f"  {label}: {slider.name}, "
            f"carriage={slider.carriage_point_world}, "
            f"rail_origin={slider.rail_reference_origin_world}, "
            f"direction={slider.slide_direction_world}, "
            f"q={slider.slide_value}"
        )
    print()

    print("Validation:")
    print(f"  passed = {report.passed}")
    for msg in report.messages:
        print(f"  - {msg}")

