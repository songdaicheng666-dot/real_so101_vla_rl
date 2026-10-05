"""Calibrate SO-101 mechanical hard stops from local shell contacts.

This is an independent calibration model.  It starts from the standalone
SO-101 MJCF, temporarily sets all joint refs to zero, disables its authored
joint/control ranges and all unrelated collision masks, then adds explicit
local contact pairs for the photographed white-shell stops and the valid
servo-housing stops found by exact mesh scans.  Wrist roll remains unlimited
when a complete revolution finds no shell or servo stop.

The small contact patches were placed on the exact visual STL triangles at
the photographed shell-to-shell contact locations.  The program first scans
for contact with no joint limits enabled.  It then uses the detected contact
angles as MuJoCo joint limits so a local patch cannot slip past its mate under
a large position command.  Servo horns, shafts, cables and persistent mounting
overlaps are excluded.  The moving joint and contacted servo may have different
numbers: joint 3's positive stop is its white bracket contacting servo 2's
housing.  No real-arm readings or follower calibration file are used here.
"""

from __future__ import annotations

import argparse
import math
import queue
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mujoco
import mujoco.viewer

if __package__:
    from .demo_mujoco_so101_unlimited import (
        ADJUSTMENT_KEYS,
        JOINT_NAMES,
        PAUSE_KEY,
        RESET_KEY,
        ROBOT_PATH,
        SERVO_SELECTION_KEYS,
        STATUS_KEY,
        adjust_control,
        reset_robot,
        restore_viewer_shortcut_side_effect,
        validate_robot,
    )
else:
    from demo_mujoco_so101_unlimited import (
        ADJUSTMENT_KEYS,
        JOINT_NAMES,
        PAUSE_KEY,
        RESET_KEY,
        ROBOT_PATH,
        SERVO_SELECTION_KEYS,
        STATUS_KEY,
        adjust_control,
        reset_robot,
        restore_viewer_shortcut_side_effect,
        validate_robot,
    )


@dataclass(frozen=True)
class HardstopPatch:
    """One local contact derived from exact source-mesh intersections."""

    joint: str
    side: str
    parent_body: str
    parent_pos: tuple[float, float, float]
    parent_quat: tuple[float, float, float, float]
    child_body: str
    child_pos: tuple[float, float, float]
    child_quat: tuple[float, float, float, float]
    mesh_contact_deg: float
    source: str = "shell"
    counterpart: str = "white shell"
    servo_number: int | None = None

    @property
    def name(self) -> str:
        source_prefix = "" if self.source == "shell" else f"{self.source}_"
        return f"hardstop_{source_prefix}{self.joint}_{self.side}"

    @property
    def parent_geom(self) -> str:
        return f"{self.name}_parent"

    @property
    def child_geom(self) -> str:
        return f"{self.name}_child"


@dataclass(frozen=True)
class ServoCandidateStatus:
    """Exact-mesh result for one servo housing and joint direction."""

    joint: str
    side: str
    status: str
    detail: str


@dataclass(frozen=True)
class ScannedEndpoint:
    """One MuJoCo local-patch contact measured without joint limits."""

    patch: HardstopPatch
    angle: float


# Each 12 x 12 mm face is centred 1.5 mm inside its source STL surface.  The
# two 1.5 mm half-thicknesses therefore meet at the original triangle contact.
CONTACT_PATCH_SIZE = (0.006, 0.006, 0.0015)
SCAN_STEP_DEG = 0.25
SCAN_TOLERANCE_DEG = 1e-6
SEARCH_BOUND_DEG = 180.0
STATUS_PERIOD_S = 0.5

# The positions and orientations below are body-local.  They came from exact
# triangle/triangle contact queries on the visual STL meshes at the shell
# contacts shown in the reference photos.  The former elbow/max shell pair is
# omitted: the real stop there is the joint-3 bracket hitting servo 2.
# mesh_contact_deg is retained only as an independent regression check;
# scan_hardstop_ranges does not return it.
SHELL_HARDSTOP_PATCHES = (
    HardstopPatch(
        "gripper",
        "min",
        "gripper",
        (-0.009400134790, -0.000100002159, -0.100566041837),
        (0.707100867901, -0.000000469140, 0.707112694422, 0.000000469148),
        "moving_jaw_so101_v1",
        (-0.010809305110, -0.080903180937, 0.018899998178),
        (0.703605758943, 0.070282379201, -0.703604806481, 0.070282284061),
        -11.4095351421,
    ),
    HardstopPatch(
        "gripper",
        "max",
        "gripper",
        (0.026115098370, -0.018307407100, -0.009421890016),
        (0.181812458586, 0.683333176352, 0.706177406137, 0.036241841304),
        "moving_jaw_so101_v1",
        (0.007389153142, -0.010825589269, 0.037699282363),
        (0.632258734728, 0.032752497021, 0.773028194362, -0.040044687787),
        114.2574319065,
    ),
    HardstopPatch(
        "wrist_flex",
        "min",
        "lower_arm",
        (-0.117238876077, 0.021560303457, 0.037800000199),
        (0.702498893575, -0.080594512621, -0.702498653406, -0.080594485068),
        "wrist",
        (-0.010497257748, -0.019934865731, 0.037800001198),
        (0.707106663449, -0.000002808714, -0.707106898913, -0.000002808714),
        -103.0888879526,
    ),
    HardstopPatch(
        "wrist_flex",
        "max",
        "lower_arm",
        (-0.117210381341, -0.013898175634, 0.039689164015),
        (0.703505716585, 0.071272209359, -0.703505686904, 0.071272206352),
        "wrist",
        (0.010499961466, -0.022257896825, 0.039689164139),
        (0.707106766568, -0.000002802729, 0.707106795794, 0.000002802729),
        101.5693486602,
    ),
    HardstopPatch(
        "elbow_flex",
        "min",
        "upper_arm",
        (-0.128300778246, -0.010983325318, 0.043152592223),
        (-0.446822787459, -0.548041418697, -0.446822693649, 0.548041495181),
        "lower_arm",
        (-0.018835484913, 0.010499967058, 0.043152592726),
        (-0.499997992359, 0.500002007633, 0.499998076192, 0.500001923801),
        -101.6182572846,
    ),
    HardstopPatch(
        "shoulder_lift",
        "min",
        "shoulder",
        (-0.016810837516, -0.021249707126, -0.036530869595),
        (0.140143756431, 0.693079885391, 0.705663590527, 0.045154147147),
        "upper_arm",
        (-0.017752168597, 0.010505185128, -0.003363514222),
        (0.531641895474, -0.466215502721, -0.466235910222, -0.531623998724),
        -105.2281935759,
    ),
    HardstopPatch(
        "shoulder_lift",
        "max",
        "shoulder",
        (-0.049386392945, 0.021666097792, -0.036527924020),
        (-0.086885444602, 0.701748473113, 0.704921247387, -0.055552092516),
        "upper_arm",
        (-0.022147122082, -0.010505377296, 0.040074770743),
        (0.488969334749, 0.510792511373, 0.510786538643, -0.488975573971),
        101.5758477902,
    ),
    HardstopPatch(
        "shoulder_pan",
        "min",
        "base",
        (0.018211533710, 0.022365973333, 0.042382407346),
        (-0.422788360966, 0.566789204053, 0.422788345001, 0.566789215963),
        "shoulder",
        (-0.031223898263, 0.007280805548, 0.020017592573),
        (-0.250524610440, 0.661239305822, 0.250524583497, 0.661239316030),
        -114.9416695134,
    ),
    HardstopPatch(
        "shoulder_pan",
        "max",
        "base",
        (0.026182127400, -0.017606096104, 0.030499999526),
        (0.531341251736, 0.466558114498, 0.531341244522, -0.466558122714),
        "shoulder",
        (-0.023937745950, 0.000998349263, 0.031900000428),
        (0.663166254123, 0.245378318913, 0.663166248343, -0.245378334535),
        123.1811807550,
    ),
)

# Valid first contacts from exact scans of the no-horn STS3215 housing against
# the relevant moving white parts.  Most pairs use the housing belonging to
# the moving joint; elbow/max is the cross-link exception where joint 3 moves
# a white bracket into upstream servo 2.
SERVO_HARDSTOP_PATCHES = (
    HardstopPatch(
        "shoulder_pan",
        "min",
        "base",
        (0.025975172595, 0.021199974861, 0.050702746633),
        (-0.5, 0.5, 0.5, 0.5),
        "shoulder",
        (-0.027227173277, -0.003114915009, 0.011697253367),
        (0.675276945970, -0.209764263501, 0.675276945970, 0.209764263501),
        -124.5130859531,
        source="servo",
        counterpart="rotation_pitch_so101_v1",
        servo_number=1,
    ),
    HardstopPatch(
        "shoulder_pan",
        "max",
        "base",
        (0.011692895008, -0.021167375158, 0.045559475133),
        (-0.497448601252, -0.502538445408, -0.496281965445, 0.503690590317),
        "shoulder",
        (-0.025838298808, -0.025596475706, 0.016833568937),
        (0.707698686322, 0.017983023850, 0.706057567902, -0.017941322102),
        93.6272342676,
        source="servo",
        counterpart="rotation_pitch_so101_v1",
        servo_number=1,
    ),
    HardstopPatch(
        "shoulder_lift",
        "min",
        "shoulder",
        (-0.009137141512, -0.004521108533, -0.028670704229),
        (0.801303919811, 0.001901148730, 0.598252788581, -0.001419395938),
        "upper_arm",
        (-0.033911471611, 0.011343395369, 0.013742456134),
        (0.704585068675, -0.077397633611, 0.701168346978, 0.077022311757),
        -119.0475194759,
        source="servo",
        counterpart="upper_arm_so101_v1",
        servo_number=2,
    ),
    HardstopPatch(
        "shoulder_lift",
        "max",
        "shoulder",
        (-0.051359663363, -0.009579379866, -0.027039026577),
        (0.800685115698, -0.001244705886, -0.599083407279, -0.000931305739),
        "upper_arm",
        (-0.031077917159, -0.019668792769, 0.008707747418),
        (0.705985895939, 0.005402349211, 0.708184553569, -0.005419173791),
        107.2681023027,
        source="servo",
        counterpart="upper_arm_so101_v1",
        servo_number=2,
    ),
    HardstopPatch(
        "elbow_flex",
        "max",
        "shoulder",
        (-0.051621863546, -0.009716303608, -0.028373037134),
        (0.766085507550, 0.002616239797, -0.642729595178, 0.002194970051),
        "lower_arm",
        (-0.137308524249, 0.021430885078, 0.008541006178),
        (-0.499790635837, -0.500209276533, -0.496362412020, 0.503611314343),
        100.4473491554,
        source="servo",
        counterpart="motor_holder_so101_wrist_v1",
        servo_number=2,
    ),
)

HARDSTOP_PATCHES = SHELL_HARDSTOP_PATCHES + SERVO_HARDSTOP_PATCHES

# Every servo is represented in the exact-mesh derivation.  These entries make
# no-contact and mounting-overlap outcomes explicit instead of manufacturing a
# false runtime stop.  The two mounting pairs remain intersecting after the
# requested 0.5 mm triangle-neighbour exclusion and are therefore excluded as
# whole pairs.
SERVO_CANDIDATE_STATUSES = (
    ServoCandidateStatus("shoulder_pan", "min", "contact", "rotation_pitch_so101_v1"),
    ServoCandidateStatus("shoulder_pan", "max", "contact", "rotation_pitch_so101_v1"),
    ServoCandidateStatus("shoulder_lift", "min", "contact", "upper_arm_so101_v1"),
    ServoCandidateStatus("shoulder_lift", "max", "contact", "upper_arm_so101_v1"),
    ServoCandidateStatus("elbow_flex", "min", "none", "no contact within 180 deg"),
    ServoCandidateStatus(
        "elbow_flex",
        "max",
        "contact",
        "joint 3 bracket contacts servo 2 housing",
    ),
    ServoCandidateStatus(
        "wrist_flex",
        "min",
        "mounting_overlap",
        "servo housing is embedded in wrist_roll_pitch_so101_v2",
    ),
    ServoCandidateStatus(
        "wrist_flex",
        "max",
        "mounting_overlap",
        "servo housing is embedded in wrist_roll_pitch_so101_v2",
    ),
    ServoCandidateStatus("wrist_roll", "min", "none", "no contact in one revolution"),
    ServoCandidateStatus("wrist_roll", "max", "none", "no contact in one revolution"),
    ServoCandidateStatus(
        "gripper",
        "min",
        "mounting_overlap",
        "moving jaw surrounds the servo output-side housing",
    ),
    ServoCandidateStatus(
        "gripper",
        "max",
        "mounting_overlap",
        "moving jaw surrounds the servo output-side housing",
    ),
)

HARDSTOP_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "gripper",
)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect SO-101 shell and servo hard stops without a real arm.",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="print shell/servo candidates and final endpoints without Viewer",
    )
    parser.add_argument(
        "--headless-check",
        action="store_true",
        help="scan, lock the ranges and run a short physics stability check",
    )
    parser.add_argument(
        "--joint-step-deg",
        type=_positive_float,
        default=2.0,
        help="keyboard target step for arm joints in degrees (default: 2)",
    )
    parser.add_argument(
        "--gripper-step-deg",
        type=_positive_float,
        default=1.0,
        help="keyboard target step for the gripper in degrees (default: 1)",
    )
    return parser.parse_args(argv)


def _require_body(spec: mujoco.MjSpec, name: str):
    body = spec.body(name)
    if body is None:
        raise RuntimeError(f"required MuJoCo body is missing: {name}")
    return body


def _add_contact_patch(
    spec: mujoco.MjSpec,
    *,
    body_name: str,
    geom_name: str,
    pos: tuple[float, float, float],
    quat: tuple[float, float, float, float],
    rgba: tuple[float, float, float, float],
) -> None:
    _require_body(spec, body_name).add_geom(
        name=geom_name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=pos,
        quat=quat,
        size=CONTACT_PATCH_SIZE,
        mass=0,
        contype=0,
        conaffinity=0,
        condim=3,
        group=4,
        rgba=rgba,
    )


def build_contact_model() -> mujoco.MjModel:
    """Build the independent model with only the explicit stop pairs."""

    if not ROBOT_PATH.is_file():
        raise FileNotFoundError(f"SO-101 robot model not found: {ROBOT_PATH}")
    if not hasattr(mujoco, "MjSpec"):
        raise RuntimeError("this calibration demo requires MuJoCo with MjSpec support")

    spec = mujoco.MjSpec.from_file(str(ROBOT_PATH))
    # The formal model's refs contain the earlier real-arm midpoint readings.
    # Remove them only in this independent model so the scanned values describe
    # pure CAD rotation from the mesh's authored rest pose.  Later endpoint
    # mapping can solve the real-to-sim offset without assuming a midpoint.
    for joint_name in JOINT_NAMES:
        joint = spec.joint(joint_name)
        if joint is None:
            raise RuntimeError(f"required MuJoCo joint is missing: {joint_name}")
        joint.ref = 0.0
    for patch in HARDSTOP_PATCHES:
        parent_rgba = (
            (1.0, 0.45, 0.05, 0.8)
            if patch.source == "servo"
            else (1.0, 0.1, 0.1, 0.7)
        )
        _add_contact_patch(
            spec,
            body_name=patch.parent_body,
            geom_name=patch.parent_geom,
            pos=patch.parent_pos,
            quat=patch.parent_quat,
            rgba=parent_rgba,
        )
        _add_contact_patch(
            spec,
            body_name=patch.child_body,
            geom_name=patch.child_geom,
            pos=patch.child_pos,
            quat=patch.child_quat,
            rgba=(0.1, 0.4, 1.0, 0.7),
        )
        spec.add_pair(
            name=patch.name,
            geomname1=patch.parent_geom,
            geomname2=patch.child_geom,
            condim=3,
            friction=(1.0, 0.005, 0.0001, 0.0001, 0.0001),
        )

    model = spec.compile()
    validate_robot(model)

    # Start the scan without any authored angular limits.  The explicit pairs
    # still collide even though all geom masks are zero.  This also prevents
    # the standalone model's coarse task collision boxes from setting an
    # endpoint before the photographed white-shell contacts do.
    model.jnt_limited[:] = False
    model.actuator_ctrllimited[:] = False
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    return model


def _object_id(model: mujoco.MjModel, object_type: int, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"required MuJoCo object is missing: {name}")
    return object_id


def _patch_geom_ids(
    model: mujoco.MjModel,
    patch: HardstopPatch,
) -> frozenset[int]:
    return frozenset(
        (
            _object_id(model, mujoco.mjtObj.mjOBJ_GEOM, patch.parent_geom),
            _object_id(model, mujoco.mjtObj.mjOBJ_GEOM, patch.child_geom),
        )
    )


def _patch_is_contacting(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    patch: HardstopPatch,
) -> bool:
    expected = _patch_geom_ids(model, patch)
    return any(
        frozenset((int(contact.geom1), int(contact.geom2))) == expected
        for contact in data.contact[: data.ncon]
    )


def _set_scan_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    joint_id: int,
    qpos: float,
) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:] = model.qpos0
    data.qpos[model.jnt_qposadr[joint_id]] = qpos
    mujoco.mj_forward(model, data)


def _scan_one_endpoint(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    patch: HardstopPatch,
    *,
    step_deg: float = SCAN_STEP_DEG,
    tolerance_deg: float = SCAN_TOLERANCE_DEG,
) -> float:
    joint_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, patch.joint)
    qpos_address = int(model.jnt_qposadr[joint_id])
    safe = float(model.qpos0[qpos_address])
    direction = -1.0 if patch.side == "min" else 1.0
    step = math.radians(step_deg) * direction
    search_bound = math.radians(SEARCH_BOUND_DEG) * direction
    contact = None

    candidate = safe
    while (candidate - search_bound) * direction < 0:
        candidate += step
        if (candidate - search_bound) * direction > 0:
            candidate = search_bound
        _set_scan_pose(model, data, joint_id=joint_id, qpos=candidate)
        if _patch_is_contacting(model, data, patch):
            contact = candidate
            break

    if contact is None:
        raise RuntimeError(
            f"no {patch.side} shell contact found for {patch.joint} "
            f"before {SEARCH_BOUND_DEG:g} degrees"
        )

    tolerance = math.radians(tolerance_deg)
    while abs(contact - safe) > tolerance:
        candidate = 0.5 * (safe + contact)
        _set_scan_pose(model, data, joint_id=joint_id, qpos=candidate)
        if _patch_is_contacting(model, data, patch):
            contact = candidate
        else:
            safe = candidate
    return contact


def scan_hardstop_candidates(
    model: mujoco.MjModel,
) -> tuple[ScannedEndpoint, ...]:
    """Measure every valid shell and servo contact with limits disabled."""

    if any(bool(value) for value in model.jnt_limited):
        raise ValueError("hard-stop scan requires all joint limits to be disabled")
    data = mujoco.MjData(model)
    endpoints = []
    for patch in HARDSTOP_PATCHES:
        endpoint = _scan_one_endpoint(model, data, patch)
        measured_deg = math.degrees(endpoint)
        if not math.isclose(
            measured_deg,
            patch.mesh_contact_deg,
            rel_tol=0.0,
            abs_tol=0.01,
        ):
            raise RuntimeError(
                f"{patch.name} moved away from its source mesh contact: "
                f"scan={measured_deg:+.6f} deg, "
                f"mesh={patch.mesh_contact_deg:+.6f} deg"
            )
        endpoints.append(ScannedEndpoint(patch=patch, angle=endpoint))
    return tuple(endpoints)


def select_hardstop_ranges(
    candidates: Sequence[ScannedEndpoint],
) -> tuple[
    dict[str, tuple[float, float]],
    dict[tuple[str, str], ScannedEndpoint],
]:
    """Select the first physical contact reached from zero in each direction."""

    grouped: dict[tuple[str, str], list[ScannedEndpoint]] = {}
    for candidate in candidates:
        grouped.setdefault((candidate.patch.joint, candidate.patch.side), []).append(
            candidate
        )

    winners: dict[tuple[str, str], ScannedEndpoint] = {}
    for joint in HARDSTOP_JOINTS:
        minimums = grouped.get((joint, "min"), [])
        maximums = grouped.get((joint, "max"), [])
        if not minimums or not maximums:
            raise RuntimeError(f"missing a two-sided hard stop for {joint}")
        # Negative direction approaches zero by increasing; positive direction
        # approaches zero by decreasing.
        winners[(joint, "min")] = max(minimums, key=lambda item: item.angle)
        winners[(joint, "max")] = min(maximums, key=lambda item: item.angle)

    ranges = {
        joint: (
            winners[(joint, "min")].angle,
            winners[(joint, "max")].angle,
        )
        for joint in HARDSTOP_JOINTS
    }
    return ranges, winners


def scan_hardstop_ranges(
    model: mujoco.MjModel,
) -> dict[str, tuple[float, float]]:
    """Return the first-contact two-sided ranges in radians."""

    ranges, _ = select_hardstop_ranges(scan_hardstop_candidates(model))
    return ranges


def apply_hardstop_ranges(
    model: mujoco.MjModel,
    ranges: Mapping[str, tuple[float, float]],
) -> None:
    """Lock joints at the contact-derived endpoints after the free scan."""

    if set(ranges) != set(HARDSTOP_JOINTS):
        raise ValueError("hard-stop ranges must contain exactly the five stopped joints")
    model.jnt_limited[:] = False
    model.actuator_ctrllimited[:] = False
    for joint, (lower, upper) in ranges.items():
        if not lower < upper:
            raise ValueError(f"invalid hard-stop range for {joint}")
        joint_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        model.jnt_range[joint_id] = (lower, upper)
        model.jnt_limited[joint_id] = True


def _candidate_by_source(
    candidates: Sequence[ScannedEndpoint],
) -> dict[tuple[str, str, str], ScannedEndpoint]:
    return {
        (candidate.patch.joint, candidate.patch.side, candidate.patch.source): candidate
        for candidate in candidates
    }


def print_ranges(
    ranges: Mapping[str, tuple[float, float]],
    candidates: Sequence[ScannedEndpoint],
    winners: Mapping[tuple[str, str], ScannedEndpoint],
) -> None:
    print("[机械接触] ref=0、无真机读数、无预设关节限位的扫描结果")
    candidate_lookup = _candidate_by_source(candidates)
    servo_status = {
        (status.joint, status.side): status for status in SERVO_CANDIDATE_STATUSES
    }
    for joint in JOINT_NAMES:
        print(f"  {joint}")
        for side in ("min", "max"):
            shell = candidate_lookup.get((joint, side, "shell"))
            servo = candidate_lookup.get((joint, side, "servo"))
            status = servo_status[(joint, side)]
            shell_text = (
                f"{math.degrees(shell.angle):+.4f} deg" if shell else "none"
            )
            if servo:
                servo_text = (
                    f"{math.degrees(servo.angle):+.4f} deg"
                    f"(servo#{servo.patch.servo_number} housing)"
                )
            elif status.status == "mounting_overlap":
                servo_text = "excluded(mounting overlap)"
            else:
                servo_text = "none"
            winner = winners.get((joint, side))
            if winner is None:
                final_text = "unlimited"
            else:
                final_text = (
                    f"{math.degrees(winner.angle):+.4f} deg/"
                    f"{winner.patch.source}"
                )
            print(
                f"    {side:3s} shell={shell_text:>13s} "
                f"servo={servo_text:>27s} final={final_text}"
            )
    print("[最终范围]")
    for joint in HARDSTOP_JOINTS:
        lower, upper = ranges[joint]
        print(
            f"  {joint:14s} "
            f"min={math.degrees(lower):+10.4f} deg  "
            f"max={math.degrees(upper):+10.4f} deg  "
            f"span={math.degrees(upper - lower):9.4f} deg"
        )
    print("  wrist_roll     no contact in one revolution; unlimited")


def _active_hardstops(model: mujoco.MjModel, data: mujoco.MjData) -> str:
    active = [
        (
            f"{patch.joint}/{patch.side}/servo#{patch.servo_number}"
            if patch.source == "servo"
            else f"{patch.joint}/{patch.side}/shell"
        )
        for patch in HARDSTOP_PATCHES
        if _patch_is_contacting(model, data, patch)
    ]
    return ",".join(active) if active else "none"


def _status_line(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ranges: Mapping[str, tuple[float, float]],
    winners: Mapping[tuple[str, str], ScannedEndpoint],
    servo_id: int,
) -> str:
    actuator_id = servo_id - 1
    joint_name = JOINT_NAMES[actuator_id]
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    actual = float(data.qpos[model.jnt_qposadr[joint_id]])
    target = float(data.ctrl[actuator_id])
    force = float(data.actuator_force[actuator_id])
    if joint_name in ranges:
        lower, upper = ranges[joint_name]
        lower_source = winners[(joint_name, "min")].patch.source
        upper_source = winners[(joint_name, "max")].patch.source
        range_text = (
            f"[{math.degrees(lower):+.2f}/{lower_source},"
            f"{math.degrees(upper):+.2f}/{upper_source}] deg"
        )
    else:
        range_text = "unlimited"
    return (
        f"[状态] {servo_id} {joint_name} "
        f"目标={math.degrees(target):+.2f} deg "
        f"实际={math.degrees(actual):+.2f} deg "
        f"接触范围={range_text} "
        f"驱动力={force:+.2f} "
        f"硬限位接触={_active_hardstops(model, data)}"
    )


def run_headless_check(
    model: mujoco.MjModel,
    ranges: Mapping[str, tuple[float, float]],
) -> None:
    for joint in HARDSTOP_JOINTS:
        joint_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        actuator_id = _object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint)
        qpos_address = int(model.jnt_qposadr[joint_id])
        for side, endpoint in zip(("min", "max"), ranges[joint], strict=True):
            data = mujoco.MjData(model)
            reset_robot(model, data)
            data.ctrl[actuator_id] = endpoint + (
                math.radians(-20.0 if side == "min" else 20.0)
            )
            for _ in range(1000):
                mujoco.mj_step(model, data)
            actual = float(data.qpos[qpos_address])
            if side == "min" and actual < endpoint - math.radians(2.0):
                raise RuntimeError(f"{joint} passed through its minimum hard stop")
            if side == "max" and actual > endpoint + math.radians(2.0):
                raise RuntimeError(f"{joint} passed through its maximum hard stop")
            if not all(math.isfinite(float(value)) for value in data.qpos):
                raise RuntimeError("headless check produced non-finite qpos")
    print("[检查] 十个接触端点均阻止了对应关节继续转动。")


def print_controls(joint_step_deg: float, gripper_step_deg: float) -> None:
    print(
        """
SO-101 白色外壳与舵机外壳机械硬限位标定

  1  shoulder_pan              4  wrist_flex
  2  shoulder_lift             5  wrist_roll（无限位）
  3  elbow_flex                6  gripper

  A / D      减小 / 增大所选关节目标
  P          立即打印状态
  Space      暂停 / 继续
  Backspace  复位
  关闭 Viewer 窗口退出

红、蓝小片表示白色件止挡；橙、蓝小片表示舵机外壳止挡。
关节范围由启动时的无约束接触扫描得到；真机数据未参与。
""".strip()
    )
    print(
        f"键盘步长：机械臂={joint_step_deg:g} deg，"
        f"夹爪={gripper_step_deg:g} deg"
    )


def run_interactive(
    model: mujoco.MjModel,
    ranges: Mapping[str, tuple[float, float]],
    winners: Mapping[tuple[str, str], ScannedEndpoint],
    *,
    joint_step_deg: float,
    gripper_step_deg: float,
) -> None:
    data = mujoco.MjData(model)
    reset_robot(model, data)
    key_events: queue.SimpleQueue[int] = queue.SimpleQueue()
    selected_servo_id = 1
    paused = False
    last_status = -math.inf
    print_controls(joint_step_deg, gripper_step_deg)
    print(f"[基础模型] {ROBOT_PATH}")

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_events.put,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = model.stat.center
            viewer.cam.distance = 0.85
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
        geomgroup_before = tuple(int(value) for value in viewer.opt.geomgroup)
        flags_before = tuple(int(value) for value in viewer.opt.flags)
        while viewer.is_running():
            loop_start = time.monotonic()
            print_status_now = False
            with viewer.lock():
                while True:
                    try:
                        keycode = key_events.get_nowait()
                    except queue.Empty:
                        break
                    restore_viewer_shortcut_side_effect(
                        viewer.opt,
                        keycode,
                        geomgroup_before=geomgroup_before,
                        flags_before=flags_before,
                    )
                    if keycode in SERVO_SELECTION_KEYS:
                        selected_servo_id = SERVO_SELECTION_KEYS[keycode] + 1
                        print_status_now = True
                    elif keycode in ADJUSTMENT_KEYS:
                        target = adjust_control(
                            model,
                            data,
                            selected_servo_id,
                            ADJUSTMENT_KEYS[keycode],
                            joint_step_rad=math.radians(joint_step_deg),
                            gripper_step_rad=math.radians(gripper_step_deg),
                        )
                        print(
                            f"[控制] {JOINT_NAMES[selected_servo_id - 1]} "
                            f"目标={math.degrees(target):+.2f} deg"
                        )
                        print_status_now = True
                    elif keycode == PAUSE_KEY:
                        paused = not paused
                        print("[仿真] 已暂停" if paused else "[仿真] 已继续")
                    elif keycode == RESET_KEY:
                        reset_robot(model, data)
                        print("[复位] 已恢复参考姿态")
                        print_status_now = True
                    elif keycode == STATUS_KEY:
                        print_status_now = True
                geomgroup_before = tuple(int(value) for value in viewer.opt.geomgroup)
                flags_before = tuple(int(value) for value in viewer.opt.flags)
                if not paused:
                    mujoco.mj_step(model, data)
                if print_status_now or loop_start - last_status >= STATUS_PERIOD_S:
                    status = _status_line(
                        model,
                        data,
                        ranges,
                        winners,
                        selected_servo_id,
                    )
                    last_status = loop_start
                else:
                    status = None
            if status is not None:
                print(status)
            viewer.sync()
            target_period = 0.01 if paused else float(model.opt.timestep)
            remaining = target_period - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)


def main() -> None:
    args = parse_args()
    model = build_contact_model()
    candidates = scan_hardstop_candidates(model)
    ranges, winners = select_hardstop_ranges(candidates)
    print_ranges(ranges, candidates, winners)
    if args.scan_only:
        return
    apply_hardstop_ranges(model, ranges)
    if args.headless_check:
        run_headless_check(model, ranges)
        return
    run_interactive(
        model,
        ranges,
        winners,
        joint_step_deg=args.joint_step_deg,
        gripper_step_deg=args.gripper_step_deg,
    )


if __name__ == "__main__":
    main()
