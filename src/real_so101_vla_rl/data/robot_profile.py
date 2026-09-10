"""SO-101 calibration provenance stored beside each project dataset."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .schema import DEFAULT_FPS, JOINT_NAMES, JOINT_UNITS, SCHEMA_VERSION

MOTOR_NAMES = tuple(name.removesuffix(".pos") for name in JOINT_NAMES)
MOTOR_IDS = dict(zip(MOTOR_NAMES, range(1, 7), strict=True))
STS3215_RESOLUTION = 4096


@dataclass(frozen=True, slots=True)
class JointRange:
    minimum: float
    maximum: float
    unit: str


@dataclass(frozen=True, slots=True)
class RobotProfile:
    schema_version: int
    robot_type: str
    robot_id: str
    use_degrees: bool
    joint_order: tuple[str, ...]
    joint_units: tuple[str, ...]
    nominal_ranges: dict[str, JointRange]
    calibration_file: str
    calibration_sha256: str
    camera_setup_id: str
    control_fps: int
    lerobot_commit: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported robot profile schema_version={self.schema_version}")
        if self.robot_type != "so101_follower":
            raise ValueError("robot_type must be so101_follower")
        if not self.robot_id.strip():
            raise ValueError("robot_id must not be empty")
        if not self.use_degrees:
            raise ValueError("The v1 data schema requires use_degrees=true")
        if self.joint_order != JOINT_NAMES or self.joint_units != JOINT_UNITS:
            raise ValueError("Robot profile joint order or units do not match SO101DataSpec")
        if set(self.nominal_ranges) != set(JOINT_NAMES):
            raise ValueError("Robot profile must contain nominal ranges for all six joints")
        for joint_name, expected_unit in zip(JOINT_NAMES, JOINT_UNITS, strict=True):
            joint_range = self.nominal_ranges[joint_name]
            if joint_range.unit != expected_unit:
                raise ValueError(
                    f"Joint {joint_name!r} must use unit={expected_unit!r}, got {joint_range.unit!r}"
                )
            if joint_range.minimum >= joint_range.maximum:
                raise ValueError(f"Joint {joint_name!r} has an invalid nominal range")
        if len(self.calibration_sha256) != 64:
            raise ValueError("calibration_sha256 must be a SHA-256 hex digest")
        try:
            int(self.calibration_sha256, 16)
        except ValueError as exc:
            raise ValueError("calibration_sha256 must contain hexadecimal characters") from exc
        if Path(self.calibration_file).name != self.calibration_file:
            raise ValueError("calibration_file must be a plain filename")
        if self.control_fps <= 0:
            raise ValueError("control_fps must be positive")
        if not self.camera_setup_id.strip() or not self.lerobot_commit.strip():
            raise ValueError("camera_setup_id and lerobot_commit must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RobotProfile:
        raw = dict(raw)
        raw["joint_order"] = tuple(raw["joint_order"])
        raw["joint_units"] = tuple(raw["joint_units"])
        raw["nominal_ranges"] = {
            name: JointRange(**value) for name, value in raw["nominal_ranges"].items()
        }
        return cls(**raw)


def calibration_sha256(calibration_bytes: bytes) -> str:
    return hashlib.sha256(calibration_bytes).hexdigest()


def _validate_calibration(raw: Any) -> dict[str, dict[str, int]]:
    if not isinstance(raw, dict) or set(raw) != set(MOTOR_NAMES):
        raise ValueError(f"Calibration must contain exactly these motors: {list(MOTOR_NAMES)}")

    validated: dict[str, dict[str, int]] = {}
    required = {"id", "drive_mode", "homing_offset", "range_min", "range_max"}
    for motor_name in MOTOR_NAMES:
        motor = raw[motor_name]
        if not isinstance(motor, dict) or set(motor) != required:
            raise ValueError(f"Invalid calibration fields for motor {motor_name!r}")
        if any(type(motor[key]) is not int for key in required):
            raise TypeError(f"Calibration values for motor {motor_name!r} must be integers")
        if motor["id"] != MOTOR_IDS[motor_name]:
            raise ValueError(
                f"Motor {motor_name!r} must use id={MOTOR_IDS[motor_name]}, got {motor['id']}"
            )
        if motor["drive_mode"] != 0:
            raise ValueError(f"The v1 profile expects drive_mode=0 for motor {motor_name!r}")
        if motor["range_min"] >= motor["range_max"]:
            raise ValueError(f"Invalid calibration range for motor {motor_name!r}")
        validated[motor_name] = dict(motor)
    return validated


def nominal_joint_ranges(calibration: dict[str, dict[str, int]]) -> dict[str, JointRange]:
    """Derive LeRobot units from a validated SO-101 calibration snapshot."""

    calibration = _validate_calibration(calibration)
    result = {}
    max_encoder_value = STS3215_RESOLUTION - 1
    for motor_name, joint_name, unit in zip(MOTOR_NAMES, JOINT_NAMES, JOINT_UNITS, strict=True):
        motor = calibration[motor_name]
        if motor_name == "gripper":
            result[joint_name] = JointRange(0.0, 100.0, unit)
            continue

        midpoint = (motor["range_min"] + motor["range_max"]) / 2
        minimum = (motor["range_min"] - midpoint) * 360 / max_encoder_value
        maximum = (motor["range_max"] - midpoint) * 360 / max_encoder_value
        result[joint_name] = JointRange(minimum, maximum, unit)
    return result


def build_robot_profile(
    calibration_path: str | Path,
    *,
    robot_id: str,
    camera_setup_id: str,
    lerobot_commit: str,
    control_fps: int = DEFAULT_FPS,
    calibration_filename: str = "calibration.json",
) -> tuple[RobotProfile, bytes]:
    """Read an actual calibration file and build a portable dataset profile."""

    calibration_bytes = Path(calibration_path).read_bytes()
    calibration = _validate_calibration(json.loads(calibration_bytes))
    profile = RobotProfile(
        schema_version=SCHEMA_VERSION,
        robot_type="so101_follower",
        robot_id=robot_id,
        use_degrees=True,
        joint_order=JOINT_NAMES,
        joint_units=JOINT_UNITS,
        nominal_ranges=nominal_joint_ranges(calibration),
        calibration_file=calibration_filename,
        calibration_sha256=calibration_sha256(calibration_bytes),
        camera_setup_id=camera_setup_id,
        control_fps=control_fps,
        lerobot_commit=lerobot_commit,
    )
    return profile, calibration_bytes


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(file_descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_robot_profile(
    project_meta_dir: str | Path,
    profile: RobotProfile,
    calibration_bytes: bytes,
) -> None:
    """Write the profile and exact calibration snapshot atomically."""

    if calibration_sha256(calibration_bytes) != profile.calibration_sha256:
        raise ValueError("Calibration bytes do not match robot profile calibration_sha256")
    project_meta_dir = Path(project_meta_dir)
    _atomic_write(project_meta_dir / profile.calibration_file, calibration_bytes)
    profile_bytes = json.dumps(
        profile.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    _atomic_write(project_meta_dir / "robot_profile.json", profile_bytes)


def load_robot_profile(path: str | Path) -> RobotProfile:
    return RobotProfile.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
