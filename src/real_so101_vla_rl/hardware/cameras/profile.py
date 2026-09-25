"""Strict, portable camera-rig profiles for real SO-101 recording."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from real_so101_vla_rl.data.schema import DEFAULT_FPS, IMAGE_HEIGHT, IMAGE_WIDTH

CAMERA_PROFILE_SCHEMA_VERSION = 2
_HEX_ID = re.compile(r"^[0-9a-f]{4}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _strict_mapping(
    value: Any,
    *,
    name: str,
    required: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    keys = set(value)
    if keys != required:
        missing = sorted(required - keys)
        extra = sorted(keys - required)
        raise ValueError(f"{name} fields differ: missing={missing}, extra={extra}")
    return dict(value)


def _non_empty(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class USBDeviceProfile:
    vendor_id: str
    product_id: str
    serial: str
    minimum_speed_mbps: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("vendor_id", self.vendor_id),
            ("product_id", self.product_id),
        ):
            if not _HEX_ID.fullmatch(value):
                raise ValueError(f"USB {field_name} must be four lowercase hex digits")
        _non_empty(self.serial, name="USB serial")
        if self.minimum_speed_mbps <= 0:
            raise ValueError("USB minimum_speed_mbps must be positive")


@dataclass(frozen=True, slots=True)
class VideoStreamProfile:
    width: int
    height: int
    fps: int
    format: str

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("stream width, height, and fps must be positive")
        _non_empty(self.format, name="stream format")


@dataclass(frozen=True, slots=True)
class CalibrationReference:
    file: str
    sha256: str

    def __post_init__(self) -> None:
        if Path(self.file).is_absolute() or ".." in Path(self.file).parts:
            raise ValueError("calibration file must be a profile-relative path")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("calibration sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class OrbbecCameraProfile:
    backend: str
    model: str
    usb: USBDeviceProfile
    firmware: str
    color: VideoStreamProfile
    output_color_order: str
    depth: VideoStreamProfile
    depth_unit: str
    aligned_output_width: int
    aligned_output_height: int
    alignment: str
    frame_sync: bool
    factory_calibration: CalibrationReference

    def __post_init__(self) -> None:
        if self.backend != "orbbec_sdk":
            raise ValueError("overview backend must be orbbec_sdk")
        _non_empty(self.model, name="overview model")
        _non_empty(self.firmware, name="overview firmware")
        expected_color = VideoStreamProfile(
            IMAGE_WIDTH, IMAGE_HEIGHT, DEFAULT_FPS, "MJPG"
        )
        expected_depth = VideoStreamProfile(640, 400, DEFAULT_FPS, "Y16")
        if self.color != expected_color:
            raise ValueError(f"overview color stream must be {expected_color}")
        if self.depth != expected_depth:
            raise ValueError(f"overview depth stream must be {expected_depth}")
        if self.output_color_order != "RGB":
            raise ValueError("overview output_color_order must be RGB")
        if self.depth_unit != "mm":
            raise ValueError("overview depth unit must be mm")
        if (self.aligned_output_width, self.aligned_output_height) != (
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
        ):
            raise ValueError("aligned depth output must match overview RGB dimensions")
        if self.alignment != "depth_to_color":
            raise ValueError("overview alignment must be depth_to_color")
        if self.frame_sync is not True:
            raise ValueError("overview frame_sync must be true")


@dataclass(frozen=True, slots=True)
class WristCameraProfile:
    backend: str
    model: str
    usb: USBDeviceProfile
    device_by_id: str
    color: VideoStreamProfile
    driver_color_order: str
    output_color_order: str
    exposure_mode: str
    exposure_value: int
    gain: int
    intrinsic_calibration_status: str
    intrinsic_calibration_file: str | None

    def __post_init__(self) -> None:
        if self.backend != "opencv_v4l2":
            raise ValueError("wrist backend must be opencv_v4l2")
        _non_empty(self.model, name="wrist model")
        path = Path(self.device_by_id)
        if path.parent != Path("/dev/v4l/by-id"):
            raise ValueError("wrist device must use an absolute /dev/v4l/by-id path")
        if not path.name.endswith("-video-index0"):
            raise ValueError("wrist device must select the capture node ending in video-index0")
        expected = VideoStreamProfile(IMAGE_WIDTH, IMAGE_HEIGHT, DEFAULT_FPS, "MJPG")
        if self.color != expected:
            raise ValueError(f"wrist color stream must be {expected}")
        if self.driver_color_order != "BGR" or self.output_color_order != "RGB":
            raise ValueError("wrist color conversion must be BGR to RGB")
        if self.exposure_mode != "manual":
            raise ValueError("wrist exposure mode must be manual")
        if type(self.exposure_value) is not int or self.exposure_value <= 0:
            raise ValueError("wrist exposure value must be a positive integer")
        if self.gain != 0:
            raise ValueError("wrist gain must be frozen at 0")
        if self.intrinsic_calibration_status != "uncalibrated":
            raise ValueError("wrist intrinsic calibration must remain uncalibrated in this profile")
        if self.intrinsic_calibration_file is not None:
            raise ValueError("uncalibrated wrist profile cannot reference a calibration file")


@dataclass(frozen=True, slots=True)
class InstallationCalibration:
    status: str
    overview_to_world: Any | None
    wrist_to_gripper: Any | None

    def __post_init__(self) -> None:
        if self.status != "uncalibrated":
            raise ValueError("installation calibration must remain uncalibrated in this profile")
        if self.overview_to_world is not None or self.wrist_to_gripper is not None:
            raise ValueError("uncalibrated installation transforms must be null")

    def require_calibrated(self) -> None:
        raise RuntimeError(
            "Camera installation extrinsics are uncalibrated; world-frame geometry is unavailable"
        )


@dataclass(frozen=True, slots=True)
class CameraRigProfile:
    schema_version: int
    rig_id: str
    overview: OrbbecCameraProfile
    wrist: WristCameraProfile
    installation_calibration: InstallationCalibration
    verified_software: dict[str, str]

    def __post_init__(self) -> None:
        if self.schema_version != CAMERA_PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported camera profile schema_version={self.schema_version}"
            )
        _non_empty(self.rig_id, name="rig_id")
        expected_versions = {"pyorbbecsdk2", "orbbec_sdk", "opencv_python", "pyav"}
        if set(self.verified_software) != expected_versions:
            raise ValueError(
                "verified_software must contain pyorbbecsdk2, orbbec_sdk, "
                "opencv_python, and pyav"
            )
        for key, value in self.verified_software.items():
            _non_empty(value, name=f"verified_software.{key}")


@dataclass(frozen=True, slots=True)
class LoadedCameraRigProfile:
    profile: CameraRigProfile
    path: Path
    raw_bytes: bytes
    sha256: str
    calibration_path: Path
    calibration_bytes: bytes
    calibration_sha256: str


def _usb_profile(raw: Any, *, name: str) -> USBDeviceProfile:
    values = _strict_mapping(
        raw,
        name=name,
        required={"vendor_id", "product_id", "serial", "minimum_speed_mbps"},
    )
    return USBDeviceProfile(**values)


def _stream_profile(raw: Any, *, name: str) -> VideoStreamProfile:
    values = _strict_mapping(
        raw,
        name=name,
        required={"width", "height", "fps", "format"},
    )
    return VideoStreamProfile(**values)


def _parse_overview(raw: Any) -> OrbbecCameraProfile:
    values = _strict_mapping(
        raw,
        name="overview",
        required={
            "backend",
            "model",
            "usb",
            "firmware",
            "color",
            "depth",
            "alignment",
            "frame_sync",
            "factory_calibration",
        },
    )
    usb = _usb_profile(values.pop("usb"), name="overview.usb")
    color_raw = _strict_mapping(
        values.pop("color"),
        name="overview.color",
        required={"width", "height", "fps", "format", "output_color_order"},
    )
    output_color_order = color_raw.pop("output_color_order")
    depth_raw = _strict_mapping(
        values.pop("depth"),
        name="overview.depth",
        required={
            "width",
            "height",
            "fps",
            "format",
            "unit",
            "aligned_output_width",
            "aligned_output_height",
        },
    )
    depth_unit = depth_raw.pop("unit")
    aligned_width = depth_raw.pop("aligned_output_width")
    aligned_height = depth_raw.pop("aligned_output_height")
    calibration_raw = _strict_mapping(
        values.pop("factory_calibration"),
        name="overview.factory_calibration",
        required={"file", "sha256"},
    )
    return OrbbecCameraProfile(
        **values,
        usb=usb,
        color=VideoStreamProfile(**color_raw),
        output_color_order=output_color_order,
        depth=VideoStreamProfile(**depth_raw),
        depth_unit=depth_unit,
        aligned_output_width=aligned_width,
        aligned_output_height=aligned_height,
        factory_calibration=CalibrationReference(**calibration_raw),
    )


def _parse_wrist(raw: Any) -> WristCameraProfile:
    values = _strict_mapping(
        raw,
        name="wrist",
        required={
            "backend",
            "model",
            "usb",
            "device_by_id",
            "color",
            "exposure",
            "intrinsic_calibration",
        },
    )
    usb = _usb_profile(values.pop("usb"), name="wrist.usb")
    color_raw = _strict_mapping(
        values.pop("color"),
        name="wrist.color",
        required={
            "width",
            "height",
            "fps",
            "format",
            "driver_color_order",
            "output_color_order",
        },
    )
    driver_color_order = color_raw.pop("driver_color_order")
    output_color_order = color_raw.pop("output_color_order")
    exposure = _strict_mapping(
        values.pop("exposure"),
        name="wrist.exposure",
        required={"mode", "value", "gain"},
    )
    calibration = _strict_mapping(
        values.pop("intrinsic_calibration"),
        name="wrist.intrinsic_calibration",
        required={"status", "file"},
    )
    return WristCameraProfile(
        **values,
        usb=usb,
        color=VideoStreamProfile(**color_raw),
        driver_color_order=driver_color_order,
        output_color_order=output_color_order,
        exposure_mode=exposure["mode"],
        exposure_value=exposure["value"],
        gain=exposure["gain"],
        intrinsic_calibration_status=calibration["status"],
        intrinsic_calibration_file=calibration["file"],
    )


def _validate_calibration_payload(
    raw: Any,
    *,
    profile: CameraRigProfile,
) -> None:
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Orbbec factory calibration must use schema_version=1")
    if raw.get("source") != "orbbec_factory":
        raise ValueError("Orbbec calibration source must be orbbec_factory")
    device = raw.get("device")
    if not isinstance(device, dict):
        raise TypeError("Orbbec calibration device must be a mapping")
    expected_device = {
        "model": profile.overview.model,
        "serial": profile.overview.usb.serial,
        "usb_vendor_id": profile.overview.usb.vendor_id,
        "usb_product_id": profile.overview.usb.product_id,
        "firmware": profile.overview.firmware,
    }
    if device != expected_device:
        raise ValueError("Orbbec calibration device identity differs from camera profile")
    for name, expected in (
        ("color", profile.overview.color),
        ("depth", profile.overview.depth),
    ):
        section = raw.get(name)
        if not isinstance(section, dict) or not isinstance(section.get("stream"), dict):
            raise TypeError(f"Orbbec calibration {name}.stream must be a mapping")
        stream = dict(section["stream"])
        stream.pop("unit", None)
        if stream != {
            "width": expected.width,
            "height": expected.height,
            "fps": expected.fps,
            "format": expected.format,
        }:
            raise ValueError(f"Orbbec calibration {name} stream differs from profile")
        intrinsic = section.get("intrinsic")
        distortion = section.get("distortion")
        if not isinstance(intrinsic, dict) or set(intrinsic) != {"fx", "fy", "cx", "cy"}:
            raise ValueError(f"Orbbec calibration {name} intrinsic is invalid")
        if not isinstance(distortion, dict):
            raise TypeError(f"Orbbec calibration {name} distortion is invalid")
        numbers = [*intrinsic.values(), *distortion.get("k", []), *distortion.get("p", [])]
        if not numbers or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in numbers):
            raise ValueError(f"Orbbec calibration {name} contains non-finite coefficients")
    extrinsic = raw.get("depth_to_color")
    if not isinstance(extrinsic, dict):
        raise TypeError("Orbbec depth_to_color calibration is missing")
    rotation = extrinsic.get("rotation")
    translation = extrinsic.get("translation_mm")
    if (
        not isinstance(rotation, list)
        or len(rotation) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in rotation)
        or not isinstance(translation, list)
        or len(translation) != 3
    ):
        raise ValueError("Orbbec depth_to_color transform must be 3x3 plus xyz")


def load_camera_rig_profile(path: str | Path) -> LoadedCameraRigProfile:
    """Load and hash one strict camera profile plus its factory calibration."""

    profile_path = Path(path).expanduser().resolve()
    raw_bytes = profile_path.read_bytes()
    raw = yaml.safe_load(raw_bytes)
    values = _strict_mapping(
        raw,
        name="camera profile",
        required={
            "schema_version",
            "rig_id",
            "overview",
            "wrist",
            "installation_calibration",
            "verified_software",
        },
    )
    overview = _parse_overview(values.pop("overview"))
    wrist = _parse_wrist(values.pop("wrist"))
    installation_raw = _strict_mapping(
        values.pop("installation_calibration"),
        name="installation_calibration",
        required={"status", "overview_to_world", "wrist_to_gripper"},
    )
    software = _strict_mapping(
        values.pop("verified_software"),
        name="verified_software",
        required={"pyorbbecsdk2", "orbbec_sdk", "opencv_python", "pyav"},
    )
    profile = CameraRigProfile(
        **values,
        overview=overview,
        wrist=wrist,
        installation_calibration=InstallationCalibration(**installation_raw),
        verified_software={key: str(value) for key, value in software.items()},
    )
    calibration_path = (
        profile_path.parent / profile.overview.factory_calibration.file
    ).resolve()
    if profile_path.parent not in calibration_path.parents:
        raise ValueError("factory calibration path escapes the camera profile directory")
    calibration_bytes = calibration_path.read_bytes()
    calibration_sha256 = hashlib.sha256(calibration_bytes).hexdigest()
    if calibration_sha256 != profile.overview.factory_calibration.sha256:
        raise ValueError("Orbbec factory calibration SHA-256 does not match profile")
    _validate_calibration_payload(
        json.loads(calibration_bytes),
        profile=profile,
    )
    return LoadedCameraRigProfile(
        profile=profile,
        path=profile_path,
        raw_bytes=raw_bytes,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        calibration_path=calibration_path,
        calibration_bytes=calibration_bytes,
        calibration_sha256=calibration_sha256,
    )
