"""Stable USB and SDK identity resolution for the configured camera rig."""

from __future__ import annotations

import importlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .profile import (
    LoadedCameraRigProfile,
    OrbbecCameraProfile,
    USBDeviceProfile,
    WristCameraProfile,
)

_VIDEO_NODE = re.compile(r"^video[0-9]+$")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


@dataclass(frozen=True, slots=True)
class USBRuntimeIdentity:
    vendor_id: str
    product_id: str
    serial: str
    manufacturer: str
    product: str
    speed_mbps: int
    topology: str


@dataclass(frozen=True, slots=True)
class WristDeviceSelection:
    configured_path: Path
    resolved_node: Path
    usb: USBRuntimeIdentity


@dataclass(slots=True)
class OrbbecSDKSelection:
    """SDK handles retained together so DeviceList never outlives Context."""

    sdk: Any
    context: Any
    devices: Any
    device: Any
    pipeline: Any
    color_profile: Any
    depth_profile: Any
    usb: USBRuntimeIdentity
    name: str
    firmware: str
    hardware_version: str
    connection_type: str


def _usb_identity_at(path: Path) -> USBRuntimeIdentity | None:
    required = ("idVendor", "idProduct", "serial", "speed")
    if any(not (path / name).is_file() for name in required):
        return None
    return USBRuntimeIdentity(
        vendor_id=_read_text(path / "idVendor").lower(),
        product_id=_read_text(path / "idProduct").lower(),
        serial=_read_text(path / "serial"),
        manufacturer=_read_text(path / "manufacturer")
        if (path / "manufacturer").is_file()
        else "",
        product=_read_text(path / "product") if (path / "product").is_file() else "",
        speed_mbps=int(float(_read_text(path / "speed"))),
        topology=path.name,
    )


def _validate_usb_identity(
    actual: USBRuntimeIdentity,
    expected: USBDeviceProfile,
    *,
    label: str,
) -> None:
    for field_name in ("vendor_id", "product_id", "serial"):
        actual_value = getattr(actual, field_name)
        expected_value = getattr(expected, field_name)
        if actual_value != expected_value:
            raise RuntimeError(
                f"{label} USB {field_name} mismatch: expected {expected_value!r}, "
                f"got {actual_value!r}"
            )
    if actual.speed_mbps < expected.minimum_speed_mbps:
        raise RuntimeError(
            f"{label} USB speed {actual.speed_mbps} Mbps is below required "
            f"{expected.minimum_speed_mbps} Mbps"
        )


def find_usb_device(
    expected: USBDeviceProfile,
    *,
    sys_usb_root: str | Path = "/sys/bus/usb/devices",
    label: str,
) -> USBRuntimeIdentity:
    """Resolve one exact USB identity without treating topology as identity."""

    matches: list[USBRuntimeIdentity] = []
    root = Path(sys_usb_root)
    for candidate in root.iterdir():
        identity = _usb_identity_at(candidate)
        if identity is None:
            continue
        if (
            identity.vendor_id == expected.vendor_id
            and identity.product_id == expected.product_id
            and identity.serial == expected.serial
        ):
            matches.append(identity)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {label} USB device "
            f"{expected.vendor_id}:{expected.product_id}/{expected.serial}, found {len(matches)}"
        )
    _validate_usb_identity(matches[0], expected, label=label)
    return matches[0]


def resolve_wrist_device(
    profile: WristCameraProfile,
    *,
    by_id_root: str | Path = "/dev/v4l/by-id",
    sys_video_root: str | Path = "/sys/class/video4linux",
) -> WristDeviceSelection:
    """Resolve the configured by-id link and verify the target's real USB parent."""

    profile_path = Path(profile.device_by_id)
    if profile_path.parent != Path("/dev/v4l/by-id"):
        raise RuntimeError("Wrist camera selection must use /dev/v4l/by-id")
    if not profile_path.name.endswith("-video-index0"):
        raise RuntimeError("Wrist camera selection must use video-index0")
    configured = Path(by_id_root) / profile_path.name
    try:
        resolved = configured.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Configured wrist camera is not connected: {configured}") from exc
    if not _VIDEO_NODE.fullmatch(resolved.name):
        raise RuntimeError(f"Wrist by-id path does not resolve to a video node: {resolved}")
    sys_device = (Path(sys_video_root) / resolved.name / "device").resolve(strict=True)
    usb = next(
        (
            identity
            for parent in (sys_device, *sys_device.parents)
            if (identity := _usb_identity_at(parent)) is not None
        ),
        None,
    )
    if usb is None:
        raise RuntimeError(f"Could not find USB identity for wrist node {resolved}")
    _validate_usb_identity(usb, profile.usb, label="wrist")
    if usb.product != profile.model:
        raise RuntimeError(
            f"Wrist model mismatch: expected {profile.model!r}, got {usb.product!r}"
        )
    return WristDeviceSelection(configured, resolved, usb)


def _sdk_format(sdk: Any, name: str) -> Any:
    try:
        return getattr(sdk.OBFormat, name)
    except AttributeError as exc:
        raise RuntimeError(f"Orbbec SDK does not expose OBFormat.{name}") from exc


def _validate_selected_profile(actual: Any, expected: Any, *, label: str) -> None:
    values = (
        int(actual.get_width()),
        int(actual.get_height()),
        int(actual.get_fps()),
        str(getattr(actual.get_format(), "name", actual.get_format())),
    )
    wanted = (expected.width, expected.height, expected.fps, expected.format)
    if values != wanted:
        raise RuntimeError(f"{label} stream mismatch: expected {wanted}, got {values}")


def select_orbbec_device(
    profile: OrbbecCameraProfile,
    *,
    sdk: Any | None = None,
    sys_usb_root: str | Path = "/sys/bus/usb/devices",
) -> OrbbecSDKSelection:
    """Select and validate the exact Orbbec device while retaining SDK owners."""

    if sdk is None:
        try:
            sdk = importlib.import_module("pyorbbecsdk")
        except ImportError as exc:
            raise RuntimeError(
                "Orbbec SDK is unavailable; install pyorbbecsdk2==2.1.2 in the hardware environment"
            ) from exc
    context = sdk.Context()
    devices = context.query_devices()
    try:
        device = devices.get_device_by_serial_number(profile.usb.serial)
    except Exception as exc:
        raise RuntimeError(
            f"Orbbec device with serial {profile.usb.serial!r} was not found"
        ) from exc
    info = device.get_device_info()
    actual_name = str(info.get_name())
    actual_serial = str(info.get_serial_number())
    actual_firmware = str(info.get_firmware_version())
    actual_vid = f"{int(info.get_vid()):04x}"
    actual_pid = f"{int(info.get_pid()):04x}"
    if actual_name != profile.model:
        raise RuntimeError(
            f"Orbbec model mismatch: expected {profile.model!r}, got {actual_name!r}"
        )
    if actual_serial != profile.usb.serial:
        raise RuntimeError(
            f"Orbbec serial mismatch: expected {profile.usb.serial!r}, got {actual_serial!r}"
        )
    if (actual_vid, actual_pid) != (profile.usb.vendor_id, profile.usb.product_id):
        raise RuntimeError(
            "Orbbec VID:PID mismatch: expected "
            f"{profile.usb.vendor_id}:{profile.usb.product_id}, got {actual_vid}:{actual_pid}"
        )
    if actual_firmware != profile.firmware:
        raise RuntimeError(
            f"Orbbec firmware mismatch: expected {profile.firmware!r}, got {actual_firmware!r}"
        )
    usb = find_usb_device(profile.usb, sys_usb_root=sys_usb_root, label="overview")
    pipeline = sdk.Pipeline(device)
    color_profile = pipeline.get_stream_profile_list(
        sdk.OBSensorType.COLOR_SENSOR
    ).get_video_stream_profile(
        profile.color.width,
        profile.color.height,
        _sdk_format(sdk, profile.color.format),
        profile.color.fps,
    )
    depth_profile = pipeline.get_stream_profile_list(
        sdk.OBSensorType.DEPTH_SENSOR
    ).get_video_stream_profile(
        profile.depth.width,
        profile.depth.height,
        _sdk_format(sdk, profile.depth.format),
        profile.depth.fps,
    )
    _validate_selected_profile(color_profile, profile.color, label="Orbbec color")
    _validate_selected_profile(depth_profile, profile.depth, label="Orbbec depth")
    return OrbbecSDKSelection(
        sdk=sdk,
        context=context,
        devices=devices,
        device=device,
        pipeline=pipeline,
        color_profile=color_profile,
        depth_profile=depth_profile,
        usb=usb,
        name=actual_name,
        firmware=actual_firmware,
        hardware_version=str(info.get_hardware_version()),
        connection_type=str(info.get_connection_type()),
    )


def _assert_close(actual: float, expected: float, *, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(f"{label} differs from factory calibration: {actual} != {expected}")


def validate_live_orbbec_calibration(
    selection: OrbbecSDKSelection,
    loaded: LoadedCameraRigProfile,
) -> None:
    """Verify that active-profile factory calibration matches the checked-in snapshot."""

    expected = json.loads(loaded.calibration_bytes)
    for label, stream, section_name in (
        ("color", selection.color_profile, "color"),
        ("depth", selection.depth_profile, "depth"),
    ):
        intrinsic = stream.get_intrinsic()
        distortion = stream.get_distortion()
        expected_section = expected[section_name]
        for name in ("fx", "fy", "cx", "cy"):
            _assert_close(
                float(getattr(intrinsic, name)),
                float(expected_section["intrinsic"][name]),
                label=f"Orbbec {label} intrinsic {name}",
            )
        expected_distortion = expected_section["distortion"]
        actual_model = str(getattr(distortion.model, "name", distortion.model))
        if actual_model != expected_distortion["model"]:
            raise RuntimeError(
                f"Orbbec {label} distortion model differs: "
                f"{actual_model!r} != {expected_distortion['model']!r}"
            )
        for index, name in enumerate(("k1", "k2", "k3", "k4", "k5", "k6")):
            _assert_close(
                float(getattr(distortion, name)),
                float(expected_distortion["k"][index]),
                label=f"Orbbec {label} distortion {name}",
            )
        for index, name in enumerate(("p1", "p2")):
            _assert_close(
                float(getattr(distortion, name)),
                float(expected_distortion["p"][index]),
                label=f"Orbbec {label} distortion {name}",
            )
    extrinsic = selection.depth_profile.get_extrinsic_to(selection.color_profile)
    expected_extrinsic = expected["depth_to_color"]
    for row_index, row in enumerate(extrinsic.rot.tolist()):
        for column_index, value in enumerate(row):
            _assert_close(
                float(value),
                float(expected_extrinsic["rotation"][row_index][column_index]),
                label=f"Orbbec depth-to-color rotation[{row_index},{column_index}]",
            )
    for index, value in enumerate(extrinsic.transform.tolist()):
        _assert_close(
            float(value),
            float(expected_extrinsic["translation_mm"][index]),
            label=f"Orbbec depth-to-color translation[{index}]",
        )
