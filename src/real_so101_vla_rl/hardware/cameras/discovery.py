"""Hardware preflight and structured reports for the frozen camera rig."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..time_matching import closest_camera_pair
from .identity import (
    OrbbecSDKSelection,
    WristDeviceSelection,
    resolve_wrist_device,
    select_orbbec_device,
    validate_live_orbbec_calibration,
)
from .profile import LoadedCameraRigProfile

_MAX_SAMPLE_ERROR_DETAILS = 10


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _software_versions(sdk: Any | None = None) -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "pyorbbecsdk2": _distribution_version("pyorbbecsdk2"),
        "opencv_python": _distribution_version("opencv-python"),
        "pyav": _distribution_version("av"),
    }
    if sdk is not None and hasattr(sdk, "get_version"):
        versions["orbbec_sdk"] = str(sdk.get_version())
    else:
        versions["orbbec_sdk"] = "unavailable"
    return versions


@dataclass(frozen=True, slots=True)
class CameraProbeReport:
    schema_version: int
    checked_at_utc: str
    rig_id: str
    profile_sha256: str
    calibration_sha256: str
    check_streams: bool
    duration_s: float
    timestamp_tolerance_ms: float
    valid: bool
    overview: dict[str, Any]
    wrist: dict[str, Any]
    software: dict[str, str]
    attempted_samples: int
    valid_samples: int
    valid_ratio: float | None
    measured_pair_fps: float | None
    first_frame_latency_ms: float | None
    p95_camera_timestamp_span_ms: float | None
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)


def _overview_report(selection: OrbbecSDKSelection) -> dict[str, Any]:
    return {
        "model": selection.name,
        "serial": selection.usb.serial,
        "vendor_id": selection.usb.vendor_id,
        "product_id": selection.usb.product_id,
        "firmware": selection.firmware,
        "hardware_version": selection.hardware_version,
        "connection_type": selection.connection_type,
        "usb_speed_mbps": selection.usb.speed_mbps,
        "usb_topology": selection.usb.topology,
        "color_stream": {
            "width": int(selection.color_profile.get_width()),
            "height": int(selection.color_profile.get_height()),
            "fps": int(selection.color_profile.get_fps()),
            "format": str(
                getattr(
                    selection.color_profile.get_format(),
                    "name",
                    selection.color_profile.get_format(),
                )
            ),
        },
        "depth_stream": {
            "width": int(selection.depth_profile.get_width()),
            "height": int(selection.depth_profile.get_height()),
            "fps": int(selection.depth_profile.get_fps()),
            "format": str(
                getattr(
                    selection.depth_profile.get_format(),
                    "name",
                    selection.depth_profile.get_format(),
                )
            ),
        },
    }


def _wrist_report(selection: WristDeviceSelection, loaded: LoadedCameraRigProfile) -> dict[str, Any]:
    stream = loaded.profile.wrist.color
    return {
        "model": loaded.profile.wrist.model,
        "serial": selection.usb.serial,
        "vendor_id": selection.usb.vendor_id,
        "product_id": selection.usb.product_id,
        "configured_path": str(selection.configured_path),
        "resolved_node": str(selection.resolved_node),
        "usb_speed_mbps": selection.usb.speed_mbps,
        "usb_topology": selection.usb.topology,
        "color_stream": {
            "width": stream.width,
            "height": stream.height,
            "fps": stream.fps,
            "format": stream.format,
        },
        "exposure": {
            "mode": loaded.profile.wrist.exposure_mode,
            "value": loaded.profile.wrist.exposure_value,
            "gain": loaded.profile.wrist.gain,
        },
    }


def probe_camera_rig(
    loaded: LoadedCameraRigProfile,
    *,
    check_streams: bool = False,
    duration_s: float = 1.0,
    timeout_ms: int = 2000,
    timestamp_tolerance_ms: float = 25.0,
    sdk: Any | None = None,
    sys_usb_root: str = "/sys/bus/usb/devices",
    by_id_root: str = "/dev/v4l/by-id",
    sys_video_root: str = "/sys/class/video4linux",
) -> CameraProbeReport:
    """Probe exact identities and optionally acquire both camera streams concurrently."""

    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if timeout_ms <= 0:
        raise ValueError("timeout_ms must be positive")
    if timestamp_tolerance_ms < 0:
        raise ValueError("timestamp_tolerance_ms must be non-negative")

    checked_at = datetime.now(UTC).isoformat()
    errors: list[str] = []
    overview_data: dict[str, Any] = {}
    wrist_data: dict[str, Any] = {}
    attempted = 0
    valid_samples = 0
    invalid_samples = 0
    first_frame_latency_ms: float | None = None
    measured_pair_fps: float | None = None
    spans_ms: list[float] = []
    selection: OrbbecSDKSelection | None = None
    wrist_selection: WristDeviceSelection | None = None
    overview_adapter: Any | None = None
    wrist_adapter: Any | None = None

    try:
        if check_streams:
            from .opencv_rgb import OpenCVRGBAdapter
            from .orbbec_rgbd import OrbbecRGBDAdapter

            overview_adapter = OrbbecRGBDAdapter.from_profile(
                loaded,
                sdk=sdk,
                sys_usb_root=sys_usb_root,
            )
            selection = overview_adapter.sdk_selection
            assert selection is not None
            wrist_adapter = OpenCVRGBAdapter.from_profile(
                loaded,
                by_id_root=by_id_root,
                sys_video_root=sys_video_root,
            )
            wrist_selection = wrist_adapter.device_selection
            assert wrist_selection is not None
            overview_data = _overview_report(selection)
            wrist_data = _wrist_report(wrist_selection, loaded)

            started = time.perf_counter()
            overview_adapter.connect()
            wrist_adapter.connect()
            sample_target = max(1, round(duration_s * loaded.profile.overview.color.fps))
            sampling_started = time.perf_counter()
            overview_buffer: deque[Any] = deque(maxlen=4)
            wrist_buffer: deque[Any] = deque(maxlen=4)
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="camera_probe") as executor:
                for sample_index in range(sample_target):
                    attempted += 1
                    overview_future = executor.submit(overview_adapter.read, timeout_ms)
                    wrist_future = executor.submit(wrist_adapter.read, timeout_ms)
                    rgbd = overview_future.result(timeout=timeout_ms / 1000 + 1.0)
                    wrist = wrist_future.result(timeout=timeout_ms / 1000 + 1.0)
                    if sample_index == 0:
                        first_frame_latency_ms = (time.perf_counter() - started) * 1000
                    if rgbd.overview.valid and rgbd.depth.valid:
                        overview_buffer.append(rgbd)
                    if wrist.valid:
                        wrist_buffer.append(wrist)
                    if not rgbd.overview.valid or not rgbd.depth.valid or not wrist.valid:
                        invalid_samples += 1
                        if invalid_samples <= _MAX_SAMPLE_ERROR_DETAILS:
                            errors.append(
                                "stream sample invalid: "
                                f"overview={rgbd.overview.error}, depth={rgbd.depth.error}, "
                                f"wrist={wrist.error}"
                            )
                        continue

                    selected = closest_camera_pair(overview_buffer, wrist_buffer)
                    for _ in range(3):
                        if (
                            selected is None
                            or selected[2] <= timestamp_tolerance_ms * 1_000_000
                        ):
                            break
                        selected_rgbd, selected_wrist, _ = selected
                        if (
                            selected_rgbd.overview.timestamp_ns
                            <= selected_wrist.timestamp_ns
                        ):
                            candidate = overview_adapter.read(timeout_ms)
                            if candidate.overview.valid and candidate.depth.valid:
                                overview_buffer.append(candidate)
                            else:
                                break
                        else:
                            candidate = wrist_adapter.read(timeout_ms)
                            if candidate.valid:
                                wrist_buffer.append(candidate)
                            else:
                                break
                        selected = closest_camera_pair(overview_buffer, wrist_buffer)

                    if (
                        selected is None
                        or selected[2] > timestamp_tolerance_ms * 1_000_000
                    ):
                        invalid_samples += 1
                        if invalid_samples <= _MAX_SAMPLE_ERROR_DETAILS:
                            best_span = None if selected is None else selected[2] / 1_000_000
                            errors.append(
                                "no unique camera pair within timestamp tolerance: "
                                f"best_span_ms={best_span}"
                            )
                        continue

                    rgbd, wrist, span_ns = selected
                    valid_samples += 1
                    if valid_samples == 1:
                        overview_data["output"] = {
                            "rgb_shape": list(rgbd.overview.value.shape),
                            "rgb_dtype": str(rgbd.overview.value.dtype),
                            "depth_shape": list(rgbd.depth.value.shape),
                            "depth_dtype": str(rgbd.depth.value.dtype),
                            "depth_unit": loaded.profile.overview.depth_unit,
                            "nonzero_depth_ratio": float(
                                np.count_nonzero(rgbd.depth.value) / rgbd.depth.value.size
                            ),
                        }
                        wrist_data["output"] = {
                            "rgb_shape": list(wrist.value.shape),
                            "rgb_dtype": str(wrist.value.dtype),
                            "color_order": loaded.profile.wrist.output_color_order,
                        }
                    spans_ms.append(span_ns / 1_000_000)
                    overview_buffer = deque(
                        (
                            item
                            for item in overview_buffer
                            if item.overview.sequence_id > rgbd.overview.sequence_id
                        ),
                        maxlen=4,
                    )
                    wrist_buffer = deque(
                        (
                            item
                            for item in wrist_buffer
                            if item.sequence_id > wrist.sequence_id
                        ),
                        maxlen=4,
                    )
            sampling_elapsed = time.perf_counter() - sampling_started
            if sampling_elapsed > 0:
                measured_pair_fps = valid_samples / sampling_elapsed
        else:
            selection = select_orbbec_device(
                loaded.profile.overview,
                sdk=sdk,
                sys_usb_root=sys_usb_root,
            )
            validate_live_orbbec_calibration(selection, loaded)
            wrist_selection = resolve_wrist_device(
                loaded.profile.wrist,
                by_id_root=by_id_root,
                sys_video_root=sys_video_root,
            )
            overview_data = _overview_report(selection)
            wrist_data = _wrist_report(wrist_selection, loaded)
    except Exception as exc:  # noqa: BLE001 - report exact hardware preflight failure
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if wrist_adapter is not None and wrist_adapter.is_connected:
            wrist_adapter.disconnect()
        if overview_adapter is not None and overview_adapter.is_connected:
            overview_adapter.disconnect()

    ratio = valid_samples / attempted if attempted else None
    p95 = float(np.percentile(spans_ms, 95)) if spans_ms else None
    if invalid_samples > _MAX_SAMPLE_ERROR_DETAILS:
        errors.append(
            f"{invalid_samples - _MAX_SAMPLE_ERROR_DETAILS} additional invalid stream "
            "samples omitted"
        )
    if check_streams and ratio is not None and ratio < 0.99:
        errors.append(f"valid camera sample ratio {ratio:.6f} is below 0.99")
    minimum_fps = loaded.profile.overview.color.fps * 0.95
    if check_streams and measured_pair_fps is not None and measured_pair_fps < minimum_fps:
        errors.append(
            f"measured paired camera rate {measured_pair_fps:.3f} fps is below "
            f"{minimum_fps:.3f} fps"
        )
    if check_streams and p95 is not None and p95 > timestamp_tolerance_ms:
        errors.append(
            f"p95 camera timestamp span {p95:.3f} ms exceeds "
            f"{timestamp_tolerance_ms:.3f} ms"
        )
    software = _software_versions(selection.sdk if selection is not None else sdk)
    return CameraProbeReport(
        schema_version=1,
        checked_at_utc=checked_at,
        rig_id=loaded.profile.rig_id,
        profile_sha256=loaded.sha256,
        calibration_sha256=loaded.calibration_sha256,
        check_streams=check_streams,
        duration_s=duration_s,
        timestamp_tolerance_ms=timestamp_tolerance_ms,
        valid=not errors,
        overview=overview_data,
        wrist=wrist_data,
        software=software,
        attempted_samples=attempted,
        valid_samples=valid_samples,
        valid_ratio=ratio,
        measured_pair_fps=measured_pair_fps,
        first_frame_latency_ms=first_frame_latency_ms,
        p95_camera_timestamp_span_ms=p95,
        errors=tuple(errors),
    )


def require_valid_camera_probe(report: CameraProbeReport) -> None:
    if not report.valid:
        raise RuntimeError("Camera preflight failed: " + "; ".join(report.errors))


def write_camera_probe_report(path: str | Path, report: CameraProbeReport) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.to_json() + "\n", encoding="utf-8")
