"""Timestamped wrist RGB adapter around LeRobot's validated OpenCVCamera."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from real_so101_vla_rl.data.schema import IMAGE_HEIGHT, IMAGE_WIDTH

from ..capture_types import CapturedSample
from .identity import WristDeviceSelection, resolve_wrist_device
from .profile import LoadedCameraRigProfile, WristCameraProfile


class OpenCVRGBAdapter:
    """Read unconsumed BGR frames and expose strict schema-v2 RGB samples."""

    def __init__(
        self,
        camera: Any,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        perf_counter_ns: Callable[[], int] = time.perf_counter_ns,
        max_consecutive_failures: int = 3,
        expected_profile: WristCameraProfile | None = None,
        device_selection: WristDeviceSelection | None = None,
    ) -> None:
        if max_consecutive_failures <= 0:
            raise ValueError("max_consecutive_failures must be positive")
        self._camera = camera
        self._clock_ns = clock_ns
        self._perf_to_monotonic_offset_ns = clock_ns() - perf_counter_ns()
        self._max_failures = max_consecutive_failures
        self._expected_profile = expected_profile
        self._device_selection = device_selection
        self._failures = 0
        self._sequence_id = 0

    @classmethod
    def from_device(
        cls,
        device: str | int,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        max_consecutive_failures: int = 3,
    ) -> OpenCVRGBAdapter:
        """Create the upstream LeRobot camera without assigning a default device."""

        try:
            from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
            from lerobot.cameras.opencv.configuration_opencv import ColorMode
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot OpenCV camera support is unavailable; install the future "
                "project hardware extra before real recording"
            ) from exc

        index_or_path = Path(device) if isinstance(device, str) else device
        camera = OpenCVCamera(
            OpenCVCameraConfig(
                index_or_path=index_or_path,
                fps=30,
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
                color_mode=ColorMode.BGR,
                fourcc="MJPG",
            )
        )
        return cls(
            camera,
            clock_ns=clock_ns,
            max_consecutive_failures=max_consecutive_failures,
        )

    @classmethod
    def from_profile(
        cls,
        loaded: LoadedCameraRigProfile,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        max_consecutive_failures: int = 3,
        by_id_root: str = "/dev/v4l/by-id",
        sys_video_root: str = "/sys/class/video4linux",
    ) -> OpenCVRGBAdapter:
        """Resolve and construct the exact wrist camera from its stable profile."""

        profile = loaded.profile.wrist
        selection = resolve_wrist_device(
            profile,
            by_id_root=by_id_root,
            sys_video_root=sys_video_root,
        )
        adapter = cls.from_device(
            str(selection.configured_path),
            clock_ns=clock_ns,
            max_consecutive_failures=max_consecutive_failures,
        )
        adapter._expected_profile = profile
        adapter._device_selection = selection
        return adapter

    @property
    def device_selection(self) -> WristDeviceSelection | None:
        return self._device_selection

    @property
    def is_connected(self) -> bool:
        return bool(getattr(self._camera, "is_connected", False))

    @property
    def is_broken(self) -> bool:
        return self._failures >= self._max_failures

    def connect(self) -> None:
        self._camera.connect()
        if self._expected_profile is None:
            return
        try:
            import cv2

            capture = self._camera.videocapture
            if capture is None:
                raise RuntimeError("Wrist camera opened without a VideoCapture handle")
            controls = self._expected_profile
            requested_controls = (
                (cv2.CAP_PROP_AUTO_EXPOSURE, 1.0, "manual exposure mode"),
                (cv2.CAP_PROP_EXPOSURE, float(controls.exposure_value), "exposure"),
                (cv2.CAP_PROP_GAIN, float(controls.gain), "gain"),
            )
            for property_id, value, name in requested_controls:
                if not capture.set(property_id, value):
                    raise RuntimeError(
                        f"Wrist camera rejected required {name} value {value:g}"
                    )
            actual_controls = {
                "auto exposure": float(capture.get(cv2.CAP_PROP_AUTO_EXPOSURE)),
                "exposure": float(capture.get(cv2.CAP_PROP_EXPOSURE)),
                "gain": float(capture.get(cv2.CAP_PROP_GAIN)),
            }
            expected_controls = {
                "auto exposure": 1.0,
                "exposure": float(controls.exposure_value),
                "gain": float(controls.gain),
            }
            mismatches = {
                name: (expected_controls[name], actual)
                for name, actual in actual_controls.items()
                if not np.isclose(actual, expected_controls[name], atol=0.5, rtol=0.0)
            }
            if mismatches:
                raise RuntimeError(
                    "Wrist camera control readback mismatch "
                    f"(expected, actual): {mismatches}"
                )
            profile = self._expected_profile.color
            actual_width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
            code = int(capture.get(cv2.CAP_PROP_FOURCC))
            actual_format = "".join(chr((code >> (8 * index)) & 0xFF) for index in range(4))
            actual = (actual_width, actual_height, actual_fps, actual_format)
            expected = (profile.width, profile.height, float(profile.fps), profile.format)
            if actual != expected:
                raise RuntimeError(
                    f"Wrist stream mismatch: expected {expected}, got {actual}"
                )
        except BaseException:
            self._camera.disconnect()
            raise

    def read(self, timeout_ms: int) -> CapturedSample[np.ndarray]:
        sequence_id = self._sequence_id
        self._sequence_id += 1
        try:
            new_frame_event = getattr(self._camera, "new_frame_event", None)
            frame_lock = getattr(self._camera, "frame_lock", None)
            if new_frame_event is not None and frame_lock is not None:
                thread = getattr(self._camera, "thread", None)
                if thread is not None and not thread.is_alive():
                    raise RuntimeError("Wrist camera read thread is not running")
                if not new_frame_event.wait(timeout=timeout_ms / 1000):
                    raise TimeoutError(
                        f"Timed out waiting for wrist frame after {timeout_ms} ms"
                    )
                with frame_lock:
                    latest_frame = getattr(self._camera, "latest_frame", None)
                    latest_timestamp = getattr(self._camera, "latest_timestamp", None)
                    new_frame_event.clear()
                if latest_frame is None or latest_timestamp is None:
                    raise RuntimeError("Wrist frame event had no image or timestamp")
                bgr = np.asarray(latest_frame)
                timestamp_ns = (
                    round(float(latest_timestamp) * 1_000_000_000)
                    + self._perf_to_monotonic_offset_ns
                )
            else:
                bgr = np.asarray(self._camera.async_read(timeout_ms=timeout_ms))
                timestamp_ns = self._clock_ns()
            if bgr.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
                raise ValueError(
                    f"wrist frame must have shape {(IMAGE_HEIGHT, IMAGE_WIDTH, 3)}, "
                    f"got {bgr.shape}"
                )
            if bgr.dtype != np.uint8:
                raise TypeError(f"wrist frame must use uint8, got {bgr.dtype}")
            rgb = np.ascontiguousarray(bgr[..., ::-1])
        except Exception as exc:  # noqa: BLE001 - adapter turns driver failures into data
            self._failures += 1
            return CapturedSample.failure(
                timestamp_ns=self._clock_ns(),
                sequence_id=sequence_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        self._failures = 0
        return CapturedSample.success(
            rgb, timestamp_ns=timestamp_ns, sequence_id=sequence_id
        )

    def disconnect(self) -> None:
        if self.is_connected:
            self._camera.disconnect()
