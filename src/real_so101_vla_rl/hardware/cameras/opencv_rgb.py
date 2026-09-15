"""Timestamped wrist RGB adapter around LeRobot's validated OpenCVCamera."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from real_so101_vla_rl.data.schema import IMAGE_HEIGHT, IMAGE_WIDTH

from ..capture_types import CapturedSample


class OpenCVRGBAdapter:
    """Read unconsumed BGR frames and expose strict schema-v2 RGB samples."""

    def __init__(
        self,
        camera: Any,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        max_consecutive_failures: int = 3,
    ) -> None:
        if max_consecutive_failures <= 0:
            raise ValueError("max_consecutive_failures must be positive")
        self._camera = camera
        self._clock_ns = clock_ns
        self._max_failures = max_consecutive_failures
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

    @property
    def is_connected(self) -> bool:
        return bool(getattr(self._camera, "is_connected", False))

    @property
    def is_broken(self) -> bool:
        return self._failures >= self._max_failures

    def connect(self) -> None:
        self._camera.connect()

    def read(self, timeout_ms: int) -> CapturedSample[np.ndarray]:
        sequence_id = self._sequence_id
        self._sequence_id += 1
        try:
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
