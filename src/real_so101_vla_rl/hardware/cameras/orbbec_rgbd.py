"""Orbbec color plus depth-to-color aligned capture for schema v2."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from real_so101_vla_rl.data.schema import IMAGE_HEIGHT, IMAGE_WIDTH

from ..capture_types import CapturedSample, RGBDFrame


def _format_name(frame: Any) -> str:
    value = frame.get_format()
    return str(getattr(value, "name", value)).upper()


def decode_orbbec_color(frame: Any) -> np.ndarray:
    """Decode an SDK color frame to contiguous HWC RGB uint8."""

    width = int(frame.get_width())
    height = int(frame.get_height())
    data = np.asarray(frame.get_data(), dtype=np.uint8).reshape(-1)
    pixel_format = _format_name(frame)

    if "MJPG" in pixel_format or "MJPEG" in pixel_format:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to decode Orbbec MJPG color") from exc
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Orbbec MJPG decode returned no image")
        rgb = bgr[..., ::-1]
    elif pixel_format.endswith("BGR") or "BGR888" in pixel_format:
        rgb = data.reshape(height, width, 3)[..., ::-1]
    elif pixel_format.endswith("RGB") or "RGB888" in pixel_format:
        rgb = data.reshape(height, width, 3)
    elif "YUYV" in pixel_format or "YUY2" in pixel_format:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to decode Orbbec YUYV color") from exc
        yuyv = data.reshape(height, width, 2)
        rgb = cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUY2)
    else:
        raise ValueError(f"Unsupported Orbbec color format: {pixel_format}")

    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if rgb.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
        raise ValueError(
            f"aligned overview RGB must have shape {(IMAGE_HEIGHT, IMAGE_WIDTH, 3)}, "
            f"got {rgb.shape}"
        )
    return rgb


def decode_orbbec_depth_mm(frame: Any) -> np.ndarray:
    """Convert one aligned Orbbec depth frame to HWC uint16 millimetres."""

    width = int(frame.get_width())
    height = int(frame.get_height())
    count = width * height
    raw = np.frombuffer(frame.get_data(), dtype=np.uint16, count=count)
    if raw.size != count:
        raise ValueError(f"Orbbec depth contains {raw.size} pixels, expected {count}")
    raw = raw.reshape(height, width)
    scale = float(frame.get_depth_scale())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Orbbec depth scale must be positive, got {scale}")
    depth_mm = np.rint(raw.astype(np.float32) * scale)
    depth_mm = np.clip(depth_mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    depth_mm = np.ascontiguousarray(depth_mm[..., None])
    if depth_mm.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 1):
        raise ValueError(
            f"aligned overview depth must have shape {(IMAGE_HEIGHT, IMAGE_WIDTH, 1)}, "
            f"got {depth_mm.shape}"
        )
    return depth_mm


class OrbbecRGBDAdapter:
    """Return paired overview RGB/depth samples from one aligned SDK frameset."""

    def __init__(
        self,
        pipeline: Any,
        *,
        config: Any | None = None,
        align_filter: Any | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        max_consecutive_failures: int = 3,
        enable_frame_sync: bool = True,
    ) -> None:
        if max_consecutive_failures <= 0:
            raise ValueError("max_consecutive_failures must be positive")
        self._pipeline = pipeline
        self._config = config
        self._align_filter = align_filter
        self._clock_ns = clock_ns
        self._max_failures = max_consecutive_failures
        self._enable_frame_sync = enable_frame_sync
        self._failures = 0
        self._sequence_id = 0
        self._connected = False

    @classmethod
    def from_sdk(
        cls,
        *,
        serial_number: str,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        max_consecutive_failures: int = 3,
    ) -> OrbbecRGBDAdapter:
        """Construct fixed streams; stable device discovery is supplied separately."""

        if not serial_number.strip():
            raise ValueError("The Orbbec serial number must be provided")
        try:
            from pyorbbecsdk import (
                AlignFilter,
                Config,
                Context,
                OBFormat,
                OBFrameAggregateOutputMode,
                OBSensorType,
                OBStreamType,
                Pipeline,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Orbbec SDK is unavailable; install pyorbbecsdk2==2.1.2 in the "
                "hardware environment"
            ) from exc

        devices = Context().query_devices()
        device = devices.get_device_by_serial_number(serial_number)
        pipeline = Pipeline(device)
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        color_profile = color_profiles.get_video_stream_profile(
            IMAGE_WIDTH, IMAGE_HEIGHT, OBFormat.MJPG, 30
        )
        depth_profile = depth_profiles.get_video_stream_profile(
            640, 400, OBFormat.Y16, 30
        )
        config = Config()
        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)
        config.set_frame_aggregate_output_mode(
            OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
        )
        align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        return cls(
            pipeline,
            config=config,
            align_filter=align_filter,
            clock_ns=clock_ns,
            max_consecutive_failures=max_consecutive_failures,
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_broken(self) -> bool:
        return self._failures >= self._max_failures

    def connect(self) -> None:
        if self._connected:
            return
        if self._enable_frame_sync:
            self._pipeline.enable_frame_sync()
        if self._config is None:
            self._pipeline.start()
        else:
            self._pipeline.start(self._config)
        self._connected = True

    def _failure_pair(self, *, timestamp_ns: int, sequence_id: int, error: str) -> RGBDFrame:
        return RGBDFrame(
            CapturedSample.failure(
                timestamp_ns=timestamp_ns, sequence_id=sequence_id, error=error
            ),
            CapturedSample.failure(
                timestamp_ns=timestamp_ns, sequence_id=sequence_id, error=error
            ),
        )

    def read(self, timeout_ms: int) -> RGBDFrame:
        sequence_id = self._sequence_id
        self._sequence_id += 1
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms)
            timestamp_ns = self._clock_ns()
            if not frames:
                raise TimeoutError(f"Orbbec produced no frameset within {timeout_ms} ms")
            if self._align_filter is not None:
                frames = self._align_filter.process(frames)
                if not frames:
                    raise ValueError("Orbbec depth-to-color alignment returned no frameset")
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                raise ValueError("Orbbec frameset is missing color or aligned depth")
            overview = decode_orbbec_color(color_frame)
            depth = decode_orbbec_depth_mm(depth_frame)
        except Exception as exc:  # noqa: BLE001 - SDK failures are capture results
            self._failures += 1
            return self._failure_pair(
                timestamp_ns=self._clock_ns(),
                sequence_id=sequence_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        self._failures = 0
        return RGBDFrame(
            CapturedSample.success(
                overview, timestamp_ns=timestamp_ns, sequence_id=sequence_id
            ),
            CapturedSample.success(
                depth, timestamp_ns=timestamp_ns, sequence_id=sequence_id
            ),
        )

    def disconnect(self) -> None:
        if self._connected:
            self._pipeline.stop()
            self._connected = False
