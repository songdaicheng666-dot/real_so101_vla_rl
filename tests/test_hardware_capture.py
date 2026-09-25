from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock
from typing import Any

import numpy as np

from real_so101_vla_rl.data import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SENSOR_TIMESTAMP_KEYS,
)
from real_so101_vla_rl.hardware import (
    CapturedSample,
    RGBDFrame,
    SO101ObservationSynchronizer,
)
from real_so101_vla_rl.hardware.cameras import (
    OpenCVRGBAdapter,
    OrbbecRGBDAdapter,
    load_camera_rig_profile,
)
from real_so101_vla_rl.hardware.cameras.orbbec_rgbd import (
    decode_orbbec_depth_mm,
)
from real_so101_vla_rl.hardware.state import SO101StateAdapter


class _Format:
    def __init__(self, name: str) -> None:
        self.name = name


class _ColorFrame:
    def __init__(self, rgb: np.ndarray) -> None:
        self.rgb = rgb

    def get_width(self) -> int:
        return self.rgb.shape[1]

    def get_height(self) -> int:
        return self.rgb.shape[0]

    def get_data(self) -> np.ndarray:
        return self.rgb.reshape(-1)

    def get_format(self) -> _Format:
        return _Format("RGB")


class _DepthFrame:
    def __init__(self, raw: np.ndarray, scale: float = 1.0) -> None:
        self.raw = raw
        self.scale = scale

    def get_width(self) -> int:
        return self.raw.shape[1]

    def get_height(self) -> int:
        return self.raw.shape[0]

    def get_data(self) -> np.ndarray:
        return self.raw

    def get_depth_scale(self) -> float:
        return self.scale


class _FrameSet:
    def __init__(self, color: Any, depth: Any) -> None:
        self.color = color
        self.depth = depth

    def get_color_frame(self) -> Any:
        return self.color

    def get_depth_frame(self) -> Any:
        return self.depth


class _Pipeline:
    def __init__(self, frames: _FrameSet) -> None:
        self.frames = frames
        self.started = False
        self.frame_sync = False

    def enable_frame_sync(self) -> None:
        self.frame_sync = True

    def start(self, config: Any = None) -> None:
        del config
        self.started = True

    def wait_for_frames(self, timeout_ms: int) -> _FrameSet:
        assert timeout_ms == 200
        return self.frames

    def stop(self) -> None:
        self.started = False


class _IdentityAlign:
    def process(self, frames: _FrameSet) -> _FrameSet:
        return frames


def test_orbbec_adapter_outputs_rgb_and_scaled_aligned_depth() -> None:
    rgb = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    rgb[..., 0] = 11
    raw_depth = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 1000, dtype=np.uint16)
    raw_depth[0, 0] = 0
    pipeline = _Pipeline(_FrameSet(_ColorFrame(rgb), _DepthFrame(raw_depth, 0.5)))
    adapter = OrbbecRGBDAdapter(
        pipeline,
        align_filter=_IdentityAlign(),
        clock_ns=lambda: 123_000_000,
    )

    adapter.connect()
    result = adapter.read(200)

    assert pipeline.frame_sync
    assert result.overview.valid and result.depth.valid
    assert result.overview.timestamp_ns == result.depth.timestamp_ns == 123_000_000
    np.testing.assert_array_equal(result.overview.value, rgb)
    assert result.depth.value.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    assert result.depth.value.dtype == np.uint16
    assert result.depth.value[0, 0, 0] == 0
    assert result.depth.value[1, 1, 0] == 500
    adapter.disconnect()


def test_orbbec_depth_rejects_unaligned_shape() -> None:
    raw = np.zeros((400, 640), dtype=np.uint16)
    try:
        decode_orbbec_depth_mm(_DepthFrame(raw))
    except ValueError as exc:
        assert "aligned overview depth" in str(exc)
    else:
        raise AssertionError("unaligned depth shape was accepted")


def test_orbbec_adapter_rejects_missing_stream() -> None:
    rgb = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    pipeline = _Pipeline(_FrameSet(_ColorFrame(rgb), None))
    adapter = OrbbecRGBDAdapter(
        pipeline,
        align_filter=_IdentityAlign(),
        clock_ns=lambda: 123,
    )

    result = adapter.read(200)

    assert not result.overview.valid
    assert not result.depth.valid
    assert result.overview.timestamp_ns == result.depth.timestamp_ns == 123
    assert "missing color or aligned depth" in result.overview.error


class _OpenCVCamera:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.is_connected = False

    def connect(self) -> None:
        self.is_connected = True

    def async_read(self, timeout_ms: int) -> np.ndarray:
        assert timeout_ms == 200
        return self.frame

    def disconnect(self) -> None:
        self.is_connected = False


class _FailingOpenCVCamera(_OpenCVCamera):
    def async_read(self, timeout_ms: int) -> np.ndarray:
        del timeout_ms
        raise TimeoutError("wrist timeout")


class _TimestampedOpenCVCamera(_OpenCVCamera):
    def __init__(self, frame: np.ndarray) -> None:
        super().__init__(frame)
        self.frame_lock = Lock()
        self.new_frame_event = Event()
        self.new_frame_event.set()
        self.latest_frame = frame
        self.latest_timestamp = 10.5


class _FakeVideoCapture:
    def __init__(self) -> None:
        import cv2

        fourcc = sum(ord(character) << (8 * index) for index, character in enumerate("MJPG"))
        self.values = {
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FOURCC: float(fourcc),
            cv2.CAP_PROP_AUTO_EXPOSURE: 3.0,
            cv2.CAP_PROP_EXPOSURE: 0.0,
            cv2.CAP_PROP_GAIN: 1.0,
        }

    def set(self, property_id: int, value: float) -> bool:
        self.values[property_id] = value
        return True

    def get(self, property_id: int) -> float:
        return self.values[property_id]


class _ProfiledOpenCVCamera(_OpenCVCamera):
    def __init__(self, frame: np.ndarray) -> None:
        super().__init__(frame)
        self.videocapture = _FakeVideoCapture()


def test_opencv_adapter_converts_bgr_to_rgb() -> None:
    bgr = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    bgr[0, 0] = (1, 2, 3)
    adapter = OpenCVRGBAdapter(_OpenCVCamera(bgr), clock_ns=lambda: 99)
    adapter.connect()
    sample = adapter.read(200)
    assert sample.valid
    assert sample.timestamp_ns == 99
    np.testing.assert_array_equal(sample.value[0, 0], (3, 2, 1))


def test_opencv_adapter_applies_and_verifies_frozen_manual_exposure() -> None:
    import cv2

    bgr = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    camera = _ProfiledOpenCVCamera(bgr)
    profile = load_camera_rig_profile(
        Path("configs/hardware/cameras/so101_competition_2026.yaml")
    ).profile.wrist
    adapter = OpenCVRGBAdapter(camera, expected_profile=profile)

    adapter.connect()

    assert camera.videocapture.get(cv2.CAP_PROP_AUTO_EXPOSURE) == 1.0
    assert camera.videocapture.get(cv2.CAP_PROP_EXPOSURE) == 300.0
    assert camera.videocapture.get(cv2.CAP_PROP_GAIN) == 0.0


def test_opencv_adapter_uses_background_hardware_completion_timestamp() -> None:
    bgr = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    adapter = OpenCVRGBAdapter(
        _TimestampedOpenCVCamera(bgr),
        clock_ns=lambda: 20_000_000_000,
        perf_counter_ns=lambda: 10_000_000_000,
    )

    sample = adapter.read(200)

    assert sample.valid
    assert sample.timestamp_ns == 20_500_000_000


def test_opencv_adapter_is_broken_after_three_consecutive_failures() -> None:
    frame = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    adapter = OpenCVRGBAdapter(
        _FailingOpenCVCamera(frame),
        clock_ns=lambda: 99,
        max_consecutive_failures=3,
    )

    for expected_sequence in range(3):
        sample = adapter.read(200)
        assert not sample.valid
        assert sample.sequence_id == expected_sequence
        assert "wrist timeout" in sample.error

    assert adapter.is_broken


class _Robot:
    is_connected = True

    def get_observation(self) -> dict[str, float]:
        return {
            "gripper.pos": 6.0,
            "wrist_roll.pos": 5.0,
            "wrist_flex.pos": 4.0,
            "elbow_flex.pos": 3.0,
            "shoulder_lift.pos": 2.0,
            "shoulder_pan.pos": 1.0,
        }


def test_state_adapter_enforces_canonical_joint_order() -> None:
    adapter = SO101StateAdapter(_Robot(), clock_ns=lambda: 77)
    sample = adapter.read(200)
    assert sample.valid
    np.testing.assert_array_equal(sample.value, np.arange(1, 7, dtype=np.float32))
    assert sample.timestamp_ns == 77


@dataclass
class _SingleAdapter:
    sample: Any
    is_broken: bool = False
    is_connected: bool = False

    def connect(self) -> None:
        self.is_connected = True

    def read(self, timeout_ms: int) -> Any:
        assert timeout_ms == 200
        return self.sample

    def disconnect(self) -> None:
        self.is_connected = False


@dataclass
class _SequenceAdapter:
    samples: list[Any]
    is_broken: bool = False
    is_connected: bool = False
    read_count: int = 0

    def connect(self) -> None:
        self.is_connected = True

    def read(self, timeout_ms: int) -> Any:
        assert timeout_ms == 200
        sample = self.samples[min(self.read_count, len(self.samples) - 1)]
        self.read_count += 1
        return sample

    def disconnect(self) -> None:
        self.is_connected = False


def _sync_inputs(*, span_ns: int = 25_000_000, sequence_id: int = 0):
    overview = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    depth = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.uint16)
    wrist = np.ones_like(overview)
    state = np.arange(6, dtype=np.float32)
    rgbd = RGBDFrame(
        CapturedSample.success(overview, timestamp_ns=100, sequence_id=sequence_id),
        CapturedSample.success(depth, timestamp_ns=100, sequence_id=sequence_id),
    )
    return (
        _SingleAdapter(rgbd),
        _SingleAdapter(
            CapturedSample.success(
                wrist, timestamp_ns=100 + span_ns, sequence_id=sequence_id
            )
        ),
        _SingleAdapter(
            CapturedSample.success(state, timestamp_ns=101, sequence_id=sequence_id)
        ),
    )


def test_synchronizer_accepts_boundary_and_emits_schema_fields() -> None:
    synchronizer = SO101ObservationSynchronizer(*_sync_inputs(), tolerance_ms=25)
    synchronizer.connect()
    result = synchronizer.capture()
    assert result.valid
    assert result.diagnostics.span_ns == 25_000_000
    fields = result.observation.to_schema_fields()
    assert [int(fields[key][0]) for key in SENSOR_TIMESTAMP_KEYS] == [
        100,
        25_000_100,
        100,
        101,
    ]
    synchronizer.disconnect()


def test_synchronizer_realigns_camera_phase_with_nearest_buffered_frame() -> None:
    overview = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    depth = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.uint16)
    wrist = np.ones_like(overview)
    state = np.arange(6, dtype=np.float32)

    def rgbd(timestamp_ns: int, sequence_id: int) -> RGBDFrame:
        return RGBDFrame(
            CapturedSample.success(
                overview, timestamp_ns=timestamp_ns, sequence_id=sequence_id
            ),
            CapturedSample.success(
                depth, timestamp_ns=timestamp_ns, sequence_id=sequence_id
            ),
        )

    overview_adapter = _SequenceAdapter(
        [rgbd(0, 0), rgbd(33_000_000, 1)]
    )
    wrist_adapter = _SequenceAdapter(
        [CapturedSample.success(wrist, timestamp_ns=34_000_000, sequence_id=0)]
    )
    state_adapter = _SequenceAdapter(
        [CapturedSample.success(state, timestamp_ns=35_000_000, sequence_id=0)]
    )
    synchronizer = SO101ObservationSynchronizer(
        overview_adapter,
        wrist_adapter,
        state_adapter,
        tolerance_ms=25,
    )

    synchronizer.connect()
    result = synchronizer.capture()

    assert result.valid
    assert result.diagnostics.span_ns == 2_000_000
    assert result.observation.overview_timestamp_ns == 33_000_000
    assert result.observation.wrist_timestamp_ns == 34_000_000
    assert overview_adapter.read_count == 2
    synchronizer.disconnect()


def test_synchronizer_rejects_skew_and_reused_sequence() -> None:
    skewed = SO101ObservationSynchronizer(
        *_sync_inputs(span_ns=25_000_001), tolerance_ms=25
    )
    skewed.connect()
    result = skewed.capture()
    assert not result.valid
    assert "exceeds" in result.error
    skewed.disconnect()

    duplicate = SO101ObservationSynchronizer(*_sync_inputs(), tolerance_ms=25)
    duplicate.connect()
    assert duplicate.capture().valid
    second = duplicate.capture()
    assert not second.valid
    assert "reused" in second.error
    duplicate.disconnect()
