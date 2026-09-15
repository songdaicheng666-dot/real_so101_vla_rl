from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from real_so101_vla_rl.data import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    JOINT_NAMES,
    build_so101_lerobot_features,
)
from real_so101_vla_rl.data.lerobot_dataset import ensure_lerobot_hub_compat
from real_so101_vla_rl.hardware import (
    SyncDiagnostics,
    SynchronizationResult,
    SynchronizedObservation,
)
from real_so101_vla_rl.recording import (
    CaptureAbort,
    SchemaV2EpisodeRecorder,
    collect_episode,
    load_recording_config,
)
from real_so101_vla_rl.recording.lerobot_v2 import RecordingControls


def _observation() -> SynchronizedObservation:
    return SynchronizedObservation(
        overview=np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8),
        wrist=np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8),
        overview_depth=np.full(
            (IMAGE_HEIGHT, IMAGE_WIDTH, 1), 800, dtype=np.uint16
        ),
        state=np.arange(6, dtype=np.float32),
        overview_timestamp_ns=100,
        wrist_timestamp_ns=200,
        overview_depth_timestamp_ns=100,
        state_timestamp_ns=300,
    )


class _Sync:
    is_broken = False

    def __init__(self, result: SynchronizationResult) -> None:
        self.result = result

    def capture(self) -> SynchronizationResult:
        return self.result


class _Teleop:
    def get_action(self) -> dict[str, float]:
        return {name: 90.0 for name in JOINT_NAMES}


class _Robot:
    def __init__(self) -> None:
        self.calls = 0

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.calls += 1
        return {name: min(value, 10.0) for name, value in action.items()}


class _Dataset:
    num_episodes = 0

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.clear_count = 0

    def add_frame(self, frame: dict[str, Any]) -> None:
        self.frames.append(frame)

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        del delete_images
        self.frames.clear()
        self.clear_count += 1

    def has_pending_frames(self) -> bool:
        return bool(self.frames)

    def save_episode(self) -> None:
        self.num_episodes += 1


def _identity(value):
    return value[0]


def test_recorder_saves_robot_returned_action_not_leader_target() -> None:
    dataset = _Dataset()
    robot = _Robot()
    recorder = SchemaV2EpisodeRecorder(
        robot=robot,
        teleop=_Teleop(),
        synchronizer=_Sync(
            SynchronizationResult(
                _observation(), SyncDiagnostics({}, 200, {})
            )
        ),
        dataset=dataset,
        task="Pick up the blue battery and place it in T0.",
        teleop_action_processor=_identity,
        robot_action_processor=_identity,
    )

    sent = recorder.record_tick()

    assert set(sent.values()) == {10.0}
    np.testing.assert_array_equal(
        dataset.frames[0]["action"], np.full(6, 10, dtype=np.float32)
    )


def test_invalid_capture_does_not_send_action_and_clears_episode(tmp_path: Path) -> None:
    invalid = SynchronizationResult(
        None,
        SyncDiagnostics(
            {"observation.timestamps.wrist_ns": 123},
            None,
            {"observation.timestamps.wrist_ns": "timeout"},
        ),
        error="one or more sensors returned an invalid sample",
    )
    dataset = _Dataset()
    dataset.frames.append({"old": "frame"})
    robot = _Robot()
    recorder = SchemaV2EpisodeRecorder(
        robot=robot,
        teleop=_Teleop(),
        synchronizer=_Sync(invalid),
        dataset=dataset,
        task="Pick up the blue battery and place it in T0.",
        teleop_action_processor=_identity,
        robot_action_processor=_identity,
    )

    summary = collect_episode(
        recorder,
        RecordingControls(),
        duration_s=1,
        fps=30,
        failure_log_path=tmp_path / "capture_failures.jsonl",
    )

    assert summary.aborted
    assert dataset.clear_count == 1
    assert dataset.frames == []
    assert robot.calls == 0
    payload = json.loads((tmp_path / "capture_failures.jsonl").read_text())
    assert payload["num_frames_discarded"] == 0
    assert payload["errors"]["observation.timestamps.wrist_ns"] == "timeout"


def test_recorder_raises_capture_abort_before_hardware_action() -> None:
    result = SynchronizationResult(
        None, SyncDiagnostics({}, None, {"state": "failed"}), error="failed"
    )
    recorder = SchemaV2EpisodeRecorder(
        robot=_Robot(),
        teleop=_Teleop(),
        synchronizer=_Sync(result),
        dataset=_Dataset(),
        task="Pick up the blue battery and place it in T0.",
        teleop_action_processor=_identity,
        robot_action_processor=_identity,
    )
    try:
        recorder.record_tick()
    except CaptureAbort:
        pass
    else:
        raise AssertionError("invalid synchronized capture did not abort")


def test_controls_are_phase_aware() -> None:
    controls = RecordingControls()
    controls.enter_phase("recording")
    controls.dispatch("enter")
    assert not controls.snapshot()[1]
    controls.dispatch("right")
    assert controls.snapshot()[0]

    controls.enter_phase("confirm")
    controls.dispatch("enter")
    assert controls.snapshot()[1]

    controls.enter_phase("recording")
    controls.dispatch("left")
    end, _, discard, _ = controls.snapshot()
    assert end and discard


def test_checked_in_config_requires_hardware_only_for_live_run() -> None:
    path = Path("configs/recording/so101_schema_v2.yaml")
    config = load_recording_config(path, require_hardware_ready=False)
    assert config.sync.tolerance_ms == 25
    assert config.dataset.rgb_use_videos is True
    try:
        load_recording_config(path, require_hardware_ready=True)
    except ValueError as exc:
        assert "follower.port" in str(exc)
    else:
        raise AssertionError("live config accepted unresolved hardware")


def test_recording_config_rejects_removed_use_videos_key(tmp_path: Path) -> None:
    source = Path("configs/recording/so101_schema_v2.yaml").read_text()
    old_config = tmp_path / "old_recording.yaml"
    old_config.write_text(
        source.replace("rgb_use_videos", "use_videos"),
        encoding="utf-8",
    )

    try:
        load_recording_config(old_config, require_hardware_ready=False)
    except ValueError as exc:
        assert "rgb_use_videos" in str(exc)
    else:
        raise AssertionError("removed dataset.use_videos key was accepted")


def test_feature_builder_keeps_depth_lossless_for_both_rgb_modes() -> None:
    video_features = build_so101_lerobot_features(rgb_use_videos=True)
    image_features = build_so101_lerobot_features(rgb_use_videos=False)

    assert {
        video_features[key]["dtype"]
        for key in ("observation.images.overview", "observation.images.wrist")
    } == {"video"}
    assert {
        image_features[key]["dtype"]
        for key in ("observation.images.overview", "observation.images.wrist")
    } == {"image"}
    for features in (video_features, image_features):
        depth = features["observation.images.overview_depth"]
        assert depth["dtype"] == "image"
        assert depth["info"] == {"is_depth_map": True, "depth_unit": "mm"}


def test_fake_hardware_tick_round_trips_through_real_lerobot_writer(
    tmp_path: Path, monkeypatch
) -> None:
    ensure_lerobot_hub_compat()
    import datasets
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset

    monkeypatch.setattr(
        datasets.config,
        "HF_DATASETS_CACHE",
        str(tmp_path / "huggingface_datasets_cache"),
    )

    root = tmp_path / "real-recording-fixture"
    dataset = LeRobotDataset.create(
        repo_id="local/schema_v2_real_recording_fixture",
        fps=30,
        root=root,
        robot_type="so101_follower",
        features=build_so101_lerobot_features(rgb_use_videos=False),
        use_videos=False,
        image_writer_threads=1,
        rgb_encoder=RGBEncoderConfig(vcodec="h264"),
    )
    recorder = SchemaV2EpisodeRecorder(
        robot=_Robot(),
        teleop=_Teleop(),
        synchronizer=_Sync(
            SynchronizationResult(
                _observation(), SyncDiagnostics({}, 200, {})
            )
        ),
        dataset=dataset,
        task="Pick up the blue battery and place it in T0.",
        teleop_action_processor=_identity,
        robot_action_processor=_identity,
    )
    recorder.record_tick()
    dataset.save_episode()
    dataset.finalize()

    reopened = LeRobotDataset(
        repo_id="local/schema_v2_real_recording_fixture",
        root=root,
        episodes=[0],
        return_uint8=True,
    )
    assert len(reopened) == 1
    np.testing.assert_array_equal(
        reopened[0]["action"].numpy(), np.full(6, 10, dtype=np.float32)
    )
