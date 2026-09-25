from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from real_so101_vla_rl.data import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    JOINT_NAMES,
    AtomicTask,
    EpisodeRecord,
    FailureType,
    SplitPolicy,
    build_so101_lerobot_features,
    load_episode_manifest,
    write_episode_manifest,
)
from real_so101_vla_rl.data.lerobot_dataset import ensure_lerobot_hub_compat
from real_so101_vla_rl.hardware import (
    SyncDiagnostics,
    SynchronizationResult,
    SynchronizedObservation,
)
from real_so101_vla_rl.recording import (
    CaptureAbort,
    EpisodeCaptureSummary,
    SchemaV2EpisodeRecorder,
    collect_episode,
    load_recording_config,
    run_recording_session,
    validate_recording_root_mode,
    validate_resumed_recording_dataset,
)
from real_so101_vla_rl.recording.lerobot_v2 import (
    RecordingControls,
    _wait_for_confirmation,
    _wait_for_layout_setup,
)


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
        task="Pick up the blue cube and place it in T0.",
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
        task="Pick up the blue cube and place it in T0.",
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
        task="Pick up the blue cube and place it in T0.",
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

    controls.enter_phase("layout_setup")
    controls.dispatch("enter")
    assert controls.snapshot()[1]


def test_operator_prompts_are_printed_without_logging(
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout_controls = RecordingControls()
    assert _wait_for_layout_setup(
        layout_controls,
        episode_index=0,
        num_episodes=20,
        task="Pick up the blue cube and place it in T0.",
        layout_id="pilot-lowlight-blue-t0-pose-001",
        sleep=lambda _: layout_controls.dispatch("enter"),
    )
    layout_output = capsys.readouterr().out
    assert "PREPARE episode 1/20" in layout_output
    assert "Press Enter/s to START RECORDING" in layout_output

    confirmation_controls = RecordingControls()
    assert _wait_for_confirmation(
        confirmation_controls,
        sleep=lambda _: confirmation_controls.dispatch("enter"),
    )
    confirmation_output = capsys.readouterr().out
    assert "EPISODE COMPLETE" in confirmation_output
    assert "Enter/s=SAVE" in confirmation_output


def test_checked_in_config_is_hardware_ready() -> None:
    path = Path("configs/recording/so101_schema_v2.yaml")
    config = load_recording_config(path, require_hardware_ready=False)
    assert config.sync.tolerance_ms == 25
    assert config.dataset.rgb_use_videos is True
    assert config.follower.robot_id == "my_follower_arm"
    assert config.leader.robot_id == "my_leader_arm"
    assert config.follower.port == (
        "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B79050068-if00"
    )
    assert config.leader.port == (
        "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B79049969-if00"
    )
    assert config.cameras.rig.profile.overview.usb.serial == "AY6M8630105"
    assert config.cameras.rig.profile.wrist.usb.serial == "20240307110322"
    live_config = load_recording_config(path, require_hardware_ready=True)
    assert live_config == config


def test_lowlight_pilot_config_has_twenty_unique_layouts() -> None:
    config = load_recording_config(
        "configs/recording/so101_real_pilot_lowlight_v2.yaml",
        require_hardware_ready=True,
    )

    assert config.dataset.num_episodes == 20
    assert config.dataset.layout_id is None
    assert config.dataset.requires_layout_setup
    assert len(config.dataset.layout_ids) == 20
    assert len(set(config.dataset.layout_ids)) == 20
    assert config.dataset.layout_id_for_episode(0).endswith("001")
    assert config.dataset.layout_id_for_episode(19).endswith("020")
    assert (
        config.dataset.split_policy
        is SplitPolicy.PILOT_SINGLE_TASK_GROUPED_V1
    )
    assert config.dataset.normalization_key == "so101_cube_dual_rgb_v2"
    with pytest.raises(ValueError, match="exactly one of layout_id or layout_ids"):
        replace(config.dataset, layout_id="also-fixed")
    with pytest.raises(ValueError, match="exactly num_episodes"):
        replace(config.dataset, layout_ids=config.dataset.layout_ids[:-1])


def test_pilot_discard_retries_layout_and_save_advances(
    tmp_path: Path, monkeypatch
) -> None:
    config = load_recording_config(
        "configs/recording/so101_real_pilot_lowlight_v2.yaml",
        require_hardware_ready=True,
    )
    config = replace(
        config,
        dataset=replace(
            config.dataset,
            root=str(tmp_path),
            num_episodes=2,
            layout_ids=config.dataset.layout_ids[:2],
            reset_time_s=0,
        ),
    )
    attempted_layouts: list[str] = []
    outcomes = iter(
        (
            EpisodeCaptureSummary(False, False, 10, 1.0, "operator discarded"),
            EpisodeCaptureSummary(False, False, 11, 1.1),
            EpisodeCaptureSummary(False, False, 12, 1.2),
        )
    )
    monkeypatch.setattr(
        "real_so101_vla_rl.recording.lerobot_v2._wait_for_layout_setup",
        lambda *args, **kwargs: attempted_layouts.append(kwargs["layout_id"])
        or True,
    )
    monkeypatch.setattr(
        "real_so101_vla_rl.recording.lerobot_v2.collect_episode",
        lambda *args, **kwargs: next(outcomes),
    )
    monkeypatch.setattr(
        "real_so101_vla_rl.recording.lerobot_v2._wait_for_confirmation",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "real_so101_vla_rl.recording.lerobot_v2._run_manual_reset",
        lambda *args, **kwargs: None,
    )

    dataset = _Dataset()
    run_recording_session(
        config,
        robot=_Robot(),
        teleop=_Teleop(),
        synchronizer=_Sync(
            SynchronizationResult(_observation(), SyncDiagnostics({}, 200, {}))
        ),
        state_adapter=object(),
        dataset=dataset,
        controls=RecordingControls(),
        teleop_action_processor=_identity,
        robot_action_processor=_identity,
    )

    assert attempted_layouts == [
        config.dataset.layout_ids[0],
        config.dataset.layout_ids[0],
        config.dataset.layout_ids[1],
    ]
    records = load_episode_manifest(tmp_path / "project_meta" / "episodes.jsonl")
    assert [record.layout_id for record in records] == list(
        config.dataset.layout_ids
    )


def test_resume_validation_accepts_only_matching_plan_prefix(tmp_path: Path) -> None:
    config = load_recording_config(
        "configs/recording/so101_real_pilot_lowlight_v2.yaml",
        require_hardware_ready=True,
    )
    config = replace(config, dataset=replace(config.dataset, root=str(tmp_path)))
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text("{}", encoding="utf-8")
    records = []
    for index in range(2):
        records.append(
            EpisodeRecord.from_task(
                episode_index=index,
                trial_id=f"{config.dataset.trial_id_prefix}-{index:06d}",
                atomic_task=AtomicTask.from_instruction(config.dataset.task),
                layout_id=config.dataset.layout_id_for_episode(index),
                success=True,
                failure_type=None,
                num_frames=10,
                duration_s=1.0,
            )
        )
    write_episode_manifest(tmp_path / "project_meta" / "episodes.jsonl", records)
    validate_recording_root_mode(config, resume=True)

    features = build_so101_lerobot_features(rgb_use_videos=True)
    dataset = SimpleNamespace(
        repo_id=config.dataset.repo_id,
        meta=SimpleNamespace(
            fps=30,
            robot_type="so101_follower",
            features=features,
            total_episodes=2,
        ),
        num_episodes=2,
    )
    validate_resumed_recording_dataset(config, dataset)

    records[1] = replace(records[1], layout_id="wrong-layout")
    write_episode_manifest(tmp_path / "project_meta" / "episodes.jsonl", records)
    with pytest.raises(ValueError, match="does not match the recording plan"):
        validate_resumed_recording_dataset(config, dataset)


def test_static_validation_config_is_diagnostic_only() -> None:
    config = load_recording_config(
        "configs/recording/so101_static_validation_v2.yaml",
        require_hardware_ready=True,
    )
    assert config.dataset.repo_id == "local/so101_static_validation_v2"
    assert config.dataset.num_episodes == 10
    assert config.dataset.episode_time_s == 8
    assert config.dataset.reset_time_s == 0
    assert config.dataset.purpose == FailureType.STATIC_VALIDATION.value
    assert config.dataset.manual_confirmation is False


def test_static_validation_auto_saves_but_is_not_a_successful_demo(
    tmp_path: Path, monkeypatch
) -> None:
    config = load_recording_config(
        "configs/recording/so101_static_validation_v2.yaml",
        require_hardware_ready=True,
    )
    config = replace(
        config,
        dataset=replace(config.dataset, root=str(tmp_path), num_episodes=1),
    )
    dataset = _Dataset()
    monkeypatch.setattr(
        "real_so101_vla_rl.recording.lerobot_v2.collect_episode",
        lambda *args, **kwargs: EpisodeCaptureSummary(False, False, 240, 8.0),
    )
    monkeypatch.setattr(
        "real_so101_vla_rl.recording.lerobot_v2._wait_for_confirmation",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("static validation unexpectedly requested confirmation")
        ),
    )
    monkeypatch.setattr(
        "real_so101_vla_rl.recording.lerobot_v2._run_manual_reset",
        lambda *args, **kwargs: None,
    )

    summaries = run_recording_session(
        config,
        robot=_Robot(),
        teleop=_Teleop(),
        synchronizer=_Sync(
            SynchronizationResult(_observation(), SyncDiagnostics({}, 200, {}))
        ),
        state_adapter=object(),
        dataset=dataset,
        controls=RecordingControls(),
        teleop_action_processor=_identity,
        robot_action_processor=_identity,
    )

    assert summaries == (EpisodeCaptureSummary(True, False, 240, 8.0),)
    records = load_episode_manifest(tmp_path / "project_meta" / "episodes.jsonl")
    assert len(records) == 1
    record: EpisodeRecord = records[0]
    assert record.success is False
    assert record.failure_type is FailureType.STATIC_VALIDATION


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


def test_recording_config_rejects_removed_camera_selectors(tmp_path: Path) -> None:
    source = Path("configs/recording/so101_schema_v2.yaml").read_text()
    old_config = tmp_path / "old_camera_config.yaml"
    profile = Path("configs/hardware/cameras/so101_competition_2026.yaml").resolve()
    old_config.write_text(
        source.replace(
            "profile: ../hardware/cameras/so101_competition_2026.yaml",
            f"profile: {profile}\n  orbbec_serial: AY6M8630105",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="were removed"):
        load_recording_config(old_config, require_hardware_ready=False)


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
        task="Pick up the blue cube and place it in T0.",
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
