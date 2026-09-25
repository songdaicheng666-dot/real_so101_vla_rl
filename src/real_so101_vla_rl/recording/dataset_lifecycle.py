"""Fail-closed creation, resumption, and provenance checks for real datasets."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from real_so101_vla_rl.data import (
    LEROBOT_COMMIT,
    build_robot_profile,
    build_so101_lerobot_features,
    load_episode_manifest,
    write_robot_profile,
)

from .config import SO101RecordingConfig

FINALIZED_METADATA_FILENAMES = (
    "splits.json",
    "norm_stats.json",
    "finalization.json",
)


def validate_recording_root_mode(config: SO101RecordingConfig, *, resume: bool) -> None:
    """Reject accidental overwrite, invalid resume roots, and finalized datasets."""

    root = Path(config.dataset.root)
    metadata_root = root / "project_meta"
    finalized = [
        str(metadata_root / name)
        for name in FINALIZED_METADATA_FILENAMES
        if (metadata_root / name).exists()
    ]
    if resume:
        if not (root / "meta" / "info.json").is_file():
            raise FileNotFoundError(
                f"Cannot resume because no finalized LeRobot metadata exists at {root}"
            )
        if finalized:
            raise RuntimeError(
                f"Cannot resume a dataset after SFT finalization; found={finalized}"
            )
        manifest_path = metadata_root / "episodes.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume without the project episode manifest: {manifest_path}"
            )
        records = load_episode_manifest(manifest_path)
        if len(records) >= config.dataset.num_episodes:
            raise RuntimeError(
                "The recording plan is already complete; finalize the dataset instead"
            )
        return

    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty dataset root: {root}; "
            "pass --resume only when continuing this exact recording plan"
        )


def _validate_recording_features(
    actual: Mapping[str, Mapping[str, Any]], *, rgb_use_videos: bool
) -> None:
    expected = build_so101_lerobot_features(rgb_use_videos=rgb_use_videos)
    for key, expected_feature in expected.items():
        actual_feature = actual.get(key)
        if not isinstance(actual_feature, Mapping):
            raise TypeError(f"Resumed LeRobotDataset is missing feature {key!r}")
        for field in ("dtype", "shape", "names"):
            expected_value = expected_feature.get(field)
            actual_value = actual_feature.get(field)
            if field in {"shape", "names"} and expected_value is not None:
                expected_value = tuple(expected_value)
                actual_value = tuple(actual_value or ())
            if actual_value != expected_value:
                raise ValueError(
                    f"Resumed feature {key!r}.{field} differs: "
                    f"expected {expected_value!r}, got {actual_value!r}"
                )
        expected_info = expected_feature.get("info", {})
        actual_info = actual_feature.get("info", {})
        for name, expected_value in expected_info.items():
            if actual_info.get(name) != expected_value:
                raise ValueError(
                    f"Resumed feature {key!r}.info.{name} differs: "
                    f"expected {expected_value!r}, got {actual_info.get(name)!r}"
                )


def validate_resumed_recording_dataset(
    config: SO101RecordingConfig, dataset: Any
) -> None:
    """Cross-check LeRobot metadata and the accepted-episode plan prefix."""

    if getattr(dataset, "repo_id", None) != config.dataset.repo_id:
        raise ValueError("Resumed LeRobotDataset repo_id differs from recording config")
    meta = getattr(dataset, "meta", None)
    if meta is None:
        raise ValueError("Resumed LeRobotDataset does not expose metadata")
    if int(meta.fps) != config.dataset.fps:
        raise ValueError("Resumed LeRobotDataset fps differs from recording config")
    if getattr(meta, "robot_type", None) not in (None, "so101_follower"):
        raise ValueError("Resumed LeRobotDataset robot_type is not so101_follower")
    _validate_recording_features(
        meta.features,
        rgb_use_videos=config.dataset.rgb_use_videos,
    )

    lerobot_count = int(meta.total_episodes)
    writer_count = int(dataset.num_episodes)
    if lerobot_count != writer_count:
        raise ValueError(
            "Resumed LeRobot episode counts disagree: "
            f"metadata={lerobot_count}, writer={writer_count}"
        )
    manifest_path = Path(config.dataset.root) / "project_meta" / "episodes.jsonl"
    records = load_episode_manifest(manifest_path)
    if len(records) != lerobot_count:
        raise ValueError(
            "LeRobot episode count and project manifest count differ: "
            f"lerobot={lerobot_count}, manifest={len(records)}"
        )
    for expected_index, record in enumerate(records):
        if record.episode_index != expected_index:
            raise ValueError(
                "Existing episode indices are not a contiguous plan prefix"
            )
        expected_layout = config.dataset.layout_id_for_episode(expected_index)
        expected_trial = f"{config.dataset.trial_id_prefix}-{expected_index:06d}"
        if (
            not record.success
            or record.task != config.dataset.task
            or record.layout_id != expected_layout
            or record.trial_id != expected_trial
        ):
            raise ValueError(
                f"Existing episode {expected_index} does not match the recording plan"
            )


def write_recording_robot_profile(config: SO101RecordingConfig, follower: Any) -> None:
    """Snapshot or validate the exact follower calibration used for recording."""

    calibration_path = Path(follower.calibration_fpath)
    if not calibration_path.is_file():
        raise FileNotFoundError(
            f"Follower calibration snapshot source is missing: {calibration_path}"
        )
    profile, calibration_bytes = build_robot_profile(
        calibration_path,
        robot_id=config.follower.robot_id,
        camera_setup_id=config.cameras.rig.profile.rig_id,
        lerobot_commit=LEROBOT_COMMIT,
        control_fps=config.dataset.fps,
    )
    write_robot_profile(
        Path(config.dataset.root) / "project_meta",
        profile,
        calibration_bytes,
    )
