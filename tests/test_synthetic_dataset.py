from __future__ import annotations

import json

import pytest

datasets = pytest.importorskip(
    "datasets", reason="install the sft-data extra for LeRobot dataset I/O"
)

from real_so101_vla_rl.data.schema import (
    OVERVIEW_DEPTH_KEY,
    OVERVIEW_IMAGE_KEY,
    SENSOR_TIMESTAMP_KEYS,
    SENSOR_VALID_KEYS,
    WRIST_IMAGE_KEY,
)
from real_so101_vla_rl.data.synthetic_dataset import (
    DEFAULT_SYNTHETIC_REPO_ID,
    create_synthetic_so101_dataset,
    synthetic_episode_specs,
    validate_synthetic_lerobot_dataset,
)


def test_synthetic_episode_specs_cover_training_contract() -> None:
    specs = synthetic_episode_specs()

    assert len(specs) == 6
    assert [spec.episode_index for spec in specs] == list(range(6))
    assert {spec.task.target_color.value for spec in specs[:4]} == {
        "red",
        "blue",
        "yellow",
        "green",
    }
    assert {spec.task.target_slot.value for spec in specs[:4]} == {
        "T0",
        "P1",
        "P2",
        "P3",
    }
    for dimension in range(6):
        assert len({spec.joint_target[dimension] for spec in specs[:4]}) == 4


def test_image_backed_synthetic_dataset_uses_real_lerobot_writer(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        datasets.config,
        "HF_DATASETS_CACHE",
        str(tmp_path / "huggingface_datasets_cache"),
    )
    root = tmp_path / "synthetic"
    summary = create_synthetic_so101_dataset(
        root,
        repo_id=DEFAULT_SYNTHETIC_REPO_ID,
        rgb_use_videos=False,
    )

    assert summary.total_frames == 144
    assert summary.rgb_use_videos is False
    assert summary.train_episodes == (0, 1, 2, 3)
    assert set(summary.val_episodes + summary.test_episodes) == {4, 5}
    assert (root / "meta" / "info.json").is_file()
    assert list((root / "data").rglob("*.parquet"))
    assert (root / "meta" / "tasks.parquet").is_file()
    assert list((root / "meta" / "episodes").rglob("*.parquet"))

    generation = json.loads(
        (root / "project_meta" / "synthetic_generation.json").read_text()
    )
    assert generation["action_source"] == (
        "synthetic_emulation_of_robot_send_action_return"
    )
    assert generation["joint_order"][-1] == "gripper.pos"
    assert generation["schema_version"] == 2
    assert generation["image_shapes"] == {
        OVERVIEW_IMAGE_KEY: [480, 640, 3],
        WRIST_IMAGE_KEY: [480, 640, 3],
        OVERVIEW_DEPTH_KEY: [480, 640, 1],
    }

    contract = validate_synthetic_lerobot_dataset(
        root, repo_id=DEFAULT_SYNTHETIC_REPO_ID
    )
    assert contract["dataset_length"] == 144
    assert contract["video_keys"] == []
    assert contract["sample_shapes"]["action"] == [8, 6]
    assert contract["sample_shapes"][OVERVIEW_IMAGE_KEY] == [3, 480, 640]
    assert contract["sample_shapes"][WRIST_IMAGE_KEY] == [3, 480, 640]
    assert contract["sample_shapes"][OVERVIEW_DEPTH_KEY] == [1, 480, 640]
    for key in (*SENSOR_TIMESTAMP_KEYS, *SENSOR_VALID_KEYS):
        assert contract["sample_shapes"][key] == []
    assert contract["tail_action_is_pad"] == [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    ]


def test_generator_refuses_to_replace_existing_data(tmp_path) -> None:
    root = tmp_path / "existing"
    root.mkdir()
    (root / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        create_synthetic_so101_dataset(root, rgb_use_videos=False)


def test_generator_rejects_removed_use_videos_keyword(tmp_path) -> None:
    with pytest.raises(TypeError, match="use_videos"):
        create_synthetic_so101_dataset(tmp_path / "old-api", use_videos=False)
