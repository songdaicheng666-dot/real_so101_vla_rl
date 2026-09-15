"""Optional lifecycle test against the real LeRobot dataset package."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

datasets = pytest.importorskip(
    "datasets", reason="install the sft-data extra for LeRobot dataset I/O"
)

from real_so101_vla_rl.data import build_so101_lerobot_features
from real_so101_vla_rl.data.lerobot_dataset import ensure_lerobot_hub_compat

ensure_lerobot_hub_compat()

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from real_so101_vla_rl.data.action_chunk import action_delta_timestamps
from real_so101_vla_rl.data.schema import (
    OVERVIEW_DEPTH_KEY,
    OVERVIEW_IMAGE_KEY,
    SENSOR_TIMESTAMP_KEYS,
    SENSOR_VALID_KEYS,
    WRIST_IMAGE_KEY,
)


def test_real_lerobot_dataset_returns_episode_safe_action_window(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        datasets.config,
        "HF_DATASETS_CACHE",
        str(tmp_path / "huggingface_datasets_cache"),
    )
    root = tmp_path / "tiny_lerobot_dataset"
    features = build_so101_lerobot_features(rgb_use_videos=True)
    assert features[OVERVIEW_IMAGE_KEY]["dtype"] == "video"
    assert features[WRIST_IMAGE_KEY]["dtype"] == "video"
    assert features[OVERVIEW_DEPTH_KEY]["dtype"] == "image"
    writer = LeRobotDataset.create(
        repo_id="local/tiny-so101",
        fps=30,
        features=features,
        root=root,
        robot_type="so101_follower",
        use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264"),
    )
    for frame_index in range(3):
        writer.add_frame(
            {
                OVERVIEW_IMAGE_KEY: np.full(
                    (480, 640, 3), frame_index, dtype=np.uint8
                ),
                WRIST_IMAGE_KEY: np.full(
                    (480, 640, 3), frame_index + 10, dtype=np.uint8
                ),
                OVERVIEW_DEPTH_KEY: np.full(
                    (480, 640, 1), 1000 + frame_index, dtype=np.uint16
                ),
                "observation.state": np.full(6, frame_index, dtype=np.float32),
                "action": np.full(6, frame_index, dtype=np.float32),
                "task": "Pick up the red battery and place it in T0.",
                **{
                    key: np.asarray([frame_index * 1_000_000 + index], dtype=np.int64)
                    for index, key in enumerate(SENSOR_TIMESTAMP_KEYS)
                },
                **{
                    key: np.asarray([True], dtype=np.bool_)
                    for key in SENSOR_VALID_KEYS
                },
            }
        )
    writer.save_episode()
    writer.finalize()

    parquet_path = next((root / "data").rglob("*.parquet"))
    table = pq.read_table(parquet_path)
    encoded_depth = table.column(OVERVIEW_DEPTH_KEY)[1].as_py()
    assert encoded_depth["path"].endswith("frame-000001.tiff")
    persisted_depth = np.asarray(Image.open(BytesIO(encoded_depth["bytes"])))
    assert persisted_depth.dtype == np.uint16
    np.testing.assert_array_equal(
        persisted_depth,
        np.full((480, 640), 1001, dtype=np.uint16),
    )

    dataset = LeRobotDataset(
        repo_id="local/tiny-so101",
        root=root,
        episodes=[0],
        delta_timestamps=action_delta_timestamps(fps=30, chunk_size=8),
        return_uint8=True,
    )
    sample = dataset[1]

    assert sample["action"].shape == (8, 6)
    assert sample["action_is_pad"].dtype == torch.bool
    assert sample["action_is_pad"].tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    assert sample["action"][-1].tolist() == pytest.approx([2.0] * 6)
    assert sample["task"] == "Pick up the red battery and place it in T0."
    assert set(dataset.meta.video_keys) == {
        OVERVIEW_IMAGE_KEY,
        WRIST_IMAGE_KEY,
    }
    assert dataset.meta.image_keys == [OVERVIEW_DEPTH_KEY]
    assert sample[OVERVIEW_IMAGE_KEY].dtype == torch.uint8
    assert sample[WRIST_IMAGE_KEY].dtype == torch.uint8
    # LeRobot exposes lossless TIFF depth as float32, without changing mm values.
    assert sample[OVERVIEW_DEPTH_KEY].dtype == torch.float32
    assert tuple(sample[OVERVIEW_DEPTH_KEY].shape) == (1, 480, 640)
    assert sample[OVERVIEW_DEPTH_KEY].unique().tolist() == [1001.0]
    for key in SENSOR_TIMESTAMP_KEYS:
        assert sample[key].dtype == torch.int64
        assert tuple(sample[key].shape) == ()
    for key in SENSOR_VALID_KEYS:
        assert sample[key].dtype == torch.bool
        assert bool(sample[key])
    assert sample["timestamp"].item() == pytest.approx(1 / 30)
