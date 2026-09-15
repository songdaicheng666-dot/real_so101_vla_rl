"""LeRobot feature metadata shared by synthetic and real SO-101 recorders."""

from __future__ import annotations

from typing import Any

from .lerobot_dataset import ensure_lerobot_hub_compat
from .schema import (
    ACTION_KEY,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    JOINT_NAMES,
    OVERVIEW_DEPTH_KEY,
    SENSOR_TIMESTAMP_KEYS,
    SENSOR_VALID_KEYS,
)


def build_so101_lerobot_features(
    *, rgb_use_videos: bool
) -> dict[str, dict[str, Any]]:
    """Build schema-v2 features with optional RGB video and lossless depth images."""

    ensure_lerobot_hub_compat()
    from lerobot.utils.feature_utils import hw_to_dataset_features

    motor_features = {name: float for name in JOINT_NAMES}
    rgb_features = {
        "overview": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        "wrist": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
    }
    depth_features = {
        "overview_depth": (IMAGE_HEIGHT, IMAGE_WIDTH, 1),
    }
    features = {
        **hw_to_dataset_features(
            motor_features, ACTION_KEY, use_video=False
        ),
        **hw_to_dataset_features(
            motor_features, "observation", use_video=False
        ),
        **hw_to_dataset_features(
            rgb_features, "observation", use_video=rgb_use_videos
        ),
        **hw_to_dataset_features(
            depth_features, "observation", use_video=False
        ),
    }
    features[OVERVIEW_DEPTH_KEY].setdefault("info", {}).update(
        {"is_depth_map": True, "depth_unit": "mm"}
    )
    features.update(
        {
            key: {"dtype": "int64", "shape": (1,), "names": None}
            for key in SENSOR_TIMESTAMP_KEYS
        }
    )
    features.update(
        {
            key: {"dtype": "bool", "shape": (1,), "names": None}
            for key in SENSOR_VALID_KEYS
        }
    )
    return features
