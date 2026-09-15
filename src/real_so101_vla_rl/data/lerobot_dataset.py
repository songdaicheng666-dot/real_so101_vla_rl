"""Load validated SO-101 splits through LeRobotDataset.

LeRobot is imported lazily so the project's lightweight base installation does
not require the SFT data dependencies.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .action_chunk import action_delta_timestamps
from .episode_manifest import EpisodeRecord, load_episode_manifest
from .normalization import (
    NormalizationStats,
    load_normalization_stats,
    training_episode_digest,
)
from .robot_profile import RobotProfile, calibration_sha256, load_robot_profile
from .schema import (
    ACTION_DIM,
    DEPTH_IMAGE_KEYS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    JOINT_NAMES,
    RGB_IMAGE_KEYS,
    SENSOR_TIMESTAMP_KEYS,
    SENSOR_VALID_KEYS,
)
from .splits import DatasetSplits, load_dataset_splits, validate_dataset_splits

if TYPE_CHECKING:
    from real_so101_vla_rl.models.sft_config import SFTConfig

SplitName = Literal["train", "val", "test"]
DatasetFactory = Callable[..., Any]
SnapshotDownloader = Callable[..., str]
DEFAULT_LEROBOT_REVISION = "v3.0"


def ensure_lerobot_hub_compat() -> None:
    """Let LeRobot import with the Hub version required by OpenVLA-OFT.

    OpenVLA-OFT's tokenizer stack currently requires ``huggingface-hub<1.0``,
    while LeRobot 0.6.2 imports the newer ``sync_bucket`` helper at module
    import time. The project only supports regular LeRobot dataset repositories,
    so provide a guarded placeholder for that unused bucket-only API.
    """

    try:
        import huggingface_hub
    except ImportError:
        return

    if hasattr(huggingface_hub, "sync_bucket"):
        return

    def unsupported_sync_bucket(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError(
            "Hugging Face bucket repositories require huggingface-hub>=1.6; "
            "use a local directory or a regular dataset repository for SFT"
        )

    huggingface_hub.sync_bucket = unsupported_sync_bucket


@dataclass(frozen=True, slots=True)
class ProjectDatasetMetadata:
    """Validated project metadata needed to construct one training split."""

    root: Path
    split: SplitName
    episode_indices: tuple[int, ...]
    episodes: tuple[EpisodeRecord, ...]
    splits: DatasetSplits
    normalization: NormalizationStats
    robot_profile: RobotProfile


@dataclass(frozen=True, slots=True)
class LoadedLeRobotSplit:
    """A selected LeRobotDataset together with its project-side metadata."""

    dataset: Any
    metadata: ProjectDatasetMetadata


def _split_episode_indices(splits: DatasetSplits, split: SplitName) -> tuple[int, ...]:
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")
    return getattr(splits, split)


def _project_metadata_paths(root: Path, directory: str) -> dict[str, Path]:
    metadata_root = root / directory
    return {
        "episodes": metadata_root / "episodes.jsonl",
        "splits": metadata_root / "splits.json",
        "normalization": metadata_root / "norm_stats.json",
        "robot_profile": metadata_root / "robot_profile.json",
    }


def _materialize_project_metadata(
    config: SFTConfig,
    *,
    snapshot_download_fn: SnapshotDownloader | None,
) -> Path:
    dataset_config = config.dataset
    requested_root = (
        Path(dataset_config.root).expanduser() if dataset_config.root else None
    )
    if requested_root is not None:
        paths = _project_metadata_paths(requested_root, dataset_config.project_meta_dir)
        if all(path.is_file() for path in paths.values()):
            return requested_root

    if dataset_config.repo_id is None:
        raise ValueError(
            "dataset.repo_id must be set before loading a LeRobotDataset split"
        )

    if snapshot_download_fn is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "Project metadata is not available locally. Install the 'sft-data' "
                "extra to load it from the Hugging Face Hub."
            ) from exc
        snapshot_download_fn = snapshot_download

    download_kwargs: dict[str, Any] = {
        "repo_id": dataset_config.repo_id,
        "repo_type": "dataset",
        "revision": dataset_config.revision or DEFAULT_LEROBOT_REVISION,
        "allow_patterns": [f"{dataset_config.project_meta_dir}/*"],
    }
    if requested_root is not None:
        download_kwargs["local_dir"] = requested_root

    resolved = Path(snapshot_download_fn(**download_kwargs))
    root = requested_root if requested_root is not None else resolved
    missing = [
        str(path)
        for path in _project_metadata_paths(
            root, dataset_config.project_meta_dir
        ).values()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Dataset project metadata is incomplete; missing={missing}"
        )
    return root


def load_project_dataset_metadata(
    config: SFTConfig,
    *,
    split: SplitName,
    root: str | Path,
) -> ProjectDatasetMetadata:
    """Load and cross-check all project metadata for one split."""

    root = Path(root)
    paths = _project_metadata_paths(root, config.dataset.project_meta_dir)
    episodes = load_episode_manifest(paths["episodes"])
    splits = load_dataset_splits(paths["splits"])
    validate_dataset_splits(splits, episodes)
    normalization = load_normalization_stats(paths["normalization"])
    robot_profile = load_robot_profile(paths["robot_profile"])

    digest, count = training_episode_digest(splits.train)
    if (
        normalization.train_episode_sha256 != digest
        or normalization.train_episode_count != count
    ):
        raise ValueError(
            "norm_stats.json was not computed from the episode indices in splits.train"
        )
    if normalization.unnorm_key != config.dataset.normalization.unnorm_key:
        raise ValueError(
            "Normalization key mismatch: "
            f"metadata={normalization.unnorm_key!r}, "
            f"config={config.dataset.normalization.unnorm_key!r}"
        )
    if robot_profile.control_fps != config.dataset.fps:
        raise ValueError(
            f"Robot profile uses {robot_profile.control_fps} fps, "
            f"but the dataset config requires {config.dataset.fps} fps"
        )

    calibration_path = paths["robot_profile"].parent / robot_profile.calibration_file
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Calibration snapshot is missing: {calibration_path}")
    if (
        calibration_sha256(calibration_path.read_bytes())
        != robot_profile.calibration_sha256
    ):
        raise ValueError("Calibration snapshot does not match robot_profile.json")

    episode_indices = _split_episode_indices(splits, split)
    if not episode_indices:
        raise ValueError(f"The {split!r} split contains no episodes")
    records_by_index = {record.episode_index: record for record in episodes}
    if any(not records_by_index[index].success for index in episode_indices):
        raise ValueError(f"The {split!r} split contains a failed episode")

    return ProjectDatasetMetadata(
        root=root,
        split=split,
        episode_indices=episode_indices,
        episodes=episodes,
        splits=splits,
        normalization=normalization,
        robot_profile=robot_profile,
    )


def _feature_shape(
    features: Mapping[str, Mapping[str, Any]], key: str
) -> tuple[int, ...]:
    try:
        shape = tuple(features[key]["shape"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"LeRobotDataset metadata is missing feature {key!r}") from exc
    return shape


def _validate_lerobot_contract(
    dataset: Any,
    config: SFTConfig,
    episode_indices: Sequence[int],
) -> None:
    meta = getattr(dataset, "meta", None)
    if meta is None:
        raise ValueError("LeRobotDataset instance does not expose metadata")
    if meta.fps != config.dataset.fps:
        raise ValueError(
            f"LeRobotDataset uses {meta.fps} fps, expected {config.dataset.fps}"
        )
    if getattr(meta, "robot_type", None) not in (None, "so101_follower"):
        raise ValueError(
            f"LeRobotDataset robot_type must be 'so101_follower', got {meta.robot_type!r}"
        )

    features = meta.features
    for key in (config.dataset.state_key, config.dataset.action_key):
        feature = features.get(key)
        if not isinstance(feature, Mapping):
            raise TypeError(
                f"LeRobotDataset metadata is missing feature {key!r}"
            )
        shape = _feature_shape(features, key)
        if shape != (ACTION_DIM,):
            raise ValueError(
                f"LeRobotDataset feature {key!r} must have shape ({ACTION_DIM},)"
            )
        if feature.get("dtype") != "float32":
            raise ValueError(
                f"LeRobotDataset feature {key!r} must use float32"
            )
        if tuple(feature.get("names", ())) != JOINT_NAMES:
            raise ValueError(
                f"LeRobotDataset feature {key!r} must use the SO-101 joint order"
            )

    rgb_features = [features.get(key) for key in RGB_IMAGE_KEYS]
    if not all(isinstance(feature, Mapping) for feature in rgb_features):
        missing = [
            key
            for key, feature in zip(RGB_IMAGE_KEYS, rgb_features, strict=True)
            if not isinstance(feature, Mapping)
        ]
        raise TypeError(f"LeRobotDataset is missing RGB features: {missing}")
    rgb_storage_types = {feature.get("dtype") for feature in rgb_features}
    if len(rgb_storage_types) != 1 or not rgb_storage_types <= {"image", "video"}:
        raise ValueError(
            "LeRobotDataset overview and wrist RGB features must use the same "
            "image or video storage type"
        )

    for key in (*RGB_IMAGE_KEYS, *DEPTH_IMAGE_KEYS):
        image_shape = _feature_shape(features, key)
        expected_channels = 3 if key in RGB_IMAGE_KEYS else 1
        expected_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, expected_channels)
        if image_shape != expected_shape:
            raise ValueError(
                f"LeRobotDataset feature {key!r} must have shape {expected_shape}"
            )
        image_feature = features[key]
        if image_feature.get("dtype") not in ("image", "video"):
            raise ValueError(
                f"LeRobotDataset feature {key!r} must use image or video storage"
            )
        if tuple(image_feature.get("names", ())) != (
            "height",
            "width",
            "channels",
        ):
            raise ValueError(
                f"LeRobotDataset feature {key!r} must use HWC names"
            )
        if key in DEPTH_IMAGE_KEYS:
            if image_feature.get("dtype") != "image":
                raise ValueError(
                    f"LeRobotDataset depth feature {key!r} must use lossless "
                    "image/TIFF storage, not video"
                )
            info = image_feature.get("info", {})
            if info.get("is_depth_map") is not True or info.get("depth_unit") != "mm":
                raise ValueError(
                    f"LeRobotDataset depth feature {key!r} must declare "
                    "is_depth_map=true and depth_unit='mm'"
                )

    for key, expected_dtype in (
        *((key, "int64") for key in SENSOR_TIMESTAMP_KEYS),
        *((key, "bool") for key in SENSOR_VALID_KEYS),
    ):
        feature = features.get(key)
        if not isinstance(feature, Mapping):
            raise TypeError(f"LeRobotDataset metadata is missing feature {key!r}")
        if _feature_shape(features, key) != (1,) or feature.get("dtype") != expected_dtype:
            raise ValueError(
                f"LeRobotDataset feature {key!r} must use {expected_dtype}[1]"
            )

    selected = tuple(getattr(dataset, "episodes", ()))
    if selected and selected != tuple(episode_indices):
        raise ValueError(
            "LeRobotDataset did not retain the requested episode selection"
        )
    if len(dataset) <= 0:
        raise ValueError("Selected LeRobotDataset split contains no frames")

    sample = dataset[0]
    depth = sample.get(DEPTH_IMAGE_KEYS[0])
    if depth is None:
        raise ValueError(
            f"LeRobotDataset sample is missing depth field {DEPTH_IMAGE_KEYS[0]!r}"
        )
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    depth = np.asarray(depth)
    expected_depth_shape = (1, IMAGE_HEIGHT, IMAGE_WIDTH)
    if depth.shape != expected_depth_shape:
        raise ValueError(
            f"Decoded LeRobot depth must have shape {expected_depth_shape}, "
            f"got {depth.shape}"
        )
    if depth.dtype not in (np.dtype(np.uint16), np.dtype(np.float32)):
        raise TypeError(
            "Decoded LeRobot depth must use uint16 or float32 millimetres, "
            f"got {depth.dtype}"
        )
    if not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError("Decoded LeRobot depth must contain finite non-negative values")
    if not np.equal(depth, np.rint(depth)).all():
        raise ValueError("Decoded LeRobot depth must contain integer millimetre values")


def create_lerobot_split_dataset(
    config: SFTConfig,
    *,
    split: SplitName,
    dataset_cls: DatasetFactory | None = None,
    snapshot_download_fn: SnapshotDownloader | None = None,
) -> LoadedLeRobotSplit:
    """Create an episode-filtered LeRobotDataset without importing OpenVLA."""

    config.validate_definition()
    root = _materialize_project_metadata(
        config,
        snapshot_download_fn=snapshot_download_fn,
    )
    metadata = load_project_dataset_metadata(config, split=split, root=root)

    if dataset_cls is None:
        ensure_lerobot_hub_compat()
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "LeRobotDataset is unavailable. Install this project with the "
                "'sft-data' extra in the Python 3.12 training environment."
            ) from exc
        dataset_cls = LeRobotDataset

    dataset = dataset_cls(
        repo_id=config.dataset.repo_id,
        root=root,
        revision=config.dataset.revision or DEFAULT_LEROBOT_REVISION,
        episodes=list(metadata.episode_indices),
        delta_timestamps=action_delta_timestamps(
            fps=config.dataset.fps,
            chunk_size=config.model.action_chunk_size,
        ),
        return_uint8=True,
    )
    _validate_lerobot_contract(dataset, config, metadata.episode_indices)
    return LoadedLeRobotSplit(dataset=dataset, metadata=metadata)
