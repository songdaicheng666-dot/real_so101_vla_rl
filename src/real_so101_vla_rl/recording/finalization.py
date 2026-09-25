"""Validate and finalize one real SO-101 dataset for SFT."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from real_so101_vla_rl.data import (
    ACTION_KEY,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    OVERVIEW_DEPTH_KEY,
    RGB_IMAGE_KEYS,
    SENSOR_TIMESTAMP_KEYS,
    SENSOR_VALID_KEYS,
    STATE_KEY,
    DatasetSplits,
    NormalizationStats,
    compute_normalization_stats,
    generate_dataset_splits,
    load_dataset_splits,
    load_episode_manifest,
    load_normalization_stats,
    load_robot_profile,
    write_dataset_splits,
    write_normalization_stats,
)
from real_so101_vla_rl.data.lerobot_dataset import ensure_lerobot_hub_compat
from real_so101_vla_rl.data.robot_profile import _atomic_write, calibration_sha256

from .config import SO101RecordingConfig
from .dataset_lifecycle import _validate_recording_features

if TYPE_CHECKING:
    from real_so101_vla_rl.models.sft_config import SFTConfig

FINALIZATION_FILENAME = "finalization.json"
DatasetFactory = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class RealDatasetFinalizationSummary:
    root: str
    repo_id: str
    complete: bool
    num_episodes: int
    num_frames: int
    train_episodes: tuple[int, ...] = ()
    val_episodes: tuple[int, ...] = ()
    test_episodes: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar_int(value: Any, *, field_name: str) -> int:
    array = _to_numpy(value)
    if array.size != 1:
        raise ValueError(f"{field_name} must contain one scalar")
    return int(array.reshape(-1)[0])


def _validate_project_provenance(
    config: SO101RecordingConfig, metadata_root: Path
) -> None:
    required = (
        metadata_root / "camera_setup.json",
        metadata_root / "camera_profile.yaml",
        metadata_root / "orbbec_factory_calibration.json",
        metadata_root / "robot_profile.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Recorded dataset provenance is incomplete; missing={missing}"
        )
    camera_setup = json.loads(
        (metadata_root / "camera_setup.json").read_text(encoding="utf-8")
    )
    if camera_setup.get("rig_id") != config.cameras.rig.profile.rig_id:
        raise ValueError("Recorded camera rig differs from recording config")

    robot_profile = load_robot_profile(metadata_root / "robot_profile.json")
    if robot_profile.robot_id != config.follower.robot_id:
        raise ValueError("Recorded follower ID differs from recording config")
    if robot_profile.camera_setup_id != config.cameras.rig.profile.rig_id:
        raise ValueError("Robot profile camera setup differs from recording config")
    if robot_profile.control_fps != config.dataset.fps:
        raise ValueError("Robot profile control rate differs from recording config")
    calibration_path = metadata_root / robot_profile.calibration_file
    if not calibration_path.is_file():
        raise FileNotFoundError(
            f"Recorded follower calibration is missing: {calibration_path}"
        )
    if (
        calibration_sha256(calibration_path.read_bytes())
        != robot_profile.calibration_sha256
    ):
        raise ValueError("Recorded follower calibration hash is invalid")


def _task_index_mapping(meta: Any) -> dict[int, str]:
    tasks = getattr(meta, "tasks", None)
    if tasks is None or not hasattr(tasks, "iterrows"):
        raise ValueError("LeRobot metadata does not expose its task table")
    mapping: dict[int, str] = {}
    for task, row in tasks.iterrows():
        task_index = int(row["task_index"])
        if task_index in mapping:
            raise ValueError(f"Duplicate LeRobot task_index={task_index}")
        mapping[task_index] = str(task)
    return mapping


def _validate_tabular_frames(
    dataset: Any,
    config: SO101RecordingConfig,
    records_by_index: dict[int, Any],
) -> tuple[dict[int, tuple[int, int]], list[dict[str, tuple[float, ...]]]]:
    columns = [
        "episode_index",
        "task_index",
        STATE_KEY,
        ACTION_KEY,
        *SENSOR_TIMESTAMP_KEYS,
        *SENSOR_VALID_KEYS,
    ]
    table = dataset.hf_dataset.select_columns(columns)
    task_by_index = _task_index_mapping(dataset.meta)
    positions: dict[int, list[int]] = defaultdict(list)
    frame_counts: Counter[int] = Counter()
    normalized_inputs: list[dict[str, tuple[float, ...]]] = []
    tolerance_ns = round(config.sync.tolerance_ms * 1_000_000)

    for position, row in enumerate(table):
        episode_index = _scalar_int(row["episode_index"], field_name="episode_index")
        record = records_by_index.get(episode_index)
        if record is None:
            raise ValueError(
                f"LeRobot frame references unknown episode {episode_index}"
            )
        task_index = _scalar_int(row["task_index"], field_name="task_index")
        if task_by_index.get(task_index) != record.task:
            raise ValueError(
                f"LeRobot task differs from episode manifest for episode {episode_index}"
            )

        timestamps = [
            _scalar_int(row[key], field_name=key) for key in SENSOR_TIMESTAMP_KEYS
        ]
        if any(timestamp < 0 for timestamp in timestamps):
            raise ValueError("Sensor timestamps must be non-negative")
        if max(timestamps) - min(timestamps) > tolerance_ns:
            raise ValueError(
                f"Episode {episode_index} contains a frame outside the "
                f"{config.sync.tolerance_ms:g} ms synchronization tolerance"
            )
        for key in SENSOR_VALID_KEYS:
            valid = _to_numpy(row[key])
            if valid.size != 1 or not bool(valid.reshape(-1)[0]):
                raise ValueError(
                    f"Episode {episode_index} contains invalid sensor field {key!r}"
                )

        state = _to_numpy(row[STATE_KEY]).astype(np.float64, copy=False)
        action = _to_numpy(row[ACTION_KEY]).astype(np.float64, copy=False)
        if state.shape != (6,) or action.shape != (6,):
            raise ValueError("Recorded state and action must both have shape (6,)")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError("Recorded state and action must contain finite values")

        positions[episode_index].append(position)
        frame_counts[episode_index] += 1
        normalized_inputs.append(
            {
                "episode_index": (float(episode_index),),
                STATE_KEY: tuple(float(value) for value in state),
                ACTION_KEY: tuple(float(value) for value in action),
            }
        )

    expected_frames = sum(record.num_frames for record in records_by_index.values())
    if len(table) != expected_frames:
        raise ValueError(
            "LeRobot frame count differs from episode manifest: "
            f"lerobot={len(table)}, manifest={expected_frames}"
        )
    for episode_index, record in records_by_index.items():
        if frame_counts[episode_index] != record.num_frames:
            raise ValueError(
                f"Episode {episode_index} frame count differs: "
                f"lerobot={frame_counts[episode_index]}, manifest={record.num_frames}"
            )
    boundaries = {
        episode_index: (episode_positions[0], episode_positions[-1])
        for episode_index, episode_positions in positions.items()
    }
    return boundaries, normalized_inputs


def _validate_decoded_sample(sample: dict[str, Any], *, label: str) -> None:
    for key in RGB_IMAGE_KEYS:
        image = _to_numpy(sample[key])
        if image.shape != (3, IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError(f"{label} decoded {key!r} shape differs: {image.shape}")
        if image.dtype == np.uint8:
            continue
        if image.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError(f"{label} decoded {key!r} must use uint8 or floating point")
        if not np.isfinite(image).all() or np.any(image < 0) or np.any(image > 1):
            raise ValueError(
                f"{label} decoded floating-point {key!r} must be in [0, 1]"
            )
    depth = _to_numpy(sample[OVERVIEW_DEPTH_KEY])
    if depth.shape != (1, IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(f"{label} decoded depth shape differs: {depth.shape}")
    if depth.dtype not in (np.dtype(np.uint16), np.dtype(np.float32)):
        raise TypeError(f"{label} decoded depth must use uint16 or float32")
    if not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError(f"{label} decoded depth must be finite and non-negative")
    if not np.equal(depth, np.rint(depth)).all():
        raise ValueError(f"{label} decoded depth must contain integer millimetres")


def _validate_media_boundaries(
    dataset: Any, boundaries: dict[int, tuple[int, int]]
) -> None:
    for episode_index, (first, last) in boundaries.items():
        _validate_decoded_sample(
            dataset[first], label=f"episode {episode_index} first frame"
        )
        if last != first:
            _validate_decoded_sample(
                dataset[last], label=f"episode {episode_index} last frame"
            )


def _write_or_validate_splits(path: Path, splits: DatasetSplits) -> None:
    if path.exists():
        if load_dataset_splits(path) != splits:
            raise RuntimeError(f"Existing split metadata differs: {path}")
        return
    write_dataset_splits(path, splits)


def _write_or_validate_normalization(
    path: Path, normalization: NormalizationStats
) -> None:
    if path.exists():
        if load_normalization_stats(path) != normalization:
            raise RuntimeError(f"Existing normalization metadata differs: {path}")
        return
    write_normalization_stats(path, normalization)


def _write_or_validate_json(path: Path, payload: dict[str, Any]) -> None:
    expected = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() != expected:
            raise RuntimeError(f"Existing finalization report differs: {path}")
        return
    _atomic_write(path, expected)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finalize_recorded_so101_dataset(
    recording_config: SO101RecordingConfig,
    sft_config: SFTConfig,
    *,
    validate_only: bool = False,
    dataset_factory: DatasetFactory | None = None,
    recording_config_sha256: str | None = None,
    sft_config_sha256: str | None = None,
) -> RealDatasetFinalizationSummary:
    """Validate a real dataset and optionally write deterministic SFT metadata."""

    if sft_config.dataset.repo_id != recording_config.dataset.repo_id:
        raise ValueError("Recording and SFT repo_id values differ")
    if sft_config.dataset.split_policy is not recording_config.dataset.split_policy:
        raise ValueError("Recording and SFT split policies differ")
    if (
        sft_config.dataset.normalization.unnorm_key
        != recording_config.dataset.normalization_key
    ):
        raise ValueError("Recording and SFT normalization keys differ")

    root = Path(recording_config.dataset.root).expanduser().resolve()
    metadata_root = root / "project_meta"
    _validate_project_provenance(recording_config, metadata_root)
    manifest_path = metadata_root / "episodes.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Episode manifest is missing: {manifest_path}")
    records = load_episode_manifest(manifest_path)
    if not records:
        raise ValueError("The recorded dataset contains no accepted episodes")
    if len(records) > recording_config.dataset.num_episodes:
        raise ValueError("Episode manifest exceeds the configured recording plan")
    records_by_index = {record.episode_index: record for record in records}
    if set(records_by_index) != set(range(len(records))):
        raise ValueError("Episode manifest must be a contiguous plan prefix")
    for episode_index, record in records_by_index.items():
        if (
            not record.success
            or record.task != recording_config.dataset.task
            or record.layout_id
            != recording_config.dataset.layout_id_for_episode(episode_index)
            or record.trial_id
            != f"{recording_config.dataset.trial_id_prefix}-{episode_index:06d}"
        ):
            raise ValueError(
                f"Episode {episode_index} does not match the recording plan"
            )

    if dataset_factory is None:
        ensure_lerobot_hub_compat()
        from lerobot.datasets import LeRobotDataset

        dataset_factory = LeRobotDataset
    dataset = dataset_factory(
        repo_id=recording_config.dataset.repo_id,
        root=root,
        return_uint8=True,
        video_backend="pyav",
    )
    meta = dataset.meta
    if int(meta.total_episodes) != len(records):
        raise ValueError(
            "LeRobot episode count differs from project manifest: "
            f"lerobot={meta.total_episodes}, manifest={len(records)}"
        )
    if int(meta.total_frames) != sum(record.num_frames for record in records):
        raise ValueError("LeRobot total frame count differs from project manifest")
    if int(meta.fps) != recording_config.dataset.fps:
        raise ValueError("LeRobot fps differs from recording config")
    if getattr(meta, "robot_type", None) not in (None, "so101_follower"):
        raise ValueError("LeRobot robot_type must be so101_follower")
    _validate_recording_features(
        meta.features,
        rgb_use_videos=recording_config.dataset.rgb_use_videos,
    )

    boundaries, frame_rows = _validate_tabular_frames(
        dataset,
        recording_config,
        records_by_index,
    )
    _validate_media_boundaries(dataset, boundaries)

    if validate_only:
        return RealDatasetFinalizationSummary(
            root=str(root),
            repo_id=recording_config.dataset.repo_id,
            complete=len(records) == recording_config.dataset.num_episodes,
            num_episodes=len(records),
            num_frames=int(meta.total_frames),
        )

    if len(records) != recording_config.dataset.num_episodes:
        raise ValueError(
            "Dataset finalization requires all configured episodes: "
            f"recorded={len(records)}, expected={recording_config.dataset.num_episodes}"
        )
    splits = generate_dataset_splits(
        records,
        seed=sft_config.training.seed,
        val_ratio=0.1,
        test_ratio=0.1,
        policy=recording_config.dataset.split_policy,
    )
    train_indices = set(splits.train)
    train_samples = [
        {STATE_KEY: row[STATE_KEY], ACTION_KEY: row[ACTION_KEY]}
        for row in frame_rows
        if int(row["episode_index"][0]) in train_indices
    ]
    normalization = compute_normalization_stats(
        train_samples,
        train_episode_indices=splits.train,
        unnorm_key=recording_config.dataset.normalization_key,
    )
    _write_or_validate_splits(metadata_root / "splits.json", splits)
    _write_or_validate_normalization(metadata_root / "norm_stats.json", normalization)
    report = {
        "schema_version": 1,
        "status": "complete",
        "repo_id": recording_config.dataset.repo_id,
        "split_policy": recording_config.dataset.split_policy.value,
        "seed": sft_config.training.seed,
        "num_episodes": len(records),
        "num_frames": int(meta.total_frames),
        "train_episodes": list(splits.train),
        "val_episodes": list(splits.val),
        "test_episodes": list(splits.test),
        "recording_config_sha256": recording_config_sha256,
        "sft_config_sha256": sft_config_sha256,
    }
    _write_or_validate_json(metadata_root / FINALIZATION_FILENAME, report)
    return RealDatasetFinalizationSummary(
        root=str(root),
        repo_id=recording_config.dataset.repo_id,
        complete=True,
        num_episodes=len(records),
        num_frames=int(meta.total_frames),
        train_episodes=splits.train,
        val_episodes=splits.val,
        test_episodes=splits.test,
    )
