"""Generate a tiny SO-101 LeRobotDataset for end-to-end SFT checks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .action_chunk import action_delta_timestamps
from .episode_manifest import EpisodeRecord, write_episode_manifest
from .lerobot_dataset import ensure_lerobot_hub_compat
from .lerobot_features import build_so101_lerobot_features
from .normalization import compute_normalization_stats, write_normalization_stats
from .robot_profile import MOTOR_IDS, build_robot_profile, write_robot_profile
from .schema import (
    ACTION_KEY,
    DEFAULT_FPS,
    DEPTH_IMAGE_KEYS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    JOINT_NAMES,
    OVERVIEW_DEPTH_KEY,
    OVERVIEW_IMAGE_KEY,
    RECORDING_FRAME_KEYS,
    RGB_IMAGE_KEYS,
    SCHEMA_VERSION,
    SENSOR_TIMESTAMP_KEYS,
    SENSOR_VALID_KEYS,
    STATE_KEY,
    WRIST_IMAGE_KEY,
    AtomicTask,
    BatteryColor,
    TargetSlot,
    TaskType,
    validate_frame,
)
from .splits import generate_dataset_splits, write_dataset_splits

DEFAULT_SYNTHETIC_REPO_ID = (
    "local/so101_synthetic_overfit_v2_lossless_depth"
)
DEFAULT_FRAMES_PER_EPISODE = 24
LEROBOT_COMMIT = "4aaff99be4a1d81568c08c8f0296b41b40c99ec4"


@dataclass(frozen=True, slots=True)
class SyntheticEpisodeSpec:
    episode_index: int
    trial_id: str
    layout_id: str
    task: AtomicTask
    joint_target: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SyntheticDatasetSummary:
    root: str
    repo_id: str
    num_episodes: int
    frames_per_episode: int
    total_frames: int
    train_episodes: tuple[int, ...]
    val_episodes: tuple[int, ...]
    test_episodes: tuple[int, ...]
    rgb_use_videos: bool


def synthetic_episode_specs() -> tuple[SyntheticEpisodeSpec, ...]:
    """Return the fixed six-episode overfit curriculum."""

    tasks_and_targets = (
        (
            AtomicTask(TaskType.SINGLE_T0, BatteryColor.RED, TargetSlot.T0, 0),
            (-30.0, -20.0, 40.0, 15.0, -20.0, 20.0),
        ),
        (
            AtomicTask(TaskType.SEQUENCE_STEP, BatteryColor.BLUE, TargetSlot.P1, 1),
            (-10.0, 0.0, 20.0, -5.0, 0.0, 40.0),
        ),
        (
            AtomicTask(TaskType.SEQUENCE_STEP, BatteryColor.YELLOW, TargetSlot.P2, 2),
            (10.0, 20.0, 0.0, -25.0, 20.0, 60.0),
        ),
        (
            AtomicTask(TaskType.SEQUENCE_STEP, BatteryColor.GREEN, TargetSlot.P3, 3),
            (30.0, 40.0, -20.0, -45.0, 40.0, 80.0),
        ),
        (
            AtomicTask(TaskType.SINGLE_T0, BatteryColor.RED, TargetSlot.T0, 0),
            (-30.0, -20.0, 40.0, 15.0, -20.0, 20.0),
        ),
        (
            AtomicTask(TaskType.SEQUENCE_STEP, BatteryColor.BLUE, TargetSlot.P1, 1),
            (-10.0, 0.0, 20.0, -5.0, 0.0, 40.0),
        ),
    )
    trial_ids = (
        "synthetic-train-t0",
        "synthetic-train-sequence",
        "synthetic-train-sequence",
        "synthetic-train-sequence",
        "synthetic-val-t0",
        "synthetic-test-sequence",
    )
    layout_ids = (
        "synthetic-layout-train",
        "synthetic-layout-train",
        "synthetic-layout-train",
        "synthetic-layout-train",
        "synthetic-layout-val",
        "synthetic-layout-test",
    )
    return tuple(
        SyntheticEpisodeSpec(index, trial_ids[index], layout_ids[index], task, target)
        for index, (task, target) in enumerate(tasks_and_targets)
    )


_COLORS = {
    BatteryColor.RED: (220, 45, 45),
    BatteryColor.BLUE: (45, 90, 220),
    BatteryColor.YELLOW: (230, 190, 35),
    BatteryColor.GREEN: (45, 175, 75),
}
_SLOT_POSITIONS = {
    TargetSlot.T0: (445, 80, 575, 155),
    TargetSlot.P1: (420, 220, 500, 300),
    TargetSlot.P2: (505, 220, 585, 300),
    TargetSlot.P3: (462, 315, 542, 395),
}


def render_synthetic_frame(
    spec: SyntheticEpisodeSpec, frame_index: int, total_frames: int
) -> np.ndarray:
    """Draw an RGB scene whose task state is easy to identify and overfit."""

    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (235, 232, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 45, 615, 430), outline=(55, 55, 55), width=4)
    draw.text((40, 18), spec.task.instruction, fill=(20, 20, 20))

    for slot, box in _SLOT_POSITIONS.items():
        draw.rectangle(box, fill=(205, 205, 195), outline=(70, 70, 70), width=3)
        draw.text((box[0] + 8, box[1] + 8), slot.value, fill=(25, 25, 25))

    completed = {
        TargetSlot.P2: ((TargetSlot.P1, BatteryColor.BLUE),),
        TargetSlot.P3: (
            (TargetSlot.P1, BatteryColor.BLUE),
            (TargetSlot.P2, BatteryColor.YELLOW),
        ),
    }.get(spec.task.target_slot, ())
    for slot, color in completed:
        left, top, right, bottom = _SLOT_POSITIONS[slot]
        draw.ellipse(
            (left + 18, top + 25, right - 18, bottom - 8),
            fill=_COLORS[color],
        )

    battery_x = 90 + spec.episode_index * 35
    battery_y = 245
    draw.rounded_rectangle(
        (battery_x, battery_y, battery_x + 105, battery_y + 55),
        radius=10,
        fill=_COLORS[spec.task.target_color],
        outline=(35, 35, 35),
        width=3,
    )
    draw.text(
        (battery_x + 12, battery_y + 18),
        spec.task.target_color.value,
        fill=(15, 15, 15),
    )

    progress = int(540 * (frame_index + 1) / total_frames)
    draw.rectangle((50, 445, 590, 462), outline=(65, 65, 65), width=2)
    draw.rectangle((50, 445, 50 + progress, 462), fill=(85, 85, 85))
    return np.asarray(image, dtype=np.uint8)


def render_synthetic_wrist_frame(
    spec: SyntheticEpisodeSpec, frame_index: int, total_frames: int
) -> np.ndarray:
    """Create a deterministic second RGB view that cannot be confused with overview."""

    overview = render_synthetic_frame(spec, frame_index, total_frames)
    wrist = np.flip(overview, axis=1).copy()
    wrist[:32, :96] = np.asarray((25, 25, 25), dtype=np.uint8)
    wrist[8:24, 8:88] = np.asarray(_COLORS[spec.task.target_color], dtype=np.uint8)
    return wrist


def render_synthetic_depth(
    spec: SyntheticEpisodeSpec, frame_index: int
) -> np.ndarray:
    """Create an aligned millimetre depth image with zero reserved as invalid."""

    x_gradient = np.linspace(900, 1500, IMAGE_WIDTH, dtype=np.uint16)
    depth = np.broadcast_to(x_gradient, (IMAGE_HEIGHT, IMAGE_WIDTH)).copy()
    depth += np.uint16(spec.episode_index * 20 + frame_index)
    depth[:4, :4] = 0
    return depth[..., None]


def _fake_calibration() -> dict[str, dict[str, int]]:
    return {
        motor_name: {
            "id": motor_id,
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": 0,
            "range_max": 4095,
        }
        for motor_name, motor_id in MOTOR_IDS.items()
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_synthetic_lerobot_dataset(
    root: str | Path,
    *,
    repo_id: str = DEFAULT_SYNTHETIC_REPO_ID,
) -> dict[str, Any]:
    """Read the finalized dataset and validate the real SO-101 disk contract."""

    ensure_lerobot_hub_compat()
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=Path(root),
        episodes=list(range(6)),
        delta_timestamps=action_delta_timestamps(
            fps=DEFAULT_FPS, chunk_size=8
        ),
        return_uint8=True,
    )
    rgb_use_videos = all(
        dataset.meta.features[key]["dtype"] == "video"
        for key in RGB_IMAGE_KEYS
    )
    expected_features = build_so101_lerobot_features(
        rgb_use_videos=rgb_use_videos
    )
    for key in (
        *RGB_IMAGE_KEYS,
        *DEPTH_IMAGE_KEYS,
        STATE_KEY,
        ACTION_KEY,
        *SENSOR_TIMESTAMP_KEYS,
        *SENSOR_VALID_KEYS,
    ):
        actual = dataset.meta.features[key]
        expected = expected_features[key]
        for field in ("dtype", "shape", "names"):
            actual_value = (
                tuple(actual[field])
                if field in ("shape", "names") and actual[field] is not None
                else actual[field]
            )
            expected_value = (
                tuple(expected[field])
                if field in ("shape", "names") and expected[field] is not None
                else expected[field]
            )
            if actual_value != expected_value:
                raise ValueError(
                    f"LeRobot feature {key!r} {field} mismatch: "
                    f"actual={actual_value!r}, expected={expected_value!r}"
                )
    if (
        dataset.meta.fps != DEFAULT_FPS
        or dataset.meta.robot_type != "so101_follower"
    ):
        raise ValueError(
            "Synthetic dataset fps or robot_type does not match SO-101 recording"
        )

    sample = dataset[DEFAULT_FRAMES_PER_EPISODE - 2]
    required = {
        *RECORDING_FRAME_KEYS,
        "action_is_pad",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    missing = sorted(required - set(sample))
    if missing:
        raise ValueError(f"Finalized LeRobot sample is missing fields: {missing}")
    if (
        tuple(sample[STATE_KEY].shape) != (6,)
        or tuple(sample[ACTION_KEY].shape) != (8, 6)
    ):
        raise ValueError("Synthetic state/action tensors have an invalid shape")
    if sample["action_is_pad"].dtype is not torch.bool:
        raise TypeError("action_is_pad must use torch.bool")
    for key in SENSOR_TIMESTAMP_KEYS:
        if sample[key].dtype is not torch.int64 or tuple(sample[key].shape) not in ((), (1,)):
            raise TypeError(f"{key} must round-trip as an int64 scalar")
    for key in SENSOR_VALID_KEYS:
        if sample[key].dtype is not torch.bool or not bool(sample[key]):
            raise TypeError(f"{key} must round-trip as a true bool scalar")
    expected_padding = [False, False, True, True, True, True, True, True]
    if sample["action_is_pad"].tolist() != expected_padding:
        raise ValueError(
            "LeRobot did not clamp and mark the episode-tail action window"
        )

    return {
        "dataset_length": len(dataset),
        "fps": dataset.meta.fps,
        "robot_type": dataset.meta.robot_type,
        "video_keys": list(dataset.meta.video_keys),
        "sample_shapes": {
            key: list(value.shape)
            for key, value in sample.items()
            if hasattr(value, "shape")
        },
        "tail_action_is_pad": sample["action_is_pad"].tolist(),
    }


def create_synthetic_so101_dataset(
    root: str | Path,
    *,
    repo_id: str = DEFAULT_SYNTHETIC_REPO_ID,
    frames_per_episode: int = DEFAULT_FRAMES_PER_EPISODE,
    seed: int = 42,
    rgb_use_videos: bool = True,
    validate: bool = True,
) -> SyntheticDatasetSummary:
    """Create and finalize the deterministic six-episode LeRobotDataset."""

    if frames_per_episode != DEFAULT_FRAMES_PER_EPISODE:
        raise ValueError(
            f"The convergence fixture requires "
            f"{DEFAULT_FRAMES_PER_EPISODE} frames per episode"
        )
    if seed != 42:
        raise ValueError("The convergence fixture uses seed=42")
    root = Path(root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"Refusing to replace non-empty dataset root: {root}"
        )
    if root.exists():
        root.rmdir()
    root.parent.mkdir(parents=True, exist_ok=True)

    ensure_lerobot_hub_compat()
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.feature_utils import build_dataset_frame

    features = build_so101_lerobot_features(
        rgb_use_videos=rgb_use_videos
    )
    writer = LeRobotDataset.create(
        repo_id=repo_id,
        fps=DEFAULT_FPS,
        features=features,
        root=root,
        robot_type="so101_follower",
        use_videos=rgb_use_videos,
        image_writer_threads=1,
        rgb_encoder=RGBEncoderConfig(vcodec="h264"),
    )
    specs = synthetic_episode_specs()
    records: list[EpisodeRecord] = []
    for spec in specs:
        state_values = dict(zip(JOINT_NAMES, spec.joint_target, strict=True))
        action_values = dict(state_values)
        for frame_index in range(frames_per_episode):
            overview = render_synthetic_frame(
                spec, frame_index, frames_per_episode
            )
            wrist = render_synthetic_wrist_frame(
                spec, frame_index, frames_per_episode
            )
            observation_values = {
                **state_values,
                "overview": overview,
                "wrist": wrist,
                "overview_depth": render_synthetic_depth(spec, frame_index),
            }
            observation_frame = build_dataset_frame(
                features, observation_values, "observation"
            )
            action_frame = build_dataset_frame(
                features, action_values, ACTION_KEY
            )
            global_frame_index = spec.episode_index * frames_per_episode + frame_index
            host_time_ns = global_frame_index * (1_000_000_000 // DEFAULT_FPS)
            timing_frame = {
                SENSOR_TIMESTAMP_KEYS[0]: np.asarray([host_time_ns], dtype=np.int64),
                SENSOR_TIMESTAMP_KEYS[1]: np.asarray([host_time_ns + 1_000_000], dtype=np.int64),
                SENSOR_TIMESTAMP_KEYS[2]: np.asarray([host_time_ns], dtype=np.int64),
                SENSOR_TIMESTAMP_KEYS[3]: np.asarray([host_time_ns + 2_000_000], dtype=np.int64),
            }
            validity_frame = {
                key: np.asarray([True], dtype=np.bool_)
                for key in SENSOR_VALID_KEYS
            }
            recording_frame = {
                **observation_frame,
                **action_frame,
                "task": spec.task.instruction,
                **timing_frame,
                **validity_frame,
            }
            validate_frame(recording_frame)
            writer.add_frame(
                recording_frame
            )
        writer.save_episode()
        records.append(
            EpisodeRecord.from_task(
                episode_index=spec.episode_index,
                trial_id=spec.trial_id,
                atomic_task=spec.task,
                layout_id=spec.layout_id,
                success=True,
                failure_type=None,
                num_frames=frames_per_episode,
                duration_s=frames_per_episode / DEFAULT_FPS,
            )
        )
    writer.finalize()

    splits = generate_dataset_splits(records, seed=seed)
    project_meta = root / "project_meta"
    write_episode_manifest(project_meta / "episodes.jsonl", records)
    write_dataset_splits(project_meta / "splits.json", splits)

    specs_by_index = {spec.episode_index: spec for spec in specs}
    train_samples = [
        {STATE_KEY: spec.joint_target, ACTION_KEY: spec.joint_target}
        for episode_index in splits.train
        for spec in (specs_by_index[episode_index],)
        for _ in range(frames_per_episode)
    ]
    normalization = compute_normalization_stats(
        train_samples,
        train_episode_indices=splits.train,
        unnorm_key="so101_battery_dual_rgb_v2",
    )
    write_normalization_stats(
        project_meta / "norm_stats.json", normalization
    )

    calibration_path = project_meta / "calibration.json"
    _write_json(calibration_path, _fake_calibration())
    profile, calibration_bytes = build_robot_profile(
        calibration_path,
        robot_id="synthetic_so101_follower",
        camera_setup_id="synthetic_dual_rgbd_640x480_v2",
        lerobot_commit=LEROBOT_COMMIT,
    )
    write_robot_profile(project_meta, profile, calibration_bytes)
    _write_json(
        project_meta / "synthetic_generation.json",
        {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "generator": "real_so101_vla_rl.data.synthetic_dataset",
            "frames_per_episode": frames_per_episode,
            "image_shapes": {
                OVERVIEW_IMAGE_KEY: [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
                WRIST_IMAGE_KEY: [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
                OVERVIEW_DEPTH_KEY: [IMAGE_HEIGHT, IMAGE_WIDTH, 1],
            },
            "action_representation": "absolute_joint_target",
            "action_source": (
                "synthetic_emulation_of_robot_send_action_return"
            ),
            "joint_order": list(JOINT_NAMES),
            "episodes": [
                {
                    "episode_index": spec.episode_index,
                    "task": spec.task.instruction,
                    "layout_id": spec.layout_id,
                    "joint_target": list(spec.joint_target),
                }
                for spec in specs
            ],
        },
    )

    if validate:
        validate_synthetic_lerobot_dataset(root, repo_id=repo_id)
    return SyntheticDatasetSummary(
        root=str(root),
        repo_id=repo_id,
        num_episodes=len(specs),
        frames_per_episode=frames_per_episode,
        total_frames=len(specs) * frames_per_episode,
        train_episodes=splits.train,
        val_episodes=splits.val,
        test_episodes=splits.test,
        rgb_use_videos=rgb_use_videos,
    )


def summary_to_dict(
    summary: SyntheticDatasetSummary,
) -> dict[str, Any]:
    return asdict(summary)
