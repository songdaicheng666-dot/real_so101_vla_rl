from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from real_so101_vla_rl.data.episode_manifest import (
    EpisodeRecord,
    write_episode_manifest,
)
from real_so101_vla_rl.data.lerobot_dataset import create_lerobot_split_dataset
from real_so101_vla_rl.data.normalization import (
    FeatureStats,
    NormalizationStats,
    training_episode_digest,
    write_normalization_stats,
)
from real_so101_vla_rl.data.openvla_oft import (
    IGNORE_INDEX,
    OpenVLABatchCollator,
    SO101OpenVLABatchTransform,
    create_openvla_oft_dataset,
)
from real_so101_vla_rl.data.robot_profile import (
    JointRange,
    RobotProfile,
    calibration_sha256,
    write_robot_profile,
)
from real_so101_vla_rl.data.schema import (
    JOINT_NAMES,
    JOINT_UNITS,
    OVERVIEW_DEPTH_KEY,
    OVERVIEW_IMAGE_KEY,
    SCHEMA_VERSION,
    SENSOR_TIMESTAMP_KEYS,
    SENSOR_VALID_KEYS,
    WRIST_IMAGE_KEY,
    AtomicTask,
    CubeColor,
    TargetSlot,
    TaskType,
)
from real_so101_vla_rl.data.splits import (
    DatasetSplits,
    SplitPolicy,
    write_dataset_splits,
)
from real_so101_vla_rl.models import load_sft_config
from real_so101_vla_rl.models.losses import masked_action_l1

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "sft" / "openvla_oft_lora.yaml"


def _episode_records() -> tuple[EpisodeRecord, ...]:
    tasks = (
        AtomicTask(TaskType.SINGLE_T0, CubeColor.RED, TargetSlot.T0, 0),
        AtomicTask(TaskType.SEQUENCE_STEP, CubeColor.BLUE, TargetSlot.P1, 1),
        AtomicTask(TaskType.SEQUENCE_STEP, CubeColor.YELLOW, TargetSlot.P2, 2),
        AtomicTask(TaskType.SEQUENCE_STEP, CubeColor.GREEN, TargetSlot.P3, 3),
    )
    return tuple(
        EpisodeRecord.from_task(
            episode_index=index,
            trial_id="single" if index == 0 else "sequence",
            atomic_task=task,
            layout_id="layout_0001",
            success=True,
            failure_type=None,
            num_frames=8,
            duration_s=8 / 30,
        )
        for index, task in enumerate(tasks)
    )


def _feature_stats() -> FeatureStats:
    return FeatureStats(
        minimum=(0.0,) * 6,
        maximum=(100.0,) * 6,
        mean=(50.0,) * 6,
        std=(25.0,) * 6,
        q01=(0.0,) * 6,
        q99=(100.0,) * 6,
        mask=(True,) * 6,
    )


def _write_project_metadata(root: Path) -> None:
    metadata_root = root / "project_meta"
    records = _episode_records()
    write_episode_manifest(metadata_root / "episodes.jsonl", records)
    splits = DatasetSplits(
        schema_version=SCHEMA_VERSION,
        seed=42,
        group_by=("layout_id",),
        train=(0, 1, 2, 3),
        val=(),
        test=(),
    )
    write_dataset_splits(metadata_root / "splits.json", splits)
    digest, count = training_episode_digest(splits.train)
    stats = _feature_stats()
    write_normalization_stats(
        metadata_root / "norm_stats.json",
        NormalizationStats(
            schema_version=SCHEMA_VERSION,
            method="bounds_q99",
            unnorm_key="so101_cube_dual_rgb_v2",
            train_episode_sha256=digest,
            train_episode_count=count,
            observation_state=stats,
            action=stats,
        ),
    )

    calibration_bytes = b'{"test":"portable calibration snapshot"}'
    nominal_ranges = {
        name: JointRange(
            minimum=0.0 if name == "gripper.pos" else -180.0,
            maximum=100.0 if name == "gripper.pos" else 180.0,
            unit=unit,
        )
        for name, unit in zip(JOINT_NAMES, JOINT_UNITS, strict=True)
    }
    profile = RobotProfile(
        schema_version=SCHEMA_VERSION,
        robot_type="so101_follower",
        robot_id="test_follower",
        use_degrees=True,
        joint_order=JOINT_NAMES,
        joint_units=JOINT_UNITS,
        nominal_ranges=nominal_ranges,
        calibration_file="calibration.json",
        calibration_sha256=calibration_sha256(calibration_bytes),
        camera_setup_id="dual_rgbd_test_v2",
        control_fps=30,
        lerobot_commit="4aaff99",
    )
    write_robot_profile(metadata_root, profile, calibration_bytes)


def _raw_sample(task: str = "Pick up the red cube and place it in T0.") -> dict:
    actions = torch.stack(
        [torch.linspace(0, 100, 6, dtype=torch.float32) + offset for offset in range(8)]
    )
    return {
        OVERVIEW_IMAGE_KEY: torch.zeros((3, 12, 16), dtype=torch.uint8),
        WRIST_IMAGE_KEY: torch.full((3, 12, 16), 255, dtype=torch.uint8),
        OVERVIEW_DEPTH_KEY: torch.full((1, 12, 16), 1000, dtype=torch.uint16),
        "observation.state": torch.linspace(0, 100, 6),
        "action": actions,
        "action_is_pad": torch.tensor(
            [False, False, False, False, False, False, True, True]
        ),
        "task": task,
        "episode_index": torch.tensor(0),
        "frame_index": torch.tensor(0),
        **{key: torch.tensor([index], dtype=torch.int64) for index, key in enumerate(SENSOR_TIMESTAMP_KEYS)},
        **{key: torch.tensor([True], dtype=torch.bool) for key in SENSOR_VALID_KEYS},
    }


class FakeLeRobotDataset:
    last_kwargs: dict | None = None

    def __init__(self, **kwargs) -> None:
        type(self).last_kwargs = kwargs
        self.episodes = kwargs["episodes"]
        self.meta = SimpleNamespace(
            fps=30,
            robot_type="so101_follower",
            features={
                OVERVIEW_IMAGE_KEY: {
                    "dtype": "video",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channels"],
                },
                WRIST_IMAGE_KEY: {
                    "dtype": "video",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channels"],
                },
                OVERVIEW_DEPTH_KEY: {
                    "dtype": "image",
                    "shape": (480, 640, 1),
                    "names": ["height", "width", "channels"],
                    "info": {"is_depth_map": True, "depth_unit": "mm"},
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": list(JOINT_NAMES),
                },
                "action": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": list(JOINT_NAMES),
                },
                **{
                    key: {"dtype": "int64", "shape": (1,), "names": None}
                    for key in SENSOR_TIMESTAMP_KEYS
                },
                **{
                    key: {"dtype": "bool", "shape": (1,), "names": None}
                    for key in SENSOR_VALID_KEYS
                },
            },
        )
        sample = _raw_sample()
        sample[OVERVIEW_DEPTH_KEY] = torch.full(
            (1, 480, 640), 1000, dtype=torch.float32
        )
        self.items = [sample]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


class FakeDepthVideoDataset(FakeLeRobotDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.meta.features[OVERVIEW_DEPTH_KEY]["dtype"] = "video"


class FakeFractionalDepthDataset(FakeLeRobotDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.items[0][OVERVIEW_DEPTH_KEY][0, 0, 0] = 1000.5


class FakePromptBuilder:
    def __init__(self, family: str) -> None:
        assert family == "openvla"
        self.turns: list[tuple[str, str]] = []

    def add_turn(self, role: str, value: str) -> None:
        self.turns.append((role, value))

    def get_prompt(self) -> str:
        human, assistant = self.turns
        return f"HUMAN:{human[1]}\nASSISTANT:{assistant[1]}\x03"


class FakeBaseTokenizer:
    pad_token_id = 0
    model_max_length = 512

    def __init__(self) -> None:
        self.last_text = ""

    def __call__(self, text: str, *, add_special_tokens: bool) -> SimpleNamespace:
        assert add_special_tokens is True
        self.last_text = text
        return SimpleNamespace(input_ids=[ord(character) for character in text])


def fake_action_tokenizer(actions: np.ndarray):
    def encode(row: np.ndarray) -> str:
        return "".join(chr(0xE000 + index) for index in range(row.size))

    if actions.ndim == 1:
        return encode(actions)
    return [encode(row) for row in actions]


def fake_image_transform(image) -> torch.Tensor:
    assert image.mode == "RGB"
    assert image.size == (16, 12)
    value = image.getpixel((0, 0))[0] / 255
    return torch.full((6, 224, 224), value, dtype=torch.float32)


def _normalization() -> NormalizationStats:
    stats = _feature_stats()
    digest, count = training_episode_digest((0, 1, 2, 3))
    return NormalizationStats(
        schema_version=SCHEMA_VERSION,
        method="bounds_q99",
        unnorm_key="so101_cube_dual_rgb_v2",
        train_episode_sha256=digest,
        train_episode_count=count,
        observation_state=stats,
        action=stats,
    )


def _transform(
    tokenizer: FakeBaseTokenizer | None = None,
) -> SO101OpenVLABatchTransform:
    return SO101OpenVLABatchTransform(
        normalization=_normalization(),
        base_tokenizer=tokenizer or FakeBaseTokenizer(),
        action_tokenizer=fake_action_tokenizer,
        image_transform=fake_image_transform,
        prompt_builder_fn=FakePromptBuilder,
    )


def test_factory_selects_split_and_requests_lerobot_action_window(tmp_path) -> None:
    _write_project_metadata(tmp_path)
    config = load_sft_config(CONFIG_PATH)
    config = replace(
        config,
        dataset=replace(config.dataset, repo_id="local/so101", root=str(tmp_path)),
    )

    loaded = create_lerobot_split_dataset(
        config,
        split="train",
        dataset_cls=FakeLeRobotDataset,
        snapshot_download_fn=lambda **_: pytest.fail(
            "local metadata must not access the Hub"
        ),
    )

    assert loaded.metadata.episode_indices == (0, 1, 2, 3)
    assert loaded.metadata.normalization.unnorm_key == "so101_cube_dual_rgb_v2"
    assert FakeLeRobotDataset.last_kwargs["episodes"] == [0, 1, 2, 3]
    assert FakeLeRobotDataset.last_kwargs["revision"] == "v3.0"
    assert FakeLeRobotDataset.last_kwargs["return_uint8"] is True
    assert FakeLeRobotDataset.last_kwargs["delta_timestamps"] == {
        "action": [index / 30 for index in range(8)]
    }


def test_factory_rejects_normalization_from_a_different_train_split(tmp_path) -> None:
    _write_project_metadata(tmp_path)
    config = load_sft_config(CONFIG_PATH)
    config = replace(
        config,
        dataset=replace(config.dataset, repo_id="local/so101", root=str(tmp_path)),
    )
    write_normalization_stats(
        tmp_path / "project_meta" / "norm_stats.json",
        replace(_normalization(), train_episode_sha256="0" * 64),
    )

    with pytest.raises(ValueError, match="splits.train"):
        create_lerobot_split_dataset(
            config,
            split="train",
            dataset_cls=FakeLeRobotDataset,
        )


def test_factory_rejects_split_policy_mismatch(tmp_path) -> None:
    _write_project_metadata(tmp_path)
    config = load_sft_config(CONFIG_PATH)
    config = replace(
        config,
        dataset=replace(
            config.dataset,
            repo_id="local/so101",
            root=str(tmp_path),
            split_policy=SplitPolicy.PILOT_SINGLE_TASK_GROUPED_V1,
        ),
    )

    with pytest.raises(ValueError, match="split policy mismatch"):
        create_lerobot_split_dataset(
            config,
            split="train",
            dataset_cls=FakeLeRobotDataset,
        )


def test_factory_rejects_depth_video_and_fractional_millimetres(tmp_path) -> None:
    _write_project_metadata(tmp_path)
    config = load_sft_config(CONFIG_PATH)
    config = replace(
        config,
        dataset=replace(config.dataset, repo_id="local/so101", root=str(tmp_path)),
    )

    with pytest.raises(ValueError, match="lossless image/TIFF"):
        create_lerobot_split_dataset(
            config,
            split="train",
            dataset_cls=FakeDepthVideoDataset,
        )
    with pytest.raises(ValueError, match="integer millimetre"):
        create_lerobot_split_dataset(
            config,
            split="train",
            dataset_cls=FakeFractionalDepthDataset,
        )


def test_transform_preserves_language_input_and_masks_only_prompt_labels() -> None:
    tokenizer = FakeBaseTokenizer()
    transformed = _transform(tokenizer)(_raw_sample())

    assert (
        "What action should the robot take to pick up the red cube and place it in t0?"
        in tokenizer.last_text
    )
    assert transformed["input_ids"].dtype == torch.long
    assert transformed["labels"].dtype == torch.long
    assert int((transformed["labels"] != IGNORE_INDEX).sum()) == 49
    assert bool((transformed["labels"] == IGNORE_INDEX).any())
    assert transformed["proprio"].shape == (6,)
    assert transformed["actions"].shape == (8, 6)
    assert transformed["action_is_pad"].shape == (8,)
    assert transformed["pixel_values"].shape == (12, 224, 224)
    assert transformed["pixel_values"][:6].eq(0).all()
    assert transformed["pixel_values"][6:].eq(1).all()
    assert transformed["dataset_name"] == "so101_cube_dual_rgb_v2"
    assert transformed["proprio"].tolist() == pytest.approx(
        [-1, -0.6, -0.2, 0.2, 0.6, 1]
    )


def test_transform_rejects_noncanonical_task_and_bad_padding() -> None:
    transform = _transform()
    bad_task = _raw_sample("把红色方块放进 T0")
    with pytest.raises(ValueError, match="canonical SO-101 instruction"):
        transform(bad_task)

    bad_padding = _raw_sample()
    bad_padding["action_is_pad"][0] = True
    with pytest.raises(ValueError, match="current action"):
        transform(bad_padding)

    invalid_sensor = _raw_sample()
    invalid_sensor[SENSOR_VALID_KEYS[0]] = torch.tensor([False])
    with pytest.raises(ValueError, match="must be true"):
        transform(invalid_sensor)


def test_collator_right_pads_tokens_and_stacks_oft_tensors() -> None:
    transform = _transform()
    first = transform(_raw_sample())
    second = transform(_raw_sample("Pick up the blue cube and place it in P1."))
    collator = OpenVLABatchCollator(model_max_length=512, pad_token_id=0)

    batch = collator([first, second])

    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].dtype == torch.bool
    assert batch["labels"][~batch["attention_mask"]].eq(IGNORE_INDEX).all()
    assert batch["pixel_values"].shape == (2, 12, 224, 224)
    assert batch["proprio"].shape == (2, 6)
    assert batch["actions"].shape == (2, 8, 6)
    assert batch["action_is_pad"].shape == (2, 8)
    assert batch["dataset_names"] == [
        "so101_cube_dual_rgb_v2",
        "so101_cube_dual_rgb_v2",
    ]


def test_complete_factory_exposes_openvla_statistics(tmp_path) -> None:
    _write_project_metadata(tmp_path)
    config = load_sft_config(CONFIG_PATH)
    config = replace(
        config,
        dataset=replace(config.dataset, repo_id="local/so101", root=str(tmp_path)),
    )

    dataset = create_openvla_oft_dataset(
        config,
        split="train",
        base_tokenizer=FakeBaseTokenizer(),
        action_tokenizer=fake_action_tokenizer,
        image_transform=fake_image_transform,
        prompt_builder_fn=FakePromptBuilder,
        dataset_cls=FakeLeRobotDataset,
        snapshot_download_fn=lambda **_: pytest.fail(
            "local metadata must not access the Hub"
        ),
    )

    assert len(dataset) == 1
    assert dataset[0]["actions"].shape == (8, 6)
    assert set(dataset.dataset_statistics["so101_cube_dual_rgb_v2"]) == {
        "observation.state",
        "action",
    }


def test_masked_action_l1_ignores_padded_tail() -> None:
    target = torch.zeros((1, 8, 6))
    predicted = target.clone()
    predicted[:, 6:] = 100
    padding = torch.tensor([[False, False, False, False, False, False, True, True]])

    assert masked_action_l1(predicted, target, padding).item() == 0

    predicted[0, 0, 0] = 6
    assert masked_action_l1(predicted, target, padding).item() == pytest.approx(1 / 6)


def test_masked_action_l1_rejects_padded_current_action() -> None:
    actions = torch.zeros((1, 8, 6))
    padding = torch.tensor([[True, False, False, False, False, False, False, False]])
    with pytest.raises(ValueError, match="valid current action"):
        masked_action_l1(actions, actions, padding)
