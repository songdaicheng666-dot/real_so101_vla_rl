"""Typed configuration for SO-101 OpenVLA-OFT LoRA fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from real_so101_vla_rl.data.schema import (
    ACTION_CHUNK_SIZE,
    ACTION_DIM,
    ACTION_KEY,
    ACTION_SOURCE,
    DEFAULT_FPS,
    RGB_IMAGE_KEYS,
    SCHEMA_VERSION,
    STATE_KEY,
    TASK_KEY,
)
from real_so101_vla_rl.data.splits import SplitPolicy


def _require_type(value: Any, expected_type: type, name: str) -> None:
    if type(value) is not expected_type:
        raise TypeError(f"{name} must be {expected_type.__name__}, got {type(value).__name__}")


def _positive_optional(value: int | None, name: str) -> None:
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError(f"{name} must be null or a positive integer")


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    method: str
    lower_quantile: float
    upper_quantile: float
    clip: bool
    stats_split: str
    unnorm_key: str

    def validate(self) -> None:
        if self.method != "bounds_q99" or self.lower_quantile != 0.01 or self.upper_quantile != 0.99:
            raise ValueError("OpenVLA-OFT v1 requires BOUNDS_Q99 with quantiles 0.01 and 0.99")
        if self.clip is not True or self.stats_split != "train":
            raise ValueError("Normalization must clip values and compute statistics from train only")
        if not isinstance(self.unnorm_key, str) or not self.unnorm_key.strip():
            raise ValueError("normalization.unnorm_key must not be empty")


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    repo_id: str | None
    root: str | None
    revision: str | None
    project_meta_dir: str
    fps: int
    image_keys: tuple[str, ...]
    state_key: str
    action_key: str
    action_source: str
    task_key: str
    task_language: str
    task_format: str
    successful_episodes_only: bool
    split_policy: SplitPolicy
    normalization: NormalizationConfig

    def validate(self) -> None:
        if self.fps != DEFAULT_FPS:
            raise ValueError(f"The initial data contract requires fps={DEFAULT_FPS}")
        if self.image_keys != RGB_IMAGE_KEYS:
            raise ValueError(
                f"The schema-v2 data contract requires image_keys={list(RGB_IMAGE_KEYS)!r}"
            )
        if (self.state_key, self.action_key, self.task_key) != (STATE_KEY, ACTION_KEY, TASK_KEY):
            raise ValueError("Dataset feature keys do not match SO101DataSpec")
        if self.action_source != ACTION_SOURCE:
            raise ValueError(
                "dataset.action_source must require the target returned by robot.send_action()"
            )
        if (self.task_language, self.task_format) != ("en", "canonical_atomic_transfer_v1"):
            raise ValueError("The v1 dataset requires canonical English atomic-transfer instructions")
        if self.project_meta_dir != "project_meta":
            raise ValueError("project_meta_dir must be 'project_meta'")
        if self.successful_episodes_only is not True:
            raise ValueError("SFT must filter to successful episodes")
        SplitPolicy(self.split_policy)
        for value, name in (
            (self.repo_id, "repo_id"),
            (self.root, "root"),
            (self.revision, "revision"),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"dataset.{name} must be null or a non-empty string")
        self.normalization.validate()


@dataclass(frozen=True, slots=True)
class ModelConfig:
    checkpoint: str
    adapter: str
    action_head: str
    loss: str
    action_dim: int
    proprio_dim: int
    action_chunk_size: int
    action_representation: str
    use_proprio: bool
    num_images: int
    image_size: int

    def validate(self) -> None:
        if not self.checkpoint.strip():
            raise ValueError("model.checkpoint must not be empty")
        expected = {
            "adapter": "openvla_oft",
            "action_head": "continuous",
            "loss": "l1",
            "action_dim": ACTION_DIM,
            "proprio_dim": ACTION_DIM,
            "action_chunk_size": ACTION_CHUNK_SIZE,
            "action_representation": "absolute_joint_target",
            "use_proprio": True,
            "num_images": len(RGB_IMAGE_KEYS),
            "image_size": 224,
        }
        for field_name, expected_value in expected.items():
            actual_value = getattr(self, field_name)
            if actual_value != expected_value:
                raise ValueError(
                    f"model.{field_name} must be {expected_value!r}, got {actual_value!r}"
                )


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    enabled: bool
    rank: int
    alpha: int
    dropout: float
    target_modules: str

    def validate(self) -> None:
        expected = (True, 32, 16, 0.0, "all-linear")
        actual = (self.enabled, self.rank, self.alpha, self.dropout, self.target_modules)
        if actual != expected:
            raise ValueError(f"LoRA v1 settings must be {expected}, got {actual}")


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    weight_decay: float
    scheduler: str
    warmup_ratio: float

    def validate(self) -> None:
        if self.name != "adamw" or self.scheduler != "cosine":
            raise ValueError("The initial optimizer must be AdamW with a cosine scheduler")
        if self.learning_rate != 5e-4:
            raise ValueError("The initial OpenVLA-OFT learning rate must be 5e-4")
        if self.weight_decay < 0 or not 0 <= self.warmup_ratio < 1:
            raise ValueError("Invalid optimizer weight_decay or warmup_ratio")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int
    max_steps: int | None
    per_device_train_batch_size: int | None
    gradient_accumulation_steps: int | None
    num_workers: int | None
    logging_steps: int
    eval_steps: int | None
    save_steps: int | None
    save_total_limit: int
    bf16: bool
    gradient_checkpointing: bool
    image_augmentation: bool
    output_dir: str | None

    def validate(self) -> None:
        if self.seed != 42:
            raise ValueError("The initial experiment seed must be 42")
        for field_name in (
            "max_steps",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "eval_steps",
            "save_steps",
        ):
            _positive_optional(getattr(self, field_name), f"training.{field_name}")
        if self.num_workers is not None and (type(self.num_workers) is not int or self.num_workers < 0):
            raise ValueError("training.num_workers must be null or a non-negative integer")
        if type(self.logging_steps) is not int or self.logging_steps <= 0:
            raise ValueError("training.logging_steps must be positive")
        if type(self.save_total_limit) is not int or self.save_total_limit <= 0:
            raise ValueError("training.save_total_limit must be positive")
        if (self.bf16, self.gradient_checkpointing) != (True, True):
            raise ValueError("bf16 and gradient_checkpointing must be enabled")
        if type(self.image_augmentation) is not bool:
            raise TypeError("training.image_augmentation must be a boolean")
        if self.output_dir is not None and not self.output_dir.strip():
            raise ValueError("training.output_dir must be null or a non-empty string")


@dataclass(frozen=True, slots=True)
class SFTConfig:
    schema_version: int
    experiment_name: str
    dataset: DatasetConfig
    model: ModelConfig
    lora: LoRAConfig
    optimizer: OptimizerConfig
    training: TrainingConfig

    def validate_definition(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported SFT config schema_version={self.schema_version}")
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must not be empty")
        self.dataset.validate()
        self.model.validate()
        self.lora.validate()
        self.optimizer.validate()
        self.training.validate()

    def validate_training_ready(self) -> None:
        """Reject values intentionally left open until data and cloud hardware exist."""

        self.validate_definition()
        missing = []
        if self.dataset.repo_id is None:
            missing.append("dataset.repo_id")
        for field_name in (
            "max_steps",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "num_workers",
            "eval_steps",
            "save_steps",
            "output_dir",
        ):
            if getattr(self.training, field_name) is None:
                missing.append(f"training.{field_name}")
        if missing:
            raise ValueError("SFT configuration is not training-ready; fill: " + ", ".join(missing))


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a YAML mapping")
    return dict(value)


def _construct(config_type: type, values: dict[str, Any], section_name: str):
    expected = set(config_type.__dataclass_fields__)
    actual = set(values)
    if expected != actual:
        raise ValueError(
            f"{section_name} fields do not match the schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    try:
        return config_type(**values)
    except TypeError as exc:
        raise TypeError(f"Invalid {section_name} configuration: {exc}") from exc


def load_sft_config(path: str | Path, *, require_training_ready: bool = False) -> SFTConfig:
    """Load a strict YAML definition and optionally require runnable values."""

    with Path(path).open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    if not isinstance(raw, dict):
        raise TypeError("SFT configuration root must be a YAML mapping")
    expected_root = {"schema_version", "experiment_name", "dataset", "model", "lora", "optimizer", "training"}
    if set(raw) != expected_root:
        raise ValueError(
            f"SFT configuration fields do not match the schema; "
            f"missing={sorted(expected_root - set(raw))}, extra={sorted(set(raw) - expected_root)}"
        )

    dataset_raw = _section(raw, "dataset")
    normalization_raw = dataset_raw.pop("normalization", None)
    if not isinstance(normalization_raw, dict):
        raise TypeError("dataset.normalization must be a YAML mapping")
    dataset_raw["normalization"] = _construct(
        NormalizationConfig, normalization_raw, "dataset.normalization"
    )
    try:
        dataset_raw["split_policy"] = SplitPolicy(dataset_raw["split_policy"])
    except KeyError as exc:
        raise ValueError("dataset.split_policy is required") from exc
    except ValueError as exc:
        raise ValueError("dataset.split_policy is unsupported") from exc
    if isinstance(dataset_raw.get("image_keys"), list):
        dataset_raw["image_keys"] = tuple(dataset_raw["image_keys"])

    config = SFTConfig(
        schema_version=raw["schema_version"],
        experiment_name=raw["experiment_name"],
        dataset=_construct(DatasetConfig, dataset_raw, "dataset"),
        model=_construct(ModelConfig, _section(raw, "model"), "model"),
        lora=_construct(LoRAConfig, _section(raw, "lora"), "lora"),
        optimizer=_construct(OptimizerConfig, _section(raw, "optimizer"), "optimizer"),
        training=_construct(TrainingConfig, _section(raw, "training"), "training"),
    )
    config.validate_definition()
    if require_training_ready:
        config.validate_training_ready()
    return config
