"""Convert LeRobot samples into the OpenVLA-OFT training contract.

The transform follows the public OpenVLA-OFT RLDS batch transform while keeping
the tokenizer, prompt builder, action tokenizer, and image transform injectable.
No model or checkpoint is imported by this module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch
from PIL import Image

from .lerobot_dataset import (
    LoadedLeRobotSplit,
    SplitName,
    create_lerobot_split_dataset,
)
from .normalization import NormalizationStats, normalize_q99
from .schema import (
    ACTION_CHUNK_SIZE,
    ACTION_DIM,
    ACTION_KEY,
    MODEL_INPUT_KEYS,
    RGB_IMAGE_KEYS,
    SENSOR_VALID_KEYS,
    STATE_KEY,
    TASK_KEY,
    AtomicTask,
)

if TYPE_CHECKING:
    from real_so101_vla_rl.models.sft_config import SFTConfig

IGNORE_INDEX = -100


class PromptBuilder(Protocol):
    def add_turn(self, role: str, value: str) -> None: ...

    def get_prompt(self) -> str: ...


PromptBuilderFactory = Callable[[str], PromptBuilder]
ImageTransform = Callable[[Image.Image], torch.Tensor]
ActionTokenizer = Callable[[np.ndarray], Any]
BaseTokenizer = Callable[..., Any]


def format_openvla_instruction(task: str) -> str:
    """Validate a canonical task and place it in OpenVLA's question template."""

    AtomicTask.from_instruction(task)
    instruction = task.removesuffix(".").lower()
    return f"What action should the robot take to {instruction}?"


def _as_rgb_pil(image: Any, *, field_name: str) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    else:
        try:
            image = np.asarray(image)
        except Exception as exc:
            raise TypeError(f"{field_name} cannot be converted to an array") from exc

    if image.ndim != 3:
        raise ValueError(
            f"{field_name} must have three dimensions, got shape={image.shape}"
        )
    if image.shape[-1] == 3:
        array = image
    elif image.shape[0] == 3:
        array = np.moveaxis(image, 0, -1)
    else:
        raise ValueError(f"{field_name} must have exactly three RGB channels")

    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError(f"{field_name} contains non-finite pixels")
        minimum = float(array.min())
        maximum = float(array.max())
        if minimum >= 0 and maximum <= 1:
            array = np.rint(array * 255)
        elif minimum < 0 or maximum > 255:
            raise ValueError(
                f"{field_name} floating pixels must be in [0, 1] or [0, 255]"
            )
        else:
            array = np.rint(array)
    elif not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{field_name} must contain integer or floating-point pixels")

    if np.any(array < 0) or np.any(array > 255):
        raise ValueError(f"{field_name} integer pixels must be in [0, 255]")
    return Image.fromarray(np.ascontiguousarray(array, dtype=np.uint8))


def _flatten_token_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Sequence):
        return "".join(_flatten_token_text(item) for item in value)
    raise TypeError(
        f"Action tokenizer returned unsupported type {type(value).__name__}"
    )


def _action_chunk_text(
    action_tokenizer: ActionTokenizer,
    actions: torch.Tensor,
) -> str:
    numpy_actions = actions.detach().cpu().numpy()
    current = _flatten_token_text(action_tokenizer(numpy_actions[0]))
    future = _flatten_token_text(action_tokenizer(numpy_actions[1:]))
    action_text = current + future
    expected_tokens = ACTION_CHUNK_SIZE * ACTION_DIM
    if len(action_text) != expected_tokens:
        raise ValueError(
            "Action tokenizer must produce one token character per action value; "
            f"expected {expected_tokens}, got {len(action_text)}"
        )
    return action_text


def _extract_input_ids(encoded: Any) -> torch.Tensor:
    if isinstance(encoded, Mapping):
        input_ids = encoded.get("input_ids")
    else:
        input_ids = getattr(encoded, "input_ids", None)
    if input_ids is None:
        raise TypeError("Base tokenizer output must expose input_ids")

    tensor = torch.as_tensor(input_ids, dtype=torch.long)
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError(
            "Base tokenizer input_ids must be a non-empty one-dimensional sequence"
        )
    return tensor


@dataclass(frozen=True, slots=True)
class SO101OpenVLABatchTransform:
    """Convert one LeRobot frame/window into one model-ready OFT sample."""

    normalization: NormalizationStats
    base_tokenizer: BaseTokenizer
    action_tokenizer: ActionTokenizer
    image_transform: ImageTransform
    prompt_builder_fn: PromptBuilderFactory
    predict_stop_token: bool = True

    @property
    def dataset_statistics(self) -> dict[str, dict[str, dict[str, list]]]:
        return self.normalization.to_openvla_dict()

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        required = {
            *MODEL_INPUT_KEYS,
            *SENSOR_VALID_KEYS,
            ACTION_KEY,
            "action_is_pad",
        }
        missing = sorted(required - set(sample))
        if missing:
            raise ValueError(f"LeRobot sample is missing required fields: {missing}")

        for key in SENSOR_VALID_KEYS:
            valid = torch.as_tensor(sample[key])
            if valid.dtype is not torch.bool or valid.shape not in ((), (1,)):
                raise TypeError(f"{key} must be a scalar or one-element bool tensor")
            if not bool(valid.reshape(-1)[0]):
                raise ValueError(f"{key} must be true before a sample enters the model")

        task = sample[TASK_KEY]
        if not isinstance(task, str):
            raise TypeError(f"{TASK_KEY} must be a string")
        question = format_openvla_instruction(task)

        state = torch.as_tensor(sample[STATE_KEY])
        if state.shape != (ACTION_DIM,):
            raise ValueError(
                f"{STATE_KEY} must have shape ({ACTION_DIM},), got {tuple(state.shape)}"
            )
        normalized_state = torch.tensor(
            normalize_q99(state.tolist(), self.normalization.observation_state),
            dtype=torch.float32,
        )

        actions = torch.as_tensor(sample[ACTION_KEY])
        expected_action_shape = (ACTION_CHUNK_SIZE, ACTION_DIM)
        if actions.shape != expected_action_shape:
            raise ValueError(
                f"{ACTION_KEY} must have shape {expected_action_shape}, got {tuple(actions.shape)}"
            )
        normalized_actions = torch.tensor(
            [
                normalize_q99(action.tolist(), self.normalization.action)
                for action in actions
            ],
            dtype=torch.float32,
        )

        raw_padding = torch.as_tensor(sample["action_is_pad"])
        if raw_padding.dtype is not torch.bool:
            raise TypeError("action_is_pad must be a boolean tensor or sequence")
        if raw_padding.shape != (ACTION_CHUNK_SIZE,):
            raise ValueError(
                f"action_is_pad must have shape ({ACTION_CHUNK_SIZE},), "
                f"got {tuple(raw_padding.shape)}"
            )
        action_is_pad = raw_padding.clone()
        if bool(action_is_pad[0]) or bool(action_is_pad.all()):
            raise ValueError("The current action must never be marked as padding")

        transformed_images = []
        for key in RGB_IMAGE_KEYS:
            transformed = self.image_transform(
                _as_rgb_pil(sample[key], field_name=key)
            )
            if not isinstance(transformed, torch.Tensor):
                raise TypeError("image_transform must return a torch.Tensor")
            if transformed.ndim != 3:
                raise ValueError(
                    "image_transform must return [C, H, W], "
                    f"got {tuple(transformed.shape)} for {key}"
                )
            if transformed_images and transformed.shape[1:] != transformed_images[0].shape[1:]:
                raise ValueError("All transformed RGB views must have the same height and width")
            transformed_images.append(transformed.to(dtype=torch.float32))
        pixel_values = torch.cat(transformed_images, dim=0).contiguous()

        action_text = _action_chunk_text(self.action_tokenizer, normalized_actions)
        prompt_builder = self.prompt_builder_fn("openvla")
        prompt_builder.add_turn("human", question)
        prompt_builder.add_turn("gpt", action_text)
        input_ids = _extract_input_ids(
            self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True)
        )

        action_token_count = ACTION_CHUNK_SIZE * ACTION_DIM
        stop_token_count = 1
        supervised_suffix = action_token_count + stop_token_count
        if input_ids.numel() <= supervised_suffix:
            raise ValueError(
                "Tokenized OpenVLA prompt does not contain a prompt prefix and "
                "the complete action-token suffix"
            )
        labels = input_ids.clone()
        labels[:-supervised_suffix] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "proprio": normalized_state,
            "actions": normalized_actions,
            "action_is_pad": action_is_pad,
            "dataset_name": self.normalization.unnorm_key,
        }


class SO101OpenVLADataset(torch.utils.data.Dataset):
    """Map-style OpenVLA view over an episode-filtered LeRobotDataset."""

    def __init__(
        self,
        source_dataset: Any,
        transform: SO101OpenVLABatchTransform,
        *,
        loaded_split: LoadedLeRobotSplit | None = None,
    ) -> None:
        self.source_dataset = source_dataset
        self.transform = transform
        self.loaded_split = loaded_split
        self.dataset_statistics = transform.dataset_statistics

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.transform(self.source_dataset[index])


def create_openvla_oft_dataset(
    config: SFTConfig,
    *,
    split: SplitName,
    base_tokenizer: BaseTokenizer,
    action_tokenizer: ActionTokenizer,
    image_transform: ImageTransform,
    prompt_builder_fn: PromptBuilderFactory,
    dataset_cls: Callable[..., Any] | None = None,
    snapshot_download_fn: Callable[..., str] | None = None,
) -> SO101OpenVLADataset:
    """Build the complete LeRobot-to-OpenVLA dataset for one split."""

    loaded = create_lerobot_split_dataset(
        config,
        split=split,
        dataset_cls=dataset_cls,
        snapshot_download_fn=snapshot_download_fn,
    )
    transform = SO101OpenVLABatchTransform(
        normalization=loaded.metadata.normalization,
        base_tokenizer=base_tokenizer,
        action_tokenizer=action_tokenizer,
        image_transform=image_transform,
        prompt_builder_fn=prompt_builder_fn,
    )
    return SO101OpenVLADataset(
        loaded.dataset,
        transform,
        loaded_split=loaded,
    )


@dataclass(frozen=True, slots=True)
class OpenVLABatchCollator:
    """Right-pad and stack model-ready SO-101 OpenVLA samples."""

    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"

    def __post_init__(self) -> None:
        if self.model_max_length <= 0:
            raise ValueError("model_max_length must be positive")
        if type(self.pad_token_id) is not int:
            raise TypeError("pad_token_id must be an integer")
        if self.padding_side != "right":
            raise ValueError("Only right-side padding is supported")

    def __call__(self, instances: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not instances:
            raise ValueError("Cannot collate an empty batch")

        lengths = []
        for index, instance in enumerate(instances):
            required = {
                "input_ids",
                "labels",
                "pixel_values",
                "proprio",
                "actions",
                "action_is_pad",
                "dataset_name",
            }
            missing = sorted(required - set(instance))
            if missing:
                raise ValueError(f"Sample {index} is missing model fields: {missing}")

            input_ids = instance["input_ids"]
            labels = instance["labels"]
            if not isinstance(input_ids, torch.Tensor) or not isinstance(
                labels, torch.Tensor
            ):
                raise TypeError(f"Sample {index} input_ids and labels must be tensors")
            if input_ids.dtype is not torch.long or labels.dtype is not torch.long:
                raise TypeError(
                    f"Sample {index} input_ids and labels must use torch.long"
                )
            if input_ids.ndim != 1 or labels.shape != input_ids.shape:
                raise ValueError(
                    f"Sample {index} token tensors must be aligned one-dimensional arrays"
                )
            if input_ids.numel() > self.model_max_length:
                raise ValueError(
                    f"Sample {index} has {input_ids.numel()} tokens, exceeding "
                    f"model_max_length={self.model_max_length}"
                )

            expected = {
                "proprio": (ACTION_DIM,),
                "actions": (ACTION_CHUNK_SIZE, ACTION_DIM),
                "action_is_pad": (ACTION_CHUNK_SIZE,),
            }
            for name, expected_shape in expected.items():
                value = instance[name]
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"Sample {index} {name} must be a tensor")
                if value.shape != expected_shape:
                    raise ValueError(
                        f"Sample {index} {name} must have shape {expected_shape}, "
                        f"got {tuple(value.shape)}"
                    )
            pixel_values = instance["pixel_values"]
            if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 3:
                raise ValueError(
                    f"Sample {index} pixel_values must have shape [C, H, W]"
                )
            if instance["action_is_pad"].dtype is not torch.bool:
                raise TypeError(f"Sample {index} action_is_pad must use torch.bool")
            if bool(instance["action_is_pad"][0]):
                raise ValueError(
                    f"Sample {index} current action must not be marked as padding"
                )
            if (
                not isinstance(instance["dataset_name"], str)
                or not instance["dataset_name"]
            ):
                raise ValueError(
                    f"Sample {index} dataset_name must be a non-empty string"
                )
            lengths.append(input_ids.numel())

        sequence_length = max(lengths)
        batch_size = len(instances)
        input_ids = torch.full(
            (batch_size, sequence_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        labels = torch.full(
            (batch_size, sequence_length),
            IGNORE_INDEX,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (batch_size, sequence_length),
            dtype=torch.bool,
        )
        for index, (instance, length) in enumerate(
            zip(instances, lengths, strict=True)
        ):
            input_ids[index, :length] = instance["input_ids"]
            labels[index, :length] = instance["labels"]
            attention_mask[index, :length] = True

        pixel_values = torch.stack([instance["pixel_values"] for instance in instances])
        proprio = torch.stack([instance["proprio"] for instance in instances])
        actions = torch.stack([instance["actions"] for instance in instances])
        action_is_pad = torch.stack(
            [instance["action_is_pad"] for instance in instances]
        )

        expected_shapes = {
            "pixel_values": (batch_size, *instances[0]["pixel_values"].shape),
            "proprio": (batch_size, ACTION_DIM),
            "actions": (batch_size, ACTION_CHUNK_SIZE, ACTION_DIM),
            "action_is_pad": (batch_size, ACTION_CHUNK_SIZE),
        }
        actual_tensors = {
            "pixel_values": pixel_values,
            "proprio": proprio,
            "actions": actions,
            "action_is_pad": action_is_pad,
        }
        for name, expected_shape in expected_shapes.items():
            if actual_tensors[name].shape != expected_shape:
                raise ValueError(
                    f"Collated {name} must have shape {expected_shape}, "
                    f"got {tuple(actual_tensors[name].shape)}"
                )
        if action_is_pad.dtype is not torch.bool:
            raise TypeError("Collated action_is_pad must use torch.bool")
        if bool(action_is_pad[:, 0].any()):
            raise ValueError("The current action must never be marked as padding")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "proprio": proprio,
            "actions": actions,
            "action_is_pad": action_is_pad,
            "dataset_names": [instance["dataset_name"] for instance in instances],
        }
