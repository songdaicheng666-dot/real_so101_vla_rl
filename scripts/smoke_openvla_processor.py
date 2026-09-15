"""Download the real OpenVLA processor and exercise the SO-101 transform."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from real_so101_vla_rl.data.normalization import (
    NormalizationStats,
    compute_normalization_stats,
)
from real_so101_vla_rl.data.openvla_oft import (
    IGNORE_INDEX,
    SO101OpenVLABatchTransform,
)
from real_so101_vla_rl.data.schema import SENSOR_VALID_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="openvla/openvla-7b")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def register_openvla_classes() -> None:
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import (
        PrismaticImageProcessor,
        PrismaticProcessor,
    )
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForVision2Seq,
        AutoProcessor,
    )

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def synthetic_normalization_stats() -> NormalizationStats:
    samples = [
        {"observation.state": [-1.0] * 6, "action": [-1.0] * 6},
        {"observation.state": [1.0] * 6, "action": [1.0] * 6},
    ]
    return compute_normalization_stats(
        samples,
        train_episode_indices=[0],
        unnorm_key="so101_battery_dual_rgb_v2",
    )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OPENVLA_ROBOT_PLATFORM", "SO101")

    register_openvla_classes()

    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=False,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    transform = SO101OpenVLABatchTransform(
        normalization=synthetic_normalization_stats(),
        base_tokenizer=processor.tokenizer,
        action_tokenizer=ActionTokenizer(processor.tokenizer),
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    sample = transform(
        {
            "observation.images.overview": np.zeros((256, 256, 3), dtype=np.uint8),
            "observation.images.wrist": np.ones((256, 256, 3), dtype=np.uint8),
            "observation.state": torch.zeros(6, dtype=torch.float32),
            "action": torch.zeros((8, 6), dtype=torch.float32),
            "action_is_pad": torch.zeros(8, dtype=torch.bool),
            "task": "Pick up the red battery and place it in T0.",
            **{key: torch.tensor([True]) for key in SENSOR_VALID_KEYS},
        }
    )

    assert (ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM) == (6, 8, 6)
    assert sample["pixel_values"].ndim == 3
    assert sample["actions"].shape == (8, 6)
    assert sample["proprio"].shape == (6,)
    assert sample["labels"].shape == sample["input_ids"].shape
    action_value_tokens = ACTION_DIM * NUM_ACTIONS_CHUNK
    supervised_tokens = int(sample["labels"].ne(IGNORE_INDEX).sum())
    assert supervised_tokens == action_value_tokens + 1
    assert int(sample["labels"][-1]) != IGNORE_INDEX
    print(
        {
            "model_id": args.model_id,
            "input_ids": tuple(sample["input_ids"].shape),
            "pixel_values": tuple(sample["pixel_values"].shape),
            "proprio": tuple(sample["proprio"].shape),
            "actions": tuple(sample["actions"].shape),
            "action_value_tokens": action_value_tokens,
            "stop_tokens": supervised_tokens - action_value_tokens,
        }
    )


if __name__ == "__main__":
    main()
