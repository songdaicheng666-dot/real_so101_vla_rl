"""Run one real OpenVLA-OFT continuous-action forward pass on CUDA."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from smoke_openvla_processor import (
    register_openvla_classes,
    synthetic_normalization_stats,
)

from real_so101_vla_rl.data.openvla_oft import (
    OpenVLABatchCollator,
    SO101OpenVLABatchTransform,
)
from real_so101_vla_rl.data.schema import SENSOR_VALID_KEYS
from real_so101_vla_rl.models.losses import masked_action_l1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="openvla/openvla-7b")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OPENVLA_ROBOT_PLATFORM", "SO101")
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a CUDA GPU")

    register_openvla_classes()

    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.models.projectors import ProprioProjector
    from prismatic.training.train_utils import (
        get_current_action_mask,
        get_next_actions_mask,
    )
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
    from transformers import AutoModelForVision2Seq, AutoProcessor

    load_kwargs = {
        # The checkpoint bundles the base OpenVLA classes. Use the locally
        # registered OpenVLA-OFT classes while loading the same weights.
        "trust_remote_code": False,
        "cache_dir": args.cache_dir,
        "local_files_only": args.local_files_only,
    }
    processor = AutoProcessor.from_pretrained(args.model_id, **load_kwargs)
    vla = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        **load_kwargs,
    ).to("cuda")
    vla.vision_backbone.set_num_images_in_input(2)
    vla.eval()

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
            "observation.state": torch.zeros(PROPRIO_DIM, dtype=torch.float32),
            "action": torch.zeros(
                (NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=torch.float32
            ),
            "action_is_pad": torch.tensor(
                [False] * (NUM_ACTIONS_CHUNK - 2) + [True, True]
            ),
            "task": "Pick up the red battery and place it in T0.",
            **{key: torch.tensor([True]) for key in SENSOR_VALID_KEYS},
        }
    )
    collator = OpenVLABatchCollator(
        model_max_length=processor.tokenizer.model_max_length,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    batch = collator([sample])
    model_batch = {
        name: value.to("cuda") if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }

    proprio_projector = ProprioProjector(
        llm_dim=vla.llm_dim,
        proprio_dim=PROPRIO_DIM,
    ).to("cuda")
    action_head = L1RegressionActionHead(
        input_dim=vla.llm_dim,
        hidden_dim=vla.llm_dim,
        action_dim=ACTION_DIM,
    ).to(device="cuda", dtype=torch.bfloat16)
    proprio_projector.eval()
    action_head.eval()

    num_patches = (
        vla.vision_backbone.get_num_patches()
        * vla.vision_backbone.get_num_images_in_input()
        + 1
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla(
            input_ids=model_batch["input_ids"],
            attention_mask=model_batch["attention_mask"],
            pixel_values=model_batch["pixel_values"].to(torch.bfloat16),
            labels=model_batch["labels"],
            output_hidden_states=True,
            proprio=model_batch["proprio"],
            proprio_projector=proprio_projector,
            use_film=False,
        )
        target_token_ids = model_batch["labels"][:, 1:]
        action_mask = get_current_action_mask(
            target_token_ids
        ) | get_next_actions_mask(target_token_ids)
        action_token_count = int(action_mask.sum())
        expected_action_tokens = NUM_ACTIONS_CHUNK * ACTION_DIM
        if action_token_count != expected_action_tokens:
            raise RuntimeError(
                f"Expected {expected_action_tokens} action tokens, got "
                f"{action_token_count}"
            )

        text_hidden_states = output.hidden_states[-1][:, num_patches:-1]
        action_hidden_states = text_hidden_states[action_mask].reshape(
            1, expected_action_tokens, -1
        )
        predicted_actions = action_head.predict_action(
            action_hidden_states.to(torch.bfloat16)
        )
        loss = masked_action_l1(
            predicted_actions,
            model_batch["actions"].to(torch.bfloat16),
            model_batch["action_is_pad"],
        )

    if predicted_actions.shape != (1, NUM_ACTIONS_CHUNK, ACTION_DIM):
        raise RuntimeError(f"Unexpected action shape: {predicted_actions.shape}")
    if not torch.isfinite(predicted_actions).all() or not torch.isfinite(loss):
        raise RuntimeError("Forward pass produced non-finite values")

    print(
        {
            "model_id": args.model_id,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "input_ids": tuple(model_batch["input_ids"].shape),
            "pixel_values": tuple(model_batch["pixel_values"].shape),
            "action_tokens": action_token_count,
            "predicted_actions": tuple(predicted_actions.shape),
            "masked_l1": float(loss),
            "peak_gpu_memory_gib": round(
                torch.cuda.max_memory_allocated() / 1024**3, 2
            ),
        }
    )


if __name__ == "__main__":
    main()
