"""Run a LoRA SFT optimizer step on one synthetic SO-101 batch."""

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
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.lora_rank <= 0:
        parser.error("--lora-rank must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    return args


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OPENVLA_ROBOT_PLATFORM", "SO101")
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a CUDA GPU")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    register_openvla_classes()

    from peft import LoraConfig, get_peft_model
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
    num_patches = (
        vla.vision_backbone.get_num_patches()
        * vla.vision_backbone.get_num_images_in_input()
        + 1
    )
    vla.config.use_cache = False
    vla.gradient_checkpointing_enable()
    vla = get_peft_model(
        vla,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=min(args.lora_rank, 16),
            lora_dropout=0.0,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        ),
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
    batch = {
        name: value.to("cuda") if isinstance(value, torch.Tensor) else value
        for name, value in collator([sample]).items()
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
    modules = (vla, proprio_projector, action_head)
    for module in modules:
        module.train()

    trainable_parameters = [
        parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    action_token_count = NUM_ACTIONS_CHUNK * ACTION_DIM
    losses: list[float] = []
    saw_lora_gradient = False
    lora_parameter_changed = False
    torch.cuda.reset_peak_memory_stats()

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"].to(torch.bfloat16),
                labels=batch["labels"],
                output_hidden_states=True,
                proprio=batch["proprio"],
                proprio_projector=proprio_projector,
                use_film=False,
            )
            target_token_ids = batch["labels"][:, 1:]
            action_mask = get_current_action_mask(
                target_token_ids
            ) | get_next_actions_mask(target_token_ids)
            if int(action_mask.sum()) != action_token_count:
                raise RuntimeError("The model batch does not contain 48 action tokens")
            text_hidden_states = output.hidden_states[-1][:, num_patches:-1]
            action_hidden_states = text_hidden_states[action_mask].reshape(
                1, action_token_count, -1
            )
            predicted_actions = action_head.predict_action(
                action_hidden_states.to(torch.bfloat16)
            )
            loss = masked_action_l1(
                predicted_actions,
                batch["actions"].to(torch.bfloat16),
                batch["action_is_pad"],
            )

        if not torch.isfinite(loss):
            raise RuntimeError(f"Step {step} produced a non-finite loss")
        loss.backward()
        if step == 0:
            lora_probe = next(
                (
                    parameter
                    for name, parameter in vla.named_parameters()
                    if "lora_" in name
                    and parameter.grad is not None
                    and bool(torch.isfinite(parameter.grad).all())
                    and bool(parameter.grad.ne(0).any())
                ),
                None,
            )
            saw_lora_gradient = lora_probe is not None
            lora_before_step = (
                lora_probe.detach().clone() if lora_probe is not None else None
            )
        torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
        optimizer.step()
        if step == 0 and lora_probe is not None and lora_before_step is not None:
            lora_parameter_changed = not torch.equal(
                lora_probe.detach(), lora_before_step
            )
        losses.append(float(loss.detach()))
        print({"step": step, "masked_l1": losses[-1]}, flush=True)

    if not saw_lora_gradient:
        raise RuntimeError("No non-zero finite LoRA gradient was produced")
    if not lora_parameter_changed:
        raise RuntimeError("The optimizer did not update the probed LoRA parameter")

    print(
        {
            "model_id": args.model_id,
            "steps": args.steps,
            "lora_rank": args.lora_rank,
            "trainable_parameters": sum(
                parameter.numel() for parameter in trainable_parameters
            ),
            "initial_masked_l1": losses[0],
            "final_masked_l1": losses[-1],
            "lora_gradient": saw_lora_gradient,
            "lora_parameter_changed": lora_parameter_changed,
            "peak_gpu_memory_gib": round(
                torch.cuda.max_memory_allocated() / 1024**3, 2
            ),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
