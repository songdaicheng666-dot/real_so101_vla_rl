"""Single-GPU OpenVLA-OFT training utilities for the SO-101 project."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Self

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from real_so101_vla_rl.data.openvla_oft import (
    OpenVLABatchCollator,
    create_openvla_oft_dataset,
)
from real_so101_vla_rl.models.losses import masked_action_l1
from real_so101_vla_rl.models.sft_config import SFTConfig

_OPENVLA_REGISTERED = False


def register_openvla_classes() -> None:
    """Register the installed OpenVLA-OFT implementation with Transformers."""

    global _OPENVLA_REGISTERED
    if _OPENVLA_REGISTERED:
        return
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import (
        OpenVLAForActionPrediction,
    )
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
    AutoModelForVision2Seq.register(
        OpenVLAConfig, OpenVLAForActionPrediction
    )
    _OPENVLA_REGISTERED = True


@dataclass(slots=True)
class OpenVLAOFTModules:
    processor: Any
    vla: torch.nn.Module
    action_head: torch.nn.Module
    proprio_projector: torch.nn.Module
    num_patches: int

    def train(self) -> None:
        self.vla.train()
        self.action_head.train()
        self.proprio_projector.train()

    def eval(self) -> None:
        self.vla.eval()
        self.action_head.eval()
        self.proprio_projector.eval()

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [
            parameter
            for module in (
                self.vla,
                self.action_head,
                self.proprio_projector,
            )
            for parameter in module.parameters()
            if parameter.requires_grad
        ]


def load_openvla_oft_modules(
    config: SFTConfig,
    *,
    cache_dir: str | Path | None,
    local_files_only: bool,
    device: torch.device,
    checkpoint: str | Path | None = None,
) -> OpenVLAOFTModules:
    """Load the base checkpoint and attach the OFT LoRA training modules."""

    register_openvla_classes()
    from peft import LoraConfig, PeftModel, get_peft_model
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from transformers import AutoModelForVision2Seq, AutoProcessor

    load_kwargs = {
        "trust_remote_code": False,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "local_files_only": local_files_only,
    }
    checkpoint_path = (
        Path(checkpoint).expanduser().resolve()
        if checkpoint is not None
        else None
    )
    if checkpoint_path is not None:
        required = (
            checkpoint_path / "lora_adapter" / "adapter_config.json",
            checkpoint_path / "processor",
            checkpoint_path / "action_head.pt",
            checkpoint_path / "proprio_projector.pt",
            checkpoint_path / "dataset_statistics.json",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Checkpoint is incomplete; missing: " + ", ".join(missing)
            )
    processor_source: str | Path = (
        checkpoint_path / "processor"
        if checkpoint_path is not None
        else config.model.checkpoint
    )
    processor = AutoProcessor.from_pretrained(processor_source, **load_kwargs)
    vla = AutoModelForVision2Seq.from_pretrained(
        config.model.checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        **load_kwargs,
    ).to(device)
    vla.vision_backbone.set_num_images_in_input(config.model.num_images)
    vla.config.use_cache = False
    if config.training.gradient_checkpointing:
        vla.gradient_checkpointing_enable()
    if checkpoint_path is None:
        vla = get_peft_model(
            vla,
            LoraConfig(
                r=config.lora.rank,
                lora_alpha=config.lora.alpha,
                lora_dropout=config.lora.dropout,
                target_modules=config.lora.target_modules,
                init_lora_weights="gaussian",
            ),
        )
    else:
        vla = PeftModel.from_pretrained(
            vla,
            checkpoint_path / "lora_adapter",
            is_trainable=False,
        )
    if config.training.gradient_checkpointing and hasattr(
        vla, "enable_input_require_grads"
    ):
        vla.enable_input_require_grads()

    proprio_projector = ProprioProjector(
        llm_dim=vla.llm_dim,
        proprio_dim=config.model.proprio_dim,
    ).to(device)
    action_head = L1RegressionActionHead(
        input_dim=vla.llm_dim,
        hidden_dim=vla.llm_dim,
        action_dim=config.model.action_dim,
    ).to(device=device, dtype=torch.bfloat16)
    if checkpoint_path is not None:
        action_head.load_state_dict(
            torch.load(
                checkpoint_path / "action_head.pt",
                map_location=device,
                weights_only=True,
            )
        )
        proprio_projector.load_state_dict(
            torch.load(
                checkpoint_path / "proprio_projector.pt",
                map_location=device,
                weights_only=True,
            )
        )
        action_head.requires_grad_(False)
        proprio_projector.requires_grad_(False)
    num_patches = (
        vla.vision_backbone.get_num_patches()
        * vla.vision_backbone.get_num_images_in_input()
        + 1
    )
    return OpenVLAOFTModules(
        processor=processor,
        vla=vla,
        action_head=action_head,
        proprio_projector=proprio_projector,
        num_patches=num_patches,
    )


def create_model_datasets(
    config: SFTConfig,
    *,
    processor: Any,
) -> dict[str, Any]:
    """Create the three model-facing datasets through the project adapter."""

    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.vla.action_tokenizer import ActionTokenizer

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    kwargs = {
        "base_tokenizer": processor.tokenizer,
        "action_tokenizer": action_tokenizer,
        "image_transform": processor.image_processor.apply_transform,
        "prompt_builder_fn": PurePromptBuilder,
    }
    return {
        split: create_openvla_oft_dataset(config, split=split, **kwargs)
        for split in ("train", "val", "test")
    }


def create_dataloaders(
    config: SFTConfig,
    *,
    datasets: Mapping[str, Any],
    processor: Any,
) -> dict[str, DataLoader]:
    """Create deterministic map-style loaders for the finite LeRobotDataset."""

    collator = OpenVLABatchCollator(
        model_max_length=processor.tokenizer.model_max_length,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    generator = torch.Generator()
    generator.manual_seed(config.training.seed)
    batch_size = config.training.per_device_train_batch_size
    workers = config.training.num_workers
    if batch_size is None or workers is None:
        raise ValueError("Training batch size and num_workers must be configured")
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collator,
            num_workers=workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=workers,
            pin_memory=True,
        ),
    }


def move_batch_to_device(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def forward_continuous_action_loss(
    modules: OpenVLAOFTModules,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the OpenVLA-OFT hidden-state action head and masked L1 loss."""

    from prismatic.training.train_utils import (
        get_current_action_mask,
        get_next_actions_mask,
    )
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

    model_batch = move_batch_to_device(batch, device)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        output = modules.vla(
            input_ids=model_batch["input_ids"],
            attention_mask=model_batch["attention_mask"],
            pixel_values=model_batch["pixel_values"].to(torch.bfloat16),
            labels=model_batch["labels"],
            output_hidden_states=True,
            proprio=model_batch["proprio"],
            proprio_projector=modules.proprio_projector,
            use_film=False,
        )
        target_token_ids = model_batch["labels"][:, 1:]
        action_mask = get_current_action_mask(
            target_token_ids
        ) | get_next_actions_mask(target_token_ids)
        batch_size = model_batch["input_ids"].shape[0]
        expected_count = batch_size * NUM_ACTIONS_CHUNK * ACTION_DIM
        actual_count = int(action_mask.sum())
        if actual_count != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} action-token states, got "
                f"{actual_count}"
            )
        text_hidden_states = output.hidden_states[-1][
            :, modules.num_patches : -1
        ]
        action_hidden_states = text_hidden_states[action_mask].reshape(
            batch_size,
            NUM_ACTIONS_CHUNK * ACTION_DIM,
            -1,
        )
        predicted_actions = modules.action_head.predict_action(
            action_hidden_states.to(torch.bfloat16)
        )
        loss = masked_action_l1(
            predicted_actions,
            model_batch["actions"].to(torch.bfloat16),
            model_batch["action_is_pad"],
        )
    return loss, predicted_actions


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    max_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if max_steps <= 0 or not 0 <= warmup_steps < max_steps:
        raise ValueError("Invalid scheduler max_steps or warmup_steps")

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def moving_average(values: list[float], window: int = 10) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    return [
        sum(values[max(0, index - window + 1) : index + 1])
        / min(window, index + 1)
        for index in range(len(values))
    ]


class MetricsWriter:
    fields = (
        "step",
        "epoch",
        "train_loss",
        "smoothed_train_loss",
        "val_loss",
        "learning_rate",
        "grad_norm",
    )

    def __init__(self, output_dir: Path) -> None:
        self.jsonl_path = output_dir / "metrics.jsonl"
        self.csv_path = output_dir / "metrics.csv"
        self._csv_file = self.csv_path.open("w", encoding="utf-8", newline="")
        self._csv = csv.DictWriter(self._csv_file, fieldnames=self.fields)
        self._csv.writeheader()

    def write(self, metrics: Mapping[str, Any]) -> None:
        row = {field: metrics.get(field) for field in self.fields}
        with self.jsonl_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, sort_keys=True) + "\n")
        self._csv.writerow(row)
        self._csv_file.flush()

    def close(self) -> None:
        self._csv_file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def save_loss_curve(
    path: str | Path, metrics: Iterable[Mapping[str, Any]]
) -> None:
    """Render raw, smoothed, and validation losses without Matplotlib."""

    rows = list(metrics)
    train = [
        (int(row["step"]), float(row["train_loss"])) for row in rows
    ]
    smooth = [
        (int(row["step"]), float(row["smoothed_train_loss"]))
        for row in rows
    ]
    validation = [
        (int(row["step"]), float(row["val_loss"]))
        for row in rows
        if row.get("val_loss") is not None
    ]
    if not train:
        raise ValueError("Cannot plot an empty training history")

    width, height = 1200, 720
    left, right, top, bottom = 90, 35, 55, 80
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_step = max(step for step, _ in train)
    all_values = [value for _, value in train] + [
        value for _, value in validation
    ]
    low = min(all_values)
    high = max(all_values)
    if math.isclose(low, high):
        high = low + 1.0

    def point(item: tuple[int, float]) -> tuple[int, int]:
        step, value = item
        x = left + round(plot_width * (step - 1) / max(1, max_step - 1))
        y = top + round(plot_height * (high - value) / (high - low))
        return x, y

    draw.line((left, top, left, top + plot_height), fill="black", width=2)
    draw.line(
        (left, top + plot_height, left + plot_width, top + plot_height),
        fill="black",
        width=2,
    )
    for tick in range(6):
        value = low + (high - low) * tick / 5
        y = point((1, value))[1]
        draw.line((left - 5, y, left, y), fill="black", width=1)
        draw.text((8, y - 7), f"{value:.4f}", fill="black")
    draw.text((left, 18), "SO-101 synthetic OpenVLA-OFT masked L1", fill="black")
    draw.text((width // 2 - 20, height - 35), "step", fill="black")
    draw.text((left, height - 62), "1", fill="black")
    draw.text((left + plot_width - 20, height - 62), str(max_step), fill="black")

    if len(train) > 1:
        draw.line([point(item) for item in train], fill=(165, 185, 220), width=2)
        draw.line([point(item) for item in smooth], fill=(25, 90, 190), width=4)
    for item in validation:
        x, y = point(item)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(210, 45, 45))
    draw.line((780, 25, 825, 25), fill=(165, 185, 220), width=2)
    draw.text((835, 17), "train raw", fill="black")
    draw.line((935, 25, 980, 25), fill=(25, 90, 190), width=4)
    draw.text((990, 17), "train MA10", fill="black")
    draw.ellipse((1090, 20, 1100, 30), fill=(210, 45, 45))
    draw.text((1110, 17), "val", fill="black")
    canvas.save(path, format="PNG")


def evaluate(
    modules: OpenVLAOFTModules,
    dataloader: DataLoader,
    *,
    device: torch.device,
) -> float:
    modules.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for batch in dataloader:
            loss, _ = forward_continuous_action_loss(
                modules, batch, device=device
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Evaluation produced a non-finite loss")
            losses.append(float(loss))
    modules.train()
    if not losses:
        raise RuntimeError("Evaluation loader produced no batches")
    return sum(losses) / len(losses)


def _safe_package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_tree_sha256() -> str:
    """Fingerprint the project code and configuration used by the run."""

    project_root = Path(__file__).resolve().parents[3]
    candidates = [project_root / "pyproject.toml"]
    for relative, pattern in (
        ("src", "*.py"),
        ("scripts", "*.py"),
        ("configs", "*.yaml"),
    ):
        candidates.extend((project_root / relative).rglob(pattern))
    digest = hashlib.sha256()
    for path in sorted(path for path in candidates if path.is_file()):
        relative_path = path.relative_to(project_root).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_run_manifest(
    config: SFTConfig,
    *,
    datasets: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    gpu = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "name": properties.name,
            "total_memory_gib": round(properties.total_memory / 1024**3, 2),
        }
    return {
        "status": "running",
        "started_at_unix": time.time(),
        "command": sys.argv,
        "git_commit": _git_commit(),
        "source_tree_sha256": _source_tree_sha256(),
        "source_archive_sha256": os.environ.get(
            "REAL_SO101_SOURCE_ARCHIVE_SHA256"
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu,
        "packages": {
            name: _safe_package_version(name)
            for name in (
                "lerobot",
                "transformers",
                "peft",
                "flash-attn",
            )
        },
        "config": asdict(config),
        "dataset": {
            split: {
                "frames": len(dataset),
                "episodes": list(
                    dataset.loaded_split.metadata.episode_indices
                ),
            }
            for split, dataset in datasets.items()
        },
    }


def save_checkpoint(
    output_dir: Path,
    *,
    step: int,
    modules: OpenVLAOFTModules,
    dataset_statistics: Mapping[str, Any],
    keep: int,
) -> Path:
    checkpoint = output_dir / f"checkpoint-{step:06d}"
    checkpoint.mkdir(parents=True, exist_ok=False)
    modules.vla.save_pretrained(checkpoint / "lora_adapter")
    modules.processor.save_pretrained(checkpoint / "processor")
    torch.save(
        modules.action_head.state_dict(),
        checkpoint / "action_head.pt",
    )
    torch.save(
        modules.proprio_projector.state_dict(),
        checkpoint / "proprio_projector.pt",
    )
    (checkpoint / "dataset_statistics.json").write_text(
        json.dumps(dataset_statistics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"step": step}, indent=2) + "\n",
        encoding="utf-8",
    )

    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    for stale in checkpoints[:-keep]:
        shutil.rmtree(stale)
    return checkpoint


def run_training(
    config: SFTConfig,
    *,
    cache_dir: str | Path | None,
    local_files_only: bool,
    config_path: str | Path,
    preflight_only: bool = False,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Run the finite-dataset SFT loop and return its final manifest."""

    config.validate_training_ready()
    if checkpoint is not None and not preflight_only:
        raise ValueError(
            "Checkpoint loading currently supports --preflight-only; "
            "optimizer-state resume is not implemented"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("OpenVLA-OFT SFT requires a CUDA GPU")
    os.environ.setdefault("OPENVLA_ROBOT_PLATFORM", "SO101")
    torch.manual_seed(config.training.seed)
    torch.cuda.manual_seed_all(config.training.seed)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    modules = load_openvla_oft_modules(
        config,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        device=device,
        checkpoint=checkpoint,
    )
    datasets = create_model_datasets(config, processor=modules.processor)
    dataloaders = create_dataloaders(
        config, datasets=datasets, processor=modules.processor
    )
    first_batch = next(iter(dataloaders["train"]))
    batch_contract = {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for key, value in first_batch.items()
        if isinstance(value, torch.Tensor)
    }
    print(json.dumps({"model_batch": batch_contract}, sort_keys=True))

    if preflight_only:
        modules.eval()
        with torch.inference_mode():
            loss, predicted = forward_continuous_action_loss(
                modules, first_batch, device=device
            )
        result = {
            "preflight": True,
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "loss": float(loss),
            "predicted_actions_shape": list(predicted.shape),
            "batch": batch_contract,
            "peak_gpu_memory_gib": round(
                torch.cuda.max_memory_allocated() / 1024**3, 2
            ),
        }
        print(json.dumps(result, sort_keys=True))
        return result

    output_dir_value = config.training.output_dir
    if output_dir_value is None:
        raise ValueError("training.output_dir is required")
    output_dir = Path(output_dir_value).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty run directory: {output_dir}"
        )
    shutil.copy2(config_path, output_dir / "config.yaml")
    manifest = build_run_manifest(
        config, datasets=datasets, device=device
    )
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    parameters = modules.trainable_parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    max_steps = config.training.max_steps
    accumulation = config.training.gradient_accumulation_steps
    if max_steps is None or accumulation is None:
        raise ValueError("max_steps and gradient accumulation are required")
    warmup_steps = round(max_steps * config.optimizer.warmup_ratio)
    scheduler = build_cosine_scheduler(
        optimizer,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
    )

    history: list[dict[str, Any]] = []
    train_losses: list[float] = []
    global_step = 0
    epoch = 0
    modules.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with MetricsWriter(output_dir) as writer:
        while global_step < max_steps:
            epoch += 1
            for microbatch_index, batch in enumerate(dataloaders["train"]):
                loss, _ = forward_continuous_action_loss(
                    modules, batch, device=device
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Training produced a non-finite loss at "
                        f"step {global_step + 1}"
                    )
                (loss / accumulation).backward()
                if (microbatch_index + 1) % accumulation:
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                train_loss = float(loss.detach())
                train_losses.append(train_loss)
                val_loss = None
                if global_step % config.training.eval_steps == 0:
                    val_loss = evaluate(
                        modules, dataloaders["val"], device=device
                    )
                row = {
                    "step": global_step,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "smoothed_train_loss": moving_average(
                        train_losses, window=10
                    )[-1],
                    "val_loss": val_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                }
                history.append(row)
                writer.write(row)
                print(json.dumps(row, sort_keys=True), flush=True)

                if global_step % config.training.save_steps == 0:
                    save_checkpoint(
                        output_dir,
                        step=global_step,
                        modules=modules,
                        dataset_statistics=datasets[
                            "train"
                        ].dataset_statistics,
                        keep=config.training.save_total_limit,
                    )
                if global_step >= max_steps:
                    break

    final_test_loss = evaluate(
        modules, dataloaders["test"], device=device
    )
    save_loss_curve(output_dir / "loss_curve.png", history)
    manifest.update(
        {
            "status": "complete",
            "finished_at_unix": time.time(),
            "elapsed_seconds": time.monotonic() - started,
            "final_train_loss": history[-1]["train_loss"],
            "final_smoothed_train_loss": history[-1][
                "smoothed_train_loss"
            ],
            "final_test_loss": final_test_loss,
            "peak_gpu_memory_gib": round(
                torch.cuda.max_memory_allocated() / 1024**3, 2
            ),
            "artifacts": {
                "metrics_jsonl": "metrics.jsonl",
                "metrics_csv": "metrics.csv",
                "loss_curve": "loss_curve.png",
                "final_checkpoint": f"checkpoint-{global_step:06d}",
            },
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return manifest
