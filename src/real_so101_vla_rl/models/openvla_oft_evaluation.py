"""Enhanced offline evaluation and acceptance checks for OpenVLA-OFT SFT."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from real_so101_vla_rl.data.normalization import FeatureStats
from real_so101_vla_rl.data.openvla_oft import OpenVLABatchCollator
from real_so101_vla_rl.data.schema import (
    ACTION_CHUNK_SIZE,
    ACTION_DIM,
    JOINT_NAMES,
    JOINT_UNITS,
    TASK_KEY,
    AtomicTask,
    CubeColor,
)
from real_so101_vla_rl.models.openvla_oft_training import (
    create_model_datasets,
    forward_continuous_action_loss,
    load_openvla_oft_modules,
)
from real_so101_vla_rl.models.sft_config import SFTConfig


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    """Hash relative paths and bytes for every regular file in a directory."""

    root = Path(path)
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Cannot hash an empty directory: {root}")
    digest = hashlib.sha256()
    for file_path in files:
        digest.update(file_path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def load_metric_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid metrics JSON on line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise TypeError(f"Metrics line {line_number} must be an object")
            rows.append(row)
    if not rows:
        raise ValueError("metrics.jsonl contains no rows")
    return rows


def select_best_checkpoint(
    rows: Sequence[Mapping[str, Any]], run_dir: str | Path
) -> tuple[int, float, Path]:
    """Choose the lowest finite validation loss, breaking ties by step."""

    candidates = []
    for row in rows:
        value = row.get("val_loss")
        if value is None:
            continue
        step = int(row["step"])
        loss = float(value)
        if not math.isfinite(loss):
            continue
        checkpoint = Path(run_dir) / f"checkpoint-{step:06d}"
        if checkpoint.is_dir():
            candidates.append((loss, step, checkpoint))
    if not candidates:
        raise FileNotFoundError(
            "No finite validation metric has a matching checkpoint"
        )
    loss, step, checkpoint = min(candidates, key=lambda item: (item[0], item[1]))
    return step, loss, checkpoint


def _metric(sum_value: float, count: int) -> dict[str, float | int | None]:
    return {"mae": sum_value / count if count else None, "count": count}


class EvaluationAccumulator:
    """Accumulate exact masked MAE sums for all requested breakdowns."""

    def __init__(self, action_stats: FeatureStats) -> None:
        self._physical_scale = torch.tensor(
            [
                0.5 * (high - low + 1e-8)
                for low, high in zip(
                    action_stats.q01, action_stats.q99, strict=True
                )
            ],
            dtype=torch.float64,
        )
        colors = tuple(color.value for color in CubeColor)
        self._overall = [0.0, 0]
        self._color = {color: [0.0, 0] for color in colors}
        self._joint = {name: [0.0, 0] for name in JOINT_NAMES}
        self._horizon = {
            str(index): [0.0, 0] for index in range(ACTION_CHUNK_SIZE)
        }
        self._physical_joint = {name: [0.0, 0] for name in JOINT_NAMES}
        self._physical_color_joint = {
            color: {name: [0.0, 0] for name in JOINT_NAMES}
            for color in colors
        }

    @staticmethod
    def _add(counter: list[float | int], values: torch.Tensor) -> None:
        counter[0] = float(counter[0]) + float(values.sum())
        counter[1] = int(counter[1]) + values.numel()

    def add_batch(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        padding: torch.Tensor,
        tasks: Sequence[str],
    ) -> None:
        expected = (len(tasks), ACTION_CHUNK_SIZE, ACTION_DIM)
        if tuple(predicted.shape) != expected or tuple(target.shape) != expected:
            raise ValueError(
                f"Predicted and target actions must have shape {expected}"
            )
        if tuple(padding.shape) != expected[:2] or padding.dtype is not torch.bool:
            raise ValueError(
                f"Padding mask must be bool with shape {expected[:2]}"
            )
        error = (predicted.detach().float().cpu() - target.float().cpu()).abs()
        if not torch.isfinite(error).all():
            raise RuntimeError("Evaluation produced non-finite action errors")
        valid = ~padding.detach().cpu()
        physical = error.to(torch.float64) * self._physical_scale.view(1, 1, -1)

        self._add(self._overall, error[valid])
        for horizon in range(ACTION_CHUNK_SIZE):
            self._add(
                self._horizon[str(horizon)],
                error[:, horizon][valid[:, horizon]],
            )
        for joint, name in enumerate(JOINT_NAMES):
            self._add(self._joint[name], error[:, :, joint][valid])
            self._add(self._physical_joint[name], physical[:, :, joint][valid])

        for batch_index, task in enumerate(tasks):
            color = AtomicTask.from_instruction(task).target_color.value
            sample_valid = valid[batch_index]
            self._add(self._color[color], error[batch_index][sample_valid])
            for joint, name in enumerate(JOINT_NAMES):
                self._add(
                    self._physical_color_joint[color][name],
                    physical[batch_index, :, joint][sample_valid],
                )

    def report(self) -> dict[str, Any]:
        return {
            "normalized": {
                "overall": _metric(float(self._overall[0]), int(self._overall[1])),
                "by_color": {
                    key: _metric(float(value[0]), int(value[1]))
                    for key, value in self._color.items()
                },
                "by_joint": {
                    key: _metric(float(value[0]), int(value[1]))
                    for key, value in self._joint.items()
                },
                "by_horizon": {
                    key: _metric(float(value[0]), int(value[1]))
                    for key, value in self._horizon.items()
                },
            },
            "physical": {
                "by_joint": {
                    name: {
                        **_metric(
                            float(self._physical_joint[name][0]),
                            int(self._physical_joint[name][1]),
                        ),
                        "unit": unit,
                    }
                    for name, unit in zip(JOINT_NAMES, JOINT_UNITS, strict=True)
                },
                "by_color_and_joint": {
                    color: {
                        name: {
                            **_metric(float(values[0]), int(values[1])),
                            "unit": unit,
                        }
                        for (name, values), unit in zip(
                            joints.items(), JOINT_UNITS, strict=True
                        )
                    }
                    for color, joints in self._physical_color_joint.items()
                },
            },
        }


class EvaluationDataset(Dataset):
    """Add task and episode provenance without changing the trainer dataset."""

    def __init__(self, model_dataset: Any) -> None:
        self.model_dataset = model_dataset

    def __len__(self) -> int:
        return len(self.model_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = self.model_dataset.source_dataset[index]
        transformed = self.model_dataset.transform(source)
        episode_index = torch.as_tensor(source["episode_index"]).reshape(-1)
        if episode_index.numel() != 1:
            raise ValueError("episode_index must contain exactly one value")
        return {
            **transformed,
            "evaluation_task": source[TASK_KEY],
            "evaluation_episode_index": int(episode_index[0]),
        }


@dataclass(frozen=True, slots=True)
class EvaluationCollator:
    model_collator: OpenVLABatchCollator

    def __call__(self, instances: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        batch = self.model_collator(instances)
        batch["evaluation_tasks"] = [
            str(instance["evaluation_task"]) for instance in instances
        ]
        batch["evaluation_episode_indices"] = [
            int(instance["evaluation_episode_index"]) for instance in instances
        ]
        return batch


def _finite_metric(entry: Mapping[str, Any]) -> bool:
    value = entry.get("mae")
    count = entry.get("count")
    return (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and isinstance(count, int)
        and count > 0
    )


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_acceptance_report(
    *,
    config: SFTConfig,
    run_dir: Path,
    metrics: Sequence[Mapping[str, Any]],
    evaluation_report: Mapping[str, Any],
    external_logs: Sequence[Path] = (),
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, details: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    max_steps = config.training.max_steps
    eval_steps = config.training.eval_steps
    save_steps = config.training.save_steps
    if max_steps is None or eval_steps is None or save_steps is None:
        raise ValueError("Acceptance requires a training-ready configuration")

    steps = [int(row["step"]) for row in metrics]
    check(
        "training_steps_contiguous",
        steps == list(range(1, max_steps + 1)),
        {"expected": max_steps, "actual": len(steps)},
    )
    metric_fields = (
        "train_loss",
        "smoothed_train_loss",
        "learning_rate",
        "grad_norm",
    )
    finite = all(
        all(math.isfinite(float(row[field])) for field in metric_fields)
        and (
            row.get("val_loss") is None
            or math.isfinite(float(row["val_loss"]))
        )
        for row in metrics
    )
    check("training_metrics_finite", finite, {"rows": len(metrics)})

    expected_val_steps = list(range(eval_steps, max_steps + 1, eval_steps))
    actual_val_steps = [
        int(row["step"]) for row in metrics if row.get("val_loss") is not None
    ]
    check(
        "validation_schedule_complete",
        actual_val_steps == expected_val_steps,
        {"expected": expected_val_steps, "actual": actual_val_steps},
    )
    expected_checkpoint_steps = list(range(save_steps, max_steps + 1, save_steps))
    actual_checkpoint_steps = sorted(
        int(path.name.removeprefix("checkpoint-"))
        for path in run_dir.glob("checkpoint-*")
        if path.is_dir()
    )
    check(
        "checkpoints_complete",
        actual_checkpoint_steps == expected_checkpoint_steps,
        {"expected": expected_checkpoint_steps, "actual": actual_checkpoint_steps},
    )

    first = sum(float(row["train_loss"]) for row in metrics[:100]) / 100
    recent = sum(float(row["train_loss"]) for row in metrics[-100:]) / 100
    check(
        "training_loss_improved",
        recent < first,
        {"first_100_mean": first, "last_100_mean": recent},
    )

    normalized = evaluation_report["metrics"]["normalized"]
    physical = evaluation_report["metrics"]["physical"]
    detailed_entries = [normalized["overall"]]
    detailed_entries.extend(normalized["by_color"].values())
    detailed_entries.extend(normalized["by_joint"].values())
    detailed_entries.extend(normalized["by_horizon"].values())
    detailed_entries.extend(physical["by_joint"].values())
    for joints in physical["by_color_and_joint"].values():
        detailed_entries.extend(joints.values())
    expected_colors = {color.value for color in CubeColor}
    expected_horizons = {str(index) for index in range(ACTION_CHUNK_SIZE)}
    complete = (
        set(normalized["by_color"]) == expected_colors
        and set(normalized["by_joint"]) == set(JOINT_NAMES)
        and set(normalized["by_horizon"]) == expected_horizons
        and set(physical["by_color_and_joint"]) == expected_colors
        and all(_finite_metric(entry) for entry in detailed_entries)
    )
    check("enhanced_evaluation_complete", complete, {"entries": len(detailed_entries)})
    check(
        "checkpoint_preflight_shape",
        evaluation_report["prediction_shapes"] == [[2, 8, 6]],
        {"actual": evaluation_report["prediction_shapes"]},
    )
    check(
        "checkpoint_statistics_match",
        evaluation_report["checkpoint_statistics_match"] is True,
        {"checkpoint": evaluation_report["best_checkpoint"]["path"]},
    )

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_sizes = {
        split: len(value["episodes"])
        for split, value in manifest["dataset"].items()
    }
    check(
        "dataset_split_contract",
        split_sizes == {"train": 80, "val": 8, "test": 12},
        split_sizes,
    )
    provenance = manifest.get("dataset_artifact", {})
    check(
        "source_and_data_fingerprints_present",
        all(
            isinstance(value, str) and len(value) == 64
            for value in (
                manifest.get("source_tree_sha256"),
                manifest.get("source_archive_sha256"),
                provenance.get("sha256"),
            )
        ),
        {
            "source_tree_sha256": manifest.get("source_tree_sha256"),
            "source_archive_sha256": manifest.get("source_archive_sha256"),
            "dataset_sha256": provenance.get("sha256"),
        },
    )

    required = [
        run_dir / "config.yaml",
        manifest_path,
        run_dir / "metrics.jsonl",
        run_dir / "metrics.csv",
        run_dir / "loss_curve.png",
        run_dir / "evaluation_report.json",
        *external_logs,
    ]
    artifact_hashes = {
        path.name: _artifact_record(path) for path in required if path.is_file()
    }
    missing = [str(path) for path in required if not path.is_file()]
    check("lightweight_artifacts_complete", not missing, {"missing": missing})

    passed = all(item["passed"] for item in checks)
    return {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "generated_at": datetime.now(UTC).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "checks": checks,
        "artifact_sha256": artifact_hashes,
        "checkpoint_sha256_algorithm": "relative_path_nul_content_nul_v1",
        "best_checkpoint": evaluation_report["best_checkpoint"],
        "final_checkpoint": evaluation_report["final_checkpoint"],
    }


def run_enhanced_evaluation(
    config: SFTConfig,
    *,
    run_dir: str | Path,
    cache_dir: str | Path | None,
    local_files_only: bool,
    external_logs: Sequence[str | Path] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate the best-val checkpoint and write evaluation/acceptance reports."""

    config.validate_training_ready()
    if not torch.cuda.is_available():
        raise RuntimeError("Enhanced OpenVLA-OFT evaluation requires a CUDA GPU")
    os.environ.setdefault("OPENVLA_ROBOT_PLATFORM", "SO101")
    torch.manual_seed(config.training.seed)
    torch.cuda.manual_seed_all(config.training.seed)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    run_dir = Path(run_dir).expanduser().resolve()
    metrics_path = run_dir / "metrics.jsonl"
    rows = load_metric_rows(metrics_path)
    best_step, best_val_loss, best_checkpoint = select_best_checkpoint(rows, run_dir)
    max_steps = config.training.max_steps
    if max_steps is None:
        raise ValueError("training.max_steps is required")
    final_checkpoint = run_dir / f"checkpoint-{max_steps:06d}"
    if not final_checkpoint.is_dir():
        raise FileNotFoundError(f"Final checkpoint is missing: {final_checkpoint}")

    modules = load_openvla_oft_modules(
        config,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        device=device,
        checkpoint=best_checkpoint,
    )
    datasets = create_model_datasets(config, processor=modules.processor)
    expected_statistics = datasets["train"].dataset_statistics
    saved_statistics = json.loads(
        (best_checkpoint / "dataset_statistics.json").read_text(encoding="utf-8")
    )
    if saved_statistics != expected_statistics:
        raise ValueError(
            "Best checkpoint dataset_statistics.json does not match the train split"
        )

    batch_size = config.training.per_device_train_batch_size
    workers = config.training.num_workers
    if batch_size is None or workers is None:
        raise ValueError("Evaluation batch size and num_workers are required")
    test_dataset = EvaluationDataset(datasets["test"])
    collator = EvaluationCollator(
        OpenVLABatchCollator(
            model_max_length=modules.processor.tokenizer.model_max_length,
            pad_token_id=modules.processor.tokenizer.pad_token_id,
        )
    )
    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=workers,
        pin_memory=True,
    )
    accumulator = EvaluationAccumulator(
        datasets["test"].loaded_split.metadata.normalization.action
    )
    prediction_shapes: set[tuple[int, ...]] = set()
    evaluated_episode_indices: set[int] = set()
    started = time.monotonic()
    modules.eval()
    with torch.inference_mode():
        for batch in loader:
            _, predicted = forward_continuous_action_loss(
                modules, batch, device=device
            )
            prediction_shapes.add(tuple(predicted.shape))
            evaluated_episode_indices.update(batch["evaluation_episode_indices"])
            accumulator.add_batch(
                predicted,
                batch["actions"],
                batch["action_is_pad"],
                batch["evaluation_tasks"],
            )

    best_sha = sha256_tree(best_checkpoint)
    final_sha = (
        best_sha if final_checkpoint == best_checkpoint else sha256_tree(final_checkpoint)
    )
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "generated_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "run_dir": str(run_dir),
        "best_checkpoint": {
            "step": best_step,
            "val_loss": best_val_loss,
            "path": str(best_checkpoint),
            "tree_sha256": best_sha,
        },
        "final_checkpoint": {
            "step": max_steps,
            "path": str(final_checkpoint),
            "tree_sha256": final_sha,
            "same_as_best": final_checkpoint == best_checkpoint,
        },
        "checkpoint_sha256_algorithm": "relative_path_nul_content_nul_v1",
        "checkpoint_statistics_match": True,
        "prediction_shapes": [list(shape) for shape in sorted(prediction_shapes)],
        "test_split": {
            "frames": len(test_dataset),
            "episodes": sorted(evaluated_episode_indices),
            "expected_episodes": list(
                datasets["test"].loaded_split.metadata.episode_indices
            ),
        },
        "provenance": {
            "metrics_sha256": sha256_file(metrics_path),
            "config_sha256": sha256_file(run_dir / "config.yaml"),
            "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
            "source_tree_sha256": manifest.get("source_tree_sha256"),
            "source_archive_sha256": manifest.get("source_archive_sha256"),
            "dataset_artifact": manifest.get("dataset_artifact"),
        },
        "metrics": accumulator.report(),
    }
    evaluation_path = run_dir / "evaluation_report.json"
    evaluation_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    acceptance = build_acceptance_report(
        config=config,
        run_dir=run_dir,
        metrics=rows,
        evaluation_report=report,
        external_logs=tuple(Path(path) for path in external_logs),
    )
    (run_dir / "acceptance_report.json").write_text(
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report, acceptance
