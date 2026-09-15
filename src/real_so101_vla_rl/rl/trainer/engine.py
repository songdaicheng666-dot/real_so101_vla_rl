"""End-to-end training orchestration for project-native RL baselines."""

from __future__ import annotations

import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from real_so101_vla_rl.rl.algorithms import (
    algorithm_requirements,
    make_algorithm,
)
from real_so101_vla_rl.rl.config import RLConfig
from real_so101_vla_rl.rl.policies import assert_finite_policy
from real_so101_vla_rl.rl.rollout import GRPOCollector, PPOCollector

from .checkpoint import restore_training_checkpoint, save_checkpoint
from .evaluation import evaluate_policy
from .factory import make_mlp_policy, make_task_environment, resolve_device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _create_run_directory(config: RLConfig) -> Path:
    root = Path(config.training.output_root)
    root.mkdir(parents=True, exist_ok=True)
    base_name = config.training.run_name or (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        + f"_{config.algorithm.name}_{config.environment.name}"
    )
    destination = root / base_name
    suffix = 1
    while destination.exists():
        destination = root / f"{base_name}-{suffix}"
        suffix += 1
    destination.mkdir()
    return destination


def _write_json_line(path: Path, values: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(values, ensure_ascii=False, sort_keys=True) + "\n")


def _collector_for_strategy(
    strategy: str,
    environments: list,
    config: RLConfig,
    device: torch.device,
):
    factories = {
        "fixed_horizon": lambda: PPOCollector(
            environments,
            steps_per_env=config.collection.steps_per_env_per_update,
            device=device,
        ),
        "grouped_full_episode": lambda: GRPOCollector(
            environments,
            group_size=config.collection.group_size,
            device=device,
        ),
    }
    try:
        return factories[strategy]()
    except KeyError as exc:
        raise ValueError(f"unsupported collection strategy: {strategy}") from exc


def train(
    config: RLConfig,
    *,
    checkpoint: str | Path | None = None,
) -> Path:
    """Run training and return the run directory."""

    _seed_everything(config.training.seed)
    device = resolve_device(config.training.device)
    strategy, requires_value = algorithm_requirements(config.algorithm.name)
    environments = [
        make_task_environment(config, seed=config.training.seed + index)
        for index in range(config.collection.num_envs)
    ]
    policy = make_mlp_policy(
        config,
        environments[0],
        has_value=requires_value,
        device=device,
    )
    algorithm = make_algorithm(
        config.algorithm.name,
        policy=policy,
        config=config.algorithm,
    )
    start_update = 0
    if checkpoint is not None:
        start_update = restore_training_checkpoint(
            checkpoint,
            algorithm=algorithm,
            environments=environments,
            device=device,
            config=config,
        )
        run_directory = Path(checkpoint).resolve().parent
    else:
        run_directory = _create_run_directory(config)
    collector = _collector_for_strategy(strategy, environments, config, device)

    (run_directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    (run_directory / "run_metadata.json").write_text(
        json.dumps(
            {
                "algorithm": config.algorithm.name,
                "task": config.environment.name,
                "route": config.training.route,
                "source": config.training.source,
                "device": str(device),
                "start_update": start_update,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics_path = run_directory / "metrics.jsonl"

    try:
        for update in range(start_update + 1, config.training.total_updates + 1):
            start_time = time.perf_counter()
            collection = collector.collect(policy)
            batch = collection.batch.to(device)
            batch = algorithm.prepare_batch(batch)
            optimization_metrics = algorithm.update(batch)
            assert_finite_policy(policy)
            elapsed = max(time.perf_counter() - start_time, 1e-9)
            metrics: dict[str, Any] = {
                "update": update,
                "algorithm": config.algorithm.name,
                "task": config.environment.name,
                "route": config.training.route,
                "source": config.training.source,
                "throughput/control_steps_per_second": float(
                    batch.action_masks.sum().item() / elapsed
                ),
                **collection.metrics,
                **optimization_metrics,
            }
            episodes = collection.metrics.get("episodes", 0.0)
            if episodes:
                metrics["episodes/success_rate"] = (
                    collection.metrics.get("successes", 0.0) / episodes
                )
            if update % config.training.evaluation_interval == 0:
                evaluation = evaluate_policy(
                    policy,
                    config,
                    episodes=config.training.evaluation_episodes,
                    device=device,
                )
                metrics.update({f"evaluation/{name}": value for name, value in evaluation.items()})
            _write_json_line(metrics_path, metrics)
            print(
                f"update={update} algorithm={config.algorithm.name} "
                f"chunks={batch.size} reward={float(batch.rewards.mean()):+.4f} "
                f"throughput={metrics['throughput/control_steps_per_second']:.1f} control_steps/s"
            )
            if (
                update % config.training.checkpoint_interval == 0
                or update == config.training.total_updates
            ):
                save_checkpoint(
                    run_directory / f"checkpoint-{update:06d}.pt",
                    update=update,
                    config=config,
                    algorithm=algorithm,
                    environments=environments,
                )
    finally:
        for environment in environments:
            environment.close()
    return run_directory
