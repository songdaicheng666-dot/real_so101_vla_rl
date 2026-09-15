"""Update-boundary checkpoint persistence."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from real_so101_vla_rl.envs.base import TaskEnvironment
from real_so101_vla_rl.rl.algorithms import RLAlgorithm
from real_so101_vla_rl.rl.config import RLConfig


def _rng_state(environments: list[TaskEnvironment]) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "environment_numpy": [env.np_random.bit_generator.state for env in environments],
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def save_checkpoint(
    path: str | Path,
    *,
    update: int,
    config: RLConfig,
    algorithm: RLAlgorithm,
    environments: list[TaskEnvironment],
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "update": update,
        "config": config.to_dict(),
        "policy": algorithm.policy.state_dict(),
        "optimizer": algorithm.optimizer.state_dict(),
        "rng": _rng_state(environments),
    }
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint_path)
    return checkpoint_path


def load_checkpoint_payload(
    path: str | Path,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported RL checkpoint format")
    return payload


def restore_training_checkpoint(
    path: str | Path,
    *,
    algorithm: RLAlgorithm,
    environments: list[TaskEnvironment],
    device: torch.device,
    config: RLConfig,
) -> int:
    payload = load_checkpoint_payload(path, map_location=device)
    saved_config = payload["config"]
    current_config = config.to_dict()
    for section in ("environment", "reward", "policy", "collection", "algorithm"):
        if saved_config[section] != current_config[section]:
            raise ValueError(f"checkpoint {section} configuration does not match")
    for field in ("seed", "route", "source"):
        if saved_config["training"][field] != current_config["training"][field]:
            raise ValueError(f"checkpoint training.{field} does not match")
    algorithm.policy.load_state_dict(payload["policy"])
    algorithm.optimizer.load_state_dict(payload["optimizer"])
    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in rng:
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
    environment_states = rng.get("environment_numpy", [])
    if len(environment_states) != len(environments):
        raise ValueError("checkpoint environment count does not match configuration")
    for index, environment in enumerate(environments):
        environment.np_random.bit_generator.state = environment_states[index]
    return int(payload["update"])
