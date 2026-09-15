"""Deterministic evaluation on a reproducible reset suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from real_so101_vla_rl.envs.components import BATTERY_COLORS
from real_so101_vla_rl.rl.config import RLConfig
from real_so101_vla_rl.rl.policies import MLPChunkPolicy

from .factory import make_task_environment


def evaluate_policy(
    policy: MLPChunkPolicy,
    config: RLConfig,
    *,
    episodes: int,
    device: torch.device,
    video_path: str | Path | None = None,
) -> dict[str, float]:
    if episodes <= 0:
        raise ValueError("evaluation episodes must be positive")
    environment = make_task_environment(
        config,
        seed=config.training.seed + 100_000,
        render_mode="rgb_array" if video_path is not None else None,
    )
    returns: list[float] = []
    lengths: list[int] = []
    successes = 0
    failures: dict[str, int] = {}
    frames: list[np.ndarray] = []
    was_training = policy.training
    policy.eval()
    try:
        for episode_index in range(episodes):
            target_color = BATTERY_COLORS[episode_index % len(BATTERY_COLORS)]
            snapshot = environment.sample_reset_snapshot(target_color)
            observation, _ = environment.reset(options={"snapshot": snapshot})
            total_reward = 0.0
            terminated = truncated = False
            final_info: dict = {}
            if video_path is not None and episode_index == 0:
                frames.append(environment.render())
            while not (terminated or truncated):
                observation_tensor = torch.as_tensor(
                    observation[None], dtype=torch.float32, device=device
                )
                with torch.no_grad():
                    action = policy.predict(observation_tensor)[0].cpu().numpy()
                observation, reward, terminated, truncated, final_info = environment.step(
                    action
                )
                total_reward += reward
                if video_path is not None and episode_index == 0:
                    frames.append(environment.render())
            returns.append(total_reward)
            lengths.append(int(final_info["episode_chunks"]))
            successes += int(final_info["success"])
            if final_info["failure_type"]:
                failure = str(final_info["failure_type"])
                failures[failure] = failures.get(failure, 0) + 1
    finally:
        environment.close()
        policy.train(was_training)

    if video_path is not None and frames:
        from PIL import Image

        destination = Path(video_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(
            destination,
            save_all=True,
            append_images=images[1:],
            duration=round(1000 * config.environment.action_chunk_size / config.environment.control_hz),
            loop=0,
        )

    metrics = {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "success_rate": successes / episodes,
        "episode_chunks_mean": float(np.mean(lengths)),
    }
    for name, count in failures.items():
        metrics[f"failures/{name}"] = count / episodes
    return metrics
