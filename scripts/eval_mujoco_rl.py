"""Evaluate a saved state-based SO101 RL checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--episodes", type=int, help="override evaluation episode count")
    parser.add_argument("--device", help="override auto/cpu/cuda from the checkpoint")
    parser.add_argument(
        "--video",
        nargs="?",
        const=Path("evaluation.gif"),
        type=Path,
        help="record the first fixed evaluation episode as a GIF",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.video is not None:
        os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        from real_so101_vla_rl.rl import rl_config_from_dict
        from real_so101_vla_rl.rl.algorithms import algorithm_requirements
        from real_so101_vla_rl.rl.trainer import load_checkpoint_payload
        from real_so101_vla_rl.rl.trainer.evaluation import evaluate_policy
        from real_so101_vla_rl.rl.trainer.factory import (
            make_mlp_policy,
            make_task_environment,
            resolve_device,
        )
    except ModuleNotFoundError as exc:
        if exc.name in {"gymnasium", "mujoco", "numpy", "PIL", "torch", "yaml"}:
            raise SystemExit(
                "RL dependencies are missing. Install them with: pip install -e '.[rl]'"
            ) from None
        raise

    payload = load_checkpoint_payload(args.checkpoint)
    config = rl_config_from_dict(payload["config"])
    device = resolve_device(args.device or config.training.device)
    _, requires_value = algorithm_requirements(config.algorithm.name)
    environment = make_task_environment(config, seed=config.training.seed)
    try:
        policy = make_mlp_policy(
            config,
            environment,
            has_value=requires_value,
            device=device,
        )
    finally:
        environment.close()
    policy.load_state_dict(payload["policy"])
    policy.eval()
    episodes = args.episodes or config.training.evaluation_episodes
    metrics = evaluate_policy(
        policy,
        config,
        episodes=episodes,
        device=device,
        video_path=args.video,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
