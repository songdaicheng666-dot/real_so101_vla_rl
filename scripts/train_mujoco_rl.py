"""Train a project-native PPO or GRPO baseline on a configured SO101 task."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="RL YAML configuration")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="resume from an update-boundary RL checkpoint",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run two short real collection/update cycles",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from real_so101_vla_rl.rl import load_rl_config, smoke_config
        from real_so101_vla_rl.rl.trainer import train
    except ModuleNotFoundError as exc:
        if exc.name in {"gymnasium", "mujoco", "numpy", "torch", "yaml"}:
            raise SystemExit(
                "RL dependencies are missing. Install them with: pip install -e '.[rl]'"
            ) from None
        raise

    config = load_rl_config(args.config)
    if args.smoke:
        config = smoke_config(config)
    run_directory = train(config, checkpoint=args.checkpoint)
    print(f"Training complete: {run_directory}")


if __name__ == "__main__":
    main()
