"""Evaluate and accept a completed SO-101 OpenVLA-OFT SFT run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from real_so101_vla_rl.models import load_sft_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--external-log", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    config_path = Path(args.config) if args.config else run_dir / "config.yaml"
    os.environ.setdefault("OPENVLA_ROBOT_PLATFORM", "SO101")
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    config = load_sft_config(config_path, require_training_ready=True)

    from real_so101_vla_rl.models.openvla_oft_evaluation import (
        run_enhanced_evaluation,
    )

    evaluation, acceptance = run_enhanced_evaluation(
        config,
        run_dir=run_dir,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        external_logs=args.external_log,
    )
    print(
        json.dumps(
            {
                "evaluation_status": evaluation["status"],
                "acceptance_status": acceptance["status"],
                "best_checkpoint": evaluation["best_checkpoint"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
