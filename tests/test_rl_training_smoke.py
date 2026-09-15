from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")

from real_so101_vla_rl.rl import load_rl_config, smoke_config
from real_so101_vla_rl.rl.trainer import load_checkpoint_payload, train


@pytest.mark.parametrize("algorithm_name", ["ppo", "grpo"])
def test_two_update_training_and_checkpoint_round_trip(tmp_path, algorithm_name: str) -> None:
    config = smoke_config(load_rl_config(f"configs/rl/{algorithm_name}_t0_mlp.yaml"))
    config = replace(
        config,
        training=replace(
            config.training,
            device="cpu",
            output_root=str(tmp_path),
            run_name=f"test-{algorithm_name}",
        ),
    )
    run_directory = train(config)

    metrics = [
        json.loads(line)
        for line in (run_directory / "metrics.jsonl").read_text().splitlines()
    ]
    assert [entry["update"] for entry in metrics] == [1, 2]
    assert all(
        math.isfinite(value)
        for entry in metrics
        for value in entry.values()
        if isinstance(value, float)
    )
    checkpoint = run_directory / "checkpoint-000002.pt"
    payload = load_checkpoint_payload(checkpoint)
    assert payload["update"] == 2
    assert payload["config"]["algorithm"]["name"] == algorithm_name
    assert all(torch.isfinite(value).all() for value in payload["policy"].values())
    assert torch.count_nonzero(payload["policy"]["actor_mean.weight"]) > 0

    resumed_config = replace(
        config,
        training=replace(config.training, total_updates=3),
    )
    resumed_directory = train(resumed_config, checkpoint=checkpoint)
    assert resumed_directory == checkpoint.resolve().parent
    assert (run_directory / "checkpoint-000003.pt").is_file()
