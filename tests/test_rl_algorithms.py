from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from real_so101_vla_rl.rl import load_rl_config, smoke_config
from real_so101_vla_rl.rl.algorithms import (
    generalized_advantage_estimate,
    group_relative_advantages,
)
from real_so101_vla_rl.rl.policies import MLPChunkPolicy
from real_so101_vla_rl.rl.rollout import TrajectoryBatch


def _batch(
    *,
    rewards: list[float],
    terminated: list[bool],
    truncated: list[bool],
    values: list[float] | None,
    next_values: list[float] | None,
    episode_ids: tuple[int, ...],
    group_ids: tuple[int, ...],
    reset_ids: tuple[str, ...],
    sequence_shape: tuple[int, int] | None,
) -> TrajectoryBatch:
    size = len(rewards)
    return TrajectoryBatch(
        observations=torch.zeros(size, 5),
        actions=torch.zeros(size, 2, 3),
        old_log_probs=torch.zeros(size, 2),
        rewards=torch.tensor(rewards),
        terminated=torch.tensor(terminated),
        truncated=torch.tensor(truncated),
        action_masks=torch.ones(size, 2),
        old_values=None if values is None else torch.tensor(values),
        next_values=None if next_values is None else torch.tensor(next_values),
        advantages=None,
        returns=None,
        episode_ids=episode_ids,
        group_ids=group_ids,
        reset_ids=reset_ids,
        sequence_shape=sequence_shape,
    )


def test_squashed_gaussian_actions_and_recomputed_log_probs_match() -> None:
    policy = MLPChunkPolicy(
        5,
        2,
        3,
        hidden_sizes=(16, 16),
        initial_action=(0.0, 0.2, -0.8),
        has_value=True,
    )
    observations = torch.randn(4, 5)
    sampled = policy.sample(observations)
    evaluated = policy.evaluate_actions(observations, sampled.action_chunk)

    assert sampled.action_chunk.shape == (4, 2, 3)
    assert sampled.log_prob.shape == (4, 2)
    assert sampled.value is not None and sampled.value.shape == (4,)
    assert torch.all(sampled.action_chunk > -1.0)
    assert torch.all(sampled.action_chunk < 1.0)
    assert torch.allclose(evaluated.log_prob, sampled.log_prob.detach(), atol=1e-4)
    expected = torch.tensor((0.0, 0.2, -0.8)).repeat(2, 1)
    assert torch.allclose(policy.predict(torch.zeros(1, 5))[0], expected, atol=1e-5)


def test_grpo_policy_has_no_value_head() -> None:
    policy = MLPChunkPolicy(5, 2, 3, hidden_sizes=(8,), has_value=False)
    output = policy.sample(torch.zeros(2, 5))
    assert output.value is None
    assert policy.value(torch.zeros(2, 5)) is None


def test_gae_distinguishes_termination_from_time_truncation() -> None:
    terminated_batch = _batch(
        rewards=[1.0],
        terminated=[True],
        truncated=[False],
        values=[0.5],
        next_values=[10.0],
        episode_ids=(0,),
        group_ids=(-1,),
        reset_ids=("a",),
        sequence_shape=(1, 1),
    )
    advantage, returns = generalized_advantage_estimate(
        terminated_batch, gamma=0.99, gae_lambda=0.95
    )
    assert advantage == pytest.approx(torch.tensor((0.5,)))
    assert returns == pytest.approx(torch.tensor((1.0,)))

    truncated_batch = replace(
        terminated_batch,
        terminated=torch.tensor((False,)),
        truncated=torch.tensor((True,)),
    )
    advantage, returns = generalized_advantage_estimate(
        truncated_batch, gamma=0.99, gae_lambda=0.95
    )
    assert advantage == pytest.approx(torch.tensor((10.4,)))
    assert returns == pytest.approx(torch.tensor((10.9,)))


def test_group_relative_advantage_uses_complete_trajectory_returns() -> None:
    batch = _batch(
        rewards=[0.5, 0.5, 1.0, 2.0],
        terminated=[False, True, False, True],
        truncated=[False] * 4,
        values=None,
        next_values=None,
        episode_ids=(10, 10, 11, 11),
        group_ids=(3, 3, 3, 3),
        reset_ids=("same",) * 4,
        sequence_shape=None,
    )
    advantages, returns = group_relative_advantages(batch)
    assert advantages == pytest.approx(torch.tensor((-1.0, -1.0, 1.0, 1.0)))
    assert returns == pytest.approx(torch.tensor((1.0, 1.0, 3.0, 3.0)))


def test_group_relative_zero_variance_and_reset_validation() -> None:
    batch = _batch(
        rewards=[1.0, 1.0],
        terminated=[True, True],
        truncated=[False, False],
        values=None,
        next_values=None,
        episode_ids=(0, 1),
        group_ids=(0, 0),
        reset_ids=("same", "same"),
        sequence_shape=None,
    )
    advantages, _ = group_relative_advantages(batch)
    assert advantages.tolist() == [0.0, 0.0]
    with pytest.raises(ValueError, match="reset snapshot"):
        group_relative_advantages(replace(batch, reset_ids=("a", "b")))


@pytest.mark.parametrize("config_name", ["ppo_t0_mlp.yaml", "grpo_t0_mlp.yaml"])
def test_rl_configs_define_fast_smoke_variants(config_name: str) -> None:
    config = load_rl_config(f"configs/rl/{config_name}")
    smoke = smoke_config(config)
    assert smoke.training.total_updates == 2
    assert smoke.environment.max_episode_chunks == 2
    assert smoke.collection.num_envs == (4 if config.algorithm.name == "grpo" else 2)
