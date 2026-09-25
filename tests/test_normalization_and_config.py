from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from real_so101_vla_rl.data import (
    SplitPolicy,
    compute_normalization_stats,
    load_normalization_stats,
    normalize_q99,
    unnormalize_q99,
    write_normalization_stats,
)
from real_so101_vla_rl.models import load_sft_config

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "sft" / "openvla_oft_lora.yaml"
SYNTHETIC_CONFIG_PATH = (
    Path(__file__).parents[1]
    / "configs"
    / "sft"
    / "openvla_oft_synthetic_overfit.yaml"
)
PILOT_CONFIG_PATH = (
    Path(__file__).parents[1]
    / "configs"
    / "sft"
    / "openvla_oft_real_pilot_lowlight.yaml"
)


def test_sft_config_definition_is_valid_but_runtime_is_explicitly_incomplete() -> None:
    config = load_sft_config(CONFIG_PATH)

    assert config.model.action_dim == 6
    assert config.model.action_chunk_size == 8
    assert config.dataset.action_source == "robot_send_action_return"
    assert config.dataset.revision is None
    assert config.dataset.task_format == "canonical_atomic_transfer_v1"
    assert config.dataset.normalization.unnorm_key == "so101_cube_dual_rgb_v2"
    assert config.dataset.image_keys == (
        "observation.images.overview",
        "observation.images.wrist",
    )
    assert config.model.num_images == 2
    with pytest.raises(ValueError, match="not training-ready"):
        config.validate_training_ready()


def test_sft_config_can_become_training_ready_without_changing_schema() -> None:
    config = load_sft_config(CONFIG_PATH)
    ready_dataset = replace(
        config.dataset,
        repo_id="sdc/so101_cube_dual_rgb_v2",
        root="/datasets/so101_cube_dual_rgb_v2",
    )
    ready_training = replace(
        config.training,
        max_steps=20_000,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_workers=4,
        eval_steps=500,
        save_steps=500,
        output_dir="outputs/so101_cube_dual_rgb_v2",
    )
    ready_config = replace(config, dataset=ready_dataset, training=ready_training)

    ready_config.validate_training_ready()


def test_sft_config_requires_repo_id_even_for_a_local_dataset() -> None:
    config = load_sft_config(CONFIG_PATH)
    local_only = replace(config, dataset=replace(config.dataset, root="/datasets/local"))

    with pytest.raises(ValueError, match="dataset.repo_id"):
        local_only.validate_training_ready()


def test_synthetic_overfit_config_is_training_ready() -> None:
    config = load_sft_config(SYNTHETIC_CONFIG_PATH, require_training_ready=True)

    assert (
        config.dataset.repo_id
        == "local/so101_synthetic_overfit_v2_lossless_depth"
    )
    assert config.training.max_steps == 200
    assert config.training.per_device_train_batch_size == 2
    assert config.training.image_augmentation is False


def test_real_lowlight_pilot_config_is_training_ready() -> None:
    config = load_sft_config(PILOT_CONFIG_PATH, require_training_ready=True)

    assert config.dataset.repo_id == "local/so101_real_pilot_lowlight_v2"
    assert (
        config.dataset.split_policy
        is SplitPolicy.PILOT_SINGLE_TASK_GROUPED_V1
    )
    assert config.dataset.normalization.unnorm_key == "so101_cube_dual_rgb_v2"
    assert config.training.max_steps == 200
    assert config.training.eval_steps == 20
    assert config.training.save_steps == 100
    assert config.training.image_augmentation is False


def test_sft_config_rejects_upstream_seven_dimensional_action_default() -> None:
    config = load_sft_config(CONFIG_PATH)
    wrong_model = replace(config.model, action_dim=7)

    with pytest.raises(ValueError, match="model.action_dim must be 6"):
        replace(config, model=wrong_model).validate_definition()


def test_train_only_normalization_stats_round_trip(tmp_path) -> None:
    samples = [
        {
            "observation.state": [index + offset for offset in range(6)],
            "action": [2 * index + offset for offset in range(6)],
        }
        for index in range(100)
    ]
    stats = compute_normalization_stats(
        samples,
        train_episode_indices=[0, 1, 2],
        unnorm_key="so101_cube_dual_rgb_v2",
    )

    assert stats.observation_state.q01[0] == pytest.approx(0.99)
    assert stats.observation_state.q99[0] == pytest.approx(98.01)
    assert set(stats.to_openvla_dict()["so101_cube_dual_rgb_v2"]) == {
        "observation.state",
        "action",
    }

    normalized = normalize_q99(stats.action.q99, stats.action)
    restored = unnormalize_q99(normalized, stats.action)
    assert normalized == pytest.approx((1, 1, 1, 1, 1, 1), abs=1e-7)
    assert restored == pytest.approx(stats.action.q99, abs=1e-7)

    path = tmp_path / "norm_stats.json"
    write_normalization_stats(path, stats)
    assert load_normalization_stats(path) == stats


def test_sft_config_rejects_schema_v1() -> None:
    config = load_sft_config(CONFIG_PATH)

    with pytest.raises(ValueError, match="schema_version=1"):
        replace(config, schema_version=1).validate_definition()
