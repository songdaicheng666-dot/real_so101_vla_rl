"""Loss functions shared by OpenVLA-OFT training code."""

from __future__ import annotations

import torch

from real_so101_vla_rl.data.schema import ACTION_CHUNK_SIZE, ACTION_DIM


def masked_action_l1(
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
    action_is_pad: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute error over real action scalars only."""

    if predicted_actions.shape != target_actions.shape:
        raise ValueError(
            "predicted_actions and target_actions must have identical shapes, "
            f"got {tuple(predicted_actions.shape)} and {tuple(target_actions.shape)}"
        )
    if predicted_actions.ndim != 3 or predicted_actions.shape[1:] != (
        ACTION_CHUNK_SIZE,
        ACTION_DIM,
    ):
        raise ValueError(
            "Action tensors must have shape "
            f"[B, {ACTION_CHUNK_SIZE}, {ACTION_DIM}], got {tuple(predicted_actions.shape)}"
        )
    if predicted_actions.shape[0] == 0:
        raise ValueError("Action tensors must contain at least one sample")
    if (
        not predicted_actions.is_floating_point()
        or not target_actions.is_floating_point()
    ):
        raise TypeError(
            "predicted_actions and target_actions must be floating-point tensors"
        )
    if action_is_pad.shape != predicted_actions.shape[:2]:
        raise ValueError(
            f"action_is_pad must have shape {tuple(predicted_actions.shape[:2])}, "
            f"got {tuple(action_is_pad.shape)}"
        )
    if action_is_pad.dtype is not torch.bool:
        raise TypeError("action_is_pad must use torch.bool")
    if bool(action_is_pad[:, 0].any()) or bool(action_is_pad.all(dim=1).any()):
        raise ValueError("Every sample must contain a valid current action")

    valid = (~action_is_pad).unsqueeze(-1).expand_as(target_actions)
    return torch.abs(predicted_actions - target_actions).masked_select(valid).mean()
