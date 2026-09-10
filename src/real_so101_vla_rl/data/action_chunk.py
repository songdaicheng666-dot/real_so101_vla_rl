"""Episode-safe action window construction."""

from __future__ import annotations

from collections.abc import Sequence

from .schema import ACTION_CHUNK_SIZE, DEFAULT_FPS, validate_joint_vector


def action_delta_timestamps(
    *, fps: int = DEFAULT_FPS, chunk_size: int = ACTION_CHUNK_SIZE
) -> dict[str, list[float]]:
    """Return LeRobot delta timestamps for ``[a_t, ..., a_t+K-1]``."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return {"action": [index / fps for index in range(chunk_size)]}


def build_action_chunk(
    episode_actions: Sequence[Sequence[float]],
    start_index: int,
    *,
    chunk_size: int = ACTION_CHUNK_SIZE,
) -> tuple[tuple[tuple[float, ...], ...], tuple[bool, ...]]:
    """Build a future action chunk without crossing an episode boundary.

    Missing tail positions repeat the final action and are marked as padding,
    matching LeRobot's ``action_is_pad`` behavior.
    """

    if not episode_actions:
        raise ValueError("episode_actions must not be empty")
    if not 0 <= start_index < len(episode_actions):
        raise IndexError(f"start_index={start_index} is outside the episode")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    actions = tuple(
        validate_joint_vector(action, field_name=f"episode_actions[{index}]")
        for index, action in enumerate(episode_actions)
    )
    final_index = len(actions) - 1
    chunk = []
    is_pad = []
    for offset in range(chunk_size):
        requested_index = start_index + offset
        padded = requested_index > final_index
        chunk.append(actions[min(requested_index, final_index)])
        is_pad.append(padded)
    return tuple(chunk), tuple(is_pad)
