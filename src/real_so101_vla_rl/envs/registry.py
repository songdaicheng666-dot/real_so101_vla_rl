"""Task environment registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from real_so101_vla_rl.envs.base import TaskEnvironment

EnvironmentFactory = Callable[..., TaskEnvironment]
_ENVIRONMENTS: dict[str, EnvironmentFactory] = {}


def register_environment(name: str, factory: EnvironmentFactory) -> None:
    if not name or name in _ENVIRONMENTS:
        raise ValueError(f"environment is already registered or invalid: {name!r}")
    _ENVIRONMENTS[name] = factory


def make_environment(name: str, **kwargs: Any) -> TaskEnvironment:
    try:
        factory = _ENVIRONMENTS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown environment {name!r}; available: {sorted(_ENVIRONMENTS)}"
        ) from exc
    return factory(**kwargs)


def available_environments() -> tuple[str, ...]:
    return tuple(sorted(_ENVIRONMENTS))


def _register_builtins() -> None:
    from real_so101_vla_rl.envs.basic_t0 import SO101BasicT0Env

    register_environment(SO101BasicT0Env.task_name, SO101BasicT0Env)


_register_builtins()
