"""Algorithm extension interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from torch import nn

from real_so101_vla_rl.rl.rollout import TrajectoryBatch


class RLAlgorithm(ABC):
    collection_strategy: str
    requires_value: bool

    def __init__(self, policy: nn.Module) -> None:
        self.policy = policy

    @abstractmethod
    def prepare_batch(self, batch: TrajectoryBatch) -> TrajectoryBatch:
        """Calculate the algorithm-specific advantages and targets."""

    @abstractmethod
    def update(self, batch: TrajectoryBatch) -> dict[str, float]:
        """Perform one policy update."""

    @property
    @abstractmethod
    def optimizer(self) -> Any:
        """Optimizer state saved in checkpoints."""


AlgorithmFactory = Callable[..., RLAlgorithm]
_ALGORITHMS: dict[str, AlgorithmFactory] = {}


def register_algorithm(name: str) -> Callable[[AlgorithmFactory], AlgorithmFactory]:
    def decorator(factory: AlgorithmFactory) -> AlgorithmFactory:
        if not name or name in _ALGORITHMS:
            raise ValueError(f"algorithm is already registered or invalid: {name!r}")
        _ALGORITHMS[name] = factory
        return factory

    return decorator


def make_algorithm(name: str, **kwargs: Any) -> RLAlgorithm:
    try:
        factory = _ALGORITHMS[name]
    except KeyError as exc:
        raise KeyError(f"unknown algorithm {name!r}; available: {sorted(_ALGORITHMS)}") from exc
    return factory(**kwargs)


def algorithm_requirements(name: str) -> tuple[str, bool]:
    try:
        factory = _ALGORITHMS[name]
    except KeyError as exc:
        raise KeyError(f"unknown algorithm {name!r}; available: {sorted(_ALGORITHMS)}") from exc
    return str(factory.collection_strategy), bool(factory.requires_value)


def available_algorithms() -> tuple[str, ...]:
    return tuple(sorted(_ALGORITHMS))
