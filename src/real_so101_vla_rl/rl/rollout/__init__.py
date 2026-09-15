"""Simulation and real-robot trajectory collection."""

from .batches import CollectionResult, TrajectoryBatch
from .collectors import GRPOCollector, PPOCollector

__all__ = ["CollectionResult", "GRPOCollector", "PPOCollector", "TrajectoryBatch"]
