"""Structural camera interfaces, kept independent from optional SDK imports."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..capture_types import CapturedSample, RGBDFrame


class RGBAdapter(Protocol):
    @property
    def is_connected(self) -> bool: ...

    @property
    def is_broken(self) -> bool: ...

    def connect(self) -> None: ...

    def read(self, timeout_ms: int) -> CapturedSample[np.ndarray]: ...

    def disconnect(self) -> None: ...


class RGBDAdapter(Protocol):
    @property
    def is_connected(self) -> bool: ...

    @property
    def is_broken(self) -> bool: ...

    def connect(self) -> None: ...

    def read(self, timeout_ms: int) -> RGBDFrame: ...

    def disconnect(self) -> None: ...
