from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class SignalGate(Protocol):
    """Protocol for signal modification gates."""

    @property
    def name(self) -> str:
        """Gate name for logging/debugging."""
        ...

    def __call__(
        self,
        positions: pd.Series,
        prices: pd.Series,
        context: dict | None = None,
    ) -> pd.Series:
        """Apply gate to positions."""
        ...
