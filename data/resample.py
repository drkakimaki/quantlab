from __future__ import annotations

"""Backward-compatible shim.

Historically, this module mixed:
- parquet loading helpers (used by rnd/webui backtests)
- Polars-based resampling utilities (used by data/download.py)

We split these into:
- :mod:`quantlab.data.loaders`    (no Polars)
- :mod:`quantlab.data.resampling` (Polars)

Keep this file to avoid breaking older imports.
"""

from .loaders import load_dukascopy_ohlc, resample_last


def __getattr__(name: str):
    # Lazy import to avoid pulling Polars into the normal backtest import path.
    if name in {
        "load_dukascopy_1s_mid",
        "load_dukascopy_mid_resample_last",
        "resample_dukascopy_1s_to_bars",
        "resample_dukascopy_1s_to_ohlc",
    }:
        from . import resampling as _r

        return getattr(_r, name)

    raise AttributeError(name)


__all__ = [
    "load_dukascopy_ohlc",
    "resample_last",
    "load_dukascopy_1s_mid",
    "load_dukascopy_mid_resample_last",
    "resample_dukascopy_1s_to_bars",
    "resample_dukascopy_1s_to_ohlc",
]
