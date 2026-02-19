from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _to_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, (pd.DataFrame,)):
        raise TypeError("Expected 1D series-like, got DataFrame")
    return pd.Series(x)


def annualization_factor(freq: str) -> float:
    """Return annualization factor for Sharpe/vol etc.

    Supported freq strings:
    - 'S'            : 1-second
    - 'T' / 'MIN'    : 1-minute
    - '5MIN'         : 5-minute
    - '15MIN'        : 15-minute
    - 'H'            : hourly
    - 'D'            : daily (calendar days)
    - 'B'            : business days
    - 'W'            : weekly
    - 'M'            : monthly

    Notes
    -----
    - For intraday, we annualize using **252 trading days/year** (not 365).
      This matches common finance convention and avoids overstating Sharpe/vol.
    """
    freq = freq.upper()

    # Common convention for FX-like markets (24h, 5d/week): ~260 trading days/year.
    TRADING_DAYS = 260
    table = {
        # Intraday: assume 24h * trading days (good approximation for FX/CFDs; weekends largely absent).
        "S": 60 * 60 * 24 * TRADING_DAYS,
        "T": 60 * 24 * TRADING_DAYS,
        "MIN": 60 * 24 * TRADING_DAYS,
        "5MIN": (60 // 5) * 24 * TRADING_DAYS,
        "15MIN": (60 // 15) * 24 * TRADING_DAYS,
        "H": 24 * TRADING_DAYS,

        # Daily calendars vs business days
        "D": 365,
        "B": 252,
        "W": 52,
        "M": 12,
    }
    if freq not in table:
        raise KeyError(f"Unknown freq {freq!r}. Known: {sorted(table)}")
    return table[freq]


def sharpe(returns, freq: str = "B") -> float:
    r = _to_series(returns).dropna()
    if r.empty:
        return float("nan")
    af = annualization_factor(freq)
    mu = r.mean()
    sig = r.std(ddof=1)
    if sig == 0:
        return float("nan")
    return float(np.sqrt(af) * mu / sig)


def volatility(returns, freq: str = "B") -> float:
    r = _to_series(returns).dropna()
    if r.empty:
        return float("nan")
    af = annualization_factor(freq)
    return float(np.sqrt(af) * r.std(ddof=1))


def max_drawdown(equity) -> float:
    e = _to_series(equity).dropna()
    if e.empty:
        return float("nan")
    peak = e.cummax()
    dd = e / peak - 1.0
    return float(dd.min())


def cagr(equity, freq: str = "B") -> float:
    e = _to_series(equity).dropna()
    if len(e) < 2:
        return float("nan")
    af = annualization_factor(freq)
    n_periods = len(e) - 1
    years = n_periods / af
    if years <= 0:
        return float("nan")
    return float(e.iloc[-1] ** (1.0 / years) - 1.0)


@dataclass(frozen=True)
class PerformanceSummary:
    cagr: float
    vol: float
    sharpe: float
    max_drawdown: float


def performance_summary(returns, equity, freq: str = "B") -> PerformanceSummary:
    return PerformanceSummary(
        cagr=cagr(equity, freq=freq),
        vol=volatility(returns, freq=freq),
        sharpe=sharpe(returns, freq=freq),
        max_drawdown=max_drawdown(equity),
    )
