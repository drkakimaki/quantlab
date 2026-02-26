from __future__ import annotations

"""Shared helpers for reporting modules.

Policy: keep this module *small* and dependency-light.
- No matplotlib.
- Pure formatting helpers + small dataclasses shared by multiple report generators.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PeriodRow:
    period: str
    pnl: float
    vs_bh: float
    max_drawdown: float
    sharpe: float
    sharpe_ci_lo: float
    sharpe_ci_hi: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    exposure: float
    n_trades: int
    start: str
    end: str


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return float("nan")


def fmt_ts(ts: pd.Timestamp) -> str:
    """Compact timestamp for report tables.

    Prefer readability over full ISO. Drop timezone suffix like "+00:00".
    """
    try:
        ts = pd.to_datetime(ts)
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert("UTC")
        return ts.strftime("%Y-%m-%d %H:%M")
    except Exception:  # noqa: BLE001
        return str(ts)
