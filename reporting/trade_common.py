from __future__ import annotations

import numpy as np
import pandas as pd


def duration_bin(bars: pd.Series) -> pd.Categorical:
    """Stable duration buckets (in bars) used across trade reports."""
    bins = [-np.inf, 1, 3, 6, 12, 24, 48, 96, 192, np.inf]
    labels = [
        "1",
        "2-3",
        "4-6",
        "7-12",
        "13-24",
        "25-48",
        "49-96",
        "97-192",
        "193+",
    ]
    return pd.cut(bars.astype(float), bins=bins, labels=labels)


def profit_factor(pnl: pd.Series) -> float:
    """Profit factor = sum(wins) / sum(|losses|)."""
    pnl = pnl.astype(float)
    wins = float(pnl[pnl > 0].sum())
    losses = float((-pnl[pnl < 0]).sum())
    if losses <= 0:
        return float("inf") if wins > 0 else float("nan")
    return wins / losses


def agg_trade_table(trades: pd.DataFrame, key: str) -> pd.DataFrame:
    """Aggregate per-trade ledger into a bucket table.

    Expects columns:
      - win (bool)
      - trade_return (float)
      - pnl_net (float)
      - bars (int)
      - costs_total (float) (optional)

    Returns a stable superset of metrics. Callers can subset columns.
    """
    g = trades.groupby(key, dropna=False)

    out = pd.DataFrame(
        {
            "n_trades": g.size(),
            "win_rate": g["win"].mean(),
            "avg_return": g["trade_return"].mean(),
            "median_return": g["trade_return"].median(),
            "sum_pnl": g["pnl_net"].sum(),
            "avg_pnl": g["pnl_net"].mean(),
            "avg_bars": g["bars"].mean(),
        }
    )

    if "costs_total" in trades.columns:
        out["sum_costs"] = g["costs_total"].sum()
        out["avg_costs"] = g["costs_total"].mean()

    # Profit factor per bucket
    pf = []
    for val, sub in trades.groupby(key, dropna=False):
        pf.append((val, profit_factor(sub["pnl_net"])))
    pf_df = pd.DataFrame(pf, columns=[key, "profit_factor"])
    out = out.reset_index().merge(pf_df, on=key, how="left")

    return out.sort_values("sum_pnl", ascending=False)
