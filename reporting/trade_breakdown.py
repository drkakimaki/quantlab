from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..engine.trades import extract_trade_log
from .trade_common import duration_bin, agg_trade_table


@dataclass(frozen=True)
class TradeBreakdownPaths:
    out_dir: Path
    trades_csv: Path
    by_month_csv: Path
    by_side_csv: Path
    by_duration_csv: Path
    summary_md: Path


def _duration_bin(bars: pd.Series) -> pd.Categorical:
    # Backward-compatible alias.
    return duration_bin(bars)


def build_trade_ledger(
    bt: pd.DataFrame,
    *,
    pos_col: str = "position",
    returns_col: str = "returns_net",
    equity_col: str = "equity",
    costs_col: str = "costs",
) -> pd.DataFrame:
    """Create an enriched per-trade ledger from a backtest dataframe.

    Minimal on purpose: uses only columns that already exist in the engine output.

    Returns a DataFrame with one row per trade and columns:
      trade_id, entry_time, exit_time, bars, side,
      entry_equity, exit_equity, pnl_net, trade_return,
      costs_total, win
    """
    if bt is None or len(bt) == 0:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "entry_time",
                "exit_time",
                "bars",
                "side",
                "entry_equity",
                "exit_equity",
                "pnl_net",
                "trade_return",
                "costs_total",
                "win",
                "open",
            ]
        )

    bt = bt.copy()
    bt.index = pd.to_datetime(bt.index)
    bt = bt.sort_index()

    for c in (pos_col, returns_col, equity_col):
        if c not in bt.columns:
            raise KeyError(f"bt missing required column {c!r}")

    trades = extract_trade_log(bt, pos_col=pos_col, returns_col=returns_col, equity_col=equity_col)
    if trades.empty:
        return trades

    # Side at entry
    pos = bt[pos_col].fillna(0.0).astype(float)
    entry_pos = pos.reindex(pd.to_datetime(trades["entry_time"]).values).to_numpy(dtype=float)
    side = np.where(entry_pos > 0, "long", np.where(entry_pos < 0, "short", "flat"))

    # Costs per trade
    if costs_col in bt.columns:
        in_pos = pos != 0.0
        prev_in_pos = in_pos.shift(1, fill_value=False)
        entry = in_pos & (~prev_in_pos)
        trade_id = entry.cumsum().astype(int)
        df = pd.DataFrame({"trade_id": trade_id, "in_pos": in_pos, "costs": bt[costs_col].fillna(0.0).astype(float)})
        costs_total = df[df["in_pos"]].groupby("trade_id")["costs"].sum()
        trades["costs_total"] = trades["trade_id"].map(costs_total).fillna(0.0).astype(float)
    else:
        trades["costs_total"] = 0.0

    trades["side"] = side
    trades["pnl_net"] = trades["pnl"].astype(float)
    trades["win"] = trades["pnl_net"] > 0

    # Keep a tidy, stable column set
    keep = [
        "trade_id",
        "entry_time",
        "exit_time",
        "bars",
        "side",
        "entry_equity",
        "exit_equity",
        "pnl_net",
        "trade_return",
        "costs_total",
        "win",
        "open",
    ]
    return trades[keep].sort_values("entry_time").reset_index(drop=True)


def _agg_table(trades: pd.DataFrame, group_key: str) -> pd.DataFrame:
    # Backward-compatible alias.
    return agg_trade_table(trades, group_key)


def write_trade_breakdown(
    bt: pd.DataFrame,
    *,
    out_dir: str | Path,
    prefix: str = "bt",
    pos_col: str = "position",
    returns_col: str = "returns_net",
    equity_col: str = "equity",
    costs_col: str = "costs",
) -> TradeBreakdownPaths:
    """Write minimal trade breakdown CSVs (diff-friendly).

    Outputs:
      - {prefix}_trades.csv (trade ledger)
      - {prefix}_by_month.csv (entry month breakdown)
      - {prefix}_by_side.csv
      - {prefix}_by_duration.csv
      - {prefix}_SUMMARY.md

    Designed so you can generate two sets (e.g. current vs search) and diff them.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = build_trade_ledger(
        bt,
        pos_col=pos_col,
        returns_col=returns_col,
        equity_col=equity_col,
        costs_col=costs_col,
    )

    trades_csv = out_dir / f"{prefix}_trades.csv"
    by_month_csv = out_dir / f"{prefix}_by_month.csv"
    by_side_csv = out_dir / f"{prefix}_by_side.csv"
    by_duration_csv = out_dir / f"{prefix}_by_duration.csv"
    summary_md = out_dir / f"{prefix}_SUMMARY.md"

    if trades.empty:
        # Still write empty artifacts for pipelines.
        trades.to_csv(trades_csv, index=False)
        pd.DataFrame().to_csv(by_month_csv, index=False)
        pd.DataFrame().to_csv(by_side_csv, index=False)
        pd.DataFrame().to_csv(by_duration_csv, index=False)
        summary_md.write_text("No trades.\n", encoding="utf-8")
        return TradeBreakdownPaths(out_dir, trades_csv, by_month_csv, by_side_csv, by_duration_csv, summary_md)

    # Add grouping keys
    et = pd.to_datetime(trades["entry_time"])
    trades = trades.copy()
    trades["entry_month"] = et.dt.to_period("M").astype(str)
    trades["duration_bin"] = duration_bin(trades["bars"])

    # Write ledger
    trades.to_csv(trades_csv, index=False)

    # Breakdown tables
    by_month = _agg_table(trades, "entry_month")
    by_side = _agg_table(trades, "side")
    by_duration = _agg_table(trades, "duration_bin")

    by_month.to_csv(by_month_csv, index=False)
    by_side.to_csv(by_side_csv, index=False)
    by_duration.to_csv(by_duration_csv, index=False)

    # Small human summary
    total_pnl = float(trades["pnl_net"].sum())
    n = int(len(trades))
    wr = float(trades["win"].mean())
    costs = float(trades["costs_total"].sum())

    top = trades.sort_values("pnl_net", ascending=False).head(10)
    bot = trades.sort_values("pnl_net", ascending=True).head(10)

    def _fmt(x: float) -> str:
        if np.isnan(x):
            return "nan"
        return f"{x:,.4f}"

    md = []
    md.append(f"# Trade breakdown: {prefix}\n")
    md.append(f"- Trades: **{n:,}**\n")
    md.append(f"- Win rate: **{wr*100:,.2f}%**\n")
    md.append(f"- Sum PnL (net): **{_fmt(total_pnl)}**\n")
    md.append(f"- Sum costs: **{_fmt(costs)}**\n")
    md.append("\n## Biggest winners (top 10 by pnl_net)\n")
    md.append(top[["trade_id", "entry_time", "exit_time", "bars", "side", "pnl_net", "trade_return", "costs_total"]].to_markdown(index=False))
    md.append("\n\n## Biggest losers (bottom 10 by pnl_net)\n")
    md.append(bot[["trade_id", "entry_time", "exit_time", "bars", "side", "pnl_net", "trade_return", "costs_total"]].to_markdown(index=False))
    md.append("\n\n## Files\n")
    md.append(f"- {trades_csv.name}\n- {by_month_csv.name}\n- {by_side_csv.name}\n- {by_duration_csv.name}\n")

    summary_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    return TradeBreakdownPaths(out_dir, trades_csv, by_month_csv, by_side_csv, by_duration_csv, summary_md)
