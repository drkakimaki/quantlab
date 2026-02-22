"""Regenerate golden series for best_trend 2024-2025.

Writes:
- tests/golden/best_trend_2024_2025_series.csv.gz
- tests/golden/best_trend_2024_2025_config.yaml

Run with the repo venv, e.g.:
  ../.venv/bin/python scripts/regression/update_golden_best_trend_2024_2025.py

This is meant to be run *only* when intentional changes are made to execution
semantics (gates, costs, time filtering, trade extraction, etc.).
"""

from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path

import pandas as pd
import yaml

from quantlab.webui.runner import WORKSPACE, load_period_data, load_fomc_mask
from quantlab.strategies import BacktestConfig, TrendStrategyWithGates


HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
GOLDEN_DIR = REPO / "tests" / "regression" / "golden"
OUT_SERIES = GOLDEN_DIR / "best_trend_2024_2025_series.csv.gz"
OUT_CFG = GOLDEN_DIR / "best_trend_2024_2025_config.yaml"


def run_best_trend(cfg: dict) -> pd.DataFrame:
    start = dt.date(2024, 1, 1)
    end = dt.date(2025, 12, 31)

    symbol = cfg.get("symbol", "XAUUSD")
    corr_symbol = cfg.get("corr_symbol", "XAGUSD")
    corr2_symbol = cfg.get("corr2_symbol", "EURUSD")

    costs = cfg.get("costs", {}) or {}
    bt_cfg = BacktestConfig(
        fee_per_lot=float(costs.get("fee_per_lot", 0.0) or 0.0),
        spread_per_lot=float(costs.get("spread_per_lot", 0.0) or 0.0),
        record_executions=False,
    )

    data = load_period_data(
        symbol,
        start,
        end,
        need_htf=True,
        need_corr=True,
        corr_symbol=corr_symbol,
        corr2_symbol=corr2_symbol,
    )

    fomc_path = WORKSPACE / cfg.get("time_filter", {}).get("fomc", {}).get(
        "days_csv", "quantlab/data/econ_calendar/fomc_decision_days.csv"
    )
    fomc_cfg = cfg.get("time_filter", {}).get("fomc", {})
    allow_mask = load_fomc_mask(data["prices"].index, start, end, fomc_path, fomc_cfg)

    strat = TrendStrategyWithGates.from_config(cfg, allow_mask=allow_mask)
    context = {
        "bars_15m": data.get("bars_15m"),
        "prices_xag": data.get("prices_xag"),
        "prices_eur": data.get("prices_eur"),
    }

    result = strat.run_backtest(data["prices"], context=context, config=bt_cfg)
    bt = result.df.copy()
    bt.index = pd.to_datetime(bt.index, utc=True)
    bt = bt.sort_index()
    return bt[["position", "returns_net", "equity"]]


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    cfg_path = WORKSPACE / "quantlab/configs/trend_based/current.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) or {}

    # Important: write an isolated snapshot of the config used for the golden.
    cfg_out = copy.deepcopy(cfg)
    OUT_CFG.write_text(yaml.safe_dump(cfg_out, sort_keys=False), encoding="utf-8")

    bt = run_best_trend(cfg_out)

    df = bt.copy()
    df.insert(0, "ts", df.index)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(OUT_SERIES, index=False, compression="gzip")

    print(f"Wrote config:  {OUT_CFG}")
    print(f"Wrote series:  {OUT_SERIES}")
    print(f"Rows: {len(df):,}")


if __name__ == "__main__":
    main()
