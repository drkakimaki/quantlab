from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from quantlab.webui.runner import WORKSPACE, load_period_data, load_fomc_mask
from quantlab.strategies import BacktestConfig, TrendStrategyWithGates


GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "best_trend_2024_2025_series.csv.gz"
CFG = Path(__file__).resolve().parents[1] / "golden" / "best_trend_2024_2025_config.yaml"


def _data_available() -> bool:
    # minimal check: 5m OHLC data root exists and is non-empty
    # (we store these as parquet shards)
    root = WORKSPACE / "quantlab/data/dukascopy_5m_ohlc"
    return root.exists() and any(root.glob("**/*.parquet"))


def _run_best_trend(cfg: dict) -> pd.DataFrame:
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

    # keep only what we regression-test
    keep = ["position", "returns_net", "equity"]
    missing = [c for c in keep if c not in bt.columns]
    if missing:
        raise KeyError(f"Backtest missing expected cols: {missing}")
    return bt[keep]


def test_best_trend_2024_2025_position_equity_series_regression() -> None:
    if not _data_available():
        # This repo often runs with local datasets that are not available in CI.
        # Skip cleanly if data is not present.
        import pytest

        pytest.skip("dukascopy_5m_ohlc data not available")

    if not GOLDEN.exists() or not CFG.exists():
        raise AssertionError(
            f"Missing golden files. Expected {GOLDEN} and {CFG}. "
            "Run tests/regression/update_golden_best_trend_2024_2025.py"
        )

    cfg = yaml.safe_load(CFG.read_text()) or {}

    actual = _run_best_trend(cfg)

    golden = pd.read_csv(GOLDEN, compression="gzip")
    golden["ts"] = pd.to_datetime(golden["ts"], utc=True)
    golden = golden.set_index("ts").sort_index()

    # Index must match exactly
    if not actual.index.equals(golden.index):
        raise AssertionError(f"Index mismatch: actual={len(actual)} golden={len(golden)}")

    # Position should be exact (this is the most sensitive canary)
    pos_a = actual["position"].astype(float).values
    pos_g = golden["position"].astype(float).values
    if not np.array_equal(pos_a, pos_g):
        # provide a small diagnostic
        diff = np.where(pos_a != pos_g)[0]
        i0 = int(diff[0]) if len(diff) else -1
        ts0 = str(actual.index[i0]) if i0 >= 0 else "n/a"
        raise AssertionError(f"position series differs (first diff @ {i0}, ts={ts0})")

    # Equity / returns should be extremely close
    for col in ["returns_net", "equity"]:
        a = actual[col].astype(float).values
        g = golden[col].astype(float).values
        if not np.allclose(a, g, rtol=0.0, atol=1e-10, equal_nan=True):
            diff = np.where(~np.isclose(a, g, rtol=0.0, atol=1e-10, equal_nan=True))[0]
            i0 = int(diff[0]) if len(diff) else -1
            ts0 = str(actual.index[i0]) if i0 >= 0 else "n/a"
            raise AssertionError(f"{col} series differs (first diff @ {i0}, ts={ts0})")
