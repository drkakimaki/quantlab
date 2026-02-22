"""Quick diagnostic: correlation gate acceptance rate at entry.

Outputs per-period:
- number of raw entries (post previous gates, pre-corr)
- number accepted by corr gate at entry
- acceptance rate
- share of entries sized to confirm_size_both vs confirm_size_one

Intended for fast iteration (not a library module).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import datetime as dt

import pandas as pd
import yaml

from quantlab.webui.periods import build_periods
from quantlab.webui.runner import load_period_data, load_fomc_mask
from quantlab.webui.config import WORKSPACE
from quantlab.strategies.trend_following import (
    TrendStrategyWithGates,
    HTFConfirmGate,
    EMASeparationGate,
    NoChopGate,
    CorrelationGate,
    TimeFilterGate,
    SeasonalitySizeCapGate,
    ChurnGate,
    RiskGate,
)


def _entry_bars(pos: pd.Series) -> pd.Series:
    on = pos.fillna(0.0).astype(float) > 0.0
    return on & (~on.shift(1, fill_value=False))


def _build_pre_corr_strategy(cfg: dict, allow_mask: pd.Series | None):
    """Build TrendStrategyWithGates but with corr_gate disabled.

    We keep everything else identical so we can measure how many entries corr *filters*.
    """
    fast = cfg.get("trend", {}).get("fast", 30)
    slow = cfg.get("trend", {}).get("slow", 75)

    htf_gate = HTFConfirmGate(fast=fast, slow=slow) if cfg.get("htf_confirm") else None

    ema_sep_gate = None
    if cfg.get("ema_sep"):
        c = cfg["ema_sep"]
        ema_sep_gate = EMASeparationGate(
            ema_fast=c.get("ema_fast", 40),
            ema_slow=c.get("ema_slow", 300),
            atr_n=c.get("atr_n", 20),
            sep_k=c.get("sep_k", 0.05),
        )

    nochop_gate = None
    if cfg.get("nochop"):
        c = cfg["nochop"]
        nochop_gate = NoChopGate(
            ema=c.get("ema", 20),
            lookback=c.get("lookback", 40),
            min_closes=c.get("min_closes", 24),
            entry_held=c.get("entry_held", False),
            exit_bad_bars=c.get("exit_bad_bars", 0),
        )

    # corr_gate intentionally None

    time_filter_gate = None
    if cfg.get("time_filter") or allow_mask is not None:
        time_filter_gate = TimeFilterGate(allow_mask=allow_mask)

    seasonality_gate = None
    month_cap = (cfg.get("seasonality", {}) or {}).get("month_size_cap")
    if isinstance(month_cap, dict) and month_cap:
        caps = {int(k): float(v) for k, v in month_cap.items()}
        seasonality_gate = SeasonalitySizeCapGate(month_size_cap=caps)

    churn_gate = None
    churn_cfg = cfg.get("churn", {}) or {}
    if churn_cfg:
        churn_gate = ChurnGate(
            min_on_bars=int(churn_cfg.get("min_on_bars", 1) or 1),
            cooldown_bars=int(churn_cfg.get("cooldown_bars", 0) or 0),
        )

    risk_gate = None
    risk_cfg = cfg.get("risk", {}) or {}
    if risk_cfg:
        risk_gate = RiskGate(
            shock_exit_abs_ret=risk_cfg.get("shock_exit_abs_ret", 0.0),
            shock_exit_sigma_k=risk_cfg.get("shock_exit_sigma_k", 0.0),
            shock_exit_sigma_window=risk_cfg.get("shock_exit_sigma_window", 96),
            shock_cooldown_bars=risk_cfg.get("shock_cooldown_bars", 0),
        )

    return TrendStrategyWithGates(
        fast=fast,
        slow=slow,
        htf_gate=htf_gate,
        ema_sep_gate=ema_sep_gate,
        nochop_gate=nochop_gate,
        corr_gate=None,
        time_filter_gate=time_filter_gate,
        seasonality_gate=seasonality_gate,
        churn_gate=churn_gate,
        risk_gate=risk_gate,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}

    symbol = cfg.get("symbol", "XAUUSD")
    corr_symbol = cfg.get("corr_symbol", "XAGUSD")
    corr2_symbol = cfg.get("corr2_symbol", "EURUSD")

    fomc_cfg = (cfg.get("time_filter", {}) or {}).get("fomc", {}) or {}
    days_csv = (cfg.get("time_filter", {}) or {}).get("fomc", {}).get(
        "days_csv", "quantlab/data/econ_calendar/fomc_decision_days.csv"
    )
    fomc_path = WORKSPACE / str(days_csv)

    periods = build_periods(cfg)

    # Build corr gate exactly as canonical strategy does.
    corr_cfg = cfg.get("corr", {}) or {}
    xag_cfg = corr_cfg.get("xag", {}) or {}
    eur_cfg = corr_cfg.get("eur", {}) or {}
    sizing_cfg = cfg.get("sizing", {}) or {}
    corr_gate = CorrelationGate(
        logic=corr_cfg.get("logic", "or"),
        xag_window=xag_cfg.get("window", 40),
        xag_min_abs=xag_cfg.get("min_abs", 0.10),
        xag_flip_lookback=xag_cfg.get("flip_lookback", 50),
        xag_max_flips=xag_cfg.get("max_flips", 0),
        eur_window=eur_cfg.get("window", 75),
        eur_min_abs=eur_cfg.get("min_abs", 0.10),
        eur_flip_lookback=eur_cfg.get("flip_lookback", 75),
        eur_max_flips=eur_cfg.get("max_flips", 5),
        confirm_size_one=sizing_cfg.get("confirm_size_one", 1.0),
        confirm_size_both=sizing_cfg.get("confirm_size_both", 2.0),
    )

    rows = []

    for name, start, end in periods:
        data = load_period_data(
            symbol,
            start,
            end,
            need_htf=True,
            need_corr=True,
            corr_symbol=corr_symbol,
            corr2_symbol=corr2_symbol,
        )

        prices = data["prices"]
        allow_mask = load_fomc_mask(prices.index, start, end, fomc_path, fomc_cfg)
        context = {
            "bars_15m": data.get("bars_15m"),
            "prices_xag": data.get("prices_xag"),
            "prices_eur": data.get("prices_eur"),
            "allow_mask": allow_mask,
        }

        # pre-corr positions (all gates except corr)
        pre = _build_pre_corr_strategy(cfg, allow_mask=allow_mask).generate_positions(prices, context=context)
        e_pre = _entry_bars(pre)
        n_pre = int(e_pre.sum())

        # Apply corr gate to pre series to see what it filters.
        post = corr_gate(pre, prices.astype(float), context=context)
        e_post = _entry_bars(post)
        n_post = int(e_post.sum())

        # Sizing tier at entry (1 vs 2). Corr gate outputs 0/1/2 size already.
        size_at_entry = post.shift(0).where(e_post, 0.0)
        n_size1 = int((size_at_entry == float(corr_gate.confirm_size_one)).sum())
        n_size2 = int((size_at_entry == float(corr_gate.confirm_size_both)).sum())

        rows.append(
            {
                "period": name,
                "start": str(start),
                "end": str(end),
                "entries_pre_corr": n_pre,
                "entries_post_corr": n_post,
                "accept_rate": (n_post / n_pre) if n_pre else float("nan"),
                "n_size1": n_size1,
                "n_size2": n_size2,
                "share_size2": (n_size2 / (n_size1 + n_size2)) if (n_size1 + n_size2) else float("nan"),
            }
        )

    df = pd.DataFrame(rows)
    pd.set_option("display.max_columns", 50)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
