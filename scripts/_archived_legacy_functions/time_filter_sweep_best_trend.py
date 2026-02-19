from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.backtest import backtest_positions_account_margin
from quantlab.metrics import performance_summary
from quantlab.report_periods import report_periods_equity_only
from quantlab.strategies import trend_following_ma_crossover
from quantlab.time_filter import (
    SessionWindow,
    apply_time_filter,
    build_allow_mask_from_sessions,
)


def _load_daily_paths(root: Path, symbol: str, start: dt.date, end: dt.date) -> list[str]:
    paths: list[str] = []
    cur = start
    one = dt.timedelta(days=1)
    while cur <= end:
        p = root / symbol / str(cur.year) / f"{cur.isoformat()}.parquet"
        if p.exists():
            paths.append(str(p))
        cur += one
    return paths


def load_ohlc_daily(*, symbol: str, start: dt.date, end: dt.date, root: Path) -> pd.DataFrame:
    paths = _load_daily_paths(root, symbol, start, end)
    if not paths:
        raise FileNotFoundError(f"No OHLC parquet files found for {symbol} in {root} between {start} and {end}")

    import polars as pl

    df = (
        pl.scan_parquet(paths)
        .select(["ts", "open", "high", "low", "close"])
        .sort("ts")
        .collect(engine="streaming")
    )
    out = df.to_pandas().set_index("ts").sort_index()
    out.index = pd.to_datetime(out.index)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    return out


@dataclass(frozen=True)
class Variant:
    id: str
    label: str
    tz: str
    sessions_mode: str  # block|allow
    sessions: list[SessionWindow]
    apply_mode: str  # force_flat|block_entry_hold_segment


def _bt_metrics(bt: pd.DataFrame) -> dict:
    summ = performance_summary(bt["returns_net"], bt["equity"], freq="5MIN")
    pnl_pct = (float(bt["equity"].iloc[-1]) / 1000.0 - 1.0) * 100.0
    peak = bt["equity"].cummax()
    maxdd_pct = float((bt["equity"] / peak - 1.0).min()) * 100.0
    execs = int((bt["position"].fillna(0.0).diff().abs() > 0).sum())
    trades = int(execs // 2)
    return {
        "pnl_pct": pnl_pct,
        "maxdd_pct": maxdd_pct,
        "sharpe": float(summ.sharpe),
        "trades": trades,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep sensible time filters on best_trend (XAUUSD).")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/trend_based/time_filter"))
    ap.add_argument("--root-5m-ohlc", type=Path, default=Path("data/dukascopy_5m_ohlc"))
    ap.add_argument("--root-15m-ohlc", type=Path, default=Path("data/dukascopy_15m_ohlc"))
    ap.add_argument("--p3-end", type=dt.date.fromisoformat, default=dt.date(2026, 2, 13))
    ap.add_argument("--make-html-top", type=int, default=5, help="Generate HTML for top-K variants by mean Sharpe")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    periods = {
        "2021-2022": (dt.date(2021, 1, 1), dt.date(2022, 12, 31)),
        "2023-2025": (dt.date(2023, 1, 1), dt.date(2025, 12, 31)),
        "2026": (dt.date(2026, 1, 1), args.p3_end),
    }

    # NOTE: data is UTC in parquet; we use UTC for sessions unless you explicitly change.
    variants: list[Variant] = [
        Variant(
            id="none",
            label="No time filter (reference)",
            tz="UTC",
            sessions_mode="block",
            sessions=[],
            apply_mode="force_flat",
        ),
        # Asia block (UTC)
        Variant(
            id="no_asia_flat",
            label="Block Asia 00:00-06:59 UTC (force flat)",
            tz="UTC",
            sessions_mode="block",
            sessions=[SessionWindow("00:00", "06:59")],
            apply_mode="force_flat",
        ),
        Variant(
            id="no_asia_no_entry",
            label="Block Asia 00:00-06:59 UTC (block entries; hold segments)",
            tz="UTC",
            sessions_mode="block",
            sessions=[SessionWindow("00:00", "06:59")],
            apply_mode="block_entry_hold_segment",
        ),
        # Only trade London+NY-ish
        Variant(
            id="only_07_21_flat",
            label="Allow only 07:00-20:59 UTC (force flat outside)",
            tz="UTC",
            sessions_mode="allow",
            sessions=[SessionWindow("07:00", "20:59")],
            apply_mode="force_flat",
        ),
        Variant(
            id="only_07_21_no_entry",
            label="Allow only 07:00-20:59 UTC (block entries outside; hold segments)",
            tz="UTC",
            sessions_mode="allow",
            sessions=[SessionWindow("07:00", "20:59")],
            apply_mode="block_entry_hold_segment",
        ),
        # Block NY open spike window (approx; 13:30-14:29 UTC)
        Variant(
            id="no_ny_open_flat",
            label="Block 13:30-14:29 UTC (force flat)",
            tz="UTC",
            sessions_mode="block",
            sessions=[SessionWindow("13:30", "14:29")],
            apply_mode="force_flat",
        ),
        Variant(
            id="no_ny_open_no_entry",
            label="Block 13:30-14:29 UTC (block entries; hold segments)",
            tz="UTC",
            sessions_mode="block",
            sessions=[SessionWindow("13:30", "14:29")],
            apply_mode="block_entry_hold_segment",
        ),
        # Block late US session (often choppy)
        Variant(
            id="no_late_us_flat",
            label="Block 20:00-23:59 UTC (force flat)",
            tz="UTC",
            sessions_mode="block",
            sessions=[SessionWindow("20:00", "23:59")],
            apply_mode="force_flat",
        ),
        Variant(
            id="no_late_us_no_entry",
            label="Block 20:00-23:59 UTC (block entries; hold segments)",
            tz="UTC",
            sessions_mode="block",
            sessions=[SessionWindow("20:00", "23:59")],
            apply_mode="block_entry_hold_segment",
        ),
    ]

    # Fixed strategy knobs (corr defaults are in strategy)
    fixed = dict(
        fast=30,
        slow=75,
        htf_rule="15min",
        ema_sep_filter=True,
        ema_fast=50,
        ema_slow=250,
        atr_n=14,
        sep_k=0.15,
        nochop_filter=True,
        nochop_ema=15,
        nochop_lookback=20,
        nochop_min_closes=12,
        corr_filter=True,
        fee_bps=0.0,
        slippage_bps=0.0,
    )

    # Load base strategy bt once per period (executed positions)
    base_periods: dict[str, dict] = {}
    for pname, (start, end) in periods.items():
        bars5 = load_ohlc_daily(symbol="XAUUSD", start=start, end=end, root=args.root_5m_ohlc)
        bars15 = load_ohlc_daily(symbol="XAUUSD", start=start, end=end, root=args.root_15m_ohlc)
        px = bars5["close"].astype(float)

        xag = load_ohlc_daily(symbol="XAGUSD", start=start, end=end, root=args.root_5m_ohlc)["close"].astype(float)
        eur = load_ohlc_daily(symbol="EURUSD", start=start, end=end, root=args.root_5m_ohlc)["close"].astype(float)

        corr_mod = {
            "logic": "or",
            "xag": {"close": xag, "window": 40, "min_abs": 0.10, "flip_lookback": 50, "max_flips": 0},
            "eur": {"close": eur, "window": 75, "min_abs": 0.10, "flip_lookback": 75, "max_flips": 5},
        }

        bt, _, _ = trend_following_ma_crossover(
            px,
            fast=int(fixed["fast"]),
            slow=int(fixed["slow"]),
            fee_bps=float(fixed["fee_bps"]),
            slippage_bps=float(fixed["slippage_bps"]),
            htf_confirm={"bars_15m": bars15, "rule": str(fixed.get("htf_rule", "15min"))},
            ema_sep={"ema_fast": int(fixed["ema_fast"]), "ema_slow": int(fixed["ema_slow"]), "atr_n": int(fixed["atr_n"]), "sep_k": float(fixed["sep_k"])},
            nochop={
                "ema": int(fixed["nochop_ema"]),
                "lookback": int(fixed["nochop_lookback"]),
                "min_closes": int(fixed["nochop_min_closes"]),
                "entry_held": False,
                "exit_bad_bars": 0,
            },
            corr=corr_mod,
            sizing={"confirm_size_one": 1.0, "confirm_size_both": 2.0},
        )

        base_periods[pname] = {
            "px": px,
            "bt": bt,
        }

    rows_long: list[dict] = []

    for v in variants:
        for pname in periods:
            px = base_periods[pname]["px"]
            bt_base = base_periods[pname]["bt"]

            # Use executed positions from base BT as the thing we time-filter.
            # IMPORTANT: since bt_base position already includes engine lag=1,
            # we re-run engine with lag=0 after filtering.
            pos_exec = bt_base["position"].astype(float).reindex(px.index).fillna(0.0)

            if v.id == "none":
                pos_f = pos_exec
            else:
                allow = build_allow_mask_from_sessions(
                    px.index,
                    tz=v.tz,
                    block=v.sessions if v.sessions_mode == "block" else None,
                    allow=v.sessions if v.sessions_mode == "allow" else None,
                )
                entry_shift = 0  # executed-pos is already lagged; entry gating should use same-bar allow
                pos_f = apply_time_filter(pos_exec, allow, mode=v.apply_mode, entry_shift=entry_shift)

            bt_f = backtest_positions_account_margin(
                prices=px,
                positions_size=pos_f,
                initial_capital=1000.0,
                leverage=20.0,
                lot_per_size=0.01,
                contract_size_per_lot=100.0,
                fee_bps=0.0,
                slippage_bps=0.0,
                lag=0,
                max_size=2.0,
                discrete_sizes=(0.0, 1.0, 2.0),
                margin_policy="skip_entry",
            )

            met = _bt_metrics(bt_f)
            rows_long.append(
                {
                    "variant_id": v.id,
                    "variant_label": v.label,
                    "period": pname,
                    **met,
                    "tz": v.tz,
                    "sessions_mode": v.sessions_mode,
                    "apply_mode": v.apply_mode,
                    "sessions": json.dumps([w.__dict__ for w in v.sessions]),
                }
            )

    df = pd.DataFrame(rows_long)
    df.to_csv(out_dir / "time_filter_results_long.csv", index=False)

    # Aggregate
    agg = (
        df.groupby(["variant_id", "variant_label"])\
          .agg(mean_sharpe=("sharpe", "mean"), worst_maxdd=("maxdd_pct", "min"), sum_pnl=("pnl_pct", "sum"))\
          .reset_index()\
          .sort_values(["mean_sharpe", "sum_pnl"], ascending=[False, False])
    )
    agg.to_csv(out_dir / "time_filter_results_agg.csv", index=False)

    # Markdown summary
    md = []
    md.append("# Time filter sweep — best_trend\n")
    md.append("Data/index assumed UTC (parquet timestamps). These filters are *heuristic* session windows.\n")
    md.append("\n## Top by mean Sharpe (across all 3 periods)\n")
    md.append(agg.head(15).to_string(index=False))

    for pname in periods:
        sub = df[df.period == pname].copy().sort_values(["sharpe", "pnl_pct"], ascending=[False, False])
        md.append(f"\n\n## {pname} — Top by Sharpe\n")
        md.append(sub[["variant_id", "sharpe", "pnl_pct", "maxdd_pct", "trades"]].head(15).to_string(index=False))

    md_path = out_dir / "time_filter_summary.md"
    md_path.write_text("\n".join(md) + "\n")

    # Optional: generate HTML for top K variants
    top_k = int(args.make_html_top)
    if top_k > 0:
        top_ids = agg.head(top_k)["variant_id"].tolist()
        for vid in top_ids:
            v = next(x for x in variants if x.id == vid)
            bts = {}
            ntr = {}
            for pname in periods:
                px = base_periods[pname]["px"]
                bt_base = base_periods[pname]["bt"]
                pos_exec = bt_base["position"].astype(float).reindex(px.index).fillna(0.0)

                if v.id == "none":
                    pos_f = pos_exec
                else:
                    allow = build_allow_mask_from_sessions(
                        px.index,
                        tz=v.tz,
                        block=v.sessions if v.sessions_mode == "block" else None,
                        allow=v.sessions if v.sessions_mode == "allow" else None,
                    )
                    pos_f = apply_time_filter(pos_exec, allow, mode=v.apply_mode, entry_shift=0)

                bt_f = backtest_positions_account_margin(
                    prices=px,
                    positions_size=pos_f,
                    initial_capital=1000.0,
                    leverage=20.0,
                    lot_per_size=0.01,
                    contract_size_per_lot=100.0,
                    fee_bps=0.0,
                    slippage_bps=0.0,
                    lag=0,
                    max_size=2.0,
                    discrete_sizes=(0.0, 1.0, 2.0),
                    margin_policy="skip_entry",
                )
                bts[pname] = bt_f
                ntr[pname] = _bt_metrics(bt_f)["trades"]

            title = f"XAUUSD best_trend + time filter | {v.label}"
            report_periods_equity_only(
                periods=bts,
                out_path=out_dir / f"time_filter_{v.id}.html",
                title=title,
                freq="5MIN",
                initial_capital={k: 1000.0 for k in bts},
                n_trades=ntr,
            )

    print(f"Wrote: {out_dir / 'time_filter_summary.md'}")
    print(f"Wrote: {out_dir / 'time_filter_results_long.csv'}")
    print(f"Wrote: {out_dir / 'time_filter_results_agg.csv'}")


if __name__ == "__main__":
    main()
