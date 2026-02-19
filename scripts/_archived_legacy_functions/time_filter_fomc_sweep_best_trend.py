from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quantlab.backtest import backtest_positions_account_margin
from quantlab.metrics import performance_summary
from quantlab.report_periods import report_periods_equity_only
from quantlab.strategies import trend_following_ma_crossover
from quantlab.time_filter import EventWindow, apply_time_filter, build_allow_mask_from_events


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
    kind: str  # whole_day | around_utc_time
    apply_mode: str  # force_flat | block_entry_hold_segment
    # whole-day params
    day_pre_h: int = 12
    day_post_h: int = 12
    # around-time params
    utc_time_hhmm: str = "19:00"
    pre_h: int = 2
    post_h: int = 2


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


def _parse_hhmm(s: str) -> tuple[int, int]:
    hh, mm = s.split(":")
    return int(hh), int(mm)


def load_fomc_days(path: Path) -> list[dt.date]:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("fomc CSV must contain 'date' column")
    days = [dt.date.fromisoformat(x) for x in df["date"].astype(str).tolist()]
    return sorted(set(days))


def events_for_variant(days: list[dt.date], v: Variant) -> list[EventWindow]:
    out: list[EventWindow] = []
    if v.kind == "whole_day":
        # Represent a UTC day by anchoring at 12:00Z and using +/- 12h (or configurable)
        for d in days:
            ts = pd.Timestamp(dt.datetime(d.year, d.month, d.day, 12, 0, tzinfo=dt.UTC))
            out.append(EventWindow(ts=ts, pre=dt.timedelta(hours=v.day_pre_h), post=dt.timedelta(hours=v.day_post_h)))
        return out

    if v.kind == "around_utc_time":
        hh, mm = _parse_hhmm(v.utc_time_hhmm)
        for d in days:
            ts = pd.Timestamp(dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=dt.UTC))
            out.append(EventWindow(ts=ts, pre=dt.timedelta(hours=v.pre_h), post=dt.timedelta(hours=v.post_h)))
        return out

    raise ValueError("unknown variant kind")


def main() -> None:
    ap = argparse.ArgumentParser(description="FOMC time-filter sweep on best_trend (XAUUSD).")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/trend_based/time_filter_fomc"))
    ap.add_argument("--root-5m-ohlc", type=Path, default=Path("data/dukascopy_5m_ohlc"))
    ap.add_argument("--root-15m-ohlc", type=Path, default=Path("data/dukascopy_15m_ohlc"))
    ap.add_argument("--fomc-days", type=Path, default=Path("data/econ_calendar/fomc_decision_days.csv"))
    ap.add_argument("--p3-end", type=dt.date.fromisoformat, default=dt.date(2026, 2, 13))
    ap.add_argument("--make-html", action="store_true")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    periods = {
        "2021-2022": (dt.date(2021, 1, 1), dt.date(2022, 12, 31)),
        "2023-2025": (dt.date(2023, 1, 1), dt.date(2025, 12, 31)),
        "2026": (dt.date(2026, 1, 1), args.p3_end),
    }

    fomc_days = load_fomc_days(args.fomc_days)

    variants: list[Variant] = [
        Variant(id="none", label="No FOMC filter (reference)", kind="whole_day", apply_mode="force_flat"),
        Variant(
            id="fomc_day_flat",
            label="Block FOMC decision *day* (UTC) — force flat",
            kind="whole_day",
            apply_mode="force_flat",
            day_pre_h=12,
            day_post_h=12,
        ),
        Variant(
            id="fomc_day_no_entry",
            label="Block FOMC decision *day* (UTC) — block entries, hold segments",
            kind="whole_day",
            apply_mode="block_entry_hold_segment",
            day_pre_h=12,
            day_post_h=12,
        ),
        Variant(
            id="fomc_19_2h_flat",
            label="Block around 19:00 UTC ±2h — force flat (approx statement time)",
            kind="around_utc_time",
            apply_mode="force_flat",
            utc_time_hhmm="19:00",
            pre_h=2,
            post_h=2,
        ),
        Variant(
            id="fomc_19_2h_no_entry",
            label="Block around 19:00 UTC ±2h — block entries, hold segments",
            kind="around_utc_time",
            apply_mode="block_entry_hold_segment",
            utc_time_hhmm="19:00",
            pre_h=2,
            post_h=2,
        ),
        Variant(
            id="fomc_19_4h_flat",
            label="Block around 19:00 UTC ±4h — force flat (approx statement time)",
            kind="around_utc_time",
            apply_mode="force_flat",
            utc_time_hhmm="19:00",
            pre_h=4,
            post_h=4,
        ),
        Variant(
            id="fomc_19_1h_flat",
            label="Block around 19:00 UTC ±1h — force flat (approx statement time)",
            kind="around_utc_time",
            apply_mode="force_flat",
            utc_time_hhmm="19:00",
            pre_h=1,
            post_h=1,
        ),
        Variant(
            id="fomc_19_1h_no_entry",
            label="Block around 19:00 UTC ±1h — block entries, hold segments",
            kind="around_utc_time",
            apply_mode="block_entry_hold_segment",
            utc_time_hhmm="19:00",
            pre_h=1,
            post_h=1,
        ),
        Variant(
            id="fomc_19_4h_no_entry",
            label="Block around 19:00 UTC ±4h — block entries, hold segments",
            kind="around_utc_time",
            apply_mode="block_entry_hold_segment",
            utc_time_hhmm="19:00",
            pre_h=4,
            post_h=4,
        ),
    ]

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

        base_periods[pname] = {"px": px, "bt": bt}

    rows: list[dict] = []

    for v in variants:
        # restrict events to days that intersect the period (done per period below)
        for pname, (start, end) in periods.items():
            px = base_periods[pname]["px"]
            bt_base = base_periods[pname]["bt"]

            pos_exec = bt_base["position"].astype(float).reindex(px.index).fillna(0.0)

            if v.id == "none":
                pos_f = pos_exec
                ev = []
            else:
                days = [d for d in fomc_days if start <= d <= end]
                ev = events_for_variant(days, v)
                allow = build_allow_mask_from_events(px.index, events=ev)
                # executed-pos is already lagged; align gating to same-bar
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

            met = _bt_metrics(bt_f)
            rows.append(
                {
                    "variant_id": v.id,
                    "variant_label": v.label,
                    "period": pname,
                    **met,
                    "apply_mode": v.apply_mode,
                    "kind": v.kind,
                    "n_events": len(ev),
                    "events": json.dumps(
                        [
                            {
                                "ts": str(x.ts),
                                "pre": str(x.pre),
                                "post": str(x.post),
                            }
                            for x in ev
                        ]
                    ),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fomc_filter_results_long.csv", index=False)

    agg = (
        df.groupby(["variant_id", "variant_label"])\
          .agg(mean_sharpe=("sharpe", "mean"), worst_maxdd=("maxdd_pct", "min"), sum_pnl=("pnl_pct", "sum"))\
          .reset_index()\
          .sort_values(["mean_sharpe", "sum_pnl"], ascending=[False, False])
    )
    agg.to_csv(out_dir / "fomc_filter_results_agg.csv", index=False)

    md = []
    md.append("# FOMC filter sweep — best_trend\n")
    md.append(f"FOMC days source: `{args.fomc_days}` (date-only; no exact decision times).\n")
    md.append("Interpretation:\n")
    md.append("- `fomc_day_*`: blocks the whole UTC day (represented as 12:00Z ± 12h).\n")
    md.append("- `fomc_19_2h_*`: blocks around 19:00 UTC ± 2h (approx statement time; ignores DST shifts).\n")
    md.append("\n## Aggregate ranking (mean Sharpe across all 3 periods)\n")
    md.append(agg.to_string(index=False))

    for pname in periods:
        sub = df[df.period == pname].copy().sort_values(["sharpe", "pnl_pct"], ascending=[False, False])
        md.append(f"\n\n## {pname} — by Sharpe\n")
        md.append(sub[["variant_id", "sharpe", "pnl_pct", "maxdd_pct", "trades", "n_events"]].to_string(index=False))

    (out_dir / "fomc_filter_summary.md").write_text("\n".join(md) + "\n")

    if args.make_html:
        for v in variants:
            bts = {}
            ntr = {}
            for pname, (start, end) in periods.items():
                px = base_periods[pname]["px"]
                bt_base = base_periods[pname]["bt"]
                pos_exec = bt_base["position"].astype(float).reindex(px.index).fillna(0.0)

                if v.id == "none":
                    pos_f = pos_exec
                else:
                    days = [d for d in fomc_days if start <= d <= end]
                    ev = events_for_variant(days, v)
                    allow = build_allow_mask_from_events(px.index, events=ev)
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

            report_periods_equity_only(
                periods=bts,
                out_path=out_dir / f"fomc_filter_{v.id}.html",
                title=f"XAUUSD best_trend + FOMC filter | {v.label}",
                freq="5MIN",
                initial_capital={k: 1000.0 for k in bts},
                n_trades=ntr,
            )

    print(f"Wrote: {out_dir / 'fomc_filter_summary.md'}")
    print(f"Wrote: {out_dir / 'fomc_filter_results_long.csv'}")
    print(f"Wrote: {out_dir / 'fomc_filter_results_agg.csv'}")


if __name__ == "__main__":
    main()
