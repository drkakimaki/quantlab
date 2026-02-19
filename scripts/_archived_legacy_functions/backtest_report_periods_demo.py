from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quantlab import PeriodResult, ReportConfig, buy_and_hold, write_html_report_periods


def main() -> None:
    # Demo price series covering 2022..2026 at 1-minute bars (synthetic).
    rng = np.random.default_rng(7)

    def make_block(start: str, end: str, base: float) -> pd.Series:
        idx = pd.date_range(start, end, freq="min", inclusive="both", tz="UTC")
        rets = pd.Series(rng.normal(0.0, 0.0004, size=len(idx)), index=idx)
        px = (1.0 + rets).cumprod() * base
        return px

    prices = pd.concat(
        [
            make_block("2022-01-01", "2022-03-31", 1800.0),
            make_block("2023-01-01", "2025-03-31", 1900.0),
            make_block("2026-01-01", "2026-02-10", 2000.0),
        ]
    ).sort_index()
    prices.name = "XAUUSD"

    bt_2022, pnl_2022, n_2022 = buy_and_hold(prices, start="2022-01-01", end="2022-03-31", fee_bps=0.5, slippage_bps=0.5, lot_size=1.0)
    bt_2325, pnl_2325, n_2325 = buy_and_hold(prices, start="2023-01-01", end="2025-03-31", fee_bps=0.5, slippage_bps=0.5, lot_size=1.0)
    bt_2026, pnl_2026, n_2026 = buy_and_hold(prices, start="2026-01-01", end="2026-02-10", fee_bps=0.5, slippage_bps=0.5, lot_size=1.0)

    periods = {
        "2022": PeriodResult(bt=bt_2022, pnl=pnl_2022, n_trades=n_2022),
        "2023-2025": PeriodResult(bt=bt_2325, pnl=pnl_2325, n_trades=n_2325),
        "2026": PeriodResult(bt=bt_2026, pnl=pnl_2026, n_trades=n_2026),
    }

    out = Path("reports") / "demo_periods" / "baseline_buy_hold.html"
    cfg = ReportConfig(title="Baseline buy&hold — demo periods", freq="MIN")
    p = write_html_report_periods(periods, out, config=cfg)
    print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
