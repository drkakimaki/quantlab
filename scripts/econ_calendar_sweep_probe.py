"""Probe whether econ_calendar window parameters actually change the allow mask.

For two configs (base and modified), prints:
- total blocked bars
- blocked bars while pre-time-filter position is ON

Usage:
  ../.venv/bin/python scripts/econ_calendar_sweep_probe.py \
    --config configs/trend_based/current.yaml

It will generate a few variants internally (small vs huge windows).
"""

from __future__ import annotations

import copy
from pathlib import Path
import argparse
import pandas as pd
import yaml

from quantlab.webui.periods import build_periods
from quantlab.webui.runner import load_period_data, load_time_filter_mask
from quantlab.webui.config import WORKSPACE
from quantlab.strategies.trend_following import TrendStrategyWithGates


def _disable_time_filter(cfg):
    c = copy.deepcopy(cfg)
    c["time_filter"] = None
    return c


def _metrics(cfg, *, label: str):
    periods = build_periods(cfg)
    symbol = cfg.get("symbol", "XAUUSD")
    corr_symbol = cfg.get("corr_symbol", "XAGUSD")
    corr2_symbol = cfg.get("corr2_symbol", "EURUSD")

    rows=[]
    for name,start,end in periods:
        data = load_period_data(symbol,start,end,need_htf=True,need_corr=True,corr_symbol=corr_symbol,corr2_symbol=corr2_symbol)
        idx = pd.DatetimeIndex(data["prices"].index)

        # pre-time-filter pos
        strat_pre = TrendStrategyWithGates.from_config(_disable_time_filter(cfg), allow_mask=None)
        pos_pre = strat_pre.generate_positions(data["prices"], context={"bars_15m":data.get("bars_15m"),"prices_xag":data.get("prices_xag"),"prices_eur":data.get("prices_eur")}).fillna(0.0)
        in_pos = pos_pre>0

        allow = load_time_filter_mask(idx,start,end,cfg=cfg,workspace=WORKSPACE)
        if allow is None:
            allow = pd.Series(True,index=idx)
        blocked = ~pd.Series(allow,index=idx).astype(bool)

        rows.append((label,name,int(blocked.sum()),int((blocked & in_pos).sum())))
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    args=ap.parse_args()
    base=yaml.safe_load(Path(args.config).read_text()) or {}

    # Variant 1: very small windows
    small=copy.deepcopy(base)
    rules=small["time_filter"]["econ_calendar"]["rules"]
    for k in ("CPI","NFP"):
        rules[k]["pre_hours"]=0.0
        rules[k]["post_hours"]=0.0

    # Variant 2: huge windows
    big=copy.deepcopy(base)
    rules=big["time_filter"]["econ_calendar"]["rules"]
    for k in ("CPI","NFP"):
        rules[k]["pre_hours"]=6.0
        rules[k]["post_hours"]=6.0

    rows=[]
    rows += _metrics(small,label='small(0/0)')
    rows += _metrics(base,label='base')
    rows += _metrics(big,label='big(6/6)')

    df=pd.DataFrame(rows,columns=['variant','period','blocked_bars','blocked_in_pos'])
    print(df.to_string(index=False))

if __name__=='__main__':
    main()
