"""Period building for WebUI backtests.

Supports:
- three_block: config-defined p1/p2/p3 blocks
- yearly: 2020, 2021, 2022, ..., current-year->today

Reads from config['periods'].

We intentionally keep this small and dependency-free.
"""

from __future__ import annotations

import datetime as dt
from typing import Any


def _parse_date(s: str) -> dt.date:
    if s == "today":
        return dt.date.today()
    return dt.date.fromisoformat(s)


def build_periods(cfg: dict[str, Any] | None) -> list[tuple[str, dt.date, dt.date]]:
    """Build periods list (name, start, end).

    Mode defaults to three_block if not provided.

    Notes:
    - three_block requires cfg['periods']['three_block'] with p1/p2/p3.
    """

    p = (cfg or {}).get("periods", {})
    mode = p.get("mode", "three_block")

    if mode == "yearly":
        start_year = 2020
        end_date = dt.date.today()
        periods: list[tuple[str, dt.date, dt.date]] = []

        for y in range(start_year, end_date.year + 1):
            start = dt.date(y, 1, 1)
            end = dt.date(y, 12, 31)
            if y == end_date.year:
                end = end_date
            periods.append((str(y), start, end))

        return periods

    # default: three_block
    tb = p.get("three_block")
    if not (isinstance(tb, dict) and tb.get("p1") and tb.get("p2") and tb.get("p3")):
        raise ValueError("periods.mode=three_block requires periods.three_block.p1/p2/p3 in config")

    out = []
    for key in ("p1", "p2", "p3"):
        item = tb[key]
        name = str(item.get("name", key))
        start = _parse_date(str(item.get("start")))
        end = _parse_date(str(item.get("end")))
        out.append((name, start, end))
    return out
