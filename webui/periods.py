"""Period building for WebUI backtests.

Supports:
- three_block: 2021-2022, 2023-2025, 2026->today
- yearly: 2021, 2022, ..., current-year->today

Reads from config['periods'] when present.

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

    Always starts at 2021.

    Backward compatible:
    - If cfg['periods']['mode'] missing -> default to three_block.
    - If cfg missing -> default to three_block.
    """

    p = (cfg or {}).get("periods", {})
    mode = p.get("mode", "three_block")

    if mode == "yearly":
        start_year = 2021
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
    if isinstance(tb, dict) and tb.get("p1") and tb.get("p2") and tb.get("p3"):
        out = []
        for key in ("p1", "p2", "p3"):
            item = tb[key]
            name = str(item.get("name", key))
            start = _parse_date(str(item.get("start")))
            end = _parse_date(str(item.get("end")))
            out.append((name, start, end))
        return out

    # legacy fallback
    return [
        ("2021-2022", dt.date(2021, 1, 1), dt.date(2022, 12, 31)),
        ("2023-2025", dt.date(2023, 1, 1), dt.date(2025, 12, 31)),
        ("2026", dt.date(2026, 1, 1), dt.date.today()),
    ]
