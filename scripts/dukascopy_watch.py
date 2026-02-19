from __future__ import annotations

import datetime as dt
import json
from pathlib import Path


STATE_PATH = Path(".cache/dukascopy_watch.json")
DATA_ROOT = Path("data/dukascopy_1s")
# We consider XAUUSD finished; watch only XAGUSD from here on.
SYMBOLS = ["XAGUSD"]


def _read_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    return json.loads(STATE_PATH.read_text())


def _write_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def latest_date(symbol: str) -> dt.date | None:
    # files look like: data/dukascopy_1s/XAUUSD/2022/2022-01-03.parquet
    root = DATA_ROOT / symbol
    if not root.exists():
        return None
    dates: list[dt.date] = []
    for p in root.rglob("*.parquet"):
        try:
            dates.append(dt.date.fromisoformat(p.stem))
        except ValueError:
            continue
    return max(dates) if dates else None


def main() -> None:
    state = _read_state()

    # expected_end is the UTC date we intended to backfill through.
    expected_end = state.get("expected_end")
    if expected_end is None:
        expected_end = dt.datetime.now(dt.UTC).date().isoformat()
        state["expected_end"] = expected_end

    expected_end_d = dt.date.fromisoformat(expected_end)

    latest = {sym: latest_date(sym) for sym in SYMBOLS}

    xag_done = (latest["XAGUSD"] is not None) and (latest["XAGUSD"] >= expected_end_d)

    xag_notified = bool(state.get("xag_notified", False))

    # Emit notifications only on state transitions.
    if xag_done and (not xag_notified):
        print(f"XAGUSD 1s build reached {expected_end} (latest={latest['XAGUSD']}).")
        state["xag_notified"] = True
        print("__REMOVE_CRON__")

    _write_state(state)


if __name__ == "__main__":
    main()
