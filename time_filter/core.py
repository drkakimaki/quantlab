from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SessionWindow:
    """A daily time-of-day window.

    Times are interpreted in `tz` when building the mask.

    Example
    -------
    Block Asia (roughly): SessionWindow("00:00", "06:59") in UTC.
    """

    start_hhmm: str  # inclusive
    end_hhmm: str  # inclusive; supports wrap-around (e.g. 22:00..01:00)
    days: tuple[int, ...] | None = None  # 0=Mon..6=Sun; None -> all days


@dataclass(frozen=True)
class EventWindow:
    """A single event timestamp with a blocking window around it."""

    ts: pd.Timestamp
    pre: dt.timedelta
    post: dt.timedelta


def _parse_hhmm(s: str) -> tuple[int, int]:
    try:
        hh, mm = s.split(":")
        return int(hh), int(mm)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Invalid HH:MM time: {s!r}") from e


def build_allow_mask_from_sessions(
    index: pd.DatetimeIndex,
    *,
    tz: str = "UTC",
    block: list[SessionWindow] | None = None,
    allow: list[SessionWindow] | None = None,
) -> pd.Series:
    """Build a boolean allow-mask for a price/position index.

    Exactly one of (block, allow) should be provided.

    - If `block` is provided: allow=True except inside any blocked session.
    - If `allow` is provided: allow=True only inside at least one allowed session.

    Wrap-around windows are supported (e.g. 22:00..01:00).
    """
    if block is not None and allow is not None:
        raise ValueError("Provide only one of block or allow")
    if block is None and allow is None:
        # default: allow everything
        return pd.Series(True, index=index)

    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx_local = idx.tz_convert(tz)

    # minutes since midnight
    mins = idx_local.hour * 60 + idx_local.minute
    dow = idx_local.dayofweek

    def in_window(w: SessionWindow) -> pd.Series:
        sh, sm = _parse_hhmm(w.start_hhmm)
        eh, em = _parse_hhmm(w.end_hhmm)
        smin = sh * 60 + sm
        emin = eh * 60 + em

        if w.days is None:
            day_ok = pd.Series(True, index=idx_local)
        else:
            day_ok = pd.Series(dow.isin(list(w.days)), index=idx_local)

        if smin <= emin:
            t_ok = (mins >= smin) & (mins <= emin)
        else:
            # wrap-around (e.g. 22:00..01:00)
            t_ok = (mins >= smin) | (mins <= emin)

        return pd.Series(t_ok, index=idx_local) & day_ok

    windows = block if block is not None else allow
    m = pd.DataFrame({f"w{i}": in_window(w) for i, w in enumerate(windows)}).any(axis=1)

    if block is not None:
        return (~m).reindex(idx_local).astype(bool)
    return m.reindex(idx_local).astype(bool)


def build_allow_mask_from_months(
    index: pd.DatetimeIndex,
    *,
    block: list[int] | tuple[int, ...] | set[int] | None = None,
) -> pd.Series:
    """Build an allow-mask that blocks entire calendar months.

    Parameters
    ----------
    block:
      Months to block (1-12). Example: block=[6] blocks June entirely.

    Semantics
    ---------
    - allow=True outside blocked months
    - allow=False inside blocked months

    Notes
    -----
    - Uses the timestamp index month in UTC (or whatever tz is already on the index).
    - This is intended as a blunt seasonal *force-flat* filter.
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")

    months = sorted({int(m) for m in (block or [])})
    if not months:
        return pd.Series(True, index=idx)

    allow = ~pd.Index(idx.month).isin(months)
    return pd.Series(allow, index=idx).astype(bool)


def build_allow_mask_from_events(index: pd.DatetimeIndex, *, events: list[EventWindow]) -> pd.Series:
    """Allow-mask that blocks within any event's [ts-pre, ts+post] interval."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")

    allow = pd.Series(True, index=idx)
    for e in events:
        ts = pd.Timestamp(e.ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        start = ts - e.pre
        end = ts + e.post
        allow.loc[(idx >= start) & (idx <= end)] = False
    return allow.astype(bool)


def build_allow_mask_from_econ_calendar(
    index: pd.DatetimeIndex,
    *,
    start: dt.date,
    end: dt.date,
    csv_path: str,
    rules: dict[str, dict],
    default_tz: str = "UTC",
) -> pd.Series | None:
    """Build allow-mask from usd_important_events.csv style calendar.

    The CSV stores dates as ts_utc at 00:00Z (date-only). We override the time-of-day
    per rule via `utc_hhmm`.

    Parameters
    ----------
    rules:
      Mapping of rule_name -> config dict with keys:
        - match: list[str] substrings to match against the `event` field (OR)
        - utc_hhmm: "HH:MM" release time in UTC
        - pre_hours: float
        - post_hours: float
        - importance: optional list[str] filter (e.g. ["high"]) (default: no filter)
    """
    if csv_path is None:
        return None

    import os

    if not os.path.exists(csv_path):
        return None

    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")

    df = pd.read_csv(csv_path)
    if df is None or len(df) == 0:
        return None

    if "ts_utc" not in df.columns or "event" not in df.columns:
        return None

    df = df.copy()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_utc"]).copy()

    # date range filter (by date part)
    d0 = pd.Timestamp(start).tz_localize("UTC")
    d1 = pd.Timestamp(end).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df = df[(df["ts_utc"] >= d0) & (df["ts_utc"] <= d1)].copy()

    if df.empty:
        return None

    events: list[EventWindow] = []

    for _rule_name, rule in (rules or {}).items():
        if not isinstance(rule, dict):
            continue

        match = rule.get("match") or []
        if isinstance(match, str):
            match = [match]
        match = [str(x) for x in match if str(x)]
        if not match:
            continue

        utc_hhmm = str(rule.get("utc_hhmm") or "00:00")
        hh, mm = _parse_hhmm(utc_hhmm)
        pre = dt.timedelta(hours=float(rule.get("pre_hours", 0.0) or 0.0))
        post = dt.timedelta(hours=float(rule.get("post_hours", 0.0) or 0.0))

        imp = rule.get("importance")
        if isinstance(imp, str):
            imp = [imp]
        imp = [str(x) for x in (imp or [])]

        # filter rows matching substrings
        m = pd.Series(False, index=df.index)
        for s in match:
            m = m | df["event"].astype(str).str.contains(s, case=False, na=False, regex=False)

        sub = df[m].copy()
        if imp and "importance" in sub.columns:
            sub = sub[sub["importance"].astype(str).isin(imp)].copy()

        if sub.empty:
            continue

        # Override time-of-day on the date
        for ts0 in sub["ts_utc"].tolist():
            ts0 = pd.Timestamp(ts0)
            if ts0.tzinfo is None:
                ts0 = ts0.tz_localize("UTC")
            ts = pd.Timestamp(dt.datetime(ts0.year, ts0.month, ts0.day, hh, mm, tzinfo=dt.UTC))
            events.append(EventWindow(ts=ts, pre=pre, post=post))

    if not events:
        return None

    return build_allow_mask_from_events(idx, events=events)


def apply_time_filter(
    pos_size: pd.Series,
    allow_mask: pd.Series,
) -> pd.Series:
    """Apply a time filter to a position/size series.

    Canonical semantics: **force_flat**.

    - allow_mask=True  -> position unchanged
    - allow_mask=False -> force position to 0

    (Other semantics were removed to avoid confusing multi-shift interactions with
    engine lag and to keep the backtest wiring unambiguous.)
    """
    if not isinstance(pos_size, pd.Series):
        raise TypeError("pos_size must be a Series")

    pos = pos_size.copy().astype(float)
    idx = pos.index
    m = pd.Series(allow_mask, index=idx).reindex(idx).fillna(False).astype(bool)

    return pos.where(m, 0.0).astype(float)
