"""Time-based filters for strategy positions.

This package provides utilities to *mask* or *gate* positions based on time-of-day
sessions (e.g. no Asia) or event windows (e.g. no FOMC).

Design goals
------------
- Pure functions operating on pandas indexes/Series.
- Strategy-agnostic: apply to any 0/1/2 (or -1/0/+1) position/size series.
- Supports two common semantics:
    1) force_flat: set position=0 during blocked windows.
    2) block_entry_hold_segment: only block *entries* that start in a blocked
       window, but allow already-open segments to continue (mirrors our corr
       entry gating behavior).

See: `quantlab.time_filter.core`.
"""

from .core import (
    EventWindow,
    SessionWindow,
    apply_time_filter,
    build_allow_mask_from_events,
    build_allow_mask_from_sessions,
)

__all__ = [
    "EventWindow",
    "SessionWindow",
    "apply_time_filter",
    "build_allow_mask_from_events",
    "build_allow_mask_from_sessions",
]
