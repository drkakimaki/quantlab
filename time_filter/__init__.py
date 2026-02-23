"""Time-based filters for strategy positions.

This package provides utilities to *mask* or *gate* positions based on time-of-day
sessions (e.g. no Asia) or event windows (e.g. no FOMC).

Design goals
------------
- Pure functions operating on pandas indexes/Series.
- Strategy-agnostic: apply to any 0/1/2 (or -1/0/+1) position/size series.
- Semantics (canonical): force_flat — set position=0 during blocked windows.

See: `quantlab.time_filter.core`.
"""

from .core import (
    EventWindow,
    SessionWindow,
    apply_time_filter,
    build_allow_mask_from_events,
    build_allow_mask_from_sessions,
    build_allow_mask_from_econ_calendar,
    build_allow_mask_from_months,
)

__all__ = [
    "EventWindow",
    "SessionWindow",
    "apply_time_filter",
    "build_allow_mask_from_events",
    "build_allow_mask_from_sessions",
    "build_allow_mask_from_econ_calendar",
    "build_allow_mask_from_months",
]
