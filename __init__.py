"""Quantlab trading strategy backtesting library."""

"""Quantlab trading strategy backtesting library."""

# Public exports (keep the surface minimal)
__all__ = ["report_periods_equity_only"]


def __getattr__(name):
    if name == "report_periods_equity_only":
        from .reporting.generate_bt_report import report_periods_equity_only
        return report_periods_equity_only
    raise AttributeError(f"module 'quantlab' has no attribute '{name}'")
