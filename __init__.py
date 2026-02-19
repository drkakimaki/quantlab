"""Quantlab trading strategy backtesting library."""

# Lazy imports to avoid circular dependencies
__all__ = ["report_periods_equity_only"]


def __getattr__(name):
    if name == "report_periods_equity_only":
        from .generate_bt_report import report_periods_equity_only
        return report_periods_equity_only
    raise AttributeError(f"module 'quantlab' has no attribute '{name}'")
