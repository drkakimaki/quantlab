from __future__ import annotations

"""Quantlab config schema + validator.

Goals
-----
- Fail fast on typos / unknown keys ("silent misconfig").
- Keep runtime strategy code mostly unchanged: validate -> get back a normal dict.
- Provide a tiny CLI in the *same file* for agent/human use.

Usage
-----
Python:
    from quantlab.config.schema import validate_config_dict
    cfg = validate_config_dict(cfg_dict)

CLI:
    .venv/bin/python -m quantlab.config.schema quantlab/configs/trend_based/current.yaml

Notes
-----
We validate the *canonical* config shape (esp. `pipeline`).
Legacy flat configs should be translated to canonical `pipeline` before validation.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


# -----------------------------
# Building blocks
# -----------------------------

class _ForbidExtra(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Roots(_ForbidExtra):
    root_5m_ohlc: str | None = None
    root_15m_ohlc: str | None = None


class PeriodBlock(_ForbidExtra):
    name: str
    start: str
    end: str


class PeriodsThreeBlock(_ForbidExtra):
    p1: PeriodBlock
    p2: PeriodBlock
    p3: PeriodBlock


class Periods(_ForbidExtra):
    mode: Literal["three_block", "yearly"] = "three_block"
    score_exclude: list[str] = Field(default_factory=list)
    three_block: PeriodsThreeBlock | None = None


class Costs(_ForbidExtra):
    fee_per_lot: float = 0.0
    spread_per_lot: float = 0.0


class Trend(_ForbidExtra):
    fast: int = 30
    slow: int = 75

    @model_validator(mode="after")
    def _check_fast_slow(self) -> "Trend":
        if self.fast <= 0 or self.slow <= 0 or self.fast >= self.slow:
            raise ValueError("trend.fast/slow must satisfy 0 < fast < slow")
        return self


class TimeFilterFOMC(_ForbidExtra):
    days_csv: str = "quantlab/data/econ_calendar/fomc_decision_days.csv"
    utc_hhmm: str = "19:00"
    pre_hours: float = 2.0
    post_hours: float = 0.5


class TimeFilterEconCalendar(_ForbidExtra):
    csv: str = "quantlab/data/econ_calendar/usd_important_events.csv"
    rules: dict[str, Any] = Field(default_factory=dict)


class TimeFilterMonths(_ForbidExtra):
    block: list[int] = Field(default_factory=list)

    @field_validator("block")
    @classmethod
    def _check_months(cls, v: list[int]) -> list[int]:
        for m in v:
            if not (1 <= int(m) <= 12):
                raise ValueError(f"Invalid month in time_filter.months.block: {m}")
        return [int(x) for x in v]


class TimeFilter(_ForbidExtra):
    kind: Literal["fomc", "econ_calendar"] = "fomc"
    fomc: TimeFilterFOMC | None = None
    econ_calendar: TimeFilterEconCalendar | None = None
    months: TimeFilterMonths | None = None


# -----------------------------
# Gate specs (pipeline)
# -----------------------------

class GateSpec(_ForbidExtra):
    gate: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("gate")
    @classmethod
    def _nonempty_gate(cls, v: str) -> str:
        v = str(v).strip()
        if not v:
            raise ValueError("gate must be a non-empty string")
        return v


# Per-gate param models.
#
# Policy (per your preference):
# - No semantic validators.
# - No special handling for open-ended rule dicts.
# - Still forbid unknown keys everywhere (`extra="forbid"`).


class EmptyParams(_ForbidExtra):
    pass


class HTFConfirmParams(_ForbidExtra):
    fast: int = 30
    slow: int = 75


class GateHTFConfirm(_ForbidExtra):
    gate: Literal["htf_confirm"] = "htf_confirm"
    params: HTFConfirmParams = Field(default_factory=HTFConfirmParams)


class EMASeparationParams(_ForbidExtra):
    ema_fast: int
    ema_slow: int
    atr_n: int
    sep_k: float


class GateEmaSep(_ForbidExtra):
    gate: Literal["ema_sep"] = "ema_sep"
    params: EMASeparationParams


class NoChopParams(_ForbidExtra):
    ema: int
    lookback: int
    min_closes: int
    entry_held: bool = False


class GateNoChop(_ForbidExtra):
    gate: Literal["nochop"] = "nochop"
    params: NoChopParams


class CorrParams(_ForbidExtra):
    logic: str = "or"
    xag_window: int = 40
    xag_min_abs: float = 0.10
    xag_flip_lookback: int = 50
    xag_max_flips: int = 0
    eur_window: int = 75
    eur_min_abs: float = 0.10
    eur_flip_lookback: int = 75
    eur_max_flips: int = 5
    confirm_size_one: float = 1.0
    confirm_size_both: float = 2.0


class GateCorr(_ForbidExtra):
    gate: Literal["corr"] = "corr"
    params: CorrParams = Field(default_factory=CorrParams)


class GateTimeFilter(_ForbidExtra):
    gate: Literal["time_filter"] = "time_filter"
    params: EmptyParams = Field(default_factory=EmptyParams)


class EmaStrengthSizingParams(_ForbidExtra):
    ema_fast: int
    ema_slow: int
    atr_n: int
    strong_k: float
    size_base: float
    size_strong: float


class GateEmaStrengthSizing(_ForbidExtra):
    gate: Literal["ema_strength_sizing"] = "ema_strength_sizing"
    params: EmaStrengthSizingParams


class SeasonalityCapParams(_ForbidExtra):
    month_size_cap: dict[int, float] | None = None


class GateSeasonalityCap(_ForbidExtra):
    gate: Literal["seasonality_cap"] = "seasonality_cap"
    params: SeasonalityCapParams = Field(default_factory=SeasonalityCapParams)


class ChurnParams(_ForbidExtra):
    min_on_bars: int
    cooldown_bars: int


class GateChurn(_ForbidExtra):
    gate: Literal["churn"] = "churn"
    params: ChurnParams


class MidLossLimiterParams(_ForbidExtra):
    min_bars: int
    max_bars: int
    stop_ret: float


class GateMidLossLimiter(_ForbidExtra):
    gate: Literal["mid_loss_limiter"] = "mid_loss_limiter"
    params: MidLossLimiterParams


class NoRecoveryExitParams(_ForbidExtra):
    bar_n: int
    min_ret: float


class GateNoRecoveryExit(_ForbidExtra):
    gate: Literal["no_recovery_exit"] = "no_recovery_exit"
    params: NoRecoveryExitParams


class ProfitMilestoneParams(_ForbidExtra):
    bar_n: int = 24
    milestone_ret: float = 0.002


class GateProfitMilestone(_ForbidExtra):
    gate: Literal["profit_milestone"] = "profit_milestone"
    params: ProfitMilestoneParams = Field(default_factory=ProfitMilestoneParams)


class RollingMaxExitParams(_ForbidExtra):
    window_bars: int = 24
    min_bars: int = 24
    min_peak_ret: float = 0.0


class GateRollingMaxExit(_ForbidExtra):
    gate: Literal["rolling_max_exit"] = "rolling_max_exit"
    params: RollingMaxExitParams = Field(default_factory=RollingMaxExitParams)


class ShockExitParams(_ForbidExtra):
    shock_exit_abs_ret: float = 0.0
    shock_exit_sigma_k: float = 0.0
    shock_exit_sigma_window: int = 96
    shock_cooldown_bars: int = 0


class GateShockExit(_ForbidExtra):
    gate: Literal["shock_exit"] = "shock_exit"
    params: ShockExitParams = Field(default_factory=ShockExitParams)


PipelineGate = (
    GateHTFConfirm
    | GateEmaSep
    | GateNoChop
    | GateCorr
    | GateTimeFilter
    | GateEmaStrengthSizing
    | GateSeasonalityCap
    | GateChurn
    | GateMidLossLimiter
    | GateNoRecoveryExit
    | GateProfitMilestone
    | GateRollingMaxExit
    | GateShockExit
)


SCHEMA_GATE_NAMES: set[str] = {
    "htf_confirm",
    "ema_sep",
    "nochop",
    "corr",
    "time_filter",
    "ema_strength_sizing",
    "seasonality_cap",
    "churn",
    "mid_loss_limiter",
    "no_recovery_exit",
    "profit_milestone",
    "rolling_max_exit",
    "shock_exit",
}


class QuantlabConfig(_ForbidExtra):
    symbol: str = "XAUUSD"
    corr_symbol: str = "XAGUSD"
    corr2_symbol: str = "EURUSD"

    roots: Roots | None = None
    periods: Periods = Field(default_factory=Periods)
    costs: Costs = Field(default_factory=Costs)
    trend: Trend = Field(default_factory=Trend)
    time_filter: TimeFilter | None = None

    pipeline: list[PipelineGate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_periods(self) -> "QuantlabConfig":
        # three_block mode requires blocks
        if self.periods.mode == "three_block" and self.periods.three_block is None:
            raise ValueError("periods.three_block must be provided when periods.mode=three_block")
        return self


def validate_config_dict(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a raw YAML config dict.

    Returns a plain dict (safe to pass to existing code).
    Raises pydantic.ValidationError on failure.
    """
    obj = QuantlabConfig.model_validate(cfg)
    # model_dump produces JSON-serializable primitives (good for stability)
    return obj.model_dump(mode="python")


def validate_config_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must parse to a dict, got {type(cfg)}")
    return validate_config_dict(cfg)


# -----------------------------
# CLI (same script)
# -----------------------------

def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate Quantlab YAML config (Pydantic schema)")
    ap.add_argument("path", help="Path to YAML config (e.g. quantlab/configs/trend_based/current.yaml)")
    ap.add_argument("--json", action="store_true", help="Print normalized config as JSON")
    ap.add_argument(
        "--no-sync-check",
        action="store_true",
        help="Disable gate registry ↔ schema coverage check",
    )
    args = ap.parse_args(argv)

    if not args.no_sync_check:
        try:
            from quantlab.strategies.gates import registered_gates

            reg = set(registered_gates())
            missing = sorted(reg - SCHEMA_GATE_NAMES)
            extra = sorted(SCHEMA_GATE_NAMES - reg)
            if missing or extra:
                print("SCHEMA/REGISTRY MISMATCH:\n")
                if missing:
                    print(f"- Gates registered in code but missing in schema: {missing}")
                if extra:
                    print(f"- Gates present in schema but not registered in code: {extra}")
                return 3
        except Exception as e:
            print(f"WARN: sync check failed: {type(e).__name__}: {e}")

    try:
        cfg = validate_config_file(args.path)
    except ValidationError as e:
        # Human-friendly errors.
        print(f"INVALID: {args.path}\n")
        print(e)
        return 1
    except Exception as e:
        print(f"ERROR: {args.path}\n{type(e).__name__}: {e}")
        return 2

    if args.json:
        print(json.dumps(cfg, indent=2, sort_keys=True))
    else:
        print(f"OK: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
