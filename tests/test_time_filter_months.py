import datetime as dt

import pandas as pd

from quantlab.webui.runner import load_time_filter_mask
from quantlab.webui.config import WORKSPACE


def test_time_filter_months_blocks_june() -> None:
    idx = pd.date_range("2026-05-31", periods=4, freq="1D", tz="UTC")
    cfg = {
        "time_filter": {
            "kind": "fomc",
            "fomc": {
                # point to a non-existent file -> mask=None, months overlay should still work
                "days_csv": "quantlab/data/econ_calendar/DOES_NOT_EXIST.csv",
            },
            "months": {"block": [6]},
        }
    }

    m = load_time_filter_mask(idx, dt.date(2026, 5, 31), dt.date(2026, 6, 3), cfg=cfg, workspace=WORKSPACE)
    assert m is not None
    assert m.astype(bool).tolist() == [True, False, False, False]
