"""Time-based split by admission date: train on the past, choose on validation, report on test, replay as monitoring.

Each admission falls in exactly one window, so the same admission can never be in both train and test.
"""

from __future__ import annotations

import pandas as pd

from costmodel.config import SplitConfig

SPLITS = ["train", "valid", "test", "monitor"]


def windows(admission_date: pd.Series, cfg: SplitConfig) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    if cfg.mode == "dates":
        return {name: (pd.Timestamp(w[0]), pd.Timestamp(w[1])) for name in SPLITS if (w := getattr(cfg, name))}
    if cfg.mode != "relative":
        raise ValueError(f"unknown split mode {cfg.mode!r}; use 'dates' or 'relative'")
    # Windows are whole calendar months ending with the last month that has data.
    last_month = admission_date.max().to_period("M")
    out: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    end = last_month
    for name, months in (("monitor", cfg.monitor_months), ("test", cfg.test_months), ("valid", cfg.valid_months)):
        start = end - (months - 1)
        out[name] = (start.start_time, end.end_time.normalize())
        end = start - 1
    out["train"] = (admission_date.min().normalize(), end.end_time.normalize())
    return {name: out[name] for name in SPLITS}


def assign(admission_date: pd.Series, cfg: SplitConfig) -> pd.Series:
    """Label each admission with its window; admissions outside every window get <NA>."""
    split = pd.Series(pd.NA, index=admission_date.index, dtype="string")
    for name, (start, end) in windows(admission_date, cfg).items():
        inside = admission_date.between(start, end)
        if (inside & split.notna()).any():
            raise ValueError(f"split window {name!r} overlaps another window")
        split = split.mask(inside, name)
    return split
