"""The explorer's numbers: totals by group and month, small groups pooled, admissions only as distributions."""

import numpy as np
import pandas as pd
import pytest

from costmodel.explore import ERROR_BANDS, OTHER, admission_view, group_series

MONTHS = ["2024-01", "2024-02"]


def frame():
    # Hospital A in both months, B only in January, C and D small.
    rows = [
        ("A", "2024-01", 100.0, 110.0, 5.0),
        ("A", "2024-01", 300.0, 290.0, 20.0),
        ("A", "2024-02", 200.0, 200.0, 0.0),
        ("B", "2024-01", 150.0, 100.0, 50.0),
        ("C", "2024-02", 10.0, 20.0, 10.0),
        ("D", "2024-01", 5.0, 5.0, 0.0),
    ]
    df = pd.DataFrame(rows, columns=["group", "month", "paid", "expected", "abs_error"])
    return df, df.pop("group")


def test_groups_are_summed_by_month_and_ordered_by_money():
    df, group = frame()
    series = group_series(df, group, MONTHS, top=10, min_admissions=1)
    assert [g["name"] for g in series] == ["A", "B", "C", "D"]  # by total paid
    a, b = series[0], series[1]
    assert a["n"] == [2, 1] and a["paid"] == [400.0, 200.0]
    assert a["expected"] == [400.0, 200.0] and a["abs_error"] == [25.0, 0.0]
    assert b["n"] == [1, None] and b["paid"] == [150.0, None]  # no admissions in February: empty, not zero
    assert all(g["pooled"] == 0 for g in series)


def test_groups_beyond_the_top_are_pooled():
    df, group = frame()
    series = group_series(df, group, MONTHS, top=2, min_admissions=1)
    assert [g["name"] for g in series] == ["A", "B", OTHER]
    other = series[-1]
    assert other["pooled"] == 2  # C and D
    assert other["n"] == [1, 1] and other["paid"] == [5.0, 10.0]
    # Nothing is lost: every month still adds up to all the admissions.
    for month in range(2):
        assert sum(g["paid"][month] or 0 for g in series) == df.loc[df["month"] == MONTHS[month], "paid"].sum()


def test_small_groups_are_pooled_whatever_their_money():
    df, group = frame()
    series = group_series(df, group, MONTHS, top=10, min_admissions=3)  # only A has three admissions
    assert [g["name"] for g in series] == ["A", OTHER] and series[-1]["pooled"] == 3
    assert series[-1]["n"] == [2, 1] and series[-1]["paid"] == [155.0, 10.0]


def test_admissions_are_shown_only_as_distributions():
    rng = np.random.default_rng(0)
    y = rng.lognormal(6, 1, 5_000)
    typical, expected = y * rng.uniform(0.5, 1.6, 5_000), y * 1.1
    view = admission_view(y, typical, expected)
    assert view["n"] == 5_000 and len(view["deciles"]) == 10
    assert [b["label"] for b in view["error_bands"]] == [label for _, label in ERROR_BANDS]
    assert sum(b["share_pct"] for b in view["error_bands"]) == pytest.approx(100)
    assert view["mae"] == pytest.approx(np.abs(typical - y).mean())
    top = view["deciles"][-1]
    assert top["mean_paid"] > view["deciles"][0]["mean_paid"]
    assert top["mean_expected"] == pytest.approx(1.1 * top["mean_paid"])  # the expected cost, decile by decile
    assert set(view) == {"n", "mae", "median_error", "deciles", "error_bands"}  # nothing per admission
