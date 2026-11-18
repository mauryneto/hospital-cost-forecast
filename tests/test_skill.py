"""The gain over a plain average: the same amount for every admission cancels out for everyone, not for a group."""

import numpy as np
import pandas as pd
import pytest

from costmodel.skill import MODEL, gain, table, totals_error

# Two hospitals in one month. A does cheap admissions (100 each), B dear ones (900 each); the average is 500.
X = pd.DataFrame(
    {
        "provider_id": ["A"] * 4 + ["B"] * 4,
        "admission_type": ["urgent"] * 8,  # one value only: not a level worth reporting
        "procedure_prefix2": ["03"] * 4 + ["04"] * 4,
        "bed_specialty": ["NA"] * 8,
        "diagnosis_chapter": list("IIJJIIJJ"),
        "procedure_prefix4": ["0301"] * 4 + ["0401"] * 4,
    }
)
PERIOD = pd.Series(["2024-01"] * 8)
Y = np.array([100.0] * 4 + [900.0] * 4)
PLAIN = np.full(8, 500.0)


def test_a_plain_average_is_right_for_everyone_and_wrong_for_each_group():
    everyone = pd.DataFrame({"month": PERIOD})
    assert totals_error(Y, PLAIN, everyone) == (0.0, 1)  # 4,000 predicted, 4,000 paid
    by_hospital = X[["provider_id"]].assign(month=PERIOD)
    error, totals = totals_error(Y, PLAIN, by_hospital)
    assert totals == 2
    assert error == pytest.approx(80.0)  # A: 2,000 against 400; B: 2,000 against 3,600; 3,200 off out of 4,000


def test_gain_is_how_much_smaller_the_error_is():
    assert gain(5.0, 20.0) == 75.0 and gain(20.0, 20.0) == 0.0 and gain(30.0, 20.0) == -50.0


def test_table_scores_each_level_and_each_admission():
    typical = Y * 1.1  # 10% high on every admission
    rows = {r["level"]: r for r in table(Y, typical, typical, {"global_mean": PLAIN, "mean_by_proc": Y.copy()}, X, PERIOD)}
    assert "admission type x month" not in rows and "bed specialty x month" not in rows  # a single value: skipped
    everyone = rows["all admissions, each month"]
    assert everyone["global_mean"] == 0.0 and np.isnan(everyone["gain_pct"])  # no gain over a rule that is exactly right
    hospital = rows["hospital x month"]
    assert hospital["totals"] == 2 and hospital["global_mean"] == pytest.approx(80.0)
    assert hospital[MODEL] == pytest.approx(10.0) and hospital["gain_pct"] == pytest.approx(87.5)
    assert hospital["mean_by_proc"] == 0.0
    each = rows["each admission"]
    assert each["what"] == "mean absolute error" and each["totals"] == 8
    assert each[MODEL] == pytest.approx(50.0) and each["global_mean"] == 400.0
