import numpy as np
import pandas as pd

from costmodel.baselines import GlobalMean, GroupStat, LogLinear
from costmodel.evaluate import monthly_totals, summary
from costmodel.features import build_features


def feature_frame(n: int, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    proc = rng.choice(["0303060190", "0411010034", "0310010039"], size=n)
    raw = pd.DataFrame(
        {
            "admission_id": [str(i) for i in range(n)], "admission_date": pd.Timestamp("2022-01-01")
            + pd.to_timedelta(rng.integers(0, 365, n), unit="D"), "age_years": rng.integers(0, 90, n).astype(float),
            "sex": rng.choice(["M", "F"], n), "diagnosis_icd10": "I219", "requested_procedure": proc,
            "procedure_code_system": "SIGTAP", "admission_type": rng.choice(["elective", "urgent"], n),
            "provider_id": "2273411", "residence_area": "330455", "bed_specialty": "03",
        }
    )  # fmt: skip
    X = build_features(raw)
    price = pd.Series(proc).map({"0303060190": 1000.0, "0411010034": 600.0, "0310010039": 300.0})
    y = price * rng.lognormal(0, 0.2, n)
    return X, y


def test_group_median_recovers_procedure_price_and_backs_off():
    X, y = feature_frame(3000)
    model = GroupStat("m", [["requested_procedure"], ["procedure_prefix4"]], "median", min_count=30).fit(X, y)
    pred = model.predict(X.head(3))
    expected = X.head(3)["requested_procedure"].map({"0303060190": 1000.0, "0411010034": 600.0, "0310010039": 300.0})
    np.testing.assert_allclose(pred, expected, rtol=0.05)

    unseen = X.head(1).assign(
        requested_procedure="0303069999", procedure_prefix4="0303"
    )  # unknown procedure, known subgroup 0303
    assert abs(model.predict(unseen)[0] - 1000.0) < 50
    unknown = X.head(1).assign(requested_procedure="9999999999", procedure_prefix4="9999")
    assert model.predict(unknown)[0] == model.global_


def test_log_linear_beats_global_mean():
    X, y = feature_frame(3000)
    Xv, yv = feature_frame(1000, seed=1)
    lin = LogLinear().fit(X, y).predict(Xv)
    mean = GlobalMean().fit(X, y).predict(Xv)
    assert np.mean(np.abs(lin - yv)) < np.mean(np.abs(mean - yv))


def test_monthly_totals_error():
    out = monthly_totals(np.array([100.0, 100.0, 50.0]), np.array([110.0, 110.0, 50.0]), pd.Series(["a", "a", "b"]))
    assert out.loc["a", "error_pct"] == 10.0
    assert out.loc["b", "error_pct"] == 0.0


def test_summary_keys():
    y = np.arange(1, 101, dtype=float)
    s = summary(y, y, pd.Series(["2022-01"] * 100))
    assert s["mae"] == 0 and s["monthly_total_mape"] == 0 and s["mae_top_decile"] == 0


def test_group_mean_matches_totals_better_than_median_on_skewed_costs():
    X, y = feature_frame(3000)
    y = y * np.random.default_rng(2).lognormal(0, 1.0, len(y))  # heavy right tail
    levels = [["requested_procedure"]]
    median = GroupStat("med", levels, "median").fit(X, y).predict(X)
    mean = GroupStat("mean", levels, "mean").fit(X, y).predict(X)
    assert abs(mean.sum() - y.sum()) < abs(median.sum() - y.sum())
    assert np.mean(np.abs(median - y)) < np.mean(np.abs(mean - y))
