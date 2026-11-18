"""The recent price uses only bills that were already known at admission; the comparison registers nothing."""

import json

import numpy as np
import pandas as pd
import pytest

from costmodel import config as config_module
from costmodel import features as F
from costmodel import recent_price, registry, train
from costmodel.config import Config, SplitConfig, TrainConfig
from costmodel.recent_price import COLUMN, PricedModel, recent_mean, reference_price
from costmodel.sources import build_table
from costmodel.synthetic import generate, write


def test_recent_mean_reads_only_the_window_that_ended_gap_months_before():
    # One code, one admission a month, paying 10, 20, 30, ... in months 1, 2, 3, ...
    month = np.arange(1, 11)
    paid = 10.0 * month
    got = recent_mean(paid, month, np.zeros(10, dtype=int), window=2, gap=4, min_count=1)
    assert np.isnan(got[:4]).all()  # months 1-4 have nothing that old
    assert got[4] == 10.0  # month 5 reads month 1 only (the window is not full yet)
    assert got[5] == 15.0  # month 6 reads months 1 and 2
    assert got[9] == 55.0  # month 10 reads months 5 and 6, nothing later


def test_recent_mean_needs_enough_admissions_and_keeps_codes_apart():
    month = np.array([1, 1, 1, 1, 5, 5])
    code = np.array(["a", "a", "a", "b", "a", "b"])
    paid = np.array([10.0, 20.0, 30.0, 500.0, 1.0, 1.0])
    got = recent_mean(paid, month, code, window=3, gap=4, min_count=3)
    assert got[4] == 20.0  # code a: three admissions in month 1
    assert np.isnan(got[5])  # code b: one admission is not enough for a price of its own


def test_later_bills_never_change_an_earlier_admissions_price():
    df = generate(6_000, "2022-01-01", "2023-12-31", seed=3)
    X = F.build_features(df)
    month = X["admission_period"].to_numpy(dtype=int)
    paid = df[F.TARGET].to_numpy(dtype=float)
    cut, gap = int(np.median(month)), 4
    changed = np.where(month >= cut, paid * 100, paid)  # everything billed from `cut` on becomes 100 times dearer
    before, level = reference_price(X, paid, window=3, gap=gap, min_count=5)
    after, _ = reference_price(X, changed, window=3, gap=gap, min_count=5)
    untouched = month < cut + gap  # these admissions were admitted before any changed bill was known
    assert np.array_equal(before[untouched], after[untouched], equal_nan=True)
    assert not np.allclose(before[~untouched], after[~untouched])
    assert (level[month < month.min() + gap] == -1).all()  # no billed history at all in the first months
    assert (level[month >= month.min() + gap + 1] >= 0).all() and (level == 0).any()


@pytest.mark.parametrize("relative", [False, True])
def test_priced_model_predicts_on_the_cost_scale(relative):
    df = generate(12_000, "2022-01-01", "2023-12-31", seed=8)
    X, y = F.build_features(df), df[F.TARGET]
    X[COLUMN], _ = reference_price(X, y, window=3, gap=4, min_count=5)
    is_train = (df["admission_date"] < "2023-07-01").to_numpy()
    model = PricedModel("log_l2", 15, 0.1, 100, 5, 42, True, relative=relative)
    model.fit(X[is_train], y[is_train], X[~is_train], y[~is_train], 20)
    assert COLUMN in model.columns
    pred, y_valid = model.predict(X[~is_train]), y[~is_train]
    assert np.isfinite(pred).all() and (pred > 0).all()
    assert 0.5 < pred.mean() / y_valid.mean() < 2
    assert np.abs(pred - y_valid).mean() < np.abs(y[is_train].mean() - y_valid).mean()
    if relative:  # the unit of the prediction: a recent price twice as high moves it, even with every other input equal
        dearer = X[~is_train].assign(**{COLUMN: lambda d: d[COLUMN] * 2})
        assert (model.predict(dearer) > pred).mean() > 0.9


@pytest.mark.slow
def test_comparison_uses_the_search_setting_and_registers_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(registry, "HOME", tmp_path)
    monkeypatch.setattr(config_module, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(recent_price, "MIN_COUNT", 5)
    write(tmp_path / "admissions.csv", 12_000, "2022-01-01", "2024-12-31", 12)
    cfg = Config(
        name="recent_price_test",
        source={"type": "file", "path": str(tmp_path / "admissions.csv")},
        split=SplitConfig(mode="relative", valid_months=6, test_months=6, monitor_months=6),
        train=TrainConfig(objectives=("log_l2",), num_leaves=(15,), use_time=(True,), min_category_count=5, max_rounds=200),
    )
    build_table(cfg)
    with pytest.raises(RuntimeError, match="costmodel train"):
        recent_price.run(cfg)

    version = train.run(cfg)
    searched = json.loads((cfg.reports_dir / "train_valid.json").read_text())[0]
    results = recent_price.run(cfg)
    assert len(results) == 1 + 2 * len(recent_price.WINDOWS)
    assert results.loc[0, "mae"] == pytest.approx(searched["mae"])  # the candidate as the search measured it
    assert (results["mae"] > 0).all() and results["monthly_total_mape"].notna().all()
    assert results["gain_share_pct"].iloc[1:].between(0, 100).all()
    saved = json.loads((cfg.reports_dir / "recent_price_valid.json").read_text())
    assert saved["gap_months"] == cfg.monitor.label_delay_months + 1  # same rule as the monitor's billing delay
    assert "not retrained" in (cfg.reports_dir / "recent_price_valid.md").read_text(encoding="utf-8")
    assert str(registry.version_for(cfg, registry.STAGING).version) == version
    assert registry.version_for(cfg, registry.PRODUCTION) is None


def test_reference_price_falls_back_to_coarser_groups():
    # Two procedures of the same subgroup: one common, one rare. The rare one takes the subgroup's price.
    n = 40
    X = pd.DataFrame(
        {
            "requested_procedure": ["0101A"] * n + ["0101B"] * 2,
            "procedure_prefix4": ["0101"] * (n + 2),
            "procedure_prefix2": ["01"] * (n + 2),
            "admission_period": [1.0] * (n // 2) + [5.0] * (n // 2) + [1.0, 5.0],
        }
    )
    paid = np.array([100.0] * n + [900.0, 900.0])
    price, level = reference_price(X, paid, window=3, gap=4, min_count=10)
    later = X["admission_period"].to_numpy() == 5.0
    assert (level[later][:-1] == 0).all() and price[later][0] == 100.0  # the common procedure: its own price
    assert level[-1] == 1  # the rare one: its subgroup's, which includes its own single earlier bill
    assert price[-1] == pytest.approx((100.0 * 20 + 900.0) / 21)
