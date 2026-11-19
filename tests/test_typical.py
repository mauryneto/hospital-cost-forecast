"""The typical cost as a median: same setting, four ways to give the number; validation only, nothing registered."""

import json

import numpy as np
import pandas as pd
import pytest

from costmodel import config as config_module
from costmodel import features as F
from costmodel import registry, train, typical
from costmodel.candidate import CostModel
from costmodel.config import Config, SplitConfig, TrainConfig
from costmodel.sources import build_table
from costmodel.synthetic import generate, write
from costmodel.typical import SCALED, UNSCALED, MedianModel, bands, lead, money_share_top_decile


@pytest.fixture(scope="module")
def data():
    df = generate(12_000, "2022-01-01", "2023-12-31", seed=8)
    train_part, valid_part = df[df["admission_date"] < "2023-07-01"], df[df["admission_date"] >= "2023-07-01"]
    return F.build_features(train_part), train_part[F.TARGET], F.build_features(valid_part), valid_part[F.TARGET]


@pytest.mark.parametrize("log_target", [True, False])
def test_median_model_splits_the_admissions_in_half_and_is_not_scaled(data, log_target):
    X, y, X_valid, y_valid = data
    model = MedianModel(log_target, num_leaves=15, min_category_count=5).fit(X, y, X_valid, y_valid, 20)
    assert model.smearing_ == 1.0 and model.name == f"lightgbm_median_{'log' if log_target else 'cost'}"
    pred = model.predict(X_valid)
    assert np.isfinite(pred).all() and (pred > 0).all()
    assert 0.4 < (y_valid.to_numpy() > pred).mean() < 0.6  # a median: about half of the admissions cost more
    # ...which the scaled number is not: it sits above the median, so fewer admissions cost more than it says.
    scaled = CostModel("log_l2", 15, min_category_count=5).fit(X, y, X_valid, y_valid, 20)
    assert scaled.smearing_ > 1.0 and (y_valid.to_numpy() > scaled.predict(X_valid)).mean() < (y_valid > pred).mean()
    unseen = X_valid.head(50).assign(requested_procedure="NEVER-SEEN", provider_id="NEW-HOSPITAL")
    assert np.isfinite(model.predict(unseen)).all()


def test_bands_split_the_error_between_the_expensive_tenth_and_the_rest():
    paid = np.array([10.0] * 18 + [1000.0, 2000.0])  # twenty admissions: the two most expensive are the top tenth
    pred = np.array([12.0] * 18 + [700.0, 1500.0])
    out = bands(paid, pred)
    assert out["mae_common"] == pytest.approx(2.0)  # the other eighteen are off by 2 each
    assert out["error_share_top_decile"] == pytest.approx(100 * 800 / (800 + 36))
    assert money_share_top_decile(paid) == pytest.approx(100 * 3000 / 3180)


def test_lead_is_measured_in_percent_of_the_baseline():
    assert lead(98.0, 100.0) == pytest.approx(2.0) and lead(105.0, 100.0) == pytest.approx(-5.0)
    assert lead(98.0, None) is None


@pytest.mark.slow
def test_typical_compares_four_numbers_and_registers_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(registry, "HOME", tmp_path)
    monkeypatch.setattr(config_module, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(typical, "DRAWS", 30)
    write(tmp_path / "admissions.csv", 12_000, "2022-01-01", "2024-12-31", 12)
    cfg = Config(
        name="typical_test",
        source={"type": "file", "path": str(tmp_path / "admissions.csv")},
        split=SplitConfig(mode="relative", valid_months=6, test_months=6, monitor_months=6),
        train=TrainConfig(objectives=("log_l2",), num_leaves=(15,), use_time=(True,), min_category_count=5, max_rounds=200),
    )
    build_table(cfg)
    with pytest.raises(RuntimeError, match="costmodel train"):
        typical.run(cfg)  # the setting comes from the search, so the search must have run

    version = train.run(cfg)
    results = typical.run(cfg)
    assert results["number"].tolist() == [SCALED, UNSCALED, "median of log(cost)", "median of cost"]
    searched = json.loads((cfg.reports_dir / "train_valid.json").read_text())[0]
    scaled, unscaled = results.iloc[0], results.iloc[1]
    # The two numbers are the ones `costmodel train` reported for this setting, with and without the factor.
    assert scaled["mae"] == pytest.approx(searched["mae"])
    assert unscaled["mae"] == pytest.approx(searched["mae_unscaled"])
    assert pd.isna(scaled["mae_gap"]) and results["mae_gap"].iloc[1:].notna().all()
    # Without the factor every prediction is lower by the same share, so the total falls and the trees are the same.
    assert unscaled["bias_pct"] < scaled["bias_pct"] and unscaled["trees"] == scaled["trees"]
    assert unscaled["mae_gap"] == pytest.approx(unscaled["mae"] - scaled["mae"])
    assert (results["mae_gap_low"].iloc[1:] <= results["mae_gap"].iloc[1:]).all()
    assert results["lead_over_baseline_pct"].isna().all()  # `costmodel baselines` did not run here

    # The two bands are the same error read apart: together they give the MAE back (a decile is about a tenth).
    apart = 0.9 * results["mae_common"] + 0.1 * results["mae_top_decile"]
    assert apart.to_numpy() == pytest.approx(results["mae"].to_numpy(), rel=1e-2)
    assert (results["mae_common"] < results["mae"]).all() and results["error_share_top_decile"].between(10, 100).all()

    saved = json.loads((cfg.reports_dir / "typical_valid.json").read_text())
    assert 10 < saved["money_share_top_decile"] < 100
    assert saved["factor"] > 1.0 and saved["baseline"] is None and len(saved["rows"]) == 4
    assert saved["in_use"] == SCALED  # this config keeps the default: the typical cost with the factor
    assert saved["best"] == results.sort_values("mae").iloc[0]["number"]
    report = (cfg.reports_dir / "typical_valid.md").read_text(encoding="utf-8")
    assert "## Typical admissions and the most expensive ones" in report
    assert "## Against the scaled number" in report and "nothing was registered" in report
    assert "gives the typical cost as the **scaled** number" in report
    # A comparison only: the registry still has just the version the search registered, and nothing in production.
    assert str(registry.version_for(cfg, registry.STAGING).version) == version
    assert registry.version_for(cfg, registry.PRODUCTION) is None
