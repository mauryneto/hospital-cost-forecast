"""Each retrain sees only admission months already billed; the comparison registers nothing."""

import json

import numpy as np
import pytest

from costmodel import base_window, registry, train
from costmodel import config as config_module
from costmodel import features as F
from costmodel.base_window import blocks, fit_known, recency_weights
from costmodel.candidate import CostModel
from costmodel.config import Config, SplitConfig, TrainConfig
from costmodel.sources import build_table
from costmodel.synthetic import generate, write

SETTING = {"objective": "log_l2", "num_leaves": 15, "use_time": True}


def test_blocks_are_consecutive_runs_of_the_horizon():
    assert blocks([7, 3, 4, 5, 6, 8, 9], horizon=3) == [[3, 4, 5], [6, 7, 8], [9]]


def test_recency_weights_halve_every_half_life():
    assert recency_weights(np.array([24, 18, 12, 0]), known=24, half_life=12).tolist() == [1.0, 0.5**0.5, 0.5, 0.25]


@pytest.fixture(scope="module")
def data():
    df = generate(9_000, "2022-01-01", "2023-06-30", seed=5)
    X = F.build_features(df)
    return X, df[F.TARGET], X["admission_period"].to_numpy(dtype=int)


@pytest.fixture(scope="module")
def small_cfg():
    tc = TrainConfig(objectives=("log_l2",), num_leaves=(15,), use_time=(True,), min_category_count=5, max_rounds=60)
    return Config(name="base_window_unit", source={"type": "file"}, train=tc)


def test_a_retrain_reads_only_the_billed_months_of_its_base(data, small_cfg):
    X, y, month = data
    known = int(np.sort(np.unique(month))[11])  # the 12th month of 18
    model, n = fit_known(small_cfg, SETTING, X, y, month, known, base=4)
    assert n == int(((month <= known) & (month > known - 4)).sum())
    _, n_all = fit_known(small_cfg, SETTING, X, y, month, known, base=0)
    assert n_all == int((month <= known).sum()) > n

    # Costs outside the base (the months not billed yet, and the ones too old) cannot change the model.
    outside = (month > known) | (month <= known - 4)
    other, _ = fit_known(small_cfg, SETTING, X, y.where(~outside, y * 50), month, known, base=4)
    later = X[month > known]
    assert np.array_equal(model.predict(later), other.predict(later))


def test_recent_months_weigh_more_in_the_weighted_variant(data, small_cfg):
    X, y, month = data
    known = int(month.max())
    recent = month > known - 3
    dearer = y.where(~recent, y * 3)  # the last three months are three times dearer
    plain, _ = fit_known(small_cfg, SETTING | {"use_time": False}, X, dearer, month, known, base=0)
    weighted, _ = fit_known(small_cfg, SETTING | {"use_time": False}, X, dearer, month, known, base=0, weighted=True)
    assert weighted.predict(X[recent]).mean() > plain.predict(X[recent]).mean()


def test_equal_weights_change_nothing(data):
    X, y, _ = data
    plain = CostModel("log_l2", 15, 0.1, 30, 5, 42, True).fit(X, y)
    same = CostModel("log_l2", 15, 0.1, 30, 5, 42, True).fit(X, y, sample_weight=np.ones(len(X)))
    assert np.allclose(plain.predict(X), same.predict(X))


@pytest.mark.slow
def test_comparison_retrains_before_each_block_and_registers_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(registry, "HOME", tmp_path)
    monkeypatch.setattr(config_module, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(base_window, "BASES", (4, 0))
    write(tmp_path / "admissions.csv", 12_000, "2022-01-01", "2024-12-31", 12)
    cfg = Config(
        name="base_window_test",
        source={"type": "file", "path": str(tmp_path / "admissions.csv")},
        split=SplitConfig(mode="relative", valid_months=6, test_months=6, monitor_months=6),
        train=TrainConfig(objectives=("log_l2",), num_leaves=(15,), use_time=(True,), min_category_count=5, max_rounds=200),
    )
    build_table(cfg)
    with pytest.raises(RuntimeError, match="costmodel train"):
        base_window.run(cfg)

    version = train.run(cfg)
    searched = json.loads((cfg.reports_dir / "train_valid.json").read_text())[0]
    results = base_window.run(cfg)
    assert results["history"].tolist() == [
        "the candidate as it is (fitted once, not retrained)",
        "last 4 billed months",
        "every billed month",
        base_window.WEIGHTED,
    ]
    assert results.loc[0, "mae"] == pytest.approx(searched["mae"])
    assert results["retrains"].tolist() == [0, 2, 2, 2]  # six validation months, three at a time
    assert results.loc[1, "mean_rows"] < results.loc[2, "mean_rows"] == results.loc[3, "mean_rows"]
    assert (results["mae"] > 0).all() and results["monthly_total_mape"].notna().all()
    saved = json.loads((cfg.reports_dir / "base_window_valid.json").read_text())
    assert saved["gap_months"] == cfg.monitor.label_delay_months + 1  # same rule as the monitor's billing delay
    assert all(len(v) == 2 for v in saved["blocks"].values())
    assert "## Block by block" in (cfg.reports_dir / "base_window_valid.md").read_text(encoding="utf-8")
    assert str(registry.version_for(cfg, registry.STAGING).version) == version
    assert registry.version_for(cfg, registry.PRODUCTION) is None
