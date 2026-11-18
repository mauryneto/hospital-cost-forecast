"""Other model families get the same data, features and metrics as the candidate; the comparison registers nothing."""

import json

import numpy as np
import pytest

from costmodel import config as config_module
from costmodel import families, registry, train
from costmodel import features as F
from costmodel.config import Config, SplitConfig, TrainConfig
from costmodel.families import CatBoostModel, ForestModel
from costmodel.sources import build_table
from costmodel.synthetic import generate


@pytest.fixture(scope="module")
def data():
    df = generate(12_000, "2022-01-01", "2023-12-31", seed=8)
    train_part, valid_part = df[df["admission_date"] < "2023-07-01"], df[df["admission_date"] >= "2023-07-01"]
    return F.build_features(train_part), train_part[F.TARGET], F.build_features(valid_part), valid_part[F.TARGET]


@pytest.mark.parametrize("log_target", [True, False])
def test_forest_beats_the_global_mean_and_survives_new_codes(data, log_target):
    X, y, X_valid, y_valid = data
    model = ForestModel(log_target, n_trees=30, min_category_count=5).fit(X, y, X_valid, y_valid, 10)
    assert model.best_iteration_ == 30  # no early stopping: every tree of a forest is independent
    pred = model.predict(X_valid)
    assert np.isfinite(pred).all() and (pred > 0).all()
    assert np.abs(pred - y_valid).mean() < np.abs(y.mean() - y_valid).mean()
    unseen = X_valid.head(50).assign(requested_procedure="NEVER-SEEN", provider_id="NEW-HOSPITAL")
    assert np.isfinite(model.predict(unseen)).all()


@pytest.mark.parametrize("loss", ["log_rmse", "tweedie"])
def test_catboost_predicts_on_the_cost_scale(data, loss):
    pytest.importorskip("catboost")
    X, y, X_valid, y_valid = data
    model = CatBoostModel(loss, iterations=60).fit(X, y, X_valid, y_valid, 20)
    pred = model.predict(X_valid)
    assert np.isfinite(pred).all() and (pred > 0).all()
    # Same scale as the cost (a doubled exponential, or none, would be off by orders of magnitude).
    assert 0.5 < pred.mean() / y_valid.mean() < 2
    assert np.abs(pred - y_valid).mean() < np.abs(y.mean() - y_valid).mean()


def test_catboost_rejects_an_unknown_loss():
    with pytest.raises(ValueError, match="loss must be"):
        CatBoostModel("mae")


@pytest.mark.slow
def test_comparison_uses_the_search_result_and_registers_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(registry, "HOME", tmp_path)
    monkeypatch.setattr(config_module, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(families, "FOREST_TREES", 20)
    monkeypatch.setattr(families, "catboost_available", lambda: False)  # the same result with or without the extra
    from costmodel.synthetic import write

    write(tmp_path / "admissions.csv", 12_000, "2022-01-01", "2024-12-31", 12)
    cfg = Config(
        name="families_test",
        source={"type": "file", "path": str(tmp_path / "admissions.csv")},
        split=SplitConfig(mode="relative", valid_months=6, test_months=6, monitor_months=6),
        train=TrainConfig(objectives=("tweedie",), num_leaves=(15,), use_time=(True,), min_category_count=5, max_rounds=200),
    )
    build_table(cfg)
    with pytest.raises(RuntimeError, match="costmodel train"):
        families.run(cfg)  # the LightGBM row comes from the search, so the search must have run

    version = train.run(cfg)
    results = families.run(cfg)
    assert results["family"].tolist() == ["LightGBM (gradient boosting)", "random forest", "random forest"]
    searched = json.loads((cfg.reports_dir / "train_valid.json").read_text())[0]
    assert results.loc[0, "mae"] == pytest.approx(searched["mae"])  # the number `costmodel train` reported, not a refit
    assert (results["mae"] > 0).all() and results["monthly_total_mape"].notna().all()
    report = (cfg.reports_dir / "families_valid.md").read_text(encoding="utf-8")
    assert "CatBoost is not installed" in report
    assert len(json.loads((cfg.reports_dir / "families_valid.json").read_text())) == 3
    # A comparison only: the registry still has just the version the search registered, and nothing in production.
    assert str(registry.version_for(cfg, registry.STAGING).version) == version
    assert registry.version_for(cfg, registry.PRODUCTION) is None
