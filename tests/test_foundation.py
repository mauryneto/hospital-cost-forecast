"""The foundation-model comparison: how admissions are handed to it as context, and that it registers nothing."""

import json

import numpy as np
import pandas as pd
import pytest

from costmodel import config as config_module
from costmodel import features as F
from costmodel import foundation, registry, train
from costmodel.config import Config, SplitConfig, TrainConfig
from costmodel.foundation import CodeOrder, FoundationModel, StatisticsModel, code_statistics, gap, sample, with_statistics
from costmodel.sources import build_table
from costmodel.synthetic import generate


def test_sample_is_random_without_replacement_and_repeatable():
    df = pd.DataFrame({"x": range(1000)})
    drawn = sample(df, 100, seed=1)
    assert len(drawn) == 100 and drawn["x"].is_unique and drawn["x"].is_monotonic_increasing
    assert drawn.equals(sample(df, 100, seed=1)) and not drawn.equals(sample(df, 100, seed=2))
    assert sample(df, 5000, seed=1) is df  # fewer rows than asked: all of them


def test_codes_become_positions_and_a_new_code_falls_between_its_neighbours():
    context = pd.DataFrame({"requested_procedure": ["0303", "0401", "0301", "0401"], "age_years": [1.0, 2.0, 3.0, 4.0]})
    order = CodeOrder().fit(context)
    assert order.transform(context)[:, 0].tolist() == [1, 2, 0, 2]  # sorted: 0301, 0303, 0401
    new = pd.DataFrame({"requested_procedure": ["0302", "0101", "0999", "0303"], "age_years": [5.0, 6.0, 7.0, 8.0]})
    matrix = order.transform(new)
    assert matrix[:, 0].tolist() == [0.5, -0.5, 2.5, 1]  # between 0301 and 0303; before all; after all; known
    assert matrix[:, 1].tolist() == [5, 6, 7, 8]  # numbers pass through


def test_code_statistics_are_mean_log_costs_and_rare_codes_take_the_overall_mean():
    X = pd.DataFrame({"requested_procedure": ["A"] * 3 + ["B"], "diagnosis_icd10": ["X"] * 4, "provider_id": ["H"] * 4})
    y = np.array([100.0, 100.0, 100.0, 10_000.0])
    learned = code_statistics(X, y, min_count=2)
    overall = np.log(y).mean()
    new = pd.DataFrame({"requested_procedure": ["A", "B", "NEW"], "diagnosis_icd10": ["X"] * 3, "provider_id": ["H"] * 3})
    out = with_statistics(new, learned)
    assert out["requested_procedure_mean_log_cost"].tolist() == pytest.approx([np.log(100), overall, overall])
    assert out["diagnosis_icd10_mean_log_cost"].tolist() == pytest.approx([overall] * 3)
    assert list(out.columns[:3]) == list(new.columns)  # the codes themselves stay


def test_gap_is_measured_on_the_same_admissions_with_an_interval():
    rng = np.random.default_rng(0)
    y = rng.lognormal(6, 1, 4_000)
    month = np.repeat(np.arange(4), 1_000)
    ours = (y * rng.uniform(0.8, 1.2, 4_000), y * 1.0)
    same = gap(y, month, ours, ours, seed=1, draws=50)
    assert set(same.values()) == {0.0}  # a model against itself: no difference in any draw
    worse = (ours[0] + 500.0, y * 1.3)  # R$ 500 further from every admission; 30% high on every month's total
    out = gap(y, month, ours, worse, seed=1, draws=200)
    assert out["mae_gap"] > 0 and out["mae_gap_low"] > 0  # worse, and the interval says so
    assert out["mae_gap_low"] <= out["mae_gap"] <= out["mae_gap_high"]
    assert out["total_gap"] == pytest.approx(30) and out["total_gap_low"] == pytest.approx(30)
    noise = (ours[0] + rng.normal(0, 1, 4_000), ours[1])
    assert gap(y, month, ours, noise, seed=1, draws=200)["mae_gap_low"] < 0 < 1  # too close to call: interval has zero


def test_candidate_accepts_the_code_statistics_as_inputs():
    df = generate(4_000, "2022-01-01", "2023-12-31", seed=3)
    X, y = F.build_features(df), df[F.TARGET]
    Xs = with_statistics(X, code_statistics(X, y, min_count=5))
    model = StatisticsModel("tweedie", 15, 0.1, 30, 5).fit(Xs, y)
    assert model.columns[-3:] == [f"{c}_mean_log_cost" for c in foundation.STAT_COLUMNS]
    assert set(model.feature_importance().index) == set(model.columns)
    assert np.isfinite(model.predict(Xs)).all()


@pytest.fixture(scope="module")
def data():
    df = generate(6_000, "2022-01-01", "2023-12-31", seed=8)
    context = sample(df[df["admission_date"] < "2023-07-01"], 400, 0)
    scored = sample(df[df["admission_date"] >= "2023-07-01"], 300, 0)
    y_scored = scored[F.TARGET].to_numpy(dtype=float)
    return F.build_features(context), context[F.TARGET], F.build_features(scored), y_scored


@pytest.mark.parametrize("log_target", [False, True])
def test_foundation_model_gives_two_numbers_on_the_cost_scale(data, log_target):
    pytest.importorskip("tabpfn")
    X, y, X_scored, y_scored = data
    model = FoundationModel(log_target, chunk_rows=120).fit(X, y)  # 300 admissions in three calls
    typical, expected = model.predict_both(X_scored.assign(requested_procedure="NEVER-SEEN").head(5))
    assert np.isfinite(typical).all() and np.isfinite(expected).all()
    typical, expected = model.predict_both(X_scored)
    assert len(typical) == len(expected) == len(y_scored)
    assert (typical > 0).all() and (expected > 0).all()
    assert np.abs(typical - y_scored).mean() < np.abs(y.mean() - y_scored).mean()  # better than one value for all
    assert expected.mean() > typical.mean()  # costs are skewed: the mean sits above the median
    assert 0.5 < expected.mean() / y_scored.mean() < 2


@pytest.mark.slow
def test_comparison_scores_every_row_on_the_same_sample_and_registers_nothing(tmp_path, monkeypatch):
    pytest.importorskip("tabpfn")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(registry, "HOME", tmp_path)
    monkeypatch.setattr(config_module, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(foundation, "CONTEXT_ROWS", 300)
    monkeypatch.setattr(foundation, "EVAL_ROWS", 200)
    monkeypatch.setattr(foundation, "variants", lambda: [("all", False, False), ("recent", True, True)])
    from costmodel.synthetic import write

    write(tmp_path / "admissions.csv", 12_000, "2022-01-01", "2024-12-31", 12)
    cfg = Config(
        name="foundation_test",
        source={"type": "file", "path": str(tmp_path / "admissions.csv")},
        split=SplitConfig(mode="relative", valid_months=6, test_months=6, monitor_months=6),
        train=TrainConfig(objectives=("tweedie", "log_l2"), num_leaves=(15,), use_time=(True,), min_category_count=5,
                          max_rounds=200),  # fmt: skip
    )
    build_table(cfg)
    with pytest.raises(RuntimeError, match="costmodel train"):
        foundation.run(cfg)  # the LightGBM rows use the search's settings, so the search must have run

    version = train.run(cfg)
    results = foundation.run(cfg)
    assert results["model"].tolist() == ["LightGBM"] * 4 + ["TabPFN-2"] * 2
    assert results["rows"].tolist()[1:] == [300] * 5 and results.loc[0, "rows"] > 300
    assert results.loc[0, list(foundation.GAPS)].isna().all()  # the first row is what the others are compared with
    assert results.loc[1:, list(foundation.GAPS)].notna().all().all()
    assert (results.loc[1:, "mae_gap"] - (results.loc[1:, "mae"] - results.loc[0, "mae"])).abs().max() < 1e-6
    assert (results["mae"] > 0).all() and results["monthly_total_mape"].notna().all()
    payload = json.loads((cfg.reports_dir / "foundation_valid.json").read_text())
    assert payload["eval_rows"] == 200 and payload["context_rows"] == 300 and len(payload["rows"]) == 6
    report = (cfg.reports_dir / "foundation_valid.md").read_text(encoding="utf-8")
    assert "TabPFN-2" in report and "nothing was registered" in report and "95% interval" in report
    # A comparison only: the registry still has just the version the search registered, and nothing in production.
    assert str(registry.version_for(cfg, registry.STAGING).version) == version
    assert registry.version_for(cfg, registry.PRODUCTION) is None
