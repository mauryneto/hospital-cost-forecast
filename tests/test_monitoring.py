"""monitor and replay on synthetic data, in a throwaway work folder and MLflow database."""

import json

import pytest

from costmodel import config as config_module
from costmodel import holdout, monitor, registry, replay, train
from costmodel.config import Config, ReplayConfig, SplitConfig, TrainConfig
from costmodel.sources import build_table


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(registry, "HOME", tmp_path)
    monkeypatch.setattr(config_module, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config_module, "REPORTS_DIR", tmp_path / "reports")
    from costmodel.synthetic import write

    write(tmp_path / "admissions.csv", 15_000, "2022-01-01", "2024-12-31", 11)
    cfg = Config(
        name="monitoring_test",
        source={"type": "file", "path": str(tmp_path / "admissions.csv")},
        split=SplitConfig(mode="relative", valid_months=6, test_months=6, monitor_months=6),
        train=TrainConfig(objectives=("tweedie",), num_leaves=(15,), use_time=(True,), min_category_count=5),
        replay=ReplayConfig(policies=("never", "rule", "monthly"), windows=(0,), gate_months=2),
    )
    build_table(cfg)
    train.run(cfg)
    holdout.run(cfg)  # promotes the candidate or, failing that, the best baseline
    return cfg


@pytest.mark.slow
def test_monitor_and_replay(cfg):
    result = monitor.run(cfg)
    assert [m["period"] for m in result["months"]] == [f"2024-{m:02d}" for m in range(7, 13)]
    for m in result["months"]:
        assert set(m["psi"]) >= {"requested_procedure", "age_years", "predicted cost"}
        assert "admission_period" not in m["psi"]  # time drifts by construction
        # Where the error comes from: per admission type, and the price level against what the model last saw.
        assert sum(t["n"] for t in m["by_admission_type"].values()) == m["n"]
        assert -50 < m["price_index_pct"] < 50
    assert all(len(s["monthly_median"]) <= 6 and s["reference_median"] > 0 for s in result["price_shifts"])
    # A month is judged only 3 months after it ends: the first decision that sees July is the end of October.
    first_mature = next(t for t in result["timeline"] if "only 0" not in t["reason"])
    assert first_mature["decision_month"] == "2024-10"
    for name in ("monitor.md", "monitor.png", "monitor.json"):
        assert (cfg.reports_dir / name).exists()

    results = {r["policy"]: r for r in replay.run(cfg)}
    assert results["never"]["attempts"] == 0
    # Tried at the end of July ... November, but only when a month newer than the model in use exists beyond the 2 gate
    # months; after a promotion that takes 3 months, so attempts + waits = 5 and there are waits only after promotions.
    monthly = results["monthly"]
    assert monthly["attempts"] + monthly["waits"] == 5
    assert monthly["attempts"] >= 1
    if monthly["promotions"] == 0:
        assert monthly["waits"] == 0
    for r in results.values():
        assert len(r["monthly"]) == 6
    report = json.loads((cfg.reports_dir / "replay.json").read_text())
    assert len(report) == 3
    assert (cfg.reports_dir / "replay.png").exists()
    # The what-if must not touch the registry.
    assert registry.version_for(cfg, registry.PRODUCTION).tags.get("gate") in {"passed", "fallback"}


@pytest.mark.slow
def test_two_numbers_go_through_the_gate_the_monitor_the_replay_and_scoring(tmp_path, monkeypatch):
    """A candidate with a typical and an expected cost: each number reaches the metric it is for, everywhere."""
    import numpy as np
    import pandas as pd

    from costmodel import features as F
    from costmodel.candidate import EXPECTED, TYPICAL
    from costmodel.score import Scorer
    from costmodel.synthetic import write

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(registry, "HOME", tmp_path)
    monkeypatch.setattr(config_module, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config_module, "REPORTS_DIR", tmp_path / "reports")
    write(tmp_path / "admissions.csv", 15_000, "2022-01-01", "2024-12-31", 11)
    cfg = Config(
        name="two_numbers_test",
        source={"type": "file", "path": str(tmp_path / "admissions.csv")},
        split=SplitConfig(mode="relative", valid_months=6, test_months=6, monitor_months=6),
        train=TrainConfig(objectives=("log_l2", "tweedie"), num_leaves=(15,), use_time=(True,), min_category_count=5),
        replay=ReplayConfig(policies=("never", "monthly"), windows=(0,), gate_months=2),
    )
    build_table(cfg)
    # Whatever this sample would choose, make the two numbers come from two different settings.
    monkeypatch.setattr(train, "choose", lambda results, cfg: results[results["objective"] == "log_l2"].iloc[0])
    monkeypatch.setattr(train, "choose_for_totals", lambda results: results[results["objective"] == "tweedie"].iloc[0])
    version = train.run(cfg)
    tags = registry.version_for(cfg, registry.STAGING).tags
    assert tags["typical_setting"].startswith("log_l2") and tags["expected_setting"].startswith("tweedie")
    assert "**Chosen for the expected cost:** tweedie" in (cfg.reports_dir / "train_valid.md").read_text(encoding="utf-8")

    monkeypatch.setattr(holdout, "should_promote", lambda *args: (True, ["forced for the test"]))
    assert holdout.run(cfg) is True
    test = json.loads((cfg.reports_dir / "test.json").read_text())
    numbers, gated = test["candidate_numbers"], test["metrics"]["candidate"]
    assert gated["mae"] == numbers[TYPICAL]["mae"] != numbers[EXPECTED]["mae"]
    assert gated["monthly_total_mape"] == numbers[EXPECTED]["monthly_total_mape"]
    assert gated["bias_pct"] == numbers[EXPECTED]["bias_pct"]
    report = (cfg.reports_dir / "test.md").read_text(encoding="utf-8")
    assert "Calibration by decile of actual cost (candidate, expected cost)" in report

    # The same split of the two numbers in the monitor, checked against the registered model itself.
    result = monitor.run(cfg)
    table = pd.read_parquet(cfg.table_path)
    july = table[table["admission_date"].dt.to_period("M") == "2024-07"]
    typical, expected = registry.predict_both(registry.load(cfg, registry.PRODUCTION), july)
    month = result["months"][0]
    assert month["period"] == "2024-07"
    assert month["predicted"] == pytest.approx(expected.sum())
    assert month["mae"] == pytest.approx(np.abs(typical - july[F.TARGET].to_numpy()).mean())
    assert not np.allclose(typical, expected)

    replayed = {r["policy"]: r for r in replay.run(cfg)}
    never = replayed["never"]  # the production model all along: the monitor's own numbers
    assert never["monthly"][0]["mae"] == pytest.approx(month["mae"])
    assert never["monthly"][0]["total_error_pct"] == pytest.approx(month["total_error_pct"])
    assert replayed["monthly"]["attempts"] >= 1

    scored = Scorer.production(cfg).score(july.drop(columns=[F.TARGET]))
    assert scored.columns.tolist() == [TYPICAL, EXPECTED] and len(scored) == len(july)
    np.testing.assert_allclose(scored[EXPECTED], expected)
    assert str(registry.version_for(cfg, registry.PRODUCTION).version) == version


@pytest.mark.slow
def test_skill_scores_whatever_is_in_production_by_level(cfg):
    from costmodel import skill

    rows = {r["level"]: r for r in skill.run(cfg)}
    assert {"all admissions, each month", "hospital x month", "each admission"} <= set(rows)
    test = json.loads((cfg.reports_dir / "test.json").read_text())
    assert rows["each admission"]["global_mean"] == pytest.approx(test["metrics"]["global_mean"]["mae"])
    assert rows["each admission"]["mean_by_proc"] == pytest.approx(test["metrics"]["mean_by_proc"]["mae"])
    # A plain average cannot tell hospitals apart: it is worse for each hospital's total than for everyone's.
    assert rows["hospital x month"]["global_mean"] > rows["all admissions, each month"]["global_mean"]
    saved = json.loads((cfg.reports_dir / "skill_test.json").read_text())
    assert saved["version"] == str(registry.version_for(cfg, registry.PRODUCTION).version)
    assert "Is the model worth it over a plain average?" in (cfg.reports_dir / "skill_test.md").read_text(encoding="utf-8")


@pytest.mark.slow
def test_explorer_numbers_are_the_monitors_and_the_page_is_written(cfg, tmp_path, monkeypatch):
    from costmodel import explore, site

    result = explore.run(cfg)
    monitored = monitor.run(cfg)
    assert [w["key"] for w in result["windows"]] == ["test", "monitor"]
    everyone = result["levels"][0]["groups"][0]
    july = result["months"].index("2024-07")
    month = monitored["months"][0]
    # The same model, the same month: the explorer's totals are the monitor's.
    assert everyone["n"][july] == month["n"]
    assert everyone["paid"][july] == pytest.approx(month["actual"], abs=1)
    assert everyone["expected"][july] == pytest.approx(month["predicted"], abs=1)
    assert everyone["abs_error"][july] / everyone["n"][july] == pytest.approx(month["mae"], rel=1e-3)
    hospitals = next(lv for lv in result["levels"] if lv["key"] == "hospital")
    assert sum(g["n"][july] or 0 for g in hospitals["groups"]) == month["n"]  # every admission is in exactly one group

    holdout_report = cfg.reports_dir / "test.json"
    assert holdout_report.exists()
    monkeypatch.setattr(site, "HOME", tmp_path)
    index = site.run(cfg)
    page = index.with_name("explore.html").read_text(encoding="utf-8")
    assert "__" not in page.split("<script id=")[0]  # no placeholder left in the page itself
    assert "function lineChart" in page and "--series-1" in page  # the main page's chart code and styles
    assert '"url": "explore.html"' in index.read_text(encoding="utf-8")  # the main page links to it
