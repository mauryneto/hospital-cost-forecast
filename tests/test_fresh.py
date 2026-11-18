"""The fresh-year check: verdicts follow the criteria stated beforehand, and the run adopts nothing."""

import json

import numpy as np
import pytest

from costmodel import config as config_module
from costmodel import foundation, fresh, registry
from costmodel.candidate import EXPECTED, TYPICAL
from costmodel.config import Config, FreshConfig, ReplayConfig, TrainConfig, load_config
from costmodel.fresh import CONFIRMED, CONTRADICTED, NOT_RUN, NOT_SHOWN, together, verdict, with_own_model
from costmodel.paths import CONFIGS_DIR
from costmodel.sources import build_table


def test_verdict_reads_an_interval_against_what_was_expected():
    assert verdict(-3.0, -1.0, "below") == CONFIRMED and verdict(1.0, 3.0, "above") == CONFIRMED
    assert verdict(1.0, 3.0, "below") == CONTRADICTED and verdict(-3.0, -1.0, "above") == CONTRADICTED
    assert verdict(-1.0, 2.0, "below") == NOT_SHOWN and verdict(0.0, 2.0, "above") == NOT_SHOWN  # zero inside
    with pytest.raises(ValueError, match="below"):
        verdict(1.0, 2.0, "smaller")


def test_several_expectations_are_confirmed_only_together():
    assert together(CONFIRMED, CONFIRMED) == CONFIRMED
    assert together(CONFIRMED, NOT_SHOWN) == NOT_SHOWN
    assert together(CONFIRMED, CONTRADICTED) == CONTRADICTED and together(NOT_SHOWN, CONTRADICTED) == CONTRADICTED


def test_own_model_replaces_only_its_group():
    typical = np.array([10.0, 20.0, 30.0, 40.0])
    out = with_own_model(typical, np.array([False, True, False, True]), np.array([21.0, 41.0]))
    assert out.tolist() == [10.0, 21.0, 30.0, 41.0] and typical.tolist() == [10.0, 20.0, 30.0, 40.0]


def test_the_2025_settings_are_the_ones_written_in_the_protocol():
    fc = load_config(CONFIGS_DIR / "sihsus_rj_2025.toml").fresh
    assert fc.window == ("2025-01-01", "2025-12-31") and fc.production_fit_until == "2022-12-31"
    assert fc.typical == {"objective": "log_l2", "num_leaves": 255, "trees": 324, "use_time": True}
    assert fc.expected == {"objective": "tweedie", "num_leaves": 63, "trees": 303, "use_time": True}
    assert fc.own_model == {"column": "procedure_prefix2", "value": "04", "trees": 107}
    assert load_config(CONFIGS_DIR / "sihsus_rj.toml").fresh.window is None  # the published dataset has no such window


@pytest.mark.slow
def test_fresh_year_answers_the_four_questions_and_adopts_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(registry, "HOME", tmp_path)
    monkeypatch.setattr(config_module, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(fresh, "DRAWS", 30)
    monkeypatch.setattr(foundation, "tabpfn_available", lambda: False)  # the same run with or without the extra
    from costmodel.synthetic import write

    write(tmp_path / "admissions.csv", 20_000, "2021-01-01", "2024-12-31", 5)
    setting = {"num_leaves": 15, "trees": 40, "use_time": True}
    cfg = Config(
        name="fresh_test",
        source={"type": "file", "path": str(tmp_path / "admissions.csv")},
        train=TrainConfig(min_category_count=5),
        replay=ReplayConfig(policies=("never", "rule", "monthly"), windows=(0,), gate_months=2),
        fresh=FreshConfig(
            window=("2024-01-01", "2024-12-31"),
            production_fit_until="2022-12-31",
            typical={"objective": "log_l2", **setting},
            expected={"objective": "tweedie", **setting},
            own_model={"column": "admission_type", "value": "elective", "trees": 20},
        ),
    )
    with pytest.raises(RuntimeError, match=r"\[fresh\]"):
        fresh.run(Config(name="none", source=cfg.source))  # no window, no settings: nothing to run
    table = build_table(cfg)
    result = fresh.run(cfg)

    scored = table[table["admission_date"] >= "2024-01-01"]
    assert result["window"] == ["2024-01", "2024-12"] and result["admissions"] == len(scored)
    # The model for the window never sees it, nor the months still being billed when it is fitted.
    assert result["known_through"] == "2023-09"
    assert result["fit_rows"] == int((table["admission_date"] < "2023-10-01").sum())
    assert [m["period"] for m in result["monthly"]] == [f"2024-{m:02d}" for m in range(1, 13)]
    assert sum(m["n"] for m in result["monthly"]) == len(scored)

    two = result["two_numbers"]
    assert set(two["numbers"]) == {TYPICAL, EXPECTED}
    gap = two["numbers"][EXPECTED]["mae"] - two["numbers"][TYPICAL]["mae"]
    assert two["expected_minus_typical"]["mae_gap"] == pytest.approx(gap)
    assert two["verdict"] == together(two["per_admission"], two["monthly_total"])
    worth = result["worth_it"]
    assert worth["best_baseline"] in worth["baselines"] and worth["skill"][-1]["level"] == "each admission"
    ideas = {i["idea"]: i for i in result["ideas"]}
    assert len(ideas) == 3 and all(i["verdict"] in {CONFIRMED, CONTRADICTED, NOT_SHOWN, NOT_RUN} for i in ideas.values())
    own = ideas["a typical-cost model of its own for one group"]
    assert own["total_gap"] == 0 and own["theirs"]["monthly_total_mape"] == own["ours"]["monthly_total_mape"]
    assert own["group_mae_own"] != own["group_mae_single"]  # only the typical cost of that group changes
    assert ideas["a foundation model (TabPFN-2) with code statistics"]["verdict"] == NOT_RUN
    never = result["retraining"]["policies"][0]
    assert never["policy"] == "never" and never["overall"]["mae"] == pytest.approx(result["model"]["mae"])
    assert [r["policy"] for r in result["retraining"]["policies"]] == ["never", "rule", "monthly"]
    assert result["production"]["rows"] == int((table["admission_date"] <= "2022-12-31").sum())

    saved = json.loads((cfg.reports_dir / "fresh_year.json").read_text(encoding="utf-8"))
    assert saved["model"]["mae"] == pytest.approx(result["model"]["mae"])
    report = (cfg.reports_dir / "fresh_year.md").read_text(encoding="utf-8")
    assert "## 1. The two numbers" in report and "## 4. Retraining policies" in report
    assert registry.version_for(cfg, registry.STAGING) is None and registry.version_for(cfg, registry.PRODUCTION) is None
