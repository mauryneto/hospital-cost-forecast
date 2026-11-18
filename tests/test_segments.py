"""One model per group: same setting as the search, small groups keep the single model, nothing is registered."""

import json

import pandas as pd
import pytest

from costmodel import config as config_module
from costmodel import registry, segments, train
from costmodel.config import Config, SplitConfig, TrainConfig
from costmodel.sources import build_table
from costmodel.synthetic import write


def test_report_shows_which_groups_kept_the_single_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "REPORTS_DIR", tmp_path / "reports")
    cfg = Config(name="segments_report", source={"type": "file"})
    metrics = {"rmse": 1.0, "mae_top_decile": 1.0, "bias_pct": -1.0, "monthly_total_mape": 5.0}
    results = pd.DataFrame(
        [
            {"scheme": segments.SINGLE, "column": None, "models": 1, "mae": 100.0, **metrics},
            {"scheme": "one model per admission type", "column": "admission_type", "models": 1, "mae": 103.0, **metrics},
        ]
    )
    errors = {"mae_single": 9.0, "mae_by_group": 9.0, "total_error_pct_single": -4.0, "total_error_pct_by_group": -4.0}
    groups = [
        {"group": "elective", "n_train": 50, "n_valid": 10, "own_model": False, **errors},
        {"group": "urgent", "n_train": 900, "n_valid": 90, "own_model": True, "trees": 37, **errors},
    ]
    segments.write_report(cfg, results, {"admission_type": groups}, "the chosen setting")
    report = (cfg.reports_dir / "segments_valid.md").read_text(encoding="utf-8")
    assert f"monthly total error at most 15%): {segments.SINGLE}." in report  # lowest MAE, as the search chooses
    lines = report.splitlines()
    elective, urgent = (next(line for line in lines if line.startswith(f"| {g}")) for g in ("elective", "urgent"))
    assert "| -" in elective and "| 37" in urgent  # no model of its own / its own model's trees
    saved = json.loads((cfg.reports_dir / "segments_valid.json").read_text())
    assert [s["scheme"] for s in saved["schemes"]] == [segments.SINGLE, "one model per admission type"]
    assert saved["groups"]["admission_type"][0]["own_model"] is False


@pytest.mark.slow
def test_groups_are_fitted_with_the_search_setting_and_nothing_is_registered(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(registry, "HOME", tmp_path)
    monkeypatch.setattr(config_module, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(segments, "MIN_SEGMENT_ROWS", 1_500)
    write(tmp_path / "admissions.csv", 12_000, "2022-01-01", "2024-12-31", 12)
    cfg = Config(
        name="segments_test",
        source={"type": "file", "path": str(tmp_path / "admissions.csv")},
        split=SplitConfig(mode="relative", valid_months=6, test_months=6, monitor_months=6),
        train=TrainConfig(objectives=("tweedie",), num_leaves=(15,), use_time=(True,), min_category_count=5, max_rounds=200),
    )
    build_table(cfg)
    with pytest.raises(RuntimeError, match="costmodel train"):
        segments.run(cfg)

    version = train.run(cfg)
    searched = json.loads((cfg.reports_dir / "train_valid.json").read_text())[0]
    results = segments.run(cfg)
    assert results["scheme"].tolist() == [segments.SINGLE, *segments.SCHEMES.values()]
    assert results.loc[0, "mae"] == pytest.approx(searched["mae"])  # the single model is the search's, refitted
    saved = json.loads((cfg.reports_dir / "segments_valid.json").read_text())
    n_valid = None
    for column, groups in saved["groups"].items():
        assert sum(g["own_model"] for g in groups) == results.set_index("column").loc[column, "models"]
        for g in groups:
            assert g["own_model"] == (g["n_train"] >= 1_500)
            if not g["own_model"]:  # too small for a model of its own: the single model's predictions, unchanged
                assert g["mae_by_group"] == g["mae_single"]
        total = sum(g["n_valid"] for g in groups)  # every validation admission is in exactly one group
        assert n_valid in (None, total)
        n_valid = total
    assert str(registry.version_for(cfg, registry.STAGING).version) == version
    assert registry.version_for(cfg, registry.PRODUCTION) is None
