"""The blend averages two settings of the search, refitted the same way; it reads validation only and registers nothing."""

import json

import pandas as pd
import pytest

from costmodel import blend, registry, train
from costmodel import config as config_module
from costmodel.config import Config, SplitConfig, TrainConfig
from costmodel.gate import GateRules
from costmodel.sources import build_table
from costmodel.synthetic import write


def search_rows(*rows):
    cols = ("objective", "mae", "monthly_total_mape")
    return pd.DataFrame([dict(zip(cols, row, strict=True)) for row in rows])


def test_pair_is_the_chosen_setting_and_the_best_on_the_months_total():
    cfg = Config(name="x", source={"type": "file"}, gate=GateRules(max_monthly_total_mape=15.0))
    search = search_rows(("a", 100.0, 16.0), ("b", 101.0, 14.0), ("c", 110.0, 9.0), ("d", 120.0, 9.0))
    first, second = blend.pair(search, cfg)
    assert first["objective"] == "b"  # lowest MAE within the gate's ceiling, as `costmodel train` chooses
    assert second["objective"] == "c"  # lowest error on the month's total; ties go to the lower MAE


def test_report_names_the_best_share_by_the_search_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "REPORTS_DIR", tmp_path / "reports")
    cfg = Config(name="blend_report", source={"type": "file"}, gate=GateRules(max_monthly_total_mape=15.0))
    metrics = {"rmse": 1.0, "mae_top_decile": 1.0, "bias_pct": -1.0}
    results = pd.DataFrame(
        [
            {"weight": 0.0, "mae": 100.0, "monthly_total_mape": 14.0, **metrics},
            {"weight": 0.5, "mae": 98.0, "monthly_total_mape": 12.0, **metrics},
            {"weight": 1.0, "mae": 99.0, "monthly_total_mape": 10.0, **metrics},
        ]
    )
    blend.write_report(cfg, results, "first setting", "second setting")
    saved = json.loads((cfg.reports_dir / "blend_valid.json").read_text())
    assert saved["best_weight"] == 0.5 and len(saved["rows"]) == 3
    report = (cfg.reports_dir / "blend_valid.md").read_text(encoding="utf-8")
    assert "50% of the month's-total model" in report
    assert "MAE goes from 100.00 to 98.00" in report and "from 14.00% to 12.00%" in report


@pytest.mark.slow
def test_blend_refits_the_search_settings_and_registers_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(registry, "HOME", tmp_path)
    monkeypatch.setattr(config_module, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config_module, "REPORTS_DIR", tmp_path / "reports")
    write(tmp_path / "admissions.csv", 12_000, "2022-01-01", "2024-12-31", 12)
    cfg = Config(
        name="blend_test",
        source={"type": "file", "path": str(tmp_path / "admissions.csv")},
        split=SplitConfig(mode="relative", valid_months=6, test_months=6, monitor_months=6),
        train=TrainConfig(objectives=("tweedie", "log_l2"), num_leaves=(15,), use_time=(True,), min_category_count=5,
                          max_rounds=200),  # fmt: skip
    )
    build_table(cfg)
    with pytest.raises(RuntimeError, match="costmodel train"):
        blend.run(cfg)

    version = train.run(cfg)
    search = pd.DataFrame(json.loads((cfg.reports_dir / "train_valid.json").read_text()))
    first, second = blend.pair(search, cfg)
    results = blend.run(cfg)
    # Alone, each model gives the number the search reported: the blend refits the same settings, nothing else.
    assert results.loc[0, "mae"] == pytest.approx(first["mae"])
    if first.name == second.name:
        assert results["weight"].tolist() == [0.0]
    else:
        assert results["weight"].tolist() == list(blend.WEIGHTS)
        assert results.iloc[-1]["monthly_total_mape"] == pytest.approx(second["monthly_total_mape"])
    assert str(registry.version_for(cfg, registry.STAGING).version) == version
    assert registry.version_for(cfg, registry.PRODUCTION) is None
