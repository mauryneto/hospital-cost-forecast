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
