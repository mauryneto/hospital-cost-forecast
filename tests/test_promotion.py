"""Champion / challenger through the registry, in a throwaway work folder and MLflow database."""

import pandas as pd
import pytest

from costmodel import config as config_module
from costmodel import holdout, registry, train
from costmodel.config import Config, SplitConfig, TrainConfig
from costmodel.sources import build_table


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(registry, "HOME", tmp_path)
    monkeypatch.setattr(config_module, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config_module, "REPORTS_DIR", tmp_path / "reports")
    from costmodel.synthetic import write

    write(tmp_path / "admissions.csv", 15_000, "2022-01-01", "2024-12-31", 7)
    cfg = Config(
        name="promotion_test",
        source={"type": "file", "path": str(tmp_path / "admissions.csv")},
        split=SplitConfig(mode="relative", valid_months=6, test_months=6, monitor_months=6),
        train=TrainConfig(objectives=("tweedie",), num_leaves=(15,), min_category_count=5),
    )
    build_table(cfg)
    return cfg


@pytest.mark.slow
def test_promotion_then_challenger_must_beat_champion(cfg, monkeypatch):
    v1 = train.run(cfg)
    assert str(registry.version_for(cfg, registry.STAGING).version) == v1
    assert registry.version_for(cfg, registry.PRODUCTION) is None

    # Force the first decision, to exercise the promotion path regardless of how this synthetic sample turns out.
    real_gate = holdout.should_promote
    monkeypatch.setattr(holdout, "should_promote", lambda *args: (True, ["forced for the test"]))
    assert holdout.run(cfg) is True
    assert str(registry.version_for(cfg, registry.PRODUCTION).version) == v1
    assert registry.version_for(cfg, registry.PRODUCTION).tags["gate"] == "passed"
    assert holdout.run(cfg) is False  # the same version cannot be promoted twice

    # A retrained candidate with the same settings and data is the same model: it cannot beat the champion by 2%.
    monkeypatch.setattr(holdout, "should_promote", real_gate)
    v2 = train.run(cfg)
    assert v2 != v1
    assert holdout.run(cfg) is False
    assert str(registry.version_for(cfg, registry.PRODUCTION).version) == v1
    assert registry.version_for(cfg, registry.STAGING).tags["gate"] == "failed"
    assert "production (version" in (cfg.reports_dir / "test.md").read_text()


@pytest.mark.slow
def test_failed_candidate_with_empty_production_puts_best_baseline_there(cfg, monkeypatch):
    train.run(cfg)
    monkeypatch.setattr(holdout, "should_promote", lambda *args: (False, ["forced for the test"]))
    assert holdout.run(cfg) is False
    prod = registry.version_for(cfg, registry.PRODUCTION)
    assert prod.tags["kind"] == "baseline"
    assert registry.version_for(cfg, registry.STAGING).tags["kind"] == "candidate"
    # The baseline in production is a normal registry model: raw admission rows in, costs out.
    table = pd.read_parquet(cfg.table_path).head(50)
    pred = registry.predict(registry.load(cfg, registry.PRODUCTION), table)
    assert len(pred) == 50 and (pred > 0).all()
