from pathlib import Path

import pytest

from costmodel import contract
from costmodel.cli import main
from costmodel.config import load_config
from costmodel.paths import CONFIGS_DIR, ROOT


@pytest.mark.parametrize("path", sorted(CONFIGS_DIR.glob("*.toml")), ids=lambda p: p.name)
def test_every_config_loads(path: Path):
    cfg = load_config(path)
    assert cfg.source["type"] in {"sihsus", "file"}
    assert cfg.split.mode in {"dates", "relative"}


def test_sihsus_config_uses_spec_windows():
    cfg = load_config(CONFIGS_DIR / "sihsus_rj.toml")
    assert cfg.split.train == ("2018-01-01", "2021-12-31")
    assert cfg.split.valid == ("2022-01-01", "2022-12-31")
    assert cfg.split.test == ("2023-01-01", "2023-12-31")
    assert cfg.split.monitor == ("2024-01-01", "2024-12-31")


def test_template_header_matches_contract():
    header = (ROOT / "templates" / "admissions_template.csv").read_text().strip().split(",")
    assert header == contract.ALL_COLUMNS


def test_validate_command(tmp_path, capsys):
    good = tmp_path / "good.csv"
    good.write_text(
        ",".join(contract.ALL_COLUMNS) + "\n"
        "A1,2023-05-02,61,F,I219,0303060190,SIGTAP,urgent,2273411,330455,03,1520.50,2023-05-09,7,2,0303060190,0\n"
    )
    main(["validate", str(good)])
    assert "OK: 1 admissions" in capsys.readouterr().out

    bad = tmp_path / "bad.csv"
    bad.write_text("admission_id,admission_date\nA1,2023-05-02\n")
    with pytest.raises(SystemExit):
        main(["validate", str(bad)])
    assert "column_in_dataframe" in capsys.readouterr().out


def test_validate_reports_bad_dates_and_duplicates_instead_of_crashing(tmp_path, capsys):
    """Regression: an unparseable date (null failure case) used to crash the report itself."""
    bad = tmp_path / "broken.csv"
    bad.write_text(
        "admission_id,admission_date,age_years,sex,diagnosis_icd10,requested_procedure,procedure_code_system,"
        "admission_type,provider_id,paid_amount\n"
        "X1,2024-02-30,200,Male,I21.9,3101,TUSS,emergency,PRV1,-5\n"
        "X1,2024-03-01,40,F,I219,3101,TUSS,urgent,PRV1,900\n"
    )
    with pytest.raises(SystemExit) as exc:
        main(["validate", str(bad)])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    for expected in ("admission_date", "<empty or invalid>", "field_uniqueness", "Male", "I21.9", "emergency", "-5.0"):
        assert expected in out, expected


def test_train_and_gate_sections_load():
    cfg = load_config(CONFIGS_DIR / "sihsus_rj.toml")
    assert cfg.train.objectives == ("tweedie", "log_l2")
    assert cfg.train.sliding_months == (24,)
    assert cfg.gate.min_mae_gain == 0.02
    assert cfg.model_name == "cost-sihsus_rj"


def test_run_stops_at_the_first_failed_step(tmp_path, capsys):
    config = tmp_path / "missing.toml"
    config.write_text(f'name = "missing"\n[source]\ntype = "file"\npath = "{(tmp_path / "nope.csv").as_posix()}"\n')
    with pytest.raises(SystemExit) as exc:
        main(["run", "--config", str(config)])
    assert exc.value.code != 0
    out = capsys.readouterr().out
    assert "step 'data' failed" in out
    assert "=== train ===" not in out


def test_the_typical_cost_is_ranked_by_the_number_the_config_gives():
    import pandas as pd

    from costmodel.config import Config, TrainConfig
    from costmodel.train import choose

    rows = pd.DataFrame(
        [
            {"objective": "log_l2", "mae": 100.0, "mae_unscaled": 96.0, "monthly_total_mape": 10.0},
            {"objective": "log_l2", "mae": 101.0, "mae_unscaled": 93.0, "monthly_total_mape": 12.0},
            {"objective": "log_l2", "mae": 99.0, "mae_unscaled": 90.0, "monthly_total_mape": 20.0},  # above the ceiling
            {"objective": "tweedie", "mae": 110.0, "mae_unscaled": 110.0, "monthly_total_mape": 3.0},
        ]
    )
    scaled = Config(name="x", source={})
    unscaled = Config(name="x", source={}, train=TrainConfig(typical_scaled=False))
    assert choose(rows, scaled).name == 0 and choose(rows, unscaled).name == 1
    # A search report written before the column existed is still read: it ranks by the only MAE it has.
    assert choose(rows.drop(columns="mae_unscaled"), unscaled).name == 0
    assert load_config("configs/sihsus_rj.toml").train.typical_scaled is False
    assert load_config("configs/sihsus_rj_2025.toml").train.typical_scaled is True  # the 2025 check keeps its recipe
