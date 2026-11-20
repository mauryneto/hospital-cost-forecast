"""The official price table as an input: which month's table an admission may read, and the comparison on validation."""

import json

import numpy as np
import pandas as pd
import pytest

from costmodel import config as config_module
from costmodel import features as F
from costmodel import registry, table_price, train
from costmodel.config import Config, SplitConfig, TrainConfig
from costmodel.sources import build_table
from costmodel.synthetic import generate
from costmodel.table_price import AS_IT_IS, COLUMNS, GROUPS, TableModel, attach, known_months, repriced_from


def price_table(procedures, months: list[str], price: dict, dearer_from: str | None = None, factor: float = 1.5):
    """A table with every procedure in every month; the first procedure costs ``factor`` times more from a month on."""
    rows = []
    for month in months:
        for i, code in enumerate(procedures):
            p = price[code] * (factor if dearer_from and i == 0 and month >= dearer_from else 1.0)
            rows.append({"month": month, "procedure": code, "table_price": p, "table_hospital": 0.8 * p,
                         "table_professional": 0.2 * p, "table_stay_days": 3.0, "table_points": 80.0,
                         "table_complexity": 2.0, "table_financing": 6.0,
                         "written": pd.Timestamp(f"{month}-01") + pd.Timedelta(days=5)})  # fmt: skip
    return pd.DataFrame(rows)


def test_an_admission_reads_only_a_table_written_before_its_month_began():
    written = pd.Series({"2022-05": pd.Timestamp("2022-05-03"), "2022-06": pd.Timestamp("2022-06-28"),
                         "2022-07": pd.Timestamp("2022-08-24"), "2022-08": pd.Timestamp("2022-08-08")})  # fmt: skip
    known = known_months(["2022-05", "2022-06", "2022-07", "2022-08", "2022-09"], written)
    assert known["2022-06"] == "2022-05"  # June's table is dated the 28th of June
    assert known["2022-07"] == "2022-06"
    assert known["2022-08"] == "2022-06"  # July's table came out on 24 August: August still reads June's
    assert known["2022-09"] == "2022-08"  # the newest one written by then, not the last one written
    assert known["2022-05"] == "2022-05"  # nothing older exists: the oldest table stands in


def test_attach_takes_the_row_of_the_procedure_in_the_given_month():
    table = price_table(["A", "B"], ["2022-01", "2022-02"], {"A": 100.0, "B": 300.0}, dearer_from="2022-02")
    X = pd.DataFrame({"age_years": [1.0, 2.0, 3.0, 4.0]})
    out = attach(X, ["A", "A", "B", "ZZZ"], ["2022-01", "2022-02", "2022-02", "2022-02"], table)
    assert out["table_price"].tolist()[:3] == [100.0, 150.0, 300.0]
    assert out[COLUMNS].iloc[3].isna().all()  # a code the table does not have
    assert out["age_years"].tolist() == [1.0, 2.0, 3.0, 4.0] and list(out.columns) == ["age_years", *COLUMNS]


def test_repriced_from_gives_the_first_month_a_price_changed_inside_the_window():
    table = price_table(["A", "B"], ["2021-12", "2022-01", "2022-02", "2022-03"], {"A": 100.0, "B": 300.0}, "2022-02")
    assert repriced_from(table, ["2022-01", "2022-02", "2022-03"]).to_dict() == {"A": "2022-02"}
    assert repriced_from(table, ["2022-03"]).empty  # a change before the window is not a change inside it


@pytest.mark.parametrize("relative", [False, True])
def test_table_model_predicts_on_the_cost_scale(relative):
    df = generate(12_000, "2022-01-01", "2023-12-31", seed=8)
    X, y = F.build_features(df), df[F.TARGET]
    price = df.groupby("requested_procedure")[F.TARGET].median().to_dict()
    months = sorted(df["admission_date"].dt.strftime("%Y-%m").unique())
    X = attach(X, df["requested_procedure"], df["admission_date"].dt.strftime("%Y-%m"), price_table(price, months, price))
    is_train = (df["admission_date"] < "2023-07-01").to_numpy()
    model = TableModel("log_l2", 15, 0.1, 100, 5, 42, True, relative=relative)
    model.fit(X[is_train], y[is_train], X[~is_train], y[~is_train], 20)
    assert set(COLUMNS) <= set(model.columns)
    pred, y_valid = model.predict(X[~is_train]), y[~is_train]
    assert np.isfinite(pred).all() and (pred > 0).all()
    assert np.abs(pred - y_valid).mean() < np.abs(y[is_train].mean() - y_valid).mean()
    no_price = X[~is_train].assign(table_price=0.0)  # a procedure the table does not price: the unit is 1, not 0
    assert np.isfinite(model.predict(no_price)).all()
    if relative:  # the unit of the prediction: a table price twice as high moves it, every other input equal
        dearer = X[~is_train].assign(table_price=lambda d: d["table_price"] * 2)
        assert (model.predict(dearer) > pred).mean() > 0.9


@pytest.mark.slow
def test_comparison_uses_the_search_settings_and_registers_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(registry, "HOME", tmp_path)
    monkeypatch.setattr(config_module, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(table_price, "DRAWS", 20)
    df = generate(12_000, "2022-01-01", "2024-12-31", seed=12)
    df.to_csv(tmp_path / "admissions.csv", index=False)
    cfg = Config(
        name="table_price_test",
        source={"type": "file", "path": str(tmp_path / "admissions.csv")},
        split=SplitConfig(mode="relative", valid_months=6, test_months=6, monitor_months=6),
        train=TrainConfig(objectives=("log_l2",), num_leaves=(15,), use_time=(True,), min_category_count=5,
                          max_rounds=200, typical_scaled=False),  # fmt: skip
    )
    build_table(cfg)
    with pytest.raises(RuntimeError, match="costmodel train"):
        table_price.run(cfg)
    version = train.run(cfg)
    with pytest.raises(RuntimeError, match="no SIGTAP procedure codes"):
        table_price.run(cfg)  # made-up admissions carry another code system: the SUS table is not theirs

    df.assign(procedure_code_system="SIGTAP").to_csv(tmp_path / "admissions.csv", index=False)
    build_table(cfg)
    price = df.groupby("requested_procedure")[F.TARGET].median().to_dict()
    asked = {}

    def fake_load(first, last):
        asked.update(first=first, last=last)
        months = [str(p) for p in pd.period_range(first, last, freq="M")]
        return price_table(sorted(price, key=lambda c: -df["requested_procedure"].eq(c).sum()), months, price, "2023-09")

    monkeypatch.setattr(table_price.sigtap, "load", fake_load)
    results = table_price.run(cfg)
    assert asked == {"first": "2021-12", "last": "2024-12"}  # from the month before the first admission
    assert results["variant"].tolist()[0] == AS_IT_IS and len(results) == 4
    searched = json.loads((cfg.reports_dir / "train_valid.json").read_text())[0]
    assert results.loc[0, "mae"] == pytest.approx(searched["mae_unscaled"])  # the typical cost as the config gives it
    assert results.loc[0, "monthly_total_mape"] == pytest.approx(searched["monthly_total_mape"])
    assert pd.isna(results.loc[0, "mae_gap"]) and results["mae_gap"].iloc[1:].notna().all()
    assert results["typical_table_gain_pct"].iloc[1:].between(0, 100).all()
    saved = json.loads((cfg.reports_dir / "table_price_valid.json").read_text())
    assert set(saved["groups"][AS_IT_IS]) == set(GROUPS)  # the most common procedure was re-priced inside the window
    assert 0 < saved["table"]["repriced_admissions_pct"] < 100 and saved["table"]["steps"][0]["month"] == "2023-09"
    assert "nothing was registered" in (cfg.reports_dir / "table_price_valid.md").read_text(encoding="utf-8")
    assert str(registry.version_for(cfg, registry.STAGING).version) == version
    assert registry.version_for(cfg, registry.PRODUCTION) is None
