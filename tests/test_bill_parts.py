"""The parts of a SIH/SUS bill: the federal complement, and the error judged with and without it."""

import numpy as np
import pandas as pd
import pytest
from test_dbf import make_dbf

from costmodel import bill_parts
from costmodel import features as F
from costmodel.bill_parts import monthly, with_parts
from costmodel.config import Config
from costmodel.sources.sihsus import read

FIELDS = [("N_AIH", "C", 13), ("IDENT", "C", 1), ("VAL_TOT", "N", 10), *[(name, "N", 10) for name in read.PARTS]]


def admissions() -> pd.DataFrame:
    return pd.DataFrame({
        "admission_id": ["1", "2", "3", "4"],
        "period": ["2024-06", "2024-06", "2024-07", "2024-07"],
        F.TARGET: [1000.0, 500.0, 5000.0, 500.0],
    })  # fmt: skip


def parts() -> pd.DataFrame:
    return pd.DataFrame({"N_AIH": ["1", "2", "3", "4", "999"], "VAL_SH_FED": [0.0, 0.0, 3000.0, np.nan, 7.0],
                         "VAL_SP_FED": [0.0, 0.0, 1000.0, np.nan, 7.0], "VAL_SH_GES": [0.0, 50.0, 0.0, 0.0, 0.0],
                         "VAL_SP_GES": 0.0})  # fmt: skip


def test_each_admission_gets_its_complements_and_none_may_be_missing():
    out = with_parts(admissions(), parts())
    assert out["federal"].tolist() == [0.0, 0.0, 4000.0, 0.0] and out["manager"].tolist() == [0.0, 50.0, 0.0, 0.0]
    assert len(out) == 4  # a bill that is not an admission of the table is left out
    with pytest.raises(RuntimeError, match="not found in the raw files"):
        with_parts(admissions(), parts().head(2))


def test_the_same_predictions_are_judged_against_the_bill_with_and_without_the_complement():
    live = with_parts(admissions(), parts()).assign(typical=[900.0, 600.0, 1000.0, 500.0], expected=1000.0)
    june, july = monthly(live)
    assert june["with_complement_pct"] == 0 and june["mae"] == june["mae_without_complement"] == 100.0
    assert june["mae_admissions_with"] is None
    assert july["with_complement_pct"] == 50.0 and july["complement_per_admission"] == 2000.0
    assert july["complement_money_pct"] == pytest.approx(100 * 4000 / 5500)
    assert july["mae"] == 2000.0 and july["mae_without_complement"] == 0.0  # the whole error is the complement
    assert july["mae_admissions_with"] == 4000.0 and july["mae_admissions_without"] == 0.0
    assert july["total_error_pct"] == pytest.approx(100 * (2000 / 5500 - 1))
    assert july["total_error_without_complement_pct"] == pytest.approx(100 * (2000 / 1500 - 1))


def test_only_a_sihsus_dataset_has_these_parts():
    with pytest.raises(RuntimeError, match="another source"):
        bill_parts.run(Config(name="x", source={"type": "file", "path": "x.csv"}))


def test_parts_are_read_one_row_per_regular_aih_and_stray_bytes_become_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(read.datasus_dbc, "decompress_bytes", lambda b: b)  # the test files are plain DBF

    def bill(n_aih, ident, total, sh, sp, sh_fed="0", sp_fed="0"):
        return (n_aih, ident, total, sh, sp, sh_fed, sp_fed, "0", "0")

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "RDRJ2406.dbc").write_bytes(make_dbf(FIELDS, [bill("1", "1", "100", "80", "20"), bill("2", "5", "9", "9", "0")]))
    (raw / "RDRJ2407.dbc").write_bytes(
        make_dbf(FIELDS, [bill("1", "1", "500", "80", "20", "300", "100"), bill("3", "1", "50", "40", "10", "\xfdI")])
    )
    got = read.read_parts(raw).set_index("N_AIH")
    assert sorted(got.index) == ["1", "3"]  # the long-stay continuation (IDENT 5) is not a regular bill
    assert got.loc["1", "VAL_TOT"] == 500 and got.loc["1", "VAL_SH_FED"] == 300  # the last file a number appears in
    assert pd.isna(got.loc["3", "VAL_SH_FED"]) and got.loc["3", "VAL_SP"] == 10
    assert (raw.parent / "bill_parts.parquet").exists()
    monkeypatch.setattr(read, "read_dbf", lambda *a, **k: pytest.fail("the Parquet copy is up to date"))
    assert len(read.read_parts(raw)) == 2
