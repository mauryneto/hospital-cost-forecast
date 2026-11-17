import pandas as pd
import pytest

from costmodel import contract
from costmodel.config import SplitConfig
from costmodel.sources.sihsus.adapter import age_in_years, to_canonical
from costmodel.sources.sihsus.download import ChecksumMismatch, months, verify_or_record
from costmodel.split import assign, windows


def sih_row(n_aih: str, **overrides) -> dict:
    row = {
        "N_AIH": n_aih, "IDENT": "1", "ANO_CMPT": "2022", "MES_CMPT": "03", "CNES": "2273411", "MUNIC_RES": "330455",
        "MUNIC_MOV": "330455", "SEXO": "1", "COD_IDADE": "4", "DIAG_PRINC": "I219", "PROC_SOLIC": "0303060190",
        "PROC_REA": "0303060190", "CAR_INT": "02", "ESPEC": "03", "COMPLEX": "02", "DT_INTER": "20220301",
        "DT_SAIDA": "20220305", "COBRANCA": "12", "IDADE": 60.0, "DIAS_PERM": 4.0, "QT_DIARIAS": 4.0,
        "UTI_MES_TO": 0.0, "UTI_INT_TO": 0.0, "MORTE": 0.0, "VAL_SH": 900.0, "VAL_SP": 100.0, "VAL_UTI": 0.0,
        "VAL_TOT": 1000.0, "SOURCE_FILE": "RDRJ2203",
    }  # fmt: skip
    row.update(overrides)
    return row


def sih_frame(*rows: dict) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return df.astype({c: "string" for c in df.columns if df[c].dtype == object})


# --- SIH/SUS source ---------------------------------------------------------------------------------------------------


def test_months_crosses_year_boundary():
    assert list(months("2024-11", "2025-02")) == [(2024, 11), (2024, 12), (2025, 1), (2025, 2)]


def test_checksum_recorded_then_enforced(tmp_path):
    f = tmp_path / "RDRJ2401.dbc"
    f.write_bytes(b"abc")
    manifest: dict[str, str] = {}
    digest = verify_or_record(f.name, f, manifest)
    assert manifest[f.name] == digest
    f.write_bytes(b"tampered")
    with pytest.raises(ChecksumMismatch):
        verify_or_record(f.name, f, manifest)


def test_download_does_not_rewrite_a_complete_manifest(tmp_path, monkeypatch):
    """Regression: a reproduction run (all checksums known) must not write the versioned manifest (read-only in Docker)."""
    from costmodel.sources.sihsus import download as dl

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "RDRJ2401.dbc").write_bytes(b"abc")
    manifest = tmp_path / "sihsus.json"
    dl.save_manifest({"RDRJ2401.dbc": dl.sha256(raw / "RDRJ2401.dbc")}, manifest)
    manifest.chmod(0o444)
    monkeypatch.setattr(dl, "MANIFEST", manifest)
    monkeypatch.setattr(dl, "load_manifest", lambda: dl.json.loads(manifest.read_text()))
    monkeypatch.setattr(dl, "save_manifest", lambda *a, **k: pytest.fail("manifest must not be written"))

    class NoFTP:
        def __init__(self, *a, **k): ...
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def login(self): ...
        def cwd(self, d): ...

    monkeypatch.setattr(dl.ftplib, "FTP", NoFTP)
    assert dl.download("RJ", "2024-01", "2024-01", raw_dir=raw) == [raw / "RDRJ2401.dbc"]


def test_age_units():
    idade = pd.Series([10, 5, 40, 2])
    unit = pd.Series(["2", "3", "4", "5"])
    assert age_in_years(idade, unit).tolist() == [0.0, 0.0, 40.0, 102.0]


def test_sihsus_adapter_rules_and_mapping():
    raw = sih_frame(
        sih_row("1111111111111", SEXO="3", CAR_INT="01"),
        sih_row("2222222222222", IDENT="5"),  # long-stay continuation bill
        sih_row("3333333333333", DT_INTER="20171230"),  # admitted before the window
        sih_row("4444444444444", VAL_TOT=0.0),  # nothing paid
        sih_row("5555555555555", VAL_TOT=500.0, SOURCE_FILE="RDRJ2203"),
        sih_row("5555555555555", VAL_TOT=700.0, SOURCE_FILE="RDRJ2205"),  # re-billed later: keep the latest
    )
    df, log = to_canonical(raw, "2018-01-01", "2024-12-31")
    assert sorted(df["admission_id"]) == ["1111111111111", "5555555555555"]
    first = df.set_index("admission_id").loc["1111111111111"]
    assert (first["sex"], first["admission_type"], first["procedure_code_system"]) == ("F", "elective", "SIGTAP")
    assert df.set_index("admission_id").loc["5555555555555", "paid_amount"] == 700.0
    assert log["raw_rows"] == 6
    contract.validate(df)  # the adapter's output satisfies the contract


# --- contract ---------------------------------------------------------------------------------------------------------


def canonical(n: int = 3) -> pd.DataFrame:
    df, _ = to_canonical(sih_frame(*[sih_row(f"{i:013d}") for i in range(1, n + 1)]), "2018-01-01", "2024-12-31")
    return df


def test_contract_reports_every_problem_at_once():
    bad = canonical().assign(sex=["X", "M", "F"], paid_amount=[-1.0, 5.0, 5.0], typo_col=1)
    with pytest.raises(contract.ContractError) as exc:
        contract.validate(bad.drop(columns=["provider_id"]))
    checks = " ".join(exc.value.report["check"].astype(str))
    for expected in ("isin", "in_range", "column_in_dataframe", "column_in_schema"):
        assert expected in checks


def test_contract_rejects_absurd_value_with_configured_ceiling():
    df = canonical().assign(paid_amount=50_000.0)
    contract.validate(df)
    with pytest.raises(contract.ContractError):
        contract.validate(df, max_paid_amount=10_000)


def test_template_csv_round_trip(tmp_path):
    path = tmp_path / "admissions.csv"
    df = canonical()
    df[contract.ALL_COLUMNS].to_csv(path, index=False, date_format="%Y-%m-%d")
    back = contract.validate(contract.read_table(path))
    assert back["requested_procedure"].iloc[0] == "0303060190"  # leading zero kept
    assert back["admission_date"].iloc[0] == pd.Timestamp("2022-03-01")


# --- split ------------------------------------------------------------------------------------------------------------


def test_split_by_dates():
    cfg = SplitConfig(
        mode="dates",
        train=("2018-01-01", "2021-12-31"),
        valid=("2022-01-01", "2022-12-31"),
        test=("2023-01-01", "2023-12-31"),
        monitor=("2024-01-01", "2024-12-31"),
    )
    dates = pd.to_datetime(pd.Series(["2018-01-01", "2021-12-31", "2022-06-01", "2023-01-01", "2024-12-31", "2025-01-02"]))
    assert assign(dates, cfg).tolist() == ["train", "train", "valid", "test", "monitor", pd.NA]


def test_split_relative_counts_back_from_last_month():
    dates = pd.Series(pd.date_range("2019-01-01", "2023-06-15", freq="D"))
    w = windows(dates, SplitConfig(mode="relative", valid_months=6, test_months=6, monitor_months=3))
    assert w["monitor"] == (pd.Timestamp("2023-04-01"), pd.Timestamp("2023-06-30"))
    assert w["test"] == (pd.Timestamp("2022-10-01"), pd.Timestamp("2023-03-31"))
    assert w["valid"] == (pd.Timestamp("2022-04-01"), pd.Timestamp("2022-09-30"))
    assert w["train"] == (pd.Timestamp("2019-01-01"), pd.Timestamp("2022-03-31"))
    labels = assign(dates, SplitConfig(mode="relative", valid_months=6, test_months=6, monitor_months=3))
    assert labels.notna().all()
