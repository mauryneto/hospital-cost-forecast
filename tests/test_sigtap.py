"""The SUS price table: one file per month, read by its own layout, pinned by a checksum manifest."""

import zipfile

import pandas as pd
import pytest

from costmodel.sources import sigtap
from costmodel.sources.sihsus.download import ChecksumMismatch

LAYOUT = [("CO_PROCEDIMENTO", 10), ("NO_PROCEDIMENTO", 12), ("TP_COMPLEXIDADE", 1), ("QT_DIAS_PERMANENCIA", 4),
          ("QT_PONTOS", 4), ("VL_SH", 10), ("VL_SA", 10), ("VL_SP", 10), ("CO_FINANCIAMENTO", 2)]  # fmt: skip


def table_zip(path, rows: list[dict]) -> None:
    """A monthly file with only what the reader needs: the procedure table and its layout, fixed width."""
    start, layout = 1, ["Coluna,Tamanho,Inicio,Fim,Tipo"]
    for name, size in LAYOUT:
        layout.append(f"{name},{size},{start},{start + size - 1},VARCHAR2")
        start += size
    body = "\r\n".join("".join(str(row[name]).ljust(size)[:size] for name, size in LAYOUT) for row in rows)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("tb_procedimento_layout.txt", "\r\n".join(layout) + "\r\n")
        z.writestr("tb_procedimento.txt", (body + "\r\n").encode("latin-1"))
        z.writestr("tb_grupo.txt", "ignored")


def row(code: str, sh: int, sp: int, stay: int = 3, name: str = "CIRURGIA") -> dict:
    return {"CO_PROCEDIMENTO": code, "NO_PROCEDIMENTO": name, "TP_COMPLEXIDADE": 2, "QT_DIAS_PERMANENCIA": f"{stay:04d}",
            "QT_PONTOS": "0080", "VL_SH": f"{sh:010d}", "VL_SA": "0" * 10, "VL_SP": f"{sp:010d}",
            "CO_FINANCIAMENTO": "06"}  # fmt: skip


def test_file_names_carry_the_month_and_the_day_the_version_was_written():
    name = "TabelaUnificada_202207_v2208241530.zip"
    assert sigtap.month_of(name) == "2022-07"
    assert sigtap.written(name) == pd.Timestamp("2022-08-24")  # written 54 days after its month began


def test_reads_prices_in_currency_and_keeps_leading_zeros(tmp_path):
    path = tmp_path / "TabelaUnificada_202207_v2208241530.zip"
    table_zip(path, [row("0407030034", 69305, 29940), row("0101010010", 0, 0, stay=9999, name="AÇÃO COLETIVA")])
    table = sigtap.read_procedures(path)
    first = table.iloc[0]
    assert first["procedure"] == "0407030034" and first["month"] == "2022-07"
    assert first["table_hospital"] == 693.05 and first["table_professional"] == 299.40
    assert first["table_price"] == pytest.approx(992.45)
    assert first["table_stay_days"] == 3 and first["table_complexity"] == 2 and first["table_financing"] == 6
    assert first["written"] == pd.Timestamp("2022-08-24")
    assert table.iloc[1]["table_price"] == 0 and pd.isna(table.iloc[1]["table_stay_days"])  # 9999 = not applicable


def test_a_file_without_the_procedure_table_is_refused(tmp_path):
    path = tmp_path / "TabelaUnificada_202207_v2208241530.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("tb_grupo.txt", "x")
    with pytest.raises(ValueError, match="layout may have changed"):
        sigtap.read_procedures(path)


def test_the_newest_version_is_taken_unless_the_manifest_already_names_one():
    names = ["TabelaUnificada_202201_v2201050900.zip", "TabelaUnificada_202201_v2203010900.zip",
             "TabelaUnificada_202202_v2202040900.zip", "leiame.txt"]  # fmt: skip
    chosen = sigtap.choose_files(names, ["2022-01", "2022-02"], {})
    assert chosen["2022-01"] == "TabelaUnificada_202201_v2203010900.zip"
    pinned = sigtap.choose_files(names, ["2022-01"], {"TabelaUnificada_202201_v2201050900.zip": "abc"})
    assert pinned["2022-01"] == "TabelaUnificada_202201_v2201050900.zip"  # a rerun reads the file it read before
    with pytest.raises(FileNotFoundError, match="2022-03"):
        sigtap.choose_files(names, ["2022-02", "2022-03"], {})


def test_files_already_here_are_checked_without_reaching_the_server(tmp_path, monkeypatch):
    raw, manifest = tmp_path / "raw", tmp_path / "sigtap.json"
    raw.mkdir()
    for month, price in (("202201", 10000), ("202202", 12000)):
        table_zip(raw / f"TabelaUnificada_{month}_v{month[2:]}050900.zip", [row("0407030034", price, 0)])

    class Server:
        def __init__(self, *a, **k): ...
        def __enter__(self):
            return self
        def __exit__(self, *a): ...
        def login(self): ...
        def cwd(self, d): ...
        def nlst(self):
            return [p.name for p in raw.iterdir()]

    monkeypatch.setattr(sigtap.ftplib, "FTP", Server)
    paths = sigtap.download("2022-01", "2022-02", raw, manifest)  # first run: lists the server, records two checksums
    assert len(paths) == 2 and len(sigtap.load_manifest(manifest)) == 2

    monkeypatch.setattr(sigtap.ftplib, "FTP", lambda *a, **k: pytest.fail("the server must not be reached"))
    assert sigtap.download("2022-01", "2022-02", raw, manifest) == paths

    table_zip(paths[1], [row("0407030034", 99999, 0)])  # the same name with other bytes
    with pytest.raises(ChecksumMismatch):
        sigtap.download("2022-01", "2022-02", raw, manifest)
