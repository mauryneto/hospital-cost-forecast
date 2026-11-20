"""The SUS price table (SIGTAP), one file per month, from the DATASUS FTP.

Each file is ``TabelaUnificada_{YYYYMM}_v{YYMMDDHHMM}.zip``: the table of one month (competencia) and the moment that
version of the file was written. A month can be written again later, under a new name, so ``data/checksums/sigtap.json``
records which file was used for each month and its SHA-256; a later run takes the same file and must find the same
bytes. The raw files are never committed.

Only the procedure table inside each file is read (``tb_procedimento.txt``, fixed width, described by the layout file
next to it): the price of the hospital part and of the professional part, the stay it assumes, its complexity and how
it is financed. This is an input the SUS adapter can offer; a payer's own contract table would take its place elsewhere.
"""

from __future__ import annotations

import ftplib
import io
import re
import zipfile
from pathlib import Path

import pandas as pd

from costmodel.paths import CHECKSUMS_DIR, DATA_DIR
from costmodel.sources.sihsus.download import load_manifest, months, save_manifest, verify_or_record

FTP_HOST = "ftp2.datasus.gov.br"
FTP_DIR = "/pub/sistemas/tup/downloads"

SIGTAP_DIR = DATA_DIR / "sigtap"
RAW_DIR = SIGTAP_DIR / "raw"
MANIFEST = CHECKSUMS_DIR / "sigtap.json"  # versioned with the code
NAME = re.compile(r"TabelaUnificada_(\d{4})(\d{2})_v(\d{2})(\d{2})(\d{2})\d{4}\.zip")

COLUMNS = {  # the table's own names -> ours
    "CO_PROCEDIMENTO": "procedure",
    "VL_SH": "table_hospital",  # in cents
    "VL_SP": "table_professional",  # in cents
    "QT_DIAS_PERMANENCIA": "table_stay_days",
    "QT_PONTOS": "table_points",
    "TP_COMPLEXIDADE": "table_complexity",
    "CO_FINANCIAMENTO": "table_financing",
}
NO_STAY = 9999  # the table's "not applicable"


def month_of(name: str) -> str:
    """'2022-07' for a file of July 2022."""
    year, month = NAME.fullmatch(name).group(1, 2)
    return f"{year}-{month}"


def written(name: str) -> pd.Timestamp:
    """The day this version of the file was written, from its name."""
    year, month, day = map(int, NAME.fullmatch(name).group(3, 4, 5))
    return pd.Timestamp(year=2000 + year, month=month, day=day)


def choose_files(names: list[str], wanted: list[str], manifest: dict[str, str]) -> dict[str, str]:
    """For each wanted month, the file the manifest already names, else the newest version among ``names``."""
    recorded = {month_of(n): n for n in manifest}
    newest: dict[str, str] = {}
    for name in sorted(n for n in names if NAME.fullmatch(n)):  # sorted: a later version replaces an earlier one
        newest[month_of(name)] = name
    chosen = {m: recorded.get(m) or newest.get(m) for m in wanted}
    missing = [m for m, name in chosen.items() if name is None]
    if missing:
        raise FileNotFoundError(f"no SIGTAP table for {missing[0]} ... {missing[-1]} ({len(missing)} months)")
    return chosen


def download(first: str, last: str, raw_dir: Path = RAW_DIR, manifest_path: Path = MANIFEST) -> list[Path]:
    """Make sure the table of every month ``first``..``last`` ("YYYY-MM") is in ``raw_dir`` and matches the manifest."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    known = dict(manifest)
    wanted = [f"{y}-{m:02d}" for y, m in months(first, last)]
    recorded = {month_of(n): n for n in manifest}
    if all(m in recorded and (raw_dir / recorded[m]).exists() for m in wanted):
        chosen = {m: recorded[m] for m in wanted}  # everything is here: no need to reach the server
    else:
        with ftplib.FTP(FTP_HOST, timeout=120) as ftp:
            ftp.login()
            ftp.cwd(FTP_DIR)
            chosen = choose_files(ftp.nlst(), wanted, manifest)
            for name in chosen.values():
                path = raw_dir / name
                if not path.exists():
                    tmp = path.with_suffix(".part")
                    with tmp.open("wb") as f:
                        ftp.retrbinary(f"RETR {name}", f.write)
                    tmp.replace(path)
                    print(f"downloaded {name} ({path.stat().st_size / 1e6:.1f} MB)")
    for name in chosen.values():
        verify_or_record(name, raw_dir / name, manifest)
    if manifest != known:
        save_manifest(manifest, manifest_path)
        print(f"recorded {len(manifest) - len(known)} new checksum(s) in {manifest_path.name}")
    return [raw_dir / name for name in chosen.values()]


def read_procedures(path: Path) -> pd.DataFrame:
    """The procedure table of one monthly file: one row per procedure, prices in the currency (not cents)."""
    with zipfile.ZipFile(path) as z:
        inside = {n.lower(): n for n in z.namelist()}
        try:
            layout = pd.read_csv(io.BytesIO(z.read(inside["tb_procedimento_layout.txt"])), encoding="latin-1")
            body = z.read(inside["tb_procedimento.txt"])
        except KeyError as exc:
            raise ValueError(f"{path.name}: {exc} not found (the SIGTAP file layout may have changed)") from None
    layout.columns = [c.strip() for c in layout.columns]
    missing = set(COLUMNS) - set(layout["Coluna"])
    if missing:
        raise ValueError(f"{path.name}: columns {sorted(missing)} not in the procedure table's layout")
    specs = [(int(a) - 1, int(b)) for a, b in zip(layout["Inicio"], layout["Fim"], strict=True)]
    raw = pd.read_fwf(io.BytesIO(body), colspecs=specs, names=list(layout["Coluna"]), dtype=str, encoding="latin-1")
    out = raw[list(COLUMNS)].rename(columns=COLUMNS)
    for col in out.columns.drop("procedure"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    out["table_hospital"] /= 100
    out["table_professional"] /= 100
    out["table_price"] = out["table_hospital"] + out["table_professional"]
    out["table_stay_days"] = out["table_stay_days"].where(out["table_stay_days"] < NO_STAY)
    out.insert(0, "month", month_of(path.name))
    out["written"] = written(path.name)
    return out


def load(first: str, last: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Every procedure of every month ``first``..``last``: download what is missing, read, and keep a Parquet copy."""
    paths = download(first, last, raw_dir)
    cache = raw_dir.parent / "procedures.parquet"
    if cache.exists():
        table = pd.read_parquet(cache)
        if "file" in table and set(table["file"]) == {p.name for p in paths}:
            return table.drop(columns="file")
    table = pd.concat((read_procedures(p).assign(file=p.name) for p in paths), ignore_index=True)
    table.to_parquet(cache, index=False)
    print(f"read {len(paths)} monthly tables ({len(table):,} rows)")
    return table.drop(columns="file")
