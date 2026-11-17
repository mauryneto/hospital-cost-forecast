"""Read raw ``.dbc`` files into one Parquet file per processing month (``data/sihsus/interim``).

Only the columns this project uses are kept. Code-like columns (CID, procedure, CNES, municipality) stay as strings so
leading zeros survive. Identifying fields that the model never needs (birth date, CEP, CPF of the authoriser) are dropped
here, so they never reach any later step.
"""

from __future__ import annotations

from pathlib import Path

import datasus_dbc
import pandas as pd

from costmodel.sources.sihsus.dbf import read_dbf
from costmodel.sources.sihsus.download import RAW_DIR, SIH_DIR

INTERIM_DIR = SIH_DIR / "interim"

STRING_COLUMNS = [
    "N_AIH", "IDENT", "ANO_CMPT", "MES_CMPT", "CNES", "MUNIC_RES", "MUNIC_MOV", "SEXO", "COD_IDADE",
    "DIAG_PRINC", "PROC_SOLIC", "PROC_REA", "CAR_INT", "ESPEC", "COMPLEX", "DT_INTER", "DT_SAIDA", "COBRANCA",
]  # fmt: skip
NUMERIC_COLUMNS = [
    "IDADE", "DIAS_PERM", "QT_DIARIAS", "UTI_MES_TO", "UTI_INT_TO", "MORTE",
    "VAL_SH", "VAL_SP", "VAL_UTI", "VAL_TOT",
]  # fmt: skip
KEEP = STRING_COLUMNS + NUMERIC_COLUMNS


def read_dbc(path: Path, columns: list[str] = KEEP) -> pd.DataFrame:
    try:
        df = read_dbf(datasus_dbc.decompress_bytes(path.read_bytes()), columns)
    except KeyError as exc:
        raise ValueError(f"{path.name}: {exc} (the RD layout may have changed)") from None
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["SOURCE_FILE"] = path.stem
    return df


def convert_all(raw_dir: Path = RAW_DIR, out_dir: Path = INTERIM_DIR) -> list[Path]:
    """Convert every raw file that has no up-to-date Parquet yet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for path in sorted(raw_dir.glob("RD*.dbc")):
        out = out_dir / f"{path.stem}.parquet"
        if not out.exists() or out.stat().st_mtime < path.stat().st_mtime:
            read_dbc(path).to_parquet(out, index=False)
            print(f"converted {path.name}")
        outputs.append(out)
    return outputs


def load_interim(uf: str, interim_dir: Path = INTERIM_DIR) -> pd.DataFrame:
    files = sorted(interim_dir.glob(f"RD{uf.upper()}*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet files in {interim_dir}; run `costmodel data` first")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
