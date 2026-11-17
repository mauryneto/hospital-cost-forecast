"""Data sources. Each one produces the canonical admissions table (see costmodel.contract), which is validated and saved.

* ``sihsus``: public SIH/SUS data, downloaded and mapped by this project.
* ``file``: a CSV or Parquet already in the canonical format (for example ``templates/admissions_template.csv`` filled
  with your own data).
"""

from __future__ import annotations

import pandas as pd

from costmodel import contract
from costmodel.config import Config
from costmodel.paths import HOME


def ingest(cfg: Config) -> pd.DataFrame:
    """Produce the canonical table from the configured source (not validated yet)."""
    kind = cfg.source.get("type")
    if kind == "sihsus":
        from costmodel.sources.sihsus.adapter import to_canonical
        from costmodel.sources.sihsus.download import download
        from costmodel.sources.sihsus.read import convert_all, load_interim

        src = cfg.source
        download(src["uf"], src["first_billing_month"], src["last_billing_month"])
        convert_all()
        df, log = to_canonical(load_interim(src["uf"]), src["first_admission"], src["last_admission"])
        for step, n in log.items():
            print(f"{step:>24}: {n:>10,}")
        return df
    if kind == "file":
        return contract.read_table(HOME / cfg.source["path"])  # relative to the work folder, or absolute
    raise ValueError(f"unknown source type {kind!r}; use 'sihsus' or 'file'")


def build_table(cfg: Config) -> pd.DataFrame:
    df = contract.validate(ingest(cfg), cfg.max_paid_amount)
    cfg.table_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.table_path, index=False)
    print(f"{len(df):,} admissions written to {cfg.table_path}")
    return df


def load_table(cfg: Config) -> pd.DataFrame:
    if not cfg.table_path.exists():
        raise FileNotFoundError(f"{cfg.table_path} not found; run `costmodel data --config {cfg.path}` first")
    return pd.read_parquet(cfg.table_path)
