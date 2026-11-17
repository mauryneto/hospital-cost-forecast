"""Scoring: the production model prices admissions whose cost is not known yet. Used by the API and the batch job.

Both go through ``Scorer.score``, so they check the input the same way (the contract's admission-time columns) and
price it the same way (the registered model builds its own features). A ``Scorer`` holds one exact model version: if
the ``production`` alias moves (a promotion or a rollback), a running scorer keeps the version it loaded until it is
restarted, and it always reports the version it actually uses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from costmodel import contract, registry
from costmodel.config import Config


@dataclass
class Scorer:
    model_name: str
    version: str
    model: object = field(repr=False)
    tags: dict[str, str] = field(default_factory=dict)

    @classmethod
    def production(cls, cfg: Config) -> Scorer:
        """The version the ``production`` alias points at now."""
        registry.setup(cfg)
        found = registry.version_for(cfg, registry.PRODUCTION)
        if found is None:
            raise RuntimeError(f"no '{registry.PRODUCTION}' model for {cfg.model_name}; run `costmodel evaluate` first")
        version = str(found.version)
        return cls(cfg.model_name, version, registry.load_version(cfg, version), dict(found.tags))

    def score(self, admissions: pd.DataFrame) -> np.ndarray:
        """Predicted cost of each admission. Raises ``contract.ContractError`` if the input breaks the contract."""
        return registry.predict(self.model, contract.validate_inputs(admissions))


def run_batch(cfg: Config, month: str | None = None, input_path: str | None = None) -> Path:
    """``costmodel score``: price every admission of a month (or of a file) and save the predictions.

    Without ``input_path`` the admissions come from the dataset's table (``costmodel data``); a file must follow the
    contract's admission-time columns, and the cost may be missing. The output, one row per admission, is written to
    ``data/predictions/<dataset>/``.
    """
    if month is None and input_path is None:
        raise ValueError("give --month, --input, or both")
    df = contract.read_table(Path(input_path)) if input_path is not None else pd.read_parquet(cfg.table_path)
    if month is not None:
        target = pd.Period(month, "M")  # raises on a malformed month
        df = df[pd.to_datetime(df["admission_date"]).dt.to_period("M") == target]
    if df.empty:
        raise ValueError(f"no admissions to score (month {month}, input {input_path or cfg.table_path})")

    scorer = Scorer.production(cfg)
    t0 = time.perf_counter()
    predicted = scorer.score(df)
    seconds = time.perf_counter() - t0
    out = pd.DataFrame(
        {
            "admission_id": df["admission_id"].astype("string").to_numpy(),
            "admission_date": pd.to_datetime(df["admission_date"]).to_numpy(),
            "predicted_cost": predicted,
            "model": scorer.model_name,
            "model_version": scorer.version,
            "scored_at": pd.Timestamp.now(tz="UTC").floor("s"),
        }
    )
    name = month or Path(input_path).stem
    path = cfg.predictions_dir / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(
        f"{len(out):,} admissions scored with {scorer.model_name} version {scorer.version} in {seconds:.1f}s:"
        f" predicted total R$ {predicted.sum():,.0f}, mean R$ {predicted.mean():,.2f} per admission -> {path}"
    )
    return path
