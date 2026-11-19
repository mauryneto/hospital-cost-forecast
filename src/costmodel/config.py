"""Run configuration: one TOML file per dataset (configs/*.toml). Code holds no dataset-specific dates or thresholds."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from costmodel.contract import DEFAULT_MAX_PAID_AMOUNT
from costmodel.drift import RetrainRules
from costmodel.gate import GateRules
from costmodel.paths import DATA_DIR, REPORTS_DIR, ROOT


@dataclass(frozen=True)
class SplitConfig:
    """Either explicit date windows (``mode = "dates"``) or month counts back from the last month (``"relative"``)."""

    mode: str = "relative"
    train: tuple[str, str] | None = None
    valid: tuple[str, str] | None = None
    test: tuple[str, str] | None = None
    monitor: tuple[str, str] | None = None
    valid_months: int = 12
    test_months: int = 12
    monitor_months: int = 12


@dataclass(frozen=True)
class TrainConfig:
    """The candidate's search space. Every combination is fitted on the training window and measured on validation."""

    objectives: tuple[str, ...] = ("tweedie", "log_l2")
    num_leaves: tuple[int, ...] = (63,)
    # The expanding window (all training data) is always tried; each entry here adds a sliding window of the last N months.
    sliding_months: tuple[int, ...] = ()
    # Whether the model sees the admission date as a number (False, True, or both to compare them).
    use_time: tuple[bool, ...] = (False,)
    # Whether the typical cost of a log_l2 setting is multiplied by the smearing factor. False = left at the median.
    typical_scaled: bool = True
    learning_rate: float = 0.1
    max_rounds: int = 2000
    early_stopping_rounds: int = 50
    min_category_count: int = 20
    seed: int = 42


@dataclass(frozen=True)
class ReplayConfig:
    """``costmodel replay``: which retraining policies to simulate over the monitoring window."""

    # "never" (keep the production model), "rule" (retrain when the retrain rule fires), "monthly" (every month).
    policies: tuple[str, ...] = ("never", "rule", "monthly")
    # Training data for each retrain: 0 = every month known so far (expanding), N = the last N months (sliding).
    windows: tuple[int, ...] = (0,)
    # The most recent mature months held out to gate each retrained candidate against the model in use.
    gate_months: int = 2


@dataclass(frozen=True)
class FreshConfig:
    """``costmodel fresh``: a year no choice was made on, scored once by settings fixed before it was read."""

    window: tuple[str, str] | None = None  # first and last day of the year that is scored
    # The model in production was fitted on admissions up to this date; its recipe is scored too, for the record.
    production_fit_until: str | None = None
    # The two settings, nothing searched: objective, num_leaves, trees, use_time.
    typical: dict = field(default_factory=dict)
    expected: dict = field(default_factory=dict)
    # One group that gets a typical-cost model of its own: column, value, trees. Empty = not tried.
    own_model: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SiteConfig:
    """``costmodel site``: the texts of the results page. The numbers come from the reports, never from here."""

    title: str = ""
    tagline: str = ""
    kicker: str = ""  # small line above the title, e.g. the data source
    note: str = ""  # a closing note, e.g. what the target is and is not
    repository: str = ""  # URL of the code repository, for the page's links
    currency: str = ""
    candidate_label: str = "LightGBM candidate"
    # Names for the codes shown on the explorer page: feature column -> code -> name (codes without one show as is).
    labels: dict = field(default_factory=dict)
    # Name of the dataset whose `costmodel fresh` report the page shows as its last section ("" = none).
    fresh_dataset: str = ""


@dataclass(frozen=True)
class Config:
    name: str
    source: dict
    split: SplitConfig = field(default_factory=SplitConfig)
    max_paid_amount: float = DEFAULT_MAX_PAID_AMOUNT
    # Extra training windows to compare, each dropping some years (e.g. the pandemic) from the training data.
    exclude_year_sets: tuple[tuple[int, ...], ...] = ()
    train: TrainConfig = field(default_factory=TrainConfig)
    gate: GateRules = field(default_factory=GateRules)
    monitor: RetrainRules = field(default_factory=RetrainRules)
    replay: ReplayConfig = field(default_factory=lambda: ReplayConfig())
    site: SiteConfig = field(default_factory=SiteConfig)
    fresh: FreshConfig = field(default_factory=FreshConfig)
    path: Path | None = None

    @property
    def model_name(self) -> str:
        """Name of the model in the MLflow registry."""
        return f"cost-{self.name}"

    @property
    def table_path(self) -> Path:
        return DATA_DIR / "processed" / self.name / "admissions.parquet"

    @property
    def reports_dir(self) -> Path:
        return REPORTS_DIR / self.name

    @property
    def predictions_dir(self) -> Path:
        """Where ``costmodel score`` writes its files (under data/, so never committed)."""
        return DATA_DIR / "predictions" / self.name


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists() and (ROOT / path).exists():
        path = ROOT / path  # e.g. "configs/sihsus_rj.toml" from any working directory
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    split = raw.get("split", {})
    for key in ("train", "valid", "test", "monitor"):
        if key in split:
            split[key] = tuple(split[key])
    return Config(
        name=raw["name"],
        source=raw["source"],
        split=SplitConfig(**split),
        max_paid_amount=float(raw.get("contract", {}).get("max_paid_amount", DEFAULT_MAX_PAID_AMOUNT)),
        exclude_year_sets=tuple(tuple(s) for s in raw.get("experiments", {}).get("exclude_year_sets", [])),
        train=TrainConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.get("train", {}).items()}),
        gate=GateRules(**raw.get("gate", {})),
        monitor=RetrainRules(**raw.get("monitor", {})),
        replay=ReplayConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.get("replay", {}).items()}),
        site=SiteConfig(**raw.get("site", {})),
        fresh=FreshConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.get("fresh", {}).items()}),
        path=path,
    )
