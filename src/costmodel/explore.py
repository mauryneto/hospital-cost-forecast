"""``costmodel explore``: the numbers behind the explorer page, the model in production seen group by group.

The production model prices the test window and the monitoring window (months it was not fitted on), and what it
predicted is added up, month by month, next to what was paid:

* for all admissions, and for each admission type, procedure group, bed specialty and hospital;
* for the admissions one by one, as distributions only: mean paid and mean predicted by decile of the amount paid,
  and how far the typical cost falls from the amount paid.

Totals use the expected cost; the error per admission uses the typical cost (see ``candidate.CostPair``). Only
aggregates are written: a level with many groups keeps the ``TOP_GROUPS`` that received most money and pools the
rest, together with any group of fewer than ``MIN_ADMISSIONS`` admissions; no admission is ever listed.
``costmodel site`` turns the result into ``docs/explore.html``.

It describes a model already in production; nothing is chosen with it and nothing is registered.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from costmodel import features as F
from costmodel import registry
from costmodel.config import Config
from costmodel.evaluate import decile_table
from costmodel.train import load_splits

LEVELS = {  # key -> (label, feature column that defines the groups; None = one group with every admission)
    "all": ("All admissions", None),
    "admission_type": ("Admission type", "admission_type"),
    "procedure_group": ("Procedure group", "procedure_prefix2"),
    "bed_specialty": ("Bed specialty", "bed_specialty"),
    "hospital": ("Hospital", "provider_id"),
}
WINDOWS = {"test": "Test window", "monitor": "Monitoring window"}
TOP_GROUPS = 40  # groups shown by name at each level; the others are pooled
MIN_ADMISSIONS = 500  # a group with fewer admissions than this is pooled too: its error would be mostly noise
OTHER = "all others"
# How far the typical cost is from what was paid, as a share of what was paid.
ERROR_BANDS = (
    (0.10, "within 10%"),
    (0.25, "10% to 25%"),
    (0.50, "25% to 50%"),
    (1.00, "50% to 100%"),
    (np.inf, "over 100%"),
)


def group_series(frame: pd.DataFrame, group: pd.Series, months: list[str], top: int = TOP_GROUPS,
                 min_admissions: int = MIN_ADMISSIONS) -> list[dict]:  # fmt: skip
    """Per group, month by month: admissions, paid, expected (predicted total) and the sum of absolute errors of the
    typical cost. Groups beyond the ``top`` by money paid, and groups with fewer than ``min_admissions`` admissions,
    are pooled into one. ``None`` where a group has no admissions."""
    money = frame["paid"].groupby(group.to_numpy()).sum().sort_values(ascending=False)
    size = group.value_counts()
    keep = [g for g in money.index if size[g] >= min_admissions][:top]
    name = pd.Series(np.where(group.isin(keep), group, OTHER), index=frame.index)
    sums = frame.assign(n=1).groupby([name, "month"])[["n", "paid", "expected", "abs_error"]].sum()
    out = []
    for g in keep + ([OTHER] if len(money) > len(keep) else []):
        table = sums.loc[g].reindex(months)
        row = {"name": str(g), "pooled": int(len(money) - len(keep)) if g == OTHER else 0}
        for col, digits in (("n", 0), ("paid", 0), ("expected", 0), ("abs_error", 0)):
            row[col] = [None if pd.isna(v) else round(float(v), digits) for v in table[col]]
        out.append(row)
    return out


def admission_view(y: np.ndarray, typical: np.ndarray, expected: np.ndarray) -> dict:
    """The admissions one by one, as distributions: by decile of the amount paid, and by size of the error."""
    deciles = decile_table(y, typical)
    decile = pd.qcut(pd.Series(y).rank(method="first"), 10, labels=range(1, 11))
    mean_expected = pd.Series(expected).groupby(decile, observed=True).mean()
    relative = np.abs(typical - y) / y
    bands, lower = [], 0.0
    for upper, label in ERROR_BANDS:
        bands.append({"label": label, "share_pct": float(100 * ((relative > lower) & (relative <= upper)).mean())})
        lower = upper
    bands[0]["share_pct"] += float(100 * (relative == 0).mean())  # an exact hit belongs to the first band
    return {
        "n": int(len(y)),
        "mae": float(np.abs(typical - y).mean()),
        "median_error": float(np.median(np.abs(typical - y))),
        "deciles": [
            {"decile": int(d), "mean_paid": float(r["mean_actual"]), "mean_typical": float(r["mean_pred"]),
             "mean_expected": float(mean_expected.loc[d]), "mae": float(r["mae"])}
            for d, r in deciles.iterrows()
        ],  # fmt: skip
        "error_bands": bands,
    }


def run(cfg: Config) -> dict:
    registry.setup(cfg)
    version = registry.version_for(cfg, registry.PRODUCTION)
    if version is None:
        raise RuntimeError(f"no '{registry.PRODUCTION}' model for {cfg.model_name}; run `costmodel evaluate` first")
    model = registry.load(cfg, registry.PRODUCTION)
    df = load_splits(cfg)
    live = df[df["split"].isin(list(WINDOWS))]
    if live.empty:
        raise RuntimeError("the test and monitoring windows are empty; check [split] in the config")
    typical, expected = registry.predict_both(model, live)
    X = F.build_features(live)
    y = live[F.TARGET].to_numpy(dtype=float)
    frame = pd.DataFrame({"month": live["period"].to_numpy(), "paid": y, "expected": expected,
                          "abs_error": np.abs(typical - y)}, index=live.index)  # fmt: skip
    months = sorted(frame["month"].unique())
    labels = cfg.site.labels
    levels = []
    for key, (label, column) in LEVELS.items():
        group = pd.Series(label, index=live.index) if column is None else X[column].astype(str)
        if column is not None and group.nunique() < 2:
            continue  # an optional column this dataset does not fill
        groups = group_series(frame, group, months)
        for g in groups:
            g["label"] = labels.get(column, {}).get(g["name"], g["name"]) if column else label
        levels.append({"key": key, "label": label, "groups": groups})
    result = {
        "version": str(version.version),
        "months": months,
        "windows": [
            {"key": key, "label": label, "months": sorted(live.loc[live["split"] == key, "period"].unique())}
            for key, label in WINDOWS.items()
            if (live["split"] == key).any()
        ],
        "levels": levels,
        "admissions": {
            key: admission_view(y[rows], typical[rows], expected[rows])
            for key in WINDOWS
            if (rows := (live["split"] == key).to_numpy()).any()
        },
    }
    out = cfg.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "explore.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    shown = ", ".join(f"{lv['label'].lower()} ({len(lv['groups'])})" for lv in levels)
    print(f"{len(live):,} admissions of {len(months)} months, version {version.version}: {shown} -> {out / 'explore.json'}")
    return result
