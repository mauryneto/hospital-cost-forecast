"""``costmodel bill-parts``: what is a bill made of, and which part moved on the monitoring window?

The contract has one cost per admission, what was paid. In SIH/SUS that total is a sum: the table's hospital and
professional parts, and **complements** paid on top of them, by the federal government and by the state or city that
runs the hospital. The table part follows the published price table; a complement follows a programme, and can be
switched on for some procedures and hospitals from one month to the next without any table changing.

When the monitor shows the error jumping while the admissions look the same, the question is which part moved. This
command reads the parts from the raw files (the admissions table does not carry them) and gives, month by month on the
monitoring window: how many admissions were paid a complement and how much; the production model's error as the monitor
measures it; and the same error **against the bill without the complement**, which is what the model would have been
judged on had the complement not existed. The gap between the two is the part of the error no retraining could have
foreseen from earlier bills.

It also gives the complement's share of the money in every year of the data (no model involved), to show whether the
monitoring window is unusual.

SIH/SUS only: other sources have their own parts, or none. It describes a window that has already been read; it makes
no choice and registers nothing.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from costmodel import features as F
from costmodel import registry
from costmodel.config import Config
from costmodel.evaluate import to_markdown
from costmodel.train import load_splits

FEDERAL, MANAGER = ["VAL_SH_FED", "VAL_SP_FED"], ["VAL_SH_GES", "VAL_SP_GES"]
PAID_ONE = 1.0  # an admission "has" a complement from this amount on (currency units)
TOP = 8  # procedures listed


def with_parts(df: pd.DataFrame, parts: pd.DataFrame) -> pd.DataFrame:
    """The admissions with the federal and the manager's complement of each bill (0 where the files have none)."""
    amounts = pd.DataFrame({"admission_id": parts["N_AIH"],
                            "federal": parts[FEDERAL].fillna(0).sum(axis=1),
                            "manager": parts[MANAGER].fillna(0).sum(axis=1)})  # fmt: skip
    out = df.merge(amounts, on="admission_id", how="left", validate="1:1")
    missing = out["federal"].isna().mean()
    if missing > 0.001:
        raise RuntimeError(f"{100 * missing:.2f}% of the admissions were not found in the raw files")
    return out.fillna({"federal": 0.0, "manager": 0.0})


def monthly(live: pd.DataFrame) -> list[dict]:
    """One row per month of admissions that carry ``federal``, ``typical`` and ``expected``."""
    rows = []
    for period, g in live.groupby("period", sort=True):
        paid, without = g[F.TARGET].to_numpy(dtype=float), (g[F.TARGET] - g["federal"]).to_numpy(dtype=float)
        has = g["federal"].to_numpy() >= PAID_ONE
        typical, expected = g["typical"].to_numpy(), g["expected"].to_numpy()
        rows.append({
            "period": str(period),
            "n": int(len(g)),
            "with_complement_pct": float(100 * has.mean()),
            "complement_per_admission": float(g["federal"].mean()),
            "complement_money_pct": float(100 * g["federal"].sum() / paid.sum()),
            "mae": float(np.abs(typical - paid).mean()),
            "mae_without_complement": float(np.abs(typical - without).mean()),
            "mae_admissions_with": float(np.abs(typical - paid)[has].mean()) if has.any() else None,
            "mae_admissions_without": float(np.abs(typical - paid)[~has].mean()),
            "total_error_pct": float(100 * (expected.sum() / paid.sum() - 1)),
            "total_error_without_complement_pct": float(100 * (expected.sum() / without.sum() - 1)),
        })  # fmt: skip
    return rows


def run(cfg: Config) -> dict:
    if cfg.source.get("type") != "sihsus":
        raise RuntimeError("bill-parts reads the parts of a SIH/SUS bill; this dataset has another source")
    from costmodel.sources.sihsus.read import read_parts

    registry.setup(cfg)
    version = registry.version_for(cfg, registry.PRODUCTION)
    if version is None:
        raise RuntimeError(f"no '{registry.PRODUCTION}' model for {cfg.model_name}; run `costmodel evaluate` first")
    model = registry.load(cfg, registry.PRODUCTION)

    df = with_parts(load_splits(cfg), read_parts())
    live = df[df["split"] == "monitor"].copy()
    if live.empty:
        raise RuntimeError("the monitoring window is empty; check [split] in the config")
    live["typical"], live["expected"] = registry.predict_both(model, live)
    months = monthly(live)

    year = df["admission_date"].dt.year
    by_year = [{"year": int(y), "window": "/".join(sorted(g["split"].unique())), "n": int(len(g)),
                "federal_money_pct": float(100 * g["federal"].sum() / g[F.TARGET].sum()),
                "with_federal_pct": float(100 * (g["federal"] >= PAID_ONE).mean()),
                "manager_money_pct": float(100 * g["manager"].sum() / g[F.TARGET].sum())}
               for y, g in df.groupby(year)]  # fmt: skip

    has = live["federal"] >= PAID_ONE
    by_type = [{"admission_type": str(t), "n": int(len(g)),
                "with_complement_pct": float(100 * (g["federal"] >= PAID_ONE).mean()),
                "complement_money_pct": float(100 * g["federal"].sum() / g[F.TARGET].sum())}
               for t, g in live.groupby("admission_type")]  # fmt: skip
    proc = live.groupby("requested_procedure").agg(n=("federal", "size"), money=("federal", "sum"),
                                                   with_pct=("federal", lambda s: 100 * (s >= PAID_ONE).mean()))  # fmt: skip
    top = proc.sort_values("money", ascending=False).head(TOP)
    hospitals = live.groupby("provider_id")["federal"].sum()
    result = {
        "version": str(version.version),
        "window": [months[0]["period"], months[-1]["period"]],
        "mae": float(np.abs(live["typical"] - live[F.TARGET]).mean()),
        "mae_without_complement": float(np.abs(live["typical"] - (live[F.TARGET] - live["federal"])).mean()),
        "complement_times_table_median": float((live["federal"] / (live[F.TARGET] - live["federal"]))[has].median()),
        "hospitals": int(len(hospitals)),
        "hospitals_with_complement": int((hospitals >= PAID_ONE).sum()),
        "months": months,
        "by_year": by_year,
        "by_admission_type": by_type,
        "top_procedures": [{"requested_procedure": str(code), "n": int(r["n"]), "with_complement_pct": float(r["with_pct"]),
                            "share_of_complement_pct": float(100 * r["money"] / live["federal"].sum())}
                           for code, r in top.iterrows()],  # fmt: skip
    }
    write_report(cfg, result)
    print(f"monitoring window MAE {result['mae']:.2f}; against the bill without the federal complement"
          f" {result['mae_without_complement']:.2f}")  # fmt: skip
    return result


def write_report(cfg: Config, result: dict) -> None:
    out = cfg.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "bill_parts_monitor.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    months = pd.DataFrame(result["months"]).rename(columns={
        "with_complement_pct": "paid a complement (%)", "complement_per_admission": "complement per admission",
        "complement_money_pct": "complement, % of the money", "mae": "MAE", "mae_without_complement": "MAE, bill without it",
        "mae_admissions_with": "MAE, admissions with it", "mae_admissions_without": "MAE, admissions without it",
        "total_error_pct": "month's total (%)", "total_error_without_complement_pct": "month's total, bill without it (%)",
    })  # fmt: skip
    lines = [
        f"# What the bills of the monitoring window are made of ({cfg.name})",
        "",
        f"Generated by `costmodel bill-parts --config {cfg.path.as_posix() if cfg.path else ''}`. What was paid for a"
        " SIH/SUS admission is the table's hospital and professional parts plus complements paid on top of them. This"
        f" reads the federal complement of each bill from the raw files and scores the months {result['window'][0]} to"
        f" {result['window'][1]} with the production model (version {result['version']}): the MAE is the typical cost's,"
        " the month's total the expected cost's, as in the monitor. \"Bill without it\" judges the same predictions"
        " against what was paid minus the complement.",
        "",
        to_markdown(months.round(2), index=False),
        "",
        f"- Over the window the MAE is **{result['mae']:.2f}**; against the bills without the federal complement it is"
        f" **{result['mae_without_complement']:.2f}**.",
        f"- Where it is paid, the complement is {result['complement_times_table_median']:.2f} times the rest of the"
        f" bill (median). {result['hospitals_with_complement']} of {result['hospitals']} hospitals received any.",
        "",
        "## By admission type",
        "",
        to_markdown(pd.DataFrame(result["by_admission_type"]).round(2), index=False),
        "",
        f"## The {len(result['top_procedures'])} procedures that received most of it",
        "",
        to_markdown(pd.DataFrame(result["top_procedures"]).round(2), index=False),
        "",
        "## The complements in every year of the data",
        "",
        "Share of the money paid as a federal complement and as a complement by the state or city (no model involved).",
        "",
        to_markdown(pd.DataFrame(result["by_year"]).round(2), index=False),
        "",
        "A window already read is described here; no choice is made and nothing is registered. The complement is not"
        " known at admission, and whether an admission will be paid one depends on a programme, not on the price table.",
    ]
    (out / "bill_parts_monitor.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
