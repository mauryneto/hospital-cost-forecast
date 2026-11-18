"""``costmodel skill``: is the model worth it over a plain average?

On the month's total for *every* admission, a plain average (the mean cost per admission, times the number of
admissions) can come close to the model: prices rise, the mix of procedures shifts, the average ignores both, and the
two mistakes partly cancel. That says nothing about the totals people actually budget for: a hospital, a specialty,
a kind of admission. There the average gives the same amount to every admission and the mix no longer cancels.

So this scores the production model on the test window at several levels: the total of each month for all
admissions, then for each admission type, procedure group, bed specialty, diagnosis chapter, procedure subgroup and
hospital, and finally each admission on its own. Next to it, two rules anyone could compute: the plain average and
the mean per procedure.

* **Error on totals** at a level: the sum of |predicted total - paid total| over every (group, month), divided by
  everything paid. Weighted by money, so tiny groups do not dominate. Totals use the model's expected cost.
* **Gain over the plain average** (a skill score): how much smaller the model's error is, in percent. 0% means the
  model adds nothing over the average at that level; 100% would be a perfect forecast.

It describes a model already promoted; it chooses nothing and registers nothing.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from costmodel import features as F
from costmodel import registry
from costmodel.baselines import all_baselines
from costmodel.config import Config
from costmodel.evaluate import to_markdown
from costmodel.train import last_months, load_splits

LEVELS = {  # name -> feature columns that define a group (each crossed with the month)
    "all admissions": [],
    "admission type": ["admission_type"],
    "procedure group": ["procedure_prefix2"],
    "bed specialty": ["bed_specialty"],
    "diagnosis chapter": ["diagnosis_chapter"],
    "procedure subgroup": ["procedure_prefix4"],
    "hospital": ["provider_id"],
}
RULES = {"global_mean": "plain average", "mean_by_proc": "mean per procedure"}
MODEL = "model"


def totals_error(y: np.ndarray, pred: np.ndarray, groups: pd.DataFrame) -> tuple[float, int]:
    """(error on totals in percent, number of totals): sum of |predicted - paid| over the groups / everything paid."""
    frame = groups.reset_index(drop=True).assign(_y=np.asarray(y, dtype=float), _pred=np.asarray(pred, dtype=float))
    totals = frame.groupby(list(groups.columns), observed=True)[["_y", "_pred"]].sum()
    return float(100 * (totals["_pred"] - totals["_y"]).abs().sum() / totals["_y"].sum()), len(totals)


def gain(model_error: float, reference_error: float) -> float:
    """How much smaller the model's error is than the reference's, in percent (not defined against a reference that
    is exactly right)."""
    return float(100 * (1 - model_error / reference_error)) if reference_error > 0 else float("nan")


def table(y: np.ndarray, typical: np.ndarray, expected: np.ndarray, rules: dict[str, np.ndarray], X: pd.DataFrame,
          period: pd.Series) -> list[dict]:  # fmt: skip
    """One row per level. ``rules`` maps a rule's name to its prediction; the first one is the plain average."""
    plain = next(iter(rules))
    rows = []
    for name, columns in LEVELS.items():
        if any(X[c].nunique() < 2 for c in columns):
            continue  # an optional column this dataset does not fill
        groups = X[columns].assign(month=period.to_numpy())
        row = {"level": name + (" x month" if columns else ", each month"), "what": "error on totals (%)"}
        row[MODEL], row["totals"] = totals_error(y, expected, groups)
        for rule, pred in rules.items():
            row[rule] = totals_error(y, pred, groups)[0]
        rows.append(row | {"gain_pct": gain(row[MODEL], row[plain])})
    row = {"level": "each admission", "what": "mean absolute error", "totals": len(y)}
    row[MODEL] = float(np.abs(typical - y).mean())
    for rule, pred in rules.items():
        row[rule] = float(np.abs(pred - y).mean())
    rows.append(row | {"gain_pct": gain(row[MODEL], row[plain])})
    return rows


def run(cfg: Config) -> list[dict]:
    registry.setup(cfg)
    version = registry.version_for(cfg, registry.PRODUCTION)
    if version is None:
        raise RuntimeError(f"no '{registry.PRODUCTION}' model for {cfg.model_name}; run `costmodel evaluate` first")
    df = load_splits(cfg)
    test = df[df["split"] == "test"]
    if test.empty:
        raise RuntimeError("the test window is empty; check [split] in the config")
    fit_data = last_months(df[df["split"].isin(["train", "valid"])], int(version.tags.get("window_months", 0)))
    X_fit, X = F.build_features(fit_data), F.build_features(test)
    typical, expected = registry.predict_both(registry.load(cfg, registry.PRODUCTION), test)
    rules = {m.name: m.fit(X_fit, fit_data[F.TARGET]).predict(X) for m in all_baselines() if m.name in RULES}
    rules = {name: rules[name] for name in RULES}  # the plain average first
    rows = table(test[F.TARGET].to_numpy(), typical, expected, rules, X, test["period"])
    write_report(cfg, rows, str(version.version), test)
    for row in rows:
        print(f"{row['level']:<30} model {row[MODEL]:9.2f}   plain average {row['global_mean']:9.2f}"
              f"   gain {row['gain_pct']:5.1f}%")  # fmt: skip
    return rows


def write_report(cfg: Config, rows: list[dict], version: str, test: pd.DataFrame) -> None:
    out = cfg.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    payload = {"version": version, "rules": RULES, "rows": json.loads(pd.DataFrame(rows).to_json(orient="records"))}
    (out / "skill_test.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    frame = pd.DataFrame(rows).rename(columns=RULES | {MODEL: "this model", "gain_pct": "gain over the plain average (%)"})
    frame = frame[["level", "what", "totals", *RULES.values(), "this model", "gain over the plain average (%)"]]
    first, last = test["admission_date"].min(), test["admission_date"].max()
    lines = [
        f"# Is the model worth it over a plain average? ({cfg.name})",
        "",
        f"Generated by `costmodel skill --config {cfg.path.as_posix() if cfg.path else ''}`. Model: `{cfg.model_name}`"
        f" version {version}, on the {len(test):,} admissions of the test window ({first:%Y-%m-%d} to {last:%Y-%m-%d})."
        " The plain average is the mean cost per admission in the model's training data, given to every admission;"
        " the mean per procedure is the same idea, one mean for each procedure.",
        "",
        "For each level, the admissions are grouped (by hospital, by specialty...) and by month, and the predicted total"
        " of each group is compared with what was paid. The error is the sum of the absolute differences over all the"
        " groups, divided by everything paid (so it weighs each group by its money, and for all admissions it is close"
        " to, but not the same as, the monthly-total error of the test report, which averages the months). Totals use"
        " the model's expected cost; the last row uses its typical cost.",
        "",
        to_markdown(frame, index=False),
        "",
        "The gain is how much smaller the model's error is than the plain average's: 0% would mean the model adds"
        " nothing at that level.",
        "",
        "This describes a model already in production. Nothing was chosen with it and nothing was registered.",
    ]
    (out / "skill_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
