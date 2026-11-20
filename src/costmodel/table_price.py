"""``costmodel table-price``: does the model do better when given the official price table?

The SUS publishes a price for every procedure, month by month (SIGTAP). The candidate learns prices from past bills, so
it follows a published change only after it is retrained on bills paid at the new price. Here every admission gets the
table's row for its requested procedure as extra inputs: the price (hospital part, professional part and their sum),
the stay the table assumes, its points, complexity and kind of financing.

**Which month's table.** Two readings, because the files are not always there when the month begins:

* **the admission month's table**: the best case, as if every change were known the day it took effect;
* **the table as written**: the newest table whose file is dated before the admission month began. This is what a
  model in use could have read, judged by the file dates on the server (a file rewritten later carries the later
  date, so this errs on the careful side).

A third variant uses the admission month's price **as the unit**: the model predicts the cost relative to the table
price, so a price that goes up 40% moves the prediction by 40% with every other input equal.

Each variant is fitted with the two settings the search chose (the typical cost's and the expected cost's), on the
training window, and compared with the candidate as it is on the same validation admissions, with a 95% interval from
drawing the admissions again. The error is also read apart for the procedures whose table price changed inside the
validation window, before and after the change: that is where a table can help, and where it can overshoot, because
what is paid for an admission is more than the table price of its main procedure.

Validation window only: the test window is not read and nothing is registered. The table over the later windows is
described (when it moved), which reads no bill.
"""

from __future__ import annotations

import json
import time

import mlflow
import numpy as np
import pandas as pd

from costmodel import features as F
from costmodel import registry
from costmodel.candidate import CostModel
from costmodel.config import Config
from costmodel.evaluate import summary, to_markdown
from costmodel.foundation import gap
from costmodel.sources import sigtap
from costmodel.train import choose, choose_for_totals, last_months, load_splits, setting_label

COLUMNS = ["table_price", "table_hospital", "table_professional", "table_stay_days", "table_points",
           "table_complexity", "table_financing"]  # fmt: skip
DRAWS = 1_000  # resamples of the validation admissions behind each interval
STEP_PCT = 0.5  # a month counts as a step of the table when the basket's mean table price moves this much
AS_IT_IS = "the candidate as it is"
SAME_MONTH = "the admission month's table, as inputs"
AS_WRITTEN = "the table as written before the month began, as inputs"
AS_UNIT = "cost predicted relative to the admission month's table price"
GROUPS = ("not re-priced", "re-priced, before the change", "re-priced, from the change on")


def known_months(admission_months, written: pd.Series) -> dict[str, str]:
    """For each admission month ("YYYY-MM"), the newest table whose file was written before that month began.

    ``written`` maps a table's month to the date of its file. The oldest table is used when none was written yet.
    """
    tables = sorted(written.index)
    out = {}
    for month in admission_months:
        start = pd.Timestamp(f"{month}-01")
        ready = [t for t in tables if t <= month and written[t] < start]
        out[month] = ready[-1] if ready else tables[0]
    return out


def attach(X: pd.DataFrame, procedure, month, table: pd.DataFrame) -> pd.DataFrame:
    """``X`` with the table's columns for each admission's procedure in the given table month (missing if not there)."""
    keys = pd.DataFrame({"month": np.asarray(month), "procedure": np.asarray(procedure)})
    row = keys.merge(table[["month", "procedure", *COLUMNS]], how="left", on=["month", "procedure"], validate="m:1")
    return X.assign(**{c: row[c].to_numpy(dtype=float) for c in COLUMNS})


def repriced_from(table: pd.DataFrame, months: list[str]) -> pd.Series:
    """Procedure -> the first of ``months`` in which its table price differs from the month before (only those that
    changed). ``table`` must also hold the month before the first one."""
    wide = table.pivot(index="procedure", columns="month", values="table_price").sort_index(axis=1)
    moved = (wide.diff(axis=1).abs() > 0.005)[months]
    return moved.idxmax(axis=1)[moved.any(axis=1)]


class TableModel(CostModel):
    """The candidate with the table's columns as inputs and, with ``relative``, the table price as its unit."""

    def __init__(self, *args, relative: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.relative = relative

    @property
    def columns(self) -> list[str]:
        return [*super().columns, *COLUMNS]

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        return super()._encode(X).assign(**{c: X[c].astype(float) for c in COLUMNS})

    @staticmethod
    def unit(X: pd.DataFrame) -> np.ndarray:
        """The table price, or 1 where the table has none (the procedure is then priced by the trees alone)."""
        price = X["table_price"].to_numpy(dtype=float)
        return np.where(price > 0, price, 1.0)

    def fit(self, X, y, X_valid=None, y_valid=None, early_stopping_rounds=None, sample_weight=None) -> TableModel:
        if self.relative:
            y = np.asarray(y, dtype=float) / self.unit(X)
            if X_valid is not None:
                y_valid = np.asarray(y_valid, dtype=float) / self.unit(X_valid)
        return super().fit(X, y, X_valid, y_valid, early_stopping_rounds, sample_weight)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = super().predict(X)
        return pred * self.unit(X) if self.relative else pred


def table_facts(table: pd.DataFrame, valid: pd.DataFrame) -> dict:
    """What the table is, measured with the validation admissions as a fixed basket: how much of the money it
    explains, when it moved over all the months read, and how late its files are dated."""
    wide = table.pivot(index="procedure", columns="month", values="table_price").sort_index(axis=1)
    written = table.groupby("month")["written"].first()
    basket = valid["requested_procedure"].value_counts()
    always = wide.reindex(basket.index).notna().all(axis=1)
    b, w = basket[always], wide.reindex(basket.index)[always]
    index = w.mul(b, axis=0).sum() / b.sum()
    steps = []
    for before, month in zip(index.index[:-1], index.index[1:], strict=True):
        pct = 100 * (index[month] / index[before] - 1)
        if abs(pct) >= STEP_PCT:
            moved = (w[month] - w[before]).abs() > 0.005
            steps.append({"month": month, "step_pct": float(pct), "procedures": int(moved.sum()),
                          "admissions_pct": float(100 * b[moved].sum() / b.sum()),
                          "file_dated": written[month].strftime("%Y-%m-%d")})  # fmt: skip
    moved_in = (wide.diff(axis=1).abs() > 0.005).iloc[:, 1:]  # the first month has nothing before it
    by_year = moved_in.T.groupby(moved_in.columns.str[:4]).any().T.sum()
    own = attach(valid[[]], valid["requested_procedure"], valid["period"], table)["table_price"].to_numpy()
    paid = valid[F.TARGET].to_numpy(dtype=float)
    start = pd.Series(pd.to_datetime([f"{m}-01" for m in written.index]), index=written.index)
    days = (written - start).dt.days
    return {
        "months": [str(index.index[0]), str(index.index[-1])],
        "basket_in_every_month_pct": float(100 * b.sum() / basket.sum()),
        "priced_pct": float(100 * (own > 0).mean()),
        "money_pct": float(100 * np.nansum(own) / paid.sum()),
        "paid_within_5pct": float(100 * (np.abs(paid / np.where(own > 0, own, np.nan) - 1) < 0.05).mean()),
        "index_first": float(index.iloc[0]),
        "index_last": float(index.iloc[-1]),
        "steps": steps,
        "procedures_repriced_by_year": {str(year): int(n) for year, n in by_year.items()},
        "file_days_median": float(days.median()),
        "files_dated_after_month": int((written >= start + pd.offsets.MonthBegin(1)).sum()),
        "files": int(len(written)),
    }


def by_group(paid: np.ndarray, group: np.ndarray, typical: np.ndarray, expected: np.ndarray) -> dict:
    out = {}
    for name in GROUPS:
        rows = group == name
        if rows.any():
            out[name] = {"mae": float(np.abs(typical[rows] - paid[rows]).mean()),
                         "expected_sum_pct": float(100 * (expected[rows].sum() / paid[rows].sum() - 1))}  # fmt: skip
    return out


def run(cfg: Config) -> pd.DataFrame:
    searched = cfg.reports_dir / "train_valid.json"
    if not searched.exists():
        raise RuntimeError(f"{searched} not found; run `costmodel train` first (its chosen settings are compared)")
    search = pd.DataFrame(json.loads(searched.read_text(encoding="utf-8")))
    settings = {"typical": choose(search, cfg), "expected": choose_for_totals(search)}
    tc = cfg.train

    df = load_splits(cfg)
    history = df[df["split"].isin(["train", "valid"])]  # bills of later windows are never read
    if not (history["procedure_code_system"] == "SIGTAP").any():
        raise RuntimeError("no SIGTAP procedure codes in this dataset: the SUS price table does not apply to it")
    in_train = (history["split"] == "train").to_numpy()
    valid = history[~in_train]
    month = history["period"]
    first = (pd.Period(month.min(), "M") - 1).strftime("%Y-%m")  # one month earlier: to see a change in the first
    table = sigtap.load(first, df["period"].max())
    written = table.groupby("month")["written"].first()
    table_month = {SAME_MONTH: month, AS_WRITTEN: month.map(known_months(month.unique(), written)), AS_UNIT: month}

    changed = repriced_from(table, sorted(valid["period"].unique()))
    since = valid["requested_procedure"].map(changed).astype(object)
    before = valid["period"].astype(str).to_numpy() < since.fillna("").to_numpy(dtype=str)
    group = np.where(since.isna(), GROUPS[0], np.where(before, GROUPS[1], GROUPS[2]))
    paid, month_code = valid[F.TARGET].to_numpy(dtype=float), pd.factorize(valid["period"])[0]
    facts = table_facts(table, valid)
    facts["repriced_admissions_pct"] = float(100 * (group != GROUPS[0]).mean())
    facts["repriced_money_pct"] = float(100 * paid[group != GROUPS[0]].sum() / paid.sum())
    own = attach(valid[[]], valid["requested_procedure"], valid["period"], table)["table_price"].to_numpy()
    facts["repriced"] = {g: {"n": int((group == g).sum()), "mean_paid": float(paid[group == g].mean()),
                             "mean_table_price": float(np.nanmean(own[group == g]))}
                         for g in GROUPS[1:] if (group == g).any()}  # fmt: skip

    X_plain = F.build_features(history)
    registry.setup(cfg)
    rows, groups, first_pair = [], {}, None
    with mlflow.start_run(run_name="table price"):
        mlflow.set_tags({"step": "table_price", "dataset": cfg.name})
        mlflow.log_params({f"setting_{k}": setting_label(s) for k, s in settings.items()} | {"draws": DRAWS})
        for variant in (AS_IT_IS, SAME_MONTH, AS_WRITTEN, AS_UNIT):
            t0 = time.perf_counter()
            if variant == AS_IT_IS:
                X_all = X_plain
            else:
                X_all = attach(X_plain, history["requested_procedure"], table_month[variant], table)
            row, pred = {"variant": variant}, {}
            for role, s in settings.items():
                train = last_months(history[in_train], int(s["window_months"]))
                args = (s["objective"], int(s["num_leaves"]), tc.learning_rate, tc.max_rounds, tc.min_category_count,
                        tc.seed, bool(s["use_time"]), tc.typical_scaled if role == "typical" else True)  # fmt: skip
                model = CostModel(*args) if variant == AS_IT_IS else TableModel(*args, relative=variant == AS_UNIT)
                model.fit(X_all.loc[train.index], train[F.TARGET], X_all[~in_train], paid, tc.early_stopping_rounds)
                pred[role] = model.predict(X_all[~in_train])
                metrics = summary(paid, pred[role], valid["period"])
                keep = ("mae", "mae_top_decile") if role == "typical" else ("monthly_total_mape", "bias_pct")
                row |= {f"{role}_trees": model.best_iteration_, **{k: metrics[k] for k in keep}}
                if variant != AS_IT_IS:
                    gain = model.feature_importance()
                    row[f"{role}_table_gain_pct"] = float(100 * gain[COLUMNS].sum() / gain.sum())
            pair = (pred["typical"], pred["expected"])
            if first_pair is None:
                first_pair = pair
            else:
                row |= gap(paid, month_code, first_pair, pair, tc.seed, DRAWS)
            rows.append(row)
            groups[variant] = by_group(paid, group, *pair)
            mlflow.log_metrics({f"valid_mae_{len(rows)}": row["mae"], f"valid_total_{len(rows)}": row["monthly_total_mape"]})
            print(f"{variant:<60} MAE {row['mae']:9.2f}  monthly total err {row['monthly_total_mape']:5.2f}%"
                  f"  ({time.perf_counter() - t0:.0f}s)", flush=True)  # fmt: skip
    results = pd.DataFrame(rows)
    write_report(cfg, results, groups, facts, {k: setting_label(s) for k, s in settings.items()})
    return results


def interval(row, key: str, unit: str = "") -> str:
    if pd.isna(row.get(key)):
        return ""
    return f"{row[key]:+.2f}{unit} ({row[key + '_low']:+.2f} to {row[key + '_high']:+.2f})"


def repriced_lines(repriced: dict) -> list[str]:
    """How far the table price and the bill moved for the re-priced procedures (needs both sides of the change)."""
    if set(GROUPS[1:]) - set(repriced):
        return []
    before, after = repriced[GROUPS[1]], repriced[GROUPS[2]]
    return [
        f"Before the change, these admissions had a mean table price of {before['mean_table_price']:.2f} and were paid"
        f" {before['mean_paid']:.2f} on average; from the change on, {after['mean_table_price']:.2f} and"
        f" {after['mean_paid']:.2f}. The table price is"
        f" {100 * (after['mean_table_price'] / before['mean_table_price'] - 1):+.1f}%, what was paid"
        f" {100 * (after['mean_paid'] / before['mean_paid'] - 1):+.1f}% (not the same admissions on each side).",
        "",
    ]


def write_report(cfg: Config, results: pd.DataFrame, groups: dict, facts: dict, settings: dict) -> None:
    out = cfg.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    payload = {"settings": settings, "typical_scaled": cfg.train.typical_scaled, "draws": DRAWS, "table": facts,
               "rows": json.loads(results.to_json(orient="records")), "groups": groups}  # fmt: skip
    (out / "table_price_valid.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")

    first = results.iloc[0]
    table = pd.DataFrame({
        "variant": results["variant"],
        "typical cost: MAE": results["mae"].map("{:.2f}".format),
        "against the first row (95% interval)": [interval(r, "mae_gap") for _, r in results.iterrows()],
        "MAE, top 10%": results["mae_top_decile"].map("{:.2f}".format),
        "expected cost: error on the month's total": results["monthly_total_mape"].map("{:.2f}%".format),
        "against the first row, in points": [interval(r, "total_gap") for _, r in results.iterrows()],
    })  # fmt: skip
    shown = [g for g in GROUPS if g in groups[AS_IT_IS]]
    by = pd.DataFrame(
        [{"variant": v, **{f"{g}: MAE": f"{groups[v][g]['mae']:.2f}" for g in shown},
          **{f"{g}: expected added up": f"{groups[v][g]['expected_sum_pct']:+.2f}%" for g in shown}} for v in groups]
    )  # fmt: skip
    steps = pd.DataFrame(facts["steps"])
    best_mae, best_total = results["mae"].min(), results["monthly_total_mape"].min()
    mae_gain = 100 * (first["mae"] - best_mae) / first["mae"]
    total_gain = 100 * (first["monthly_total_mape"] - best_total) / first["monthly_total_mape"]
    written_total = results.loc[results["variant"] == AS_WRITTEN, "monthly_total_mape"].iloc[0]
    written_gain = 100 * (first["monthly_total_mape"] - written_total) / first["monthly_total_mape"]
    lines = [
        f"# The official price table as an input, on the validation window ({cfg.name})",
        "",
        f"Generated by `costmodel table-price --config {cfg.path.as_posix() if cfg.path else ''}`. Every admission gets"
        " the SIGTAP row of its requested procedure as extra inputs (price, the stay the table assumes, points,"
        " complexity, financing). The typical cost uses the setting the search chose for it"
        f" ({settings['typical']}) and is given {'with' if cfg.train.typical_scaled else 'without'} the smearing"
        f" factor; the expected cost uses its own ({settings['expected']}). Fitted on the training window, compared on"
        f" the same validation admissions; intervals from {DRAWS:,} resamples (negative = better than the first row).",
        "",
        "## The table itself",
        "",
        f"- Tables read: {facts['files']} monthly files, {facts['months'][0]} to {facts['months'][1]}.",
        f"- {facts['priced_pct']:.1f}% of the validation admissions have a table price for their requested procedure."
        f" Added up, those prices are **{facts['money_pct']:.1f}%** of what was paid; {facts['paid_within_5pct']:.1f}% of"
        " the admissions were paid within 5% of the table price. The rest of a bill is what happens during the stay"
        " (intensive care days, other procedures, materials) and complements paid on top of the table.",
        f"- The same basket of admissions (the validation window's) costs {facts['index_first']:.2f} per admission at"
        f" the first table and {facts['index_last']:.2f} at the last. Months in which it moved by at least"
        f" {STEP_PCT}%:",
        "",
        to_markdown(steps.round(2), index=False) if len(steps) else "(none)",
        "",
        "- Procedures whose table price changed at least once, by year: "
        + ", ".join(f"{year}: {n}" for year, n in facts["procedures_repriced_by_year"].items()) + ".",
        f"- File dates: half of the tables are dated within {facts['file_days_median']:.0f} days of the 1st of their"
        f" month; {facts['files_dated_after_month']} of {facts['files']} are dated after their month had ended.",
        "",
        "## With the table as an input",
        "",
        to_markdown(table, index=False),
        "",
        f"The best variant lowers the typical cost's MAE by {mae_gain:.2f}% (the gate asks"
        f" {100 * cfg.gate.min_mae_gain:.0f}% to replace a model) and the expected cost's error on the"
        f" month's total by {total_gain:.1f}% of its value"
        f" (the gate asks {100 * cfg.gate.min_total_gain:.0f}%); with the table as written, the second is"
        f" {written_gain:.1f}%.",
        "",
        "## Where the difference is: procedures re-priced inside the window",
        "",
        f"{facts['repriced_admissions_pct']:.2f}% of the validation admissions ({facts['repriced_money_pct']:.2f}% of"
        " the money) are of a procedure whose table price changed inside the window. \"Expected added up\" is the"
        " expected cost of the group against what was paid for it (negative = too low).",
        "",
        to_markdown(by, index=False),
        "",
        *repriced_lines(facts.get("repriced", {})),
        "A gain on the month's total is only a gain if each group gets closer to what was paid. A variant that moves"
        " the re-priced procedures from too low to too high, while the others stay too low, improves the total by"
        " adding two errors of opposite sign.",
        "",
        "Validation window only: the test window was not read and nothing was registered. Serving a model like this"
        " needs the current table at prediction time.",
    ]
    (out / "table_price_valid.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
