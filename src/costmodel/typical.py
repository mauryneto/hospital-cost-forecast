"""``costmodel typical``: should the typical cost be a median?

The typical cost is the number meant to be close to each admission. It started as least squares on log(cost),
exponentiated and then scaled up by Duan's smearing factor so that it adds up better. Since the expected cost now does
the adding up, that factor may be working against what the typical cost is for: on a right-skewed cost the number
closest to each admission is the median, and a median is not scaled up. On the 2025 check a plain table of medians
nearly tied with the model per admission, which is what raised the question. What this found is why
``[train] typical_scaled`` exists; the comparison is the same whichever way the config now gives the number.

Everything here is fitted on the training window and measured on validation, with the setting the search chose for
the typical cost (leaves, window, time feature). What changes is only what the number is:

* **scaled**: exp(prediction) times the smearing factor. Every other row is compared with this one.
* **unscaled**: the same fitted model without the factor. A median if the errors in log(cost) are symmetric.
* **median of log(cost)**: least absolute error on log(cost). The fitted value is the median itself, with no
  assumption about symmetry; the median of the log is the log of the median.
* **median of cost**: least absolute error on the cost. The same target, fitted in money.

Each is compared with the scaled number on the same admissions, with a 95% interval from drawing the admissions
again. A median is expected to be closer to each admission and to fall well short on the month's total; the second
does not matter to the model (the total comes from the expected cost) but it is reported, so the price of the change
is on the table.

The error is also read in two bands, because one average hides both: the 10% most expensive admissions hold most of
the money and most of the error, and the other 90% are what "a typical admission" means. The bands are cut by what
was paid, which is known only afterwards: they are a way to read the error, not something the model can use.

Validation window only: the test window is not read and nothing is registered.
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
from costmodel.evaluate import decile_table, summary, to_markdown
from costmodel.foundation import GAPS, gap
from costmodel.train import choose, last_months, load_splits, setting_label

DRAWS = 1_000  # resamples of the validation admissions behind each interval
KEEP = ("mae", "median_ae", "mae_top_decile", "bias_pct", "monthly_total_mape")
BANDS = ("mae_common", "error_share_top_decile")
SCALED, UNSCALED = "scaled", "unscaled"


class MedianModel(CostModel):
    """Least absolute error on log(cost) or on the cost itself: the fitted value is the conditional median."""

    def __init__(self, log_target: bool = True, num_leaves: int = 255, learning_rate: float = 0.1,
                 n_estimators: int = 2000, min_category_count: int = 20, seed: int = 42, use_time: bool = True):  # fmt: skip
        super().__init__("log_l2", num_leaves, learning_rate, n_estimators, min_category_count, seed, use_time)
        self.log_target = log_target
        self.objective = "median_log" if log_target else "median_cost"

    def _target(self, y) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        return np.log(y) if self.log_target else y

    def _booster(self):
        return super()._booster().set_params(objective="l1")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.booster_.predict(self._encode(X), num_iteration=self.best_iteration_)
        return np.exp(raw) if self.log_target else raw


def bands(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """The error in two bands of the paid amount: the MAE on the 90% least expensive admissions, and the share (%) of
    all the absolute error that falls on the 10% most expensive."""
    deciles = decile_table(y, pred)
    error = deciles["mae"] * deciles["n"]
    common = deciles.index != 10
    return {"mae_common": float(error[common].sum() / deciles.loc[common, "n"].sum()),
            "error_share_top_decile": float(100 * error.loc[10] / error.sum())}  # fmt: skip


def money_share_top_decile(y: np.ndarray) -> float:
    """Share (%) of everything paid that went to the 10% most expensive admissions."""
    deciles = decile_table(y, y)
    money = deciles["mean_actual"] * deciles["n"]
    return float(100 * money.loc[10] / money.sum())


def lead(mae: float, baseline_mae: float | None) -> float | None:
    """How much lower the MAE is than the best baseline's, in % of the baseline's (what the gate asks 2% of)."""
    return None if baseline_mae is None else 100 * (baseline_mae - mae) / baseline_mae


def best_baseline(cfg: Config) -> tuple[str, float] | None:
    """The baseline with the lowest validation MAE, from ``costmodel baselines`` (all training years)."""
    path = cfg.reports_dir / "baselines_valid.json"
    if not path.exists():
        return None
    rows = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    rows = rows[rows["train_window"] == rows["train_window"].iloc[0]]
    best = rows.sort_values("mae").iloc[0]
    return str(best["model"]), float(best["mae"])


def run(cfg: Config) -> pd.DataFrame:
    searched = cfg.reports_dir / "train_valid.json"
    if not searched.exists():
        raise RuntimeError(f"{searched} not found; run `costmodel train` first (this uses the typical cost's setting)")
    setting = choose(pd.DataFrame(json.loads(searched.read_text(encoding="utf-8"))), cfg)
    tc, leaves, use_time = cfg.train, int(setting["num_leaves"]), bool(setting["use_time"])
    baseline = best_baseline(cfg)

    df = load_splits(cfg)
    train, valid = df[df["split"] == "train"], df[df["split"] == "valid"]
    window = last_months(train, int(setting["window_months"]))
    X, y = F.build_features(window), window[F.TARGET]
    X_valid, y_valid = F.build_features(valid), valid[F.TARGET]
    paid, month = y_valid.to_numpy(dtype=float), pd.factorize(valid["period"])[0]

    registry.setup(cfg)
    rows, first = [], None

    def add(number: str, model: CostModel, pred: np.ndarray, seconds: float) -> None:
        metrics = summary(paid, pred, valid["period"])
        row = {"number": number, "trees": model.best_iteration_, **{k: metrics[k] for k in KEEP}, **bands(paid, pred),
               "lead_over_baseline_pct": lead(metrics["mae"], baseline[1] if baseline else None)}  # fmt: skip
        row |= dict.fromkeys(GAPS) if first is None else gap(paid, month, (first, first), (pred, pred), tc.seed, DRAWS)
        row["seconds"] = round(seconds, 1)
        rows.append(row)
        mlflow.log_metrics({f"valid_{k}_{len(rows)}": metrics[k] for k in ("mae", "monthly_total_mape")})
        print(f"{number:<22} trees {row['trees']:<5} MAE {row['mae']:9.2f}  top decile {row['mae_top_decile']:9.2f}"
              f"  monthly total err {row['monthly_total_mape']:5.2f}%  ({row['seconds']}s)", flush=True)  # fmt: skip

    with mlflow.start_run(run_name="typical"):
        mlflow.set_tags({"step": "typical", "dataset": cfg.name})
        mlflow.log_params({"setting": setting_label(setting), "draws": DRAWS})
        t0 = time.perf_counter()
        scaled = CostModel("log_l2", leaves, tc.learning_rate, tc.max_rounds, tc.min_category_count, tc.seed, use_time)
        scaled.fit(X, y, X_valid, y_valid, tc.early_stopping_rounds)
        pred = scaled.predict(X_valid)
        add(SCALED, scaled, pred, time.perf_counter() - t0)
        first, factor = pred, scaled.smearing_
        add(UNSCALED, scaled, pred / factor, 0.0)
        for log_target in (True, False):
            t0 = time.perf_counter()
            model = MedianModel(log_target, leaves, tc.learning_rate, tc.max_rounds, tc.min_category_count, tc.seed,
                                use_time)  # fmt: skip
            model.fit(X, y, X_valid, y_valid, tc.early_stopping_rounds)
            name = "median of log(cost)" if log_target else "median of cost"
            add(name, model, model.predict(X_valid), time.perf_counter() - t0)
    results = pd.DataFrame(rows)
    write_report(cfg, results, setting_label(setting), factor, len(window), len(valid), baseline,
                 money_share_top_decile(paid))  # fmt: skip
    return results


def interval(row, key: str, unit: str = "") -> str:
    if row[key] is None or pd.isna(row[key]):
        return "-"
    return f"{row[key]:+.2f}{unit} ({row[f'{key}_low']:+.2f} to {row[f'{key}_high']:+.2f})"


def write_report(cfg: Config, results: pd.DataFrame, setting: str, factor: float, n_train: int, n_valid: int,
                 baseline: tuple[str, float] | None, money_top: float) -> None:  # fmt: skip
    out = cfg.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    best = results.sort_values("mae").iloc[0]
    scaled = results.iloc[0]
    in_use = SCALED if cfg.train.typical_scaled else UNSCALED
    payload = {
        "setting": setting,
        "in_use": in_use,
        "factor": factor,
        "n_train": n_train,
        "n_valid": n_valid,
        "draws": DRAWS,
        "money_share_top_decile": money_top,
        "baseline": {"model": baseline[0], "mae": baseline[1]} if baseline else None,
        "best": best["number"],
        "rows": json.loads(results.to_json(orient="records")),
    }
    (out / "typical_valid.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")

    table = results[["number", "trees", *KEEP]].copy()
    if baseline:
        table["lead over the best baseline (%)"] = results["lead_over_baseline_pct"]
    split = results[["number", "mae", "mae_common", "mae_top_decile", "error_share_top_decile", "median_ae"]].rename(
        columns={
            "mae": "MAE, all admissions",
            "mae_common": "MAE, the 90% least expensive",
            "mae_top_decile": "MAE, the 10% most expensive",
            "error_share_top_decile": "share of the error on the 10% most expensive (%)",
            "median_ae": "error of the middle admission",
        }
    )
    gaps = pd.DataFrame(
        {
            "number": results["number"],
            "MAE against the scaled": [interval(r, "mae_gap") for _, r in results.iterrows()],
            "its own monthly total against the scaled (points)": [interval(r, "total_gap") for _, r in results.iterrows()],
        }
    ).iloc[1:]
    lines = [
        f"# Should the typical cost be a median? Validation window ({cfg.name})",
        "",
        f"Generated by `costmodel typical --config {cfg.path.as_posix() if cfg.path else ''}`. Fitted on the training"
        f" window ({n_train:,} admissions) and measured on validation ({n_valid:,}), all with the setting the search"
        f" chose for the typical cost: {setting}. Only what the number is changes.",
        "",
        f"- **{SCALED}**: exp(prediction of log cost) times the smearing factor, here {factor:.3f}.",
        f"- **{UNSCALED}**: the same fitted model without the factor.",
        "- **median of log(cost)**, **median of cost**: least absolute error, on log(cost) and on the cost.",
        "",
        f"The config (`[train] typical_scaled`) gives the typical cost as the **{in_use}** number.",
        "",
        to_markdown(table, index=False),
        "",
        "`bias_pct` and `monthly_total_mape` are those of the number itself, as if it were added up. The model does"
        " not add up the typical cost (the expected cost is for that); they show what a median gives up."
        + (
            f" The best baseline on this window is `{baseline[0]}` (MAE {baseline[1]:.2f}); the gate asks for an MAE at"
            f" least {100 * cfg.gate.min_mae_gain:.0f}% lower."
            if baseline
            else ""
        ),
        "",
        "## Typical admissions and the most expensive ones",
        "",
        f"The 10% most expensive admissions received {money_top:.1f}% of everything paid in this window, and most of the"
        " error per admission falls on them. They are real bills (intensive care, major surgery, long stays), and"
        " which admission will be one of them is not known when the patient is admitted, so they are not left out:"
        " the error is read in two bands instead. The bands are cut by what was paid, known only afterwards.",
        "",
        to_markdown(split, index=False),
        "",
        "The error of the middle admission is the median absolute error: half of the admissions are priced closer"
        " than this.",
        "",
        f"## Against the scaled number, on the same admissions (95% interval, {DRAWS:,} resamples)",
        "",
        "Negative = lower error than the scaled number's.",
        "",
        to_markdown(gaps, index=False),
        "",
        f"**Lowest MAE:** {best['number']} ({best['mae']:.2f} against {scaled['mae']:.2f} scaled).",
        "",
        "Validation window only: the test window was not read and nothing was registered.",
    ]
    (out / "typical_valid.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
