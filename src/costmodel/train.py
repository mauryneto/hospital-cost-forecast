"""Train the candidate: choose its settings on validation, refit, and register it as ``staging``.

1. **Search.** Every combination of training window x objective x tree size x time feature in ``[train]`` is fitted on
   the training window and measured on the validation window, each as an MLflow run. The test window is not read here.
2. **Choose two settings,** one for each number the model gives (see ``candidate.CostPair``). For the **typical
   cost**: lowest validation MAE among the settings whose monthly-total error is within the gate's ceiling (if none
   is, lowest MAE overall). For the **expected cost**: lowest error on the month's total. They can be the same.
   With ``typical_scaled = false`` the typical cost of a ``log_l2`` setting is given without the smearing factor, and
   the MAE that ranks the settings is that of the number as given (``mae_unscaled``).
3. **Refit** both on training + validation (the most recent data the model can have before the test year), each with
   the number of trees found by early stopping, and register them as one version in MLflow with alias ``staging``.

Whether it goes to ``production`` is decided later, by ``costmodel evaluate`` (see ``holdout.py``).
"""

from __future__ import annotations

import copy
import itertools
import json
import time

import mlflow
import numpy as np
import pandas as pd

from costmodel import features as F
from costmodel import registry
from costmodel.candidate import CostModel, CostPair
from costmodel.config import Config
from costmodel.evaluate import summary, to_markdown
from costmodel.sources import load_table
from costmodel.split import assign

EXPANDING = 0  # "sliding months" value meaning: use every month available


def window_label(months: int) -> str:
    return "expanding" if months == EXPANDING else f"sliding {months} months"


def setting_label(row) -> str:
    """One row of the search in words: 'log_l2, 255 leaves, expanding, with time'."""
    return f"{row['objective']}, {int(row['num_leaves'])} leaves, {row['window']}{', with time' if row['use_time'] else ''}"


def last_months(df: pd.DataFrame, months: int) -> pd.DataFrame:
    """Keep the last ``months`` whole calendar months of admissions (all of them for the expanding window)."""
    if months == EXPANDING:
        return df
    period = df["admission_date"].dt.to_period("M")
    return df[period > period.max() - months]


def load_splits(cfg: Config) -> pd.DataFrame:
    df = load_table(cfg)
    df["split"] = assign(df["admission_date"], cfg.split)
    df["period"] = df["admission_date"].dt.to_period("M").astype("string")
    return df


def search(cfg: Config, train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    tc = cfg.train
    X_valid, y_valid = F.build_features(valid), valid[F.TARGET]
    rows = []
    for months in (EXPANDING, *tc.sliding_months):
        window = last_months(train, months)
        X, y = F.build_features(window), window[F.TARGET]
        for objective, leaves, use_time in itertools.product(tc.objectives, tc.num_leaves, tc.use_time):
            model = CostModel(objective, leaves, tc.learning_rate, tc.max_rounds, tc.min_category_count, tc.seed, use_time)
            t0 = time.perf_counter()
            label = f"{model.name} {leaves} leaves, {window_label(months)}{', with time' if use_time else ''}"
            with mlflow.start_run(run_name=label, nested=True):
                model.fit(X, y, X_valid, y_valid, tc.early_stopping_rounds)
                pred = model.predict(X_valid)
                metrics = summary(y_valid.to_numpy(), pred, valid["period"])
                # The same fit without the smearing factor (1 unless log_l2): what an unscaled typical cost would give.
                metrics["mae_unscaled"] = float(np.abs(pred / model.smearing_ - y_valid.to_numpy()).mean())
                row = {
                    "objective": objective,
                    "num_leaves": leaves,
                    "use_time": use_time,
                    "window_months": months,
                    "window": window_label(months),
                    "n_train": len(window),
                    "trees": model.best_iteration_,
                    **metrics,
                    "seconds": round(time.perf_counter() - t0, 1),
                }
                mlflow.log_params(model.params() | {"window": window_label(months), "n_train": len(window)})
                mlflow.log_metrics({f"valid_{k}": v for k, v in metrics.items()} | {"trees": model.best_iteration_})
            rows.append(row)
            print(
                f"{row['window']:<20} {objective:<8} leaves {leaves:<4} time {'yes' if use_time else 'no ':<4}"
                f"trees {row['trees']:<5} "
                f"MAE {row['mae']:9.2f}  monthly total err {row['monthly_total_mape']:5.2f}%  ({row['seconds']}s)"
            )
    return pd.DataFrame(rows)


def typical_mae(results: pd.DataFrame, cfg: Config) -> str:
    """The column holding the MAE of the typical cost as the config gives it (older reports have only ``mae``)."""
    return "mae" if cfg.train.typical_scaled or "mae_unscaled" not in results else "mae_unscaled"


def choose(results: pd.DataFrame, cfg: Config) -> pd.Series:
    ceiling = cfg.gate.max_monthly_total_mape
    eligible = results if ceiling is None else results[results["monthly_total_mape"] <= ceiling]
    return (eligible if len(eligible) else results).sort_values(typical_mae(results, cfg)).iloc[0]


def choose_for_totals(results: pd.DataFrame) -> pd.Series:
    """The setting for the expected cost: lowest error on the month's total (ties: lowest MAE)."""
    return results.sort_values(["monthly_total_mape", "mae"]).iloc[0]


def refit(cfg: Config, setting: pd.Series, known: pd.DataFrame, scaled: bool = True) -> CostModel:
    """One setting of the search, fitted on its window of ``known`` with the number of trees the search found."""
    tc, months = cfg.train, int(setting["window_months"])
    data = last_months(known, months)
    model = CostModel(setting["objective"], int(setting["num_leaves"]), tc.learning_rate, int(setting["trees"]),
                      tc.min_category_count, tc.seed, bool(setting["use_time"]), scaled)  # fmt: skip
    print(f"refitting {model.name} ({window_label(months)}) on {len(data):,} admissions (train + valid)")
    return model.fit(F.build_features(data), data[F.TARGET])


def run(cfg: Config) -> str:
    """Returns the registered version number."""
    df = load_splits(cfg)
    train, valid = df[df["split"] == "train"], df[df["split"] == "valid"]
    registry.setup(cfg)
    with mlflow.start_run(run_name="train") as parent:
        mlflow.set_tags({"step": "train", "dataset": cfg.name})
        results = search(cfg, train, valid)
        best, best_total = choose(results, cfg), choose_for_totals(results)
        known = pd.concat([train, valid])
        typical = refit(cfg, best, known, cfg.train.typical_scaled)
        if best_total.name != best.name:
            expected = refit(cfg, best_total, known)
        elif typical.scaled or typical.objective != "log_l2":
            expected = typical
        else:  # one setting for both numbers: the same trees, read with the factor for the number that is added up
            expected = copy.copy(typical)
            expected.scaled = True
        model = CostPair(typical, expected)
        months = int(best["window_months"])
        refit_data = last_months(known, months)
        mlflow.log_params(typical.params() | {"window_months": months, "n_train": len(refit_data)})
        mlflow.log_params({f"expected_{k}": v for k, v in expected.params().items()})
        mlflow.log_param("expected_window_months", int(best_total["window_months"]))
        mlflow.log_metrics({"valid_mae": float(best[typical_mae(results, cfg)]), "valid_rmse": float(best["rmse"])})
        mlflow.log_metrics({f"valid_{k}": float(best_total[k]) for k in ("bias_pct", "monthly_total_mape")})
        version = registry.log_model(cfg, model, refit_data)
    registry.tag_version(
        cfg,
        version,
        {
            "kind": "candidate",
            "model": model.name,
            "typical_setting": setting_label(best),
            "typical_number": typical_number(typical),
            "expected_setting": setting_label(best_total),
            "window_months": str(months),
            "fitted_on": f"{refit_data['admission_date'].min():%Y-%m-%d} to {refit_data['admission_date'].max():%Y-%m-%d}",
            "valid_mae": f"{best[typical_mae(results, cfg)]:.2f}",
            "valid_monthly_total_mape": f"{best_total['monthly_total_mape']:.2f}",
            "train_run_id": parent.info.run_id,
        },
    )
    registry.set_alias(cfg, registry.STAGING, version)
    write_report(cfg, results, best, best_total, model, version)
    print(f"registered {cfg.model_name} version {version} as '{registry.STAGING}'")
    return version


def typical_number(model: CostModel) -> str:
    """How a fitted model gives its number, in words (for the registry tag and the report)."""
    if model.objective != "log_l2":
        return "the mean, as fitted"
    return "scaled by the smearing factor" if model.scaled else "the median: no smearing factor"


def gain_shares(model: CostModel) -> pd.Series:
    importance = model.feature_importance()
    return (100 * importance / importance.sum()).round(1).rename("share of gain (%)")


def write_report(cfg: Config, results: pd.DataFrame, best: pd.Series, best_total: pd.Series, model: CostPair,
                 version: str) -> None:  # fmt: skip
    out = cfg.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    records = json.loads(results.to_json(orient="records"))
    (out / "train_valid.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8", newline="\n")
    cols = [
        "window",
        "objective",
        "num_leaves",
        "use_time",
        "trees",
        "mae",
        "mae_unscaled",
        "rmse",
        "mae_top_decile",
        "bias_pct",
        "monthly_total_mape",
    ]
    ceiling = cfg.gate.max_monthly_total_mape
    key = typical_mae(results, cfg)
    same = model.expected is model.typical
    lines = [
        f"# Candidate search on the validation window ({cfg.name})",
        "",
        f"Generated by `costmodel train --config {cfg.path.as_posix() if cfg.path else ''}`. Each row was fitted on the"
        " training window (or its last N months) and measured on validation; trees chosen by early stopping.",
        "",
        to_markdown(results[cols].sort_values("mae"), index=False),
        "",
        "`mae_unscaled` is the MAE of the same fit without the smearing factor (only `log_l2` has one); every other"
        " column is of the number with the factor.",
        "",
        f"**Chosen for the typical cost:** {best['objective']}, {int(best['num_leaves'])} leaves, {best['window']},"
        f" {'with' if best['use_time'] else 'without'} the time feature, {int(best['trees'])} trees"
        + (
            f" (lowest {'`mae`' if key == 'mae' else '`mae_unscaled`'} among settings with monthly total error at most"
            f" {ceiling:.0f}%)."
            if ceiling is not None
            else "."
        )
        + f" The number is given as {typical_number(model.typical)}"
        + (f" (validation MAE {best[key]:.2f})." if key != "mae" else "."),
        "",
        "**Chosen for the expected cost:** "
        + (
            "the same setting (it also has the lowest error on the month's total)."
            if same
            else f"{best_total['objective']}, {int(best_total['num_leaves'])} leaves, {best_total['window']},"
            f" {'with' if best_total['use_time'] else 'without'} the time feature, {int(best_total['trees'])} trees"
            " (lowest error on the month's total)."
        ),
        "",
        f"Refitted on training + validation and registered together as `{cfg.model_name}` version {version}, alias"
        " `staging`.",
        "",
        "## What the refitted model relies on" + ("" if same else " (typical cost)"),
        "",
        gain_shares(model.typical).to_markdown(floatfmt=".1f"),
    ]
    if not same:
        expected_shares = gain_shares(model.expected).to_markdown(floatfmt=".1f")
        lines += ["", "## What the refitted model relies on (expected cost)", "", expected_shares]
    (out / "train_valid.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
