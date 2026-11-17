"""Train the candidate: choose its settings on validation, refit, and register it as ``staging``.

1. **Search.** Every combination of training window x objective x tree size x time feature in ``[train]`` is fitted on
   the training window and measured on the validation window, each as an MLflow run. The test window is not read here.
2. **Choose.** Lowest validation MAE among the settings whose monthly-total error is within the gate's ceiling (the
   gate would reject the others anyway); if none is, lowest MAE overall, and the gate will say no.
3. **Refit** the chosen settings on training + validation (the most recent data the model can have before the test
   year), with the number of trees found by early stopping, and register it in MLflow with alias ``staging``.

Whether it goes to ``production`` is decided later, by ``costmodel evaluate`` (see ``holdout.py``).
"""

from __future__ import annotations

import itertools
import json
import time

import mlflow
import pandas as pd

from costmodel import features as F
from costmodel import registry
from costmodel.candidate import CostModel
from costmodel.config import Config
from costmodel.evaluate import summary, to_markdown
from costmodel.sources import load_table
from costmodel.split import assign

EXPANDING = 0  # "sliding months" value meaning: use every month available


def window_label(months: int) -> str:
    return "expanding" if months == EXPANDING else f"sliding {months} months"


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
                metrics = summary(y_valid.to_numpy(), model.predict(X_valid), valid["period"])
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


def choose(results: pd.DataFrame, cfg: Config) -> pd.Series:
    ceiling = cfg.gate.max_monthly_total_mape
    eligible = results if ceiling is None else results[results["monthly_total_mape"] <= ceiling]
    return (eligible if len(eligible) else results).sort_values("mae").iloc[0]


def run(cfg: Config) -> str:
    """Returns the registered version number."""
    df = load_splits(cfg)
    train, valid = df[df["split"] == "train"], df[df["split"] == "valid"]
    registry.setup(cfg)
    with mlflow.start_run(run_name="train") as parent:
        mlflow.set_tags({"step": "train", "dataset": cfg.name})
        results = search(cfg, train, valid)
        best = choose(results, cfg)
        tc = cfg.train
        months = int(best["window_months"])
        refit_data = last_months(pd.concat([train, valid]), months)
        model = CostModel(
            best["objective"],
            int(best["num_leaves"]),
            tc.learning_rate,
            int(best["trees"]),
            tc.min_category_count,
            tc.seed,
            bool(best["use_time"]),
        )
        print(f"refitting {model.name} ({window_label(months)}) on {len(refit_data):,} admissions (train + valid)")
        model.fit(F.build_features(refit_data), refit_data[F.TARGET])
        mlflow.log_params(model.params() | {"window_months": months, "n_train": len(refit_data)})
        mlflow.log_metrics({f"valid_{k}": float(best[k]) for k in ("mae", "rmse", "bias_pct", "monthly_total_mape")})
        version = registry.log_model(cfg, model, refit_data)
    registry.tag_version(
        cfg,
        version,
        {
            "kind": "candidate",
            "model": model.name,
            "window_months": str(months),
            "fitted_on": f"{refit_data['admission_date'].min():%Y-%m-%d} to {refit_data['admission_date'].max():%Y-%m-%d}",
            "valid_mae": f"{best['mae']:.2f}",
            "valid_monthly_total_mape": f"{best['monthly_total_mape']:.2f}",
            "train_run_id": parent.info.run_id,
        },
    )
    registry.set_alias(cfg, registry.STAGING, version)
    write_report(cfg, results, best, model, version)
    print(f"registered {cfg.model_name} version {version} as '{registry.STAGING}'")
    return version


def write_report(cfg: Config, results: pd.DataFrame, best: pd.Series, model: CostModel, version: str) -> None:
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
        "rmse",
        "mae_top_decile",
        "bias_pct",
        "monthly_total_mape",
    ]
    ceiling = cfg.gate.max_monthly_total_mape
    importance = model.feature_importance()
    importance = (100 * importance / importance.sum()).round(1).rename("share of gain (%)")
    lines = [
        f"# Candidate search on the validation window ({cfg.name})",
        "",
        f"Generated by `costmodel train --config {cfg.path.as_posix() if cfg.path else ''}`. Each row was fitted on the"
        " training window (or its last N months) and measured on validation; trees chosen by early stopping.",
        "",
        to_markdown(results[cols].sort_values("mae"), index=False),
        "",
        f"**Chosen:** {best['objective']}, {int(best['num_leaves'])} leaves, {best['window']},"
        f" {'with' if best['use_time'] else 'without'} the time feature, {int(best['trees'])} trees"
        + (f" (lowest MAE among settings with monthly total error at most {ceiling:.0f}%)." if ceiling is not None else "."),
        f"Refitted on training + validation and registered as `{cfg.model_name}` version {version}, alias `staging`.",
        "",
        "## What the refitted model relies on",
        "",
        importance.to_markdown(floatfmt=".1f"),
    ]
    (out / "train_valid.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
