"""The one look at the test window: score the ``staging`` model, run the gate, and promote it if it passes.

* The candidate is loaded **from the registry** (not from memory), so what is measured is exactly what would be served.
  It gives two numbers per admission, each judged by the metric it is for: the typical cost by the error per
  admission, the expected cost by the error on the month's total (``evaluate.pair_summary``).
* Every baseline is refitted on the same data the candidate was (training + validation, same window), then scored on
  the test window. The gate's baseline reference is the best of them *on the test window*: the strictest comparison.
* If a model is already in ``production``, it is scored too and is a second reference: a new version must beat the
  one in use, not only the baselines.
* Passing moves the ``production`` alias to the candidate. Failing leaves ``production`` where it was; if nothing was in
  production yet, the best baseline is registered and put there instead, so there is always a model to serve (and the
  next candidate has to beat it). Either way the decision and its reasons are written to the model version's tags, an
  MLflow run and ``reports/<dataset>/test.md``.

The test window should drive decisions only here. Re-running this command regenerates the report; it is not a licence
to tune against the test year.
"""

from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd

from costmodel import features as F
from costmodel import registry
from costmodel.baselines import age_band, all_baselines
from costmodel.candidate import EXPECTED, TYPICAL
from costmodel.config import Config
from costmodel.evaluate import decile_table, pair_summary, segment_mae, summary, to_markdown
from costmodel.gate import should_promote
from costmodel.train import last_months, load_splits


def run(cfg: Config) -> bool:
    """Returns whether the candidate was promoted."""
    registry.setup(cfg)
    staging = registry.version_for(cfg, registry.STAGING)
    if staging is None:
        raise RuntimeError(f"no '{registry.STAGING}' model for {cfg.model_name}; run `costmodel train` first")
    production = registry.version_for(cfg, registry.PRODUCTION)
    if production is not None and str(production.version) == str(staging.version):
        print(f"version {staging.version} is already in production; train a new candidate first")
        return False

    df = load_splits(cfg)
    test = df[df["split"] == "test"]
    if test.empty:
        raise RuntimeError("the test window is empty; check [split] in the config")
    months = int(staging.tags.get("window_months", 0))
    fit_data = last_months(df[df["split"].isin(["train", "valid"])], months)
    y = test[F.TARGET].to_numpy()

    preds: dict[str, np.ndarray] = {}  # one number per admission and model: the typical cost
    fitted = {}
    metrics = {}
    preds["candidate"], expected = registry.predict_both(registry.load(cfg, registry.STAGING), test)
    metrics["candidate"] = pair_summary(y, preds["candidate"], expected, test["period"])
    numbers = {TYPICAL: summary(y, preds["candidate"], test["period"]), EXPECTED: summary(y, expected, test["period"])}
    if production is not None:
        preds["production"], in_use_expected = registry.predict_both(registry.load(cfg, registry.PRODUCTION), test)
        metrics["production"] = pair_summary(y, preds["production"], in_use_expected, test["period"])
    X_fit, X_test = F.build_features(fit_data), F.build_features(test)
    for model in all_baselines():
        fitted[model.name] = model.fit(X_fit, fit_data[F.TARGET])
        preds[model.name] = model.predict(X_test)
        metrics[model.name] = summary(y, preds[model.name], test["period"])
        print(f"scored baseline {model.name}")

    baseline_names = [m.name for m in all_baselines()]
    best_baseline = min(baseline_names, key=lambda n: metrics[n]["mae"])
    baselines = {f"best baseline ({best_baseline})": metrics[best_baseline]}
    in_use = (f"production (version {production.version})", metrics["production"]) if production is not None else None
    promote, reasons = should_promote(metrics["candidate"], baselines, cfg.gate, in_use)

    with mlflow.start_run(run_name="evaluate on test"):
        mlflow.set_tags({"step": "evaluate", "dataset": cfg.name, "candidate_version": str(staging.version)})
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics["candidate"].items()})
        mlflow.log_metrics({f"test_{best_baseline}_{k}": v for k, v in metrics[best_baseline].items()})
        mlflow.set_tag("gate", "passed" if promote else "failed")
        mlflow.log_text("\n".join(reasons), "gate_reasons.txt")
    registry.tag_version(
        cfg,
        staging.version,
        {
            "gate": "passed" if promote else "failed",
            "test_mae": f"{metrics['candidate']['mae']:.2f}",
            "test_monthly_total_mape": f"{metrics['candidate']['monthly_total_mape']:.2f}",
        },
    )
    if promote:
        registry.set_alias(cfg, registry.PRODUCTION, staging.version)
        outcome = f"PROMOTED: {cfg.model_name} version {staging.version} is now '{registry.PRODUCTION}'"
    elif production is not None:
        outcome = f"NOT PROMOTED: version {production.version} stays in production"
    else:
        version = register_baseline(cfg, fitted[best_baseline], fit_data, metrics[best_baseline])
        outcome = (
            f"NOT PROMOTED: nothing was in production, so the best baseline ({best_baseline}) was registered as"
            f" version {version} and put in production"
        )

    write_report(cfg, test, preds, expected, metrics, numbers, best_baseline, staging.version, promote, reasons, outcome)
    print("\n".join(reasons))
    print(outcome)
    return promote


def register_baseline(cfg: Config, model, fit_data: pd.DataFrame, test_metrics: dict) -> str:
    """Register a fitted baseline as a model version and put it in production (used when no candidate ever passed)."""
    with mlflow.start_run(run_name=f"baseline {model.name} to production"):
        mlflow.set_tags({"step": "evaluate", "dataset": cfg.name, "kind": "baseline"})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        version = registry.log_model(cfg, model, fit_data)
    registry.tag_version(
        cfg,
        version,
        {"kind": "baseline", "model": model.name, "gate": "fallback", "test_mae": f"{test_metrics['mae']:.2f}"},
    )
    registry.set_alias(cfg, registry.PRODUCTION, version)
    return version


def segments(test: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "age band": age_band(test["age_years"].astype(float)),
        "sex": test["sex"].astype("string"),
        "admission type": test["admission_type"].astype("string"),
    }


def write_report(cfg, test, preds, expected, metrics, numbers, best_baseline, version, promote, reasons,
                 outcome) -> None:  # fmt: skip
    out = cfg.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([{"model": name, **m} for name, m in metrics.items()])
    two_numbers = expected is not preds["candidate"]
    payload = {"candidate_version": version, "promoted": promote, "reasons": reasons, "metrics": metrics}
    if two_numbers:
        payload["candidate_numbers"] = numbers  # each of the candidate's two numbers, on every metric
    (out / "test.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    cols = ["model", "mae", "rmse", "median_ae", "mae_top_decile", "bias_pct", "monthly_total_mape"]
    y = test[F.TARGET].to_numpy()
    first, last = test["admission_date"].min(), test["admission_date"].max()
    lines = [
        f"# Test window ({cfg.name})",
        "",
        f"Generated by `costmodel evaluate --config {cfg.path.as_posix() if cfg.path else ''}`. Candidate = registry"
        f" version {version} (`{cfg.model_name}`). All models fitted on the same data, then scored on"
        f" {len(test):,} admissions from {first:%Y-%m-%d} to {last:%Y-%m-%d}.",
        "",
        to_markdown(table[cols].sort_values("mae"), index=False),
        "",
    ]
    if two_numbers:
        both = pd.DataFrame([{"the candidate's": name.replace("_", " "), **m} for name, m in numbers.items()])
        lines += [
            "The candidate gives two numbers per admission. Its row above takes each metric from the number meant for"
            " it: `mae`, `rmse`, `median_ae` and `mae_top_decile` from the typical cost, `bias_pct` and"
            " `monthly_total_mape` from the expected cost. Each number on every metric:",
            "",
            to_markdown(both[["the candidate's", *cols[1:]]], index=False),
            "",
        ]
    lines += [
        f"## Gate: {'PASSED, promoted to production' if promote else 'FAILED, not promoted'}",
        "",
        *[f"- {r}" for r in reasons],
        "",
        f"Outcome: {outcome.split(': ', 1)[1]}.",
        "",
        "## Error by segment (MAE" + (" of the typical cost)" if two_numbers else ")"),
        "",
    ]
    for name, seg in segments(test).items():
        cand = segment_mae(y, preds["candidate"], seg)
        base = segment_mae(y, preds[best_baseline], seg)
        seg_table = pd.DataFrame(
            {"n": cand["n"], "mean actual": cand["mean_actual"], "candidate": cand["mae"], best_baseline: base["mae"]}
        ).rename_axis(name)
        lines += [f"### {name}", "", to_markdown(seg_table), ""]
    lines += [
        "## Calibration by decile of actual cost (candidate" + (", typical cost)" if two_numbers else ")"),
        "",
        to_markdown(decile_table(y, preds["candidate"])),
    ]
    if two_numbers:
        title = "## Calibration by decile of actual cost (candidate, expected cost)"
        lines += ["", title, "", to_markdown(decile_table(y, expected))]
    (out / "test.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
