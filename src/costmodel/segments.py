"""``costmodel segments``: one model for every admission, or one model per group of admissions?

The candidate is a single model for all admissions. This fits the setting the search chose (``costmodel train``) once
per group instead, with only that group's admissions, and puts the predictions back together so the whole validation
window is measured with the same two metrics. Two ways of grouping are tried: by admission type (elective, urgent)
and by procedure group (the first two digits of the procedure code).

A group gets its own model only with at least ``MIN_SEGMENT_ROWS`` training admissions; smaller groups keep the single
model's predictions. Like ``costmodel families`` and ``costmodel blend``, this reads the validation window only and
registers nothing.
"""

from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd

from costmodel import features as F
from costmodel import registry
from costmodel.blend import fit_setting
from costmodel.config import Config
from costmodel.evaluate import summary, to_markdown
from costmodel.train import choose, last_months, load_splits, setting_label

SCHEMES = {"admission_type": "one model per admission type", "procedure_prefix2": "one model per procedure group"}
SINGLE = "one model for every admission"
MIN_SEGMENT_ROWS = 20_000  # training admissions a group needs to get its own model
KEEP = ("mae", "rmse", "mae_top_decile", "bias_pct", "monthly_total_mape")


def total_error_pct(y: np.ndarray, pred: np.ndarray) -> float:
    return float(100 * (pred.sum() - y.sum()) / y.sum())


def by_group(cfg: Config, setting: pd.Series, train: pd.DataFrame, valid: pd.DataFrame, column: str,
             single: np.ndarray) -> tuple[np.ndarray, list[dict]]:  # fmt: skip
    """Predictions for ``valid`` from one model per value of ``column``, and what happened in each group."""
    window = last_months(train, int(setting["window_months"]))
    train_group = F.build_features(window)[column].to_numpy()
    X_valid, y_valid = F.build_features(valid), valid[F.TARGET]
    valid_group = X_valid[column].to_numpy()
    pred, groups = single.copy(), []
    for value in sorted(set(valid_group)):
        in_train, in_valid = train_group == value, valid_group == value
        row = {"group": value, "n_train": int(in_train.sum()), "n_valid": int(in_valid.sum()), "own_model": False}
        y_group = y_valid[in_valid].to_numpy()
        if row["n_train"] >= MIN_SEGMENT_ROWS:
            model = fit_setting(cfg, setting, window[in_train], X_valid[in_valid], y_valid[in_valid])
            pred[in_valid] = model.predict(X_valid[in_valid])
            row |= {"own_model": True, "trees": model.best_iteration_}
        row |= {
            "mae_single": float(np.abs(single[in_valid] - y_group).mean()),
            "mae_by_group": float(np.abs(pred[in_valid] - y_group).mean()),
            "total_error_pct_single": total_error_pct(y_group, single[in_valid]),
            "total_error_pct_by_group": total_error_pct(y_group, pred[in_valid]),
        }
        groups.append(row)
    return pred, groups


def run(cfg: Config) -> pd.DataFrame:
    searched = cfg.reports_dir / "train_valid.json"
    if not searched.exists():
        raise RuntimeError(f"{searched} not found; run `costmodel train` first (each group uses its chosen setting)")
    setting = choose(pd.DataFrame(json.loads(searched.read_text(encoding="utf-8"))), cfg)

    df = load_splits(cfg)
    train, valid = df[df["split"] == "train"], df[df["split"] == "valid"]
    X_valid, y_valid = F.build_features(valid), valid[F.TARGET]
    registry.setup(cfg)
    rows, detail = [], {}
    with mlflow.start_run(run_name="segments"):
        mlflow.set_tags({"step": "segments", "dataset": cfg.name})
        mlflow.log_params({"setting": setting_label(setting), "min_segment_rows": MIN_SEGMENT_ROWS})
        single = fit_setting(cfg, setting, train, X_valid, y_valid).predict(X_valid)
        schemes = [(SINGLE, None, single, 1)]
        for column, name in SCHEMES.items():
            pred, groups = by_group(cfg, setting, train, valid, column, single)
            detail[column] = groups
            schemes.append((name, column, pred, sum(g["own_model"] for g in groups)))
        for name, column, pred, models in schemes:
            metrics = summary(y_valid.to_numpy(), pred, valid["period"])
            rows.append({"scheme": name, "column": column, "models": models, **{k: metrics[k] for k in KEEP}})
            mlflow.log_metrics({f"valid_{k}_{column or 'single'}": metrics[k] for k in ("mae", "monthly_total_mape")})
            print(f"{name:<32} models {models:<3} MAE {metrics['mae']:9.2f}"
                  f"  monthly total err {metrics['monthly_total_mape']:5.2f}%")  # fmt: skip
    results = pd.DataFrame(rows)
    write_report(cfg, results, detail, setting_label(setting))
    return results


def write_report(cfg: Config, results: pd.DataFrame, detail: dict[str, list[dict]], setting: str) -> None:
    out = cfg.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "setting": setting,
        "min_segment_rows": MIN_SEGMENT_ROWS,
        "schemes": json.loads(results.to_json(orient="records")),
        "groups": detail,
    }
    (out / "segments_valid.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    best = choose(results, cfg)
    ceiling = cfg.gate.max_monthly_total_mape
    lines = [
        f"# One model or one per group, on the validation window ({cfg.name})",
        "",
        f"Generated by `costmodel segments --config {cfg.path.as_posix() if cfg.path else ''}`. The setting the search"
        f" chose ({setting}) fitted once on every training admission, and once per group with only that group's"
        f" admissions. A group gets its own model with at least {MIN_SEGMENT_ROWS:,} training admissions; smaller groups"
        " keep the single model. Every row is measured on the whole validation window.",
        "",
        to_markdown(results.drop(columns="column"), index=False),
        "",
        "**Best by the search's own rule** (lowest MAE"
        + (f" among rows with monthly total error at most {ceiling:.0f}%" if ceiling is not None else "")
        + f"): {best['scheme']}.",
    ]
    for column, groups in detail.items():
        table = pd.DataFrame(groups).drop(columns=["own_model", "trees"], errors="ignore").rename(columns={"group": column})
        table.insert(3, "trees", [str(g["trees"]) if g["own_model"] else "-" for g in groups])
        lines += [
            "",
            f"## {SCHEMES[column].capitalize()}",
            "",
            "`single` is the single model on that group's admissions, `by_group` the group's own model (the same number"
            " when the group is too small to have one, shown with `-` trees). The total error is over the whole window.",
            "",
            to_markdown(table, index=False),
        ]
    lines += ["", "Validation window only: the test window was not read and nothing was registered."]
    (out / "segments_valid.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
