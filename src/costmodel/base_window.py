"""``costmodel base-window``: how much billed history should each retrain use?

Prices change every few months, so old admissions teach old prices; but a short history has few examples of each
procedure. This measures the trade-off the way the model would be used: the validation window is cut into blocks of
``HORIZON`` months, and before each block the model is fitted again with only what was already billed, then prices
that block. Every variant is retrained before every block, so the only thing that differs is the history it is given:

* only the last 4, 12 or 24 billed months, or every billed month;
* every billed month, with recent months weighing more (a month counts half as much every ``HALF_LIFE`` months back).

**What "already billed" means.** Costs are known ``label_delay_months`` after the admission month ends (the rule the
monitor uses), so before a block that starts in month *m* the last usable admission month is
*m* - ``label_delay_months`` - 1. Later months are still incomplete, and what is missing from them is the expensive
bills, which arrive last.

The number of trees cannot be chosen on the block being priced, so the last billed month is held out to choose it and
the model is then fitted again on the whole history. The setting is the one the search chose. Like the other side
comparisons, this reads the validation window only and registers nothing.
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
from costmodel.drift import monthly_metrics
from costmodel.evaluate import summary, to_markdown
from costmodel.train import choose, load_splits, setting_label

HORIZON = 3  # months priced by each retrain
BASES = (4, 12, 24, 0)  # billed months of history; 0 = all of them
HALF_LIFE = 12  # for the weighted variant: months back at which an admission counts half
KEEP = ("mae", "rmse", "mae_top_decile", "bias_pct", "monthly_total_mape")
WEIGHTED = "every billed month, recent months weigh more"


def base_label(months: int) -> str:
    return "every billed month" if months == 0 else f"last {months} billed months"


def blocks(months: list[int], horizon: int = HORIZON) -> list[list[int]]:
    """Consecutive runs of ``horizon`` months (the last one may be shorter)."""
    months = sorted(months)
    return [months[i : i + horizon] for i in range(0, len(months), horizon)]


def recency_weights(month: np.ndarray, known: int, half_life: int = HALF_LIFE) -> np.ndarray:
    """1 for the last billed month, 0.5 for ``half_life`` months earlier, 0.25 for twice that, and so on."""
    return 0.5 ** ((known - month) / half_life)


def fit_known(cfg: Config, setting: pd.Series, X: pd.DataFrame, y: pd.Series, month: np.ndarray, known: int,
              base: int, weighted: bool = False) -> tuple[CostModel, int]:  # fmt: skip
    """The setting fitted on admission months up to ``known`` (only the last ``base`` of them, if not 0)."""
    tc = cfg.train
    in_base = (month <= known) & ((month > known - base) if base else True)
    last = in_base & (month == known)  # held out to choose the number of trees, then used again
    weight = recency_weights(month, known) if weighted else None

    def model(trees: int) -> CostModel:
        return CostModel(setting["objective"], int(setting["num_leaves"]), tc.learning_rate, trees,
                         tc.min_category_count, tc.seed, bool(setting["use_time"]))  # fmt: skip

    rest = in_base & ~last
    probe = model(tc.max_rounds).fit(X[rest], y[rest], X[last], y[last], tc.early_stopping_rounds,
                                     sample_weight=None if weight is None else weight[rest])  # fmt: skip
    final = model(probe.best_iteration_)
    final.fit(X[in_base], y[in_base], sample_weight=None if weight is None else weight[in_base])
    return final, int(in_base.sum())


def run(cfg: Config) -> pd.DataFrame:
    searched = cfg.reports_dir / "train_valid.json"
    if not searched.exists():
        raise RuntimeError(f"{searched} not found; run `costmodel train` first (its chosen setting is the one retrained)")
    best = choose(pd.DataFrame(json.loads(searched.read_text(encoding="utf-8"))), cfg)
    gap = cfg.monitor.label_delay_months + 1

    df = load_splits(cfg)
    history = df[df["split"].isin(["train", "valid"])]  # bills of later windows are never read
    X, y = F.build_features(history), history[F.TARGET]
    month = X["admission_period"].to_numpy(dtype=int)
    is_valid = (history["split"] == "valid").to_numpy()
    y_valid, period = y[is_valid].to_numpy(), history.loc[is_valid, "period"]
    quarters = blocks(sorted(set(month[is_valid])))
    names = {m: p for m, p in zip(month[is_valid], period, strict=True)}
    block_of = {m: f"{names[q[0]]} to {names[q[-1]]}" for q in quarters for m in q}

    rows = [{"history": "the candidate as it is (fitted once, not retrained)", "retrains": 0, "mean_rows": None,
             "mean_trees": float(best["trees"]), **{k: float(best[k]) for k in KEEP}}]  # fmt: skip
    detail = {}
    registry.setup(cfg)
    with mlflow.start_run(run_name="base window"):
        mlflow.set_tags({"step": "base_window", "dataset": cfg.name})
        mlflow.log_params({"setting": setting_label(best), "horizon": HORIZON, "gap_months": gap, "half_life": HALF_LIFE})
        variants = [(base_label(b), b, False) for b in BASES] + [(WEIGHTED, 0, True)]
        for name, base, weighted in variants:
            t0, pred, sizes, trees = time.perf_counter(), np.full(len(history), np.nan), [], []
            for quarter in quarters:
                model, n = fit_known(cfg, best, X, y, month, quarter[0] - gap, base, weighted)
                priced = np.isin(month, quarter) & is_valid
                pred[priced] = model.predict(X[priced])
                sizes.append(n)
                trees.append(model.n_estimators)
            metrics = summary(y_valid, pred[is_valid], period)
            rows.append({"history": name, "retrains": len(quarters), "mean_rows": float(np.mean(sizes)),
                         "mean_trees": float(np.mean(trees)), **{k: metrics[k] for k in KEEP}})  # fmt: skip
            by_block = monthly_metrics(y_valid, pred[is_valid], pd.Series([block_of[m] for m in month[is_valid]]))
            detail[name] = [{"block": b["period"], **{k: b[k] for k in ("mae", "total_error_pct")}} for b in by_block]
            tag = "weighted" if weighted else str(base)
            mlflow.log_metrics({f"valid_{k}_{tag}": metrics[k] for k in ("mae", "monthly_total_mape")})
            print(f"{name:<46} MAE {metrics['mae']:9.2f}  monthly total err {metrics['monthly_total_mape']:5.2f}%"
                  f"  ({time.perf_counter() - t0:.0f}s)")  # fmt: skip
    results = pd.DataFrame(rows)
    write_report(cfg, results, detail, setting_label(best), gap)
    return results


def write_report(cfg: Config, results: pd.DataFrame, detail: dict[str, list[dict]], setting: str, gap: int) -> None:
    out = cfg.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads(results.to_json(orient="records"))
    payload = {"setting": setting, "horizon": HORIZON, "gap_months": gap, "half_life": HALF_LIFE}
    payload |= {"rows": rows, "blocks": detail}
    (out / "base_window_valid.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    retrained = results.iloc[1:]
    best = choose(retrained, cfg)
    ceiling = cfg.gate.max_monthly_total_mape
    table = results.rename(columns={"mean_rows": "admissions per fit", "mean_trees": "trees"})
    table["admissions per fit"] = table["admissions per fit"].map(lambda v: "-" if pd.isna(v) else f"{v:,.0f}")
    table["trees"] = table["trees"].map(lambda v: f"{v:.0f}")
    lines = [
        f"# How much billed history each retrain should use, on the validation window ({cfg.name})",
        "",
        f"Generated by `costmodel base-window --config {cfg.path.as_posix() if cfg.path else ''}`. The setting the search"
        f" chose ({setting}) is fitted again before every block of {HORIZON} months of the validation window, with only"
        f" the admission months already billed by then (up to {gap} months before the block starts), and prices that"
        " block. Every variant is retrained before every block; only the history it is given differs. In the weighted"
        f" variant a month counts half as much every {HALF_LIFE} months back. The first row is the search's own number,"
        " for reference: one model fitted on the whole training window and never retrained.",
        "",
        to_markdown(table, index=False),
        "",
        "**Best of the retrained variants by the search's own rule** (lowest MAE"
        + (f" among rows with monthly total error at most {ceiling:.0f}%" if ceiling is not None else "")
        + f"): {best['history']}.",
        "",
        "## Block by block",
        "",
        "MAE per admission, and the error on the block's total (negative = under-predicted).",
        "",
    ]
    names = list(detail)
    block_names = [b["block"] for b in detail[names[0]]] if names else []
    for metric, title in (("mae", "MAE"), ("total_error_pct", "Error on the block's total (%)")):
        frame = pd.DataFrame({n: [b[metric] for b in detail[n]] for n in names}, index=block_names).T
        frame = frame.rename_axis("history").reset_index()
        lines += [f"**{title}**", "", frame.to_markdown(index=False, floatfmt=".2f"), ""]
    lines += [
        "Admission months count as billed by the delay rule above; the amounts are the final ones in the table, as in"
        " the retraining replay. Validation window only: the test window was not read and nothing was registered.",
    ]
    (out / "base_window_valid.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
