"""``costmodel recent-price``: does the model do better when told what each procedure has been costing lately?

The candidate learns prices from its training data and keeps them until it is retrained, so it under-predicts when
prices rise. Here every admission gets one more input, built from bills that were already known when the patient was
admitted: the **recent price**, the mean paid for the same procedure over the last few billed months.

**What "already known" means.** Costs are known ``label_delay_months`` after the admission month ends (the same rule
the monitor uses). So an admission in month *m* may only use admissions from month *m* - ``label_delay_months`` - 1
or earlier. Nothing from its own month, or from the months still being billed, reaches it.

Two ways of using the recent price are measured against the candidate as it is, each with a short and a long window:

* **as one more input**: the trees decide how much to rely on it;
* **as the unit**: the model predicts the cost *relative* to the recent price, so a price that doubles moves the
  prediction with it once the bills arrive.

A procedure with fewer than ``MIN_COUNT`` billed admissions in the window takes the price of its subgroup, then of its
group, then of all admissions. The model itself is not retrained along the validation window; only this input moves.
Like the other side comparisons, this reads the validation window only and registers nothing.
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
from costmodel.train import choose, last_months, load_splits, setting_label

COLUMN = "recent_price"
WINDOWS = (3, 12)  # billed months averaged
MIN_COUNT = 20  # billed admissions a code needs in the window to have a price of its own
LEVELS = ("requested_procedure", "procedure_prefix4", "procedure_prefix2")  # then: every admission
KEEP = ("mae", "rmse", "mae_top_decile", "bias_pct", "monthly_total_mape")
AS_INPUT, AS_UNIT = "recent price as one more input", "cost predicted relative to the recent price"


def recent_mean(paid, month, code, window: int, gap: int, min_count: int = MIN_COUNT) -> np.ndarray:
    """For each admission, the mean paid for its ``code`` in the ``window`` months that end ``gap`` months before its
    own month (``month`` is a month number, e.g. year * 12 + month). NaN with fewer than ``min_count`` admissions."""
    df = pd.DataFrame({"paid": np.asarray(paid, dtype=float), "month": np.asarray(month, dtype=int)})
    df["code"] = np.asarray(code)
    months = range(df["month"].min(), df["month"].max() + 1)
    cells = df.groupby(["code", "month"])["paid"].agg(["sum", "count"])
    total = cells["sum"].unstack(fill_value=0.0).reindex(columns=months, fill_value=0.0)
    count = cells["count"].unstack(fill_value=0).reindex(columns=months, fill_value=0)
    # Column t now holds months t-window+1 .. t; an admission in month m reads column m-gap.
    total, count = (t.T.rolling(window, min_periods=1).sum().T for t in (total, count))
    mean = (total / count.where(count >= min_count)).rename(columns=lambda t: t + gap)
    lookup = mean.stack().rename(COLUMN)  # stack drops the missing ones
    keys = pd.MultiIndex.from_frame(df[["code", "month"]])
    return lookup.reindex(keys).to_numpy(dtype=float)


def reference_price(X: pd.DataFrame, paid, window: int, gap: int,
                    min_count: int = MIN_COUNT) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """The recent price of each admission's procedure, falling back to its subgroup, its group and every admission.

    Returns the price and the level it came from (0 = the procedure itself, ..., 3 = every admission, -1 = none: the
    first months of the data have no billed history at all).
    """
    month = X["admission_period"].to_numpy(dtype=int)
    price, level = np.full(len(X), np.nan), np.full(len(X), -1)
    codes = [X[col].to_numpy() for col in LEVELS] + [np.zeros(len(X), dtype=int)]
    for i, code in enumerate(codes):
        missing = np.isnan(price)
        found = recent_mean(paid, month, code, window, gap, min_count)
        take = missing & ~np.isnan(found)
        price[take], level[take] = found[take], i
    return price, level


class PricedModel(CostModel):
    """The candidate with the recent price as one more input and, with ``relative``, as the unit of its target."""

    def __init__(self, *args, relative: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.relative = relative

    @property
    def columns(self) -> list[str]:
        return [*super().columns, COLUMN]

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        return super()._encode(X).assign(**{COLUMN: X[COLUMN].astype(float)})

    def fit(self, X, y, X_valid=None, y_valid=None, early_stopping_rounds=None) -> PricedModel:
        if not self.relative:
            return super().fit(X, y, X_valid, y_valid, early_stopping_rounds)
        priced = X[COLUMN].notna().to_numpy()  # the first months of the data have no billed history: left out
        X, y = X[priced], np.asarray(y, dtype=float)[priced] / X.loc[priced, COLUMN].to_numpy()
        if X_valid is not None:
            y_valid = np.asarray(y_valid, dtype=float) / X_valid[COLUMN].to_numpy()
        return super().fit(X, y, X_valid, y_valid, early_stopping_rounds)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = super().predict(X)
        return pred * X[COLUMN].to_numpy() if self.relative else pred


def run(cfg: Config) -> pd.DataFrame:
    searched = cfg.reports_dir / "train_valid.json"
    if not searched.exists():
        raise RuntimeError(f"{searched} not found; run `costmodel train` first (its chosen setting is the comparison)")
    search = pd.DataFrame(json.loads(searched.read_text(encoding="utf-8")))
    best = choose(search, cfg)
    tc, gap = cfg.train, cfg.monitor.label_delay_months + 1
    rows = [{"variant": "the candidate as it is", "window": "-", "trees": int(best["trees"]),
             **{k: float(best[k]) for k in KEEP}, "gain_share_pct": None, "own_price_pct": None}]  # fmt: skip

    df = load_splits(cfg)
    history = df[df["split"].isin(["train", "valid"])]  # bills of later windows are never read
    X_all, paid = F.build_features(history), history[F.TARGET]
    registry.setup(cfg)
    with mlflow.start_run(run_name="recent price"):
        mlflow.set_tags({"step": "recent_price", "dataset": cfg.name})
        mlflow.log_params({"setting": setting_label(best), "gap_months": gap, "min_count": MIN_COUNT})
        for window in WINDOWS:
            price, level = reference_price(X_all, paid, window, gap)
            X_all[COLUMN] = price
            in_train = (history["split"] == "train").to_numpy()
            train = last_months(history[in_train], int(best["window_months"]))
            X, y = X_all.loc[train.index], train[F.TARGET]
            X_valid, y_valid, valid = X_all[~in_train], paid[~in_train], history[~in_train]
            own = float(100 * (level[~in_train] == 0).mean())
            for variant in (AS_INPUT, AS_UNIT):
                t0 = time.perf_counter()
                model = PricedModel(best["objective"], int(best["num_leaves"]), tc.learning_rate, tc.max_rounds,
                                    tc.min_category_count, tc.seed, bool(best["use_time"]),
                                    relative=variant == AS_UNIT)  # fmt: skip
                model.fit(X, y, X_valid, y_valid, tc.early_stopping_rounds)
                metrics = summary(y_valid.to_numpy(), model.predict(X_valid), valid["period"])
                gain = model.feature_importance()
                rows.append({"variant": variant, "window": f"last {window} billed months", "trees": model.best_iteration_,
                             **{k: metrics[k] for k in KEEP}, "gain_share_pct": float(100 * gain[COLUMN] / gain.sum()),
                             "own_price_pct": own})  # fmt: skip
                seconds = time.perf_counter() - t0
                print(f"{variant:<44} {window:>2} months  trees {model.best_iteration_:<5} MAE {metrics['mae']:9.2f}"
                      f"  monthly total err {metrics['monthly_total_mape']:5.2f}%  ({seconds:.0f}s)")  # fmt: skip
    results = pd.DataFrame(rows)
    write_report(cfg, results, setting_label(best), gap)
    return results


def write_report(cfg: Config, results: pd.DataFrame, setting: str, gap: int) -> None:
    out = cfg.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads(results.to_json(orient="records"))
    payload = {"setting": setting, "gap_months": gap, "min_count": MIN_COUNT, "rows": rows}
    (out / "recent_price_valid.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    best, alone = choose(results, cfg), results.iloc[0]
    ceiling = cfg.gate.max_monthly_total_mape
    table = results.rename(columns={"gain_share_pct": "recent price, share of gain (%)",
                                    "own_price_pct": "priced by its own procedure (%)"})  # fmt: skip
    for col in table.columns[-2:]:
        table[col] = table[col].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")
    verdict = (
        "No variant with the recent price beats it under that rule."
        if best.name == alone.name
        else f"Against the candidate as it is, MAE goes from {alone['mae']:.2f} to {best['mae']:.2f} and the monthly total"
        f" error from {alone['monthly_total_mape']:.2f}% to {best['monthly_total_mape']:.2f}%."
    )
    lines = [
        f"# The recent price of each procedure as an input, on the validation window ({cfg.name})",
        "",
        f"Generated by `costmodel recent-price --config {cfg.path.as_posix() if cfg.path else ''}`. The setting the search"
        f" chose ({setting}), with one more input: the mean paid for the same procedure over the last billed months. An"
        f" admission only uses bills from {gap} months before its own month or earlier (costs are known"
        f" {gap - 1} months after the admission month ends). A procedure with fewer than {MIN_COUNT} billed admissions in"
        " the window takes the price of its subgroup, its group, or every admission. The model is fitted once on the"
        " training window and not retrained; only the recent price moves along the validation window.",
        "",
        to_markdown(table, index=False),
        "",
        "**Best by the search's own rule** (lowest MAE"
        + (f" among rows with monthly total error at most {ceiling:.0f}%" if ceiling is not None else "")
        + f"): {best['variant']}" + ("" if best["window"] == "-" else f", {best['window']}") + f". {verdict}",
        "",
        "Validation window only: the test window was not read and nothing was registered. Serving a model like this"
        " needs the table of recent prices at prediction time, refreshed every month.",
    ]
    (out / "recent_price_valid.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
