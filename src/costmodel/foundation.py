"""``costmodel foundation``: does a tabular foundation model do better than the candidate?

A tabular foundation model is trained once, by its authors, on millions of made-up tables. It is not fitted to this
data: it is handed labelled admissions as *context* and prices the others in one pass. Here it is **TabPFN-2**, the
checkpoint of TabPFN with open weights (the newer ones need an account and a non-commercial licence).

**The catch is scale.** TabPFN-2 reads at most ``CONTEXT_ROWS`` rows of context; the training window has millions,
with thousands of procedure, diagnosis and hospital codes. So three things are measured on the same validation
admissions:

* **LightGBM on every training row**: the two settings in production (typical cost and expected cost);
* **LightGBM on the context rows only**: the same two settings, with the same small budget of rows;
* **TabPFN-2 on the context rows**, with the context drawn from the whole training window or from its last
  ``RECENT_MONTHS`` months, predicting the cost or log(cost), and once more with **code statistics**: for the
  procedure, the diagnosis and the hospital, the mean log(cost) of that code over every training row outside the
  context. That is one way of letting a model that reads few rows use what the others say. LightGBM gets the same
  statistics on the same rows, to tell what the statistics do from what the model does.

Each row is also compared with the first on the same admissions, with a 95% interval obtained by drawing the scored
admissions again with replacement (``DRAWS`` times): a difference whose interval contains zero is not one this sample
can show.

TabPFN predicts a whole distribution for each admission, so it gives both numbers by itself: the median is the typical
cost and the mean is the expected cost.

Every call recomputes the context, so the validation window is not scored whole: ``EVAL_ROWS`` admissions are drawn
from it at random and every row of the table is scored on those same admissions. Totals of a sample are noisier than
totals of the window, so compare rows with each other, not with the other reports.

Optional: ``uv sync --extra foundation`` (it brings PyTorch; a GPU is needed at this size). Like the other side
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
from costmodel.blend import fit_setting, pair
from costmodel.candidate import CostModel
from costmodel.config import Config
from costmodel.evaluate import pair_summary, to_markdown
from costmodel.train import last_months, load_splits, setting_label

CHECKPOINT = "v2"  # the TabPFN checkpoint with open weights
CONTEXT_ROWS = 10_000  # what that checkpoint was built to read
EVAL_ROWS = 30_000  # validation admissions scored, drawn at random
CHUNK_ROWS = 5_000  # admissions priced per call; bounds the GPU memory
RECENT_MONTHS = 12
MAX_CATEGORIES = 30  # a column with more codes than this is read as a number (its position in sorted order)
STAT_COLUMNS = ("requested_procedure", "diagnosis_icd10", "provider_id")
DRAWS = 1_000  # resamples of the scored admissions behind each interval
QUANTILES = tuple(float(q) for q in np.round(np.arange(0.005, 1, 0.01), 3))  # 100 levels, to average exp(log cost)
KEEP = ("mae", "rmse", "mae_top_decile", "bias_pct", "monthly_total_mape")
GAPS = ("mae_gap", "mae_gap_low", "mae_gap_high", "total_gap", "total_gap_low", "total_gap_high")


def tabpfn_available() -> bool:
    try:
        import tabpfn  # noqa: F401
    except ImportError:
        return False
    return True


def sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """``n`` rows drawn at random without replacement, in their original order (all of them if there are fewer)."""
    if len(df) <= n:
        return df
    return df.iloc[np.sort(np.random.default_rng(seed).choice(len(df), n, replace=False))]


class CodeOrder:
    """Codes as numbers: each categorical column becomes the position of the code in sorted order, so neighbours in
    the code tables (procedures, ICD-10) stay neighbours. A code the context does not have falls halfway between
    its two neighbours."""

    def fit(self, X: pd.DataFrame) -> CodeOrder:
        self.codes_ = {c: np.unique(X[c].to_numpy(dtype=str)) for c in X.columns if c in F.CATEGORICAL}
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        out = X.copy()
        for col, codes in self.codes_.items():
            values = X[col].to_numpy(dtype=str)
            position = np.searchsorted(codes, values)
            known = codes[np.minimum(position, len(codes) - 1)] == values
            out[col] = np.where(known, position, position - 0.5)
        return out.to_numpy(dtype=float)


def code_statistics(X: pd.DataFrame, y, min_count: int) -> dict[str, tuple[pd.Series, float]]:
    """Per column of ``STAT_COLUMNS``: the mean log(cost) of each code, and the mean of all rows for codes with fewer
    than ``min_count`` rows (or never seen)."""
    log_cost = pd.Series(np.log(np.asarray(y, dtype=float)), index=X.index)
    overall = float(log_cost.mean())
    out = {}
    for col in STAT_COLUMNS:
        by_code = log_cost.groupby(X[col].astype(str)).agg(["mean", "count"])
        out[col] = (by_code.loc[by_code["count"] >= min_count, "mean"], overall)
    return out


def with_statistics(X: pd.DataFrame, statistics: dict[str, tuple[pd.Series, float]]) -> pd.DataFrame:
    out = X.copy()
    for col, (means, overall) in statistics.items():
        out[f"{col}_mean_log_cost"] = X[col].astype(str).map(means).fillna(overall).to_numpy(dtype=float)
    return out


def statistic_columns() -> list[str]:
    return [f"{col}_mean_log_cost" for col in STAT_COLUMNS]


class StatisticsModel(CostModel):
    """The candidate with the code statistics as extra inputs."""

    @property
    def columns(self) -> list[str]:
        return [*super().columns, *statistic_columns()]

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        return super()._encode(X).assign(**{c: X[c].astype(float) for c in statistic_columns()})


def measures(y: np.ndarray, month: np.ndarray, typical: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    """MAE of the typical cost, and mean absolute error (%) of the expected cost on each month's total. ``month`` is
    a month number starting at 0."""
    paid, predicted = np.bincount(month, y), np.bincount(month, expected)
    return float(np.abs(typical - y).mean()), float(100 * np.abs(predicted / paid - 1).mean())


def gap(y, month, ours: tuple[np.ndarray, np.ndarray], theirs: tuple[np.ndarray, np.ndarray], seed: int,
        draws: int = DRAWS) -> dict[str, float]:  # fmt: skip
    """``theirs`` minus ``ours`` on the same admissions (positive = ``theirs`` is worse): the MAE, and the error on the
    month's total in percentage points, each with a 95% interval from drawing the admissions again with replacement."""
    y, month = np.asarray(y, dtype=float), np.asarray(month)
    point = np.subtract(measures(y, month, *theirs), measures(y, month, *ours))
    rng = np.random.default_rng(seed)
    diffs = np.empty((draws, 2))
    for i in range(draws):
        rows = rng.integers(0, len(y), len(y))
        a = measures(y[rows], month[rows], ours[0][rows], ours[1][rows])
        b = measures(y[rows], month[rows], theirs[0][rows], theirs[1][rows])
        diffs[i] = np.subtract(b, a)
    low, high = np.percentile(diffs, [2.5, 97.5], axis=0)
    return {"mae_gap": float(point[0]), "mae_gap_low": float(low[0]), "mae_gap_high": float(high[0]),
            "total_gap": float(point[1]), "total_gap_low": float(low[1]), "total_gap_high": float(high[1])}  # fmt: skip


class FoundationModel:
    """TabPFN with a set of admissions as context. ``predict_both`` gives the two numbers: the median of the predicted
    distribution (typical cost) and its mean (expected cost)."""

    def __init__(self, log_target: bool = False, checkpoint: str = CHECKPOINT, chunk_rows: int = CHUNK_ROWS,
                 seed: int = 42, device: str = "auto"):  # fmt: skip
        self.log_target, self.checkpoint, self.chunk_rows = log_target, checkpoint, chunk_rows
        self.seed, self.device = seed, device

    @property
    def name(self) -> str:
        return f"tabpfn_{self.checkpoint}_{'log_cost' if self.log_target else 'cost'}"

    def fit(self, X: pd.DataFrame, y) -> FoundationModel:
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        self.order_ = CodeOrder().fit(X)
        few = [i for i, c in enumerate(X.columns) if len(self.order_.codes_.get(c, ())) in range(1, MAX_CATEGORIES + 1)]
        self.model_ = TabPFNRegressor.create_default_for_version(
            ModelVersion(self.checkpoint), categorical_features_indices=few, random_state=self.seed, device=self.device
        )
        y = np.asarray(y, dtype=float)
        self.model_.fit(self.order_.transform(X), np.log(y) if self.log_target else y)
        return self

    def predict_both(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        matrix = self.order_.transform(X)
        typical, expected = [], []
        for start in range(0, len(matrix), self.chunk_rows):
            out = self.model_.predict(matrix[start : start + self.chunk_rows], output_type="main", quantiles=list(QUANTILES))
            if self.log_target:
                # exp() of a median is the median; the mean has to be taken after exp(), over the whole distribution.
                typical.append(np.exp(out["median"]))
                expected.append(np.exp(np.stack(out["quantiles"])).mean(axis=0))
            else:
                typical.append(np.asarray(out["median"], dtype=float))
                expected.append(np.asarray(out["mean"], dtype=float))
        return np.concatenate(typical), np.concatenate(expected)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_both(X)[0]


def variants() -> list[tuple[str, bool, bool]]:
    """(context, log target, code statistics) for every TabPFN row."""
    return [("all", False, False), ("recent", False, False), ("recent", True, False), ("recent", True, True)]


def run(cfg: Config) -> pd.DataFrame:
    if not tabpfn_available():
        raise RuntimeError("TabPFN is not installed; run `uv sync --extra foundation` first")
    searched = cfg.reports_dir / "train_valid.json"
    if not searched.exists():
        raise RuntimeError(f"{searched} not found; run `costmodel train` first (the LightGBM rows use two of its settings)")
    search = pd.DataFrame(json.loads(searched.read_text(encoding="utf-8")))
    first, second = pair(search, cfg)
    tc = cfg.train

    df = load_splits(cfg)
    train, valid = df[df["split"] == "train"], df[df["split"] == "valid"]
    scored = sample(valid, EVAL_ROWS, tc.seed)
    X_eval, y_eval, period = F.build_features(scored), scored[F.TARGET].to_numpy(dtype=float), scored["period"]
    month = pd.factorize(period, sort=True)[0]
    contexts = {
        "all": ("the whole training window", sample(train, CONTEXT_ROWS, tc.seed)),
        "recent": (f"its last {RECENT_MONTHS} months", sample(last_months(train, RECENT_MONTHS), CONTEXT_ROWS, tc.seed)),
    }
    rows, production = [], []  # production: the two numbers of the first row, which every other row is compared with

    def add(model: str, reads: str, setting: str, n: int, typical: np.ndarray, expected: np.ndarray, t0: float) -> None:
        metrics = pair_summary(y_eval, typical, expected, period)
        seconds = round(time.perf_counter() - t0, 1)
        if not production:
            production.extend([typical, expected])
            against = dict.fromkeys(GAPS)
        else:
            against = gap(y_eval, month, tuple(production), (typical, expected), tc.seed)
        rows.append({"model": model, "reads": reads, "setting": setting, "rows": n,
                     **{k: metrics[k] for k in KEEP}, **against, "seconds": seconds})  # fmt: skip
        mlflow.log_metrics({f"valid_{k}_{len(rows)}": metrics[k] for k in ("mae", "monthly_total_mape")})
        print(f"{model:<9} {reads:<46} {setting:<36} MAE {metrics['mae']:9.2f}"
              f"  monthly total err {metrics['monthly_total_mape']:5.2f}%  ({seconds}s)", flush=True)  # fmt: skip

    registry.setup(cfg)
    with mlflow.start_run(run_name="foundation model"):
        mlflow.set_tags({"step": "foundation", "dataset": cfg.name})
        mlflow.log_params({"checkpoint": CHECKPOINT, "context_rows": CONTEXT_ROWS, "eval_rows": len(scored)})
        both = "the two settings in production"

        # The models in production, fitted again exactly as the search left them (same rows, same number of trees).
        t0 = time.perf_counter()
        preds = []
        for setting in (first, second):
            window = last_months(train, int(setting["window_months"]))
            model = CostModel(setting["objective"], int(setting["num_leaves"]), tc.learning_rate, int(setting["trees"]),
                              tc.min_category_count, tc.seed, bool(setting["use_time"]))  # fmt: skip
            preds.append(model.fit(F.build_features(window), window[F.TARGET]).predict(X_eval))
        add("LightGBM", "every training admission", both, len(train), preds[0], preds[1], t0)

        for label, context in contexts.values():
            t0 = time.perf_counter()
            preds = [fit_setting(cfg, setting, context, X_eval, y_eval).predict(X_eval) for setting in (first, second)]
            add("LightGBM", f"{len(context):,} admissions of {label}", both, len(context), preds[0], preds[1], t0)

        label, context = contexts["recent"]
        others = train.drop(index=context.index)  # no context row's own cost goes into the number it is given
        learned = code_statistics(F.build_features(others), others[F.TARGET], tc.min_category_count)
        with_codes = with_statistics(F.build_features(context), learned), with_statistics(X_eval, learned)
        t0 = time.perf_counter()
        preds = []
        for setting in (first, second):
            model = StatisticsModel(setting["objective"], int(setting["num_leaves"]), tc.learning_rate, tc.max_rounds,
                                    tc.min_category_count, tc.seed, bool(setting["use_time"]))  # fmt: skip
            model.fit(with_codes[0], context[F.TARGET], with_codes[1], y_eval, tc.early_stopping_rounds)
            preds.append(model.predict(with_codes[1]))
        add("LightGBM", f"{len(context):,} admissions of {label}", f"{both}, with code statistics", len(context),
            preds[0], preds[1], t0)  # fmt: skip

        for key, log_target, statistics in variants():
            label, context = contexts[key]
            t0 = time.perf_counter()
            X_context, X_scored = F.build_features(context), X_eval
            if statistics:
                if key != "recent":
                    raise ValueError("code statistics are computed for the recent context only")
                X_context, X_scored = with_codes
            model = FoundationModel(log_target, seed=tc.seed).fit(X_context, context[F.TARGET])
            typical, expected = model.predict_both(X_scored)
            setting = ("predicts log(cost)" if log_target else "predicts the cost") + (
                ", with code statistics" if statistics else ""
            )
            add(f"TabPFN-{CHECKPOINT[1:]}", f"{len(context):,} admissions of {label}", setting, len(context), typical,
                expected, t0)  # fmt: skip
    results = pd.DataFrame(rows)
    write_report(cfg, results, setting_label(first), setting_label(second), len(train), len(valid))
    return results


def write_report(cfg: Config, results: pd.DataFrame, first: str, second: str, n_train: int, n_valid: int) -> None:
    out = cfg.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": f"TabPFN-{CHECKPOINT[1:]}",
        "context_rows": CONTEXT_ROWS,
        "eval_rows": min(EVAL_ROWS, n_valid),
        "valid_rows": n_valid,
        "train_rows": n_train,
        "typical_setting": first,
        "expected_setting": second,
        "rows": json.loads(results.to_json(orient="records")),
    }
    (out / "foundation_valid.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    full = results.iloc[0]
    tab = results[results["model"].str.startswith("TabPFN")]
    best, best_total = tab.loc[tab["mae"].idxmin()], tab.loc[tab["monthly_total_mape"].idxmin()]
    twin = results[(results["model"] == "LightGBM") & (results["reads"] == best["reads"])]
    twin = twin[twin["setting"].str.contains("statistics") == ("statistics" in best["setting"])].iloc[0]

    def interval(row: pd.Series, key: str, unit: str) -> str:
        return f"{row[key]:+.2f}{unit} ({row[key + '_low']:+.2f} to {row[key + '_high']:+.2f})"

    def verdict(row: pd.Series, key: str) -> str:
        if row[key + "_low"] <= 0 <= row[key + "_high"]:
            return "a difference this sample cannot tell from zero"
        return "worse" if row[key] > 0 else "better"

    others = results.iloc[1:]
    against = others[["model", "reads", "setting"]].copy()
    against["MAE difference"] = [interval(r, "mae_gap", "") for _, r in others.iterrows()]
    against["monthly total difference"] = [interval(r, "total_gap", " pts") for _, r in others.iterrows()]
    lines = [
        f"# A tabular foundation model on the validation window ({cfg.name})",
        "",
        f"Generated by `costmodel foundation --config {cfg.path.as_posix() if cfg.path else ''}`. {payload['checkpoint']}"
        f" reads at most {CONTEXT_ROWS:,} admissions as context; the training window has {n_train:,}. Every row below is"
        f" scored on the same {payload['eval_rows']:,} admissions, drawn at random from the {n_valid:,} of the validation"
        " window, so the numbers compare with each other but not with the other reports (a sample's monthly totals are"
        " noisier than the window's).",
        "",
        "Each row gives two numbers per admission. **MAE** and the columns next to it are measured on the typical cost;"
        " **bias** and **monthly total error** on the expected cost. For LightGBM those are the two settings in"
        f" production (typical: {first}; expected: {second}); for TabPFN, the median and the mean of the distribution it"
        " predicts. On the small sets of rows LightGBM stops adding trees when the scored admissions stop improving, as"
        " in the search.",
        "",
        to_markdown(results.drop(columns=list(GAPS)), index=False),
        "",
        "## Against LightGBM on every training admission",
        "",
        "Each row minus the first, on the same admissions: positive means worse than the model in production. In"
        f" brackets, the 95% interval from drawing the {payload['eval_rows']:,} admissions again with replacement"
        f" {DRAWS:,} times. An interval that contains zero is a difference this sample cannot show.",
        "",
        to_markdown(against, index=False),
        "",
        f"**Per admission**, the best TabPFN row ({best['reads']}; {best['setting']}) has MAE {best['mae']:.2f}."
        f" LightGBM with the same rows and inputs has {twin['mae']:.2f}, and LightGBM on every training admission"
        f" {full['mae']:.2f}: {interval(best, 'mae_gap', '')}, {verdict(best, 'mae_gap')}.",
        "",
        f"**On the month's total**, the best TabPFN row ({best_total['reads']}; {best_total['setting']}) misses by"
        f" {best_total['monthly_total_mape']:.2f}%, against {full['monthly_total_mape']:.2f}% for LightGBM on every"
        f" training admission: {interval(best_total, 'total_gap', ' pts')}, {verdict(best_total, 'total_gap')}.",
        "",
        "Validation window only: the test window was not read and nothing was registered.",
    ]
    (out / "foundation_valid.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
