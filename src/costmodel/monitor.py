"""Watch the production model over the monitoring window, month by month, as if it were live.

Two kinds of drift, available at different times:

* **Data drift** (PSI of each input, and of the predicted cost) is known the moment admissions arrive: it needs no cost.
* **Performance** (MAE, error on the month's total) needs the cost, which is known only once the admission is billed.
  On SIH/SUS, 3 months after the admission month covers 98.7% of admissions and 95.4% of the money; expensive stays bill
  later, so judging a month earlier would make the error look smaller than it is. A month counts only once it is
  ``label_delay_months`` old ("mature").

When the error grows, the report also says **where**: the error on the month's total by admission type, and how
prices moved. The price level compares what was paid with what the same admissions would cost at the prices the model
last saw (the mean per procedure in the last months of its data), so a change in mix does not count. A change in what
is paid for the same procedure shows there and in no input drift.

The **retrain rule** fires when the last ``patience`` mature months all breach: MAE more than ``mae_tolerance`` above the
model's reference MAE (its error on the test window), or a monthly-total error above the gate's ceiling. The rule only
*recommends*: retraining is a decision a person makes (``costmodel run`` with the new data). ``costmodel replay`` shows
what following it would have done.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from costmodel import features as F  # noqa: E402
from costmodel import registry  # noqa: E402
from costmodel.config import Config  # noqa: E402
from costmodel.drift import (  # noqa: E402
    PSI_ALERT,
    PSI_STABLE,
    breaches,
    error_by_group,
    known_through,
    pair_monthly_metrics,
    price_level,
    price_shifts,
    psi,
    should_retrain,
)
from costmodel.evaluate import to_markdown  # noqa: E402

TIME_FEATURES = ["admission_month", "admission_period"]  # they drift by construction; not a signal
DRIFT_FEATURES = [c for c in F.FEATURES if c not in TIME_FEATURES]
RECENT_MONTHS = 12  # "what the model last saw": the last months of the data it was fitted on
MIN_SHARE = 0.02  # admission types with a smaller share of the admissions are left out of the chart (too noisy)


# --- the command ---------------------------------------------------------------------------------------------------


def reference_rows(cfg: Config, df: pd.DataFrame, version) -> pd.DataFrame:
    """The admissions the production model was fitted on (training + validation, its window)."""
    from costmodel.train import last_months

    return last_months(df[df["split"].isin(["train", "valid"])], int(version.tags.get("window_months", 0)))


def run(cfg: Config) -> dict:
    from costmodel.train import load_splits

    registry.setup(cfg)
    version = registry.version_for(cfg, registry.PRODUCTION)
    if version is None:
        raise RuntimeError(f"no '{registry.PRODUCTION}' model for {cfg.model_name}; run `costmodel evaluate` first")
    if "test_mae" not in version.tags:
        raise RuntimeError(f"version {version.version} has no test_mae tag; it did not go through `costmodel evaluate`")
    reference_mae = float(version.tags["test_mae"])
    model = registry.load(cfg, registry.PRODUCTION)
    rules, ceiling = cfg.monitor, cfg.gate.max_monthly_total_mape

    df = load_splits(cfg)
    live = df[df["split"] == "monitor"]
    if live.empty:
        raise RuntimeError("the monitoring window is empty; check [split] in the config")
    ref = reference_rows(cfg, df, version)
    ref_sample = ref.sample(min(len(ref), 200_000), random_state=0)
    X_ref = F.build_features(ref)
    pred_ref = registry.predict(model, ref_sample)

    # Two numbers per admission: the MAE is the typical cost's, every total (and its error) the expected cost's.
    typical, pred = registry.predict_both(model, live)
    X_live = F.build_features(live)
    months = pair_monthly_metrics(live[F.TARGET].to_numpy(), typical, pred, live["period"])
    pred_s = pd.Series(typical, index=live.index)
    for m in months:
        rows = live["period"] == m["period"]
        drift = {c: psi(X_ref[c], X_live.loc[rows, c]) for c in DRIFT_FEATURES}
        drift["predicted cost"] = psi(pd.Series(pred_ref), pred_s[rows])
        m["psi"] = drift
        m["breaches"] = breaches(m, reference_mae, rules, ceiling)

    # Where the error comes from: by admission type, and prices against the level the model last saw.
    y_live = live[F.TARGET].to_numpy()
    by_type = error_by_group(y_live, pred, live["period"], live["admission_type"])
    ref_period = ref["admission_date"].dt.to_period("M")
    recent = ref[ref_period > ref_period.max() - RECENT_MONTHS]
    last_seen = recent.groupby(recent["requested_procedure"].astype("string"))[F.TARGET].agg(["mean", "median"])
    prices = price_level(y_live, live["period"], live["requested_procedure"], last_seen["mean"])
    for m, level in zip(months, prices, strict=True):
        m["by_admission_type"] = by_type[m["period"]]
        m["price_index_pct"] = level["price_index_pct"]
        m["price_covered_pct"] = level["covered_pct"]
    shifts = price_shifts(y_live, pred, live["period"], live["requested_procedure"], last_seen)

    # Replay time: at the end of each month, which months are mature, and what does the rule say?
    periods = [pd.Period(m["period"], "M") for m in months]
    timeline = []
    fired_at = None
    for decision in pd.period_range(periods[0], periods[-1] + rules.label_delay_months, freq="M"):
        mature = [m for m, p in zip(months, periods, strict=True) if p <= known_through(decision, rules)]
        fire, reason = should_retrain(mature, reference_mae, rules, ceiling)
        timeline.append({"decision_month": str(decision), "mature_through": str(known_through(decision, rules)),
                         "retrain": fire, "reason": reason})  # fmt: skip
        if fire and fired_at is None:
            fired_at = str(decision)

    result = {
        "version": str(version.version),
        "two_numbers": pred is not typical,  # a typical and an expected cost, or one number for both
        "reference_mae": reference_mae,
        "rules": rules.__dict__ | {"max_monthly_total_error": ceiling},
        "months": months,
        "price_shifts": shifts,
        "timeline": timeline,
        "first_retrain_recommendation": fired_at,
    }
    with mlflow.start_run(run_name="monitor"):
        mlflow.set_tags({"step": "monitor", "dataset": cfg.name, "model_version": str(version.version)})
        for i, m in enumerate(months):
            mlflow.log_metrics(
                {
                    "monthly_mae": m["mae"],
                    "monthly_total_error_pct": m["total_error_pct"],
                    "max_psi": max(m["psi"].values()),
                },
                step=i,
            )
        mlflow.set_tag("first_retrain_recommendation", fired_at or "none")
    write_report(cfg, result)
    status = f"RETRAIN recommended at the end of {fired_at}" if fired_at else "OK: the retrain rule did not fire"
    print(status)
    return result


def write_report(cfg: Config, result: dict) -> None:
    out = cfg.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "monitor.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    months, rules = result["months"], result["rules"]
    ref = result["reference_mae"]
    chart(cfg, months, ref, rules, out / "monitor.png", result["first_retrain_recommendation"])
    types = shown_types(months)
    by_type = pd.DataFrame({"month": [m["period"] for m in months]})
    for t in types:
        by_type[f"{t}: total error %"] = [m["by_admission_type"].get(t, {}).get("total_error_pct") for m in months]
    for t in types:
        by_type[f"{t}: paid (millions)"] = [m["by_admission_type"].get(t, {}).get("actual", 0) / 1e6 for m in months]
    by_type.insert(1, "prices vs last seen %", [m["price_index_pct"] for m in months])
    shifts = result.get("price_shifts", [])
    shift_table = pd.DataFrame(
        [
            {
                "procedure": g["code"],
                "n": g["n"],
                "paid (millions)": g["paid"] / 1e6,
                "at last-seen prices (millions)": g["at_reference_prices"] / 1e6,
                "extra (millions)": g["extra"] / 1e6,
                "predicted (millions)": g["predicted"] / 1e6,
            }
            for g in shifts
        ]
    )
    median_table = pd.DataFrame(
        [{"procedure": g["code"], "model last saw": g["reference_median"], **g["monthly_median"]} for g in shifts]
    )

    table = pd.DataFrame(
        [
            {
                "month": m["period"],
                "n": int(m["n"]),
                "MAE": m["mae"],
                "total error %": m["total_error_pct"],
                "max PSI": max(m["psi"].values()),
                "most drifted": max(m["psi"], key=m["psi"].get),
                "out of bounds": "; ".join(m["breaches"]) or "-",
            }
            for m in months
        ]
    )
    psi_table = pd.DataFrame({m["period"]: m["psi"] for m in months}).T
    psi_table = psi_table[psi_table.max().sort_values(ascending=False).index]
    fired = result["first_retrain_recommendation"]
    lines = [
        f"# Monitoring the production model ({cfg.name})",
        "",
        f"Generated by `costmodel monitor --config {cfg.path.as_posix() if cfg.path else ''}`. Model: `{cfg.model_name}`"
        f" version {result['version']}, reference MAE {ref:.2f} (its error on the test window)."
        + (
            " The model gives two numbers per admission: the MAE below is the typical cost's; every total, and its error,"
            " is the expected cost's."
            if result.get("two_numbers")
            else ""
        ),
        "",
        f"Retrain rule: {rules['patience']} mature months in a row with MAE above {100 * rules['mae_tolerance']:.0f}% over"
        f" the reference ({(1 + rules['mae_tolerance']) * ref:.2f})"
        + (
            f" or a monthly-total error beyond {rules['max_monthly_total_error']:.0f}%"
            if rules["max_monthly_total_error"] is not None
            else ""
        )
        + f". A month is mature {rules['label_delay_months']} months after it ends (costs are known only once billed).",
        "",
        f"**{'Retraining recommended at the end of ' + fired if fired else 'The rule did not fire in this window'}.**",
        "",
        "![Monitoring](monitor.png)",
        "",
        "## Month by month",
        "",
        to_markdown(table, index=False),
        "",
        f"## Data drift: PSI per input against the training data (stable < {PSI_STABLE}, shift > {PSI_ALERT})",
        "",
        "Known as soon as the month's admissions arrive, months before their costs.",
        "",
        to_markdown(psi_table.round(3)),
        "",
        "## Where the error comes from",
        "",
        f"\"Prices vs last seen\" compares what was paid with what the same admissions would cost at the mean price per"
        f" procedure in the last {RECENT_MONTHS} months of the model's data (same mix, so only price counts). Then the error"
        " on the month's total by admission type (negative = under-predicted), and what was paid:",
        "",
        to_markdown(by_type.round(2), index=False),
        "",
        "The procedures that cost most above the prices the model last saw, over the whole window:",
        "",
        to_markdown(shift_table.round(2), index=False) if len(shift_table) else "-",
        "",
        "Median paid for each of them, by admission month. A jump is a change in what is paid for the same procedure:",
        "",
        to_markdown(median_table.round(0), index=False) if len(median_table) else "-",
        "",
        "## When each decision could be made",
        "",
        to_markdown(pd.DataFrame(result["timeline"]), index=False),
    ]
    (out / "monitor.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def shown_types(months: list[dict]) -> list[str]:
    """Admission types with enough admissions over the window to be read month by month, largest first."""
    counts: dict[str, float] = {}
    for m in months:
        for name, v in m.get("by_admission_type", {}).items():
            counts[name] = counts.get(name, 0) + v["n"]
    total = sum(counts.values())
    return [t for t, n in sorted(counts.items(), key=lambda kv: -kv[1]) if total and n / total >= MIN_SHARE]


def chart(cfg: Config, months: list[dict], reference_mae: float, rules: dict, path, fired_at: str | None = None) -> None:
    from costmodel.eda import BLUE, INK, MUTED, ORANGE, SURFACE, _style

    x = [pd.Period(m["period"], "M").to_timestamp() for m in months]
    fig, axes = plt.subplots(4, 1, figsize=(10, 9.8), sharex=True, facecolor=SURFACE)
    limit = (1 + rules["mae_tolerance"]) * reference_mae
    ax = axes[0]
    _style(ax)
    ax.plot(x, [m["mae"] for m in months], color=BLUE, linewidth=2, marker="o", markersize=4)
    ax.axhline(reference_mae, color=MUTED, linewidth=1, linestyle=":")
    ax.axhline(limit, color=ORANGE, linewidth=1, linestyle="--")
    ax.text(x[0], limit, f" retrain limit {limit:,.0f}", color=ORANGE, fontsize=8, va="bottom")
    ax.text(x[0], reference_mae, f" reference {reference_mae:,.0f}", color=MUTED, fontsize=8, va="bottom")
    ax.set_title("MAE per admission, by admission month", loc="left", fontsize=11, color=INK)
    ax = axes[1]
    _style(ax)
    ax.plot(x, [m["total_error_pct"] for m in months], color=BLUE, linewidth=2, marker="o", markersize=4)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ceiling = rules["max_monthly_total_error"]
    if ceiling is not None:
        for c in (ceiling, -ceiling):
            ax.axhline(c, color=ORANGE, linewidth=1, linestyle="--")
    ax.set_title("Error on the month's total cost (%; negative = under-predicted)", loc="left", fontsize=11, color=INK)
    ax = axes[2]
    _style(ax)
    for name, color in zip(shown_types(months), [BLUE, "#1f9e74", "#8a5cd1", MUTED], strict=False):
        ys = [m["by_admission_type"].get(name, {}).get("total_error_pct") for m in months]
        ax.plot(x, ys, color=color, linewidth=2, marker="o", markersize=4, label=name)
    ax.plot(x, [m["price_index_pct"] for m in months], color=ORANGE, linewidth=2, marker="o", markersize=4,
            label="prices vs what the model last saw")  # fmt: skip
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="center left")
    ax.set_title("The same error by admission type, and the price level (%)", loc="left", fontsize=11, color=INK)
    ax = axes[3]
    _style(ax)
    ax.bar(x, [max(m["psi"].values()) for m in months], width=20, color=BLUE)
    ax.axhline(PSI_ALERT, color=ORANGE, linewidth=1, linestyle="--")
    ax.set_title("Largest PSI among the inputs (data drift)", loc="left", fontsize=11, color=INK)
    if fired_at:
        # Decided at the end of that month: draw the line where the next month starts.
        when = (pd.Period(fired_at, "M") + 1).to_timestamp()
        for ax in axes:
            ax.axvline(when, color=INK, linewidth=1)
        axes[0].text(when, axes[0].get_ylim()[1], f" retrain recommended (end of {fired_at})", color=INK, fontsize=8,
                     va="top")  # fmt: skip
    fig.suptitle(f"{cfg.name}: production model, month by month", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
