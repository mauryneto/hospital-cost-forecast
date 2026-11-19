"""Replay the monitoring window under different retraining policies, to see what each would have done.

Month by month, as if live: the model in use prices the month's admissions; at the end of the month, only the months
that are mature by then (costs billed, see ``drift.known_through``) can be looked at. Policies:

* ``never``: keep the production model all along.
* ``rule``: retrain when the retrain rule fires (``drift.should_retrain``).
* ``monthly``: try to retrain at the end of every month.

A retrain uses only data known at that moment, with the production model's settings (both of them, when it gives a
typical and an expected cost), on an expanding or a sliding window. Like any candidate, it must pass the gate before
it is used: it is fitted without the last ``gate_months`` mature months, compared with the model in use on those
months, and, if it passes, refitted on everything known and put in use from the next month.

The gate months must be new to *both* models. So a retrain is tried only when at least one month beyond the gate months
is newer than the data of the model in use; otherwise the candidate would learn nothing new, and the model in use would
be judged on months it was trained on. With 2 gate months, a "monthly" policy can therefore change models at most every
3 months. Nothing here touches the registry: it is a what-if, logged as one MLflow run.
"""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from costmodel import features as F  # noqa: E402
from costmodel import registry  # noqa: E402
from costmodel.candidate import CostModel, CostPair, predict_both  # noqa: E402
from costmodel.config import Config  # noqa: E402
from costmodel.drift import known_through, pair_monthly_metrics, should_retrain  # noqa: E402
from costmodel.evaluate import pair_summary, to_markdown  # noqa: E402
from costmodel.gate import should_promote  # noqa: E402
from costmodel.train import last_months, load_splits, window_label  # noqa: E402

POLICIES = ("never", "rule", "monthly")


def like(template):
    """A fresh model with the same settings (and number of trees) as ``template``."""
    if isinstance(template, CostPair):
        typical = like(template.typical)
        return CostPair(typical, typical if template.expected is template.typical else like(template.expected))
    return CostModel(
        template.objective,
        template.num_leaves,
        template.learning_rate,
        template.best_iteration_,
        template.min_category_count,
        template.seed,
        getattr(template, "use_time", False),
        template.scaled,
    )


def simulate(cfg: Config, df: pd.DataFrame, live: pd.DataFrame, start_model, start_cutoff: pd.Period,
             reference_mae: float, template: CostModel, policy: str, window: int) -> dict:  # fmt: skip
    rules, ceiling, gate_months = cfg.monitor, cfg.gate.max_monthly_total_mape, cfg.replay.gate_months
    period = df["admission_date"].dt.to_period("M")
    live_period = live["admission_date"].dt.to_period("M")
    months = sorted(live_period.unique())
    X_live, y_live = F.build_features(live), live[F.TARGET].to_numpy()

    current, current_ref, deployed = start_model, reference_mae, 0  # index of the first month served by `current`
    cutoff = start_cutoff  # last admission month in the data of the model in use
    typical, pred = np.empty(len(live)), np.empty(len(live))  # `pred` is the expected cost: the one that is added up
    events, waits = [], 0
    for i, month in enumerate(months):
        rows = (live_period == month).to_numpy()
        typical[rows], pred[rows] = predict_both(current, X_live[rows])
        if policy == "never" or i == len(months) - 1:
            continue  # a model retrained after the last month would serve outside the window
        known = known_through(month, rules)
        served = ((live_period >= months[deployed]) & (live_period <= known)).to_numpy()
        mature = pair_monthly_metrics(y_live[served], typical[served], pred[served], live.loc[served, "period"])
        if policy == "rule":
            fire, reason = should_retrain(mature, current_ref, rules, ceiling)
        else:
            fire, reason = True, "scheduled"
        if not fire:
            continue
        if known - gate_months <= cutoff:
            waits += 1  # no month the model in use has not seen, beyond the gate months: nothing new to learn yet
            continue

        t0 = time.perf_counter()
        data = last_months(df[period <= known], window)
        holdout = period[data.index] > known - gate_months
        fit_part, hold = data[~holdout], data[holdout]
        if fit_part.empty or len(hold) < 10:
            continue  # not enough known data yet to train and gate a candidate
        X_hold = F.build_features(hold)
        candidate = like(template).fit(F.build_features(fit_part), fit_part[F.TARGET])
        y_hold = hold[F.TARGET].to_numpy()
        cand_m = pair_summary(y_hold, *predict_both(candidate, X_hold), hold["period"])
        cur_m = pair_summary(y_hold, *predict_both(current, X_hold), hold["period"])
        promote, gate_reasons = should_promote(cand_m, {}, cfg.gate, ("model in use", cur_m))
        if promote:
            current = like(template).fit(F.build_features(data), data[F.TARGET])
            current_ref, deployed, cutoff = cand_m["mae"], i + 1, known
        events.append(
            {
                "decided": str(month),
                "why": reason,
                "data through": str(known),
                "candidate MAE": cand_m["mae"],
                "in-use MAE": cur_m["mae"],
                "promoted": promote,
                "serves from": str(months[i + 1]) if promote else "-",
                "gate": "; ".join(gate_reasons),
            }
        )
        print(
            f"  {policy:<8} {window_label(window):<18} end of {month}: data through {known}, candidate MAE"
            f" {cand_m['mae']:.2f} vs in use {cur_m['mae']:.2f} -> {'PROMOTED' if promote else 'kept'}"
            f" ({time.perf_counter() - t0:.0f}s)"
        )

    return {
        "policy": policy,
        "window": window_label(window) if policy != "never" else "-",
        "overall": pair_summary(y_live, typical, pred, live["period"]),
        "monthly": pair_monthly_metrics(y_live, typical, pred, live["period"]),
        "attempts": len(events),
        "promotions": sum(e["promoted"] for e in events),
        "waits": waits,
        "events": events,
    }


def run(cfg: Config) -> list[dict]:
    registry.setup(cfg)
    version = registry.version_for(cfg, registry.PRODUCTION)
    if version is None or "test_mae" not in version.tags:
        raise RuntimeError(f"no evaluated '{registry.PRODUCTION}' model for {cfg.model_name}; run `costmodel evaluate`")
    production = registry.unwrap(registry.load(cfg, registry.PRODUCTION))
    template = production
    if not isinstance(template, CostModel | CostPair):  # a baseline in production: retrain with the candidate's settings
        template = registry.unwrap(registry.load(cfg, registry.STAGING))
    unknown = set(cfg.replay.policies) - set(POLICIES)
    if unknown:
        raise ValueError(f"unknown replay policies {sorted(unknown)}; use {POLICIES}")

    df = load_splits(cfg)
    live = df[df["split"] == "monitor"]
    reference_mae = float(version.tags["test_mae"])
    start_cutoff = df.loc[df["split"].isin(["train", "valid"]), "admission_date"].max().to_period("M")
    results = []
    for policy in cfg.replay.policies:
        for window in [0] if policy == "never" else cfg.replay.windows:
            print(f"replaying '{policy}' ({window_label(window)})")
            results.append(simulate(cfg, df, live, production, start_cutoff, reference_mae, template, policy, window))

    with mlflow.start_run(run_name="replay retraining policies"):
        mlflow.set_tags({"step": "replay", "dataset": cfg.name, "model_version": str(version.version)})
        for r in results:
            key = f"{r['policy']}_{r['window']}".replace(" ", "_").replace(",", "")
            mlflow.log_metrics(
                {f"{key}_mae": r["overall"]["mae"], f"{key}_monthly_total_mape": r["overall"]["monthly_total_mape"]}
            )
    write_report(cfg, results, version, reference_mae)
    return results


def write_report(cfg: Config, results: list[dict], version, reference_mae: float) -> None:
    out = cfg.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "replay.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    chart(cfg, results, out / "replay.png")
    names = [label(r) for r in results]
    summary_table = pd.DataFrame(
        [
            {
                "policy": label(r),
                "MAE": r["overall"]["mae"],
                "monthly total error %": r["overall"]["monthly_total_mape"],
                "bias %": r["overall"]["bias_pct"],
                "retrain attempts": r["attempts"],
                "waits (no new data)": r["waits"],
                "models put in use": r["promotions"],
            }
            for r in results
        ]
    )
    mae = pd.DataFrame({n: {m["period"]: m["mae"] for m in r["monthly"]} for n, r in zip(names, results, strict=True)})
    total = pd.DataFrame(
        {n: {m["period"]: m["total_error_pct"] for m in r["monthly"]} for n, r in zip(names, results, strict=True)}
    )
    rules = cfg.monitor
    lines = [
        f"# Retraining policies, replayed over the monitoring window ({cfg.name})",
        "",
        f"Generated by `costmodel replay --config {cfg.path.as_posix() if cfg.path else ''}`. Starting model:"
        f" `{cfg.model_name}` version {version.version} (reference MAE {reference_mae:.2f}). Costs become known"
        f" {rules.label_delay_months} months after the admission month; retrained models use only data known at the"
        f" time, and each must pass the gate on the last {cfg.replay.gate_months} known months before it is used.",
        "",
        to_markdown(summary_table, index=False),
        "",
        "![Retraining policies](replay.png)",
        "",
        "## MAE by month",
        "",
        to_markdown(mae),
        "",
        "## Error on the month's total (%)",
        "",
        to_markdown(total),
    ]
    for r in results:
        if r["events"]:
            ev = pd.DataFrame(r["events"]).drop(columns=["gate"])
            lines += ["", f"## Retrain decisions: {label(r)}", "", to_markdown(ev, index=False)]
    (out / "replay.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def label(r: dict) -> str:
    return r["policy"] if r["policy"] == "never" else f"{r['policy']} ({r['window']})"


def chart(cfg: Config, results: list[dict], path) -> None:
    from costmodel.eda import INK, MUTED, SURFACE, _style

    colors = ["#52514e", "#2a78d6", "#eb6834", "#1f9e74", "#8a5cd1", "#c2417a"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, facecolor=SURFACE)
    for ax, key, title in [
        (axes[0], "mae", "MAE per admission, by admission month"),
        (axes[1], "total_error_pct", "Error on the month's total cost (%; negative = under-predicted)"),
    ]:
        _style(ax)
        for r, color in zip(results, colors, strict=False):
            x = [pd.Period(m["period"], "M").to_timestamp() for m in r["monthly"]]
            ax.plot(x, [m[key] for m in r["monthly"]], color=color, linewidth=2, marker="o", markersize=3, label=label(r))
            for e in r["events"]:
                if e["promoted"]:
                    xs = pd.Period(e["serves from"], "M").to_timestamp()
                    ax.axvline(xs, color=color, linewidth=0.8, linestyle=":")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    axes[1].axhline(0, color=MUTED, linewidth=0.8)
    axes[1].legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
    fig.suptitle(f"{cfg.name}: retraining policies (dotted line = a retrained model starts serving)", x=0.01, ha="left",
                 fontsize=12, color=INK)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
