"""``costmodel fresh``: a year no choice was made on, scored once, with everything fixed beforehand.

Every choice in this project was made on one validation year and judged on a test year and a monitoring year that ended
up being looked at more than once. This command is the check that is left: one more year, read for the first time, by
settings written down before it was read (``[fresh]`` in the config; the protocol is in DECISIONS.md).

**The model that is scored.** The two settings in ``[fresh]`` (typical cost and expected cost), with their numbers of
trees, fitted at the end of the month before the window on the admissions whose costs are known by then
(``[monitor] label_delay_months``). Nothing is searched.

**Four questions**, each with a verdict computed here from a criterion stated beforehand:

1. *Do the two numbers hold?* The typical cost should have the lower error per admission, the expected cost the lower
   error on the month's total.
2. *Is the model still worth it?* The gate's rule against the best baseline, and ``costmodel skill``'s table: a smaller
   error on totals than a plain average at every level.
3. *Do the ideas that looked good on validation repeat?* Recent months weighing more (``costmodel base-window``), a
   typical-cost model of its own for one group (``costmodel segments``), and a foundation model reading recent
   admissions and code statistics (``costmodel foundation``, on a random sample of the window).
4. *Is retraining every month better than waiting for the rule?* ``costmodel replay``'s simulation, over this window.

Differences come with a 95% interval, from drawing the scored admissions again with replacement; a difference counts
only when its interval leaves zero out. The model in production, fitted on older data and never retrained, is scored
too, for the record. Nothing is registered and nothing is adopted: this describes, it does not decide.
"""

from __future__ import annotations

import json
import time

import mlflow
import numpy as np
import pandas as pd

from costmodel import features as F
from costmodel import foundation, registry, skill
from costmodel.base_window import HALF_LIFE, recency_weights
from costmodel.baselines import all_baselines
from costmodel.candidate import EXPECTED, TYPICAL, CostModel, CostPair, predict_both
from costmodel.config import Config
from costmodel.drift import known_through, pair_monthly_metrics
from costmodel.evaluate import pair_summary, summary, to_markdown
from costmodel.gate import should_promote
from costmodel.replay import label, like, simulate
from costmodel.sources import load_table
from costmodel.train import window_label

DRAWS = 1_000  # resamples of the scored admissions behind each interval
CONFIRMED, CONTRADICTED, NOT_SHOWN = "confirmed", "contradicted", "not shown"
NOT_CONFIRMED, NOT_RUN = "not confirmed", "not run"
KEEP = ("mae", "rmse", "mae_top_decile", "bias_pct", "monthly_total_mape")


def setting_model(cfg: Config, setting: dict, trees: int | None = None) -> CostModel:
    """One of the settings of ``[fresh]``, not fitted yet."""
    tc = cfg.train
    model = CostModel(setting["objective"], int(setting["num_leaves"]), tc.learning_rate, int(trees or setting["trees"]),
                      tc.min_category_count, tc.seed, bool(setting.get("use_time", False)))  # fmt: skip
    model.best_iteration_ = model.n_estimators  # what `replay.like` copies: the trees are fixed, not searched
    return model


def template(cfg: Config) -> CostPair:
    return CostPair(setting_model(cfg, cfg.fresh.typical), setting_model(cfg, cfg.fresh.expected))


def verdict(low: float, high: float, want: str) -> str:
    """What a 95% interval of a difference says about an expectation that it would be ``below`` or ``above`` zero."""
    if want not in ("below", "above"):
        raise ValueError("want must be 'below' or 'above'")
    if low <= 0 <= high:
        return NOT_SHOWN
    return CONFIRMED if (high < 0) == (want == "below") else CONTRADICTED


def together(*parts: str) -> str:
    """Several expectations as one: confirmed only if all are, contradicted if any is."""
    if all(p == CONFIRMED for p in parts):
        return CONFIRMED
    return CONTRADICTED if CONTRADICTED in parts else NOT_SHOWN


def with_own_model(typical: np.ndarray, in_group: np.ndarray, own: np.ndarray) -> np.ndarray:
    """The typical cost with the admissions of one group priced by that group's own model."""
    out = typical.copy()
    out[in_group] = own
    return out


def run(cfg: Config) -> dict:
    fc, tc = cfg.fresh, cfg.train
    if fc.window is None or not fc.typical or not fc.expected:
        raise RuntimeError("the config has no [fresh] section with a window and the two settings")
    t_start = time.perf_counter()
    df = load_table(cfg)
    df["period"] = df["admission_date"].dt.to_period("M").astype("string")
    month = df["admission_date"].dt.to_period("M")
    first, last = pd.Period(fc.window[0], "M"), pd.Period(fc.window[1], "M")
    live = df[(month >= first) & (month <= last)]
    if live.empty:
        raise RuntimeError(f"no admissions between {first} and {last}; run `costmodel data` with this config first")
    known_month = known_through(first - 1, cfg.monitor)  # costs known when the model for the window is fitted
    is_known = (month <= known_month).to_numpy()
    known = df[is_known]
    known_period = month[is_known]
    X_known, y_known = F.build_features(known), known[F.TARGET]
    X, y, period = F.build_features(live), live[F.TARGET].to_numpy(dtype=float), live["period"]
    month_no = pd.factorize(period, sort=True)[0]
    pair = template(cfg)
    print(f"{len(live):,} admissions from {first} to {last}; model fitted on {len(known):,} known through {known_month}")

    def against(theirs: tuple[np.ndarray, np.ndarray], rows=slice(None)) -> dict:
        ours = (typical[rows], expected[rows])
        return foundation.gap(y[rows], pd.factorize(month_no[rows], sort=True)[0], ours, theirs, tc.seed, DRAWS)

    # ---- the model for the window, fitted the way a retrain is: first without the last known months, to have an
    # error on months it has not seen (the retrain rule's reference), then on everything known.
    held = (known_period > known_month - cfg.replay.gate_months).to_numpy()
    probe = like(pair).fit(X_known[~held], y_known[~held])
    reference = pair_summary(y_known[held].to_numpy(), *predict_both(probe, X_known[held]), known.loc[held, "period"])
    model = like(pair).fit(X_known, y_known)
    typical, expected = predict_both(model, X)
    overall = pair_summary(y, typical, expected, period)
    print(f"model for the window: MAE {overall['mae']:.2f}, monthly total error {overall['monthly_total_mape']:.2f}%")

    # ---- 1. the two numbers
    numbers = {TYPICAL: summary(y, typical, period), EXPECTED: summary(y, expected, period)}
    swap = foundation.gap(y, month_no, (typical, typical), (expected, expected), tc.seed, DRAWS)  # expected - typical
    two = {
        "numbers": numbers,
        "expected_minus_typical": swap,
        "per_admission": verdict(swap["mae_gap_low"], swap["mae_gap_high"], "above"),
        "monthly_total": verdict(swap["total_gap_low"], swap["total_gap_high"], "below"),
    }
    two["verdict"] = together(two["per_admission"], two["monthly_total"])

    # ---- 2. worth it: the gate's rule against the best baseline, and the totals against a plain average
    baseline_metrics, rules = {}, {}
    for baseline in all_baselines():
        pred = baseline.fit(X_known, y_known).predict(X)
        baseline_metrics[baseline.name] = summary(y, pred, period)
        if baseline.name in skill.RULES:
            rules[baseline.name] = pred
    best = min(baseline_metrics, key=lambda name: baseline_metrics[name]["mae"])
    passes, reasons = should_promote(overall, {f"best baseline ({best})": baseline_metrics[best]}, cfg.gate, None)
    skill_rows = skill.table(y, typical, expected, {name: rules[name] for name in skill.RULES}, X, period)
    gains = all(row["gain_pct"] > 0 for row in skill_rows)
    worth = {
        "baselines": baseline_metrics,
        "best_baseline": best,
        "gate_passes": bool(passes),
        "gate_reasons": reasons,
        "skill": json.loads(pd.DataFrame(skill_rows).to_json(orient="records")),
        "gain_at_every_level": bool(gains),
        "verdict": CONFIRMED if passes and gains else NOT_CONFIRMED,
    }

    # ---- 3. the ideas that looked good on validation
    ideas = []
    known_no = X_known["admission_period"].to_numpy(dtype=int)
    weights = recency_weights(known_no, int(known_no.max()), HALF_LIFE)
    weighted = (
        setting_model(cfg, fc.typical).fit(X_known, y_known, sample_weight=weights).predict(X),
        setting_model(cfg, fc.expected).fit(X_known, y_known, sample_weight=weights).predict(X),
    )
    diff = against(weighted)
    ideas.append({
        "idea": "recent months weigh more",
        "detail": f"half-life of {HALF_LIFE} months",
        "scored_on": len(live),
        "ours": {k: overall[k] for k in KEEP},
        "theirs": {k: v for k, v in pair_summary(y, *weighted, period).items() if k in KEEP},
        **diff,
        "expected_from_validation": "better on the month's total",
        "verdict": verdict(diff["total_gap_low"], diff["total_gap_high"], "below"),
    })  # fmt: skip

    if fc.own_model:
        column, value = fc.own_model["column"], str(fc.own_model["value"])
        in_known, in_live = (X_known[column] == value).to_numpy(), (X[column] == value).to_numpy()
        own = setting_model(cfg, fc.typical, int(fc.own_model["trees"])).fit(X_known[in_known], y_known[in_known])
        split = (with_own_model(typical, in_live, own.predict(X[in_live])), expected)
        diff = against(split)
        ideas.append({
            "idea": "a typical-cost model of its own for one group",
            "detail": f"{column} = {value}, {int(in_known.sum()):,} admissions to learn from, {int(in_live.sum()):,} scored",
            "scored_on": len(live),
            "ours": {k: overall[k] for k in KEEP},
            "theirs": {k: v for k, v in pair_summary(y, *split, period).items() if k in KEEP},
            **diff,
            "group_mae_single": float(np.abs(typical[in_live] - y[in_live]).mean()),
            "group_mae_own": float(np.abs(split[0][in_live] - y[in_live]).mean()),
            "expected_from_validation": "better per admission",
            "verdict": verdict(diff["mae_gap_low"], diff["mae_gap_high"], "below"),
        })  # fmt: skip

    reads = f"{foundation.CONTEXT_ROWS:,} admissions of the last {foundation.RECENT_MONTHS} known months"
    tabpfn = {"idea": "a foundation model (TabPFN-2) with code statistics", "detail": reads,
              "expected_from_validation": "better on the month's total, not worse per admission"}  # fmt: skip
    if foundation.tabpfn_available():
        rows = np.sort(np.random.default_rng(tc.seed).choice(len(live), min(foundation.EVAL_ROWS, len(live)), replace=False))
        recent = (known_period > known_month - foundation.RECENT_MONTHS).to_numpy()
        context = foundation.sample(known[recent], foundation.CONTEXT_ROWS, tc.seed)
        rest = ~known.index.isin(context.index)  # no context row's own cost goes into the number it is given
        learned = foundation.code_statistics(X_known[rest], y_known[rest], tc.min_category_count)
        reader = foundation.FoundationModel(log_target=True, seed=tc.seed)
        reader.fit(foundation.with_statistics(X_known.loc[context.index], learned), context[F.TARGET])
        read = reader.predict_both(foundation.with_statistics(X.iloc[rows], learned))
        diff = against(read, rows)
        per_admission = verdict(diff["mae_gap_low"], diff["mae_gap_high"], "below")
        on_total = verdict(diff["total_gap_low"], diff["total_gap_high"], "below")
        ours = pair_summary(y[rows], typical[rows], expected[rows], period.iloc[rows])
        tabpfn |= {
            "scored_on": len(rows),
            "ours": {k: ours[k] for k in KEEP},
            "theirs": {k: v for k, v in pair_summary(y[rows], *read, period.iloc[rows]).items() if k in KEEP},
            **diff,
            "verdict": CONTRADICTED if CONTRADICTED in (per_admission, on_total) else on_total,
        }  # fmt: skip
    else:
        tabpfn |= {"scored_on": 0, "verdict": NOT_RUN}
    ideas.append(tabpfn)
    for idea in ideas:
        print(f"idea: {idea['idea']:<52} {idea['verdict']}")

    # ---- 4. retraining policies, replayed over the window from the model above
    policies = []
    for policy in cfg.replay.policies:
        for window in [0] if policy == "never" else cfg.replay.windows:
            print(f"replaying '{policy}' ({window_label(window)})")
            policies.append(simulate(cfg, df, live, model, known_month, reference["mae"], pair, policy, window))
    by_label = {label(r): r["overall"]["monthly_total_mape"] for r in policies}
    monthly, rule = f"monthly ({window_label(0)})", f"rule ({window_label(0)})"
    retrain = {"policies": policies, "verdict": NOT_RUN}
    if {"never", monthly, rule} <= set(by_label):
        better = by_label[monthly] < by_label["never"] and by_label[monthly] < by_label[rule]
        retrain["verdict"] = CONFIRMED if better else NOT_CONFIRMED

    # ---- for the record: the recipe of the model in production, never retrained
    production = None
    if fc.production_fit_until:
        old = (known_period <= pd.Period(fc.production_fit_until, "M")).to_numpy()
        stale = predict_both(like(pair).fit(X_known[old], y_known[old]), X)
        production = {"fit_until": fc.production_fit_until, "rows": int(old.sum()),
                      **{k: v for k, v in pair_summary(y, *stale, period).items() if k in KEEP},
                      "monthly": pair_monthly_metrics(y, *stale, period)}  # fmt: skip

    result = {
        "window": [str(first), str(last)],
        "admissions": len(live),
        "known_through": str(known_month),
        "fit_rows": len(known),
        "draws": DRAWS,
        "settings": {TYPICAL: dict(fc.typical), EXPECTED: dict(fc.expected)},
        "reference_mae": reference["mae"],
        "model": {k: overall[k] for k in KEEP},
        "monthly": pair_monthly_metrics(y, typical, expected, period),
        "two_numbers": two,
        "worth_it": worth,
        "ideas": ideas,
        "retraining": retrain,
        "production": production,
        "minutes": round((time.perf_counter() - t_start) / 60, 1),
    }
    registry.setup(cfg)
    with mlflow.start_run(run_name="fresh year"):
        mlflow.set_tags({"step": "fresh", "dataset": cfg.name, "window": f"{first} to {last}"})
        mlflow.log_metrics({f"fresh_{k}": v for k, v in result["model"].items()})
    write_report(cfg, result)
    return result


def _span(row: dict, key: str, unit: str = "") -> str:
    if row.get(key) is None:
        return "-"
    return f"{row[key]:+.2f}{unit} ({row[key + '_low']:+.2f} to {row[key + '_high']:+.2f})"


def write_report(cfg: Config, result: dict) -> None:
    out = cfg.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    clean = json.loads(pd.Series([result]).to_json(orient="records"))[0]  # NaN -> null, numpy -> plain numbers
    (out / "fresh_year.json").write_text(json.dumps(clean, indent=2) + "\n", encoding="utf-8", newline="\n")

    two, worth, retrain = result["two_numbers"], result["worth_it"], result["retraining"]
    first, last = result["window"]
    verdicts = pd.DataFrame(
        [
            {"question": "1. Do the two numbers hold?", "verdict": two["verdict"]},
            {"question": "2. Is the model still worth it?", "verdict": worth["verdict"]},
            *({"question": f"3. Idea: {i['idea']}", "verdict": i["verdict"]} for i in result["ideas"]),
            {"question": "4. Is retraining every month better than waiting for the rule?", "verdict": retrain["verdict"]},
        ]
    )
    numbers = pd.DataFrame(
        [{"number": name, **{k: two["numbers"][name][k] for k in KEEP}} for name in (TYPICAL, EXPECTED)]
    )
    baselines = pd.DataFrame(
        [{"model": "this model (two numbers)", **result["model"]}]
        + [{"model": name, **{k: m[k] for k in KEEP}} for name, m in worth["baselines"].items()]
    ).sort_values("mae")
    levels = pd.DataFrame(worth["skill"]).rename(
        columns=skill.RULES | {skill.MODEL: "this model", "gain_pct": "gain over the plain average (%)"}
    )
    ideas = pd.DataFrame(
        [
            {
                "idea": i["idea"],
                "detail": i["detail"],
                "scored on": i["scored_on"],
                "MAE": i.get("theirs", {}).get("mae"),
                "MAE of the model, same admissions": i.get("ours", {}).get("mae"),
                "MAE difference": _span(i, "mae_gap"),
                "monthly total error %": i.get("theirs", {}).get("monthly_total_mape"),
                "of the model": i.get("ours", {}).get("monthly_total_mape"),
                "monthly total difference": _span(i, "total_gap", " pts"),
                "on validation it was": i["expected_from_validation"],
                "verdict": i["verdict"],
            }
            for i in result["ideas"]
        ]
    )
    policies = pd.DataFrame(
        [
            {
                "policy": label(r),
                "MAE": r["overall"]["mae"],
                "monthly total error %": r["overall"]["monthly_total_mape"],
                "bias %": r["overall"]["bias_pct"],
                "retrain attempts": r["attempts"],
                "models put in use": r["promotions"],
            }
            for r in retrain["policies"]
        ]
    )
    months = pd.DataFrame(result["monthly"])
    swap = two["expected_minus_typical"]
    lines = [
        f"# A year no choice was made on: {first} to {last} ({cfg.name})",
        "",
        f"Generated by `costmodel fresh --config {cfg.path.as_posix() if cfg.path else ''}`, by settings fixed before the"
        " window was read (`[fresh]` in the config; the protocol is in DECISIONS.md). The model is the two settings"
        f" below, fitted on the {result['fit_rows']:,} admissions whose costs were known at the end of the month before"
        f" the window (admission months through {result['known_through']}), and it prices the {result['admissions']:,}"
        " admissions of the window. Nothing was searched, registered or adopted.",
        "",
        f"- Typical cost: {json.dumps(result['settings'][TYPICAL])}",
        f"- Expected cost: {json.dumps(result['settings'][EXPECTED])}",
        "",
        f"A difference comes with its 95% interval in brackets, from drawing the scored admissions again with replacement"
        f" {result['draws']:,} times. **{CONFIRMED}**: the interval is all on the side that was expected."
        f" **{NOT_SHOWN}**: it contains zero. **{CONTRADICTED}**: it is all on the other side.",
        "",
        to_markdown(verdicts, index=False),
        "",
        "## 1. The two numbers",
        "",
        to_markdown(numbers, index=False),
        "",
        f"Expected cost minus typical cost: MAE {_span(swap, 'mae_gap')} (expected to be above zero:"
        f" {two['per_admission']}); monthly total error {_span(swap, 'total_gap', ' pts')} (expected to be below zero:"
        f" {two['monthly_total']}).",
        "",
        "## 2. Worth it",
        "",
        "MAE and the columns next to it are measured on the typical cost; bias and monthly total error on the expected"
        " cost. Baselines are fitted on the same admissions as the model.",
        "",
        to_markdown(baselines, index=False),
        "",
        f"The gate's rule against the best baseline ({worth['best_baseline']}):"
        f" **{'passes' if worth['gate_passes'] else 'fails'}**."
        + "".join(f"\n- {reason}" for reason in worth["gate_reasons"]),
        "",
        "Error on totals, group by group and month by month, next to a plain average's (`costmodel skill`):",
        "",
        to_markdown(levels, index=False),
        "",
        f"A gain at every level: **{'yes' if worth['gain_at_every_level'] else 'no'}**.",
        "",
        "## 3. The ideas that looked good on validation",
        "",
        "Each fitted on the same admissions as the model and compared with it on the same scored admissions: negative"
        " differences mean the idea did better.",
        "",
        to_markdown(ideas, index=False),
        "",
        "## 4. Retraining policies",
        "",
        f"The replay of `costmodel replay`, starting from the model above (reference MAE {result['reference_mae']:.2f}, on"
        f" the last {cfg.replay.gate_months} known months it was first fitted without).",
        "",
        to_markdown(policies, index=False),
        "",
        f"Retraining every month, on all known months, better on the month's total than never and than the rule:"
        f" **{retrain['verdict']}**.",
    ]
    for r in retrain["policies"]:
        if r["events"]:
            events = pd.DataFrame(r["events"]).drop(columns=["gate"])
            lines += ["", f"### Retrain decisions: {label(r)}", "", to_markdown(events, index=False)]
    lines += ["", "## The model month by month", "", to_markdown(months, index=False)]
    if result["production"]:
        p = result["production"]
        lines += [
            "",
            "## For the record: the model in production, never retrained",
            "",
            f"The same two settings fitted on the {p['rows']:,} admissions up to {p['fit_until']} and never again: MAE"
            f" {p['mae']:.2f}, monthly total error {p['monthly_total_mape']:.2f}%, bias {p['bias_pct']:.2f}%.",
        ]
    (out / "fresh_year.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
