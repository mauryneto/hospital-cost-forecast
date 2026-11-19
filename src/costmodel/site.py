"""``costmodel site``: one static page with the results, built from the JSON reports.

The page (``docs/index.html`` in the work folder, which GitHub Pages can serve) has no server behind it and loads no
data: the numbers are embedded when it is generated, read from ``reports/<dataset>/test.json``, ``monitor.json`` and,
where they exist, ``replay.json`` and the side comparisons. So the page cannot disagree with
the reports, and regenerating it after a new run is one command. Only aggregates are embedded (per model, per month,
per procedure); never an admission.

When ``costmodel explore`` has run, a second page is written next to it, ``docs/explore.html``: the model in
production group by group (hospital, specialty...) and month by month, with filters. It reuses the first page's
styles and chart code, so the two cannot drift apart.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

from costmodel.config import Config
from costmodel.paths import HOME

TEMPLATE = Path(__file__).with_name("site_template.html")
EXPLORE_TEMPLATE = Path(__file__).with_name("explore_template.html")
SHARED_JS = ("  // ==== shared with the explorer page: start ====\n", "  // ==== shared with the explorer page: end ====\n")
EXPLORER_LINK = {"label": "Explorer: by hospital, specialty and month", "url": "explore.html"}
TEST_METRICS = ("mae", "monthly_total_mape", "bias_pct", "mae_top_decile")
SHIFT_FIELDS = ("code", "n", "extra", "reference_median", "monthly_median")
FAMILY_FIELDS = ("family", "setting", "trees", "seconds", *TEST_METRICS)
FOUNDATION_GAPS = ("mae_gap", "mae_gap_low", "mae_gap_high", "total_gap", "total_gap_low", "total_gap_high")
FOUNDATION_FIELDS = ("model", "reads", "setting", "rows", "seconds", *TEST_METRICS, *FOUNDATION_GAPS)
TYPICAL_BANDS = ("mae_common", "error_share_top_decile")
TYPICAL_FIELDS = ("number", "trees", *TEST_METRICS, *TYPICAL_BANDS, "lead_over_baseline_pct", *FOUNDATION_GAPS)
BLEND_FIELDS = ("weight", *TEST_METRICS)
SEGMENT_FIELDS = ("scheme", "models", *TEST_METRICS)
RECENT_PRICE_FIELDS = ("variant", "window", *TEST_METRICS)
BASE_WINDOW_FIELDS = ("history", "retrains", "mean_rows", *TEST_METRICS)


GAP_KEYS = ("mae_gap", "mae_gap_low", "mae_gap_high", "total_gap", "total_gap_low", "total_gap_high")


def fresh_data(report: dict) -> dict:
    """What the page shows of a `costmodel fresh` report: the verdicts and the numbers behind each."""
    two, worth, retrain = report["two_numbers"], report["worth_it"], report["retraining"]
    short = ("mae", "monthly_total_mape", "bias_pct")

    def months(rows: list[dict]) -> list[dict]:
        return [{"period": m["period"], "total_error_pct": m["total_error_pct"], "mae": m["mae"]} for m in rows]

    production = report.get("production")
    return {
        **{k: report[k] for k in ("window", "admissions", "known_through", "fit_rows", "draws")},
        "model": {k: report["model"][k] for k in TEST_METRICS},
        "monthly": months(report["monthly"]),
        "two_numbers": {
            "numbers": {name: {k: m[k] for k in short} for name, m in two["numbers"].items()},
            "gap": two["expected_minus_typical"],
            **{k: two[k] for k in ("per_admission", "monthly_total", "verdict")},
        },
        "worth_it": {
            "best_baseline": worth["best_baseline"],
            "baseline": {k: worth["baselines"][worth["best_baseline"]][k] for k in short},
            **{k: worth[k] for k in ("gate_passes", "gain_at_every_level", "verdict")},
            "skill": [{k: row[k] for k in ("level", "model", "global_mean", "gain_pct")} for row in worth["skill"]],
        },
        "ideas": [
            {
                **{k: idea[k] for k in ("idea", "detail", "scored_on", "expected_from_validation", "verdict")},
                **{k: idea.get(k) for k in GAP_KEYS},
                "ours": idea.get("ours") and {k: idea["ours"][k] for k in short},
                "theirs": idea.get("theirs") and {k: idea["theirs"][k] for k in short},
            }
            for idea in report["ideas"]
        ],
        "retraining": {
            "verdict": retrain["verdict"],
            "policies": [
                {
                    "policy": r["policy"],
                    "window": r["window"],
                    "label": r["policy"] if r["policy"] == "never" else f"{r['policy']} ({r['window']})",
                    "mae": r["overall"]["mae"],
                    "monthly_total_mape": r["overall"]["monthly_total_mape"],
                    "attempts": r["attempts"],
                    "promotions": r["promotions"],
                    "serves": [e["serves from"] for e in r["events"] if e["promoted"]],
                    "monthly": [{"period": m["period"], "total_error_pct": m["total_error_pct"]} for m in r["monthly"]],
                }
                for r in retrain["policies"]
            ],
        },
        "production": production
        and {**{k: production[k] for k in ("fit_until", *short)}, "monthly": months(production["monthly"])},
    }


def _read(cfg: Config, name: str, required: bool = True):
    path = cfg.reports_dir / name
    if not path.exists():
        if required:
            raise RuntimeError(f"{path} not found; run `costmodel evaluate` and `costmodel monitor` first")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def page_data(cfg: Config) -> dict:
    """Everything the page shows, taken from the reports."""
    test, monitor, replay = _read(cfg, "test.json"), _read(cfg, "monitor.json"), _read(cfg, "replay.json", required=False)
    # Optional side comparisons on the validation window (`costmodel families`, `blend`, `segments`, `recent-price`).
    recent = _read(cfg, "recent_price_valid.json", required=False) or {"rows": []}
    skill = _read(cfg, "skill_test.json", required=False) or {"rows": []}  # `costmodel skill`, on the test window
    base_window = _read(cfg, "base_window_valid.json", required=False) or {"rows": []}
    families = _read(cfg, "families_valid.json", required=False) or []
    foundation = _read(cfg, "foundation_valid.json", required=False)
    fresh = None  # `costmodel fresh`, run with another dataset's config
    if cfg.site.fresh_dataset and (path := cfg.reports_dir.parent / cfg.site.fresh_dataset / "fresh_year.json").exists():
        fresh = fresh_data(json.loads(path.read_text(encoding="utf-8")))
    blend = _read(cfg, "blend_valid.json", required=False)
    typical = _read(cfg, "typical_valid.json", required=False)
    segments = _read(cfg, "segments_valid.json", required=False) or {"schemes": []}
    if "price_shifts" not in monitor:
        raise RuntimeError("monitor.json has no price breakdown; run `costmodel monitor` again")
    from costmodel.monitor import shown_types

    site = cfg.site
    repo = site.repository.rstrip("/")
    links = (
        [
            {"label": "Code and README", "url": repo},
            {"label": "Model card", "url": f"{repo}/blob/main/MODEL_CARD.md"},
            {"label": "Architecture", "url": f"{repo}/blob/main/ARCHITECTURE.md"},
            {"label": "Generated reports", "url": f"{repo}/tree/main/reports/{cfg.name}"},
        ]
        if repo
        else []
    )
    if (cfg.reports_dir / "explore.json").exists():
        links = [EXPLORER_LINK, *links]
    months = [
        {
            "period": m["period"],
            "n": m["n"],
            "mae": m["mae"],
            "total_error_pct": m["total_error_pct"],
            "max_psi": max(v for k, v in m["psi"].items() if k != "predicted cost"),
            "most_drifted": max((k for k in m["psi"] if k != "predicted cost"), key=m["psi"].get),
            "psi_predicted": m["psi"].get("predicted cost"),
            "price_index_pct": m["price_index_pct"],
            "types": {k: {"n": v["n"], "total_error_pct": v["total_error_pct"]} for k, v in m["by_admission_type"].items()},
        }
        for m in monitor["months"]
    ]
    return {
        "name": cfg.name,
        "title": site.title or cfg.name,
        "tagline": site.tagline,
        "kicker": site.kicker or cfg.name,
        "note": site.note,
        "currency": site.currency,
        "candidate_label": site.candidate_label,
        "links": links,
        "generated": date.today().isoformat(),
        "test": {
            "version": test["candidate_version"],
            "promoted": test["promoted"],
            "reasons": test["reasons"],
            "models": [
                {"name": name, "candidate": name == "candidate", **{k: metrics[k] for k in TEST_METRICS}}
                for name, metrics in test["metrics"].items()
                if name != "production"  # the version it replaced: in the report, not one of the page's baselines
            ],
            # When the candidate gives a typical and an expected cost: each of the two on every metric.
            "numbers": [
                {"name": name, **{k: metrics[k] for k in TEST_METRICS}}
                for name, metrics in test.get("candidate_numbers", {}).items()
            ],
            # The model against a plain average, on the totals of each month for groups of different sizes.
            "skill": skill["rows"],
        },
        "monitor": {
            "version": monitor["version"],
            "reference_mae": monitor["reference_mae"],
            "rules": monitor["rules"],
            "fired": monitor["first_retrain_recommendation"],
            "types": shown_types(monitor["months"]),
            "months": months,
            "price_shifts": [{k: s[k] for k in SHIFT_FIELDS} for s in monitor["price_shifts"]],
        },
        "replay": [
            {
                "policy": r["policy"],
                "window": r["window"],
                "label": r["policy"] if r["policy"] == "never" else f"{r['policy']} ({r['window']})",
                "mae": r["overall"]["mae"],
                "monthly_total_mape": r["overall"]["monthly_total_mape"],
                "attempts": r["attempts"],
                "promotions": r["promotions"],
                "serves": [e["serves from"] for e in r["events"] if e["promoted"]],
                "monthly": [{"period": m["period"], "total_error_pct": m["total_error_pct"]} for m in r["monthly"]],
            }
            for r in (replay or [])
        ],
        # The first row of the family comparison is always the setting the search chose.
        "families": [{"candidate": i == 0, **{k: row[k] for k in FAMILY_FIELDS}} for i, row in enumerate(families)],
        "fresh": fresh,
        # `costmodel foundation`: the first row is the model in production; all rows share one sample of validation.
        "foundation": foundation
        and {
            **{k: foundation[k] for k in ("checkpoint", "context_rows", "eval_rows", "valid_rows", "train_rows")},
            "rows": [{k: row[k] for k in FOUNDATION_FIELDS} for row in foundation["rows"]],
        },
        # `costmodel typical`: the first row is today's typical cost; the others are the same setting read as a median.
        "typical": typical
        and {
            **{k: typical[k] for k in ("factor", "n_valid", "baseline", "money_share_top_decile")},
            "min_mae_gain": cfg.gate.min_mae_gain,
            "rows": [{k: row[k] for k in TYPICAL_FIELDS} for row in typical["rows"]],
        },
        "blend": blend
        and {
            "per_admission": blend["per_admission"],
            "monthly_total": blend["monthly_total"],
            "best_weight": blend["best_weight"],
            "rows": [{k: row[k] for k in BLEND_FIELDS} for row in blend["rows"]],
        },
        # The first row is the single model; each other row is one way of splitting the admissions into groups.
        "segments": [{k: row[k] for k in SEGMENT_FIELDS} for row in segments["schemes"]],
        # The first row is the candidate as it is; each other row gives it the recent billed price in some way.
        "recent_price": [{k: row[k] for k in RECENT_PRICE_FIELDS} for row in recent["rows"]],
        # The first row is the candidate fitted once; each other row is retrained before every block of months.
        "base_window": {
            "horizon": base_window.get("horizon"),
            "rows": [{k: row[k] for k in BASE_WINDOW_FIELDS} for row in base_window["rows"]],
        },
    }


def _clean(value):
    """NaN and infinity are not JSON: the page gets ``null`` instead."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def render(data: dict) -> str:
    # "</" inside the embedded JSON would end the <script> block early.
    payload = json.dumps(_clean(data), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    for key in ("title", "tagline"):
        html = html.replace(f"__{key.upper()}__", _escape(data[key]))
    return html.replace("__DATA__", payload)


def explorer_data(cfg: Config) -> dict | None:
    """What the explorer page shows (``costmodel explore``), or None if that has not run."""
    explore = _read(cfg, "explore.json", required=False)
    if explore is None:
        return None
    from costmodel.explore import MIN_ADMISSIONS, TOP_GROUPS

    site = cfg.site
    repo = site.repository.rstrip("/")
    return {
        "title": site.title or cfg.name,
        "kicker": site.kicker or cfg.name,
        "note": site.note,
        "currency": site.currency,
        "links": (
            [{"label": "Code and README", "url": repo}, {"label": "Model card", "url": f"{repo}/blob/main/MODEL_CARD.md"}]
            if repo
            else []
        ),
        "generated": date.today().isoformat(),
        "top_groups": TOP_GROUPS,
        "min_admissions": MIN_ADMISSIONS,
        "explore": explore,
    }


def _between(text: str, start: str, end: str) -> str:
    return text[text.index(start) + len(start) : text.index(end)]


def render_explorer(data: dict) -> str:
    """The explorer page: its own layout, with the styles and the chart code of the main page."""
    main = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(_clean(data), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    html = EXPLORE_TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("__STYLE__", _between(main, "<style>\n", "</style>"))
    html = html.replace("__SHARED_JS__", _between(main, *SHARED_JS))
    return html.replace("__TITLE__", _escape(data["title"])).replace("__DATA__", payload)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def run(cfg: Config) -> Path:
    path = HOME / "docs" / "index.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(page_data(cfg)), encoding="utf-8", newline="\n")
    print(f"page written to {path}")
    explorer = explorer_data(cfg)
    if explorer is not None:
        explore_path = path.with_name("explore.html")
        explore_path.write_text(render_explorer(explorer), encoding="utf-8", newline="\n")
        print(f"explorer written to {explore_path}")
    return path
