"""The results page: built from the reports only, with nothing left unfilled and nothing but aggregates inside."""

import json
import re

import pytest

from costmodel import config as config_module
from costmodel import site
from costmodel.config import Config, SiteConfig

MONTHS = ["2024-01", "2024-02", "2024-03"]


def month(period, error, price):
    return {
        "period": period,
        "n": 100.0,
        "actual": 1000.0,
        "predicted": 900.0,
        "mae": 50.0,
        "total_error_pct": error,
        "psi": {"requested_procedure": 0.2, "age_years": 0.01, "predicted cost": 0.9},
        "breaches": [],
        "by_admission_type": {
            "urgent": {"n": 70.0, "actual": 600.0, "predicted": 560.0, "total_error_pct": error / 2},
            "elective": {"n": 29.0, "actual": 390.0, "predicted": 330.0, "total_error_pct": error * 2},
            "other": {"n": 1.0, "actual": 10.0, "predicted": 10.0, "total_error_pct": 0.0},
        },
        "price_index_pct": price,
        "price_covered_pct": 99.0,
    }


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(site, "HOME", tmp_path)
    cfg = Config(
        name="site_test",
        source={"type": "file"},
        site=SiteConfig(title="A <title> & more", tagline="What it is", repository="https://example.org/repo/"),
    )
    cfg.reports_dir.mkdir(parents=True)
    metrics = {"mae": 10.0, "bias_pct": -5.0, "mae_top_decile": 80.0, "monthly_total_mape": 6.0}
    test = {
        "candidate_version": 3,
        "promoted": True,
        "reasons": ["MAE 10.00 is 50.0% below best baseline"],
        "metrics": {"candidate": metrics, "global_mean": metrics | {"mae": 20.0, "monthly_total_mape": 4.0}},
    }
    monitor = {
        "version": "3",
        "reference_mae": 10.0,
        "rules": {"label_delay_months": 3, "mae_tolerance": 0.1, "patience": 2, "max_monthly_total_error": 15.0},
        "months": [month(p, e, x) for p, e, x in zip(MONTHS, [-5.0, -6.0, -20.0], [1.0, 2.0, float("nan")], strict=True)],
        "price_shifts": [
            {"code": "0001", "n": 30, "paid": 9000.0, "at_reference_prices": 3000.0, "extra": 6000.0, "predicted": 3100.0,
             "reference_median": 100.0, "monthly_median": {"2024-01": 100.0, "2024-02": 100.0, "2024-03": 700.0}}
        ],  # fmt: skip
        "timeline": [],
        "first_retrain_recommendation": "2024-02",
    }
    (cfg.reports_dir / "test.json").write_text(json.dumps(test), encoding="utf-8")
    (cfg.reports_dir / "monitor.json").write_text(json.dumps(monitor), encoding="utf-8")
    return cfg


def embedded(html: str) -> dict:
    return json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1))


def test_page_is_filled_from_the_reports(cfg):
    html = site.run(cfg).read_text(encoding="utf-8")
    assert not re.search(r"__[A-Z]+__", html)  # no placeholder left
    assert "<title>A &lt;title&gt; &amp; more</title>" in html
    data = embedded(html)
    assert [m["period"] for m in data["monitor"]["months"]] == MONTHS
    assert data["monitor"]["types"] == ["urgent", "elective"]  # 'other' is 1% of the admissions: left out
    assert data["monitor"]["months"][2]["price_index_pct"] is None  # NaN is not JSON
    # The predicted cost's own PSI is reported apart from the inputs' largest PSI.
    assert data["monitor"]["months"][0]["max_psi"] == 0.2
    assert data["monitor"]["months"][0]["most_drifted"] == "requested_procedure"
    assert [m["name"] for m in data["test"]["models"] if m["candidate"]] == ["candidate"]
    assert data["replay"] == []  # replay.json is optional
    assert data["links"][0]["url"] == "https://example.org/repo"


def test_page_embeds_the_replay_when_there_is_one(cfg):
    def policy(name, window, total, events):
        return {
            "policy": name,
            "window": window,
            "overall": {"mae": 12.0, "monthly_total_mape": total},
            "attempts": len(events),
            "promotions": sum(e["promoted"] for e in events),
            "waits": 0,
            "events": events,
            "monthly": [{"period": p, "mae": 12.0, "total_error_pct": -total} for p in MONTHS],
        }

    events = [{"promoted": True, "serves from": "2024-02"}, {"promoted": False, "serves from": "-"}]
    replay = [policy("never", "-", 9.0, []), policy("monthly", "expanding", 5.0, events)]
    (cfg.reports_dir / "replay.json").write_text(json.dumps(replay), encoding="utf-8")
    data = embedded(site.run(cfg).read_text(encoding="utf-8"))
    assert [r["label"] for r in data["replay"]] == ["never", "monthly (expanding)"]
    assert data["replay"][1]["serves"] == ["2024-02"]


def test_page_embeds_the_side_comparisons_when_they_exist(cfg):
    data = embedded(site.run(cfg).read_text(encoding="utf-8"))
    assert data["families"] == [] and data["blend"] is None  # both are optional

    metrics = {"mae": 10.0, "rmse": 30.0, "bias_pct": -5.0, "mae_top_decile": 80.0, "monthly_total_mape": 6.0}
    families = [
        {"family": "LightGBM (gradient boosting)", "setting": "chosen", "trees": 100, "seconds": 5.0, **metrics},
        {"family": "random forest", "setting": "on the cost", "trees": 300, "seconds": 9.0, **metrics | {"mae": 11.0}},
    ]
    blend = {"per_admission": "a", "monthly_total": "b", "best_weight": 0.5,
             "rows": [{"weight": w, **metrics} for w in (0.0, 0.5, 1.0)]}  # fmt: skip
    (cfg.reports_dir / "families_valid.json").write_text(json.dumps(families), encoding="utf-8")
    (cfg.reports_dir / "blend_valid.json").write_text(json.dumps(blend), encoding="utf-8")
    data = embedded(site.run(cfg).read_text(encoding="utf-8"))
    assert [(r["family"], r["candidate"]) for r in data["families"]] == [
        ("LightGBM (gradient boosting)", True),  # the first row is the setting the search chose
        ("random forest", False),
    ]
    assert "rmse" not in data["families"][0]  # only what the page shows is embedded
    assert data["blend"]["best_weight"] == 0.5 and [r["weight"] for r in data["blend"]["rows"]] == [0.0, 0.5, 1.0]


def test_embedded_text_cannot_close_the_script_block(cfg):
    html = site.render(site.page_data(cfg) | {"note": "</script><b>x"})
    assert html.count("</script>") == 2  # the data block and the page's own script
    assert embedded(html)["note"] == "</script><b>x"


def test_page_needs_the_reports(cfg):
    (cfg.reports_dir / "monitor.json").unlink()
    with pytest.raises(RuntimeError, match="costmodel monitor"):
        site.run(cfg)
    (cfg.reports_dir / "monitor.json").write_text(json.dumps({"months": []}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="run `costmodel monitor` again"):
        site.run(cfg)
