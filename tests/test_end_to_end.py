"""Run the real command line, in a separate process and a throwaway work folder, on synthetic data in the contract format.

This is the path a new user follows: fill the template (here: generate it), validate, build the table, explore, baselines,
then train a candidate, register it, and let the gate decide on the test window.
"""

import json
import os
import subprocess
import sys

import pytest


def costmodel(home, *args):
    env = {**os.environ, "COSTMODEL_HOME": str(home), "MPLBACKEND": "Agg"}
    result = subprocess.run([sys.executable, "-m", "costmodel.cli", *args], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.slow
def test_file_source_pipeline_end_to_end(tmp_path):
    config = tmp_path / "plan.toml"
    config.write_text(
        'name = "e2e"\n[source]\ntype = "file"\npath = "data/admissions.csv"\n'
        '[split]\nmode = "relative"\nmonitor_months = 6\ntest_months = 6\nvalid_months = 6\n'
        '[train]\nobjectives = ["tweedie", "log_l2"]\nnum_leaves = [15]\nsliding_months = [12]\nmin_category_count = 5\n'
        "[gate]\nmax_monthly_total_mape = 50.0\n"
    )
    costmodel(
        tmp_path, "synthetic", "--out", "data/admissions.csv", "--n", "20000", "--start", "2022-01-01", "--end", "2024-12-31"
    )
    assert "OK: 20,000 admissions" in costmodel(tmp_path, "validate", str(tmp_path / "data" / "admissions.csv"))
    costmodel(tmp_path, "data", "--config", str(config))
    costmodel(tmp_path, "eda", "--config", str(config))
    costmodel(tmp_path, "baselines", "--config", str(config))

    reports = tmp_path / "reports" / "e2e"
    for name in ("eda.md", "monthly_volume_and_cost.png", "cost_distribution.png", "baselines_valid.md"):
        assert (reports / name).exists(), name
    results = {r["model"]: r for r in json.loads((reports / "baselines_valid.json").read_text())}
    # The synthetic cost is driven by the procedure price, so the procedure rule must beat the global mean.
    assert results["median_by_proc"]["mae"] < results["global_mean"]["mae"]
    assert results["log_linear"]["mae"] < results["global_mean"]["mae"]

    # Candidate: searched on validation (2 objectives x 2 windows), refitted, registered, judged once on test.
    costmodel(tmp_path, "train", "--config", str(config))
    assert len(json.loads((reports / "train_valid.json").read_text())) == 4
    out = costmodel(tmp_path, "evaluate", "--config", str(config))
    test = json.loads((reports / "test.json").read_text())
    assert {"candidate", "global_mean", "median_by_proc"} <= set(test["metrics"])
    assert test["metrics"]["candidate"]["mae"] < test["metrics"]["global_mean"]["mae"]
    assert ("NOT PROMOTED" not in out) == test["promoted"]
    assert (tmp_path / "mlflow.db").exists()
    # Either way something is now in production: the candidate, or the best baseline as a fallback.
    assert "PROMOTED: " in out or "the best baseline" in out
    again = costmodel(tmp_path, "evaluate", "--config", str(config))
    if test["promoted"]:
        assert "already in production" in again  # the same version cannot be promoted twice
    else:
        assert "stays in production" in again  # the candidate now also has to beat the baseline in production

    # The one-command pipeline: rebuilds the table, trains a new version, and judges it against production.
    out = costmodel(tmp_path, "run", "--config", str(config))
    assert "=== data ===" in out and "=== evaluate ===" in out
    assert "stays in production" in out or "PROMOTED: " in out
