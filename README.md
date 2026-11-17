# hospital-cost-forecast

Predicts the cost of a hospital admission at the moment of admission, using public SUS data. Built like production:
honest baselines, explicit leakage controls, time-based validation, a registered model behind an API, and monthly
drift monitoring that shows when the model needs retraining.

The pipeline does not depend on SUS: it reads one canonical admissions table defined by a
[data contract](DATA_CONTRACT.md). Public SIH/SUS data (state of Rio de Janeiro, 2018–2024, 5.4 million admissions) is
the first source; a health plan can fill the same table with its own claims and run the same commands.

![The production model, month by month through 2024](reports/sihsus_rj/monitor.png)

## Results

**Test year (2023), scored once.** Every model was fitted on 2018–2022 and scored on the same 877,760 admissions.

| Model | MAE per admission (R$) | Error on the month's total | Bias |
|---|---:|---:|---:|
| **LightGBM (in production)** | **778.62** | 11.32% | −11.42% |
| Median by procedure × age band × admission type (business rule) | 872.80 | 40.81% | −40.89% |
| Median by procedure | 886.29 | 41.53% | −41.61% |
| Linear regression on log(cost) | 937.81 | 12.89% | −12.89% |
| Mean by procedure × age band × admission type | 972.10 | 13.44% | −13.57% |
| Mean by procedure | 1,003.53 | 13.50% | −13.63% |
| Global mean | 1,579.64 | **7.15%** | −7.36% |

- The model beats the strongest baseline by 10.8% per admission and passed the promotion gate.
- **A global mean has a smaller error on the month's total.** That comes from mix effects cancelling out, not skill,
  since it misses by R$ 1,580 per admission. It is in the table because it is the honest comparison for anyone who
  only needs the total.
- Medians win per admission but miss the month's total by 40%, because SUS costs are skewed (the 10% most expensive
  admissions take 57% of the money). That is why the gate checks both metrics.

**Monitoring (2024, simulated production).** The model under-predicts the month's total by about 14% early in 2024
and by 20–24% from July. The retrain rule fires at the end of September, as soon as the bad months have been billed.
Input drift (PSI) stayed flat while the error jumped: the admissions looked the same, but the prices paid for them
went up.

**Retraining policies, replayed over 2024** with only the data known at each moment:

| Policy | MAE (R$) | Error on the month's total |
|---|---:|---:|
| Never retrain | 904.69 | 19.03% |
| Retrain when the rule fires | 905.55 | 16.72% |
| Try to retrain every month (each candidate still gated) | 898.11 | 11.75% |

Retrains use all data known so far; using only the last 24 months gives nearly the same numbers.

![Retraining policies over 2024](reports/sihsus_rj/replay.png)

Details: [model card](MODEL_CARD.md) (intended use, segments, limitations), and the generated reports in
[`reports/sihsus_rj/`](reports/sihsus_rj/).

## Quickstart

Needs [uv](https://docs.astral.sh/uv/) and about 1.5 GB of disk. From a clone of this repository:

```bash
uv sync
uv run costmodel run      # download SIH/SUS RJ (~400 MB, checksummed), validate, train, gate, monitor
uv run costmodel serve    # API on http://127.0.0.1:8000/docs
```

`costmodel run` takes about half an hour on a desktop PC. The same with Docker, writing everything to `./docker-work`:

```bash
docker compose run --rm costmodel run
```

```bash
docker compose up api
```

### Commands

| Command | What it does |
|---|---|
| `costmodel data` | Ingest the configured source, check it against the contract, save the admissions table |
| `costmodel eda` / `baselines` | Exploratory report; fit and compare the baselines on validation |
| `costmodel train` | Search the candidate's settings on validation, refit, register it in MLflow as `staging` |
| `costmodel evaluate` | Score `staging` once on test, run the gate, promote it to `production` if it passes |
| `costmodel monitor` | Score the monitoring months: PSI per input, error once billed, and the retrain rule |
| `costmodel replay` | What-if of the retraining policies over the monitoring months (slow: it retrains) |
| `costmodel score --month 2024-03` | Batch: price a month of admissions with `production` |
| `costmodel serve` | HTTP API: `POST /predict`, `GET /health` |
| `costmodel run` | `data`, `train`, `evaluate`, `monitor`, in order, stopping at the first failure |
| `costmodel validate file.csv` | Check a file against the data contract and list every problem |

Every command takes `--config configs/<dataset>.toml` (default: the SIH/SUS demo). Runs are manual; nothing is
scheduled. Experiments and the model registry can be browsed with
`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000`.

### Use your own admissions

1. Fill [`templates/admissions_template.csv`](templates/admissions_template.csv) following
   [DATA_CONTRACT.md](DATA_CONTRACT.md), and check it: `uv run costmodel validate my_admissions.csv`.
2. Copy [`configs/example_file.toml`](configs/example_file.toml), point it at the file and set the windows.
3. `uv run costmodel run --config configs/my_plan.toml`.

To try this path without real data, `costmodel synthetic` writes made-up plan admissions (TUSS codes) and
[`configs/synthetic_plan.toml`](configs/synthetic_plan.toml) runs the pipeline on them. A model trained on SUS prices
does not transfer to a private plan; the pipeline does.

## How it works

```
 SIH/SUS (DATASUS) ─► adapter ─┐
                               ├─► admissions table ─► contract ─► time split ─► features ─► baselines + LightGBM
 your CSV / Parquet ───────────┘     (canonical)        (pandera)                (allow-list)          │
                                                                                                       ▼
        monitor: PSI now, error after billing ◄── API + batch scoring ◄── gate ◄── MLflow registry ('staging')
                       │                                                    │
                       └─► "retraining recommended" ─► a person retrains ───┘ (same gate)
```

More in [ARCHITECTURE.md](ARCHITECTURE.md).

## Design decisions

- **A data contract between the source and the model.** Everything after ingestion reads one table, checked by
  pandera before anything is trained; a bad file fails with a list of every problem. SIH/SUS is one adapter, so other
  data needs a CSV, not code.
- **Prediction at admission, enforced in code.** Features are built from an allow-list of columns known at admission.
  Length of stay, ICU days, performed procedure, death and discharge stay in the table for analysis, and a test fails
  the build if any of them changes the features.
- **Time-based validation.** Train 2018–2021, choose on 2022, test once on 2023, then replay 2024 as production. A
  random split would mix future prices into training and hide the drift that matters most here.
- **A strong business-rule baseline.** The median by procedure, age band and admission type is what an analyst would
  build by hand; the requested procedure alone explains 65% of the variance of log(cost). The model has to beat it.
- **Two metrics, because money is counted in totals.** MAE per admission, and the error on each month's total. The
  gate requires a better MAE than every reference (by 2%), a month's-total error no worse than theirs, and never above
  15%. If the candidate fails and nothing is in production, the best baseline is put in production instead.
- **Time as an input.** Trees cannot extrapolate a price trend. Giving the model the admission date as a number made
  it apply the latest price level to new admissions, cutting the validation error on the month's total from 14–17% to
  10–15% at the same MAE.
- **Billing delay is part of monitoring.** Costs are known only once billed: 3 months after the admission month covers
  98.7% of admissions but 95.4% of the money, because expensive stays bill later. A month is judged only then;
  judging earlier would understate the error.
- **The monitor recommends; a person retrains.** Retraining changes what is in production, so the trigger is human
  and the new model goes through the same gate. The replay of 2024 suggests the rule reacts late; the rule was fixed
  before looking at 2024 and is left unchanged rather than fitted to it.
- **The registered model builds its own features.** It takes raw contract columns, so the API and the batch job
  cannot prepare inputs differently from training. Both load one exact version and report it with every prediction;
  rolling back is moving the `production` alias.
- **SIH/SUS cleaning rules,** each checked on the real files: only regular AIHs (`IDENT = 1`; type 5 are monthly
  continuation bills of long stays), one row per AIH, amount paid above zero, dated by admission rather than billing
  month, and billing files read until 2025-06 so that late bills of 2024 are complete.

## Limitations

- **SUS reimbursement, not cost of care,** and for Rio de Janeiro only. The target is what the SUS paid.
- **It under-predicts when prices rise:** by 11% on 2023 and up to 24% in late 2024 without retraining.
- **Individual predictions for the most expensive admissions are low:** the top decile is under-predicted by about
  38% on average. What happens during the stay is not known at admission.
- **The requested procedure is assumed to be as known at admission.** Public files hold the AIH as billed; if the
  field was amended later, the measured error is somewhat optimistic.
- **Not for decisions about patients, people or hospitals.** See the [model card](MODEL_CARD.md) for what it must not
  be used for.
- **The API is for local or internal use:** no authentication or rate limiting.

## Project

Python 3.12, uv, LightGBM, pandera, MLflow (SQLite), FastAPI. Lint with ruff, tests with pytest; CI runs a secret scan
(gitleaks), lint and the test suite, both directly and inside the Docker image. Only public data is used; raw data is never
committed, only its checksums.

MIT License.
