# hospital-cost-forecast

Predicts the cost of a hospital admission at the moment of admission, using public SUS data. Built like production:
honest baselines, explicit leakage controls, time-based validation, a registered model behind an API, and monthly
drift monitoring that shows when the model needs retraining.

The pipeline does not depend on SUS: it reads one canonical admissions table defined by a
[data contract](DATA_CONTRACT.md). Public SIH/SUS data (state of Rio de Janeiro, 2018–2024, 5.4 million admissions) is
the first source; a health plan can fill the same table with its own claims and run the same commands.

**[Results page](https://mauryneto.github.io/hospital-cost-forecast/)**: the numbers below as interactive charts.

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

**Why LightGBM: other model families on the validation year (2022).** Same training admissions, same inputs, same
two metrics ([families_valid.md](reports/sihsus_rj/families_valid.md)):

| Model family | Setting | MAE per admission (R$) | Error on the month's total | Time to fit |
|---|---|---:|---:|---:|
| **LightGBM (gradient boosting)** | least squares on log(cost); chosen among 16 settings | **792.15** | 14.98% | 75 s |
| Random forest | on log(cost) | 835.15 | 18.34% | 220 s |
| CatBoost | Tweedie | 837.15 | 11.28% | 40 min |
| CatBoost | least squares on log(cost) | 838.64 | 15.76% | 30 min |
| Random forest | on the cost | 841.43 | **9.09%** | 150 s |

- No other family came close per admission: all four are 5–6% worse than LightGBM.
- The error on the month's total follows what the model is fitted on more than the family. Models fitted on the cost
  itself miss the total by less (9–11%) and each admission by more, and LightGBM does the same when given a Tweedie
  objective with the same inputs (10.69% to 11.46% at an MAE of 800 to 811). The random forest on the cost has the lowest total error of all.
- The comparison is not even: LightGBM's row is the best of 16 settings and each other family got two, and CatBoost
  was still improving when it reached its limit of 1,000 trees. More tuning would narrow the gap; at 30 to 40 minutes
  per fit against 75 seconds, it was not pursued.
- Nothing was promoted from this comparison, and the test year was not looked at again.

**Averaging two settings of LightGBM** ([blend_valid.md](reports/sihsus_rj/blend_valid.md)). The setting chosen for the
error per admission (least squares on log(cost)) and the one with the lowest error on the month's total in the search
(Tweedie), fitted as in the search and averaged, on the same validation year:

| Share of the Tweedie model | MAE per admission (R$) | Error on the month's total |
|---|---:|---:|
| 0% (the chosen setting alone) | 792.15 | 14.98% |
| 25% | **787.00** | 13.78% |
| 50% | 787.47 | 12.58% |
| 75% | 793.89 | 11.52% |
| 100% (the Tweedie setting alone) | 807.86 | **10.69%** |

Half and half is better than the chosen setting on both metrics: by 0.6% per admission and by 2.4 points on the
month's total. The model in production was not changed. The gate asks a new model to be 2% better per admission than
the one in production, and this gain is mostly on the total, so as the rule stands the average would not replace it
(see the roadmap).

**Monitoring (2024, simulated production).** The model under-predicts the month's total by about 14% early in 2024
and by 20–24% from July. The retrain rule fires at the end of September, as soon as the bad months have been billed.
Input drift (PSI) stayed flat while the error jumped: the admissions looked the same, but the prices paid for them
went up.

**What happened in July 2024: a price change in elective surgery.** The monitor compares what was paid with what the
same admissions would cost at the prices the model last saw (mean per procedure in 2022, same mix):

| | January–April | May–June | July–December |
|---|---:|---:|---:|
| Prices against the level the model last saw | +7.91% to +9.12% | +11.08% to +13.07% | +16.90% to +21.89% |
| Error on the month's total, urgent admissions | −12.63% to −14.20% | −13.13% to −14.37% | −12.85% to −16.58% |
| Error on the month's total, elective admissions | −13.79% to −17.75% | −20.15% to −21.35% | −28.25% to −34.10% |

Urgent admissions were paid as before; elective ones were not. The median paid for cataract surgery
(phacoemulsification, SIGTAP 0405050372) was R$ 772 in every month up to June and R$ 3,858 from July; for laparoscopic
cholecystectomy (0407030034) it went from R$ 1,057 to between R$ 1,397 and R$ 2,026. A model trained on earlier prices
cannot know this, and no input drift shows it. The rule behind the change was not identified; a payment complement
for elective surgery is the likely kind of cause.

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
| `costmodel monitor` | Score the monitoring months: PSI per input, error once billed, where the error comes from (admission type, price level), and the retrain rule |
| `costmodel replay` | What-if of the retraining policies over the monitoring months (slow: it retrains) |
| `costmodel score --month 2024-03` | Batch: price a month of admissions with `production` |
| `costmodel serve` | HTTP API: `POST /predict`, `GET /health` |
| `costmodel families` | Compare other model families (random forest, CatBoost) with the chosen candidate on validation; CatBoost needs `uv sync --extra families` |
| `costmodel blend` | Average the candidate's best per-admission and best month's-total settings, on validation |
| `costmodel site` | One static page with the results ([`docs/index.html`](docs/index.html)), built from the reports |
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
- **It under-predicts when prices rise:** by 11% on 2023 and up to 24% in late 2024 without retraining. A change
  in payment rules, like the one for elective surgery in July 2024, is seen only after the bills arrive.
- **Individual predictions for the most expensive admissions are low:** the top decile is under-predicted by about
  38% on average. What happens during the stay is not known at admission.
- **The requested procedure is assumed to be as known at admission.** Public files hold the AIH as billed; if the
  field was amended later, the measured error is somewhat optimistic.
- **Not for decisions about patients, people or hospitals.** See the [model card](MODEL_CARD.md) for what it must not
  be used for.
- **The API is for local or internal use:** no authentication or rate limiting.

## Roadmap

Not done yet, in the order I would do them:

- **Prediction intervals.** A range ("between X and Y") next to each predicted cost, from quantile models or conformal
  prediction, with its coverage checked on the test year. It matters most for the expensive admissions, where a single
  number is least reliable.
- **A tabular foundation model in the results table.** Models such as NVIDIA's Kumo Tabular or TabPFN predict without
  training, from examples given as context. Scored on the same windows and through the same gate, whichever way it goes
  is a result. The open question is scale: they take tens of thousands of context rows, and this training set has 3.6
  million rows with thousands of procedure, diagnosis and hospital codes.
- **The average of two settings as a candidate.** On validation it is a little better per admission and clearly
  better on the month's total. Serving it means packaging the two models as one and passing the gate, and the gate
  would have to weigh the month's total (the retraining policy item below) for the gain to count.
- **Kinds of prediction side by side on the results page:** point estimate, interval and month's total, per model.
- **The retraining policy, tested on a new period.** The 2024 replay suggests retraining whenever a new billed month
  exists, and a gate that weighs the month's total. Changing the rule on the same year that suggested it would be
  fitting it to the result; it needs 2025 data.
- **The rule behind the July 2024 price change,** and whether known payment-rule changes can be given to the model as
  an input.
- **Other states,** to see how much of the model is specific to Rio de Janeiro.

## How this was built

The code was written with [Claude Code](https://claude.com/claude-code), an AI coding assistant, and every commit
says so. The direction, the questions along the way and what stayed in were the author's.
[DECISIONS.md](DECISIONS.md) lists the decisions that shaped the project and where each one came from.

## Project

Python 3.12, uv, LightGBM, pandera, MLflow (SQLite), FastAPI (CatBoost is optional, for the family comparison); the results page is plain HTML, CSS and JavaScript. Lint with ruff, tests with pytest; CI runs a secret scan
(gitleaks), lint and the test suite, both directly and inside the Docker image. Only public data is used; raw data is never
committed, only its checksums.

MIT License.
