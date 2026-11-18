# Architecture

The project is a pipeline that is **independent of where the data comes from**. Everything downstream of ingestion reads
one canonical table, defined by a data contract ([DATA_CONTRACT.md](DATA_CONTRACT.md)). Public SIH/SUS data is just the
first source: an adapter downloads it and fills that table. Anyone else (for example a health plan with its own claims)
fills the same table from a CSV, runs the same commands, and gets a model trained, validated, gated and monitored on
their own data.

What transfers is the **pipeline**, not the model. A model trained on SUS prices does not apply to a private plan
(different procedure codes, TUSS instead of SIGTAP, and contract prices). Each dataset is trained on its own.

```
 sources                     contract                 modelling                         operation
 ───────                     ────────                 ─────────                         ─────────
 SIH/SUS FTP ─► sihsus adapter ─┐
                                ├─► admissions table ─► validate ─► split ─► features ─► baselines + candidates
 your CSV / Parquet ────────────┘    (canonical)         (pandera)   (time)   (allow-list)        │
                                                                                                  ▼
                                    monitor (PSI, monthly MAE) ◄── serve (FastAPI) ◄── gate + registry (MLflow)
                                          │                                                       ▲
                                          └──── retrain rule fires ──► retrain on recent data ────┘
```

## Layers

| Layer | Module | Contract with the next layer |
|---|---|---|
| Sources | `costmodel.sources.sihsus` (download with checksums, `.dbc` reader, mapping) | Produces a DataFrame in the canonical schema. Source-only columns go in `src_*`. |
| Contract | `costmodel.contract` | `validate(df)` fails with a readable report on a missing column, wrong type or absurd value. |
| Split | `costmodel.split` | Time-based train / valid / test / monitor windows from the run config, by **admission date**. |
| Features | `costmodel.features` | The only place that decides what the model sees. Reads admission-time columns only. |
| Models | `costmodel.baselines`, `costmodel.candidate` (LightGBM) | `fit(X, y)` / `predict(X)` on the feature frame. |
| Training | `costmodel.train` | Searches the candidate's settings on validation, refits, registers it as `staging`. |
| Evaluation | `costmodel.evaluate` | MAE, RMSE, error by cost decile, error on the month's total, slices. |
| Gate + registry | `costmodel.gate`, `costmodel.holdout`, `costmodel.registry` + MLflow (SQLite) | Scores `staging` once on test; moves the `production` alias only if it passes the gate. |
| Serving | `costmodel.score`, `costmodel.api` | One exact `production` version prices admissions: a month in batch (`costmodel score`) or 1 to 1,000 per request (`POST /predict`). Both check the contract's admission-time columns the same way. |
| Monitoring | `costmodel.drift`, `costmodel.monitor`, `costmodel.replay` | Monthly PSI per input and error of the production model; the retrain rule's recommendation; a replay of retraining policies. |

## The pipeline, end to end

Steps 1 to 6 run from one command, by hand (nothing is scheduled):

```
uv run costmodel run --config configs/sihsus_rj.toml        # public demo
uv run costmodel run --config configs/my_plan.toml          # your own admissions table
```

1. **Ingest.** The config names a source: `sihsus` (download + map) or `file` (a CSV/Parquet already in the canonical
   format, e.g. filled from `templates/admissions_template.csv`).
2. **Validate.** The contract is checked before anything is trained. A failure stops the run and lists every problem
   (`uv run costmodel validate my_file.csv` runs this step alone).
3. **Split by time.** Train on the past, choose on validation, report once on test, then replay the most recent months
   as simulated production. Windows are either explicit dates or relative to the last month in the data, so a dataset
   with a different date range needs no code change.
4. **Train** (`costmodel train`). Every combination of training window (expanding, or only the last N months) x
   objective x tree size is fitted on the training window and measured on validation, each as an MLflow run. The best
   one is refitted on training + validation and registered in the MLflow model registry with the alias `staging`.
   `costmodel families` is a side check, outside `costmodel run`: it fits other model families (random forest,
   CatBoost) on the same windows and features and reports them next to the chosen setting. It reads only the
   validation window and registers nothing; a family that won there would enter the search as a candidate and face
   the same gate. `costmodel blend` is the same kind of check for an average of two settings from the search: the one
   chosen for the error per admission and the one with the lowest error on the month's total. `costmodel segments`
   asks whether one model per group of admissions (by admission type, by procedure group) does better than one model
   for all of them. `costmodel recent-price` gives the model the mean paid for each procedure in the last billed
   months, built only from bills already known at admission (the monitor's billing delay decides which).
   `costmodel base-window` retrains before every block of three months with histories of different lengths (the
   last 4, 12 or 24 billed months, all of them, all of them with recent months weighing more).
   `costmodel foundation` puts a tabular foundation model (TabPFN-2, the checkpoint with open weights) next to the
   candidate. It is not fitted: it reads 10,000 admissions as context and prices a random sample of the validation
   window, on which LightGBM is scored too, both with all the training rows and with those same 10,000. It needs
   `uv sync --extra foundation` (PyTorch) and a GPU.
5. **Gate** (`costmodel evaluate`). The `staging` model is loaded back from the registry and scored on the test
   window, next to every baseline refitted on the same data and the current `production` model, if any. Each of its
   two numbers is judged by the metric it is for: the typical cost by the MAE, the expected cost by the error on the
   month's total. It is promoted (the `production` alias moves to it) only if it beats the best baseline on MAE by
   the configured margin **and** is not worse than it on the monthly total; if, against the model in production, at
   least one of the two numbers is better by its margin and the other is not worse; and if its monthly-total error
   stays under an absolute ceiling. If it fails and nothing is in production yet, the best baseline is registered
   and put in production, so there is always a model to serve. The decision and its reasons go to the model version's
   tags, an MLflow run and `reports/<dataset>/test.md`. `costmodel skill` then answers "is it worth it over a plain
   average?" for the model in production: the error on each month's totals for all admissions, and for each hospital,
   specialty, procedure group and so on, next to a plain average's.
6. **Monitor** (`costmodel monitor`). The production model prices the monitoring months one by one. Data drift (PSI of
   each input, and of the predicted cost, against the data the model was fitted on) is known as soon as admissions
   arrive. Performance (the MAE of the typical cost, the error of the expected cost on the month's total) is known
   only once costs are billed, so a month is judged only
   `label_delay_months` after it ends (3 for SIH/SUS: 98.7% of admissions and 95.4% of the money are billed by then;
   expensive stays bill later, so judging earlier would understate the error). When the error grows, the report
   also says where: the error by admission type, and the price level, which compares what was paid with what the same
   admissions would cost at the prices the model last saw (same mix of procedures, so only price counts).
7. **Retrain rule.** When the last `patience` mature months all have an MAE more than `mae_tolerance` above the model's
   test MAE, or a monthly-total error beyond the gate's ceiling, the monitor recommends retraining. It does not retrain
   by itself: a person runs `costmodel run` with the new data, and the new candidate goes through the same gate.
   `costmodel replay` shows what each policy (never, by the rule, every month; expanding or sliding window) would have
   done over the monitoring window, using only data known at each moment.
8. **Serve.** `costmodel score --month YYYY-MM` prices a month of admissions (from the table or from a file without
   the cost) and writes one file per month to `data/predictions/<dataset>/`, with the model version on every row.
   `costmodel serve` starts the HTTP API (`POST /predict`, `GET /health`, docs at `/docs`). Both return a
   `typical_cost` and an `expected_cost` for every admission (the same number twice, if the model in production gives
   only one: a baseline, or a version registered before the two numbers existed). Both load one exact model
   version, not the alias, so the version they report is always the one that priced the admission; a promotion or a
   rollback reaches the API when it restarts. The API does not start without a `production` model, and it rejects any
   field the contract does not know at admission (the cost, the length of stay) instead of ignoring it.
9. **Show.** `costmodel explore` adds up, group by group and month by month (all admissions, admission type,
   procedure group, bed specialty, hospital), what the model in production predicted and what was paid, over the test
   and monitoring windows; small groups are pooled and no admission is listed. `costmodel site` then writes that as
   `docs/explore.html`, a screen with filters, next to the main page.
   `costmodel site` writes one static page (`docs/index.html`) from the JSON reports: the test table, the
   monitoring months, where prices changed, the retraining replay and, when they were run, the side comparisons
   (other model families, a tabular foundation model, the blend, one model per group, the recent price, the length
   of the history). The numbers
   are embedded when the page is generated, so it cannot disagree with the reports, and only aggregates are published.

## Leakage control is part of the contract

The contract separates columns known **at admission** from columns known only **after** (length of stay, ICU days,
performed procedure, death, discharge). Post-admission columns are allowed in the table (they are useful for analysis
and for leakage tests) but `costmodel.features` can only read admission-time columns, and a test fails the build if
that ever changes.

## Running it

Locally with `uv run costmodel <command>`, or in Docker with `docker compose run --rm costmodel <command>` (data and
reports go to `./docker-work`). `docker compose run --rm tests` runs lint and the whole test suite in the image.

MLflow keeps runs and the registry in `mlflow.db` in the work folder. To browse them:
`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000` (or `docker compose up mlflow`),
then open http://127.0.0.1:5000. Rolling back is moving the `production` alias to an earlier version; versions are
never deleted.

The API: `uv run costmodel serve` (or `docker compose up api`), then http://127.0.0.1:8000/docs. After moving the
`production` alias, restart it.

To try the "your own data" path without real data, `costmodel synthetic` writes made-up admissions in the canonical
format (TUSS codes, contract-like prices) and `configs/synthetic_plan.toml` runs the pipeline on them. The end-to-end
test does exactly this in a throwaway folder.

## Configuration

One TOML file per dataset (`configs/*.toml`): the source, the split windows, gate thresholds and the retrain policy.
Code holds no dataset-specific dates or thresholds.

## What is deliberately out of scope

- No real patient or health-plan data in this repository, ever: only the public SIH/SUS adapter and an empty template.
- No automatic scheduling: retraining is a decision the run reports and a person executes.
