# Model card: `cost-sihsus_rj` version 2

Predicts, at the moment of admission, how much the SUS will pay for a hospital admission in the state of Rio de
Janeiro. It is the model currently in `production` in the MLflow registry. All numbers below come from the reports in
[`reports/sihsus_rj/`](reports/sihsus_rj/) and can be regenerated with the commands in the [README](README.md).

## Model details

| | |
|---|---|
| Type | Gradient-boosted trees (LightGBM), regression |
| Objective | Least squares on log(cost), returned to reais with Duan's smearing factor (so predictions estimate the mean, not the median) |
| Size | 255 leaves per tree, 324 trees (chosen by early stopping on 2022), learning rate 0.1, seed 42 |
| Trained on | 3,606,614 admissions from 2018-01-01 to 2022-12-31 (training + validation windows, refitted after the search) |
| Chosen by | Lowest MAE on 2022 among 16 settings whose monthly-total error stayed at or under 15% ([train_valid.md](reports/sihsus_rj/train_valid.md)) |
| Alternatives tried | Random forest and CatBoost on the same 2022 validation year: 5–6% worse per admission ([families_valid.md](reports/sihsus_rj/families_valid.md)) |
| Promoted by | The gate on 2023, scored once ([test.md](reports/sihsus_rj/test.md)) |
| Packaging | MLflow pyfunc that takes the raw admission columns and builds its own features, so callers cannot build them differently |
| Served by | `costmodel score` (a month in batch) and `costmodel serve` (`POST /predict`) |

**Inputs**, all known when the patient is admitted ([DATA_CONTRACT.md](DATA_CONTRACT.md)): admission date, age, sex,
main diagnosis (ICD-10), requested procedure (SIGTAP code), admission type (elective or urgent), hospital, municipality
of residence, bed specialty. The model derives coarser groups from the codes (ICD-10 category and chapter, procedure
group and subgroup), the month of year, and the admission date as a number of months. The last one lets the trees
apply the most recent price level to new admissions.

**What it relies on** (share of the trees' gain): requested procedure 48%, diagnosis category 21%, procedure subgroup
13%, hospital 11%, everything else under 5% each.

**Output:** the expected amount paid, in reais.

## Intended use

- **Forecasting money, not cases.** Summing the predictions of a month's admissions estimates what those admissions will
  cost months before the bills arrive (SUS hospitals bill up to about 5 months later). This is the use the gate and the
  monitor measure ("error on the month's total").
- **A first estimate per admission**, for planning and for prioritising which admissions to look at, with an error of
  about R$ 780 per admission on 2023 (see below).
- **A reference implementation.** The same pipeline can be trained on another payer's own admissions (see the README).
  This model itself applies only to SUS admissions in Rio de Janeiro.

## Out of scope: do not use it to

- **Decide anything about a patient's care**: authorising, denying, delaying or prioritising an admission or a
  treatment. The model predicts a reimbursement, not a clinical need.
- **Price or select people**: premiums, eligibility or any decision about an individual.
- **Judge or rank hospitals.** The hospital is an input, so the model learns each hospital's typical billing. That
  says nothing about its efficiency or quality, and a hospital with a sicker case mix will simply look "expensive".
- **Flag a bill as wrong or fraudulent.** A large gap between the bill and the prediction is expected for the most
  expensive admissions (see "Errors on the most expensive admissions").
- **Estimate what an admission costs the hospital.** The target is what the SUS paid (`VAL_TOT`, the reimbursement),
  which can be far from the real cost of care.
- **Predict private-plan, other-state or post-2024 costs without retraining.** Codes, prices and mix differ, and the
  model is already more than 20% low on late 2024 (see "Monitoring").

## Data

Public SIH/SUS hospital admission files (RD), state of Rio de Janeiro, downloaded from DATASUS with SHA-256 checksums
([`data/checksums/sihsus.json`](data/checksums/sihsus.json)). No raw data is stored in the repository.

- **Admissions:** 5,385,538 from 2018 to 2024, after these rules: only regular AIHs (`IDENT = 1`; type 5 are monthly
  continuation bills of long stays, not new admissions), one row per AIH, amount paid above zero, and dated by
  **admission**, not by billing month. Billing months up to 2025-06 are read so that late bills of 2024 admissions are
  included.
- **Target:** `VAL_TOT`, the total amount paid for the AIH. Median R$ 671, mean R$ 1,583; the 10% most expensive
  admissions take 57% of the money.
- **Windows, by admission date:** train 2018–2021 (2,819,848), validation 2022 (786,766), test 2023 (877,760),
  monitoring 2024 (901,164).
- **Excluded from the inputs** because they are known only during or after the stay: length of stay, ICU days,
  performed procedure, death, discharge, secondary diagnoses, complexity and the reason for discharge. A test fails
  the build if any of them reaches the features.

## Performance on the test year (2023, scored once)

All models were fitted on the same data (2018–2022) and scored on the same 877,760 admissions.

| Model | MAE (R$) | Error on the month's total | Bias | MAE, top 10% most expensive |
|---|---:|---:|---:|---:|
| **This model** | **778.62** | 11.32% | −11.42% | **4,724.83** |
| Median by procedure × age band × admission type | 872.80 | 40.81% | −40.89% | 6,191.20 |
| Median by procedure | 886.29 | 41.53% | −41.61% | 6,283.98 |
| Linear regression on log(cost) | 937.81 | 12.89% | −12.89% | 5,848.08 |
| Mean by procedure × age band × admission type | 972.10 | 13.44% | −13.57% | 5,376.14 |
| Mean by procedure | 1,003.53 | 13.50% | −13.63% | 5,546.33 |
| Global mean | 1,579.64 | **7.15%** | −7.36% | 7,763.69 |

"Error on the month's total" is the mean, over the 12 months, of |predicted total − paid total| / paid total.

- The model has the lowest error per admission, 10.8% below the strongest baseline (the business-rule median by
  procedure, age and admission type), and the lowest error on the most expensive admissions.
- **On the month's total, a global mean does better (7.2% against 11.3%).** That is a coincidence of mix (cheaper and
  dearer procedures shifting in opposite directions), not skill: the same global mean is off by R$ 1,580 per
  admission. Anyone who needs only the monthly total should still compare against a simple average.
- The model under-predicts 2023 by 11%: prices kept rising after the training data ended.

### Error by segment (MAE, R$)

| Segment | Admissions | Mean paid | This model | Best baseline |
|---|---:|---:|---:|---:|
| Age < 1 | 45,381 | 2,307.35 | 1,496.71 | 1,708.15 |
| Age 1–4 | 33,080 | 963.31 | 414.24 | **402.42** |
| Age 5–14 | 45,669 | 922.22 | 402.22 | 404.24 |
| Age 15–29 | 175,620 | 884.67 | 323.88 | 334.13 |
| Age 30–44 | 151,432 | 1,173.50 | 499.36 | 542.16 |
| Age 45–59 | 145,430 | 1,913.02 | 892.68 | 1,024.90 |
| Age 60–74 | 184,356 | 2,355.87 | 1,120.41 | 1,297.82 |
| Age 75+ | 96,792 | 2,185.72 | 1,183.72 | 1,319.58 |
| Female | 506,388 | 1,420.73 | 648.51 | 717.16 |
| Male | 371,372 | 1,929.10 | 956.04 | 1,085.02 |
| Elective | 265,323 | 2,186.54 | 926.10 | 1,119.74 |
| Urgent | 610,135 | 1,394.76 | 713.40 | 764.37 |
| Other admission type | 2,302 | 2,050.30 | 1,069.11 | 1,150.10 |

"Best baseline" is the median by procedure × age band × admission type. The model is better in every segment except
children aged 1–4, where the two are within 3%. Errors are larger where admissions cost more (babies, older adults,
men, elective surgery); relative to the mean paid, the gap between groups is much smaller. This table is a check of
where the model errs, not a fairness audit.

### Errors on the most expensive admissions

Grouping 2023 admissions by decile of what was actually paid:

| Decile of amount paid | Mean paid (R$) | Mean predicted (R$) |
|---:|---:|---:|
| 1 (cheapest) | 138.16 | 238.93 |
| 5 | 629.21 | 837.38 |
| 9 | 2,251.76 | 2,332.75 |
| 10 (most expensive) | 9,279.05 | 5,724.60 |

Like any model of this kind, it pulls extremes towards the middle: it over-predicts cheap admissions and
under-predicts the most expensive ones by about 38% on average. What happens during an expensive stay (complications,
ICU) is not known at admission. Individual predictions for high-cost cases should be read as "this is likely to be
expensive", not as an amount.

## Monitoring (2024, simulated production)

The model priced every month of 2024 as if it were live ([monitor.md](reports/sihsus_rj/monitor.md)). A month is
judged only 3 months after it ends, when 98.7% of admissions and 95.4% of the money have been billed (measured on
2018–2021).

![Monitoring](reports/sihsus_rj/monitor.png)

- **It under-predicts throughout 2024:** the month's total is about 14% low from January to April, 16–17% in May and June,
  and 20–24% from July to December. The MAE steps up from about 850 to over 1,000 in July.
- **The cause is price, and the July step is in elective surgery.** Against the prices the model last saw (mean per
  procedure in 2022, same mix), what was paid is 7.91–9.12% higher from January to April, 11.08–13.07% in May and June
  and 16.90–21.89% from July. The error on urgent admissions stays between −12.63% and −16.58% all year; on elective
  admissions it goes from −13.79% to −17.75% (January to April) to −28.25% to −34.10% (July to December). The median
  paid for cataract surgery (SIGTAP 0405050372) is R$ 772 in every month up to June and R$ 3,858 from July. The rule
  behind this change was not identified.
- **The retrain rule fires at the end of September 2024.** May and June broke the limits (MAE more than 10% above
  the test MAE, or the month's total off by more than 15%), but their costs were known only in September.
- **Input drift did not warn about it.** The largest PSI among the inputs stays between 0.24 and 0.28 all year, with no
  change when the error jumps, and the PSI of the predicted cost stays around 0.01. Admissions looked the same; what
  changed was the price paid for them (concept drift). Only the error, known after billing, shows it.

**Retraining.** The monitor recommends; a person decides and retrains, and the new model must pass the same gate. A
replay of 2024 under three policies, using only data known at each moment
([replay.md](reports/sihsus_rj/replay.md)):

| Policy | MAE (R$) | Error on the month's total | Retrained models put in use |
|---|---:|---:|---|
| Never retrain | 904.69 | 19.03% | none |
| Retrain when the rule fires (all data / last 24 months) | 905.55 / 907.55 | 16.72% / 16.57% | 1, from October |
| Try to retrain every month (all data / last 24 months) | 898.11 / 903.10 | 11.75% / 11.30% | 2, from February and September |

The rule reacts late to a steady price rise; retraining whenever new billed data exist cuts the error on the month's
total from 19% to 11–12%. The rule and the gate thresholds were fixed before looking at 2024 and have not been changed to fit
these results.

## Known limitations

- **Payment-rule changes are invisible until billed.** The July 2024 change in what is paid for elective surgery
  could not be learned from earlier data, and showed in no input. The monitor's price level and its breakdown by
  admission type show where such a change happened, three months later.
- **Systematic under-prediction when prices rise.** The time input carries the latest price level forward but cannot
  anticipate increases. Without retraining, the error grows month after month.
- **Requested procedure.** It is the strongest input and matches the performed procedure in 95.9% of admissions. The
  public files hold the AIH as billed, so it cannot be confirmed that this field was never amended after admission.
  If it was, the measured error is somewhat optimistic.
- **New or rare codes.** Procedures, diagnoses or hospitals seen 20 times or fewer in the training data are treated as
  unknown; the model then falls back on coarser groups (procedure subgroup, ICD-10 chapter).
- **Late feedback.** The error of a month is known only 3 months after it ends, so problems are detected at least that
  late.
- **Sex and age are inputs.** They are known at admission and help explain cost (stays of babies and older adults
  cost more). This is one more reason to keep to the intended uses above.
- **The serving API is for local or internal use**: no authentication, no rate limiting, one process. It binds to
  127.0.0.1 by default.
