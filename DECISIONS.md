# Decision log

This project was implemented with [Claude Code](https://claude.com/claude-code), an AI coding assistant, and every
commit says so. The author set the direction, questioned the work as it went and decided what stayed. This file
records the decisions that shaped the project and where each one came from, so that the split is visible instead of
implied.

"Author" means a decision or a question raised by the author. "Assistant" means a choice proposed by the assistant
and kept after review.

## Raised by the author

| # | Decision | Why | What it became |
|---|---|---|---|
| 1 | **Do not model SUS data directly. Define a standard table that any source fills.** | Other payers should be able to fill the same table with their own admissions and run the same thing, instead of adapting code written around one source. | The data contract ([DATA_CONTRACT.md](DATA_CONTRACT.md)), the SIH/SUS adapter as one source among others, `costmodel validate`, the template and the synthetic plan data. |
| 2 | **What transfers is the pipeline, not the model, so the pipeline must train, check and retrain on its own.** | A model trained on SUS prices is useless to a plan with TUSS codes and contract prices. A plan should drop its data in and get a model that was validated and gated without editing code. | `costmodel run`, relative time windows, the promotion gate with a baseline fallback, the monitor and the retrain rule. |
| 3 | **Should the architecture come first? Write it down, and keep building.** | To have the layers agreed in writing before the code grew around them, without stopping the work. | [ARCHITECTURE.md](ARCHITECTURE.md), written before the model work. |
| 4 | **Nothing counts as working until it has run in Docker.** | "How can it go public if it was never tested?" | The Docker image, the test suite running inside it in CI, and a real bug found that way (the full run died when every step shared one process). |
| 5 | **The public repository holds only what is meant to be public.** | No internal notes, no personal details, no old history. | The repository was recreated with a clean history and scanned for secrets and personal data before the first push. |
| 6 | **Before publishing, look at what is new and at what others have done.** | To check the project against the current state of the field instead of assuming it was finished. | The roadmap in the README (prediction intervals, tabular foundation models), and a closer look at 2024, which located the July step in elective surgery. |
| 7 | **Show the results on a page, not only in Markdown.** | Results should be visible without reading reports. | `costmodel site`: a [static page](https://mauryneto.github.io/hospital-cost-forecast/) generated from the reports. |
| 8 | **Is one model family enough? Test others.** | The candidate had only been compared with simple rules and with other settings of itself. | `costmodel families`: random forest and CatBoost on the same split, features and metrics. Result: see below. |
| 9 | **Make visible which decisions were the author's.** | The commits show that an assistant wrote the code. They do not show who decided what. | This file. |
| 10 | **Everything is predicted by one model for all admissions. Would one model per kind of admission be better?** | A model fitted to one group could learn that group better; it could also lose what the groups have in common. Not obvious either way. | `costmodel segments`: the chosen setting fitted per admission type and per procedure group. Result: see below. |
| 11 | **Prices change every quarter, so train on only the last 3 or 4 months and predict the next 3. And an admission is paid months later: is that accounted for?** | People who work in health care say a long history teaches old prices. And part of what is paid in a month comes from admissions of two or three months before. | `costmodel base-window`: retrained before every quarter with only the months already billed, with histories of 4, 12 and 24 months and all of them. Result: see below. The billing delay was measured again on the data; a cash-flow forecast went to the roadmap. |
| 12 | **If a plain average misses the month's total by 7%, why build all this? Is there a score that says whether the model is worth it?** | The results table showed a global mean almost tied with the model on the month's total. That deserved a straight answer, not a footnote. | `costmodel skill`: the error on totals for all admissions and for each hospital, specialty and procedure group, next to a plain average's, and how much smaller the model's is. Result: see below. |
| 13 | **Show the final model on a screen like a BI tool: the whole state, admission type, procedure group, bed specialty, hospital and each admission, month by month.** | A table in a report answers one question. A screen with filters lets someone look for their own. | `costmodel explore` and the [explorer page](https://mauryneto.github.io/hospital-cost-forecast/explore.html). Each admission is shown only as distributions: no admission is published. |
| 14 | **No prediction ranges. Next: a tabular foundation model; then cash flow; then the official price table as an input; other states and patient history later. No neural network.** | Whoever uses this wants a value, not a range. Several kinds of model were already compared, so the recent one that is missing comes first. | The roadmap in the README, in this order. |

## Fixed in the brief, before any code

The project brief, which the author brought to the first session, set these constraints, and they were not
renegotiated during implementation: only
public data; raw data never committed, only its checksums; prediction at the moment of admission; time-based
validation; the results table shows the numbers as they are, including where a baseline does better; retraining is
recommended by the system and decided by a person, with nothing scheduled.

## Proposed by the assistant and kept after review

- LightGBM as the candidate, with the search over objective, tree size, training window and the time input.
- The admission date as an input, so that trees apply the latest price level to new admissions.
- Two metrics in the gate (error per admission and error on the month's total), and its thresholds.
- Measuring the billing delay on the data and judging a month only three months after it ends.
- The replay of retraining policies, and the rule that a gate month must be unseen by both models.
- The price level in the monitor (same admissions at the prices the model last saw), which located the July 2024 step
  in elective surgery.
- Generating the results page from the reports, so that it cannot disagree with them.
- Giving two numbers for every admission instead of one, and the gate's rule for replacing a model that has them
  (see "Two numbers instead of one").
- Rejecting LazyPredict, PCA and Naive Bayes for decision 8, in favour of a controlled comparison: the first cannot
  handle this data size or report the month's total, the second would destroy the information in the procedure codes,
  and the third is a classifier.

## Open

- **The retraining policy.** The 2024 replay suggests retraining whenever a new billed month exists. The retrain rule
  was fixed before looking at 2024 and has not been changed, because changing it on the year that suggested the
  change would be fitting it to the result. It needs a new period to test. The author's call.
- **The licence of the newer foundation models.** TabPFN-2.5 and later read up to a million rows instead of 10,000,
  the limit that hurt in the comparison. They need an account and a non-commercial licence that also covers the
  model's outputs. The author's call.
- **A year not used for any choice.** The second of the model's two numbers, and the gate rule written for it, were
  judged on years already seen (see "Two numbers instead of one"). 2025 is the check that is missing.

## Result of decision 8

On the 2022 validation year, with the same training admissions and inputs
([families_valid.md](reports/sihsus_rj/families_valid.md)):

| Model family | Setting | MAE per admission (R$) | Error on the month's total |
|---|---|---:|---:|
| **LightGBM** | least squares on log(cost); chosen among 16 settings | **792.15** | 14.98% |
| Random forest | on log(cost) | 835.15 | 18.34% |
| CatBoost | Tweedie | 837.15 | 11.28% |
| CatBoost | least squares on log(cost) | 838.64 | 15.76% |
| Random forest | on the cost | 841.43 | **9.09%** |

LightGBM stays. The other families are 5–6% worse per admission, and CatBoost takes 30 to 40 minutes per fit against
75 seconds. The comparison favours LightGBM in tuning (16 settings against two each), so the gap would narrow with
more work, but nothing here suggests it would close.

The question still paid off, in a different place: the models fitted on the cost itself miss the month's total by
9–11%, against 15% for the chosen one. That is a trade-off between the error per
admission and the error on the total, and it now shows up across three families instead of one.

That led to one more experiment, proposed by the assistant and approved by the author: averaging the chosen setting
with the Tweedie setting that has the lowest error on the month's total
([blend_valid.md](reports/sihsus_rj/blend_valid.md)). Half and half is better on both metrics on 2022: 787.47 against
792.15 per admission, and 12.58% against 14.98% on the month's total. The gain per admission (0.6%) was below the 2%
the gate asked, so the average was not promoted. The two settings ended up being served side by side instead (see
"Two numbers instead of one").

## Result of decision 10

On the same validation year ([segments_valid.md](reports/sihsus_rj/segments_valid.md)):

| | Models | MAE per admission (R$) | Error on the month's total |
|---|---:|---:|---:|
| **One model for every admission** | 1 | 792.15 | **14.98%** |
| One model per admission type | 2 | 791.17 | 15.28% |
| One model per procedure group | 2 | **777.98** | 16.49% |

The single model stays. By admission type nothing changes. By procedure group the error per admission falls 1.8%,
all of it in surgery (R$ 575 against R$ 628), and the error on the month's total rises above the gate's ceiling.
It is the same trade-off for the third time: what helps each admission does not help the total. Surgery with a
model of its own is the lead worth following, on a year not used for any choice.

## Result of decision 11

On the same validation year, retraining before every quarter with only what was already billed
([base_window_valid.md](reports/sihsus_rj/base_window_valid.md)):

| History given to each retrain | MAE per admission (R$) | Error on the month's total |
|---|---:|---:|
| Last 4 billed months | 818.56 | 14.20% |
| Last 12 billed months | 800.17 | 13.49% |
| Last 24 billed months | 796.58 | 13.03% |
| **Every billed month** | **788.65** | 13.39% |
| Every billed month, recent months weigh more | 796.15 | **12.37%** |

On this data the short history loses: 3.8% worse per admission than using everything, in every quarter, and no better
on the month's total. Two reasons. There are thousands of procedure codes and four months hold few examples of most
of them. And because bills take months to arrive, the last four billed months are already 4 to 7 months old when the
quarter is priced, so the short history is not much fresher than the long one. The author's second point is what
explains the first.

The idea is not dismissed for other data. SUS prices move slowly; a payer with quarterly contract prices and faster
bills can run the same command and may get the opposite answer. Weighing recent months more, the middle ground the
assistant added, gave the lowest error on the month's total.

## Result of decision 12

On the test year ([skill_test.md](reports/sihsus_rj/skill_test.md)):

| Totals of each month, for | Plain average | This model | Model's error is smaller by |
|---|---:|---:|---:|
| All admissions | 7.36% | 6.64% | 10% |
| Each admission type | 17.61% | 6.65% | 62% |
| Each bed specialty | 28.11% | 6.83% | 76% |
| Each hospital | 49.25% | 11.16% | 77% |
| *Each admission (MAE, R$)* | *1,579.64* | *778.62* | *51%* |

The author's doubt was right for the one total it was about: for all admissions together, a plain average is nearly
enough, because a price rise and a shift to cheaper procedures partly cancelled. It stops being enough for any
smaller total, which is what a payer budgets by. The score is now part of the reports and of the results page, so
the question is answered where the numbers are.

## Two numbers instead of one

After five experiments on 2022 (decisions 8, 10 and 11, the average of two settings and the recent price), the error
per admission had moved by 1.8% at most, and every change that helped it hurt the error on the month's total. The
author asked what else could improve the cost prediction before moving on. The assistant's answer was that the two
metrics pull apart for a structural reason, the skew of the costs, and proposed giving two numbers instead of
looking for one: a typical cost for a single admission and an expected cost to add up. The author asked how the error
on the total is computed and what the API would return, and approved.

What changed:

- **The model** is two settings from the same search, registered as one version: the one with the lowest error per
  admission (typical cost) and the one with the lowest error on the month's total (expected cost). The API and the
  batch job return both.
- **The gate** judges each number by its own metric. The rule against the best baseline did not change. The rule
  against the model in production did, and this wording was the assistant's: a new version replaces it when at least
  one of the two numbers is better by its margin (2% for the error per admission, 10% for the month's total) and
  the other is not worse. Under the old rule (2% better per admission) the new version could not have entered,
  because its typical cost is the same model as before. The margins were set before running anything.
- **The test year was scored a second time,** to judge the expected cost. The typical cost's numbers on 2023 are
  those of the first and only look (MAE 778.62). The expected cost's are new: 6.53% on the month's total, against
  11.32% for the typical cost. Version 3 passed the gate and replaced version 2.
- **The monitor's alarm moved.** With a better number for totals, June 2024 is no longer out of bounds (May still
  is, on the error per admission). The first two months in a row out of bounds are now July and August, so the
  retrain rule fires at the end of November instead of September.

What this does not settle: the expected cost and the new gate rule were judged on years that had already been looked
at, after seeing what the first version did on them. That is the reason for the open item on a fresh year.

## Result of decision 14, first step: a tabular foundation model

The author approved installing TabPFN and downloading its pre-trained model. Two things were found before running
anything, and decided by the assistant within that approval:

- The newest checkpoints (TabPFN-2.5 to 3.5, which read up to a million rows) need an account and the acceptance of
  a non-commercial licence that also covers the model's outputs. That is the author's decision to make, so the
  comparison used **TabPFN-2**, the checkpoint with open weights, which reads 10,000 rows.
- A new release of the package had come out the day before. The version installed is one that had been public for
  five weeks.

`costmodel foundation` ([foundation_valid.md](reports/sihsus_rj/foundation_valid.md)), on 30,000 admissions drawn at
random from 2022, the same for every row:

| | MAE per admission (R$) | Error on the month's total |
|---|---:|---:|
| LightGBM as in production, all 2,819,848 training admissions | 787.13 | 11.69% |
| LightGBM, 10,000 admissions of the last 12 months | 965.15 | 19.69% |
| LightGBM, the same plus code statistics | 882.57 | 19.35% |
| TabPFN-2, the same 10,000 admissions | 877.30 | 10.84% |
| TabPFN-2, the same plus code statistics | 792.38 | 9.35% |

On equal terms (same rows, same inputs) the foundation model is 9% to 10% better per admission than LightGBM. Against
the model in production, which learns from 280 times more rows, it is 11.5% worse alone. Given statistics of the
codes computed on the rest of the training window, it is R$ 5.25 worse per admission with a 95% interval of -6.00 to
+16.93, which is a tie, and 2.34 points better on the month's total (interval 1.30 to 3.04).

Not adopted, for four reasons: it is one sample of one year with one draw of the context; the best of four variants
was picked on that same sample; it needs a GPU and minutes where LightGBM needs seconds; and a new candidate would
face a test year already looked at twice. The assistant's reading is that this is the most promising result among
the side comparisons, and that it deserves a fresh year more than a quick adoption. Whether to accept the licence of
the newer checkpoints is open, and the author's call.

## Tried and not adopted: the recent price of each procedure as an input

Asked what would improve the predictions most, the assistant bet on telling the model what each procedure has been
costing lately, since the largest error measured is an outdated price level. The author asked whether a payer would
really have that information (it would: it is computed from its own paid bills) and approved the test
([recent_price_valid.md](reports/sihsus_rj/recent_price_valid.md)).

The bet did not pay off on 2022. As one more input it made the error per admission slightly worse (795.56 to 803.94
against 792.15). Predicting the cost relative to the recent price cut the error on the month's total to 10.92%, at
3.8% more per admission, which is what the Tweedie setting already does without any new input. It was built for a
sudden price change like July 2024; 2022 has none, so that part of the idea remains untested.
