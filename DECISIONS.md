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
| 15 | **Before moving on to the cash-flow forecast, do the 2025 check.** | Every choice so far was made on 2022 and judged on 2023 and 2024, which were looked at more than once. A year no choice was made on is the only clean check left. | The protocol below, written and committed before any 2025 cost was read; then `costmodel fresh`, run once. |

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
- **What to do with the 2025 check** (see "Result of the 2025 check"): a typical cost that is a median (tried on
  2022 and better per admission, see "The typical cost as a median"; not adopted), the gate's rule for retrained
  models, and a typical-cost model of its own for surgery. The author's call.
- **The next period not used for any choice.** 2025 has been read. The first half of 2026 is next, once billed.

## The 2025 check: written before looking

This section was written and committed before any 2025 cost was read, by a model or by a person. Only the raw files
were downloaded and their checksums recorded. The settings are in `configs/sihsus_rj_2025.toml`, a dataset of its
own, so the reports of 2018-2024 do not change. The author approved this protocol as proposed by the assistant.

**The model that is scored.** The two settings of version 3 (typical cost: least squares on log(cost), 255 leaves,
324 trees; expected cost: Tweedie, 63 leaves, 303 trees), fitted at the end of December 2024 on the admissions whose
costs were known by then (admission months up to September 2024, because of the 3-month billing delay). Nothing is
searched or tuned. It prices every admission of 2025.

**Four questions, each with what counts as confirmed.** Differences come with a 95% interval, from drawing the scored
admissions again with replacement 1,000 times. A difference counts only when its interval leaves zero out.

1. **Do the two numbers hold?** Confirmed if the typical cost has the lower error per admission and the expected
   cost the lower error on the month's total, both beyond their intervals.
2. **Is the model still worth it?** Confirmed if it passes the gate's rule against the best baseline (error per
   admission at least 2% lower, month's total not worse, and the month's total within 15%) and its error on totals is
   smaller than a plain average's at every level (all admissions, admission type, procedure group, bed specialty,
   diagnosis chapter, procedure subgroup, hospital, each admission).
3. **Do the ideas that looked good on 2022 repeat?** Each against the model above, fitted on the same admissions:
   - *Recent months weigh more* (half-life of 12 months; on 2022: better on the month's total, worse per admission).
     Confirmed if it is better on the month's total.
   - *A typical-cost model of its own for surgical admissions* (107 trees, the number found on 2022; there: better
     per admission). Confirmed if the error per admission over all admissions is lower.
   - *TabPFN-2 reading 10,000 admissions of the last 12 known months, with code statistics from the rest* (on a 2022
     sample: a tie per admission, better on the month's total). Scored on 30,000 admissions of 2025 drawn at random.
     Confirmed if it is better on the month's total and not worse per admission.
4. **Is retraining every month better than waiting for the rule?** The 2024 replay, on 2025: never retrain, retrain
   when the rule fires, try every month, each with all known months or the last 24. Confirmed if "every month" on
   all known months has a lower error on the month's total than "never" and than "rule" on the same window. This one
   has no interval: the replay gives totals, not paired predictions.

For the record, and not one of the four questions: version 3 exactly as it is in production (fitted on 2018-2022,
never retrained) is scored on 2025 too.

**Rules.**

- One command (`costmodel fresh`), run once. The code is committed before it runs.
- Whatever comes out is published, including what is not confirmed.
- If a mistake in the code is found after the run, it is fixed, the run repeated and the fact stated here.
- Nothing is adopted by this check. The model in production stays as it is; what to do with the result is the
  author's decision.

## Result of the 2025 check

Run once, on the code of commit `a04832a`, which was pushed before the run; the protocol is commit `82383d7`. It
took 46 minutes. [The report](reports/sihsus_rj_2025/fresh_year.md) has every number; the model priced 919,818
admissions with an error of R$ 1,029.00 per admission and 2.79% on the month's total.

| | Verdict | |
|---|---|---|
| 1. The two numbers | Confirmed | Typical cost 1,029.00 against 1,049.47 per admission; expected cost 2.79% against 5.37% on the month's total |
| 2. Still worth it | Not confirmed | A gain over a plain average at every level (42% to 83%), but per admission only 0.6% better than the best baseline, where the gate asks for 2% |
| 3. Recent months weigh more | Contradicted | Worse on both metrics |
| 3. A model of its own for surgery | Confirmed | 40.68 less per admission over all admissions; 1,055.48 against 1,144.98 on surgical admissions |
| 3. TabPFN-2 with code statistics | Contradicted | Better per admission, 34.50 points worse on the month's total |
| 4. Retraining every month | Confirmed | 2.12% on the month's total against 2.79% for never and for the rule |

Three things the verdicts do not say by themselves, in the assistant's reading:

- **Question 4 passed by a small margin and for an awkward reason.** The gate refused 18 of the 21 retrained
  candidates, all of them 6% to 13% better per admission on the two months they were judged on, for being worse on
  those months' total. So "by the rule" ended up identical to "never", and "every month" used a single retrained
  model. The gate's rule against the model in use, written with the two numbers, had not been exercised on a fresh
  period before; on this one it is too strict.
- **The error per admission rewards predicting low.** A table of medians (44.29% off on the month's total) nearly
  tied with the model per admission, and the model in production, never retrained since 2022 and 17.55% off on the
  month's total, was closer to each admission (960.25) than the model fitted for 2025. The typical cost is the log
  model's prediction scaled up (by 17% in version 3); an estimate meant to be close to each admission would not be
  scaled. This was not tested and is the first thing to try next, on the validation year.
- **TabPFN-2's two numbers went opposite ways.** Its median was closer to each admission than the model's typical
  cost, and its mean overshot the month's total in every month of the year. A diagnostic run afterwards on the same
  admissions points at how the mean is taken out of it (an average of exp() over a hundred levels of the distribution
  it predicts for log(cost)): 98% of the paid amounts fell below the level it gave as its 95%, and the few levels
  above that one carry almost all of the excess. On 2022 the same recipe was 8.50% low. That is a finding about this
  way of reading an expected cost from the model, not a verdict on foundation models.

One diagnostic was run after the check, on the same 30,000 admissions, to understand the TabPFN result. It changed no
verdict and no code. Nothing was adopted: the model in production is still version 3, and what to do with each
finding (a median for the typical cost, the gate's rule for retrains, a model of its own for surgery) is the
author's decision. 2025 has now been read; the next period no choice was made on is the first half of 2026.

## The typical cost as a median: tried on the validation year

The first thing the 2025 check left to try. `costmodel typical` fits the setting chosen for the typical cost on the
training window and reads it four ways on 2022 ([the report](reports/sihsus_rj/typical_valid.md)); each difference
has a 95% interval from drawing the 786,766 admissions again.

| The typical cost as | MAE | Against today's | MAE, other 90% | MAE, top 10% | If added up | Lead over the best baseline |
|---|---:|---:|---:|---:|---:|---:|
| Today's: scaled by the factor (1.157) | 792.15 | | 313.72 | 5,098.02 | 14.98% low | 6.65% |
| The same model without the factor | 747.38 | −44.77 (−45.58 to −43.90) | 251.26 | 5,212.42 | 26.50% low | 11.93% |
| A median model, on log(cost) | 733.45 | −58.69 (−60.37 to −56.91) | 219.72 | 5,356.99 | 31.71% low | 13.57% |
| A median model, on the cost | 746.11 | −46.04 (−48.00 to −44.01) | 234.37 | 5,351.75 | 31.17% low | 12.08% |

In the assistant's reading:

- **The hypothesis holds on 2022.** The factor that makes the typical cost add up better makes it 5.7% worse at what
  it is for. Three quarters of what a median model gains comes from just leaving the factor out.
- **Leaving the factor out is the cheap option:** no new model, no retraining, the same 324 trees. The median model
  on log(cost) is better by another 13.93 per admission, but it ran into the limit of 2,000 trees without stopping,
  so it is six times larger and its number here is not its best. A fair version of it needs a higher limit or a
  higher learning rate, which is one more search.
- **The price is on the expensive admissions:** 2.2% further from the top 10% without the factor, 5.1% with the
  median model. Whoever reads the typical cost of a likely expensive admission gets a lower number than today.
- **Whether to leave the expensive admissions out** was asked by the author on seeing these numbers: they look like
  outliers and they dominate the average error. They are kept. They are real bills, 57.3% of the money paid, and
  nobody knows at admission which admission will be one of them, so a model cannot leave them out and an error
  measured without them would describe a case that cannot be chosen in advance. What the question changed is how
  the error is read: in two bands, the 90% least expensive admissions and the 10% most expensive, cut by what was
  paid. On the first band the error is 313.72 today and 251.26 without the factor; on the second, 5,098.02 and
  5,212.42. The typical cost without the factor is the "price of an ordinary admission" the question was after.
- **It does not settle the 2025 question.** On 2025 the model led the table of medians by 0.6%; on 2022 leaving the
  factor out adds about five points to that lead. Whether it does the same on a year like 2025 was not measured:
  2025 has been read, and scoring this on it now would be choosing with a year that was meant to choose nothing.

Not adopted. Adopting it would be a new version (the typical cost without the factor), judged by the gate, and
confirmed on the first half of 2026. The author's call.

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
