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
- Rejecting LazyPredict, PCA and Naive Bayes for decision 8, in favour of a controlled comparison: the first cannot
  handle this data size or report the month's total, the second would destroy the information in the procedure codes,
  and the third is a classifier.

## Open

- **The retraining policy.** The 2024 replay suggests retraining whenever a new billed month exists, and a gate that
  weighs the month's total more. The rule was fixed before looking at 2024 and has not been changed, because changing
  it on the year that suggested the change would be fitting it to the result. It needs a new period to test. The
  author's call.
- **Whether the average of two settings replaces the model in production** (see the end of this file). It depends on
  the gate weighing the month's total, and on a period that has not been used to choose anything.

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
9–11%, against 15% for the chosen one. That is the same trade-off as in the open item above (error per admission
against error on the total), and it now shows up across three families instead of one.

That led to one more experiment, proposed by the assistant and approved by the author: averaging the chosen setting
with the Tweedie setting that has the lowest error on the month's total
([blend_valid.md](reports/sihsus_rj/blend_valid.md)). Half and half is better on both metrics on 2022: 787.47 against
792.15 per admission, and 12.58% against 14.98% on the month's total. The model in production was not changed. The
gain per admission (0.6%) is below the 2% the gate asks, so whether the average replaces it depends on the open item
above: how much the gate should weigh the month's total.

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
model of its own is the lead worth following, once the gate question is settled.

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

## Tried and not adopted: the recent price of each procedure as an input

Asked what would improve the predictions most, the assistant bet on telling the model what each procedure has been
costing lately, since the largest error measured is an outdated price level. The author asked whether a payer would
really have that information (it would: it is computed from its own paid bills) and approved the test
([recent_price_valid.md](reports/sihsus_rj/recent_price_valid.md)).

The bet did not pay off on 2022. As one more input it made the error per admission slightly worse (795.56 to 803.94
against 792.15). Predicting the cost relative to the recent price cut the error on the month's total to 10.92%, at
3.8% more per admission, which is what the Tweedie setting already does without any new input. It was built for a
sudden price change like July 2024; 2022 has none, so that part of the idea remains untested.
