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
| 7 | **Show the results on a page, not only in Markdown.** | Results should be visible without reading reports. | `costmodel site`: a static page generated from the reports. |
| 8 | **Is one model family enough? Test others.** | The candidate had only been compared with simple rules and with other settings of itself. | `costmodel families`: random forest and CatBoost on the same split, features and metrics. Result: see below. |
| 9 | **Make visible which decisions were the author's.** | The commits show that an assistant wrote the code. They do not show who decided what. | This file. |

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
