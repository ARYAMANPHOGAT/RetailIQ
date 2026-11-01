# RetailIQ verification report

This report separates recorded test executions from inspection of saved outputs. The documentation update itself
did not run tests or retrain models. The latest project test count supplied for this update is **78 passed tests**;
the historical environment report below recorded an earlier 35-test run.

## Test execution record

| Check | Evidence and result |
|---|---|
| Latest pytest suite | **78 passed, 0 failed** (latest reported project run; not rerun while editing documentation) |
| Earlier pytest suite | 35 passed, 0 failed in a fresh Linux virtual environment from the extracted ZIP |
| Earlier browser end-to-end suite | 42/42 Playwright checks passed against the packaged app; separate from the pytest count |
| Earlier original-model training | `python train.py` succeeded in about 35 seconds; extracted-ZIP retraining produced identical metrics |
| Earlier dependency installation | `pip install -r requirements.txt` succeeded in a fresh Linux virtual environment |
| Earlier app launch | Streamlit started and its health endpoint responded in the Linux environment |
| Windows batch scripts | **Not independently executed**: Windows was unavailable in the Linux verification environment |

The 78 pytest result is the latest reported test count. The 35 pytest result and 42 browser checks above are retained
as historical evidence and are not additions to the latest pytest count.

## M5 checks represented in the test suite

The 78-test suite covers the M5 validation, training, integration and dashboard paths, including:

- Validation of sales, calendar and price file structure, identifiers, day/date mapping, duplicate keys and value
  constraints.
- Cutoff-safe M5 feature generation, lag and rolling history, split boundaries, and protection against target-day
  leakage.
- Stage 3 LightGBM and seasonal-naive comparison, validation-based selection, metadata and artifact behavior.
- Stage 4 and Stage 4.1 feature/model comparisons, Stage 5 hybrid routing, validation-only experiment boundaries,
  selection locking, and final-test safeguards.
- Selected-model inference integration, generated M5 forecast shape and values, and separation from RetailIQ upload
  data.
- M5 Forecasting and M5 Model Performance dashboard behavior, including saved metric and model evidence.

The historical integration checks also exercised the original Streamlit app headlessly through its then-existing
seven original sections. The current application has **nine sidebar sections**: those seven plus M5 Forecasting and
M5 Model Performance.

## Saved M5 model comparison

The metrics below were read from the saved M5 reports under `outputs/m5/`. Validation covers d_1886–d_1913; final
holdout covers d_1914–d_1941. Each split contains 853,720 rows from 30,490 series. Metrics are per-series-day MAE
and RMSE in units sold, plus aggregate WAPE.

| Candidate | Validation MAE | Validation RMSE | Validation WAPE | Final holdout MAE | Final holdout RMSE | Final holdout WAPE |
|---|---:|---:|---:|---:|---:|---:|
| Stage 3 LightGBM | 0.8853 | 2.0012 | 63.86% | 0.9190 | 2.0556 | 63.69% |
| **Stage 4 LightGBM (selected)** | **0.8818** | 2.0072 | **63.60%** | **0.9157** | 2.0584 | **63.46%** |
| Stage 4.1 same-weekday experiment | 0.8804 | 2.0094 | 63.50% | Not evaluated | Not evaluated | Not evaluated |
| Stage 5 LightGBM + TSB hybrid | 0.9048 | 2.0155 | 65.26% | 0.9393 | 2.0713 | 65.10% |

Selection is Stage 4, based on validation criteria; its final holdout was evaluated after selection was fixed.
Stage 5 was not promoted. Stage 4.1 has validation-only results and its final holdout was not accessed. The Stage 5
holdout figures are an evaluation of the unpromoted candidate after the Stage 4 selection had been locked; they
did not change selection. The low-demand group remains difficult: Stage 4 final-test WAPE is 101.36% (low),
100.01% (medium) and 54.83% (high). Stage 5 validation low-demand WAPE is 137.16%, one reason it was not promoted.

The earlier Stage 3 baseline report also records seasonal-naive lag-7 MAE/RMSE/WAPE of 1.2054 / 2.6601 / 86.94%
on validation and 1.2447 / 2.6815 / 86.27% on final holdout.

## Artifact and data checks inspected for this update

- Saved `outputs/m5/validation_report.json` records successful validation of the three raw M5 inputs: 30,490 sales
  series, 1,969 calendar dates and 6,841,121 price rows, with no reported structural errors.
- Saved Stage 3, Stage 4, Stage 4.1 and Stage 5 metadata/metric JSON files support the split ranges, row counts,
  validation and holdout metrics, and promotion status stated above.
- `models/m5/retailiq_selected_model.json` identifies Stage 4 as the selected M5 inference model and says Stage 5
  was not promoted.
- `app.py`, M5 integration code and `.gitignore` were inspected for current dashboard sections, M5 data separation,
  and the exclusion of `data/m5/` from Git.

These are checks of saved reports and implementation, not fresh executions of M5 validation, training, inference or
dashboard tests during the documentation update. The original synthetic retail dataset and the Walmart M5 dataset
are separate. The M5 raw files are ignored by Git and are not asserted to be present on GitHub.

## Checks not performed for this documentation update

- No pytest suite, browser suite or dashboard test was executed while editing these documents.
- No model training, validation pipeline, final holdout evaluation, or forecast generation was run.
- Windows `setup.bat`, `run_retailiq.bat` and `train_models.bat` were not independently executed in the Linux
  verification environment. Windows installation and launch instructions therefore remain documented commands,
  not a Windows execution result.
- No fresh dependency installation or cross-platform visual review was performed.

The previous verification environment used Linux, Python 3.13.18 and the versions pinned in `requirements.txt`.
The current requirements specify Python 3.12 or 3.13 (64-bit). Narrow screens were not tuned in the earlier visual
review.
