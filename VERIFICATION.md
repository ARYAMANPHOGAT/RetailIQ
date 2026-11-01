# RetailIQ — verification report

Build environment: Linux container, Python 3.13.18, packages exactly as pinned in `requirements.txt`.
Windows itself was **not** available, so the `.bat` scripts were written for Windows but not executed.

## What was executed

| Check | Result |
|---|---|
| Raw CSV inspected (shape, columns, dtypes, dates, IDs, missing values, duplicates, continuity, correlations) | Done. 73,100 × 15, 2022-01-01 → 2024-01-01, 5 stores × 20 products, 0 missing, 0 duplicates, all 100 series daily and continuous |
| `python train.py` | Succeeded in about 35 s. Artifacts are in `models/` (version RIQ-20261031-1236) |
| Retraining from the extracted ZIP in a fresh venv | Succeeded with identical metrics (reproducible, `random_state=42`) |
| `pip install -r requirements.txt` in a fresh venv (from the ZIP) | Succeeded |
| `python -m pytest` (from the ZIP, fresh venv) | **35 passed, 0 failed** |
| App launch (`streamlit run app.py`) from the extracted ZIP | Started; health endpoint OK |
| Browser end-to-end run (Playwright + Chromium) against the packaged app | **42/42 checks passed** (list below) |

## Test layers

* **Unit tests** (`tests/test_pipeline.py`): validation and row-numbered errors, CSV/XLSX/corrupt/empty/unsupported
  files, lag and rolling correctness, order independence, leakage (changing target-day sales leaves features
  unchanged), gaps not filled, next-day row, Demand Forecast excluded, chronological split, date-grouped folds,
  selection by validation score, artifact load/predict/round-trip, missing and corrupt artifacts, unseen IDs,
  output schema, metrics, risk thresholds including zero demand and zero stock, complete-window profiles,
  K-Means/PCA assignment.
* **Integration tests** (`tests/test_storage_app.py`): SQLite save/read/delete, cascade, clear, NULL actuals for
  future dates, rollback of a failed save, CSV/PDF export, a failed training run keeping the last good models,
  Streamlit AppTest through all 7 sections, the app without models showing the train command, and the mode-switch
  confirmation and reset keeping history.
* **Browser E2E** (not shipped; it needs Playwright): welcome screen with both modes; Live uploads of a file with
  a missing column, a corrupt .xlsx, bad values (non-numeric, negative, invalid date, duplicate keys) and too-short
  history, all blocked with clear messages and no tracebacks; a valid .xlsx with messy headers, gaps, an unseen
  store and a zero-demand series analysed successfully; forecast and risk CSV exports checked; segmentation and
  model pages; mode-switch confirmation (cancel and confirm); demo run; PDF from Overview; reset keeping history;
  history filter, PDF and insights CSV; reopening a saved run; delete with confirmation; the full 73,100-row file
  uploaded in Live mode (analysed and saved in about 6 s, with an in-sample warning shown); the About dialog with
  the synthetic-data disclosure.

## Model results (holdout = final 20% of dates)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 88.75 | 108.20 | −0.0005 |
| Random Forest (selected by CV MAE 89.30 vs 89.36) | 88.78 | 108.21 | −0.0008 |
| Naive: yesterday's sales | 119.75 | 153.49 | −1.014 |
| Naive: 7-day mean | 93.19 | 115.45 | −0.139 |

R² ≈ 0 is the honest outcome. Next-day sales in this synthetic file are uncorrelated with past sales (≈ 0.00).

## Known limitations / manual steps

* Run `setup.bat`, then `run_retailiq.bat` on Windows. The scripts were not executed on Windows here.
* The pinned versions need Python 3.12+ (numpy 2.5.3). If a pinned wheel is unavailable on your machine, install
  the unpinned packages and run `python train.py` (see README › Troubleshooting).
* Visual checks were done on screenshots at 1440 px and 1600 px widths; very narrow screens were not tuned.
