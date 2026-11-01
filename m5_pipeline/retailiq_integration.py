"""Dataset-specific M5 inference used only by RetailIQ's M5 page."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C
from . import stage4_config as S4
from . import stage4_1_config as S41
from .features import _load_calendar
from .stage4_features import NEW_FEATURES, compute_intermittent_features
from .stage4_1_experiment import same_weekday_features
from .training import feature_matrix

SELECTION_MANIFEST = C.MODELS_DIR / "retailiq_selected_model.json"
SERIES_METADATA = S4.SERIES_METADATA_PATH
FEATURE_METADATA = S4.FEATURE_METADATA_PATH
M5_ID_COLS = list(C.SERIES_ID_COLUMNS)


def write_selected_model_manifest(model_variant: str) -> dict:
    """Point the separate M5 page at the model selected on validation."""
    candidates = {
        "stage4": {
            "model": S4.MODEL_DIR / "m5_lightgbm_stage4.txt",
            "metadata": S4.MODEL_DIR / "m5_training_metadata.json",
            "metrics": S4.RESULTS_DIR / "stage4_metrics.json",
            "test_metrics": S4.RESULTS_DIR / "stage4_test_metrics.json",
            "version": "M5 Stage 4 LightGBM",
        },
        "stage4_1": {
            "model": S41.MODEL_PATH,
            "metadata": S41.MODEL_METADATA_PATH,
            "metrics": S41.METRICS_PATH,
            "test_metrics": S41.RESULTS_DIR / "stage4_1_test_metrics.json",
            "version": "M5 Stage 4.1 LightGBM",
        },
    }
    if model_variant not in candidates:
        raise ValueError("model_variant must be stage4 or stage4_1")
    if SELECTION_MANIFEST.exists():
        raise FileExistsError(f"Refusing to overwrite selected-model manifest: {SELECTION_MANIFEST}")
    chosen = candidates[model_variant]
    required = (chosen["model"], chosen["metadata"], chosen["metrics"])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Selected M5 model files are missing: " + ", ".join(missing))
    manifest = {
        "model_variant": model_variant, "model_version": chosen["version"],
        "model_path": str(chosen["model"].resolve()),
        "metadata_path": str(chosen["metadata"].resolve()),
        "validation_metrics_path": str(chosen["metrics"].resolve()),
        "test_metrics_path": str(chosen["test_metrics"].resolve()) if chosen["test_metrics"].is_file() else None,
        "selected_by": ("Validation low-demand MAE/WAPE non-regression gate, then overall validation MAE; "
                        "final test excluded from selection."),
        "dataset": "Walmart M5 Forecasting; isolated from the synthetic RetailIQ upload pipeline.",
        "stage5_promoted": False,
    }
    SELECTION_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_deployment_info() -> tuple[dict, dict, dict | None]:
    if not SELECTION_MANIFEST.is_file():
        raise FileNotFoundError(f"M5 selected-model manifest not found: {SELECTION_MANIFEST}")
    manifest = json.loads(SELECTION_MANIFEST.read_text(encoding="utf-8"))
    model_path = Path(manifest["model_path"])
    metadata_path = Path(manifest["metadata_path"])
    metrics_path = Path(manifest["validation_metrics_path"])
    for path in (model_path, metadata_path, metrics_path, SERIES_METADATA, FEATURE_METADATA):
        if not path.is_file():
            raise FileNotFoundError(f"M5 model or metadata file is missing: {path}")
    return (manifest, json.loads(metadata_path.read_text(encoding="utf-8")),
            json.loads(metrics_path.read_text(encoding="utf-8")))


def _load_cutoff_price_matrix(calendar: dict, pair_to_index: dict, n_series: int,
                              cutoff_position: int) -> tuple[np.ndarray, int]:
    """Load only sell-price weeks known by the prediction cutoff."""
    max_week = int(calendar["day_week_index"][cutoff_position])
    weeks = calendar["week_ids"][:max_week + 1]
    week_to_index = {week: idx for idx, week in enumerate(weeks)}
    matrix = np.full((n_series, len(weeks)), np.nan, dtype=np.float32)
    path = C.PRICES_PATH
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            week_idx = week_to_index.get(row["wm_yr_wk"])
            if week_idx is None:
                continue
            series_idx = pair_to_index.get((row["store_id"], row["item_id"]))
            if series_idx is not None:
                matrix[series_idx, week_idx] = np.float32(row["sell_price"])
    all_weeks = np.arange(matrix.shape[1])
    for idx in range(n_series):
        known = np.flatnonzero(np.isfinite(matrix[idx]))
        if known.size:
            previous = np.searchsorted(known, all_weeks, side="right") - 1
            available = previous >= 0
            matrix[idx, available] = matrix[idx, known[previous[available]]]
    return matrix, max_week


def _forecast_feature_columns(sales: np.ndarray, rows: pd.DataFrame, calendar: dict,
                              prices: np.ndarray, target_position: int) -> dict[str, np.ndarray]:
    n = len(sales)
    day = target_position + 1
    out: dict[str, np.ndarray] = {
        "item_idx": rows.item_id_idx.to_numpy(np.uint16),
        "store_idx": rows.store_id_idx.to_numpy(np.uint16),
        "state_idx": rows.state_id_idx.to_numpy(np.uint16),
        "dept_idx": rows.dept_id_idx.to_numpy(np.uint16),
        "cat_idx": rows.cat_id_idx.to_numpy(np.uint16),
        "target_day": np.full(n, day, dtype=np.uint16),
    }
    for lag in (1, 7, 14, 28):
        out[f"lag_{lag}"] = sales[:, target_position - lag].astype(np.int16, copy=False)
    for window in (7, 28):
        hist = sales[:, target_position-window:target_position].astype(np.float64)
        out[f"roll_mean_{window}"] = hist.mean(axis=1).astype(np.float32)
        out[f"roll_std_{window}"] = hist.std(axis=1, ddof=1).astype(np.float32)

    for name, key in (("target_dow", "dow"), ("target_wday", "wday"),
                      ("target_month", "month"), ("target_year", "year"),
                      ("snap_CA", "snap_CA"), ("snap_TX", "snap_TX"), ("snap_WI", "snap_WI")):
        out[name] = np.full(n, calendar[key][target_position], dtype=np.asarray(calendar[key]).dtype)
    for column in ("event_name_1", "event_type_1", "event_name_2", "event_type_2"):
        out[column] = np.full(n, calendar[column][target_position], dtype=np.uint16)
    state = rows.state_id.astype(str).to_numpy()
    snap_state = np.zeros(n, dtype=np.uint8)
    for key in ("CA", "TX", "WI"):
        mask = state == key
        snap_state[mask] = calendar[f"snap_{key}"][target_position]
    out["snap_state"] = snap_state

    cutoff_position = target_position - 1
    week = int(calendar["day_week_index"][cutoff_position])
    previous_week = max(week - 1, 0)
    current = prices[:, week]
    previous = prices[:, previous_week]
    out["price_at_cutoff"] = current.astype(np.float32)
    out["price_prev_week"] = previous.astype(np.float32)
    out["price_missing"] = (~np.isfinite(current)).astype(np.uint8)
    out["price_change_pct"] = np.divide(current - previous, previous,
        out=np.full(n, np.nan, dtype=np.float32), where=previous > 0)

    padded = np.pad(sales, ((0, 0), (0, 1)), mode="constant")
    out.update(compute_intermittent_features(padded, np.asarray([target_position])))
    return out


def build_next_day_forecasts() -> pd.DataFrame:
    """Predict M5 d_(last observed + 1) using only M5-specific files."""
    manifest, model_meta, _ = load_deployment_info()
    sales_path = C.SALES_PATH
    calendar_path = C.CALENDAR_PATH
    for path in (sales_path, calendar_path, C.PRICES_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required M5 input is missing: {path}")
    series = pd.read_csv(SERIES_METADATA, dtype={"id": "string", "item_id": "string", "store_id": "string"})
    input_ids = pd.read_csv(sales_path, usecols=M5_ID_COLS, dtype="string")
    if len(input_ids) != len(series) or not input_ids["id"].astype(str).equals(series["id"].astype(str)):
        raise ValueError("M5 sales series ordering does not match the selected model's series metadata")
    calendar, current_event_codebooks = _load_calendar()
    headers = pd.read_csv(sales_path, nrows=0).columns
    day_ids = sorted(int(col[2:]) for col in headers if col.startswith("d_") and col[2:].isdigit())
    if not day_ids or day_ids != list(range(1, day_ids[-1] + 1)):
        raise ValueError("M5 sales day columns are not a continuous d_1..d_N sequence")
    last_day = day_ids[-1]
    target_position = last_day
    if target_position >= len(calendar["dates"]):
        raise ValueError("M5 calendar does not include the next forecast day")
    feature_metadata = json.loads(FEATURE_METADATA.read_text(encoding="utf-8"))
    if feature_metadata["event_codebooks"] != current_event_codebooks:
        raise ValueError("M5 calendar event encoding differs from the selected model metadata")
    pair_to_index = {(row.store_id, row.item_id): int(row.series_idx)
                     for row in series[["store_id", "item_id", "series_idx"]].itertuples(index=False)}
    prices, _ = _load_cutoff_price_matrix(calendar, pair_to_index, len(series), target_position - 1)
    model = lgb.Booster(model_file=manifest["model_path"])
    feature_names = list(model_meta["feature_names"])
    if model.num_feature() != len(feature_names):
        raise ValueError("Selected M5 model feature count differs from its metadata")
    added = {"same_weekday_nonzero_rate_3_lags", "same_weekday_positive_mean_3_lags"}
    daily_columns = [f"d_{d}" for d in range(1, last_day + 1)]
    outputs = []
    offset = 0
    for frame in pd.read_csv(sales_path, usecols=daily_columns, dtype=np.int16, chunksize=256):
        sales = frame.to_numpy(copy=True)
        rows = series.iloc[offset:offset + len(sales)]
        columns = _forecast_feature_columns(sales, rows, calendar,
                                            prices[offset:offset + len(sales)], target_position)
        if added.intersection(feature_names):
            columns.update(same_weekday_features(columns["lag_7"], columns["lag_14"], columns["lag_28"]))
        matrix = feature_matrix(columns, feature_names)
        prediction = np.maximum(0, model.predict(matrix)).astype(np.float32)
        outputs.append(pd.DataFrame({
            "series_idx": rows.series_idx.to_numpy(np.uint32), "series_id": rows.id.astype(str),
            "item_id": rows.item_id.astype(str), "store_id": rows.store_id.astype(str),
            "state_id": rows.state_id.astype(str), "category": rows.cat_id.astype(str),
            "forecast_date": pd.Timestamp(calendar["dates"][target_position]),
            "predicted_units": prediction,
        }))
        offset += len(sales)
    if offset != len(series):
        raise RuntimeError(f"Forecasted {offset} M5 series; expected {len(series)}")
    result = pd.concat(outputs, ignore_index=True)
    if not np.isfinite(result.predicted_units.to_numpy()).all():
        raise RuntimeError("M5 forecast contains nonfinite predictions")
    result.attrs["model_version"] = manifest["model_version"]
    result.attrs["forecast_day"] = target_position + 1
    return result
