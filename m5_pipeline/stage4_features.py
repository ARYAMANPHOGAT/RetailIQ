"""Stage 4 cutoff-safe intermittent-demand features, isolated from Stage 2."""
from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import stage4_config as S4
from .features import _load_calendar, _peak_rss_mb, _series_table, _make_feature_block, _save_split_shard

NEW_FEATURES = [*(f"nonzero_count_{w}" for w in S4.COUNT_WINDOWS),
                *(f"nonzero_rate_{w}" for w in S4.COUNT_WINDOWS),
                *(f"positive_mean_{w}" for w in S4.POSITIVE_MEAN_WINDOWS),
                "days_since_last_sale", "no_prior_sale",
                *(f"roll_mean_{w}" for w in S4.LATE_WINDOWS),
                *(f"roll_std_{w}" for w in S4.LATE_WINDOWS),
                *(f"demand_cv_{w}" for w in S4.LATE_WINDOWS)]


def _logger() -> logging.Logger:
    S4.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("m5.stage4.features")
    if not logger.handlers:
        handler = logging.FileHandler(S4.FEATURE_LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def compute_intermittent_features(sales: np.ndarray, target_positions: np.ndarray) -> dict[str, np.ndarray]:
    """Compute features from complete windows strictly before target positions.

    Target position is the zero-based target column; the latest eligible sale
    is position - 1. Long-window features remain NaN until a full window is
    available. No positive history yields NaN days-since and no_prior_sale=1.
    """
    values = np.asarray(sales)
    positions = np.asarray(target_positions, dtype=np.int64)
    if (values.ndim != 2 or positions.size == 0 or np.any(positions < 1)
            or np.any(positions >= values.shape[1])):
        raise ValueError("sales must be 2-D and target positions must have a prior cutoff day")
    # Slice before any cumulative operation: target-day and future cells are
    # not even inputs to these feature calculations.
    history = values[:, :int(positions.max())]
    positive = history > 0
    positive_i64 = np.where(positive, history, 0).astype(np.int64, copy=False)
    count_prefix = np.pad(np.cumsum(positive, axis=1, dtype=np.int32), ((0, 0), (1, 0)))
    sum_prefix = np.pad(np.cumsum(history, axis=1, dtype=np.int64), ((0, 0), (1, 0)))
    square_prefix = np.pad(np.cumsum(history.astype(np.int64) ** 2, axis=1, dtype=np.int64),
                           ((0, 0), (1, 0)))
    positive_sum_prefix = np.pad(np.cumsum(positive_i64, axis=1, dtype=np.int64), ((0, 0), (1, 0)))
    out: dict[str, np.ndarray] = {}

    for window in S4.COUNT_WINDOWS:
        counts = np.full((values.shape[0], len(positions)), np.nan, dtype=np.float32)
        valid = positions >= window
        if np.any(valid):
            p = positions[valid]
            counts[:, valid] = count_prefix[:, p] - count_prefix[:, p - window]
        out[f"nonzero_count_{window}"] = counts.reshape(-1)
        out[f"nonzero_rate_{window}"] = (counts / np.float32(window)).reshape(-1).astype(np.float32)

    for window in S4.POSITIVE_MEAN_WINDOWS:
        positive_means = np.full((values.shape[0], len(positions)), np.nan, dtype=np.float32)
        valid = positions >= window
        if np.any(valid):
            p = positions[valid]
            counts = count_prefix[:, p] - count_prefix[:, p - window]
            sums = positive_sum_prefix[:, p] - positive_sum_prefix[:, p - window]
            means = np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=np.float64), where=counts > 0)
            positive_means[:, valid] = means.astype(np.float32)
        out[f"positive_mean_{window}"] = positive_means.reshape(-1)

    cutoff_positions = positions - 1
    day_indices = np.arange(history.shape[1], dtype=np.int32)
    latest_positive = np.maximum.accumulate(np.where(positive, day_indices[None, :], -1), axis=1)
    last_sale_indices = latest_positive[:, cutoff_positions]
    no_prior = last_sale_indices < 0
    days_since = (cutoff_positions[None, :] - last_sale_indices).astype(np.float32)
    days_since[no_prior] = np.nan
    out["days_since_last_sale"] = days_since.reshape(-1)
    out["no_prior_sale"] = no_prior.reshape(-1).astype(np.uint8)

    for window in S4.LATE_WINDOWS:
        means = np.full((values.shape[0], len(positions)), np.nan, dtype=np.float32)
        stds = np.full_like(means, np.nan)
        cvs = np.full_like(means, np.nan)
        valid = positions >= window
        if np.any(valid):
            p = positions[valid]
            sums = sum_prefix[:, p] - sum_prefix[:, p - window]
            squares = square_prefix[:, p] - square_prefix[:, p - window]
            mean = sums / window
            variance = np.maximum(0.0, (squares - sums.astype(np.float64) ** 2 / window) / (window - 1))
            std = np.sqrt(variance)
            cv = np.divide(std, mean, out=np.full_like(std, np.nan), where=mean > 0)
            means[:, valid] = mean.astype(np.float32)
            stds[:, valid] = std.astype(np.float32)
            cvs[:, valid] = cv.astype(np.float32)
        out[f"roll_mean_{window}"] = means.reshape(-1)
        out[f"roll_std_{window}"] = stds.reshape(-1)
        out[f"demand_cv_{window}"] = cvs.reshape(-1)
    return out


def _price_matrix_through_cutoff(calendar: dict[str, object], pair_to_index: dict,
                                 n_series: int) -> np.ndarray:
    """Build weekly prices only through the last validation prediction cutoff."""
    cutoff_position = S4.VALIDATION_END - 2  # d_1912 as a zero-based calendar position
    max_week = int(calendar["day_week_index"][cutoff_position])
    weeks = calendar["week_ids"][:max_week + 1]
    week_to_index = {week: index for index, week in enumerate(weeks)}
    matrix = np.full((n_series, len(weeks)), np.nan, dtype=np.float32)
    with C.PRICES_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
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
    return matrix


def required_feature_disk_bytes() -> tuple[int, int]:
    """Conservative estimate from existing Stage 2 train/validation shard sizes."""
    base_manifest = json.loads(C.FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
    base_paths = [path for split in ("train", "validation")
                  for path in (C.FEATURES_DIR / split).glob("part_*.npz")]
    if not base_paths:
        raise FileNotFoundError("Stage 2 train/validation feature shards are required for disk estimation")
    base_bytes = sum(path.stat().st_size for path in base_paths)
    base_columns = len(base_manifest["features"])
    stage4_columns = base_columns + len(NEW_FEATURES)
    estimate = int(base_bytes * (stage4_columns / base_columns) * S4.DISK_SIZE_SAFETY_FACTOR
                   + S4.DISK_RESERVE_BYTES)
    return estimate, base_bytes


def require_disk_space(estimated_bytes: int, available_bytes: int) -> None:
    if available_bytes < estimated_bytes:
        raise OSError(f"Insufficient disk space for Stage 4 features: need about {estimated_bytes:,} bytes, "
                      f"only {available_bytes:,} bytes are free")


def generate_stage4_features() -> dict:
    """Write Stage 4 train/validation features after a free-space check.

    No target rows for d_1914 onward are generated or saved.
    """
    if S4.FEATURES_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite Stage 4 features: {S4.FEATURES_DIR}")
    staging = S4.STAGE4_ROOT / "_features_staging"
    if staging.exists():
        raise FileExistsError(f"Stage 4 staging directory already exists: {staging}")
    estimate, stage2_bytes = required_feature_disk_bytes()
    free_bytes = shutil.disk_usage(C.OUTPUTS_DIR).free
    require_disk_space(estimate, free_bytes)

    logger = _logger()
    started = time.time()
    logger.info("Disk check passed: free=%d estimated_required=%d stage2_bytes=%d",
                free_bytes, estimate, stage2_bytes)
    staging.mkdir(parents=True)
    split_dirs = {name: staging / name for name in ("train", "validation")}
    for path in split_dirs.values():
        path.mkdir()

    calendar, event_codebooks = _load_calendar()
    series, pair_to_index, entity_codebooks = _series_table()
    prices = _price_matrix_through_cutoff(calendar, pair_to_index, len(series))
    series_cols = ["series_idx", "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
                   "item_id_idx", "store_id_idx", "state_id_idx", "dept_id_idx", "cat_id_idx"]
    series[series_cols].to_csv(staging / "series_metadata.csv", index=False)

    last_read_day = S4.VALIDATION_END  # validation labels are known through d_1913
    daily_columns = [f"d_{day}" for day in range(1, last_read_day + 1)]
    rows = {name: 0 for name in split_dirs}
    shards = {name: 0 for name in split_dirs}
    series_start = 0
    peak = _peak_rss_mb()
    logger.info("Generating Stage 4 features for %d series in %d-row blocks", len(series), S4.SERIES_BATCH_SIZE)
    try:
        for batch_number, frame in enumerate(pd.read_csv(
                C.SALES_PATH, usecols=daily_columns, dtype=np.int16, chunksize=S4.SERIES_BATCH_SIZE)):
            sales = frame.to_numpy(dtype=np.int16, copy=True)
            end = series_start + len(sales)
            series_batch = series.iloc[series_start:end]
            base = _make_feature_block(sales, series_batch, calendar, prices, series_start)
            positions = np.arange(28, sales.shape[1], dtype=np.int64)
            extra = compute_intermittent_features(sales, positions)
            block = {**base, **extra}
            day = block["target_day"]
            masks = {"train": day <= S4.TRAIN_END,
                     "validation": (day >= S4.VALIDATION_START) & (day <= S4.VALIDATION_END)}
            for split, mask in masks.items():
                count = _save_split_shard(split_dirs[split], shards[split], block, mask)
                if count:
                    rows[split] += count
                    shards[split] += 1
            series_start = end
            current = _peak_rss_mb()
            if current is not None:
                peak = current if peak is None else max(peak, current)
            if (batch_number + 1) % 10 == 0 or series_start == len(series):
                logger.info("Processed %d/%d series", series_start, len(series))
        if series_start != len(series):
            raise RuntimeError(f"Processed {series_start} series; expected {len(series)}")

        expected = {"train": len(series) * (S4.TRAIN_END - 28),
                    "validation": len(series) * (S4.VALIDATION_END - S4.VALIDATION_START + 1)}
        if rows != expected:
            raise RuntimeError(f"Stage 4 split row counts mismatch: {rows} expected {expected}")
        base_metadata = json.loads(C.FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
        metadata = {
            "status": "complete", "series_count": len(series), "series_batch_size": S4.SERIES_BATCH_SIZE,
            "split": {
                "train": {"target_days": [29, S4.TRAIN_END], "rows": rows["train"], "shards": shards["train"]},
                "validation": {"target_days": [S4.VALIDATION_START, S4.VALIDATION_END],
                               "rows": rows["validation"], "shards": shards["validation"]},
            },
            "features": [*base_metadata["features"][:-1], *NEW_FEATURES, "target"],
            "new_features": NEW_FEATURES,
            "window_semantics": {
                "all_sales_windows": "End at target_day - 1; full calendar windows only.",
                "positive_mean": "Mean sales conditional on sales > 0 in the stated full window; NaN if none.",
                "days_since_last_sale": "Cutoff position minus most recent positive-sales position; NaN if none before cutoff.",
                "no_prior_sale": "1 if no positive sale exists before cutoff, else 0.",
                "demand_cv": "Sample rolling standard deviation divided by rolling mean; NaN when mean is zero or window incomplete.",
            },
            "missing_value_policy": "Keep incomplete-window and no-positive-history features as NaN; LightGBM native missing handling; no imputation.",
            "feature_cutoff": "Feature values stop at each target's preceding cutoff; validation labels end at d_1913. No d_1914+ sales columns or target rows are read/generated.",
            "event_codebooks": event_codebooks, "entity_codebooks": entity_codebooks,
            "disk_check": {"free_bytes_before_generation": free_bytes,
                           "estimated_required_bytes": estimate, "stage2_train_validation_bytes": stage2_bytes},
            "training_models_fit": False, "peak_process_rss_mb": peak,
            "generation_seconds": round(time.time() - started, 2),
        }
        (staging / "feature_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        staging.rename(S4.FEATURES_DIR)
        logger.info("Stage 4 feature generation complete: %s", rows)
        return metadata
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    print(json.dumps(generate_stage4_features(), indent=2))
