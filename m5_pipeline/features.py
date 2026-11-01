"""Bounded-memory, cutoff-safe feature generation for Walmart M5."""
from __future__ import annotations

import csv
import ctypes
import json
import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

LAGS = (1, 7, 14, 28)
ROLL_WINDOWS = (7, 28)
TEST_DAYS = set(range(C.TEST_DAY_START, C.TEST_DAY_END + 1))
VALIDATION_START = C.TEST_DAY_START - C.VALIDATION_DAYS
VALIDATION_END = C.TEST_DAY_START - 1
TRAIN_END = VALIDATION_START - 1

SALES_FEATURES = [*(f"lag_{lag}" for lag in LAGS),
                  *(f"roll_mean_{window}" for window in ROLL_WINDOWS),
                  *(f"roll_std_{window}" for window in ROLL_WINDOWS)]
CALENDAR_FEATURES = ["target_dow", "target_wday", "target_month", "target_year",
                     "event_name_1", "event_type_1", "event_name_2", "event_type_2",
                     "snap_CA", "snap_TX", "snap_WI", "snap_state"]
PRICE_FEATURES = ["price_at_cutoff", "price_prev_week", "price_change_pct", "price_missing"]
ID_FEATURES = ["series_idx", "item_idx", "store_idx", "state_idx", "dept_idx", "cat_idx"]
ALL_FEATURES = ID_FEATURES + ["target_day"] + SALES_FEATURES + CALENDAR_FEATURES + PRICE_FEATURES


def get_logger() -> logging.Logger:
    C.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("m5.features")
    if not logger.handlers:
        handler = logging.FileHandler(C.FEATURE_LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _peak_rss_mb() -> float | None:
    """Return process peak working set where the host exposes it."""
    try:
        if os.name == "nt":
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
            return counters.PeakWorkingSetSize / (1024 * 1024) if ok else None
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / 1024 if value > 10_000 else value / (1024 * 1024)
    except (AttributeError, OSError, ImportError, OverflowError, TypeError):
        return None


def temporal_split_masks(target_days: np.ndarray) -> dict[str, np.ndarray]:
    """Split target day IDs: validation is the 28 days before d_1914."""
    days = np.asarray(target_days)
    return {
        "train": days <= VALIDATION_START - 1,
        "validation": (days >= VALIDATION_START) & (days <= VALIDATION_END),
        "test": (days >= C.TEST_DAY_START) & (days <= C.TEST_DAY_END),
    }


def compute_sales_features(sales: np.ndarray, target_positions: np.ndarray) -> dict[str, np.ndarray]:
    """Return flattened labels and historical features for 2-D int sales.

    A position denotes the target's zero-based column in the observed sales
    matrix. All source slices end at target_position - 1, the prediction
    cutoff. Target values are copied only into `target`.
    """
    values = np.asarray(sales)
    positions = np.asarray(target_positions, dtype=np.int64)
    if values.ndim != 2 or np.any(positions < max(LAGS)) or np.any(positions >= values.shape[1]):
        raise ValueError("Sales must be 2-D and target positions must have 28 observed history days")
    out: dict[str, np.ndarray] = {}
    for lag in LAGS:
        out[f"lag_{lag}"] = values[:, positions - lag].reshape(-1).astype(np.int16, copy=False)
    prefix = np.pad(np.cumsum(values, axis=1, dtype=np.int64), ((0, 0), (1, 0)))
    squared_prefix = np.pad(np.cumsum(values.astype(np.int64) ** 2, axis=1, dtype=np.int64),
                            ((0, 0), (1, 0)))
    for window in ROLL_WINDOWS:
        end = positions
        start = positions - window
        sums = prefix[:, end] - prefix[:, start]
        square_sums = squared_prefix[:, end] - squared_prefix[:, start]
        mean = sums / window
        variance = np.maximum(0.0, (square_sums - (sums.astype(np.float64) ** 2 / window)) / (window - 1))
        out[f"roll_mean_{window}"] = mean.reshape(-1).astype(np.float32)
        out[f"roll_std_{window}"] = np.sqrt(variance).reshape(-1).astype(np.float32)
    out["target"] = values[:, positions].reshape(-1).astype(np.int16, copy=False)
    return out


def _categorical_codes(values: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    labels = sorted({v for v in values if v})
    codes = {value: index + 1 for index, value in enumerate(labels)}
    return np.asarray([codes.get(v, 0) for v in values], dtype=np.uint16), codes


def _load_calendar() -> tuple[dict[str, object], dict[str, dict[str, int]]]:
    with C.CALENDAR_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda row: int(row["d"].removeprefix("d_")))
    day_ids = [int(row["d"].removeprefix("d_")) for row in rows]
    if day_ids != list(range(1, len(rows) + 1)):
        raise ValueError("Calendar day IDs are not a continuous d_1...d_N sequence")
    date_values = [pd.Timestamp(row["date"]) for row in rows]
    if any((right - left).days != 1 for left, right in zip(date_values, date_values[1:])):
        raise ValueError("Calendar dates are not continuous")

    codebooks: dict[str, dict[str, int]] = {}
    arrays: dict[str, object] = {
        "dates": date_values,
        "weeks": [row["wm_yr_wk"] for row in rows],
        "dow": np.asarray([date.dayofweek for date in date_values], dtype=np.uint8),
        "wday": np.asarray([int(row["wday"]) for row in rows], dtype=np.uint8),
        "month": np.asarray([date.month for date in date_values], dtype=np.uint8),
        "year": np.asarray([date.year for date in date_values], dtype=np.uint16),
    }
    for column in ("event_name_1", "event_type_1", "event_name_2", "event_type_2"):
        values = [row.get(column, "") or "" for row in rows]
        codes, codebook = _categorical_codes(values)
        arrays[column] = codes
        codebooks[column] = codebook
    for column in ("snap_CA", "snap_TX", "snap_WI"):
        arrays[column] = np.asarray([int(row.get(column, "0") or 0) for row in rows], dtype=np.uint8)
    arrays["week_ids"] = sorted(set(arrays["weeks"]), key=lambda value: min(
        i for i, week in enumerate(arrays["weeks"]) if week == value))
    week_index = {week: index for index, week in enumerate(arrays["week_ids"])}
    arrays["day_week_index"] = np.asarray([week_index[week] for week in arrays["weeks"]], dtype=np.uint16)
    return arrays, codebooks


def _series_table() -> tuple[pd.DataFrame, dict[tuple[str, str], int], dict[str, dict[str, int]]]:
    id_columns = list(C.SERIES_ID_COLUMNS)
    frame = pd.read_csv(C.SALES_PATH, usecols=id_columns, dtype="string")
    if frame[["id", "item_id", "store_id"]].isna().any().any():
        raise ValueError("Sales series metadata contains missing identifiers")
    if frame.duplicated(["store_id", "item_id"]).any():
        raise ValueError("Expected exactly one sales series for each store-item pair")
    frame = frame.reset_index(drop=True)
    category_maps: dict[str, dict[str, int]] = {}
    for col in ("item_id", "store_id", "state_id", "dept_id", "cat_id"):
        labels = sorted(frame[col].astype(str).unique())
        category_maps[col] = {value: index for index, value in enumerate(labels)}
        frame[f"{col}_idx"] = frame[col].astype(str).map(category_maps[col]).astype(np.uint16)
    frame.insert(0, "series_idx", np.arange(len(frame), dtype=np.uint32))
    pair_to_index = {(row.store_id, row.item_id): int(row.series_idx)
                     for row in frame[["store_id", "item_id", "series_idx"]].itertuples(index=False)}
    return frame, pair_to_index, category_maps


def _load_price_matrix(calendar: dict[str, object], pair_to_index: dict[tuple[str, str], int], n_series: int) -> np.ndarray:
    weeks: list[str] = calendar["week_ids"]
    week_to_index = {week: index for index, week in enumerate(weeks)}
    matrix = np.full((n_series, len(weeks)), np.nan, dtype=np.float32)
    with C.PRICES_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            series_idx = pair_to_index.get((row["store_id"], row["item_id"]))
            if series_idx is None:
                continue
            week_idx = week_to_index.get(row["wm_yr_wk"])
            if week_idx is not None:
                matrix[series_idx, week_idx] = np.float32(row["sell_price"])
    # Forward-fill only within each series. A price is not invented before its
    # first observed week; those early features remain NaN for later handling.
    all_weeks = np.arange(matrix.shape[1])
    for idx in range(n_series):
        known = np.flatnonzero(np.isfinite(matrix[idx]))
        if not known.size:
            continue
        take = np.searchsorted(known, all_weeks, side="right") - 1
        available = take >= 0
        matrix[idx, available] = matrix[idx, known[take[available]]]
    return matrix


def _make_feature_block(
    sales: np.ndarray,
    series_rows: pd.DataFrame,
    calendar: dict[str, object],
    price_matrix: np.ndarray,
    series_start: int,
) -> dict[str, np.ndarray]:
    positions = np.arange(28, min(C.LAST_SALES_DAY, sales.shape[1]), dtype=np.int64)
    n_dates = len(positions)
    days = positions + 1
    base = compute_sales_features(sales, positions)
    n_series = sales.shape[0]
    out: dict[str, np.ndarray] = {
        "series_idx": np.repeat(np.arange(series_start, series_start + n_series, dtype=np.uint32), n_dates),
        "target_day": np.tile(days.astype(np.uint16), n_series),
    }
    for old, new in (("item_id_idx", "item_idx"), ("store_id_idx", "store_idx"),
                     ("state_id_idx", "state_idx"), ("dept_id_idx", "dept_idx"),
                     ("cat_id_idx", "cat_idx")):
        out[new] = np.repeat(series_rows[old].to_numpy(dtype=np.uint16), n_dates)
    for name, array in base.items():
        out[name] = array

    for name, key in (("target_dow", "dow"), ("target_wday", "wday"),
                      ("target_month", "month"), ("target_year", "year"),
                      ("event_name_1", "event_name_1"), ("event_type_1", "event_type_1"),
                      ("event_name_2", "event_name_2"), ("event_type_2", "event_type_2"),
                      ("snap_CA", "snap_CA"), ("snap_TX", "snap_TX"), ("snap_WI", "snap_WI")):
        out[name] = np.tile(calendar[key][positions], n_series)
    state_codes = series_rows["state_id"].astype(str).to_numpy()
    snap_state = np.zeros((n_series, n_dates), dtype=np.uint8)
    for row_idx, state in enumerate(state_codes):
        if state in ("CA", "TX", "WI"):
            snap_state[row_idx] = calendar[f"snap_{state}"][positions]
    out["snap_state"] = snap_state.reshape(-1)

    cutoff_positions = positions - 1
    week_positions = calendar["day_week_index"][cutoff_positions].astype(np.int64)
    previous_week_positions = np.maximum(week_positions - 1, 0)
    batch_prices = price_matrix[series_start:series_start + n_series]
    current_prices = batch_prices[:, week_positions]
    previous_prices = batch_prices[:, previous_week_positions]
    out["price_at_cutoff"] = current_prices.reshape(-1).astype(np.float32)
    out["price_prev_week"] = previous_prices.reshape(-1).astype(np.float32)
    out["price_missing"] = (~np.isfinite(current_prices)).reshape(-1).astype(np.uint8)
    change = np.divide(current_prices - previous_prices, previous_prices,
                        out=np.full_like(current_prices, np.nan), where=previous_prices > 0)
    out["price_change_pct"] = change.reshape(-1).astype(np.float32)
    return out


def _save_split_shard(split_dir: Path, shard_index: int, features: dict[str, np.ndarray],
                      mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    # Select row arrays in small blocks from the same batch to keep transient
    # allocations bounded while writing compressed, independently readable files.
    selected = {key: value[mask] for key, value in features.items()}
    path = split_dir / f"part_{shard_index:05d}.npz"
    np.savez_compressed(path, **selected)
    return int(mask.sum())


def generate_feature_dataset() -> dict:
    """Generate train/validation/test feature shards without wide-to-long melt."""
    if C.FEATURES_DIR.exists() and any(C.FEATURES_DIR.iterdir()):
        if (C.FEATURE_METADATA_PATH.exists()
                and json.loads(C.FEATURE_METADATA_PATH.read_text(encoding="utf-8")).get("status") == "complete"):
            raise FileExistsError(f"Refusing to overwrite complete M5 feature artifacts: {C.FEATURES_DIR}")
        # An interrupted run can leave partial shards. The directory is an
        # M5-only generated artifact location; clear it before a clean retry.
        shutil.rmtree(C.FEATURES_DIR)
    logger = get_logger()
    started = time.time()
    C.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    split_dirs = {name: C.FEATURES_DIR / name for name in ("train", "validation", "test")}
    for directory in split_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    logger.info("Preparing calendar, series map, and weekly prices")
    calendar, event_codebooks = _load_calendar()
    series, pair_to_index, category_maps = _series_table()
    price_matrix = _load_price_matrix(calendar, pair_to_index, len(series))
    series_export_cols = ["series_idx", "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
                          "item_id_idx", "store_id_idx", "state_id_idx", "dept_id_idx", "cat_id_idx"]
    series[series_export_cols].to_csv(C.SERIES_METADATA_PATH, index=False)

    daily_cols = [f"d_{i}" for i in range(1, C.LAST_SALES_DAY + 1)]
    row_counts = {name: 0 for name in split_dirs}
    shard_counts = {name: 0 for name in split_dirs}
    next_start = 0
    peak_rss_mb = _peak_rss_mb()
    logger.info("Generating features for %d series in batches of %d", len(series), C.SERIES_BATCH_SIZE)
    for batch_number, frame in enumerate(pd.read_csv(
            C.SALES_PATH, usecols=daily_cols, dtype=np.int16, chunksize=C.SERIES_BATCH_SIZE)):
        sales = frame.to_numpy(dtype=np.int16, copy=True)
        batch_end = next_start + len(sales)
        series_batch = series.iloc[next_start:batch_end]
        feature_block = _make_feature_block(sales, series_batch, calendar, price_matrix, next_start)
        days = feature_block["target_day"]
        masks = temporal_split_masks(days)
        for split, mask in masks.items():
            written = _save_split_shard(split_dirs[split], shard_counts[split], feature_block, mask)
            if written:
                row_counts[split] += written
                shard_counts[split] += 1
        next_start = batch_end
        current_peak = _peak_rss_mb()
        if current_peak is not None:
            peak_rss_mb = current_peak if peak_rss_mb is None else max(peak_rss_mb, current_peak)
        if (batch_number + 1) % 10 == 0 or next_start == len(series):
            logger.info("Processed %d/%d series (%d batches)", next_start, len(series), batch_number + 1)
    if next_start != len(series):
        raise RuntimeError(f"Sales row count changed during generation: processed {next_start}, expected {len(series)}")

    expected_days = {
        "train": TRAIN_END - 28,
        "validation": C.VALIDATION_DAYS,
        "test": C.TEST_DAY_END - C.TEST_DAY_START + 1,
    }
    expected_rows = {name: len(series) * days for name, days in expected_days.items()}
    if row_counts != expected_rows:
        raise RuntimeError(f"Generated row counts do not match the planned temporal split: {row_counts} vs {expected_rows}")

    metadata = {
        "status": "complete",
        "source": {"sales": str(C.SALES_PATH.relative_to(C.ROOT_DIR)),
                   "calendar": str(C.CALENDAR_PATH.relative_to(C.ROOT_DIR)),
                   "prices": str(C.PRICES_PATH.relative_to(C.ROOT_DIR))},
        "series_count": len(series),
        "series_batch_size": C.SERIES_BATCH_SIZE,
        "split": {
            "train": {"target_days": [29, TRAIN_END], "rows": row_counts["train"], "shards": shard_counts["train"]},
            "validation": {"target_days": [VALIDATION_START, VALIDATION_END],
                           "rows": row_counts["validation"], "shards": shard_counts["validation"]},
            "test": {"target_days": [C.TEST_DAY_START, C.TEST_DAY_END],
                     "rows": row_counts["test"], "shards": shard_counts["test"]},
        },
        "features": ALL_FEATURES + ["target"],
        "dtypes": {"series_idx": "uint32", "target_day": "uint16", "target": "int16",
                   "lags": "int16", "rolling": "float32", "calendar_codes": "uint16",
                   "calendar_numbers": "uint16/uint8", "prices": "float32",
                   "price_missing": "uint8", "entity_codes": "uint16"},
        "lag_days": list(LAGS),
        "rolling_windows": list(ROLL_WINDOWS),
        "leakage_controls": [
            "Lag and rolling features stop at the cutoff day (target day minus one).",
            "Price features use the price effective in the cutoff week's calendar week, and the preceding week's price.",
            "Target-day calendar and SNAP fields are known calendar inputs; no target-day sales or price is used.",
            "Test labels are d_1914 through d_1941 and are not used to define earlier training or validation features.",
        ],
        "event_codebooks": event_codebooks,
        "entity_codebooks": category_maps,
        "price_missing_feature_rows": None,
        "missing_value_policy": {
            "price_missing": "Explicit 0/1 indicator: 1 when no historical price is available by the cutoff.",
            "LightGBM": "Pass NaN price features through; LightGBM handles missing numeric values natively.",
            "Linear Regression": "At model-fit time, median-impute each numeric price feature using training rows only; retain price_missing.",
            "Random Forest": "At model-fit time, median-impute each numeric price feature using training rows only; retain price_missing for consistent cross-version behavior.",
            "leakage_rule": "Never calculate imputation statistics from validation/test rows or from later dates.",
        },
        "training_models_fit": False,
        "generation_seconds": round(time.time() - started, 2),
        "peak_process_rss_mb": round(peak_rss_mb, 1) if peak_rss_mb is not None else None,
    }
    # Count missing historical price values from the generated shards without
    # retaining feature rows in RAM.
    missing_prices = 0
    for split_dir in split_dirs.values():
        for shard in split_dir.glob("*.npz"):
            with np.load(shard) as data:
                missing_prices += int(np.isnan(data["price_at_cutoff"]).sum())
    metadata["price_missing_feature_rows"] = missing_prices
    C.FEATURE_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Feature generation complete in %.1fs; rows=%s", metadata["generation_seconds"], row_counts)
    return metadata
