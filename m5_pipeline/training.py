"""Stage 3 baseline comparison and memory-bounded LightGBM training."""
from __future__ import annotations

import gzip
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C
from .features import _peak_rss_mb
from .metrics import regression_metrics

MODEL_DIR = C.MODELS_DIR
MODEL_PATH = MODEL_DIR / "m5_lightgbm.txt"
MODEL_METADATA_PATH = MODEL_DIR / "m5_training_metadata.json"
METRICS_PATH = C.OUTPUTS_DIR / "m5_metrics.json"
VALIDATION_PREDICTIONS_PATH = C.OUTPUTS_DIR / "m5_validation_predictions.csv.gz"
TEST_PREDICTIONS_PATH = C.OUTPUTS_DIR / "m5_test_predictions.csv.gz"

ID_COLUMNS = ("series_idx", "target_day", "target", "lag_7")
CATEGORICAL_FEATURES = ("item_idx", "store_idx", "state_idx", "dept_idx", "cat_idx",
                        "target_dow", "target_wday", "target_month", "event_name_1",
                        "event_type_1", "event_name_2", "event_type_2")


def _logger() -> logging.Logger:
    C.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("m5.training")
    if not logger.handlers:
        handler = logging.FileHandler(C.TRAINING_LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def sample_row_positions(total_rows: int, max_rows: int, seed: int) -> np.ndarray:
    """Return sorted, reproducible uniform row positions without replacement."""
    if total_rows < 0 or max_rows < 1:
        raise ValueError("total_rows must be nonnegative and max_rows must be positive")
    sample_size = min(total_rows, max_rows)
    if sample_size == total_rows:
        return np.arange(total_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total_rows, size=sample_size, replace=False))


def feature_matrix(columns: dict[str, np.ndarray], feature_names: list[str]) -> np.ndarray:
    """Stack one shard's model columns into compact float32, preserving NaNs."""
    if not feature_names:
        raise ValueError("feature_names cannot be empty")
    lengths = {len(columns[name]) for name in feature_names}
    if len(lengths) != 1:
        raise ValueError("Feature columns have inconsistent row counts")
    matrix = np.empty((next(iter(lengths)), len(feature_names)), dtype=np.float32)
    for index, name in enumerate(feature_names):
        matrix[:, index] = columns[name]
    return matrix


def _load_split(split: str, feature_names: list[str], max_rows: int | None = None,
                seed: int = C.RANDOM_SEED) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], set[int]]:
    split_dir = C.FEATURES_DIR / split
    paths = sorted(split_dir.glob("part_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No feature shards found for split '{split}' in {split_dir}")
    total_rows = int(C.FEATURE_METADATA_PATH.exists() and
                     json.loads(C.FEATURE_METADATA_PATH.read_text(encoding="utf-8"))["split"][split]["rows"])
    positions = sample_row_positions(total_rows, max_rows or total_rows, seed)
    out_x = np.empty((len(positions), len(feature_names)), dtype=np.float32)
    out_y = np.empty(len(positions), dtype=np.float32)
    side = {name: np.empty(len(positions), dtype=np.uint32 if name == "series_idx" else np.uint16)
            for name in ("series_idx", "target_day")}
    selected_series: set[int] = set()
    offset = output_offset = 0
    for path in paths:
        with np.load(path) as shard:
            shard_rows = len(shard["target"])
            left = np.searchsorted(positions, offset, side="left")
            right = np.searchsorted(positions, offset + shard_rows, side="left")
            if left < right:
                local = positions[left:right] - offset
                columns = {name: shard[name][local] for name in feature_names}
                out_x[output_offset:output_offset + len(local)] = feature_matrix(columns, feature_names)
                out_y[output_offset:output_offset + len(local)] = shard["target"][local]
                for name in side:
                    side[name][output_offset:output_offset + len(local)] = shard[name][local]
                selected_series.update(np.unique(side["series_idx"][output_offset:output_offset + len(local)]).tolist())
                output_offset += len(local)
            offset += shard_rows
    if offset != total_rows or output_offset != len(positions):
        raise RuntimeError(f"Read {offset} rows / selected {output_offset}; metadata expects {total_rows}/{len(positions)}")
    return out_x, out_y, side, selected_series


def _prediction_frame(side: dict[str, np.ndarray], actual: np.ndarray, seasonal: np.ndarray,
                      lightgbm: np.ndarray) -> pd.DataFrame:
    series = pd.read_csv(C.SERIES_METADATA_PATH, usecols=["series_idx", "id"])
    ids = series.sort_values("series_idx")["id"].astype(str).to_numpy()
    with C.CALENDAR_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        cal = pd.read_csv(stream, usecols=["d", "date"], dtype={"d": "string", "date": "string"})
    dates = {int(row.d.removeprefix("d_")): row.date for row in cal.itertuples(index=False)}
    day_ids = side["target_day"].astype(np.int32, copy=False)
    return pd.DataFrame({
        "series_idx": side["series_idx"],
        "series_id": ids[side["series_idx"]],
        "target_day": day_ids,
        "target_date": [dates[int(day)] for day in day_ids],
        "actual": actual,
        "seasonal_naive_lag7": seasonal,
        "lightgbm": lightgbm,
    })


def _write_predictions(path: Path, frame: pd.DataFrame) -> Path:
    temp = path.with_name(path.stem + ".tmp" + path.suffix)
    frame.to_csv(temp, index=False, compression={"method": "gzip", "compresslevel": 1, "mtime": 0})
    return temp


def train_and_evaluate() -> dict:
    """Fit on the approved training split, select on validation, then test once."""
    paths = (MODEL_PATH, MODEL_METADATA_PATH, METRICS_PATH,
             VALIDATION_PREDICTIONS_PATH, TEST_PREDICTIONS_PATH)
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing M5 model outputs: " + ", ".join(existing))
    feature_meta = json.loads(C.FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
    if feature_meta.get("status") != "complete" or "price_missing" not in feature_meta["features"]:
        raise ValueError("A complete Stage 2 feature dataset with price_missing is required")
    logger = _logger()
    started = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    C.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    model_features = [name for name in feature_meta["features"] if name not in ("series_idx", "target")]
    categorical = [name for name in CATEGORICAL_FEATURES if name in model_features]
    total_train = feature_meta["split"]["train"]["rows"]
    cap = C.MAX_TRAIN_ROWS_PER_MODEL
    logger.info("Loading training sample: cap=%d of %d rows; seed=%d", cap, total_train, C.RANDOM_SEED)
    x_train, y_train, _, sampled_series = _load_split("train", model_features, cap, C.RANDOM_SEED)
    sampled_train_rows = len(y_train)
    logger.info("Loaded %d training rows across %d series; matrix=%s", len(y_train), len(sampled_series), x_train.shape)
    logger.info("Loading full validation split (%d rows)", feature_meta["split"]["validation"]["rows"])
    x_val, y_val, val_side, val_series = _load_split("validation", model_features)
    if len(val_series) != feature_meta["series_count"]:
        raise RuntimeError("Validation split does not contain the full series population")

    params = {
        "objective": "regression_l1",
        "metric": "l1",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "max_bin": 63,
        "histogram_pool_size": 512,
        "cat_smooth": 10.0,
        "max_cat_threshold": 32,
        "use_missing": True,
        "zero_as_missing": False,
        "verbosity": -1,
        "num_threads": C.LIGHTGBM_NUM_THREADS,
        "seed": C.RANDOM_SEED,
        "data_random_seed": C.RANDOM_SEED,
        "feature_fraction_seed": C.RANDOM_SEED,
        "bagging_seed": C.RANDOM_SEED,
        "deterministic": True,
        "force_col_wise": True,
    }
    logger.info("Constructing LightGBM datasets; NaNs are retained for native handling")
    train_set = lgb.Dataset(x_train, label=y_train, feature_name=model_features,
                            categorical_feature=categorical, params=params, free_raw_data=True)
    train_set.construct()
    del x_train, y_train
    valid_set = lgb.Dataset(x_val, label=y_val, reference=train_set, feature_name=model_features,
                            categorical_feature=categorical, params=params, free_raw_data=True)
    valid_set.construct()
    logger.info("Starting LightGBM: max_rounds=%d early_stop=%d", C.LIGHTGBM_MAX_BOOST_ROUNDS,
                C.LIGHTGBM_EARLY_STOPPING_ROUNDS)
    booster = lgb.train(
        params, train_set, num_boost_round=C.LIGHTGBM_MAX_BOOST_ROUNDS,
        valid_sets=[valid_set], valid_names=["validation"],
        callbacks=[lgb.early_stopping(C.LIGHTGBM_EARLY_STOPPING_ROUNDS, first_metric_only=True),
                   lgb.log_evaluation(C.LIGHTGBM_LOG_PERIOD)],
    )
    best_iteration = int(booster.best_iteration or C.LIGHTGBM_MAX_BOOST_ROUNDS)
    val_lgbm = np.maximum(0, booster.predict(x_val, num_iteration=best_iteration)).astype(np.float32)
    val_naive = x_val[:, model_features.index("lag_7")].copy()
    val_metrics = {
        "Seasonal Naive (lag_7)": regression_metrics(y_val, val_naive),
        "LightGBM": regression_metrics(y_val, val_lgbm),
    }
    selected = min(val_metrics, key=lambda name: val_metrics[name]["MAE"])
    logger.info("Validation selection: %s; naive MAE=%.5f LightGBM MAE=%.5f at iteration %d",
                selected, val_metrics["Seasonal Naive (lag_7)"]["MAE"], val_metrics["LightGBM"]["MAE"], best_iteration)
    validation_frame = _prediction_frame(val_side, y_val, val_naive, val_lgbm)
    val_temp = _write_predictions(VALIDATION_PREDICTIONS_PATH, validation_frame)
    del validation_frame, valid_set, val_side

    # The final test files are first opened only after the validation-based
    # model selection above has been finalized.
    logger.info("Selection recorded; now loading untouched final test (%d rows)",
                feature_meta["split"]["test"]["rows"])
    x_test, y_test, test_side, test_series = _load_split("test", model_features)
    if len(test_series) != feature_meta["series_count"]:
        raise RuntimeError("Final test split does not contain the full series population")
    test_lgbm = np.maximum(0, booster.predict(x_test, num_iteration=best_iteration)).astype(np.float32)
    test_naive = x_test[:, model_features.index("lag_7")].copy()
    test_metrics = {
        "Seasonal Naive (lag_7)": regression_metrics(y_test, test_naive),
        "LightGBM": regression_metrics(y_test, test_lgbm),
    }
    test_frame = _prediction_frame(test_side, y_test, test_naive, test_lgbm)
    test_temp = _write_predictions(TEST_PREDICTIONS_PATH, test_frame)
    del test_frame, x_test

    run_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "lightgbm_version": lgb.__version__,
        "selected_model": selected,
        "selection_rule": "Lowest validation MAE; final test was not read until selection was fixed.",
        "training_period_target_days": feature_meta["split"]["train"]["target_days"],
        "validation_period_target_days": feature_meta["split"]["validation"]["target_days"],
        "test_period_target_days": feature_meta["split"]["test"]["target_days"],
        "total_training_rows_available": int(total_train),
        "training_rows_used": int(sampled_train_rows),
        "training_series_used": len(sampled_series),
        "validation_rows": int(feature_meta["split"]["validation"]["rows"]),
        "validation_series": len(val_series) if val_series is not None else feature_meta["series_count"],
        "test_rows": int(feature_meta["split"]["test"]["rows"]),
        "test_series": len(test_series),
        "sampling_policy": "Uniform global row sample without replacement, sorted by source row position; all validation/test rows retained.",
        "random_seed": C.RANDOM_SEED,
        "feature_names": model_features,
        "categorical_features": categorical,
        "missing_values": "LightGBM native NaN handling enabled; price_missing retained; no imputation.",
        "lightgbm_parameters": params,
        "max_boost_rounds": C.LIGHTGBM_MAX_BOOST_ROUNDS,
        "early_stopping_rounds": C.LIGHTGBM_EARLY_STOPPING_ROUNDS,
        "best_iteration": best_iteration,
        "model_training_seconds": round(time.time() - started, 2),
        "peak_process_rss_mb": _peak_rss_mb(),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    model_temp = MODEL_PATH.with_name("." + MODEL_PATH.name + ".tmp")
    meta_temp = MODEL_METADATA_PATH.with_name("." + MODEL_METADATA_PATH.name + ".tmp")
    metrics_temp = METRICS_PATH.with_name("." + METRICS_PATH.name + ".tmp")
    booster.save_model(str(model_temp), num_iteration=best_iteration)
    # Verify the serialized model reloads and gives the same small-sample output.
    reloaded = lgb.Booster(model_file=str(model_temp))
    probe = min(128, len(x_val))
    if not np.allclose(booster.predict(x_val[:probe], num_iteration=best_iteration),
                       reloaded.predict(x_val[:probe]), rtol=1e-6, atol=1e-6):
        raise RuntimeError("Saved LightGBM model prediction verification failed")
    meta_temp.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    metrics = {
        "selected_model": selected,
        "selection_metric": "MAE",
        "validation": val_metrics,
        "final_test": test_metrics,
        "row_counts": {"available_train": int(total_train), "training_used": sampled_train_rows,
                       "validation": int(feature_meta["split"]["validation"]["rows"]),
                       "test": int(feature_meta["split"]["test"]["rows"])},
        "series_counts": {"all": int(feature_meta["series_count"]), "training_sample": len(sampled_series),
                          "validation": int(feature_meta["series_count"]), "test": len(test_series)},
        "best_iteration": best_iteration,
    }
    metrics_temp.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Publish only after model reload and all metric/prediction files exist.
    os.replace(model_temp, MODEL_PATH)
    os.replace(meta_temp, MODEL_METADATA_PATH)
    os.replace(metrics_temp, METRICS_PATH)
    os.replace(val_temp, VALIDATION_PREDICTIONS_PATH)
    os.replace(test_temp, TEST_PREDICTIONS_PATH)
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        raise RuntimeError("One or more M5 training artifacts failed final file verification")
    with gzip.open(VALIDATION_PREDICTIONS_PATH, "rt", encoding="utf-8") as stream:
        first_line = stream.readline()
    if "actual" not in first_line or "lightgbm" not in first_line:
        raise RuntimeError("Validation prediction artifact header verification failed")
    logger.info("Training complete: selected=%s best_iteration=%d duration=%.1fs", selected,
                best_iteration, run_meta["model_training_seconds"])
    return {"metrics": metrics, "training_metadata": run_meta,
            "artifacts": [str(path.relative_to(C.ROOT_DIR)) for path in paths]}
