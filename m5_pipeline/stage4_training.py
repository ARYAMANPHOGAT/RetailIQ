"""Stage 4 vs Stage 3 validation comparison; never reads final-test data."""
from __future__ import annotations

import gzip
import json
import logging
import os
import time
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C
from . import stage4_config as S4
from .features import _peak_rss_mb
from .metrics import regression_metrics
from .training import feature_matrix, sample_row_positions, CATEGORICAL_FEATURES

STAGE3_MODEL_META = C.MODELS_DIR / "m5_training_metadata.json"
STAGE3_METRICS = C.OUTPUTS_DIR / "m5_metrics.json"
STAGE3_VAL_PREDICTIONS = C.OUTPUTS_DIR / "m5_validation_predictions.csv.gz"
STAGE3_DEMAND_REPORT = C.ANALYSIS_DIR / "report.json"
MODEL_PATH = S4.MODEL_DIR / "m5_lightgbm_stage4.txt"
MODEL_META_PATH = S4.MODEL_DIR / "m5_training_metadata.json"
METRICS_PATH = S4.RESULTS_DIR / "stage4_metrics.json"
PREDICTIONS_PATH = S4.RESULTS_DIR / "stage4_validation_predictions.csv.gz"
IMPORTANCE_PATH = S4.RESULTS_DIR / "stage4_feature_importance.csv"


def _logger() -> logging.Logger:
    S4.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("m5.stage4.training")
    if not logger.handlers:
        handler = logging.FileHandler(S4.TRAINING_LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _load_stage4_split(split: str, feature_names: list[str], max_rows: int | None = None):
    shard_paths = sorted((S4.FEATURES_DIR / split).glob("part_*.npz"))
    manifest = json.loads(S4.FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
    total = int(manifest["split"][split]["rows"])
    positions = sample_row_positions(total, max_rows or total, S4.RANDOM_SEED)
    x = np.empty((len(positions), len(feature_names)), dtype=np.float32)
    y = np.empty(len(positions), dtype=np.float32)
    side = {name: np.empty(len(positions), dtype=np.uint32 if name == "series_idx" else np.uint16)
            for name in ("series_idx", "target_day")}
    series_seen: set[int] = set()
    offset = out_offset = 0
    for path in shard_paths:
        with np.load(path) as shard:
            count = len(shard["target"])
            lo, hi = np.searchsorted(positions, [offset, offset + count], side="left")
            if lo < hi:
                local = positions[lo:hi] - offset
                columns = {name: shard[name][local] for name in feature_names}
                n = len(local)
                x[out_offset:out_offset+n] = feature_matrix(columns, feature_names)
                y[out_offset:out_offset+n] = shard["target"][local]
                for name in side:
                    side[name][out_offset:out_offset+n] = shard[name][local]
                series_seen.update(np.unique(side["series_idx"][out_offset:out_offset+n]).tolist())
                out_offset += n
            offset += count
    if not shard_paths or offset != total or out_offset != len(positions):
        raise RuntimeError(f"Invalid Stage 4 {split} shards: read={offset}, selected={out_offset}, expected={total}")
    return x, y, side, series_seen


def _tier_results(y: np.ndarray, prediction: np.ndarray, series_idx: np.ndarray,
                  low_max: float, medium_max: float) -> dict:
    series_n = int(series_idx.max()) + 1
    counts = np.bincount(series_idx, minlength=series_n)
    actual_sum = np.bincount(series_idx, weights=y, minlength=series_n)
    mean_demand = np.divide(actual_sum, counts, out=np.zeros_like(actual_sum), where=counts > 0)
    tier_by_series = np.where(mean_demand <= low_max, 0, np.where(mean_demand <= medium_max, 1, 2))
    labels = ("low", "medium", "high")
    result = {}
    for tier, label in enumerate(labels):
        mask = tier_by_series[series_idx] == tier
        error = prediction[mask] - y[mask]
        result[label] = {**regression_metrics(y[mask], prediction[mask]),
                         "mean_bias": float(error.mean()),
                         "series_count": int(np.count_nonzero(tier_by_series == tier)),
                         "rows": int(mask.sum())}
    return result


def _metric_bundle(y: np.ndarray, prediction: np.ndarray, series_idx: np.ndarray,
                   low_max: float, medium_max: float) -> dict:
    return {"overall": {**regression_metrics(y, prediction), "mean_bias": float((prediction-y).mean())},
            "demand_levels": _tier_results(y, prediction, series_idx, low_max, medium_max)}


def _write_predictions(path, frame: pd.DataFrame):
    temp = path.with_name(path.stem + ".tmp" + path.suffix)
    frame.to_csv(temp, index=False, compression={"method": "gzip", "compresslevel": 1, "mtime": 0})
    return temp


def train_stage4_comparison() -> dict:
    """Train only Stage 4 model; compare with existing Stage 3 validation outputs."""
    inputs = (S4.FEATURE_METADATA_PATH, S4.SERIES_METADATA_PATH, STAGE3_MODEL_META,
              STAGE3_METRICS, STAGE3_VAL_PREDICTIONS, STAGE3_DEMAND_REPORT)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Stage 4/Stage 3 comparison inputs: " + ", ".join(missing))
    outputs = (MODEL_PATH, MODEL_META_PATH, METRICS_PATH, PREDICTIONS_PATH, IMPORTANCE_PATH)
    if any(path.exists() for path in outputs):
        raise FileExistsError("Refusing to overwrite Stage 4 artifacts")
    stage4_manifest = json.loads(S4.FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
    stage3_meta = json.loads(STAGE3_MODEL_META.read_text(encoding="utf-8"))
    stage3_demand = json.loads(STAGE3_DEMAND_REPORT.read_text(encoding="utf-8"))
    if stage4_manifest.get("status") != "complete":
        raise ValueError("Complete Stage 4 feature shards are required")
    if stage4_manifest["split"]["train"]["rows"] != stage3_meta["total_training_rows_available"]:
        raise ValueError("Stage 4 train rows differ from Stage 3")
    if S4.SAMPLE_ROWS != stage3_meta["training_rows_used"] or S4.RANDOM_SEED != stage3_meta["random_seed"]:
        raise ValueError("Stage 4 sample policy differs from the Stage 3 benchmark")

    logger = _logger()
    started = time.time()
    feature_names = [name for name in stage4_manifest["features"] if name not in ("series_idx", "target")]
    if feature_names[:len(stage3_meta["feature_names"])] != stage3_meta["feature_names"]:
        raise ValueError("Stage 4 base feature ordering does not match Stage 3")
    categorical = [name for name in CATEGORICAL_FEATURES if name in feature_names]
    logger.info("Loading same seeded Stage 3 row sample: %d rows", S4.SAMPLE_ROWS)
    x_train, y_train, _, sampled_series = _load_stage4_split("train", feature_names, S4.SAMPLE_ROWS)
    sampled_train_rows = len(y_train)
    x_val, y_val, val_side, val_series = _load_stage4_split("validation", feature_names)
    if len(val_series) != stage4_manifest["series_count"]:
        raise RuntimeError("Stage 4 validation split is missing series")

    params = stage3_meta["lightgbm_parameters"]
    train_set = lgb.Dataset(x_train, label=y_train, feature_name=feature_names,
                            categorical_feature=categorical, params=params, free_raw_data=True)
    train_set.construct()
    del x_train, y_train
    valid_set = lgb.Dataset(x_val, label=y_val, reference=train_set, feature_name=feature_names,
                            categorical_feature=categorical, params=params, free_raw_data=True)
    valid_set.construct()
    logger.info("Training Stage 4 LightGBM: same Stage 3 parameters and early stopping")
    booster = lgb.train(
        params, train_set, num_boost_round=stage3_meta["max_boost_rounds"],
        valid_sets=[valid_set], valid_names=["validation"],
        callbacks=[lgb.early_stopping(stage3_meta["early_stopping_rounds"], first_metric_only=True),
                   lgb.log_evaluation(C.LIGHTGBM_LOG_PERIOD)],
    )
    best_iteration = int(booster.best_iteration or stage3_meta["max_boost_rounds"])
    stage4_prediction = np.maximum(0, booster.predict(x_val, num_iteration=best_iteration)).astype(np.float32)

    # Demand-tier cutpoints are fixed from the existing Stage 3 analysis.
    cuts = stage3_demand["demand_level_cutpoints_mean_daily_units"]
    low_max, medium_max = cuts["low_max"], cuts["medium_max"]
    stage3_frame = pd.read_csv(STAGE3_VAL_PREDICTIONS)
    if len(stage3_frame) != len(y_val):
        raise ValueError("Stage 3 and Stage 4 validation row counts differ")
    if not np.array_equal(stage3_frame.series_idx.to_numpy(), val_side["series_idx"]):
        raise ValueError("Stage 3 and Stage 4 validation series ordering differs")
    if not np.array_equal(stage3_frame.target_day.to_numpy(), val_side["target_day"]):
        raise ValueError("Stage 3 and Stage 4 validation target dates differ")
    if not np.array_equal(stage3_frame.actual.to_numpy(dtype=np.float32), y_val):
        raise ValueError("Stage 3 and Stage 4 validation actuals differ")
    stage3_prediction = stage3_frame.lightgbm.to_numpy(dtype=np.float32)
    stage3_metrics = _metric_bundle(y_val, stage3_prediction, val_side["series_idx"], low_max, medium_max)
    stage4_metrics = _metric_bundle(y_val, stage4_prediction, val_side["series_idx"], low_max, medium_max)
    comparison = {
        "scope": "Validation only; final test data was not read.",
        "demand_tier_cutpoints_from_stage3": cuts,
        "validation_rows": len(y_val), "series_count": stage4_manifest["series_count"],
        "stage3_existing": stage3_metrics, "stage4_new_features": stage4_metrics,
        "stage4_best_iteration": best_iteration,
        "auto_promotion": False,
    }

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "Stage 4 features vs frozen Stage 3 LightGBM benchmark",
        "feature_names": feature_names, "new_features": stage4_manifest["new_features"],
        "training_rows_available": stage4_manifest["split"]["train"]["rows"],
        "training_rows_used": sampled_train_rows,
        "training_series_used": len(sampled_series),
        "validation_rows": len(y_val), "validation_series": len(val_series),
        "training_period_target_days": stage4_manifest["split"]["train"]["target_days"],
        "validation_period_target_days": stage4_manifest["split"]["validation"]["target_days"],
        "sample_policy": stage3_meta["sampling_policy"], "random_seed": S4.RANDOM_SEED,
        "lightgbm_parameters": params, "best_iteration": best_iteration,
        "missing_value_policy": stage4_manifest["missing_value_policy"],
        "test_accessed": False, "promotion_status": "not_promoted",
        "peak_process_rss_mb": _peak_rss_mb(), "elapsed_seconds": round(time.time()-started, 2),
    }
    importance = pd.DataFrame({"feature": feature_names,
                               "gain": booster.feature_importance(importance_type="gain"),
                               "split": booster.feature_importance(importance_type="split")})
    importance["gain_share_percent"] = np.where(importance.gain.sum() > 0,
                                                importance.gain / importance.gain.sum() * 100, 0.0)
    importance = importance.sort_values("gain", ascending=False)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    S4.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_temp = MODEL_PATH.with_name(".m5_lightgbm_stage4.tmp.txt")
    metadata_temp = MODEL_META_PATH.with_name(".m5_training_metadata.tmp.json")
    metrics_temp = METRICS_PATH.with_name(".stage4_metrics.tmp.json")
    predictions_frame = pd.DataFrame({
        "series_idx": val_side["series_idx"], "target_day": val_side["target_day"],
        "actual": y_val, "stage3_lightgbm": stage3_prediction, "stage4_lightgbm": stage4_prediction,
    })
    pred_temp = _write_predictions(PREDICTIONS_PATH, predictions_frame)
    importance_temp = IMPORTANCE_PATH.with_name(".stage4_feature_importance.tmp.csv")
    importance.to_csv(importance_temp, index=False)
    booster.save_model(str(model_temp), num_iteration=best_iteration)
    reloaded = lgb.Booster(model_file=str(model_temp))
    if not np.allclose(booster.predict(x_val[:128], num_iteration=best_iteration),
                       reloaded.predict(x_val[:128]), rtol=1e-6, atol=1e-6):
        raise RuntimeError("Stage 4 saved model failed reload verification")
    metadata_temp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics_temp.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    os.replace(model_temp, MODEL_PATH)
    os.replace(metadata_temp, MODEL_META_PATH)
    os.replace(metrics_temp, METRICS_PATH)
    os.replace(pred_temp, PREDICTIONS_PATH)
    os.replace(importance_temp, IMPORTANCE_PATH)
    logger.info("Stage 4 model saved with best iteration %d; no promotion", best_iteration)
    return {"comparison": comparison, "training_metadata": metadata,
            "artifacts": [str(path) for path in outputs]}


if __name__ == "__main__":
    print(json.dumps(train_stage4_comparison(), indent=2))
