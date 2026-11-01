"""Stage 4.1: validation-only same-weekday summaries of existing lags."""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C
from . import stage4_1_config as S41
from . import stage4_config as S4
from .features import _peak_rss_mb
from .stage4_training import _metric_bundle
from .training import CATEGORICAL_FEATURES, feature_matrix, sample_row_positions

STAGE4_METADATA = S4.MODEL_DIR / "m5_training_metadata.json"
STAGE4_METRICS = S4.RESULTS_DIR / "stage4_metrics.json"
STAGE4_VALIDATION_PREDS = S4.RESULTS_DIR / "stage4_validation_predictions.csv.gz"
NEW_FEATURES = ("same_weekday_nonzero_rate_3_lags", "same_weekday_positive_mean_3_lags")


def same_weekday_features(lag_7, lag_14, lag_28) -> dict[str, np.ndarray]:
    """Summarize only the existing target-safe same-weekday lag columns."""
    values = np.column_stack([np.asarray(lag_7), np.asarray(lag_14), np.asarray(lag_28)]).astype(np.float32)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("same-weekday lags must be finite, nonnegative sales")
    positive = values > 0
    count = positive.sum(axis=1)
    mean = np.divide((values * positive).sum(axis=1), count,
                     out=np.full(len(values), np.nan, dtype=np.float32), where=count > 0)
    return {
        NEW_FEATURES[0]: positive.mean(axis=1, dtype=np.float32),
        NEW_FEATURES[1]: mean.astype(np.float32),
    }


def _logger() -> logging.Logger:
    S41.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("m5.stage4_1.training")
    if not logger.handlers:
        handler = logging.FileHandler(S41.TRAINING_LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _load_split(split: str, feature_names: list[str], max_rows: int | None = None):
    paths = sorted((S41.FEATURES_DIR / split).glob("part_*.npz"))
    manifest = json.loads(S4.FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
    total = int(manifest["split"][split]["rows"])
    positions = sample_row_positions(total, max_rows or total, S41.RANDOM_SEED)
    x = np.empty((len(positions), len(feature_names)), dtype=np.float32)
    y = np.empty(len(positions), dtype=np.float32)
    side = {name: np.empty(len(positions), dtype=np.uint32 if name == "series_idx" else np.uint16)
            for name in ("series_idx", "target_day")}
    selected_series: set[int] = set()
    base_names = [name for name in feature_names if name not in NEW_FEATURES]
    offset = out = 0
    for path in paths:
        with np.load(path) as shard:
            count = len(shard["target"])
            lo, hi = np.searchsorted(positions, [offset, offset + count], side="left")
            if lo < hi:
                local = positions[lo:hi] - offset
                columns = {name: shard[name][local] for name in base_names}
                n = len(local)
                x[out:out+n, :len(base_names)] = feature_matrix(columns, base_names)
                extra = same_weekday_features(columns["lag_7"], columns["lag_14"], columns["lag_28"])
                for j, name in enumerate(NEW_FEATURES, start=len(base_names)):
                    x[out:out+n, j] = extra[name]
                y[out:out+n] = shard["target"][local]
                for name in side:
                    side[name][out:out+n] = shard[name][local]
                selected_series.update(np.unique(side["series_idx"][out:out+n]).tolist())
                out += n
            offset += count
    if not paths or offset != total or out != len(positions):
        raise RuntimeError(f"Invalid Stage 4 {split} shards: read={offset}, selected={out}, expected={total}")
    return x, y, side, selected_series


def train_stage4_1_validation() -> dict:
    """Fit one fixed-config candidate and score validation only."""
    outputs = (S41.MODEL_PATH, S41.MODEL_METADATA_PATH, S41.METRICS_PATH,
               S41.PREDICTIONS_PATH, S41.IMPORTANCE_PATH)
    if any(path.exists() for path in outputs):
        raise FileExistsError("Refusing to overwrite Stage 4.1 experiment artifacts")
    for path in (S4.FEATURE_METADATA_PATH, S41.FEATURES_DIR / "series_metadata.csv",
                 STAGE4_METADATA, STAGE4_METRICS, STAGE4_VALIDATION_PREDS):
        if not path.is_file():
            raise FileNotFoundError(f"Required Stage 4 input is missing: {path}")
    stage4_meta = json.loads(STAGE4_METADATA.read_text(encoding="utf-8"))
    stage4_manifest = json.loads(S4.FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
    if S41.SAMPLE_ROWS != stage4_meta["training_rows_used"] or S41.RANDOM_SEED != stage4_meta["random_seed"]:
        raise ValueError("Stage 4.1 sampling policy differs from Stage 4")
    base_names = list(stage4_meta["feature_names"])
    feature_names = [*base_names, *NEW_FEATURES]
    categorical = [name for name in CATEGORICAL_FEATURES if name in feature_names]
    started = time.time()
    logger = _logger()
    logger.info("Reusing Stage 4 shards; sample=%d, seed=%d", S41.SAMPLE_ROWS, S41.RANDOM_SEED)
    x_train, y_train, _, train_series = _load_split("train", feature_names, S41.SAMPLE_ROWS)
    x_val, y_val, side, val_series = _load_split("validation", feature_names)
    if len(val_series) != int(stage4_manifest["series_count"]):
        raise RuntimeError("Stage 4.1 validation series population differs from Stage 4")

    params = stage4_meta["lightgbm_parameters"]
    train_set = lgb.Dataset(x_train, label=y_train, feature_name=feature_names,
                            categorical_feature=categorical, params=params, free_raw_data=True)
    train_set.construct()
    del x_train, y_train
    valid_set = lgb.Dataset(x_val, label=y_val, reference=train_set, feature_name=feature_names,
                            categorical_feature=categorical, params=params, free_raw_data=True)
    valid_set.construct()
    logger.info("Training Stage 4.1 fixed LightGBM configuration")
    booster = lgb.train(
        params, train_set, num_boost_round=C.LIGHTGBM_MAX_BOOST_ROUNDS,
        valid_sets=[valid_set], valid_names=["validation"],
        callbacks=[lgb.early_stopping(C.LIGHTGBM_EARLY_STOPPING_ROUNDS, first_metric_only=True),
                   lgb.log_evaluation(C.LIGHTGBM_LOG_PERIOD)],
    )
    best_iteration = int(booster.best_iteration or C.LIGHTGBM_MAX_BOOST_ROUNDS)
    prediction = np.maximum(0, booster.predict(x_val, num_iteration=best_iteration)).astype(np.float32)
    prior = pd.read_csv(STAGE4_VALIDATION_PREDS)
    if len(prior) != len(y_val) or not np.array_equal(prior.series_idx.to_numpy(), side["series_idx"]):
        raise ValueError("Stage 4.1 validation series rows do not match Stage 4")
    if not np.array_equal(prior.target_day.to_numpy(), side["target_day"]):
        raise ValueError("Stage 4.1 validation dates do not match Stage 4")
    if not np.array_equal(prior.actual.to_numpy(np.float32), y_val):
        raise ValueError("Stage 4.1 validation actuals do not match Stage 4")
    cuts = json.loads(STAGE4_METRICS.read_text(encoding="utf-8"))["demand_tier_cutpoints_from_stage3"]
    metrics = {
        "scope": "Validation only; final test was not accessed.",
        "validation_rows": len(y_val), "series_count": len(val_series),
        "demand_tier_cutpoints_from_stage3": cuts,
        "stage3_existing": _metric_bundle(y_val, prior.stage3_lightgbm.to_numpy(np.float32), side["series_idx"], cuts["low_max"], cuts["medium_max"]),
        "stage4_existing": _metric_bundle(y_val, prior.stage4_lightgbm.to_numpy(np.float32), side["series_idx"], cuts["low_max"], cuts["medium_max"]),
        "stage4_1_same_weekday": _metric_bundle(y_val, prediction, side["series_idx"], cuts["low_max"], cuts["medium_max"]),
        "auto_promotion": False,
    }
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "Stage 4.1: same-weekday lag summaries, derived from Stage 4 features",
        "feature_names": feature_names, "new_features": list(NEW_FEATURES),
        "feature_definition": "Nonzero rate and positive-sales mean over lag_7, lag_14, lag_28; all sales are before target date.",
        "training_rows_available": int(stage4_manifest["split"]["train"]["rows"]),
        "training_rows_used": S41.SAMPLE_ROWS, "training_series_used": len(train_series),
        "validation_rows": len(y_val), "validation_series": len(val_series),
        "training_period_target_days": stage4_meta["training_period_target_days"],
        "validation_period_target_days": stage4_meta["validation_period_target_days"],
        "sample_policy": stage4_meta["sample_policy"], "random_seed": S41.RANDOM_SEED,
        "lightgbm_parameters": params, "best_iteration": best_iteration,
        "test_accessed": False, "promotion_status": "not_promoted",
        "peak_process_rss_mb": _peak_rss_mb(), "elapsed_seconds": round(time.time()-started, 2),
    }
    importance = pd.DataFrame({"feature": feature_names,
                               "gain": booster.feature_importance(importance_type="gain"),
                               "split": booster.feature_importance(importance_type="split")}).sort_values("gain", ascending=False)
    S41.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    S41.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_model = S41.MODEL_PATH.with_name(".m5_stage4_1.tmp.txt")
    booster.save_model(str(tmp_model), num_iteration=best_iteration)
    reloaded = lgb.Booster(model_file=str(tmp_model))
    if not np.allclose(booster.predict(x_val[:128], num_iteration=best_iteration), reloaded.predict(x_val[:128]), rtol=1e-6, atol=1e-6):
        tmp_model.unlink(missing_ok=True)
        raise RuntimeError("Stage 4.1 saved model reload check failed")
    os.replace(tmp_model, S41.MODEL_PATH)
    S41.MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    S41.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"series_idx": side["series_idx"], "target_day": side["target_day"], "actual": y_val,
                  "stage3_lightgbm": prior.stage3_lightgbm, "stage4_lightgbm": prior.stage4_lightgbm,
                  "stage4_1_lightgbm": prediction}).to_csv(
        S41.PREDICTIONS_PATH, index=False,
        compression={"method": "gzip", "compresslevel": 1, "mtime": 0})
    importance.to_csv(S41.IMPORTANCE_PATH, index=False)
    logger.info("Stage 4.1 validation complete at iteration %d; final test untouched", best_iteration)
    return {"metrics": metrics, "metadata": metadata,
            "artifacts": [str(path) for path in outputs]}


if __name__ == "__main__":
    print(json.dumps(train_stage4_1_validation(), indent=2))
