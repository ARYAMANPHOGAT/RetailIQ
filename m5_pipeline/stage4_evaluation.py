"""One-time final holdout evaluation for the selected Stage 4 candidate."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C
from . import stage4_config as S4
from .features import _peak_rss_mb
from .metrics import regression_metrics
from .stage4_features import NEW_FEATURES, compute_intermittent_features
from .stage4_training import (
    MODEL_META_PATH, MODEL_PATH, METRICS_PATH, _metric_bundle,
)
from .training import feature_matrix

STAGE3_TEST_PREDICTIONS = C.OUTPUTS_DIR / "m5_test_predictions.csv.gz"
TEST_SHARDS = C.FEATURES_DIR / "test"
STAGE4_RESULTS = S4.RESULTS_DIR
TEST_METRICS_PATH = STAGE4_RESULTS / "stage4_test_metrics.json"
TEST_PREDICTIONS_PATH = STAGE4_RESULTS / "stage4_test_predictions.csv.gz"
TEST_LOG_PATH = S4.LOG_DIR / "test_evaluation.log"


def _logger() -> logging.Logger:
    S4.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("m5.stage4.test_evaluation")
    if not logger.handlers:
        handler = logging.FileHandler(TEST_LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def evaluate_selected_stage4_on_test() -> dict:
    """Evaluate only after Stage 4 wins the predeclared validation MAE rule.

    Existing Stage 2 test feature shards provide the unchanged feature columns
    and labels. Stage 4 historical features are computed in matching bounded
    series batches from sales through d_1941; each feature calculation itself
    ends at the prediction cutoff. Nothing from this run feeds back into model
    selection or training.
    """
    required = (MODEL_PATH, MODEL_META_PATH, METRICS_PATH, STAGE3_TEST_PREDICTIONS)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Stage 4 test-evaluation inputs: " + ", ".join(missing))
    if TEST_METRICS_PATH.exists() or TEST_PREDICTIONS_PATH.exists():
        raise FileExistsError("Refusing to overwrite existing Stage 4 final-test results")
    if not TEST_SHARDS.is_dir() or not list(TEST_SHARDS.glob("part_*.npz")):
        raise FileNotFoundError(f"Stage 2 test feature shards not found: {TEST_SHARDS}")

    validation = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    stage3_mae = validation["stage3_existing"]["overall"]["MAE"]
    stage4_mae = validation["stage4_new_features"]["overall"]["MAE"]
    if not stage4_mae < stage3_mae:
        raise ValueError("Stage 4 did not win the validation MAE selection rule; final test remains unopened")
    metadata = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    if metadata.get("test_accessed") is not False or metadata.get("promotion_status") != "not_promoted":
        raise ValueError("Stage 4 model metadata indicates test access or promotion")
    feature_names = metadata["feature_names"]
    if feature_names[-len(NEW_FEATURES):] != NEW_FEATURES:
        raise ValueError("Stage 4 model feature ordering does not match the feature generator")

    logger = _logger()
    started = time.time()
    model = lgb.Booster(model_file=str(MODEL_PATH))
    stage3 = pd.read_csv(STAGE3_TEST_PREDICTIONS)
    total_rows = C.TEST_DAY_END - C.TEST_DAY_START + 1
    total_rows *= int(metadata["validation_series"])
    predictions = np.empty(total_rows, dtype=np.float32)
    actual = np.empty(total_rows, dtype=np.float32)
    series_idx = np.empty(total_rows, dtype=np.uint32)
    target_day = np.empty(total_rows, dtype=np.uint16)
    offset = 0
    shards = sorted(TEST_SHARDS.glob("part_*.npz"))
    logger.info("Selection fixed from validation MAE: Stage 4 %.8f vs Stage 3 %.8f; evaluating %d rows",
                stage4_mae, stage3_mae, total_rows)

    sales_columns = [f"d_{day}" for day in range(1, C.TEST_DAY_END + 1)]
    target_positions = np.arange(C.TEST_DAY_START - 1, C.TEST_DAY_END, dtype=np.int64)
    # Each Stage 2 test shard corresponds to one source series batch (256 rows).
    for batch_index, (path, sales_frame) in enumerate(zip(
            shards,
            pd.read_csv(C.SALES_PATH, usecols=sales_columns, dtype=np.int16,
                        chunksize=S4.SERIES_BATCH_SIZE),
    )):
        sales = sales_frame.to_numpy(dtype=np.int16, copy=True)
        extra = compute_intermittent_features(sales, target_positions)
        with np.load(path) as shard:
            n = len(shard["target"])
            expected_n = len(sales) * len(target_positions)
            if n != expected_n:
                raise RuntimeError(f"Test shard {path.name} has {n} rows; expected {expected_n}")
            batch_series = np.repeat(np.arange(
                batch_index * S4.SERIES_BATCH_SIZE,
                batch_index * S4.SERIES_BATCH_SIZE + len(sales), dtype=np.uint32), len(target_positions))
            batch_days = np.tile(np.arange(C.TEST_DAY_START, C.TEST_DAY_END + 1, dtype=np.uint16), len(sales))
            if not np.array_equal(shard["series_idx"], batch_series) or not np.array_equal(shard["target_day"], batch_days):
                raise RuntimeError(f"Series/date alignment check failed for {path.name}")
            columns = {name: shard[name] for name in feature_names if name not in NEW_FEATURES}
            columns.update(extra)
            matrix = feature_matrix(columns, feature_names)
            batch_prediction = np.maximum(0, model.predict(matrix, num_iteration=metadata["best_iteration"])).astype(np.float32)
            actual[offset:offset+n] = shard["target"].astype(np.float32, copy=False)
            predictions[offset:offset+n] = batch_prediction
            series_idx[offset:offset+n] = shard["series_idx"]
            target_day[offset:offset+n] = shard["target_day"]
            offset += n
        if batch_index and batch_index % 20 == 0:
            logger.info("Evaluated %d/%d test feature batches (%d/%d rows)",
                        batch_index, len(shards), offset, total_rows)

    if offset != total_rows or len(shards) != (int(metadata["validation_series"]) + S4.SERIES_BATCH_SIZE - 1) // S4.SERIES_BATCH_SIZE:
        raise RuntimeError(f"Test row coverage mismatch: evaluated {offset}, expected {total_rows}")
    if len(stage3) != total_rows:
        raise ValueError("Stage 3 and Stage 4 final-test row counts differ")
    for col, values in (("series_idx", series_idx), ("target_day", target_day)):
        if not np.array_equal(stage3[col].to_numpy(), values):
            raise ValueError(f"Stage 3 and Stage 4 final-test {col} ordering differs")
    if not np.array_equal(stage3.actual.to_numpy(dtype=np.float32), actual):
        raise ValueError("Stage 3 and Stage 4 final-test actuals differ")

    cuts = validation["demand_tier_cutpoints_from_stage3"]
    metrics = {
        "scope": "Final holdout d_1914-d_1941; evaluated once after validation selection; no tuning followed.",
        "selected_candidate": "Stage 4 LightGBM by lowest validation MAE; not promoted to the RetailIQ application.",
        "selection_validation_mae": {"stage3": stage3_mae, "stage4": stage4_mae},
        "test_rows": total_rows, "series_count": int(metadata["validation_series"]),
        "target_days": [C.TEST_DAY_START, C.TEST_DAY_END],
        "demand_tier_cutpoints_from_stage3": cuts,
        "stage3_existing": _metric_bundle(actual, stage3.lightgbm.to_numpy(dtype=np.float32), series_idx,
                                           cuts["low_max"], cuts["medium_max"]),
        "stage4_selected": _metric_bundle(actual, predictions, series_idx,
                                           cuts["low_max"], cuts["medium_max"]),
    }
    result_metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "training_rows_used": metadata["training_rows_used"],
        "training_series_used": metadata["training_series_used"],
        "best_iteration": metadata["best_iteration"],
        "training_elapsed_seconds": metadata["elapsed_seconds"],
        "training_peak_process_rss_mb": metadata["peak_process_rss_mb"],
        "test_evaluation_elapsed_seconds": round(time.time() - started, 2),
        "test_evaluation_peak_process_rss_mb": _peak_rss_mb(),
        "test_accessed_after_model_selection": True,
        "model_selection_changed_after_test": False,
    }
    prediction_frame = pd.DataFrame({
        "series_idx": series_idx, "target_day": target_day, "actual": actual,
        "stage3_lightgbm": stage3.lightgbm.to_numpy(dtype=np.float32),
        "stage4_lightgbm": predictions,
    })
    temp_predictions = TEST_PREDICTIONS_PATH.with_name(".stage4_test_predictions.tmp.csv.gz")
    temp_metrics = TEST_METRICS_PATH.with_name(".stage4_test_metrics.tmp.json")
    prediction_frame.to_csv(temp_predictions, index=False,
                            compression={"method": "gzip", "compresslevel": 1, "mtime": 0})
    metrics["run_metadata"] = result_metadata
    temp_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    temp_predictions.replace(TEST_PREDICTIONS_PATH)
    temp_metrics.replace(TEST_METRICS_PATH)
    logger.info("Final test evaluation saved: Stage 4 MAE %.8f, rows %d", metrics["stage4_selected"]["overall"]["MAE"], total_rows)
    return metrics


if __name__ == "__main__":
    print(json.dumps(evaluate_selected_stage4_on_test(), indent=2))
