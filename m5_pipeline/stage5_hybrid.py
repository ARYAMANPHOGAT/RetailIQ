"""Stage 5 LightGBM + TSB hybrid; all writes are isolated under Stage 5."""
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
from . import stage4_config as S4
from . import stage5_config as S5
from .features import _peak_rss_mb
from .stage4_features import NEW_FEATURES, compute_intermittent_features
from .stage4_training import _metric_bundle
from .training import CATEGORICAL_FEATURES, feature_matrix, sample_row_positions

STAGE4_METADATA = S4.MODEL_DIR / "m5_training_metadata.json"
STAGE4_VALIDATION_METRICS = S4.RESULTS_DIR / "stage4_metrics.json"
STAGE4_VALIDATION_PREDS = S4.RESULTS_DIR / "stage4_validation_predictions.csv.gz"
STAGE4_TEST_PREDS = S4.RESULTS_DIR / "stage4_test_predictions.csv.gz"
STAGE2_TEST_DIR = C.FEATURES_DIR / "test"
ROUTE_FEATURE = "nonzero_rate_84"


def route_to_tsb(nonzero_rate_84, history_days, threshold: float = S5.INTERMITTENT_RATE_THRESHOLD):
    """Route only complete-history series at/below the recent nonzero-rate cut."""
    rates = np.asarray(nonzero_rate_84, dtype=np.float32)
    history = np.asarray(history_days)
    if rates.shape != history.shape:
        raise ValueError("rate and history_days must have matching shapes")
    return np.isfinite(rates) & (history >= S5.MIN_HISTORY_DAYS) & (rates <= threshold)


def compute_tsb_forecasts(
    sales: np.ndarray,
    target_positions: np.ndarray,
    alpha: float = S5.TSB_ALPHA,
    beta: float = S5.TSB_BETA,
    route_threshold: float = S5.INTERMITTENT_RATE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one-step TSB forecasts, prior 84-day rates and route flags.

    Sales columns are day IDs minus one. The forecast for target position p is
    emitted before sales[:, p] is read into the recursive state. TSB is
    initialized from d_1..d_28; routing waits until a full 84-day history.
    """
    values = np.asarray(sales)
    positions = np.asarray(target_positions, dtype=np.int64)
    if values.ndim != 2 or positions.size == 0 or np.any(np.diff(positions) <= 0):
        raise ValueError("sales must be 2-D and target_positions strictly increasing")
    if positions[0] < 28 or positions[-1] > values.shape[1]:
        raise ValueError("TSB targets require 28 observed history days through the target cutoff")
    if not (0 < alpha <= 1 and 0 < beta <= 1):
        raise ValueError("TSB alpha and beta must be in (0, 1]")

    # Discard target and future columns before any state or prefix calculation.
    history = values[:, :int(positions[-1])]
    initial = history[:, :28].astype(np.float64)
    positive_initial = initial > 0
    probability = positive_initial.mean(axis=1)
    positive_count = positive_initial.sum(axis=1)
    size = np.divide((initial * positive_initial).sum(axis=1), positive_count,
                     out=np.zeros(values.shape[0], dtype=np.float64), where=positive_count > 0)
    prefix = np.pad(np.cumsum(history > 0, axis=1, dtype=np.int32), ((0, 0), (1, 0)))
    target_lookup = {int(position): index for index, position in enumerate(positions)}
    forecasts = np.empty((values.shape[0], len(positions)), dtype=np.float32)
    rates = np.full_like(forecasts, np.nan)

    for position in range(28, int(positions[-1]) + 1):
        output_idx = target_lookup.get(position)
        if output_idx is not None:
            forecasts[:, output_idx] = (probability * size).astype(np.float32)
            if position >= S5.MIN_HISTORY_DAYS:
                rates[:, output_idx] = (prefix[:, position] - prefix[:, position - S5.MIN_HISTORY_DAYS]) / float(S5.MIN_HISTORY_DAYS)
        # Update only after emitting the forecast. The final target value is
        # deliberately not read; no later forecast in this batch needs it.
        if position == int(positions[-1]):
            continue
        observed = history[:, position].astype(np.float64)
        occurrence = observed > 0
        probability += beta * (occurrence.astype(np.float64) - probability)
        size[occurrence] += alpha * (observed[occurrence] - size[occurrence])

    routed = route_to_tsb(rates, np.broadcast_to(positions, rates.shape), route_threshold)
    return forecasts.reshape(-1), rates.reshape(-1), routed.reshape(-1)


def _logger(test: bool = False) -> logging.Logger:
    S5.LOG_DIR.mkdir(parents=True, exist_ok=True)
    name = "m5.stage5.test" if test else "m5.stage5.training"
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.FileHandler(S5.TEST_LOG_PATH if test else S5.TRAINING_LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _load_split(split: str, feature_names: list[str], max_rows: int | None = None):
    paths = sorted((S5.FEATURES_DIR / split).glob("part_*.npz"))
    manifest = json.loads(S4.FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
    total = int(manifest["split"][split]["rows"])
    positions = sample_row_positions(total, max_rows or total, S5.RANDOM_SEED)
    x = np.empty((len(positions), len(feature_names)), dtype=np.float32)
    y = np.empty(len(positions), dtype=np.float32)
    rates = np.empty(len(positions), dtype=np.float32)
    side = {name: np.empty(len(positions), dtype=np.uint32 if name == "series_idx" else np.uint16)
            for name in ("series_idx", "target_day")}
    offset = out = 0
    for path in paths:
        with np.load(path) as shard:
            count = len(shard["target"])
            lo, hi = np.searchsorted(positions, [offset, offset + count], side="left")
            if lo < hi:
                local = positions[lo:hi] - offset
                n = len(local)
                columns = {name: shard[name][local] for name in feature_names}
                x[out:out+n] = feature_matrix(columns, feature_names)
                y[out:out+n] = shard["target"][local]
                rates[out:out+n] = shard[ROUTE_FEATURE][local]
                for name in side:
                    side[name][out:out+n] = shard[name][local]
                out += n
            offset += count
    if not paths or offset != total or out != len(positions):
        raise RuntimeError(f"Invalid Stage 4 {split} shards: read={offset}, selected={out}, expected={total}")
    return x, y, rates, side


def _tsb_batches(target_start: int, target_end: int, last_sales_day: int):
    """Yield aligned TSB outputs for each source series batch, without a long melt."""
    target_positions = np.arange(target_start - 1, target_end, dtype=np.int64)
    daily = [f"d_{day}" for day in range(1, last_sales_day + 1)]
    reader = pd.read_csv(C.SALES_PATH, usecols=daily, dtype=np.int16, chunksize=S5.SERIES_BATCH_SIZE)
    for frame in reader:
        yield compute_tsb_forecasts(frame.to_numpy(copy=True), target_positions)


def train_stage5_validation() -> dict:
    """Train the routed LightGBM and compare the hybrid on validation only."""
    outputs = (S5.MODEL_PATH, S5.MODEL_METADATA_PATH,
               S5.VALIDATION_METRICS_PATH, S5.VALIDATION_PREDICTIONS_PATH)
    if any(path.exists() for path in outputs):
        raise FileExistsError("Refusing to overwrite Stage 5 artifacts")
    required = (S4.FEATURE_METADATA_PATH, S4.SERIES_METADATA_PATH, STAGE4_METADATA,
                STAGE4_VALIDATION_METRICS, STAGE4_VALIDATION_PREDS)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Stage 4 inputs: " + ", ".join(missing))

    logger = _logger()
    started = time.time()
    s4_meta = json.loads(STAGE4_METADATA.read_text(encoding="utf-8"))
    s4_manifest = json.loads(S4.FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
    if s4_manifest.get("status") != "complete":
        raise ValueError("Complete Stage 4 train/validation shards are required")
    if s4_meta["training_rows_used"] != S5.SAMPLE_ROWS or s4_meta["random_seed"] != S5.RANDOM_SEED:
        raise ValueError("Stage 5 sample policy or seed differs from the Stage 4 benchmark")
    feature_names = list(s4_meta["feature_names"])
    if feature_names[-len(NEW_FEATURES):] != NEW_FEATURES:
        raise ValueError("Stage 4 feature metadata does not match the hybrid model inputs")

    logger.info("Loading Stage 4 seeded sample (%d candidate rows)", S5.SAMPLE_ROWS)
    x_sample, y_sample, rates_sample, train_side = _load_split("train", feature_names, S5.SAMPLE_ROWS)
    # A full rate window is represented by a finite Stage 4 rate; early rows
    # with NaN route to LightGBM as specified in the approved plan.
    intermittent_sample = route_to_tsb(rates_sample, np.where(np.isfinite(rates_sample), S5.MIN_HISTORY_DAYS, 0))
    regular = ~intermittent_sample
    if not regular.any():
        raise RuntimeError("Routing left no regular-demand sample rows for LightGBM")
    x_train = x_sample[regular]
    y_train = y_sample[regular]
    regular_rows = int(regular.sum())
    training_series_used = int(np.unique(train_side["series_idx"][regular]).size)
    del x_sample, y_sample, rates_sample, train_side, regular, intermittent_sample
    categorical = [name for name in CATEGORICAL_FEATURES if name in feature_names]
    params = s4_meta["lightgbm_parameters"]
    train_set = lgb.Dataset(x_train, label=y_train, feature_name=feature_names,
                            categorical_feature=categorical, params=params, free_raw_data=True)
    train_set.construct()
    del x_train, y_train

    logger.info("Loading full Stage 4 validation population")
    x_val, y_val, rate_val, side = _load_split("validation", feature_names)
    val_rows = len(y_val)
    # Early stopping measures only the rows the LightGBM branch will serve.
    tsb_parts, rate_parts, route_parts = [], [], []
    for forecasts, rates, routed in _tsb_batches(S4.VALIDATION_START, S4.VALIDATION_END,
                                                  S4.VALIDATION_END - 1):
        tsb_parts.append(forecasts)
        rate_parts.append(rates)
        route_parts.append(routed)
    tsb_prediction = np.concatenate(tsb_parts)
    tsb_rate = np.concatenate(rate_parts)
    route = np.concatenate(route_parts)
    if len(route) != val_rows or not np.array_equal(route, route_to_tsb(rate_val, np.where(np.isfinite(rate_val), S5.MIN_HISTORY_DAYS, 0))):
        raise RuntimeError("TSB routing does not align with Stage 4 validation feature rows")
    if not np.allclose(tsb_rate, rate_val, rtol=0, atol=1e-7, equal_nan=True):
        raise RuntimeError("TSB rolling rates differ from the Stage 4 cutoff-safe rate feature")
    regular_val = ~route
    if not regular_val.any():
        raise RuntimeError("Validation set contains no regular-demand rows for LightGBM early stopping")
    x_val_regular = x_val[regular_val]
    y_val_regular = y_val[regular_val]
    valid_set = lgb.Dataset(x_val_regular, label=y_val_regular, reference=train_set, feature_name=feature_names,
                            categorical_feature=categorical, params=params, free_raw_data=True)
    valid_set.construct()
    del x_val_regular, y_val_regular, regular_val
    logger.info("Training regular-demand LightGBM on %d/%d sampled rows", regular_rows, S5.SAMPLE_ROWS)
    booster = lgb.train(
        params, train_set, num_boost_round=C.LIGHTGBM_MAX_BOOST_ROUNDS,
        valid_sets=[valid_set], valid_names=["validation"],
        callbacks=[lgb.early_stopping(C.LIGHTGBM_EARLY_STOPPING_ROUNDS, first_metric_only=True),
                   lgb.log_evaluation(C.LIGHTGBM_LOG_PERIOD)],
    )
    best_iteration = int(booster.best_iteration or C.LIGHTGBM_MAX_BOOST_ROUNDS)
    lightgbm_prediction = np.maximum(0, booster.predict(x_val, num_iteration=best_iteration)).astype(np.float32)

    target_start = S4.VALIDATION_START
    target_end = S4.VALIDATION_END
    hybrid_prediction = np.where(route, tsb_prediction, lightgbm_prediction).astype(np.float32)

    baselines = pd.read_csv(STAGE4_VALIDATION_PREDS)
    if len(baselines) != val_rows:
        raise ValueError("Stage 4 and Stage 5 validation populations differ")
    for name, values in (("series_idx", side["series_idx"]), ("target_day", side["target_day"])):
        if not np.array_equal(baselines[name].to_numpy(), values):
            raise ValueError(f"Validation {name} order differs from Stage 4")
    if not np.array_equal(baselines.actual.to_numpy(dtype=np.float32), y_val):
        raise ValueError("Validation actuals differ from Stage 4")
    cuts = json.loads(STAGE4_VALIDATION_METRICS.read_text(encoding="utf-8"))["demand_tier_cutpoints_from_stage3"]
    comparison = {
        "scope": "Validation only; final test not accessed.",
        "validation_rows": val_rows, "series_count": int(s4_manifest["series_count"]),
        "intermittent_rows_routed_to_tsb": int(route.sum()),
        "regular_rows_routed_to_lightgbm": int((~route).sum()),
        "demand_tier_cutpoints": cuts,
        "stage3_existing": _metric_bundle(y_val, baselines.stage3_lightgbm.to_numpy(np.float32), side["series_idx"],
                                           cuts["low_max"], cuts["medium_max"]),
        "stage4_existing": _metric_bundle(y_val, baselines.stage4_lightgbm.to_numpy(np.float32), side["series_idx"],
                                           cuts["low_max"], cuts["medium_max"]),
        "stage5_hybrid": _metric_bundle(y_val, hybrid_prediction, side["series_idx"],
                                         cuts["low_max"], cuts["medium_max"]),
    }
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "Stage 5 Stage 4-feature LightGBM + TSB routing",
        "feature_names": feature_names, "sample_policy": s4_meta["sample_policy"],
        "training_rows_available": int(s4_manifest["split"]["train"]["rows"]),
        "candidate_rows_sampled": S5.SAMPLE_ROWS, "lightgbm_training_rows_used": regular_rows,
        "lightgbm_training_series_used": training_series_used,
        "validation_rows": val_rows, "validation_series": int(s4_manifest["series_count"]),
        "training_period_target_days": s4_meta["training_period_target_days"],
        "validation_period_target_days": [target_start, target_end],
        "random_seed": S5.RANDOM_SEED, "lightgbm_parameters": params,
        "max_boost_rounds": C.LIGHTGBM_MAX_BOOST_ROUNDS,
        "early_stopping_rounds": C.LIGHTGBM_EARLY_STOPPING_ROUNDS,
        "best_iteration": best_iteration,
        "router": {"feature": ROUTE_FEATURE, "threshold": S5.INTERMITTENT_RATE_THRESHOLD,
                   "minimum_history_days": S5.MIN_HISTORY_DAYS,
                   "insufficient_history_policy": "route to LightGBM"},
        "tsb": {"alpha": S5.TSB_ALPHA, "beta": S5.TSB_BETA,
                "initialization": "d_1..d_28 occurrence rate and positive-size mean",
                "update_order": "forecast before observing target-day sales"},
        "missing_value_policy": "Keep Stage 4 NaNs; incomplete 84-day history routes to LightGBM; LightGBM native missing handling.",
        "test_accessed": False, "selection_status": "pending_validation_review",
        "promotion_status": "not_promoted", "peak_process_rss_mb": _peak_rss_mb(),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    S5.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    S5.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_temp = S5.MODEL_PATH.with_name(".m5_hybrid_stage5.tmp.txt")
    booster.save_model(str(model_temp), num_iteration=best_iteration)
    reloaded = lgb.Booster(model_file=str(model_temp))
    if not np.allclose(booster.predict(x_val[:128], num_iteration=best_iteration), reloaded.predict(x_val[:128]),
                       rtol=1e-6, atol=1e-6):
        model_temp.unlink(missing_ok=True)
        raise RuntimeError("Stage 5 saved model failed reload verification")
    os.replace(model_temp, S5.MODEL_PATH)
    S5.MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    S5.VALIDATION_METRICS_PATH.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    preds = pd.DataFrame({
        "series_idx": side["series_idx"], "target_day": side["target_day"], "actual": y_val,
        "intermittent_route": route, "nonzero_rate_84": rate_val, "tsb_prediction": tsb_prediction,
        "lightgbm_prediction": lightgbm_prediction, "hybrid_prediction": hybrid_prediction,
        "stage3_lightgbm": baselines.stage3_lightgbm.to_numpy(np.float32),
        "stage4_lightgbm": baselines.stage4_lightgbm.to_numpy(np.float32),
    })
    preds.to_csv(S5.VALIDATION_PREDICTIONS_PATH, index=False,
                 compression={"method": "gzip", "compresslevel": 1, "mtime": 0})
    logger.info("Validation complete; best iteration=%d, routed TSB rows=%d/%d; test untouched",
                best_iteration, int(route.sum()), val_rows)
    return {"validation_metrics": comparison, "metadata": metadata,
            "artifacts": [str(p) for p in outputs]}


def lock_stage5_selection(selected_candidate: str) -> dict:
    """Record an explicit post-validation choice before any final-test read."""
    if selected_candidate not in {"stage3", "stage4", "stage5_hybrid"}:
        raise ValueError("selected_candidate must be stage3, stage4, or stage5_hybrid")
    if not S5.VALIDATION_METRICS_PATH.is_file() or not S5.MODEL_METADATA_PATH.is_file():
        raise FileNotFoundError("Stage 5 validation artifacts are required before selection is locked")
    if S5.SELECTION_PATH.exists():
        raise FileExistsError("Stage 5 validation selection is already locked")
    metrics = json.loads(S5.VALIDATION_METRICS_PATH.read_text(encoding="utf-8"))
    selection = {"selected_candidate": selected_candidate,
                 "selection_metric": "validation MAE",
                 "validation_metrics": {name: metrics[name]["overall"]["MAE"]
                                        for name in ("stage3_existing", "stage4_existing", "stage5_hybrid")},
                 "selection_locked_at": datetime.now().isoformat(timespec="seconds"),
                 "test_accessed": False}
    S5.SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    S5.SELECTION_PATH.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    model_metadata = json.loads(S5.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    model_metadata["selection_status"] = "locked"
    model_metadata["selected_candidate"] = selected_candidate
    model_metadata["selection_locked_at"] = selection["selection_locked_at"]
    S5.MODEL_METADATA_PATH.write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")
    return selection


def evaluate_stage5_final_test() -> dict:
    """One-time held-out evaluation, available only after explicit selection lock."""
    if not S5.SELECTION_PATH.is_file():
        raise RuntimeError("Final test is locked; first review validation and explicitly lock candidate selection")
    selection = json.loads(S5.SELECTION_PATH.read_text(encoding="utf-8"))
    if selection.get("selected_candidate") not in {"stage3", "stage4", "stage5_hybrid"} or selection.get("test_accessed") is not False:
        raise RuntimeError("A valid validation model selection must be locked before final-test access")
    if S5.TEST_METRICS_PATH.exists() or S5.TEST_PREDICTIONS_PATH.exists():
        raise FileExistsError("Refusing to overwrite Stage 5 final-test artifacts")
    required = (S5.MODEL_PATH, S5.MODEL_METADATA_PATH, STAGE4_TEST_PREDS, C.FEATURE_METADATA_PATH)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing final-test inputs: " + ", ".join(missing))
    # Only after selection lock is verified do we enumerate/open test shards.
    shard_paths = sorted(STAGE2_TEST_DIR.glob("part_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"Stage 2 test shards are missing: {STAGE2_TEST_DIR}")
    logger = _logger(test=True)
    started = time.time()
    model_meta = json.loads(S5.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    expected_shards = (int(model_meta["validation_series"]) + S5.SERIES_BATCH_SIZE - 1) // S5.SERIES_BATCH_SIZE
    if len(shard_paths) != expected_shards:
        raise RuntimeError(f"Expected {expected_shards} test shards, found {len(shard_paths)}")
    model = lgb.Booster(model_file=str(S5.MODEL_PATH))
    stage4_test = pd.read_csv(STAGE4_TEST_PREDS)
    # Stage 2 test base features are read unchanged. Stage 4/5 historical
    # features and TSB state are computed in memory in bounded series batches.
    target_days = np.arange(C.TEST_DAY_START, C.TEST_DAY_END + 1, dtype=np.uint16)
    target_positions = target_days.astype(np.int64) - 1
    total = int(model_meta["validation_series"]) * len(target_days)
    prediction = np.empty(total, dtype=np.float32)
    actual = np.empty(total, dtype=np.float32)
    route_all = np.empty(total, dtype=bool)
    series_all = np.empty(total, dtype=np.uint32)
    days_all = np.empty(total, dtype=np.uint16)
    offset = series_start = 0
    daily_columns = [f"d_{day}" for day in range(1, C.TEST_DAY_END + 1)]
    reader = pd.read_csv(C.SALES_PATH, usecols=daily_columns, dtype=np.int16, chunksize=S5.SERIES_BATCH_SIZE)
    feature_names = model_meta["feature_names"]
    base_names = [name for name in feature_names if name not in NEW_FEATURES]
    logger.info("Selection locked to %s; evaluating Stage 5 candidate on %d final-test rows",
                selection["selected_candidate"], total)
    for batch_index, (path, frame) in enumerate(zip(shard_paths, reader)):
        sales = frame.to_numpy(dtype=np.int16, copy=True)
        extra = compute_intermittent_features(sales, target_positions)
        tsb, rates, routed = compute_tsb_forecasts(sales, target_positions)
        with np.load(path) as shard:
            n = len(shard["target"])
            if n != len(sales) * len(target_days):
                raise RuntimeError(f"Test shard row count mismatch: {path.name}")
            expected_series = np.repeat(np.arange(series_start, series_start + len(sales), dtype=np.uint32), len(target_days))
            expected_days = np.tile(target_days, len(sales))
            if not np.array_equal(shard["series_idx"], expected_series) or not np.array_equal(shard["target_day"], expected_days):
                raise RuntimeError(f"Test shard series/date alignment failed: {path.name}")
            missing_features = [name for name in base_names if name not in shard.files]
            if missing_features:
                raise RuntimeError(f"Stage 2 test shard lacks base features: {missing_features}")
            cols = {name: shard[name] for name in base_names}
            cols.update(extra)
            matrix = feature_matrix(cols, feature_names)
            pred_lgb = np.maximum(0, model.predict(matrix, num_iteration=model_meta["best_iteration"])).astype(np.float32)
            hybrid = np.where(routed, tsb, pred_lgb).astype(np.float32)
            sl = slice(offset, offset+n)
            prediction[sl] = hybrid
            actual[sl] = shard["target"]
            route_all[sl] = routed
            series_all[sl] = shard["series_idx"]
            days_all[sl] = shard["target_day"]
            offset += n
        series_start += len(sales)
    if offset != total or series_start != int(model_meta["validation_series"]):
        raise RuntimeError(f"Final-test coverage mismatch: rows={offset}/{total}, series={series_start}/{model_meta['validation_series']}")
    if len(stage4_test) != total or not np.array_equal(stage4_test.series_idx.to_numpy(), series_all) or not np.array_equal(stage4_test.target_day.to_numpy(), days_all) or not np.array_equal(stage4_test.actual.to_numpy(np.float32), actual):
        raise ValueError("Stage 4 and Stage 5 final-test rows are not aligned")
    cuts = json.loads(S5.VALIDATION_METRICS_PATH.read_text(encoding="utf-8"))["demand_tier_cutpoints"]
    metrics = {
        "scope": "Final test evaluated once after validation selection lock; no tuning follows.",
        "selected_candidate": selection["selected_candidate"],
        "evaluated_candidate": "stage5_hybrid", "selection_locked_at": selection["selection_locked_at"],
        "test_rows": total,
        "series_count": int(model_meta["validation_series"]), "target_days": [C.TEST_DAY_START, C.TEST_DAY_END],
        "intermittent_rows_routed_to_tsb": int(route_all.sum()),
        "stage3_existing": _metric_bundle(actual, stage4_test.stage3_lightgbm.to_numpy(np.float32), series_all,
                                           cuts["low_max"], cuts["medium_max"]),
        "stage4_existing": _metric_bundle(actual, stage4_test.stage4_lightgbm.to_numpy(np.float32), series_all,
                                           cuts["low_max"], cuts["medium_max"]),
        "stage5_hybrid": _metric_bundle(actual, prediction, series_all, cuts["low_max"], cuts["medium_max"]),
        "test_accessed_after_selection": True,
        "model_selection_changed_after_test": False,
        "elapsed_seconds": round(time.time() - started, 2),
        "peak_process_rss_mb": _peak_rss_mb(),
    }
    preds = pd.DataFrame({"series_idx": series_all, "target_day": days_all, "actual": actual,
                          "intermittent_route": route_all, "stage3_lightgbm": stage4_test.stage3_lightgbm,
                          "stage4_lightgbm": stage4_test.stage4_lightgbm, "stage5_hybrid": prediction})
    S5.TEST_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    preds.to_csv(S5.TEST_PREDICTIONS_PATH, index=False,
                 compression={"method": "gzip", "compresslevel": 1, "mtime": 0})
    logger.info("Final test evaluation saved; selection unchanged")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_stage5_validation(), indent=2))
