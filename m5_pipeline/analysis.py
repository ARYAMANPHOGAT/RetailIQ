"""Validation-only explainability and error analysis for the saved M5 model."""
from __future__ import annotations

import html
import json
import logging
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C
from .features import _peak_rss_mb
from .metrics import regression_metrics
from .training import _load_split

MODEL_PATH = C.MODELS_DIR / "m5_lightgbm.txt"
MODEL_META_PATH = C.MODELS_DIR / "m5_training_metadata.json"
PREDICTIONS_PATH = C.OUTPUTS_DIR / "m5_validation_predictions.csv.gz"


def _logger() -> logging.Logger:
    C.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("m5.analysis")
    if not logger.handlers:
        handler = logging.FileHandler(C.ANALYSIS_LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _group_metrics(series: pd.DataFrame, group_col: str) -> list[dict]:
    grouped = series.groupby(group_col, dropna=False, sort=True, observed=True).agg(
        series_count=("series_idx", "count"),
        rows=("row_count", "sum"),
        actual_sum=("actual_sum", "sum"),
        absolute_error_sum=("absolute_error_sum", "sum"),
        signed_error_sum=("signed_error_sum", "sum"),
    ).reset_index()
    grouped["MAE"] = grouped.absolute_error_sum / grouped.rows
    grouped["WAPE_percent"] = np.where(grouped.actual_sum != 0,
                                       100 * grouped.absolute_error_sum / grouped.actual_sum, np.nan)
    grouped["mean_bias"] = grouped.signed_error_sum / grouped.rows
    grouped["bias_direction"] = np.where(grouped.mean_bias > 0, "overpredict",
                                         np.where(grouped.mean_bias < 0, "underpredict", "balanced"))
    grouped["mean_actual_per_row"] = grouped.actual_sum / grouped.rows
    return grouped.to_dict(orient="records")


def _bias_extremes(groups: pd.DataFrame, count: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return genuinely positive-bias groups and most negative-bias groups."""
    over = groups[groups.mean_bias > 0].nlargest(count, "mean_bias")
    under = groups[groups.mean_bias < 0].nsmallest(count, "mean_bias")
    return over, under


def _svg_bar_chart(path: Path, title: str, labels: list[str], values: list[float],
                   value_label: str, color: str = "#4f81bd", signed: bool = False) -> None:
    width, row_height, left, right, top = 900, 34, 250, 80, 70
    height = max(170, top + row_height * len(labels) + 35)
    finite = [abs(v) for v in values if np.isfinite(v)]
    max_value = max(finite, default=1.0) or 1.0
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="#ffffff"/>',
              f'<text x="{left}" y="36" font-family="Arial" font-size="20" font-weight="bold">{html.escape(title)}</text>']
    plot_width = width - left - right
    center_x = left + plot_width // 2
    scale_width = plot_width / 2 if signed else plot_width
    if signed:
        pieces.append(f'<line x1="{center_x}" y1="{top-5}" x2="{center_x}" y2="{height-25}" stroke="#777" stroke-width="1"/>')
    for index, (label, value) in enumerate(zip(labels, values)):
        y = top + index * row_height
        bar_width = max(1, int(scale_width * abs(value) / max_value)) if np.isfinite(value) else 1
        bar_x = center_x - bar_width if signed and value < 0 else (center_x if signed else left)
        bar_color = ("#4f81bd" if value < 0 else "#c26a52") if signed else color
        pieces.append(f'<text x="{left-10}" y="{y+17}" text-anchor="end" font-family="Arial" font-size="12">{html.escape(label)}</text>')
        pieces.append(f'<rect x="{bar_x}" y="{y}" width="{bar_width}" height="22" fill="{bar_color}"/>')
        shown = f"{value:.4g}" if np.isfinite(value) else "n/a"
        text_x = bar_x + bar_width + 6 if not signed or value >= 0 else bar_x - 6
        anchor = "start" if not signed or value >= 0 else "end"
        pieces.append(f'<text x="{text_x}" y="{y+16}" text-anchor="{anchor}" font-family="Arial" font-size="12">{shown} {html.escape(value_label)}</text>')
    pieces.append('</svg>')
    path.write_text("\n".join(pieces), encoding="utf-8")


def _permutation_importance(model: lgb.Booster, feature_names: list[str], sample_size: int,
                            seed: int) -> tuple[list[dict], dict]:
    x, y, side, series = _load_split("validation", feature_names, sample_size, seed)
    baseline_prediction = np.maximum(0, model.predict(x)).astype(np.float32)
    baseline = regression_metrics(y, baseline_prediction)
    rng = np.random.default_rng(seed)
    rows = []
    for column, name in enumerate(feature_names):
        deltas = []
        for _ in range(C.ANALYSIS_PERMUTATION_REPEATS):
            permuted = x.copy()
            order = rng.permutation(len(permuted))
            permuted[:, column] = x[order, column]
            prediction = np.maximum(0, model.predict(permuted)).astype(np.float32)
            deltas.append(regression_metrics(y, prediction)["MAE"] - baseline["MAE"])
        rows.append({"feature": name, "permutation_delta_MAE": float(np.mean(deltas)),
                     "std_delta_MAE": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                     "baseline_sample_MAE": baseline["MAE"], "permutation_repeats": len(deltas)})
    rows.sort(key=lambda row: row["permutation_delta_MAE"], reverse=True)
    return rows, {"rows": len(y), "series_covered": len(series), "target_day_min": int(side["target_day"].min()),
                  "target_day_max": int(side["target_day"].max()), "seed": seed,
                  "baseline_metrics": baseline}


def run_validation_analysis() -> dict:
    """Write explainability outputs under outputs/m5/analysis; never access test data."""
    required = (MODEL_PATH, MODEL_META_PATH, PREDICTIONS_PATH, C.SERIES_METADATA_PATH,
                C.FEATURE_METADATA_PATH)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required M5 analysis input(s): " + ", ".join(missing))
    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    prior_report = C.ANALYSIS_DIR / "report.json"
    if prior_report.exists() and json.loads(prior_report.read_text(encoding="utf-8")).get("analysis_scope") != \
            "Validation split only; final test predictions and shards were not accessed.":
        raise FileExistsError(f"Refusing to overwrite existing M5 analysis outputs: {C.ANALYSIS_DIR}")
    logger = _logger()
    logger.info("Starting validation-only explainability and error analysis")
    model_meta = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    feature_meta = json.loads(C.FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
    model = lgb.Booster(model_file=str(MODEL_PATH))
    feature_names = model_meta["feature_names"]
    if model.feature_name() != feature_names:
        raise ValueError("Saved booster feature names do not match its training metadata")

    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    })
    feature_importance["gain_share_percent"] = np.where(
        feature_importance.gain.sum() > 0,
        feature_importance.gain / feature_importance.gain.sum() * 100, 0.0)
    feature_importance = feature_importance.sort_values(["gain", "split"], ascending=False)
    feature_importance.to_csv(C.ANALYSIS_DIR / "feature_importance.csv", index=False)

    metadata = pd.read_csv(C.SERIES_METADATA_PATH).sort_values("series_idx").reset_index(drop=True)
    expected_series = int(feature_meta["series_count"])
    if len(metadata) != expected_series or not np.array_equal(metadata.series_idx.to_numpy(), np.arange(expected_series)):
        raise ValueError("Series metadata is incomplete or not indexed consecutively")
    arrays = {name: np.zeros(expected_series, dtype=np.float64)
              for name in ("row_count", "actual_sum", "prediction_sum", "absolute_error_sum",
                           "squared_error_sum", "signed_error_sum")}
    validation_rows = 0
    for chunk in pd.read_csv(PREDICTIONS_PATH, chunksize=100_000):
        sid = chunk.series_idx.to_numpy(dtype=np.int64)
        if np.any(sid < 0) or np.any(sid >= expected_series):
            raise ValueError("Validation predictions contain an unknown series_idx")
        day = chunk.target_day.to_numpy(dtype=np.int16)
        if np.any(day < 1886) or np.any(day > 1913):
            raise ValueError("Analysis prediction file contains dates outside the validation split")
        if not np.array_equal(chunk.series_id.astype(str).to_numpy(), metadata.id.to_numpy()[sid]):
            raise ValueError("Validation prediction series IDs are misaligned")
        actual = chunk.actual.to_numpy(dtype=np.float64)
        prediction = chunk.lightgbm.to_numpy(dtype=np.float64)
        error = prediction - actual
        validation_rows += len(chunk)
        arrays["row_count"] += np.bincount(sid, minlength=expected_series)
        arrays["actual_sum"] += np.bincount(sid, weights=actual, minlength=expected_series)
        arrays["prediction_sum"] += np.bincount(sid, weights=prediction, minlength=expected_series)
        arrays["absolute_error_sum"] += np.bincount(sid, weights=np.abs(error), minlength=expected_series)
        arrays["squared_error_sum"] += np.bincount(sid, weights=np.square(error), minlength=expected_series)
        arrays["signed_error_sum"] += np.bincount(sid, weights=error, minlength=expected_series)
    expected_rows = int(feature_meta["split"]["validation"]["rows"])
    if validation_rows != expected_rows or np.any(arrays["row_count"] == 0):
        raise ValueError(f"Validation predictions contain {validation_rows} rows; expected {expected_rows}")

    series = metadata[["series_idx", "id", "item_id", "store_id", "state_id", "cat_id"]].copy()
    for key, values in arrays.items():
        series[key] = values
    series["mean_demand"] = series.actual_sum / series.row_count
    series["MAE"] = series.absolute_error_sum / series.row_count
    series["RMSE"] = np.sqrt(series.squared_error_sum / series.row_count)
    series["WAPE_percent"] = np.where(series.actual_sum != 0,
                                      100 * series.absolute_error_sum / series.actual_sum, np.nan)
    series["mean_bias"] = series.signed_error_sum / series.row_count
    series["bias_direction"] = np.where(series.mean_bias > 0, "overpredict",
                                        np.where(series.mean_bias < 0, "underpredict", "balanced"))
    thresholds = np.quantile(series.mean_demand, [1 / 3, 2 / 3])
    series["demand_level"] = np.select(
        [series.mean_demand <= thresholds[0], series.mean_demand <= thresholds[1]],
        ["low", "medium"], default="high")
    series["demand_level"] = pd.Categorical(series.demand_level, categories=["low", "medium", "high"], ordered=True)
    series.sort_values(["MAE", "WAPE_percent"], ascending=False).head(50).to_csv(
        C.ANALYSIS_DIR / "worst_series.csv", index=False)

    group_output = []
    for column, dimension in (("store_id", "store"), ("state_id", "state"),
                              ("cat_id", "category"), ("demand_level", "demand_level")):
        for record in _group_metrics(series, column):
            record["dimension"] = dimension
            record["group"] = str(record.pop(column))
            group_output.append(record)
    groups = pd.DataFrame(group_output)
    groups.to_csv(C.ANALYSIS_DIR / "group_metrics.csv", index=False)

    importance_rows, permutation_info = _permutation_importance(
        model, feature_names, C.ANALYSIS_SAMPLE_ROWS, C.RANDOM_SEED)
    pd.DataFrame(importance_rows).to_csv(C.ANALYSIS_DIR / "permutation_importance.csv", index=False)
    demand_groups = groups[groups.dimension == "demand_level"].set_index("group").reindex(["low", "medium", "high"])
    bias_candidates = groups[groups.dimension.isin(["store", "state", "category", "demand_level"])]
    most_over, most_under = _bias_extremes(bias_candidates)
    report = {
        "analysis_scope": "Validation split only; final test predictions and shards were not accessed.",
        "model": str(MODEL_PATH.relative_to(C.ROOT_DIR)),
        "best_iteration": int(model_meta["best_iteration"]),
        "validation_rows": validation_rows,
        "series_count": expected_series,
        "permutation_sampling": {**permutation_info,
                                  "policy": "Uniform global row sample without replacement from validation shards.",
                                  "feature_permutations": "Shuffle one feature column at a time; three seeded repeats; report increase in sample MAE."},
        "demand_level_cutpoints_mean_daily_units": {"low_max": float(thresholds[0]), "medium_max": float(thresholds[1])},
        "demand_level_metrics": demand_groups[["series_count", "rows", "MAE", "WAPE_percent", "mean_bias", "bias_direction"]].reset_index().to_dict(orient="records"),
        "most_overpredicting_groups": [{"dimension": r.dimension, "group": r.group, "mean_bias": r.mean_bias} for r in most_over.itertuples()],
        "most_underpredicting_groups": [{"dimension": r.dimension, "group": r.group, "mean_bias": r.mean_bias} for r in most_under.itertuples()],
        "worst_series_by_MAE": series.nlargest(10, "MAE")[["id", "store_id", "state_id", "cat_id", "mean_demand", "MAE", "WAPE_percent", "mean_bias", "bias_direction"]].to_dict(orient="records"),
        "importance_top_gain": feature_importance.head(10).to_dict(orient="records"),
        "permutation_top_MAE_increase": importance_rows[:10],
        "peak_process_rss_mb": _peak_rss_mb(),
    }
    (C.ANALYSIS_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    top_gain = feature_importance.head(12).sort_values("gain")
    _svg_bar_chart(C.ANALYSIS_DIR / "feature_gain.svg", "LightGBM gain importance (top 12)",
                   top_gain.feature.tolist(), top_gain.gain_share_percent.tolist(), "% gain")
    _svg_bar_chart(C.ANALYSIS_DIR / "demand_level_mae.svg", "Validation MAE by series demand level",
                   [str(value) for value in demand_groups.index], demand_groups.MAE.astype(float).tolist(), "units")
    bias_chart = pd.concat([most_under.head(5), most_over.head(5)]).drop_duplicates(["dimension", "group"])
    _svg_bar_chart(C.ANALYSIS_DIR / "group_bias.svg", "Mean signed prediction error (positive = overpredict)",
                   [f"{row.dimension}: {row.group}" for row in bias_chart.itertuples()],
                   bias_chart.mean_bias.astype(float).tolist(), "units", signed=True)
    logger.info("Analysis complete; validation_rows=%d, permutation_sample=%d", validation_rows,
                permutation_info["rows"])
    return report


if __name__ == "__main__":
    print(json.dumps(run_validation_analysis(), indent=2, default=str))
