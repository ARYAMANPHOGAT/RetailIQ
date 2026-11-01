"""Lightweight post-hoc analysis using Stage 4-owned saved predictions only."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import stage4_config as S4

ANALYSIS_DIR = S4.STAGE4_ROOT.parent / "stage4_analysis"
SERIES_PATH = S4.FEATURES_DIR / "series_metadata.csv"
VALIDATION_PATH = S4.RESULTS_DIR / "stage4_validation_predictions.csv.gz"
TEST_PATH = S4.RESULTS_DIR / "stage4_test_predictions.csv.gz"
VALIDATION_METRICS_PATH = S4.RESULTS_DIR / "stage4_metrics.json"
TEST_METRICS_PATH = S4.RESULTS_DIR / "stage4_test_metrics.json"
MODEL_COLUMNS = {"Stage 3 baseline": "stage3_lightgbm", "Stage 4": "stage4_lightgbm"}
MARKDOWN_REPORT_PATH = ANALYSIS_DIR / "analysis_report.md"


def _metric_rows(frame: pd.DataFrame, split: str, group: str) -> pd.DataFrame:
    rows = []
    for model, prediction_col in MODEL_COLUMNS.items():
        pred = frame[prediction_col].to_numpy(dtype=np.float64)
        actual = frame.actual.to_numpy(dtype=np.float64)
        error = pred - actual
        work = pd.DataFrame({
            "group": frame[group].astype(str).to_numpy(), "actual": actual,
            "abs_error": np.abs(error), "squared_error": error * error,
            "error": error, "over": error > 0, "under": error < 0,
        })
        grouped = work.groupby("group", sort=True, observed=True).agg(
            rows=("actual", "size"), actual_units=("actual", "sum"),
            absolute_error=("abs_error", "sum"), mae=("abs_error", "mean"),
            mse=("squared_error", "mean"), mean_bias=("error", "mean"),
            overprediction_rate=("over", "mean"), underprediction_rate=("under", "mean"),
        ).reset_index()
        grouped["rmse"] = np.sqrt(grouped.pop("mse"))
        grouped["wape_percent"] = np.where(
            grouped.actual_units > 0, grouped.absolute_error / grouped.actual_units * 100, np.nan)
        grouped["split"] = split
        grouped["group_type"] = group
        grouped["model"] = model
        rows.append(grouped.rename(columns={"group": "group_value"}))
    return pd.concat(rows, ignore_index=True)


def _overall(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["all"] = "All"
    return _metric_rows(frame, split, "all")


def _series_report(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for model, pred_col in MODEL_COLUMNS.items():
        err = frame[pred_col].to_numpy(dtype=np.float64) - frame.actual.to_numpy(dtype=np.float64)
        tmp = pd.DataFrame({"series_idx": frame.series_idx.to_numpy(),
                            "actual": frame.actual.to_numpy(dtype=np.float64),
                            "abs_error": np.abs(err), "sq_error": err * err,
                            "bias_sum": err})
        agg = tmp.groupby("series_idx", sort=False).agg(
            rows=("actual", "size"), actual_units=("actual", "sum"),
            mae=("abs_error", "mean"), absolute_error=("abs_error", "sum"),
            mse=("sq_error", "mean"), mean_bias=("bias_sum", "mean"),
        ).reset_index()
        agg["rmse"] = np.sqrt(agg.pop("mse"))
        agg["wape_percent"] = np.where(agg.actual_units > 0,
                                       agg.absolute_error / agg.actual_units * 100, np.nan)
        agg["model"] = model
        agg["split"] = split
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def _svg_bar_chart(path: Path, title: str, labels: list[str], series: list[tuple[str, list[float], str]]) -> None:
    width, row_h, left, right, top = 980, 27, 230, 170, 66
    height = top + row_h * len(labels) + 38
    values = [v for _, vals, _ in series for v in vals if np.isfinite(v)]
    maximum = max(values, default=1.0)
    maximum = maximum if maximum > 0 else 1.0
    plot_width = width - left - right
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              f'<text x="{left}" y="30" font-family="Arial" font-size="20" font-weight="bold">{html.escape(title)}</text>']
    for j, (name, _, color) in enumerate(series):
        x = left + j * 190
        chunks.extend([f'<rect x="{x}" y="43" width="13" height="13" fill="{color}"/>',
                       f'<text x="{x+19}" y="55" font-family="Arial" font-size="12">{html.escape(name)}</text>'])
    for i, label in enumerate(labels):
        y = top + i * row_h
        chunks.append(f'<text x="{left-12}" y="{y+12}" text-anchor="end" font-family="Arial" font-size="12">{html.escape(label)}</text>')
        for j, (name, vals, color) in enumerate(series):
            value = vals[i]
            if not np.isfinite(value):
                continue
            bar_y = y + j * 5
            bar_w = max(0, value / maximum * plot_width)
            chunks.append(f'<rect x="{left}" y="{bar_y}" width="{bar_w:.2f}" height="4" fill="{color}"/>')
        chunks.append(f'<line x1="{left}" y1="{y+row_h-2}" x2="{width-right}" y2="{y+row_h-2}" stroke="#eeeeee"/>')
    chunks.append('</svg>')
    path.write_text("\n".join(chunks), encoding="utf-8")


def render_markdown_report(overwrite: bool = False) -> Path:
    """Render a concise human-readable report from the generated CSVs only."""
    if MARKDOWN_REPORT_PATH.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite report: {MARKDOWN_REPORT_PATH}")
    demand = pd.read_csv(ANALYSIS_DIR / "stage4_demand_level_deltas.csv")
    frequency = pd.read_csv(ANALYSIS_DIR / "sales_frequency_comparison.csv")
    overall = pd.read_csv(ANALYSIS_DIR / "group_error_breakdown.csv")
    stores = pd.read_csv(ANALYSIS_DIR / "store_error_breakdown_stage4.csv")
    worst_mae = pd.read_csv(ANALYSIS_DIR / "worst_products_mae_test.csv").head(5)
    worst_wape = pd.read_csv(ANALYSIS_DIR / "worst_products_wape_test.csv").head(5)

    def table(headers: list[str], rows: list[list[object]]) -> str:
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
        return "\n".join(lines)

    def f(value, digits=3):
        return f"{float(value):.{digits}f}"

    all_rows = overall[(overall.group_type == "all") & (overall.model == "Stage 4")]
    overall_rows = [[r.split, f(r.mae), f(r.rmse), f(r.wape_percent, 2) + "%", f(r.mean_bias)]
                    for r in all_rows.itertuples()]
    demand_rows = []
    for split in ("validation", "test"):
        for level in ("low", "medium", "high"):
            r = demand[(demand.split == split) & (demand.demand_level == level)].iloc[0]
            demand_rows.append([split, level, f(r.mae_change_stage4_minus_stage3),
                                f(r.rmse_change_stage4_minus_stage3),
                                f(r.wape_change_pp_stage4_minus_stage3) + " pp"])
    freq_rows = []
    freq_s4 = frequency[frequency.model == "Stage 4"]
    for split in ("validation", "test"):
        for level in ("intermittent (<=20% days)", "middle frequency", "frequent (>=80% days)"):
            r = freq_s4[(freq_s4.split == split) & (freq_s4.group_value == level)].iloc[0]
            freq_rows.append([split, level, int(r.rows), f(r.mae), f(r.wape_percent, 2) + "%", f(r.mean_bias)])
    freq_pivot = frequency.pivot(index=["split", "group_value"], columns="model", values=["mae", "wape_percent"])
    freq_delta_rows = []
    for (split, level), r in freq_pivot.iterrows():
        freq_delta_rows.append([split, level,
                                f(r[("mae", "Stage 4")] - r[("mae", "Stage 3 baseline")]),
                                f(r[("wape_percent", "Stage 4")] - r[("wape_percent", "Stage 3 baseline")], 2) + " pp"])
    store_rows = [[r.group_value, f(r.wape_percent, 2) + "%", f(r.mae), f(r.mean_bias)]
                  for r in stores[stores.split == "test"].sort_values("wape_percent", ascending=False).head(5).itertuples()]

    def product_rows(frame):
        return [[r.id, r.store_id, r.cat_id, f(r.mae), f(r.wape_percent, 1) + "%", f(r.mean_bias)]
                for r in frame.itertuples()]

    content = [
        "# Stage 4 error analysis",
        "",
        "This report uses Stage 4-owned saved validation and final-test predictions, Stage 4 series metadata, and Stage 4 metrics. The Stage 3 comparison is read only from baseline prediction columns embedded in those Stage 4 prediction files; no Stage 3 source artifact was opened. No model was trained.",
        "",
        "Demand levels use the fixed Stage 3 cutpoints copied into the Stage 4 metrics. Intermittent/frequent groups use each split's saved actuals: intermittent means sales on at most 20% of days; frequent means sales on at least 80% of days.",
        "",
        "## Stage 4 overall performance",
        "",
        table(["Split", "MAE", "RMSE", "WAPE", "Mean bias"], overall_rows),
        "",
        "Negative mean bias indicates underprediction. Stage 4 row-level overprediction is more common than underprediction, but the size of underpredictions outweighs the overpredictions in aggregate.",
        "",
        "## Effect by demand level",
        "",
        "Changes are Stage 4 minus the embedded Stage 3 baseline; negative values favor Stage 4.",
        "",
        table(["Split", "Demand", "MAE change", "RMSE change", "WAPE change"], demand_rows),
        "",
        "Stage 4 improves MAE and WAPE in all three demand tiers on both splits. The largest WAPE gain is for medium demand (about 1.06 percentage points on validation and 0.90 points on test). RMSE is slightly worse for low and high demand, so the gain is not uniform across error metrics.",
        "",
        "## Intermittent versus frequent sales",
        "",
        table(["Split", "Frequency", "Rows", "MAE", "WAPE", "Mean bias"], freq_rows),
        "",
        "Intermittent WAPE is above 100% because absolute prediction error is about as large as or larger than the small total actual volume. Many zero-actual days contribute little denominator; false-positive forecasts can therefore dominate WAPE. Frequent sellers have lower WAPE but larger absolute errors and stronger negative unit bias.",
        "",
        "### Stage 4 change by sales frequency",
        "",
        "Changes are Stage 4 minus the embedded Stage 3 baseline.",
        "",
        table(["Split", "Frequency", "MAE change", "WAPE change"], freq_delta_rows),
        "",
        "Stage 4's intermittent test improvement is small (about 0.09 WAPE points and 0.0001 MAE); its stronger intermittent WAPE gain appears on validation (about 1.56 points). Treat this as limited evidence of transfer to the test window.",
        "",
        "## Stores and product series",
        "",
        "### Highest Stage 4 test store WAPE",
        "",
        table(["Store", "WAPE", "MAE", "Mean bias"], store_rows),
        "",
        "### Highest Stage 4 test series MAE",
        "",
        table(["Series", "Store", "Category", "MAE", "WAPE", "Mean bias"], product_rows(worst_mae)),
        "",
        "### Highest Stage 4 test series WAPE",
        "",
        table(["Series", "Store", "Category", "MAE", "WAPE", "Mean bias"], product_rows(worst_wape)),
        "",
        "Series WAPE can become very large when actual volume is only one or two units over 28 days; use MAE alongside it when prioritizing products.",
        "",
        "## Charts",
        "",
        "![Demand-level MAE comparison](demand_level_mae.svg)",
        "",
        "![Worst test stores by Stage 4 WAPE](worst_test_stores_wape.svg)",
        "",
        "## Three promising future experiments",
        "",
        "1. Test a demand-occurrence plus positive-size model, or a TSB/Croston-style baseline, for intermittent series to reduce false-positive demand and improve WAPE.",
        "2. Test store/category residual calibration or hierarchical shrinkage for persistent group bias; select any calibration using validation only.",
        "3. Test event and price-change interactions or spike-aware recent-demand features for high-volume series, tracking RMSE as well as MAE/WAPE.",
        "",
        "Limitations: frequency groups are defined separately within each evaluated 28-day period; group summaries describe errors but do not establish causes.",
    ]
    MARKDOWN_REPORT_PATH.write_text("\n".join(content) + "\n", encoding="utf-8")
    return MARKDOWN_REPORT_PATH


def _make_split(path: Path, split: str, series_meta: pd.DataFrame, cuts: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path)
    needed = {"series_idx", "target_day", "actual", *MODEL_COLUMNS.values()}
    if not needed.issubset(frame.columns):
        raise ValueError(f"Unexpected {split} prediction columns: {sorted(set(frame.columns) ^ needed)}")
    if frame.duplicated(["series_idx", "target_day"]).any():
        raise ValueError(f"Duplicate series/date rows in {split} predictions")
    demand = frame.groupby("series_idx", sort=False).actual.mean()
    nonzero = frame.assign(_positive=frame.actual.gt(0)).groupby("series_idx", sort=False)._positive.mean()
    features = pd.DataFrame({"series_idx": demand.index.to_numpy(),
                             "mean_daily_demand": demand.to_numpy(),
                             "nonzero_day_rate": nonzero.reindex(demand.index).to_numpy()})
    features["demand_level"] = np.select(
        [features.mean_daily_demand <= cuts["low_max"],
         features.mean_daily_demand <= cuts["medium_max"]],
        ["low", "medium"], default="high")
    features["sales_frequency"] = np.select(
        [features.nonzero_day_rate <= 0.20, features.nonzero_day_rate >= 0.80],
        ["intermittent (<=20% days)", "frequent (>=80% days)"], default="middle frequency")
    frame = frame.merge(features, on="series_idx", how="left", validate="many_to_one")
    frame = frame.merge(series_meta[["series_idx", "id", "item_id", "store_id", "state_id", "cat_id", "dept_id"]],
                        on="series_idx", how="left", validate="many_to_one")
    if frame.item_id.isna().any():
        raise ValueError(f"Could not map all {split} series to Stage 4 metadata")
    groups = ["demand_level", "sales_frequency", "store_id", "state_id", "cat_id"]
    reports = [_overall(frame, split)] + [_metric_rows(frame, split, col) for col in groups]
    return frame, pd.concat(reports, ignore_index=True)


def run_stage4_error_analysis() -> dict:
    """Create compact reports and vector SVG charts from saved Stage 4 outputs."""
    inputs = (SERIES_PATH, VALIDATION_PATH, TEST_PATH, VALIDATION_METRICS_PATH, TEST_METRICS_PATH)
    missing = [str(p) for p in inputs if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing Stage 4 analysis inputs: " + ", ".join(missing))
    expected_outputs = {
        "group_error_breakdown.csv", "series_error_metrics.csv", "worst_products_mae_validation.csv",
        "worst_products_wape_validation.csv", "worst_products_mae_test.csv", "worst_products_wape_test.csv",
        "store_error_breakdown_stage4.csv", "demand_level_comparison.csv", "sales_frequency_comparison.csv",
        "product_category_comparison.csv", "stage4_demand_level_deltas.csv", "demand_level_mae.svg",
        "worst_test_stores_wape.svg", "analysis_summary.json", "analysis_report.md",
    }
    if ANALYSIS_DIR.exists():
        existing = {p.name for p in ANALYSIS_DIR.iterdir()}
        if "analysis_summary.json" in existing or not existing.issubset(expected_outputs):
            raise FileExistsError(f"Refusing to overwrite existing analysis results: {ANALYSIS_DIR}")
        # Resume our own interrupted run when it left only known partial outputs.
    else:
        ANALYSIS_DIR.mkdir(parents=True)
    series = pd.read_csv(SERIES_PATH, usecols=["series_idx", "id", "item_id", "store_id", "state_id", "cat_id", "dept_id"])
    if series.series_idx.duplicated().any():
        raise ValueError("Duplicate series_idx in Stage 4 series metadata")
    validation_metrics = json.loads(VALIDATION_METRICS_PATH.read_text(encoding="utf-8"))
    test_metrics = json.loads(TEST_METRICS_PATH.read_text(encoding="utf-8"))
    cuts = validation_metrics["demand_tier_cutpoints_from_stage3"]
    val, val_groups = _make_split(VALIDATION_PATH, "validation", series, cuts)
    test, test_groups = _make_split(TEST_PATH, "test", series, cuts)
    group_metrics = pd.concat([val_groups, test_groups], ignore_index=True)
    group_metrics.to_csv(ANALYSIS_DIR / "group_error_breakdown.csv", index=False)

    # Per-series rows for both models, ranked by Stage 4 error in separate tables.
    products = pd.concat([_series_report(val, "validation"), _series_report(test, "test")], ignore_index=True)
    products = products.merge(series, on="series_idx", how="left", validate="many_to_one")
    products.to_csv(ANALYSIS_DIR / "series_error_metrics.csv", index=False)
    worst = products[products.model.eq("Stage 4")].copy()
    for split in ("validation", "test"):
        sub = worst[worst.split.eq(split)]
        sub.sort_values("mae", ascending=False).head(10).to_csv(ANALYSIS_DIR / f"worst_products_mae_{split}.csv", index=False)
        sub.sort_values("wape_percent", ascending=False).head(10).to_csv(ANALYSIS_DIR / f"worst_products_wape_{split}.csv", index=False)

    store_table = group_metrics[(group_metrics.group_type == "store_id") &
                                (group_metrics.model == "Stage 4")].copy()
    store_table.to_csv(ANALYSIS_DIR / "store_error_breakdown_stage4.csv", index=False)
    demand_table = group_metrics[(group_metrics.group_type == "demand_level")].copy()
    demand_table.to_csv(ANALYSIS_DIR / "demand_level_comparison.csv", index=False)
    frequency_table = group_metrics[(group_metrics.group_type == "sales_frequency")].copy()
    frequency_table.to_csv(ANALYSIS_DIR / "sales_frequency_comparison.csv", index=False)
    category_table = group_metrics[(group_metrics.group_type == "cat_id")].copy()
    category_table.to_csv(ANALYSIS_DIR / "product_category_comparison.csv", index=False)

    # Stage 4 deltas use paired rows and the frozen baseline embedded in the
    # Stage 4-owned prediction files; no Stage 3 source artifact is opened.
    pivot = demand_table.pivot(index=["split", "group_value"], columns="model", values=["mae", "rmse", "wape_percent", "mean_bias"])
    deltas = []
    for (split, level), row in pivot.iterrows():
        deltas.append({"split": split, "demand_level": level,
                       **{f"{metric}_stage3": row[(metric, "Stage 3 baseline")]
                          for metric in ("mae", "rmse", "wape_percent", "mean_bias")},
                       **{f"{metric}_stage4": row[(metric, "Stage 4")]
                          for metric in ("mae", "rmse", "wape_percent", "mean_bias")},
                       "mae_change_stage4_minus_stage3": row[("mae", "Stage 4")] - row[("mae", "Stage 3 baseline")],
                       "wape_change_pp_stage4_minus_stage3": row[("wape_percent", "Stage 4")] - row[("wape_percent", "Stage 3 baseline")],
                       "rmse_change_stage4_minus_stage3": row[("rmse", "Stage 4")] - row[("rmse", "Stage 3 baseline")]})
    delta_frame = pd.DataFrame(deltas)
    delta_frame.to_csv(ANALYSIS_DIR / "stage4_demand_level_deltas.csv", index=False)

    labels = ["Validation / " + r.demand_level for r in delta_frame[delta_frame.split.eq("validation")].itertuples()]
    labels += ["Test / " + r.demand_level for r in delta_frame[delta_frame.split.eq("test")].itertuples()]
    mae_s3 = delta_frame[delta_frame.split.eq("validation")].mae_stage3.tolist() + delta_frame[delta_frame.split.eq("test")].mae_stage3.tolist()
    mae_s4 = delta_frame[delta_frame.split.eq("validation")].mae_stage4.tolist() + delta_frame[delta_frame.split.eq("test")].mae_stage4.tolist()
    _svg_bar_chart(ANALYSIS_DIR / "demand_level_mae.svg", "MAE by demand level: saved prediction comparison",
                   labels, [("Stage 3 baseline", mae_s3, "#8a8f98"), ("Stage 4", mae_s4, "#2878b5")])
    test_store = store_table[store_table.split.eq("test")].sort_values("wape_percent", ascending=False).head(10)
    _svg_bar_chart(ANALYSIS_DIR / "worst_test_stores_wape.svg", "Stage 4: 10 stores with highest test WAPE",
                   test_store.group_value.tolist(), [("WAPE %", test_store.wape_percent.tolist(), "#bd4b4b")])

    def row_for(metric_source, split, model, group_type, group_value):
        row = metric_source[(metric_source.split == split) & (metric_source.model == model) &
                            (metric_source.group_type == group_type) & (metric_source.group_value == group_value)]
        return row.iloc[0].to_dict() if len(row) else None

    findings = {
        "inputs": ["Stage 4-owned validation predictions", "Stage 4-owned final-test predictions",
                   "Stage 4 series metadata", "Stage 4 saved metrics"],
        "stage3_source_artifacts_opened": False,
        "demand_cutoffs": cuts,
        "frequency_definition": {"intermittent": "nonzero actuals on <=20% of days in that split",
                                 "frequent": "nonzero actuals on >=80% of days in that split",
                                 "middle": "between those thresholds"},
        "validation_overall_stage4": row_for(group_metrics, "validation", "Stage 4", "all", "All"),
        "test_overall_stage4": row_for(group_metrics, "test", "Stage 4", "all", "All"),
        "demand_level_deltas": delta_frame.to_dict(orient="records"),
        "worst_test_stores_stage4_wape": test_store[["group_value", "wape_percent", "mae", "mean_bias"]].to_dict(orient="records"),
        "limitations": ["Intermittency and demand level are classified from each evaluated 28-day window's saved actuals.",
                        "Store and category summaries are aggregate error descriptions, not causal explanations.",
                        "Stage 3 comparison values come only from baseline columns already stored in Stage 4 prediction files."],
        "future_experiment_ideas": [
            "For sparse series, test a two-part occurrence and positive-demand model or a TSB/Croston-style baseline.",
            "For high-volume series, test event-aware recent-demand features and robust treatment of demand spikes; monitor RMSE.",
            "Test store/category-level calibration or hierarchical shrinkage for consistently biased groups, using validation only.",
        ],
        "saved_metric_cross_check": {
            "validation_stage4_mae": validation_metrics["stage4_new_features"]["overall"]["MAE"],
            "test_stage4_mae": test_metrics["stage4_selected"]["overall"]["MAE"],
        },
    }
    (ANALYSIS_DIR / "analysis_summary.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    render_markdown_report()
    return {"output_dir": str(ANALYSIS_DIR), "validation_rows": len(val), "test_rows": len(test),
            "files": sorted(p.name for p in ANALYSIS_DIR.iterdir()), "findings": findings}


if __name__ == "__main__":
    print(json.dumps(run_stage4_error_analysis(), indent=2))
