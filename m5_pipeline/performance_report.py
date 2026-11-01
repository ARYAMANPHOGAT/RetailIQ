"""Read-only helpers for the saved M5 model performance artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def comparison_table(root: Path) -> pd.DataFrame:
    """Return only persisted comparable split metrics; absent results stay absent."""
    base = root / "outputs" / "m5"
    m5 = _read_json(base / "m5_metrics.json")
    stage4_val = _read_json(base / "stage4" / "results" / "stage4_metrics.json")
    stage4_test = _read_json(base / "stage4" / "results" / "stage4_test_metrics.json")
    stage41 = _read_json(base / "stage4_1" / "results" / "stage4_1_metrics.json")
    stage5_val = _read_json(base / "stage5" / "results" / "stage5_validation_metrics.json")
    stage5_test = _read_json(base / "stage5" / "results" / "stage5_test_metrics.json")
    rows = []

    def add(model, split, metric):
        rows.append({"Model": model, "Split": split, "MAE": metric.get("MAE"),
                     "RMSE": metric.get("RMSE"), "WAPE (%)": metric.get("WAPE_percent")})

    for split, key in (("Validation", "validation"), ("Final test", "final_test")):
        baseline = m5.get(key, {})
        for label, metric in baseline.items():
            add("Seasonal Naive" if "Naive" in label else "Stage 3 LightGBM", split, metric)
    for split, report, key in (("Validation", stage4_val, "stage4_new_features"),
                               ("Final test", stage4_test, "stage4_selected")):
        metric = report.get(key, {}).get("overall")
        if metric:
            add("Stage 4 LightGBM", split, metric)
    for label, report, key in (("Validation", stage4_val, "stage4_1_same_weekday"),):
        metric = stage41.get(key, {}).get("overall")
        if metric:
            add("Stage 4.1 LightGBM", label, metric)
    for split, report in (("Validation", stage5_val), ("Final test", stage5_test)):
        metric = report.get("stage5_hybrid", {}).get("overall")
        if metric:
            add("Stage 5 Hybrid", split, metric)
    return pd.DataFrame(rows)


def tier_table(path: Path, model_key: str = "stage4_new_features") -> pd.DataFrame:
    report = _read_json(path)
    tiers = report.get(model_key, {}).get("demand_levels", {})
    return pd.DataFrame([{"Demand tier": tier.title(), **values} for tier, values in tiers.items()])


def tier_comparison(root: Path, split: str) -> pd.DataFrame:
    base = root / "outputs" / "m5"
    if split == "Validation":
        reports = [
            ("Stage 3 LightGBM", _read_json(base / "stage4/results/stage4_metrics.json"), "stage3_existing"),
            ("Stage 4 LightGBM", _read_json(base / "stage4/results/stage4_metrics.json"), "stage4_new_features"),
            ("Stage 4.1 LightGBM", _read_json(base / "stage4_1/results/stage4_1_metrics.json"), "stage4_1_same_weekday"),
            ("Stage 5 Hybrid", _read_json(base / "stage5/results/stage5_validation_metrics.json"), "stage5_hybrid"),
        ]
    else:
        reports = [
            ("Stage 3 LightGBM", _read_json(base / "stage4/results/stage4_test_metrics.json"), "stage3_existing"),
            ("Stage 4 LightGBM", _read_json(base / "stage4/results/stage4_test_metrics.json"), "stage4_selected"),
            ("Stage 5 Hybrid", _read_json(base / "stage5/results/stage5_test_metrics.json"), "stage5_hybrid"),
        ]
    rows = []
    for name, report, key in reports:
        for tier, values in report.get(key, {}).get("demand_levels", {}).items():
            rows.append({"Model": name, "Demand tier": tier.title(), "MAE": values.get("MAE"),
                         "RMSE": values.get("RMSE"), "WAPE (%)": values.get("WAPE_percent")})
    return pd.DataFrame(rows)


def prediction_frame(path: Path, split: str, max_rows: int = 20_000) -> pd.DataFrame:
    """Load saved Stage 4 predictions, retaining a deterministic sample for row charts."""
    frame = pd.read_csv(path, usecols=["series_idx", "target_day", "actual", "stage4_lightgbm"])
    frame.rename(columns={"stage4_lightgbm": "predicted"}, inplace=True)
    # All rows are used for daily totals. The deterministic row sample bounds plot payload size.
    if len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=42).sort_values(["series_idx", "target_day"])
    frame["residual"] = frame["predicted"] - frame["actual"]
    frame.attrs["split"] = split
    return frame


def series_extremes(path: Path, split: str, count: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["series_idx", "mae", "wape_percent", "actual_units", "mean_bias", "id", "store_id", "cat_id", "split", "model"]
    frame = pd.read_csv(path, usecols=columns)
    frame = frame[(frame["split"].eq(split)) & frame["model"].str.contains("Stage 4", case=False, na=False)]
    frame = frame.drop_duplicates("series_idx")
    shown = ["id", "store_id", "cat_id", "actual_units", "mae", "wape_percent", "mean_bias"]
    return frame.nsmallest(count, "mae")[shown], frame.nlargest(count, "mae")[shown]
