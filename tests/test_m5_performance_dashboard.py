from pathlib import Path

from m5_pipeline.performance_report import comparison_table, tier_comparison


def test_saved_model_comparison_keeps_only_available_splits():
    root = Path(__file__).resolve().parents[1]
    report = comparison_table(root)
    assert {"Seasonal Naive", "Stage 3 LightGBM", "Stage 4 LightGBM", "Stage 4.1 LightGBM", "Stage 5 Hybrid"} <= set(report.Model)
    assert not ((report.Model == "Stage 4.1 LightGBM") & (report.Split == "Final test")).any()
    assert report.loc[report.Model == "Stage 4 LightGBM", "MAE"].notna().all()
    assert "Training" not in set(report.Split)


def test_demand_tier_comparison_has_shared_models_and_metric_columns():
    root = Path(__file__).resolve().parents[1]
    for split in ("Validation", "Final test"):
        report = tier_comparison(root, split)
        assert {"MAE", "RMSE", "WAPE (%)", "Demand tier", "Model"} <= set(report.columns)
        assert set(report["Demand tier"]) == {"Low", "Medium", "High"}
        assert "Stage 4 LightGBM" in set(report.Model)
    test = tier_comparison(root, "Final test")
    assert "Stage 4.1 LightGBM" not in set(test.Model)
