"""Unit tests for Stage 3.5 validation error summaries."""
import numpy as np
import pandas as pd

from m5_pipeline import analysis
from m5_pipeline.analysis import _bias_extremes, _group_metrics


def test_group_summary_reports_weighted_mae_wape_and_bias_direction():
    series = pd.DataFrame({
        "series_idx": [0, 1, 2],
        "store_id": ["A", "A", "B"],
        "row_count": [2, 2, 1],
        "actual_sum": [10.0, 20.0, 0.0],
        "absolute_error_sum": [4.0, 2.0, 3.0],
        "signed_error_sum": [2.0, -2.0, -3.0],
    })
    rows = _group_metrics(series, "store_id")
    by_store = {row["store_id"]: row for row in rows}

    assert by_store["A"]["MAE"] == 1.5
    assert by_store["A"]["WAPE_percent"] == 20.0
    assert by_store["A"]["bias_direction"] == "balanced"
    assert by_store["B"]["bias_direction"] == "underpredict"
    assert np.isnan(by_store["B"]["WAPE_percent"])


def test_bias_extremes_does_not_label_underprediction_as_overprediction():
    groups = pd.DataFrame({"mean_bias": [-0.1, -0.5], "group": ["A", "B"]})
    over, under = _bias_extremes(groups)
    assert over.empty
    assert under.group.tolist() == ["B", "A"]


def test_permutation_importance_is_reproducible(monkeypatch):
    x = np.array([[0, 1], [1, 0], [2, 1], [3, 0]], dtype=np.float32)
    y = np.array([0, 1, 2, 3], dtype=np.float32)
    side = {"target_day": np.array([1886, 1887, 1888, 1889])}
    monkeypatch.setattr(analysis, "_load_split", lambda *args: (x, y, side, {0, 1}))

    class SimplePredictor:
        def predict(self, values):
            return values[:, 0]

    first = analysis._permutation_importance(SimplePredictor(), ["signal", "noise"], 4, 42)
    second = analysis._permutation_importance(SimplePredictor(), ["signal", "noise"], 4, 42)
    assert first == second
    assert first[1]["target_day_min"] == 1886
    assert first[1]["target_day_max"] == 1889
