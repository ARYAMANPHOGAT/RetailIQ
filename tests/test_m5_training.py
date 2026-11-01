"""Unit tests for Stage 3 sampling and metric helpers."""
import numpy as np
import pytest

from m5_pipeline.metrics import regression_metrics
from m5_pipeline.training import feature_matrix, sample_row_positions


def test_row_sampling_is_reproducible_without_replacement():
    first = sample_row_positions(1000, 100, 42)
    second = sample_row_positions(1000, 100, 42)
    assert len(first) == 100
    assert len(np.unique(first)) == 100
    assert np.all(np.diff(first) > 0)
    np.testing.assert_array_equal(first, second)


def test_row_sampling_uses_all_rows_under_the_cap():
    np.testing.assert_array_equal(sample_row_positions(5, 20, 42), np.arange(5))


def test_feature_matrix_is_float32_and_preserves_native_missing_values():
    matrix = feature_matrix({"x": np.array([1, 2], dtype=np.int16),
                             "price": np.array([np.nan, 3], dtype=np.float32)}, ["x", "price"])
    assert matrix.dtype == np.float32
    assert np.isnan(matrix[0, 1])
    assert matrix[1].tolist() == [2.0, 3.0]


def test_regression_metrics_match_hand_calculation():
    metrics = regression_metrics([1, 2, 3], [2, 2, 5])
    assert metrics["MAE"] == pytest.approx(1.0)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(5 / 3))
    assert metrics["WAPE_percent"] == pytest.approx(100 * 3 / 6)


def test_metrics_reject_empty_or_nonfinite_inputs():
    with pytest.raises(ValueError):
        regression_metrics([], [])
    with pytest.raises(ValueError):
        regression_metrics([1], [np.nan])
