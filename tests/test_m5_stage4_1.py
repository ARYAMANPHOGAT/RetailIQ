"""Focused tests for Stage 4.1 summaries derived from prior same-weekday lags."""
import numpy as np
import pytest

from m5_pipeline.stage4_1_experiment import same_weekday_features
from m5_pipeline.stage4_features import compute_intermittent_features


def test_same_weekday_features_summarize_only_positive_lags():
    result = same_weekday_features(
        np.array([0, 3, 4]), np.array([0, 0, 8]), np.array([0, 9, 0]))
    np.testing.assert_allclose(result["same_weekday_nonzero_rate_3_lags"], [0, 2/3, 2/3])
    assert np.isnan(result["same_weekday_positive_mean_3_lags"][0])
    np.testing.assert_allclose(result["same_weekday_positive_mean_3_lags"][1:], [6, 6])


def test_same_weekday_features_reject_invalid_sales_values():
    with pytest.raises(ValueError, match="finite, nonnegative"):
        same_weekday_features([1, np.nan], [2, 3], [4, 5])
    with pytest.raises(ValueError, match="finite, nonnegative"):
        same_weekday_features([1, -1], [2, 3], [4, 5])


def test_stage4_intermittent_features_are_strictly_cutoff_safe():
    sales = np.arange(1, 110, dtype=np.int16).reshape(1, -1)
    target = np.array([84])
    before = compute_intermittent_features(sales, target)
    changed = sales.copy()
    changed[:, 84:] = 0
    after = compute_intermittent_features(changed, target)
    for name in before:
        np.testing.assert_equal(before[name], after[name])
