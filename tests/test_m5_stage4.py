"""Focused tests for Stage 4 cutoff-safe intermittent-demand features."""
import numpy as np
import pytest

from m5_pipeline import stage4_config as S4
from m5_pipeline.features import temporal_split_masks
from m5_pipeline.stage4_features import compute_intermittent_features, require_disk_space
from m5_pipeline.stage4_training import _metric_bundle
from m5_pipeline.training import sample_row_positions


def test_window_counts_rates_positive_means_and_long_rolls():
    sales = np.zeros((1, 100), dtype=np.int16)
    sales[0, [0, 10, 20, 40, 60, 80]] = [2, 4, 6, 8, 10, 12]
    features = compute_intermittent_features(sales, np.array([84]))

    assert features["nonzero_count_7"][0] == 1
    assert features["nonzero_rate_7"][0] == pytest.approx(1 / 7)
    assert features["nonzero_count_28"][0] == 2
    assert features["nonzero_count_56"][0] == 3
    assert features["nonzero_count_84"][0] == 6
    assert features["positive_mean_28"][0] == 11
    assert features["positive_mean_84"][0] == 7
    assert features["days_since_last_sale"][0] == 3
    assert features["no_prior_sale"][0] == 0
    assert features["roll_mean_84"][0] == pytest.approx(0.5)
    assert features["roll_std_84"][0] > 0
    assert features["demand_cv_84"][0] == pytest.approx(features["roll_std_84"][0] / 0.5)


def test_target_and_future_sales_are_not_inputs_to_features():
    sales = np.arange(1, 101, dtype=np.int16).reshape(1, -1)
    changed = sales.copy()
    target_position = 85
    changed[0, target_position:] = 0
    before = compute_intermittent_features(sales, np.array([target_position]))
    after = compute_intermittent_features(changed, np.array([target_position]))
    for feature in before:
        np.testing.assert_equal(before[feature], after[feature])


def test_incomplete_windows_and_no_prior_sales_remain_missing():
    sales = np.zeros((1, 40), dtype=np.int16)
    features = compute_intermittent_features(sales, np.array([28]))
    assert np.isnan(features["nonzero_count_56"][0])
    assert np.isnan(features["nonzero_rate_84"][0])
    assert np.isnan(features["positive_mean_28"][0])
    assert np.isnan(features["days_since_last_sale"][0])
    assert features["no_prior_sale"][0] == 1
    assert np.isnan(features["roll_mean_84"][0])
    assert np.isnan(features["demand_cv_56"][0])


def test_stage4_splits_exclude_final_test_rows_and_keep_same_training_sample():
    days = np.arange(29, 1914, dtype=np.uint16)
    masks = temporal_split_masks(days)
    assert days[masks["train"]].min() == 29
    assert days[masks["train"]].max() == S4.TRAIN_END
    assert days[masks["validation"]].tolist() == list(range(S4.VALIDATION_START, S4.VALIDATION_END + 1))
    assert not masks["test"].any()
    stage3_positions = sample_row_positions(56_619_930, 2_000_000, 42)
    stage4_positions = sample_row_positions(56_619_930, S4.SAMPLE_ROWS, S4.RANDOM_SEED)
    np.testing.assert_array_equal(stage3_positions, stage4_positions)


def test_disk_preflight_stops_before_generation_when_space_is_short():
    with pytest.raises(OSError, match="Insufficient disk space"):
        require_disk_space(1000, 999)
    require_disk_space(999, 1000)


def test_stage3_and_stage4_tier_metric_helper_uses_inclusive_shared_cutpoints():
    actual = np.array([0, 1, 2, 3, 5, 7], dtype=np.float32)
    prediction = np.array([0, 0, 2, 4, 7, 8], dtype=np.float32)
    series = np.array([0, 0, 1, 1, 2, 2], dtype=np.uint32)
    result = _metric_bundle(actual, prediction, series, low_max=0.5, medium_max=2.5)
    assert result["demand_levels"]["low"]["series_count"] == 1
    assert result["demand_levels"]["medium"]["series_count"] == 1
    assert result["demand_levels"]["high"]["series_count"] == 1
    assert result["demand_levels"]["low"]["WAPE_percent"] == 100.0
    assert result["demand_levels"]["high"]["mean_bias"] == 1.5
