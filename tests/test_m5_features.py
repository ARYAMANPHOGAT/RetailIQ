"""Tests for M5 chronological feature generation and leakage boundaries."""
import numpy as np
import pandas as pd

from m5_pipeline.features import compute_sales_features, temporal_split_masks


def test_lag_and_rolling_features_use_only_history_before_target():
    sales = np.arange(1, 41, dtype=np.int16).reshape(1, -1)
    target_positions = np.array([28, 29, 30])
    features = compute_sales_features(sales, target_positions)

    assert features["lag_1"].tolist() == [28, 29, 30]
    assert features["lag_7"].tolist() == [22, 23, 24]
    assert features["lag_14"].tolist() == [15, 16, 17]
    assert features["lag_28"].tolist() == [1, 2, 3]
    assert features["roll_mean_7"][0] == np.mean(np.arange(22, 29))
    assert np.isclose(features["roll_std_7"][0], np.std(np.arange(22, 29), ddof=1))
    assert features["roll_mean_28"][0] == np.mean(np.arange(1, 29))


def test_changing_target_day_sales_does_not_change_its_features():
    sales = np.arange(1, 41, dtype=np.int16).reshape(1, -1)
    changed = sales.copy()
    target_position = 30
    changed[0, target_position] = 999

    original_features = compute_sales_features(sales, np.array([target_position]))
    changed_features = compute_sales_features(changed, np.array([target_position]))
    for name in original_features:
        if name == "target":
            assert original_features[name][0] != changed_features[name][0]
        else:
            np.testing.assert_array_equal(original_features[name], changed_features[name])


def test_temporal_splits_are_disjoint_and_reserve_requested_test_days():
    days = np.arange(29, 1942, dtype=np.uint16)
    masks = temporal_split_masks(days)
    split_values = {name: set(days[mask].tolist()) for name, mask in masks.items()}

    assert split_values["train"] == set(range(29, 1886))
    assert split_values["validation"] == set(range(1886, 1914))
    assert split_values["test"] == set(range(1914, 1942))
    assert not (split_values["train"] & split_values["validation"])
    assert not (split_values["train"] & split_values["test"])
    assert not (split_values["validation"] & split_values["test"])


def test_price_feature_construction_uses_cutoff_week_not_target_week():
    from m5_pipeline.features import _make_feature_block

    n_days = 35
    calendar = {
        "day_week_index": np.arange(n_days, dtype=np.uint16),
        "dow": np.zeros(n_days, dtype=np.uint8),
        "wday": np.ones(n_days, dtype=np.uint8),
        "month": np.ones(n_days, dtype=np.uint8),
        "year": np.full(n_days, 2011, dtype=np.uint16),
        "event_name_1": np.zeros(n_days, dtype=np.uint16),
        "event_type_1": np.zeros(n_days, dtype=np.uint16),
        "event_name_2": np.zeros(n_days, dtype=np.uint16),
        "event_type_2": np.zeros(n_days, dtype=np.uint16),
        "snap_CA": np.zeros(n_days, dtype=np.uint8),
        "snap_TX": np.zeros(n_days, dtype=np.uint8),
        "snap_WI": np.zeros(n_days, dtype=np.uint8),
    }
    series = pd.DataFrame({"series_idx": [0], "item_id_idx": [0], "store_id_idx": [0],
                           "state_id_idx": [0], "dept_id_idx": [0], "cat_id_idx": [0], "state_id": ["CA"]})
    sales = np.arange(n_days, dtype=np.int16).reshape(1, -1)
    prices = np.arange(n_days, dtype=np.float32).reshape(1, -1) + 1
    baseline = _make_feature_block(sales, series, calendar, prices, 0)
    # Feature row index for target d_30 (0-based within d_29 onward) uses cutoff d_29/week 28.
    row = 1
    altered = prices.copy()
    altered[0, 29] = 5000  # price effective on target d_30, unavailable at cutoff d_29
    changed = _make_feature_block(sales, series, calendar, altered, 0)
    assert baseline["price_at_cutoff"][row] == changed["price_at_cutoff"][row]
    assert baseline["price_prev_week"][row] == changed["price_prev_week"][row]
    assert baseline["price_missing"][row] == changed["price_missing"][row] == 0


def test_missing_historical_price_is_nan_and_has_indicator():
    from m5_pipeline.features import _make_feature_block

    n_days = 35
    calendar = {
        "day_week_index": np.arange(n_days, dtype=np.uint16),
        "dow": np.zeros(n_days, dtype=np.uint8), "wday": np.ones(n_days, dtype=np.uint8),
        "month": np.ones(n_days, dtype=np.uint8), "year": np.full(n_days, 2011, dtype=np.uint16),
        "event_name_1": np.zeros(n_days, dtype=np.uint16), "event_type_1": np.zeros(n_days, dtype=np.uint16),
        "event_name_2": np.zeros(n_days, dtype=np.uint16), "event_type_2": np.zeros(n_days, dtype=np.uint16),
        "snap_CA": np.zeros(n_days, dtype=np.uint8), "snap_TX": np.zeros(n_days, dtype=np.uint8),
        "snap_WI": np.zeros(n_days, dtype=np.uint8),
    }
    series = pd.DataFrame({"series_idx": [0], "item_id_idx": [0], "store_id_idx": [0],
                           "state_id_idx": [0], "dept_id_idx": [0], "cat_id_idx": [0], "state_id": ["CA"]})
    prices = np.full((1, n_days), np.nan, dtype=np.float32)
    features = _make_feature_block(np.ones((1, n_days), dtype=np.int16), series, calendar, prices, 0)
    assert np.isnan(features["price_at_cutoff"]).all()
    assert np.isnan(features["price_prev_week"]).all()
    assert features["price_missing"].tolist() == [1] * len(features["target"])
