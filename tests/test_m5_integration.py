"""Small schema and cutoff checks for M5 RetailIQ inference features."""
import numpy as np
import pandas as pd

from m5_pipeline.retailiq_integration import _forecast_feature_columns


def test_next_day_features_use_cutoff_history_and_price_week():
    target_position = 84
    sales = np.zeros((1, 100), dtype=np.int16)
    sales[0, [56, 70, 77, 83]] = [4, 7, 10, 13]
    rows = pd.DataFrame({
        "item_id_idx": [1], "store_id_idx": [2], "state_id_idx": [0],
        "dept_id_idx": [3], "cat_id_idx": [4], "state_id": ["CA"],
    })
    calendar = {
        "dow": np.arange(100, dtype=np.uint8) % 7,
        "wday": np.ones(100, dtype=np.uint8), "month": np.ones(100, dtype=np.uint8) * 3,
        "year": np.ones(100, dtype=np.uint16) * 2016,
        "event_name_1": np.zeros(100, dtype=np.uint16), "event_type_1": np.zeros(100, dtype=np.uint16),
        "event_name_2": np.zeros(100, dtype=np.uint16), "event_type_2": np.zeros(100, dtype=np.uint16),
        "snap_CA": np.ones(100, dtype=np.uint8), "snap_TX": np.zeros(100, dtype=np.uint8),
        "snap_WI": np.zeros(100, dtype=np.uint8), "day_week_index": np.ones(100, dtype=np.uint16),
    }
    prices = np.array([[10.0, 12.0]], dtype=np.float32)
    features = _forecast_feature_columns(sales, rows, calendar, prices, target_position)
    assert features["target_day"][0] == 85
    assert [features[f"lag_{lag}"][0] for lag in (1, 7, 14, 28)] == [13, 10, 7, 4]
    assert features["price_at_cutoff"][0] == 12
    assert features["price_prev_week"][0] == 10
    assert features["price_change_pct"][0] == np.float32(0.2)
    assert features["nonzero_rate_84"][0] == 4 / 84

    changed = sales.copy()
    changed[:, target_position:] = 99
    changed_features = _forecast_feature_columns(changed, rows, calendar, prices, target_position)
    for name in features:
        np.testing.assert_equal(features[name], changed_features[name])
