"""Small unit tests for Stage 5's cutoff-safe router and TSB core."""
import numpy as np
import pytest

from m5_pipeline import stage5_config as S5
from m5_pipeline.stage5_hybrid import compute_tsb_forecasts, route_to_tsb


def test_router_requires_full_history_and_uses_inclusive_threshold():
    rates = np.array([0.20, 0.20001, np.nan, 0.0], dtype=np.float32)
    history = np.array([84, 84, 100, 83])
    assert route_to_tsb(rates, history).tolist() == [True, False, False, False]


def test_tsb_initialization_and_cutoff_excludes_target_and_future_sales():
    sales = np.zeros((1, 100), dtype=np.int16)
    sales[0, [0, 10]] = [2, 4]
    target = np.array([28])
    forecast, rate, routed = compute_tsb_forecasts(sales, target)
    assert forecast[0] == pytest.approx((2 / 28) * 3)
    assert np.isnan(rate[0])  # fewer than 84 days available for routing
    assert routed.tolist() == [False]

    changed = sales.copy()
    changed[:, 28:] = 7  # includes target day and all later days
    future_forecast, future_rate, future_route = compute_tsb_forecasts(changed, target)
    np.testing.assert_equal(forecast, future_forecast)
    np.testing.assert_equal(rate, future_rate)
    np.testing.assert_equal(routed, future_route)


def test_route_and_tsb_rate_use_only_prior_84_days():
    sales = np.zeros((2, 100), dtype=np.int16)
    sales[0, 0] = 1
    sales[1, 0:20] = 1
    target = np.array([84])
    _, rates, routed = compute_tsb_forecasts(sales, target)
    np.testing.assert_allclose(rates, [1 / 84, 20 / 84], rtol=0, atol=1e-7)
    assert routed.tolist() == [True, False]

    changed = sales.copy()
    changed[:, 84:] = 9
    _, changed_rates, changed_route = compute_tsb_forecasts(changed, target)
    np.testing.assert_equal(rates, changed_rates)
    np.testing.assert_equal(routed, changed_route)


def test_final_test_evaluator_is_locked_before_reading_test_data(tmp_path, monkeypatch):
    monkeypatch.setattr(S5, "SELECTION_PATH", tmp_path / "selection.json")
    with pytest.raises(RuntimeError, match="Final test is locked"):
        from m5_pipeline.stage5_hybrid import evaluate_stage5_final_test
        evaluate_stage5_final_test()
