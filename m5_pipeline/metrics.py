"""Small, dependency-light regression metrics for M5 model evaluation."""
from __future__ import annotations

import numpy as np


def regression_metrics(actual, predicted) -> dict[str, float]:
    y = np.asarray(actual, dtype=np.float64).reshape(-1)
    p = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("actual and predicted must have the same nonzero length")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("actual and predicted must contain only finite values")
    error = p - y
    mae = float(np.abs(error).mean())
    rmse = float(np.sqrt(np.square(error).mean()))
    denominator = float(np.abs(y).sum())
    wape = float(np.abs(error).sum() / denominator * 100) if denominator else float("nan")
    return {"MAE": mae, "RMSE": rmse, "WAPE_percent": wape}
