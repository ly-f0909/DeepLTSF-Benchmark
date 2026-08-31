"""Leaderboard-aligned forecasting metrics."""

from __future__ import annotations

import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> dict[str, float]:
    """Compute MAE, MSE, RMSE, MAPE, SMAPE, and WAPE (lower is better)."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    err = y_pred - y_true
    abs_err = np.abs(err)

    mae = float(np.mean(abs_err))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))

    mape_denom = np.maximum(np.abs(y_true), eps)
    mape = float(np.mean(abs_err / mape_denom) * 100.0)

    smape_denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    smape = float(np.mean(2.0 * abs_err / smape_denom) * 100.0)

    wape_denom = np.maximum(np.sum(np.abs(y_true)), eps)
    wape = float(np.sum(abs_err) / wape_denom * 100.0)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "SMAPE": smape,
        "WAPE": wape,
    }


def format_metrics(metrics: dict[str, float]) -> str:
    return ", ".join(f"{name} = {value:.6f}" for name, value in metrics.items())
