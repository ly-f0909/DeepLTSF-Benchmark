"""Data loading helpers for training and inference."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_train_frame(input_dir: Path) -> pd.DataFrame:
    """Load train.csv which contains historical targets and covariates."""
    path = input_dir / "train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing train.csv in {input_dir}.")
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def load_cov_frame(input_dir: Path) -> pd.DataFrame | None:
    """Load validation/test input with known future covariates (no target column)."""
    for name in ("validation_input.csv", "test_input.csv"):
        path = input_dir / name
        if path.exists():
            frame = pd.read_csv(path)
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
            return frame
    return None


def load_forecast_index(input_dir: Path) -> pd.DataFrame:
    for name in ("forecast_index_test.csv", "forecast_index_validation.csv"):
        path = input_dir / name
        if path.exists():
            frame = pd.read_csv(path)
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
            return frame
    raise FileNotFoundError(
        "Expected forecast_index_test.csv or forecast_index_validation.csv in input_dir."
    )


def get_series_history(train_frame: pd.DataFrame, series_id: str, t0: pd.Timestamp) -> pd.DataFrame:
    """Historical rows with true targets strictly before forecast start."""
    hist = train_frame.loc[
        train_frame["series_id"].eq(series_id) & train_frame["timestamp"].lt(t0)
    ]
    return hist.sort_values("timestamp")


def get_series_future_cov(
    cov_frame: pd.DataFrame | None,
    train_frame: pd.DataFrame,
    series_id: str,
    future_index: pd.DataFrame,
) -> pd.DataFrame:
    """
    Known future covariates for one series.

    Prefer validation/test input; fall back to train.csv for local backtests.
    """
    if cov_frame is not None:
        merged = cov_frame.merge(future_index, on=["series_id", "timestamp"], how="inner")
        part = merged.loc[merged["series_id"].eq(series_id)].sort_values("timestamp")
        if not part.empty:
            return part

    return (
        train_frame.merge(future_index, on=["series_id", "timestamp"], how="inner")
        .loc[lambda df: df["series_id"].eq(series_id)]
        .sort_values("timestamp")
    )
