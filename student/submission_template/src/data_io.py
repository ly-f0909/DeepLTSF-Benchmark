"""Data loading helpers for training and inference."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SUBMISSION_ROOT = Path(__file__).resolve().parents[1]


def resolve_data_dir(explicit: Path | None = None) -> Path:
    """Find a directory that contains train.csv."""
    if explicit is not None:
        data_dir = explicit.expanduser().resolve()
        if (data_dir / "train.csv").exists():
            return data_dir
        if data_dir.name == "train.csv" and data_dir.exists():
            return data_dir.parent
        raise FileNotFoundError(
            f"train.csv not found under {data_dir}. "
            "Pass --data-dir to the folder containing train.csv, "
            "or --train_csv to the train.csv file directly."
        )

    candidates = [
        SUBMISSION_ROOT / "data",
        SUBMISSION_ROOT.parent.parent / "data",
        SUBMISSION_ROOT / "../../data",
        Path.cwd() / "data",
        Path.cwd().parent / "data",
    ]
    for candidate in candidates:
        data_dir = candidate.expanduser().resolve()
        if (data_dir / "train.csv").exists():
            return data_dir

    raise FileNotFoundError(
        "Could not locate train.csv. Download the HF dataset and either:\n"
        "  1) place train.csv under ../../data relative to submission_template, or\n"
        "  2) run: python train.py --data-dir ../../data\n"
        "  3) run: python train.py --train_csv ../../data/train.csv"
    )


def resolve_train_csv(train_csv: Path | None = None, data_dir: Path | None = None) -> Path:
    """Resolve train.csv from an explicit file path or data directory."""
    if train_csv is not None:
        path = train_csv.expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"train.csv not found: {path}")
    return resolve_data_dir(data_dir) / "train.csv"


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
