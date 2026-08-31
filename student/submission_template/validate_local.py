"""Local validation: OA-style inference + backtest metrics on train.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from predict import build_model_from_checkpoint, forecast_series, resolve_device
from src.data_io import (
    get_series_future_cov,
    get_series_history,
    load_cov_frame,
    load_forecast_index,
    load_train_frame,
)
from src.features import TARGET_COL
from src.metrics import compute_metrics, format_metrics


def run_oa_inference(
    data_dir: Path,
    checkpoint_path: Path,
    output_file: Path,
    device: torch.device,
) -> pd.DataFrame:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        checkpoint = {"state_dict": checkpoint, "config": {}}
        state = checkpoint["state_dict"]
    else:
        raise ValueError("Checkpoint must be a state_dict or a dict containing `state_dict`.")

    model = build_model_from_checkpoint(checkpoint)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    forecast_index = load_forecast_index(data_dir)
    train_frame = load_train_frame(data_dir)
    cov_frame = load_cov_frame(data_dir)

    rows = []
    for series_id, index_part in forecast_index.groupby("series_id", sort=False):
        t0 = index_part["timestamp"].min()
        hist = get_series_history(train_frame, series_id, t0)
        if hist.empty:
            raise ValueError(f"No history in train.csv for series {series_id!r} before {t0}.")

        future_cov = get_series_future_cov(cov_frame, train_frame, series_id, index_part)
        yhat = forecast_series(model, hist, index_part, future_cov, device)
        part = index_part[["series_id", "timestamp"]].copy()
        part["prediction"] = yhat
        rows.append(part)

    predictions = pd.concat(rows, ignore_index=True)
    predictions["timestamp"] = predictions["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_file, index=False)
    return predictions


def backtest_last_horizon(
    train_csv: Path,
    checkpoint_path: Path,
    horizon: int,
    device: torch.device,
) -> dict[str, float]:
    frame = pd.read_csv(train_csv)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        checkpoint = {"state_dict": checkpoint, "config": {}}
        state = checkpoint["state_dict"]
    else:
        raise ValueError("Checkpoint must be a state_dict or a dict containing `state_dict`.")

    model = build_model_from_checkpoint(checkpoint)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    y_true_all: list[float] = []
    y_pred_all: list[float] = []

    for series_id, group in frame.groupby("series_id", sort=False):
        sorted_group = group.sort_values("timestamp").reset_index(drop=True)
        if len(sorted_group) <= horizon:
            continue

        history = sorted_group.iloc[:-horizon]
        future_index = sorted_group.iloc[-horizon:][["series_id", "timestamp"]]
        future_cov = sorted_group.iloc[-horizon:]
        y_true = sorted_group.iloc[-horizon:][TARGET_COL].to_numpy(dtype=np.float64)
        y_pred = forecast_series(model, history, future_index, future_cov, device)
        y_true_all.extend(y_true.tolist())
        y_pred_all.extend(y_pred.tolist())

    return compute_metrics(np.asarray(y_true_all), np.asarray(y_pred_all))


def score_with_labels(predictions: pd.DataFrame, labels_csv: Path) -> dict[str, float]:
    labels = pd.read_csv(labels_csv)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    predictions = predictions.copy()
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])

    merged = predictions.merge(labels, on=["series_id", "timestamp"], how="inner")
    if merged.empty:
        raise ValueError("No overlapping rows between predictions and labels.")

    return compute_metrics(merged[TARGET_COL].to_numpy(), merged["prediction"].to_numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description="Local validation aligned with OA metrics.")
    parser.add_argument("--data-dir", type=Path, default=Path("../../data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint.pt"))
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=336)
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    args = parser.parse_args()

    train_csv = args.train_csv or (args.data_dir / "train.csv")
    device = resolve_device()
    print(f"--> validation device: {device}")

    predictions: pd.DataFrame | None = None
    if not args.skip_inference:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
        print("running OA-style inference...")
        predictions = run_oa_inference(args.data_dir, args.checkpoint, args.output_file, device)
        forecast_index = load_forecast_index(args.data_dir)
        print(f"wrote {len(predictions)} predictions -> {args.output_file}")
        if len(predictions) != len(forecast_index):
            print(
                f"warning: prediction rows ({len(predictions)}) != "
                f"forecast_index rows ({len(forecast_index)})"
            )

    if args.labels is not None:
        if predictions is None:
            predictions = pd.read_csv(args.output_file)
        print("scoring against provided labels...")
        metrics = score_with_labels(predictions, args.labels)
        print("Labels scored:", format_metrics(metrics))
    else:
        print("note: validation labels are hidden on HF; upload predictions.csv to OA for true score.")

    if not args.skip_backtest:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
        print(f"running per-series backtest on last {args.horizon} steps of train.csv...")
        metrics = backtest_last_horizon(train_csv, args.checkpoint, args.horizon, device)
        print("Backtest scored:", format_metrics(metrics))


if __name__ == "__main__":
    main()
