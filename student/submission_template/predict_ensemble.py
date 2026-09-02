"""Ensemble TiDE checkpoints (multi-seed / multi-seq_len) + seasonal_mean blend.

Examples:
  # Average several checkpoints, then blend with seasonal_mean
  python predict_ensemble.py \\
    --input_dir ../../data \\
    --checkpoints checkpoint_seq336.pt,checkpoint_seq504.pt,checkpoint_seed42.pt \\
    --tide-weight 0.8 --seasonal-weight 0.2 \\
    --output_file predictions.csv

  # Or resolve seed files next to a stem checkpoint
  python predict_ensemble.py \\
    --input_dir ../../data \\
    --checkpoint checkpoint.pt \\
    --ensemble-seeds 42,2024,777 \\
    --tide-weight 0.8 --seasonal-weight 0.2 \\
    --output_file predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from predict import (
    build_model_from_checkpoint,
    load_checkpoint_dict,
    predict_with_model,
    print_prediction_diagnostics,
    resolve_device,
)
from src.data_io import load_cov_frame, load_forecast_index, load_train_frame, resolve_data_dir
from src.features import TARGET_COL


def seasonal_mean_forecast(
    train_frame: pd.DataFrame,
    forecast_index: pd.DataFrame,
) -> pd.DataFrame:
    """Per-series same weekday/hour historical mean, with series/global fallbacks."""
    train = train_frame.copy()
    index = forecast_index.copy()
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    index["timestamp"] = pd.to_datetime(index["timestamp"])

    train["_hour"] = train["timestamp"].dt.hour
    train["_dayofweek"] = train["timestamp"].dt.dayofweek
    index["_hour"] = index["timestamp"].dt.hour
    index["_dayofweek"] = index["timestamp"].dt.dayofweek

    seasonal = (
        train.groupby(["series_id", "_dayofweek", "_hour"], as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "prediction"})
    )
    result = index[["series_id", "timestamp", "_dayofweek", "_hour"]].merge(
        seasonal,
        on=["series_id", "_dayofweek", "_hour"],
        how="left",
    )
    series_means = (
        train.groupby("series_id", as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "_series_mean"})
    )
    result = result.merge(series_means, on="series_id", how="left")
    global_mean = float(train[TARGET_COL].mean())
    result["prediction"] = (
        result["prediction"].fillna(result["_series_mean"]).fillna(global_mean)
    )
    return result[["series_id", "timestamp", "prediction"]]


def parse_checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.checkpoints:
        paths.extend(Path(part.strip()) for part in args.checkpoints.split(",") if part.strip())
    if args.ensemble_seeds:
        seeds = [part.strip() for part in args.ensemble_seeds.split(",") if part.strip()]
        stem = args.checkpoint.stem
        suffix = args.checkpoint.suffix or ".pt"
        parent = args.checkpoint.parent
        for seed in seeds:
            paths.append(parent / f"{stem}_seed{seed}{suffix}")
    if not paths:
        if args.checkpoint.exists():
            paths.append(args.checkpoint)
        else:
            raise FileNotFoundError(
                "No checkpoints provided. Use --checkpoints and/or --ensemble-seeds, "
                "or a valid --checkpoint."
            )

    # Deduplicate while preserving order.
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)

    missing = [str(path) for path in unique if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint(s): {missing}")
    return unique


def align_to_forecast_index(
    predictions: pd.DataFrame,
    forecast_index: pd.DataFrame,
) -> pd.DataFrame:
    """Force exact forecast_index row order and schema."""
    fi = forecast_index[["series_id", "timestamp"]].copy()
    pred = predictions.copy()
    fi["timestamp"] = pd.to_datetime(fi["timestamp"])
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    aligned = fi.merge(pred, on=["series_id", "timestamp"], how="left", validate="one_to_one")
    if aligned["prediction"].isna().any():
        n_missing = int(aligned["prediction"].isna().sum())
        raise ValueError(f"Missing {n_missing} predictions after aligning to forecast_index.")
    return aligned


def run_tide_ensemble(
    checkpoint_paths: list[Path],
    forecast_index: pd.DataFrame,
    train_frame: pd.DataFrame,
    cov_frame: pd.DataFrame | None,
    device: torch.device,
) -> np.ndarray:
    stack: list[np.ndarray] = []
    for path in checkpoint_paths:
        checkpoint = load_checkpoint_dict(path, device)
        config = checkpoint.get("config", {})
        print(
            f"[TiDE] {path.name}: seq_len={config.get('seq_len')} "
            f"seed={config.get('seed')} num_covariates={config.get('num_covariates')}",
            flush=True,
        )
        model = build_model_from_checkpoint(checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model.to(device)
        preds = predict_with_model(model, forecast_index, train_frame, cov_frame, device)
        aligned = align_to_forecast_index(preds, forecast_index)
        values = aligned["prediction"].to_numpy(dtype=np.float64)
        print(
            f"  pred stats: mean={values.mean():.4f} min={values.min():.4f} max={values.max():.4f}",
            flush=True,
        )
        stack.append(values)

    averaged = np.mean(np.stack(stack, axis=0), axis=0)
    return averaged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TiDE multi-checkpoint ensemble + seasonal_mean blend."
    )
    parser.add_argument("--input_dir", type=Path, default=Path("../../data"))
    parser.add_argument("--output_file", type=Path, default=Path("predictions.csv"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoint.pt"),
        help="Fallback single checkpoint / stem for --ensemble-seeds.",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma-separated checkpoint paths (different seeds and/or seq_len).",
    )
    parser.add_argument(
        "--ensemble-seeds",
        type=str,
        default=None,
        help="Comma-separated seeds; loads {stem}_seed{seed}.pt next to --checkpoint.",
    )
    parser.add_argument("--tide-weight", type=float, default=0.8)
    parser.add_argument("--seasonal-weight", type=float, default=0.2)
    parser.add_argument(
        "--no-seasonal",
        action="store_true",
        help="Disable seasonal_mean blend (TiDE ensemble only).",
    )
    args = parser.parse_args()

    tide_w = float(args.tide_weight)
    seasonal_w = 0.0 if args.no_seasonal else float(args.seasonal_weight)
    if tide_w < 0 or seasonal_w < 0:
        raise ValueError("Weights must be non-negative.")
    weight_sum = tide_w + seasonal_w
    if weight_sum <= 0:
        raise ValueError("At least one of tide-weight / seasonal-weight must be > 0.")
    tide_w /= weight_sum
    seasonal_w /= weight_sum

    device = resolve_device()
    print(f"--> inference device: {device}")

    checkpoint_paths = parse_checkpoint_paths(args)
    print(f"--> TiDE checkpoints ({len(checkpoint_paths)}):")
    for path in checkpoint_paths:
        print(f"    - {path}")

    data_dir = resolve_data_dir(args.input_dir)
    print(f"using data_dir: {data_dir}")
    forecast_index = load_forecast_index(data_dir)
    train_frame = load_train_frame(data_dir)
    cov_frame = load_cov_frame(data_dir)

    tide_pred = run_tide_ensemble(
        checkpoint_paths, forecast_index, train_frame, cov_frame, device
    )
    tide_pred = np.clip(tide_pred, 0.0, None)

    if seasonal_w > 0:
        print("computing seasonal_mean baseline...", flush=True)
        seasonal = seasonal_mean_forecast(train_frame, forecast_index)
        seasonal_aligned = align_to_forecast_index(seasonal, forecast_index)
        seasonal_pred = seasonal_aligned["prediction"].to_numpy(dtype=np.float64)
        seasonal_pred = np.clip(seasonal_pred, 0.0, None)
        print(
            f"seasonal_mean stats: mean={seasonal_pred.mean():.4f} "
            f"min={seasonal_pred.min():.4f} max={seasonal_pred.max():.4f}",
            flush=True,
        )
        blended = tide_w * tide_pred + seasonal_w * seasonal_pred
    else:
        blended = tide_pred

    blended = np.clip(blended, 0.0, None)

    predictions = forecast_index[["series_id", "timestamp"]].copy()
    predictions["prediction"] = blended
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"]).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    print(
        f"blend weights: tide={tide_w:.3f}, seasonal_mean={seasonal_w:.3f}",
        flush=True,
    )
    print_prediction_diagnostics(predictions, train_frame, forecast_index)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_file, index=False)
    print(f"wrote {len(predictions)} predictions -> {args.output_file}")


if __name__ == "__main__":
    main()
