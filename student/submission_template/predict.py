"""Inference entrypoint for TiDE private evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data_io import (
    get_series_future_cov,
    get_series_history,
    load_cov_frame,
    load_forecast_index,
    load_train_frame,
    resolve_data_dir,
)
from src.features import COVARIATE_COLS, FEATURE_COLS, TARGET_COL, TARGET_IDX
from src.model import ForecastModel


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model_from_checkpoint(checkpoint: dict) -> ForecastModel:
    config = checkpoint.get("config", {})
    return ForecastModel(
        seq_len=config.get("seq_len", 336),
        pred_len=config.get("pred_len", 24),
        num_features=config.get("num_features", len(FEATURE_COLS)),
        num_covariates=config.get("num_covariates", len(COVARIATE_COLS)),
        target_idx=config.get("target_idx", TARGET_IDX),
        hidden_dim=config.get("hidden_dim", 256),
        num_encoder_layers=config.get("num_encoder_layers", 2),
        num_decoder_layers=config.get("num_decoder_layers", 2),
        dropout=config.get("dropout", 0.2),
        use_revin=config.get("use_revin", True),
    )


def extract_future_covariates(
    future_cov_frame: pd.DataFrame,
    future_index: pd.DataFrame,
    start: int,
    length: int,
) -> np.ndarray:
    """Slice future covariates by timestamp alignment (not positional guess)."""
    block_index = future_index.iloc[start : start + length]
    merged = block_index.merge(
        future_cov_frame,
        on=["series_id", "timestamp"],
        how="left",
    )
    values = (
        merged.reindex(columns=COVARIATE_COLS)
        .astype(np.float32)
        .fillna(0.0)
        .to_numpy()
    )
    if values.shape[0] < length:
        pad = np.zeros((length - values.shape[0], len(COVARIATE_COLS)), dtype=np.float32)
        values = np.concatenate([values, pad], axis=0)
    return values


def build_past_window(values: np.ndarray, seq_len: int) -> np.ndarray:
    if len(values) < seq_len:
        pad = np.repeat(values[:1], seq_len - len(values), axis=0)
        return np.concatenate([pad, values], axis=0)
    return values[-seq_len:]


def print_prediction_diagnostics(
    predictions: pd.DataFrame,
    train_frame: pd.DataFrame,
    forecast_index: pd.DataFrame,
) -> None:
    pred = predictions["prediction"].to_numpy(dtype=np.float64)
    target = train_frame[TARGET_COL].to_numpy(dtype=np.float64)

    print("=== prediction diagnostics ===")
    print(
        f"train target : mean={target.mean():.4f}  min={target.min():.4f}  "
        f"max={target.max():.4f}  std={target.std():.4f}"
    )
    print(
        f"predictions  : mean={pred.mean():.4f}  min={pred.min():.4f}  "
        f"max={pred.max():.4f}  std={pred.std():.4f}"
    )
    print(f"negative predictions: {(pred < 0).sum()} / {len(pred)}")
    print(f"nan predictions: {np.isnan(pred).sum()}")

    fi = forecast_index.copy()
    fi["timestamp"] = pd.to_datetime(fi["timestamp"])
    pred_check = predictions.copy()
    pred_check["timestamp"] = pd.to_datetime(pred_check["timestamp"])

    aligned = fi.merge(
        pred_check,
        on=["series_id", "timestamp"],
        how="left",
        validate="one_to_one",
    )
    if len(aligned) != len(fi):
        raise ValueError("Prediction row count does not match forecast_index.")
    if aligned["prediction"].isna().any():
        raise ValueError("Missing predictions after aligning to forecast_index.")

    same_order = fi[["series_id", "timestamp"]].reset_index(drop=True).equals(
        pred_check[["series_id", "timestamp"]].reset_index(drop=True)
    )
    print(f"forecast_index order preserved: {same_order}")
    if not same_order:
        raise ValueError("predictions.csv row order does not match forecast_index.")

    ratio = pred.mean() / max(target.mean(), 1e-6)
    if ratio < 0.3 or ratio > 3.0:
        print(f"WARNING: prediction mean / train mean ratio = {ratio:.3f} (expected ~0.5-2.0)")


@torch.no_grad()
def forecast_series(
    model: ForecastModel,
    history: pd.DataFrame,
    future_index: pd.DataFrame,
    future_cov_frame: pd.DataFrame,
    device: torch.device,
) -> np.ndarray:
    """Roll out 24-step blocks; RevIN stats recomputed each step (same as training)."""
    seq_len, pred_len = model.seq_len, model.pred_len

    values = (
        history.reindex(columns=FEATURE_COLS)
        .astype(np.float32)
        .fillna(0.0)
        .to_numpy()
    )
    if values.shape[0] == 0:
        raise ValueError("History is empty; cannot forecast.")

    preds: list[float] = []
    horizon = len(future_index)
    steps_done = 0

    while steps_done < horizon:
        take = min(pred_len, horizon - steps_done)
        x_past = build_past_window(values, seq_len)
        x_future = extract_future_covariates(
            future_cov_frame, future_index, steps_done, pred_len
        )

        pred_block = model(
            torch.from_numpy(x_past).unsqueeze(0).to(device),
            torch.from_numpy(x_future).unsqueeze(0).to(device),
        )[0].detach().cpu().numpy()[:take]

        # Operational load index should be non-negative.
        pred_block = np.clip(pred_block, 0.0, None)
        preds.extend(pred_block.tolist())

        future_cov_known = x_future[:take]
        appended = np.zeros((take, len(FEATURE_COLS)), dtype=np.float32)
        appended[:, : len(COVARIATE_COLS)] = future_cov_known
        appended[:, TARGET_IDX] = pred_block
        values = np.concatenate([values, appended], axis=0)
        steps_done += take

    return np.asarray(preds[:horizon], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate private test predictions.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Directory with train.csv and validation_input.csv (auto-detected if omitted).",
    )
    parser.add_argument("--output_file", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    device = resolve_device()
    print(f"--> inference device: {device}")

    data_dir = resolve_data_dir(args.input_dir)
    print(f"using data_dir: {data_dir}")
    forecast_index = load_forecast_index(data_dir)
    train_frame = load_train_frame(data_dir)
    cov_frame = load_cov_frame(data_dir)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        config = checkpoint.get("config", {})
    elif isinstance(checkpoint, dict):
        state = checkpoint
        config = {}
    else:
        raise ValueError("Checkpoint must be a state_dict or a dict containing `state_dict`.")

    print(f"checkpoint config: seq_len={config.get('seq_len')} pred_len={config.get('pred_len')} "
          f"use_revin={config.get('use_revin', True)}")

    model = build_model_from_checkpoint(checkpoint if isinstance(checkpoint, dict) else {"config": config, "state_dict": state})
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    pred_parts: list[pd.DataFrame] = []
    for series_id, index_part in forecast_index.groupby("series_id", sort=False):
        t0 = index_part["timestamp"].min()
        hist = get_series_history(train_frame, series_id, t0)
        if hist.empty:
            raise ValueError(
                f"No history in train.csv for series {series_id!r} before {t0}."
            )

        future_cov = get_series_future_cov(cov_frame, train_frame, series_id, index_part)
        if len(future_cov) != len(index_part):
            raise ValueError(
                f"Future covariate rows ({len(future_cov)}) != forecast rows ({len(index_part)}) "
                f"for series {series_id!r}."
            )

        yhat = forecast_series(model, hist, index_part, future_cov, device)
        part = index_part[["series_id", "timestamp"]].copy()
        part["prediction"] = yhat
        pred_parts.append(part)

    pred_long = pd.concat(pred_parts, ignore_index=True)

    # Strictly preserve forecast_index row order.
    predictions = forecast_index[["series_id", "timestamp"]].merge(
        pred_long,
        on=["series_id", "timestamp"],
        how="left",
        validate="one_to_one",
    )
    predictions["timestamp"] = predictions["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    print_prediction_diagnostics(predictions, train_frame, forecast_index)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_file, index=False)
    print(f"wrote {len(predictions)} predictions -> {args.output_file}")


if __name__ == "__main__":
    main()
