"""Inference entrypoint for TiDE private evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.features import COVARIATE_COLS, FEATURE_COLS, TARGET_COL, TARGET_IDX
from src.model import ForecastModel


def load_forecast_index(input_dir: Path) -> pd.DataFrame:
    candidates = [
        input_dir / "forecast_index_test.csv",
        input_dir / "forecast_index_validation.csv",
    ]
    for forecast_index in candidates:
        if forecast_index.exists():
            return pd.read_csv(forecast_index)
    expected = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(f"Expected one of {expected} in input_dir.")


def load_history_frame(input_dir: Path) -> pd.DataFrame:
    """Private eval uses test_input.csv; local validation uses validation_input.csv."""
    candidates = [
        input_dir / "test_input.csv",
        input_dir / "validation_input.csv",
        input_dir / "train.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
    raise FileNotFoundError(
        f"Expected one of {[p.name for p in candidates]} in {input_dir}."
    )


def build_model_from_checkpoint(checkpoint: dict) -> ForecastModel:
    """Instantiate TiDE ForecastModel from checkpoint config."""
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
    start: int,
    length: int,
) -> np.ndarray:
    block = future_cov_frame.iloc[start : start + length]
    values = (
        block.reindex(columns=COVARIATE_COLS)
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


@torch.no_grad()
def forecast_series(
    model: ForecastModel,
    history: pd.DataFrame,
    future_index: pd.DataFrame,
    input_frame: pd.DataFrame,
    device: torch.device,
) -> np.ndarray:
    """Roll out 24-step TiDE blocks using known future covariates."""
    seq_len, pred_len = model.seq_len, model.pred_len
    hist = history.sort_values("timestamp").copy()

    future_cov_frame = (
        input_frame.merge(future_index, on=["series_id", "timestamp"], how="inner")
        .sort_values("timestamp")
    )

    values = (
        hist.reindex(columns=FEATURE_COLS)
        .astype(np.float32)
        .fillna(0.0)
        .to_numpy()
    )

    preds: list[float] = []
    horizon = len(future_index)
    steps_done = 0

    while steps_done < horizon:
        take = min(pred_len, horizon - steps_done)
        x_past = build_past_window(values, seq_len)
        x_future = extract_future_covariates(future_cov_frame, steps_done, pred_len)

        pred_block = model(
            torch.from_numpy(x_past).unsqueeze(0).to(device),
            torch.from_numpy(x_future).unsqueeze(0).to(device),
        )[0].detach().cpu().numpy()[:take]

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
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--> 当前推理使用的计算设备: {device}")

    forecast_index = load_forecast_index(args.input_dir)
    forecast_index["timestamp"] = pd.to_datetime(forecast_index["timestamp"])
    history_frame = load_history_frame(args.input_dir)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
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

    rows = []
    for series_id, index_part in forecast_index.groupby("series_id", sort=False):
        hist = history_frame.loc[history_frame["series_id"].eq(series_id)]
        t0 = index_part["timestamp"].min()
        hist = hist.loc[hist["timestamp"] < t0]
        if hist.empty:
            raise ValueError(f"No history for series {series_id!r} before {t0}.")

        yhat = forecast_series(model, hist, index_part, history_frame, device)
        part = index_part[["series_id", "timestamp"]].copy()
        part["prediction"] = yhat
        rows.append(part)

    predictions = pd.concat(rows, ignore_index=True)
    predictions["timestamp"] = predictions["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
