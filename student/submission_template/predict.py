"""Inference entrypoint for TiDE private evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.model import ForecastModel

COVARIATE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "trend",
    "workload_intensity", "demand_forecast", "staffing_forecast",
    "upstream_quality_forecast", "promotion_intensity", "shock_risk",
    "maintenance_known", "unit_reliability_forecast", "queue_pressure_forecast",
    "network_pressure_forecast", "event_load_forecast",
    "service_irregularity_risk_forecast", "throughput_disruption_risk_forecast",
    "nominal_capacity", "zone_sin", "zone_cos",
]
TARGET_COL = "target"


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
        num_covariates=config.get("num_covariates", len(COVARIATE_COLS)),
        hidden_dim=config.get("hidden_dim", 256),
        num_encoder_layers=config.get("num_encoder_layers", 2),
        num_decoder_layers=config.get("num_decoder_layers", 2),
        dropout=config.get("dropout", 0.1),
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

    past_target = (
        hist.reindex(columns=[TARGET_COL])
        .astype(np.float32)
        .fillna(0.0)
        .to_numpy()
        .reshape(-1)
    )
    past_cov = (
        hist.reindex(columns=COVARIATE_COLS)
        .astype(np.float32)
        .fillna(0.0)
        .to_numpy()
    )

    preds: list[float] = []
    horizon = len(future_index)
    steps_done = 0

    while steps_done < horizon:
        if len(past_target) < seq_len:
            pad_len = seq_len - len(past_target)
            past_target = np.concatenate([np.repeat(past_target[:1], pad_len), past_target])
            past_cov = np.concatenate([np.repeat(past_cov[:1], pad_len, axis=0), past_cov], axis=0)
        else:
            past_target = past_target[-seq_len:]
            past_cov = past_cov[-seq_len:]

        take = min(pred_len, horizon - steps_done)
        future_cov = extract_future_covariates(future_cov_frame, steps_done, take)
        if take < pred_len:
            pad = np.zeros((pred_len - take, len(COVARIATE_COLS)), dtype=np.float32)
            future_cov = np.concatenate([future_cov, pad], axis=0)

        pred_block = model(
            torch.from_numpy(past_target).unsqueeze(0).to(device),
            torch.from_numpy(past_cov).unsqueeze(0).to(device),
            torch.from_numpy(future_cov).unsqueeze(0).to(device),
        )[0].detach().cpu().numpy()[:take]

        preds.extend(pred_block.tolist())
        past_target = np.concatenate([past_target, pred_block])
        past_cov = np.concatenate([past_cov, future_cov[:take]], axis=0)
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
