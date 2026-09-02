"""Inference entrypoint for TiDE with optional multi-seed ensemble averaging."""

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
from src.features import COVARIATE_COLS, FEATURE_COLS, TARGET_COL, TARGET_IDX, enrich_features
from src.model import ForecastModel

DEFAULT_ENSEMBLE_SEEDS = (42, 2024, 777)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model_from_checkpoint(checkpoint: dict) -> ForecastModel:
    config = checkpoint.get("config", {})
    return ForecastModel(
        seq_len=config.get("seq_len", 504),
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


def load_checkpoint_dict(path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint
    if isinstance(checkpoint, dict):
        return {"state_dict": checkpoint, "config": {}}
    raise ValueError(f"Unsupported checkpoint format: {path}")


def resolve_checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.checkpoints:
        paths.extend(Path(part.strip()) for part in args.checkpoints.split(",") if part.strip())
    elif args.ensemble_seeds:
        seeds = [part.strip() for part in args.ensemble_seeds.split(",") if part.strip()]
        stem = args.checkpoint.stem
        suffix = args.checkpoint.suffix or ".pt"
        parent = args.checkpoint.parent
        for seed in seeds:
            paths.append(parent / f"{stem}_seed{seed}{suffix}")
    else:
        paths.append(args.checkpoint)

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint(s): {missing}")
    return paths


def extract_future_covariates(
    future_cov_frame: pd.DataFrame,
    future_index: pd.DataFrame,
    start: int,
    length: int,
) -> np.ndarray:
    """Slice future covariates by timestamp alignment."""
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

    history = enrich_features(history)
    future_cov_frame = enrich_features(future_cov_frame)

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

        pred_block = np.clip(pred_block, 0.0, None)
        preds.extend(pred_block.tolist())

        future_cov_known = x_future[:take]
        appended = np.zeros((take, len(FEATURE_COLS)), dtype=np.float32)
        appended[:, : len(COVARIATE_COLS)] = future_cov_known
        appended[:, TARGET_IDX] = pred_block
        values = np.concatenate([values, appended], axis=0)
        steps_done += take

    return np.asarray(preds[:horizon], dtype=np.float64)


def predict_with_model(
    model: ForecastModel,
    forecast_index: pd.DataFrame,
    train_frame: pd.DataFrame,
    cov_frame: pd.DataFrame | None,
    device: torch.device,
) -> pd.DataFrame:
    pred_parts: list[pd.DataFrame] = []
    for series_id, index_part in forecast_index.groupby("series_id", sort=False):
        t0 = index_part["timestamp"].min()
        hist = get_series_history(train_frame, series_id, t0)
        if hist.empty:
            raise ValueError(f"No history in train.csv for series {series_id!r} before {t0}.")

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
    predictions = forecast_index[["series_id", "timestamp"]].merge(
        pred_long,
        on=["series_id", "timestamp"],
        how="left",
        validate="one_to_one",
    )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate private test predictions.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("../../data"),
        help="Directory with train.csv and validation_input.csv (default: ../../data).",
    )
    parser.add_argument("--output_file", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoint.pt"),
        help="Single checkpoint path (also used as stem for --ensemble-seeds).",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma-separated checkpoint paths for seed averaging.",
    )
    parser.add_argument(
        "--ensemble-seeds",
        type=str,
        default=None,
        help="Comma-separated seeds; loads checkpoint_seed{seed}.pt next to --checkpoint. "
        f"Example: {','.join(str(s) for s in DEFAULT_ENSEMBLE_SEEDS)}",
    )
    args = parser.parse_args()

    device = resolve_device()
    print(f"--> inference device: {device}")

    checkpoint_paths = resolve_checkpoint_paths(args)
    print(f"--> using {len(checkpoint_paths)} checkpoint(s):")
    for path in checkpoint_paths:
        print(f"    - {path}")

    data_dir = resolve_data_dir(args.input_dir)
    print(f"using data_dir: {data_dir}")
    forecast_index = load_forecast_index(data_dir)
    train_frame = load_train_frame(data_dir)
    cov_frame = load_cov_frame(data_dir)

    ensemble_preds: list[np.ndarray] = []
    for path in checkpoint_paths:
        checkpoint = load_checkpoint_dict(path, device)
        config = checkpoint.get("config", {})
        print(
            f"loading {path.name}: seq_len={config.get('seq_len')} "
            f"pred_len={config.get('pred_len')} use_revin={config.get('use_revin', True)} "
            f"num_covariates={config.get('num_covariates')}"
        )
        model = build_model_from_checkpoint(checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model.to(device)

        predictions = predict_with_model(model, forecast_index, train_frame, cov_frame, device)
        ensemble_preds.append(predictions["prediction"].to_numpy(dtype=np.float64))

    averaged = np.mean(np.stack(ensemble_preds, axis=0), axis=0)
    averaged = np.clip(averaged, 0.0, None)

    predictions = forecast_index[["series_id", "timestamp"]].copy()
    predictions["prediction"] = averaged
    predictions["timestamp"] = predictions["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    print_prediction_diagnostics(predictions, train_frame, forecast_index)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_file, index=False)
    print(f"wrote {len(predictions)} predictions -> {args.output_file}")


if __name__ == "__main__":
    main()
