"""Ensemble TiDE checkpoints with per-checkpoint weights, seasonal blend, and per-series calibration.

Examples:
  python predict_ensemble.py \\
    --input_dir ../../data \\
    --checkpoints checkpoint_seq336.pt,checkpoint_seq504.pt,checkpoint_seq672.pt \\
    --checkpoint-weights 0.3,0.4,0.3 \\
    --tide-weight 0.8 --seasonal-weight 0.2 \\
    --calibrate \\
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
    forecast_series,
    load_checkpoint_dict,
    predict_with_model,
    print_prediction_diagnostics,
    resolve_device,
)
from src.data_io import (
    get_series_future_cov,
    get_series_history,
    load_cov_frame,
    load_forecast_index,
    load_train_frame,
    resolve_data_dir,
)
from src.features import TARGET_COL
from src.model import ForecastModel


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


def _list_local_checkpoints(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.pt"))


def resolve_seed_checkpoint(base: Path, seed: str) -> Path | None:
    stem = base.stem
    suffix = base.suffix or ".pt"
    parent = base.parent
    candidates = [
        parent / f"{stem}_seed{seed}{suffix}",
        parent / f"{stem}_{seed}{suffix}",
        parent / f"checkpoint_seed{seed}{suffix}",
        parent / f"checkpoint_{seed}{suffix}",
    ]
    for path in candidates:
        if path.exists():
            return path
    if base.exists():
        return base
    plain = parent / f"checkpoint{suffix}"
    if plain.exists():
        return plain
    return None


def parse_checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    warnings: list[str] = []

    if args.checkpoints:
        paths.extend(Path(part.strip()) for part in args.checkpoints.split(",") if part.strip())

    if args.ensemble_seeds:
        seeds = [part.strip() for part in args.ensemble_seeds.split(",") if part.strip()]
        resolved_for_seeds: list[Path] = []
        missing_seeds: list[str] = []
        for seed in seeds:
            found = resolve_seed_checkpoint(args.checkpoint, seed)
            if found is None:
                missing_seeds.append(seed)
                continue
            expected = args.checkpoint.parent / (
                f"{args.checkpoint.stem}_seed{seed}{args.checkpoint.suffix or '.pt'}"
            )
            if found.resolve() != expected.resolve():
                warnings.append(
                    f"seed={seed}: expected {expected.name}, using fallback {found.name}"
                )
            resolved_for_seeds.append(found)
        if missing_seeds:
            available = _list_local_checkpoints(args.checkpoint.parent)
            available_msg = (
                ", ".join(path.name for path in available) if available else "(none found)"
            )
            raise FileNotFoundError(
                "Missing seed checkpoints for seeds: "
                + ", ".join(missing_seeds)
                + f"\nAvailable .pt files: {available_msg}"
            )
        paths.extend(resolved_for_seeds)

    if getattr(args, "auto_checkpoints", False):
        paths.extend(_list_local_checkpoints(args.checkpoint.parent))

    if not paths:
        if args.checkpoint.exists():
            paths.append(args.checkpoint)
        else:
            raise FileNotFoundError("No checkpoints provided / found.")

    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)

    missing = [str(path) for path in unique if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint(s): {missing}")

    if warnings:
        print("checkpoint resolution notes:")
        for msg in warnings:
            print(f"  - {msg}")
        if len(unique) == 1 and args.ensemble_seeds and "," in args.ensemble_seeds:
            print(
                "WARNING: multiple seeds resolved to the same checkpoint file "
                f"({unique[0].name}). Predictions will NOT be multi-seed averaged."
            )
    return unique


def parse_checkpoint_weights(raw: str | None, n: int) -> np.ndarray:
    if raw is None or raw.strip() == "":
        weights = np.ones(n, dtype=np.float64)
    else:
        parts = [float(x.strip()) for x in raw.split(",") if x.strip()]
        if len(parts) != n:
            raise ValueError(
                f"--checkpoint-weights has {len(parts)} values but {n} checkpoints were provided."
            )
        weights = np.asarray(parts, dtype=np.float64)
        if np.any(weights < 0):
            raise ValueError("checkpoint weights must be non-negative.")
        if float(weights.sum()) <= 0:
            raise ValueError("checkpoint weights must sum to > 0.")
    weights = weights / weights.sum()
    return weights


def align_to_forecast_index(
    predictions: pd.DataFrame,
    forecast_index: pd.DataFrame,
) -> pd.DataFrame:
    fi = forecast_index[["series_id", "timestamp"]].copy()
    pred = predictions.copy()
    fi["timestamp"] = pd.to_datetime(fi["timestamp"])
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    aligned = fi.merge(pred, on=["series_id", "timestamp"], how="left", validate="one_to_one")
    if aligned["prediction"].isna().any():
        n_missing = int(aligned["prediction"].isna().sum())
        raise ValueError(f"Missing {n_missing} predictions after aligning to forecast_index.")
    return aligned


def load_models(
    checkpoint_paths: list[Path],
    device: torch.device,
) -> list[tuple[Path, ForecastModel, dict]]:
    models: list[tuple[Path, ForecastModel, dict]] = []
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
        models.append((path, model, config))
    return models


def run_weighted_tide_ensemble(
    models: list[tuple[Path, ForecastModel, dict]],
    weights: np.ndarray,
    forecast_index: pd.DataFrame,
    train_frame: pd.DataFrame,
    cov_frame: pd.DataFrame | None,
    device: torch.device,
) -> np.ndarray:
    stack: list[np.ndarray] = []
    for (path, model, _), weight in zip(models, weights):
        preds = predict_with_model(model, forecast_index, train_frame, cov_frame, device)
        aligned = align_to_forecast_index(preds, forecast_index)
        values = aligned["prediction"].to_numpy(dtype=np.float64)
        print(
            f"  {path.name} weight={weight:.4f} "
            f"mean={values.mean():.4f} min={values.min():.4f} max={values.max():.4f}",
            flush=True,
        )
        stack.append(values)
    stacked = np.stack(stack, axis=0)
    return np.average(stacked, axis=0, weights=weights)


def fit_series_calibration(
    models: list[tuple[Path, ForecastModel, dict]],
    weights: np.ndarray,
    train_frame: pd.DataFrame,
    device: torch.device,
    *,
    horizon: int = 168,
    min_points: int = 48,
    a_clip: tuple[float, float] = (0.5, 1.5),
    mode: str = "bias",
    shrinkage: float = 0.5,
    b_clip: float = 3.0,
    fit_models: list[tuple[Path, ForecastModel, dict]] | None = None,
) -> dict[str, tuple[float, float]]:
    """
    Fit per-series correction on the last `horizon` train steps.

    Calibration is fit on TiDE ensemble predictions only (before seasonal blend).
    No future leakage: history is strictly before the calibration block.
    """
    coeffs: dict[str, tuple[float, float]] = {}
    train_frame = train_frame.copy()
    train_frame["timestamp"] = pd.to_datetime(train_frame["timestamp"])
    fit_models = fit_models or models
    fit_weights = weights[: len(fit_models)]
    if fit_weights.sum() > 0:
        fit_weights = fit_weights / fit_weights.sum()

    print(
        f"fitting per-series calibration ({mode}) on last {horizon} train steps "
        f"using {len(fit_models)} model(s)...",
        flush=True,
    )

    for series_id, group in train_frame.groupby("series_id", sort=False):
        sorted_group = group.sort_values("timestamp").reset_index(drop=True)
        if len(sorted_group) <= horizon + 24:
            coeffs[str(series_id)] = (1.0, 0.0)
            continue

        history = sorted_group.iloc[:-horizon]
        future_block = sorted_group.iloc[-horizon:]
        future_index = future_block[["series_id", "timestamp"]]
        y_true = future_block[TARGET_COL].to_numpy(dtype=np.float64)

        member_preds: list[np.ndarray] = []
        for (_, model, _), weight in zip(fit_models, fit_weights):
            y_hat = forecast_series(
                model,
                history,
                future_index,
                future_block,
                device,
            )
            member_preds.append(y_hat.astype(np.float64) * float(weight))
        y_pred = np.sum(np.stack(member_preds, axis=0), axis=0)

        if len(y_pred) < min_points or np.allclose(y_pred.std(), 0.0):
            coeffs[str(series_id)] = (1.0, 0.0)
            continue

        strength = min(1.0, len(y_pred) / max(float(horizon), 1.0)) * shrinkage
        if mode == "bias":
            raw_b = float(np.mean(y_true - y_pred))
            b = float(np.clip(strength * raw_b, -b_clip, b_clip))
            coeffs[str(series_id)] = (1.0, b)
            continue

        design = np.column_stack([y_pred, np.ones_like(y_pred)])
        try:
            ab, _, _, _ = np.linalg.lstsq(design, y_true, rcond=None)
            a_raw = float(ab[0])
            b_raw = float(ab[1])
            a = float(np.clip(1.0 + strength * (a_raw - 1.0), a_clip[0], a_clip[1]))
            b = float(np.clip(strength * b_raw, -b_clip, b_clip))
        except np.linalg.LinAlgError:
            a, b = 1.0, 0.0
        coeffs[str(series_id)] = (a, b)

    a_vals = np.array([a for a, _ in coeffs.values()], dtype=np.float64)
    b_vals = np.array([b for _, b in coeffs.values()], dtype=np.float64)
    print(
        f"calibration stats: n_series={len(coeffs)} "
        f"a mean={a_vals.mean():.4f} [{a_vals.min():.4f}, {a_vals.max():.4f}] "
        f"b mean={b_vals.mean():.4f} [{b_vals.min():.4f}, {b_vals.max():.4f}]",
        flush=True,
    )
    return coeffs


def apply_series_calibration(
    predictions: pd.DataFrame,
    coeffs: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    out = predictions.copy()
    calibrated = []
    for series_id, value in zip(out["series_id"].astype(str), out["prediction"].to_numpy()):
        a, b = coeffs.get(series_id, (1.0, 0.0))
        calibrated.append(a * float(value) + b)
    out["prediction"] = np.clip(np.asarray(calibrated, dtype=np.float64), 0.0, None)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TiDE weighted ensemble + seasonal_mean blend + per-series calibration."
    )
    parser.add_argument("--input_dir", type=Path, default=Path("../../data"))
    parser.add_argument("--output_file", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint.pt"))
    parser.add_argument("--checkpoints", type=str, default=None)
    parser.add_argument("--ensemble-seeds", type=str, default=None)
    parser.add_argument("--auto-checkpoints", action="store_true")
    parser.add_argument(
        "--checkpoint-weights",
        type=str,
        default=None,
        help="Comma-separated weights aligned with resolved checkpoint order. Default: uniform.",
    )
    parser.add_argument("--tide-weight", type=float, default=0.8)
    parser.add_argument("--seasonal-weight", type=float, default=0.2)
    parser.add_argument("--no-seasonal", action="store_true")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Optional: fit/apply per-series bias/linear correction on TiDE before seasonal blend.",
    )
    parser.add_argument(
        "--calibrate-mode",
        choices=["bias", "linear"],
        default="bias",
        help="bias=intercept-only (fast/safer); linear=slope+intercept.",
    )
    parser.add_argument(
        "--calibrate-horizon",
        type=int,
        default=168,
        help="Train-tail backtest length for calibration fit (default 168 for speed).",
    )
    parser.add_argument(
        "--calibrate-fit-models",
        type=int,
        default=1,
        help="Use only the first N checkpoints when fitting calibration (speed).",
    )
    parser.add_argument("--calibrate-shrinkage", type=float, default=0.5)
    parser.add_argument("--calibrate-b-max", type=float, default=3.0)
    parser.add_argument("--calibrate-a-min", type=float, default=0.5)
    parser.add_argument("--calibrate-a-max", type=float, default=1.5)
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
    ckpt_weights = parse_checkpoint_weights(args.checkpoint_weights, len(checkpoint_paths))
    print(f"--> TiDE checkpoints ({len(checkpoint_paths)}):")
    for path, weight in zip(checkpoint_paths, ckpt_weights):
        print(f"    - {path}  (weight={weight:.4f})")

    data_dir = resolve_data_dir(args.input_dir)
    print(f"using data_dir: {data_dir}")
    forecast_index = load_forecast_index(data_dir)
    train_frame = load_train_frame(data_dir)
    cov_frame = load_cov_frame(data_dir)

    models = load_models(checkpoint_paths, device)

    tide_pred = run_weighted_tide_ensemble(
        models, ckpt_weights, forecast_index, train_frame, cov_frame, device
    )
    tide_pred = np.clip(tide_pred, 0.0, None)

    coeffs: dict[str, tuple[float, float]] | None = None
    if args.calibrate:
        fit_n = max(1, min(args.calibrate_fit_models, len(models)))
        fit_models = models[:fit_n]
        fit_weights = ckpt_weights[:fit_n]
        coeffs = fit_series_calibration(
            models,
            ckpt_weights,
            train_frame,
            device,
            horizon=args.calibrate_horizon,
            mode=args.calibrate_mode,
            shrinkage=args.calibrate_shrinkage,
            b_clip=args.calibrate_b_max,
            a_clip=(args.calibrate_a_min, args.calibrate_a_max),
            fit_models=fit_models,
        )
        tide_df = forecast_index[["series_id", "timestamp"]].copy()
        tide_df["prediction"] = tide_pred
        print("applying per-series calibration to TiDE before seasonal blend...", flush=True)
        tide_df = apply_series_calibration(tide_df, coeffs)
        tide_pred = tide_df["prediction"].to_numpy(dtype=np.float64)

    if seasonal_w > 0:
        print("computing seasonal_mean baseline...", flush=True)
        seasonal = seasonal_mean_forecast(train_frame, forecast_index)
        seasonal_aligned = align_to_forecast_index(seasonal, forecast_index)
        seasonal_pred = np.clip(
            seasonal_aligned["prediction"].to_numpy(dtype=np.float64), 0.0, None
        )
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

    predictions["prediction"] = np.clip(
        predictions["prediction"].to_numpy(dtype=np.float64), 0.0, None
    )
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
