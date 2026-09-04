"""Inference entrypoint for TiDE with optional multi-seed ensemble averaging."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data_io import (
    detect_forecast_index_path,
    get_series_future_cov,
    get_series_history,
    load_cov_frame,
    load_forecast_index,
    load_train_frame,
    resolve_data_dir,
)
from src.features import (
    BASE_COVARIATE_COLS,
    COVARIATE_COLS,
    FEATURE_COLS,
    TARGET_COL,
    TARGET_IDX,
    enrich_features,
)
from src.lgb_infer import infer_feature_columns, predict_lightgbm
from src.model import ForecastModel

TIDE_WEIGHT = 0.7
LGB_WEIGHT = 0.3
UPPER_SCALE = 1.1


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _state_tensor(state: dict, *suffixes: str) -> torch.Tensor | None:
    for key, value in state.items():
        for suffix in suffixes:
            if key == suffix or key.endswith("." + suffix) or key.endswith(suffix):
                if torch.is_tensor(value):
                    return value
    return None


def _count_residual_blocks(state: dict, prefix: str) -> int:
    indexes: set[int] = set()
    for key in state:
        if not key.startswith(prefix) or ".ff.0.weight" not in key:
            continue
        part = key[len(prefix) :].split(".", 1)[0]
        if part.isdigit():
            indexes.add(int(part))
    return len(indexes)


def infer_tide_config_from_state_dict(state: dict) -> dict:
    """Recover architecture dims from Linear weight shapes (authoritative)."""
    past_w = _state_tensor(state, "past_feat_proj.weight")
    future_w = _state_tensor(state, "future_feat_proj.weight")
    skip_w = _state_tensor(state, "global_skip.weight")
    enc0_w = _state_tensor(state, "encoder.0.weight")
    if past_w is None or future_w is None:
        raise ValueError("Cannot infer TiDE dims: past_feat_proj / future_feat_proj missing.")

    hidden_dim = int(past_w.shape[0])
    num_features = int(past_w.shape[1])
    num_covariates = int(future_w.shape[1])
    if num_features != num_covariates + 1:
        print(
            f"WARNING: past in_features={num_features} != num_covariates+1 "
            f"({num_covariates}+1); using weight shapes as-is.",
            flush=True,
        )

    if skip_w is not None:
        pred_len = int(skip_w.shape[0])
        seq_len = int(skip_w.shape[1])
    elif enc0_w is not None and hidden_dim > 0 and enc0_w.shape[1] % hidden_dim == 0:
        seq_len = int(enc0_w.shape[1] // hidden_dim)
        pred_len = 24
    else:
        raise ValueError("Cannot infer seq_len/pred_len from checkpoint weights.")

    if enc0_w is not None:
        expected = seq_len * hidden_dim
        if int(enc0_w.shape[1]) != expected:
            seq_len = int(enc0_w.shape[1] // hidden_dim)
            print(f"WARNING: encoder in_features implies seq_len={seq_len}", flush=True)

    return {
        "seq_len": seq_len,
        "pred_len": pred_len,
        "num_features": num_features,
        "num_covariates": num_covariates,
        "target_idx": num_features - 1,
        "hidden_dim": hidden_dim,
        "num_encoder_layers": _count_residual_blocks(state, "encoder."),
        "num_decoder_layers": _count_residual_blocks(state, "decoder."),
        "use_revin": any(key.split(".")[0] == "revin" or ".revin." in key for key in state),
    }


def resolve_feature_layout(config: dict) -> tuple[list[str], list[str], int]:
    """Align feature/covariate column lists with the model's channel count."""
    n_cov = int(config.get("num_covariates", len(COVARIATE_COLS)))
    saved_cov = config.get("covariate_cols")
    saved_feat = config.get("feature_cols")
    target_col = str(config.get("target_col", TARGET_COL))

    if isinstance(saved_cov, (list, tuple)) and len(saved_cov) == n_cov:
        covariate_cols = [str(c) for c in saved_cov]
    elif n_cov == len(COVARIATE_COLS):
        covariate_cols = list(COVARIATE_COLS)
    elif n_cov == len(BASE_COVARIATE_COLS):
        covariate_cols = list(BASE_COVARIATE_COLS)
    else:
        pool = list(COVARIATE_COLS)
        covariate_cols = pool[:n_cov] if len(pool) >= n_cov else pool
        if len(covariate_cols) < n_cov:
            raise ValueError(
                f"Need {n_cov} covariate columns but only {len(covariate_cols)} are defined."
            )

    if isinstance(saved_feat, (list, tuple)) and len(saved_feat) == n_cov + 1:
        feature_cols = [str(c) for c in saved_feat]
    else:
        feature_cols = covariate_cols + [target_col]

    if target_col in feature_cols:
        target_idx = int(config.get("target_idx", feature_cols.index(target_col)))
        if feature_cols[target_idx] != target_col:
            target_idx = feature_cols.index(target_col)
    else:
        target_idx = len(feature_cols) - 1
    return feature_cols, covariate_cols, target_idx


def resolve_tide_config(checkpoint: dict) -> dict:
    """Prefer saved config, then overwrite architectural dims from weight shapes."""
    state, saved = extract_tide_bundle(checkpoint)
    merged = dict(saved or {})
    for alt_key in ("config", "args", "hyperparameters", "hparams", "model_args"):
        extra = checkpoint.get(alt_key)
        if isinstance(extra, dict):
            for key, value in extra.items():
                merged.setdefault(key, value)

    inferred = infer_tide_config_from_state_dict(state)
    for key, value in inferred.items():
        if merged.get(key) != value:
            if key in merged:
                print(f"  checkpoint config {key}={merged.get(key)} overridden by weights -> {value}", flush=True)
            merged[key] = value

    feature_cols, covariate_cols, target_idx = resolve_feature_layout(merged)
    merged["feature_cols"] = feature_cols
    merged["covariate_cols"] = covariate_cols
    merged["target_idx"] = target_idx
    merged["num_features"] = len(feature_cols)
    merged["num_covariates"] = len(covariate_cols)
    return merged


def build_model_from_checkpoint(checkpoint: dict) -> ForecastModel:
    if "tide_state_dict" in checkpoint or "state_dict" in checkpoint:
        config = resolve_tide_config(checkpoint)
    else:
        config = checkpoint.get("tide_config") or checkpoint.get("config", {})
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
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    if "tide_state_dict" in checkpoint:
        return checkpoint
    if "state_dict" in checkpoint:
        return checkpoint
    return {"state_dict": checkpoint, "config": {}}


def extract_tide_bundle(checkpoint: dict) -> tuple[dict, dict]:
    if "tide_state_dict" in checkpoint:
        return checkpoint["tide_state_dict"], checkpoint.get("tide_config", {})
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint.get("config", {})
    raise ValueError("Checkpoint is missing tide_state_dict / state_dict.")


def series_max_targets(train_frame: pd.DataFrame) -> pd.Series:
    target = pd.to_numeric(train_frame[TARGET_COL], errors="coerce")
    caps = train_frame.assign(_target=target).groupby("series_id", sort=False)["_target"].max()
    if caps.isna().any():
        raise ValueError("Some series have no valid historical target in train.csv.")
    return caps


def apply_physical_clip(predictions: pd.DataFrame, train_frame: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    values = out["prediction"].to_numpy(dtype=np.float64)
    caps = series_max_targets(train_frame)
    upper = out["series_id"].map(caps) * UPPER_SCALE
    if upper.isna().any():
        missing = sorted(out.loc[upper.isna(), "series_id"].astype(str).unique())
        raise ValueError(f"No historical max_target for series: {missing}")
    upper_vals = upper.to_numpy(dtype=np.float64)
    clipped = np.minimum(values, upper_vals)
    clipped = np.maximum(clipped, 0.0)
    print(
        f"physical clip: to_zero={(values < 0).sum()} "
        f"to_upper={(values > upper_vals).sum()} "
        f"cap_scale={UPPER_SCALE}",
        flush=True,
    )
    out["prediction"] = clipped
    return out


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
    covariate_cols: list[str] | None = None,
) -> np.ndarray:
    """Slice future covariates by timestamp alignment."""
    cols = covariate_cols or COVARIATE_COLS
    block_index = future_index.iloc[start : start + length]
    merged = block_index.merge(
        future_cov_frame,
        on=["series_id", "timestamp"],
        how="left",
    )
    values = (
        merged.reindex(columns=cols)
        .astype(np.float32)
        .fillna(0.0)
        .to_numpy()
    )
    if values.shape[0] < length:
        pad = np.zeros((length - values.shape[0], len(cols)), dtype=np.float32)
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
    feature_cols: list[str] | None = None,
    covariate_cols: list[str] | None = None,
    target_idx: int | None = None,
) -> np.ndarray:
    """Roll out pred_len-step blocks using the checkpoint's seq_len / channel layout."""
    seq_len, pred_len = model.seq_len, model.pred_len
    feat_cols = feature_cols or FEATURE_COLS
    cov_cols = covariate_cols or COVARIATE_COLS
    tgt_idx = TARGET_IDX if target_idx is None else target_idx

    history = enrich_features(history)
    future_cov_frame = enrich_features(future_cov_frame)

    values = (
        history.reindex(columns=feat_cols)
        .astype(np.float32)
        .fillna(0.0)
        .to_numpy()
    )
    if values.shape[0] == 0:
        raise ValueError("History is empty; cannot forecast.")
    if values.shape[1] != model.num_features:
        raise ValueError(
            f"History feature dim {values.shape[1]} != model.num_features {model.num_features}."
        )

    preds: list[float] = []
    horizon = len(future_index)
    steps_done = 0

    while steps_done < horizon:
        take = min(pred_len, horizon - steps_done)
        x_past = build_past_window(values, seq_len)
        x_future = extract_future_covariates(
            future_cov_frame,
            future_index,
            steps_done,
            pred_len,
            covariate_cols=cov_cols,
        )
        if x_past.shape != (seq_len, model.num_features):
            raise ValueError(
                f"x_past shape {x_past.shape} != {(seq_len, model.num_features)}"
            )
        if x_future.shape[1] != model.num_covariates:
            raise ValueError(
                f"x_future channels {x_future.shape[1]} != model.num_covariates {model.num_covariates}"
            )

        pred_block = model(
            torch.from_numpy(x_past).unsqueeze(0).to(device),
            torch.from_numpy(x_future).unsqueeze(0).to(device),
        )[0].detach().cpu().numpy()[:take]

        pred_block = np.clip(pred_block, 0.0, None)
        preds.extend(pred_block.tolist())

        future_cov_known = x_future[:take]
        appended = np.zeros((take, len(feat_cols)), dtype=np.float32)
        appended[:, : len(cov_cols)] = future_cov_known
        appended[:, tgt_idx] = pred_block
        values = np.concatenate([values, appended], axis=0)
        steps_done += take

    return np.asarray(preds[:horizon], dtype=np.float64)


def predict_with_model(
    model: ForecastModel,
    forecast_index: pd.DataFrame,
    train_frame: pd.DataFrame,
    cov_frame: pd.DataFrame | None,
    device: torch.device,
    feature_cols: list[str] | None = None,
    covariate_cols: list[str] | None = None,
    target_idx: int | None = None,
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

        yhat = forecast_series(
            model,
            hist,
            index_part,
            future_cov,
            device,
            feature_cols=feature_cols,
            covariate_cols=covariate_cols,
            target_idx=target_idx,
        )
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
    parser = argparse.ArgumentParser(description="Official TiDE + LightGBM inference.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("../../data"),
        help="Directory with train.csv and forecast_index_*.csv.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("test_predictions.csv"),
        help="Output CSV path (default: test_predictions.csv).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoint.pt"),
        help="Packed checkpoint.pt with tide_state_dict and lgb_model.",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    device = resolve_device()
    print(f"--> inference device: {device}")

    data_dir = resolve_data_dir(args.input_dir)
    index_path = detect_forecast_index_path(data_dir)
    print(f"using data_dir: {data_dir}")
    print(f"detected forecast index: {index_path.name}")

    forecast_index = load_forecast_index(data_dir)
    train_frame = load_train_frame(data_dir)
    cov_frame = load_cov_frame(data_dir)

    checkpoint = load_checkpoint_dict(args.checkpoint, device)
    tide_state, _saved_config = extract_tide_bundle(checkpoint)
    tide_config = resolve_tide_config(checkpoint)
    feature_cols = list(tide_config["feature_cols"])
    covariate_cols = list(tide_config["covariate_cols"])
    target_idx = int(tide_config["target_idx"])
    print(
        f"loading TiDE from weights: seq_len={tide_config.get('seq_len')} "
        f"pred_len={tide_config.get('pred_len')} "
        f"num_features={tide_config.get('num_features')} "
        f"num_covariates={tide_config.get('num_covariates')} "
        f"target_idx={target_idx} hidden_dim={tide_config.get('hidden_dim')}",
        flush=True,
    )
    model = build_model_from_checkpoint(checkpoint)
    incompatible = model.load_state_dict(tide_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"  load_state_dict notes missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}",
            flush=True,
        )
    model.eval()
    model.to(device)

    print("running TiDE inference...", flush=True)
    tide_pred = predict_with_model(
        model,
        forecast_index,
        train_frame,
        cov_frame,
        device,
        feature_cols=feature_cols,
        covariate_cols=covariate_cols,
        target_idx=target_idx,
    )
    tide_vals = tide_pred["prediction"].to_numpy(dtype=np.float64)

    lgb_raw = checkpoint.get("lgb_model")
    if lgb_raw is None:
        print("checkpoint has no lgb_model; using TiDE-only predictions.", flush=True)
        blended = tide_vals
    else:
        lgb_feature_cols = checkpoint.get("lgb_feature_cols") or infer_feature_columns(train_frame)
        print("running LightGBM inference...", flush=True)
        lgb_pred = predict_lightgbm(
            lgb_raw,
            forecast_index,
            train_frame,
            cov_frame,
            feature_cols=lgb_feature_cols,
        )
        lgb_vals = lgb_pred["prediction"].to_numpy(dtype=np.float64)
        blended = TIDE_WEIGHT * tide_vals + LGB_WEIGHT * lgb_vals
        print(
            f"blend weights: tide={TIDE_WEIGHT:.3f} lgb={LGB_WEIGHT:.3f} "
            f"tide_mean={tide_vals.mean():.4f} lgb_mean={lgb_vals.mean():.4f}",
            flush=True,
        )

    predictions = forecast_index[["series_id", "timestamp"]].copy()
    predictions["prediction"] = blended
    predictions = apply_physical_clip(predictions, train_frame)
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"]).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    print_prediction_diagnostics(predictions, train_frame, forecast_index)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_file, index=False)
    print(f"wrote {len(predictions)} predictions -> {args.output_file}")


if __name__ == "__main__":
    main()
