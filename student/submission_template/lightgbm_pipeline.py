"""Standalone LightGBM hourly forecasting pipeline (independent of TiDE).

Trains on train.csv with lag / rolling / future-known covariates, then writes
predictions_lgb.csv aligned to the official forecast index.

Example:
  python lightgbm_pipeline.py --data-dir ../../data --output_file predictions_lgb.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _import_lightgbm():
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "lightgbm is required for this script. Install with: pip install lightgbm"
        ) from exc
    return lgb


TARGET_COL = "target"
LAGS = (1, 2, 24, 48, 168)
ROLL_WINDOWS = (24, 168)

KNOWN_FUTURE_COVARIATES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "trend",
    "workload_intensity",
    "demand_forecast",
    "staffing_forecast",
    "upstream_quality_forecast",
    "promotion_intensity",
    "shock_risk",
    "maintenance_known",
    "unit_reliability_forecast",
    "queue_pressure_forecast",
    "network_pressure_forecast",
    "event_load_forecast",
    "service_irregularity_risk_forecast",
    "throughput_disruption_risk_forecast",
    "nominal_capacity",
    "zone_sin",
    "zone_cos",
]

TIME_FEATURE_COLS = ["hour", "dayofweek"]
LAG_COLS = [f"lag_{lag}" for lag in LAGS]
ROLL_COLS = [f"rolling_{stat}_{window}" for window in ROLL_WINDOWS for stat in ("mean", "std")]


def resolve_data_dir(explicit: Path) -> Path:
    data_dir = explicit.expanduser().resolve()
    if (data_dir / "train.csv").exists():
        return data_dir
    raise FileNotFoundError(f"train.csv not found under {data_dir}")


def load_forecast_index(data_dir: Path) -> pd.DataFrame:
    for name in ("forecast_index_test.csv", "forecast_index_validation.csv"):
        path = data_dir / name
        if path.exists():
            frame = pd.read_csv(path)
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
            return frame
    raise FileNotFoundError(
        "Expected forecast_index_test.csv or forecast_index_validation.csv."
    )


def load_future_covariates(data_dir: Path) -> pd.DataFrame | None:
    for name in ("validation_input.csv", "test_input.csv"):
        path = data_dir / name
        if path.exists():
            frame = pd.read_csv(path)
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
            return frame
    return None


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    timestamps = pd.to_datetime(out["timestamp"])
    out["hour"] = timestamps.dt.hour.astype(np.int16)
    out["dayofweek"] = timestamps.dt.dayofweek.astype(np.int16)
    if "demand_forecast" in out.columns and "staffing_forecast" in out.columns:
        demand = pd.to_numeric(out["demand_forecast"], errors="coerce")
        staffing = pd.to_numeric(out["staffing_forecast"], errors="coerce")
        out["demand_staffing_gap"] = (demand - staffing).astype(np.float32)
    return out


def available_covariate_cols(frame: pd.DataFrame) -> list[str]:
    cols = [col for col in KNOWN_FUTURE_COVARIATES if col in frame.columns]
    if "demand_staffing_gap" in frame.columns:
        cols.append("demand_staffing_gap")
    return cols


def add_lag_rolling_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-series lags and rolling stats. Rolling windows use only past targets."""

    def _one_series(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("timestamp").copy()
        target = pd.to_numeric(group[TARGET_COL], errors="coerce")
        for lag in LAGS:
            group[f"lag_{lag}"] = target.shift(lag)
        previous = target.shift(1)
        for window in ROLL_WINDOWS:
            rolled = previous.rolling(window=window, min_periods=window)
            group[f"rolling_mean_{window}"] = rolled.mean()
            group[f"rolling_std_{window}"] = rolled.std()
        return group

    parts = [_one_series(group) for _, group in frame.groupby("series_id", sort=False)]
    return pd.concat(parts, ignore_index=True)


def feature_columns(covariate_cols: list[str]) -> list[str]:
    return TIME_FEATURE_COLS + LAG_COLS + ROLL_COLS + covariate_cols


def chronological_masks(timestamps: pd.Series, train_ratio: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    times = np.sort(pd.to_datetime(timestamps).unique())
    cut_idx = max(int(len(times) * train_ratio) - 1, 0)
    cut = times[cut_idx]
    train_mask = timestamps <= cut
    val_mask = timestamps > cut
    return train_mask.to_numpy(), val_mask.to_numpy()


def fit_lightgbm(X_train, y_train, X_val, y_val):
    lgb = _import_lightgbm()
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=1000,
        learning_rate=0.03,
        random_state=42,
        n_jobs=-1,
    )
    fit_kwargs = {
        "eval_set": [(X_val, y_val)],
        "eval_metric": "l1",
    }
    try:
        model.fit(
            X_train,
            y_train,
            early_stopping_rounds=50,
            **fit_kwargs,
        )
    except TypeError:
        model.fit(
            X_train,
            y_train,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50),
            ],
            **fit_kwargs,
        )
    return model


def _safe_lag(history: np.ndarray, lag: int) -> float:
    if history.size < lag:
        return np.nan
    return float(history[-lag])


def _safe_roll(history: np.ndarray, window: int) -> tuple[float, float]:
    if history.size < window:
        return np.nan, np.nan
    block = history[-window:]
    return float(block.mean()), float(block.std(ddof=1) if block.size > 1 else 0.0)


def features_from_history(
    history_targets: np.ndarray,
    cov_row: pd.Series,
    feature_cols: list[str],
) -> np.ndarray:
    values: dict[str, float] = {
        "hour": float(cov_row["hour"]),
        "dayofweek": float(cov_row["dayofweek"]),
    }
    for lag in LAGS:
        values[f"lag_{lag}"] = _safe_lag(history_targets, lag)
    for window in ROLL_WINDOWS:
        mean, std = _safe_roll(history_targets, window)
        values[f"rolling_mean_{window}"] = mean
        values[f"rolling_std_{window}"] = std
    for col in feature_cols:
        if col in values:
            continue
        raw = cov_row[col] if col in cov_row.index else np.nan
        values[col] = float(raw) if pd.notna(raw) else np.nan
    return np.asarray([values[col] for col in feature_cols], dtype=np.float32)


def recursive_forecast_series(
    model,
    history_targets: np.ndarray,
    future_cov: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    """One-step recursive rollout over the forecast window (no future target leakage)."""
    preds: list[float] = []
    history = history_targets.astype(np.float64, copy=True)
    for _, cov_row in future_cov.iterrows():
        x = features_from_history(history, cov_row, feature_cols).reshape(1, -1)
        pred = float(model.predict(x)[0])
        pred = max(pred, 0.0)
        preds.append(pred)
        history = np.append(history, pred)
    return np.asarray(preds, dtype=np.float64)


def align_to_forecast_index(
    predictions: pd.DataFrame,
    forecast_index: pd.DataFrame,
) -> pd.DataFrame:
    index = forecast_index[["series_id", "timestamp"]].copy()
    pred = predictions.copy()
    index["timestamp"] = pd.to_datetime(index["timestamp"])
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    aligned = index.merge(pred, on=["series_id", "timestamp"], how="left", validate="one_to_one")
    if aligned["prediction"].isna().any():
        n_missing = int(aligned["prediction"].isna().sum())
        raise ValueError(f"Missing {n_missing} LightGBM predictions after index alignment.")
    aligned["prediction"] = np.clip(aligned["prediction"].to_numpy(dtype=np.float64), 0.0, None)
    aligned["timestamp"] = pd.to_datetime(aligned["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return aligned[["series_id", "timestamp", "prediction"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone LightGBM forecast pipeline.")
    parser.add_argument("--data-dir", type=Path, default=Path("../../data"))
    parser.add_argument("--input_dir", type=Path, default=None, help="Alias for --data-dir.")
    parser.add_argument("--output_file", type=Path, default=Path("predictions_lgb.csv"))
    parser.add_argument("--model-path", type=Path, default=Path("lgb_model.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_data_dir(args.input_dir or args.data_dir)
    print(f"using data_dir: {data_dir}", flush=True)

    train = pd.read_csv(data_dir / "train.csv")
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    train = add_time_features(train)

    covariate_cols = available_covariate_cols(train)
    feat_cols = feature_columns(covariate_cols)
    print(f"feature count={len(feat_cols)}", flush=True)

    print("building lag / rolling features...", flush=True)
    featured = add_lag_rolling_features(train)
    featured = featured.dropna(subset=LAG_COLS + ROLL_COLS + [TARGET_COL]).reset_index(drop=True)

    train_mask, val_mask = chronological_masks(featured["timestamp"], train_ratio=0.8)
    X = featured[feat_cols]
    y = featured[TARGET_COL].astype(np.float32)
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[val_mask], y.loc[val_mask]
    print(
        f"train rows={len(X_train)} val rows={len(X_val)} "
        f"cut after last 20% timestamps",
        flush=True,
    )

    print("fitting LGBMRegressor(objective='regression_l1')...", flush=True)
    model = fit_lightgbm(X_train, y_train, X_val, y_val)
    best_iter = getattr(model, "best_iteration_", None)
    print(f"best_iteration={best_iter}", flush=True)
    model.booster_.save_model(str(args.model_path))
    print(f"saved LightGBM model -> {args.model_path}", flush=True)

    val_pred = np.clip(model.predict(X_val), 0.0, None)
    val_mae = float(np.mean(np.abs(val_pred - y_val.to_numpy())))
    print(f"validation MAE (direct, not recursive)={val_mae:.6f}", flush=True)

    forecast_index = load_forecast_index(data_dir)
    cov_frame = load_future_covariates(data_dir)
    print(
        f"forecast rows={len(forecast_index)} "
        f"future_cov_source={'validation/test input' if cov_frame is not None else 'train.csv fallback'}",
        flush=True,
    )

    pred_parts: list[pd.DataFrame] = []
    n_series = forecast_index["series_id"].nunique()
    for i, (series_id, index_part) in enumerate(forecast_index.groupby("series_id", sort=False), start=1):
        t0 = index_part["timestamp"].min()
        hist = train.loc[
            train["series_id"].eq(series_id) & train["timestamp"].lt(t0)
        ].sort_values("timestamp")
        if hist.empty:
            raise ValueError(f"No history for series {series_id!r} before {t0}.")

        future_index = index_part[["series_id", "timestamp"]].copy()
        if cov_frame is not None:
            future_cov = cov_frame.merge(future_index, on=["series_id", "timestamp"], how="inner")
        else:
            future_cov = train.merge(future_index, on=["series_id", "timestamp"], how="inner")
        future_cov = add_time_features(future_cov).sort_values("timestamp")
        if len(future_cov) != len(future_index):
            raise ValueError(
                f"Future covariate rows ({len(future_cov)}) != forecast rows "
                f"({len(future_index)}) for series {series_id!r}."
            )

        yhat = recursive_forecast_series(
            model,
            hist[TARGET_COL].to_numpy(dtype=np.float64),
            future_cov,
            feat_cols,
        )
        part = future_index.copy()
        part["prediction"] = yhat
        pred_parts.append(part)
        if i == 1 or i % 16 == 0 or i == n_series:
            print(f"  forecasted {i}/{n_series} series", flush=True)

    predictions = align_to_forecast_index(pd.concat(pred_parts, ignore_index=True), forecast_index)
    pred = predictions["prediction"].to_numpy(dtype=np.float64)
    print(
        f"LGB predictions: n={len(predictions)} mean={pred.mean():.4f} "
        f"min={pred.min():.4f} max={pred.max():.4f} negatives={(pred < 0).sum()}",
        flush=True,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_file, index=False)
    print(f"wrote {len(predictions)} rows -> {args.output_file}", flush=True)


if __name__ == "__main__":
    main()
