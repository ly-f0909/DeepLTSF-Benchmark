"""Standalone LightGBM hourly forecasting pipeline (independent of TiDE).

Feature design avoids mid/long-horizon autoregressive drift:
  - future-known covariate interactions and calendar features
  - static per-series target encodings
  - long lag anchors (lag_336 / lag_504) looked up from real history only

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
LAGS = (1, 2, 24, 48, 168, 336, 504)
ROLL_WINDOWS = (24, 168)
EPS = 1e-5

KNOWN_FUTURE_COVARIATES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
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

TIME_FEATURE_COLS = ["hour", "dayofweek", "is_weekend", "hour_dow_cat"]
INTERACT_COLS = ["demand_staffing_gap", "demand_staffing_ratio"]
STATIC_COLS = ["series_mean", "series_median", "series_std", "series_q75"]
LAG_COLS = [f"lag_{lag}" for lag in LAGS]
ROLL_COLS = [f"rolling_{stat}_{window}" for window in ROLL_WINDOWS for stat in ("mean", "std")]
CATEGORICAL_COLS = ["hour_dow_cat"]


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


def add_deterministic_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Calendar + future-known demand/staffing interactions. No target used."""
    out = frame.copy()
    timestamps = pd.to_datetime(out["timestamp"])
    hour = timestamps.dt.hour.astype(np.int16)
    dayofweek = timestamps.dt.dayofweek.astype(np.int16)
    out["hour"] = hour
    out["dayofweek"] = dayofweek
    out["is_weekend"] = (dayofweek >= 5).astype(np.int8)
    out["hour_dow_cat"] = (hour.astype(np.int32) * 7 + dayofweek.astype(np.int32)).astype(
        np.int32
    )

    if "demand_forecast" in out.columns and "staffing_forecast" in out.columns:
        demand = pd.to_numeric(out["demand_forecast"], errors="coerce")
        staffing = pd.to_numeric(out["staffing_forecast"], errors="coerce")
        out["demand_staffing_gap"] = (demand - staffing).astype(np.float32)
        out["demand_staffing_ratio"] = (demand / (staffing + EPS)).astype(np.float32)
    else:
        out["demand_staffing_gap"] = np.float32(0.0)
        out["demand_staffing_ratio"] = np.float32(0.0)
    return out


def compute_series_target_encoding(history: pd.DataFrame) -> pd.DataFrame:
    """Static per-series target stats from historical rows only."""
    target = pd.to_numeric(history[TARGET_COL], errors="coerce")
    stats = (
        history.assign(_target=target)
        .groupby("series_id", sort=False)["_target"]
        .agg(
            series_mean="mean",
            series_median="median",
            series_std="std",
            series_q75=lambda s: s.quantile(0.75),
        )
    )
    return stats


def add_static_encoding(frame: pd.DataFrame, series_stats: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    mapped = series_stats.reindex(out["series_id"].to_numpy())
    for col in STATIC_COLS:
        out[col] = mapped[col].to_numpy(dtype=np.float32)
    return out


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


def attach_history_only_lags(
    future: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Look up lags / rolling stats from real history only.

    Forecast timestamps never append predicted targets, so lag_336 / lag_504
    stay on observed values throughout a 336-step horizon.
    """
    out = future.copy()
    hist = history.sort_values("timestamp")
    hist_ts = pd.to_datetime(hist["timestamp"])
    hist_vals = pd.to_numeric(hist[TARGET_COL], errors="coerce")
    hist_lookup = pd.Series(hist_vals.to_numpy(dtype=np.float64), index=hist_ts)

    query_ts = pd.to_datetime(out["timestamp"])
    for lag in LAGS:
        out[f"lag_{lag}"] = (query_ts - pd.Timedelta(hours=lag)).map(hist_lookup)

    hist_time_np = hist_ts.to_numpy(dtype="datetime64[ns]")
    hist_val_np = hist_vals.to_numpy(dtype=np.float64)
    query_np = query_ts.to_numpy(dtype="datetime64[ns]")
    right = np.searchsorted(hist_time_np, query_np, side="left")
    for window in ROLL_WINDOWS:
        left = np.searchsorted(
            hist_time_np,
            query_np - np.timedelta64(window, "h"),
            side="left",
        )
        means = np.full(len(out), np.nan, dtype=np.float64)
        stds = np.full(len(out), np.nan, dtype=np.float64)
        for i, (lo, hi) in enumerate(zip(left, right)):
            if hi - lo < window:
                continue
            block = hist_val_np[lo:hi]
            means[i] = float(np.nanmean(block))
            stds[i] = float(np.nanstd(block, ddof=1)) if block.size > 1 else 0.0
        out[f"rolling_mean_{window}"] = means
        out[f"rolling_std_{window}"] = stds
    return out


def available_covariate_cols(frame: pd.DataFrame) -> list[str]:
    reserved = set(TIME_FEATURE_COLS + INTERACT_COLS + STATIC_COLS + LAG_COLS + ROLL_COLS)
    return [col for col in KNOWN_FUTURE_COVARIATES if col in frame.columns and col not in reserved]


def feature_columns(covariate_cols: list[str]) -> list[str]:
    return TIME_FEATURE_COLS + INTERACT_COLS + STATIC_COLS + LAG_COLS + ROLL_COLS + covariate_cols


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
    cat_cols = [col for col in CATEGORICAL_COLS if col in X_train.columns]
    fit_kwargs = {
        "eval_set": [(X_val, y_val)],
        "eval_metric": "l1",
        "categorical_feature": cat_cols,
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
    train = add_deterministic_features(train)

    series_stats = compute_series_target_encoding(train)
    train = add_static_encoding(train, series_stats)
    print(
        f"series encodings: n={len(series_stats)} "
        f"mean-of-mean={series_stats['series_mean'].mean():.4f} "
        f"mean-of-median={series_stats['series_median'].mean():.4f}",
        flush=True,
    )

    covariate_cols = available_covariate_cols(train)
    feat_cols = feature_columns(covariate_cols)
    print(f"feature count={len(feat_cols)} categorical={CATEGORICAL_COLS}", flush=True)

    print("building lag / rolling / long-anchor features...", flush=True)
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
    print(f"validation MAE (direct, history-anchored)={val_mae:.6f}", flush=True)

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
        if len(future_cov) != len(future_index):
            raise ValueError(
                f"Future covariate rows ({len(future_cov)}) != forecast rows "
                f"({len(future_index)}) for series {series_id!r}."
            )

        future_feat = add_deterministic_features(future_cov)
        future_feat = add_static_encoding(future_feat, series_stats)
        future_feat = attach_history_only_lags(future_feat, hist)
        future_feat = future_feat.sort_values("timestamp")

        yhat = np.clip(model.predict(future_feat[feat_cols]), 0.0, None)
        part = future_feat[["series_id", "timestamp"]].copy()
        part["prediction"] = yhat
        pred_parts.append(part)
        if i == 1 or i % 16 == 0 or i == n_series:
            print(f"  forecasted {i}/{n_series} series (direct, no recursion)", flush=True)

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
