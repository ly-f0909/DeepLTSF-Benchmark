"""LightGBM feature construction and recursive inference (submission-safe)."""

from __future__ import annotations

import numpy as np
import pandas as pd

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


def feature_columns(covariate_cols: list[str]) -> list[str]:
    return TIME_FEATURE_COLS + LAG_COLS + ROLL_COLS + covariate_cols


def infer_feature_columns(train_frame: pd.DataFrame) -> list[str]:
    featured = add_time_features(train_frame)
    return feature_columns(available_covariate_cols(featured))


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


class LGBPredictor:
    """Thin predict() wrapper around a LightGBM Booster or sklearn-style model."""

    def __init__(self, raw) -> None:
        self.raw = raw

    def predict(self, x: np.ndarray) -> np.ndarray:
        model = self.raw
        if hasattr(model, "booster_") and not hasattr(model, "predict"):
            model = model.booster_
        if hasattr(model, "predict"):
            out = model.predict(x)
        else:
            raise TypeError(f"Unsupported LightGBM object: {type(self.raw)!r}")
        return np.asarray(out, dtype=np.float64)


def wrap_lgb_model(raw) -> LGBPredictor:
    if isinstance(raw, LGBPredictor):
        return raw
    if isinstance(raw, str):
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise SystemExit("lightgbm is required. Install with: pip install lightgbm") from exc
        return LGBPredictor(lgb.Booster(model_str=raw))
    return LGBPredictor(raw)


def recursive_forecast_series(
    model,
    history_targets: np.ndarray,
    future_cov: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    predictor = wrap_lgb_model(model)
    preds: list[float] = []
    history = history_targets.astype(np.float64, copy=True)
    for _, cov_row in future_cov.iterrows():
        x = features_from_history(history, cov_row, feature_cols).reshape(1, -1)
        pred = float(predictor.predict(x)[0])
        pred = max(pred, 0.0)
        preds.append(pred)
        history = np.append(history, pred)
    return np.asarray(preds, dtype=np.float64)


def predict_lightgbm(
    model,
    forecast_index: pd.DataFrame,
    train_frame: pd.DataFrame,
    cov_frame: pd.DataFrame | None,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    train = add_time_features(train_frame)
    feat_cols = feature_cols or infer_feature_columns(train)
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
            print(f"  [LGB] forecasted {i}/{n_series} series", flush=True)

    pred_long = pd.concat(pred_parts, ignore_index=True)
    aligned = forecast_index[["series_id", "timestamp"]].merge(
        pred_long,
        on=["series_id", "timestamp"],
        how="left",
        validate="one_to_one",
    )
    if aligned["prediction"].isna().any():
        n_missing = int(aligned["prediction"].isna().sum())
        raise ValueError(f"Missing {n_missing} LightGBM predictions after index alignment.")
    aligned["prediction"] = np.clip(aligned["prediction"].to_numpy(dtype=np.float64), 0.0, None)
    return aligned
