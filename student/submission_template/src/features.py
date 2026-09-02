"""Shared feature column definitions and engineered feature helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COL = "target"

# Base covariates present in the HF dataset.
BASE_COVARIATE_COLS = [
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

# Extra engineered covariates (cyclic harmonics + interaction gap).
ENGINEERED_COVARIATE_COLS = [
    "sin_hour",
    "cos_hour",
    "sin_dow",
    "cos_dow",
    "hour_sin_2",
    "hour_cos_2",
    "demand_staffing_gap",
]

COVARIATE_COLS = BASE_COVARIATE_COLS + ENGINEERED_COVARIATE_COLS
FEATURE_COLS = COVARIATE_COLS + [TARGET_COL]
TARGET_IDX = FEATURE_COLS.index(TARGET_COL)
NUM_FEATURES = len(FEATURE_COLS)
NUM_COVARIATES = len(COVARIATE_COLS)


def enrich_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclic time encodings and demand-staffing gap.

    - Recomputes hour/day-of-week sin/cos from timestamp for continuity.
    - Adds alias columns sin_hour/cos_hour/sin_dow/cos_dow.
    - Adds 2nd harmonic hour encodings and demand_staffing_gap.
    """
    out = frame.copy()
    if "timestamp" not in out.columns:
        raise ValueError("enrich_features requires a `timestamp` column.")

    timestamps = pd.to_datetime(out["timestamp"])
    hour = timestamps.dt.hour.to_numpy(dtype=np.float64)
    dow = timestamps.dt.dayofweek.to_numpy(dtype=np.float64)

    # Primary cyclic encodings (also overwrite dataset-provided values for consistency).
    hour_sin = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32)
    hour_cos = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32)
    dow_sin = np.sin(2.0 * np.pi * dow / 7.0).astype(np.float32)
    dow_cos = np.cos(2.0 * np.pi * dow / 7.0).astype(np.float32)

    out["hour_sin"] = hour_sin
    out["hour_cos"] = hour_cos
    out["dow_sin"] = dow_sin
    out["dow_cos"] = dow_cos

    # Explicit aliases requested for clearer cyclic continuity.
    out["sin_hour"] = hour_sin
    out["cos_hour"] = hour_cos
    out["sin_dow"] = dow_sin
    out["cos_dow"] = dow_cos

    # Second harmonic (two cycles per day) for sharper daily patterns.
    out["hour_sin_2"] = np.sin(4.0 * np.pi * hour / 24.0).astype(np.float32)
    out["hour_cos_2"] = np.cos(4.0 * np.pi * hour / 24.0).astype(np.float32)

    if "demand_forecast" in out.columns and "staffing_forecast" in out.columns:
        demand = pd.to_numeric(out["demand_forecast"], errors="coerce").fillna(0.0)
        staffing = pd.to_numeric(out["staffing_forecast"], errors="coerce").fillna(0.0)
        out["demand_staffing_gap"] = (demand - staffing).astype(np.float32)
    else:
        out["demand_staffing_gap"] = np.float32(0.0)

    return out
