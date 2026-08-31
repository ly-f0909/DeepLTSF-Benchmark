"""Shared feature column definitions for training and inference."""

from __future__ import annotations

TARGET_COL = "target"
COVARIATE_COLS = [
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
FEATURE_COLS = COVARIATE_COLS + [TARGET_COL]
TARGET_IDX = FEATURE_COLS.index(TARGET_COL)
NUM_FEATURES = len(FEATURE_COLS)
NUM_COVARIATES = len(COVARIATE_COLS)
