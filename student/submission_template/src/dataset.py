"""Pre-materialized TiDE dataset with engineered features and in-memory tensors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features import COVARIATE_COLS, FEATURE_COLS, TARGET_COL, enrich_features


def _build_series_windows(
    features: np.ndarray,
    covariates: np.ndarray,
    target: np.ndarray,
    seq_len: int,
    pred_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build aligned windows for one series.

    For anchor time t (= start + seq_len):
      x_past   = features[t - seq_len : t]
      x_future = covariates[t : t + pred_len]   (known future covariates only, NO target)
      y_target = target[t : t + pred_len]
    """
    n = target.shape[0]
    n_windows = n - seq_len - pred_len + 1
    if n_windows <= 0:
        return (
            np.empty((0, seq_len, features.shape[1]), dtype=np.float32),
            np.empty((0, pred_len, covariates.shape[1]), dtype=np.float32),
            np.empty((0, pred_len), dtype=np.float32),
        )

    starts = np.arange(n_windows, dtype=np.int64)
    t = starts + seq_len
    past_idx = starts[:, None] + np.arange(seq_len, dtype=np.int64)[None, :]
    future_idx = t[:, None] + np.arange(pred_len, dtype=np.int64)[None, :]

    x_past = features[past_idx]
    x_future = covariates[future_idx]
    y_target = target[future_idx]

    return (
        np.ascontiguousarray(x_past, dtype=np.float32),
        np.ascontiguousarray(x_future, dtype=np.float32),
        np.ascontiguousarray(y_target, dtype=np.float32),
    )


class TiDEDataset(Dataset):
    """
    Materialize all sliding windows as stacked float32 tensors at init time.

    Each sample returns:
      x_past:   [seq_len, num_features]
      x_future: [pred_len, num_covariates]
      y_target: [pred_len]
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        seq_len: int,
        pred_len: int,
        feature_cols: list[str] | None = None,
        covariate_cols: list[str] | None = None,
        target_col: str = TARGET_COL,
    ) -> None:
        feature_cols = feature_cols or FEATURE_COLS
        covariate_cols = covariate_cols or COVARIATE_COLS

        if target_col in covariate_cols:
            raise ValueError("target_col must not appear in covariate_cols (label leakage).")

        frame = enrich_features(frame)

        x_past_parts: list[np.ndarray] = []
        x_future_parts: list[np.ndarray] = []
        y_target_parts: list[np.ndarray] = []

        for _, group in frame.groupby("series_id", sort=False):
            sorted_group = group.sort_values("timestamp")
            features = sorted_group[feature_cols].astype(np.float32).fillna(0.0).to_numpy()
            covariates = sorted_group[covariate_cols].astype(np.float32).fillna(0.0).to_numpy()
            target = sorted_group[target_col].astype(np.float32).fillna(0.0).to_numpy()

            x_past, x_future, y_target = _build_series_windows(
                features, covariates, target, seq_len, pred_len
            )
            if x_past.shape[0] == 0:
                continue

            x_past_parts.append(x_past)
            x_future_parts.append(x_future)
            y_target_parts.append(y_target)

        if not x_past_parts:
            raise ValueError(
                "No training windows could be created; check seq_len/pred_len and data size."
            )

        self.x_past = torch.from_numpy(np.concatenate(x_past_parts, axis=0))
        self.x_future = torch.from_numpy(np.concatenate(x_future_parts, axis=0))
        self.y_target = torch.from_numpy(np.concatenate(y_target_parts, axis=0))
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_covariates = len(covariate_cols)
        self._validate_tensors()

    def _validate_tensors(self) -> None:
        if self.x_future.shape[-1] != self.num_covariates:
            raise ValueError("x_future must contain covariates only.")
        if self.x_past.shape[1] != self.seq_len:
            raise ValueError("x_past seq_len mismatch.")
        if self.x_future.shape[1] != self.pred_len:
            raise ValueError("x_future pred_len mismatch.")
        if self.y_target.shape[1] != self.pred_len:
            raise ValueError("y_target pred_len mismatch.")

    def __len__(self) -> int:
        return self.x_past.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_past[idx], self.x_future[idx], self.y_target[idx]


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by global timestamp order (no shuffle)."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    times = np.sort(df["timestamp"].unique())
    cut = times[int(len(times) * train_ratio) - 1]
    train_df = df[df["timestamp"] <= cut]
    val_df = df[df["timestamp"] > cut]
    return train_df, val_df
