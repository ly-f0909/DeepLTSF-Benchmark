"""Memory-efficient TiDE dataset with optional multi-step rollout targets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features import COVARIATE_COLS, FEATURE_COLS, TARGET_COL, enrich_features


class TiDEDataset(Dataset):
    """
    Index-based sliding windows over per-series float32 tensors.

    When rollout_steps=1:
      x_past:   [seq_len, num_features]
      x_future: [pred_len, num_covariates]
      y_target: [pred_len]

    When rollout_steps>1:
      x_future: [pred_len * rollout_steps, num_covariates]
      y_target: [pred_len * rollout_steps]
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        seq_len: int,
        pred_len: int,
        feature_cols: list[str] | None = None,
        covariate_cols: list[str] | None = None,
        target_col: str = TARGET_COL,
        rollout_steps: int = 1,
    ) -> None:
        feature_cols = feature_cols or FEATURE_COLS
        covariate_cols = covariate_cols or COVARIATE_COLS
        if target_col in covariate_cols:
            raise ValueError("target_col must not appear in covariate_cols (label leakage).")
        if rollout_steps < 1:
            raise ValueError("rollout_steps must be >= 1")

        frame = enrich_features(frame)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.rollout_steps = rollout_steps
        self.horizon = pred_len * rollout_steps
        self.num_covariates = len(covariate_cols)
        self.target_col = target_col

        self.series_features: list[torch.Tensor] = []
        self.series_covariates: list[torch.Tensor] = []
        self.series_targets: list[torch.Tensor] = []
        self.index: list[tuple[int, int]] = []

        series_idx = 0
        for _, group in frame.groupby("series_id", sort=False):
            sorted_group = group.sort_values("timestamp")
            features = torch.from_numpy(
                sorted_group[feature_cols].astype(np.float32).fillna(0.0).to_numpy()
            )
            covariates = torch.from_numpy(
                sorted_group[covariate_cols].astype(np.float32).fillna(0.0).to_numpy()
            )
            target = torch.from_numpy(
                sorted_group[target_col].astype(np.float32).fillna(0.0).to_numpy()
            )

            n = int(target.shape[0])
            n_windows = n - seq_len - self.horizon + 1
            if n_windows <= 0:
                continue

            self.series_features.append(features)
            self.series_covariates.append(covariates)
            self.series_targets.append(target)
            for start in range(n_windows):
                self.index.append((series_idx, start))
            series_idx += 1

        if not self.index:
            raise ValueError(
                "No training windows could be created; check seq_len/pred_len/rollout_steps."
            )

        bytes_est = sum(t.numel() for t in self.series_features) * 4
        print(
            f"  dataset series={len(self.series_features)} windows={len(self.index)} "
            f"rollout_steps={rollout_steps} feature_bytes≈{bytes_est / 1e6:.1f}MB",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        series_idx, start = self.index[idx]
        end = start + self.seq_len
        fut_end = end + self.horizon
        x_past = self.series_features[series_idx][start:end]
        x_future = self.series_covariates[series_idx][end:fut_end]
        y_target = self.series_targets[series_idx][end:fut_end]
        return x_past, x_future, y_target


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
