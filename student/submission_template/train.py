"""Train TiDE on train.csv with pre-materialized tensor windows."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from src.model import ForecastModel

TARGET_COL = "target"
COVARIATE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "trend",
    "workload_intensity", "demand_forecast", "staffing_forecast",
    "upstream_quality_forecast", "promotion_intensity", "shock_risk",
    "maintenance_known", "unit_reliability_forecast", "queue_pressure_forecast",
    "network_pressure_forecast", "event_load_forecast",
    "service_irregularity_risk_forecast", "throughput_disruption_risk_forecast",
    "nominal_capacity", "zone_sin", "zone_cos",
]
FEATURE_COLS = COVARIATE_COLS + [TARGET_COL]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SlidingWindowDataset(Dataset):
    """Pre-materialized float32 tensor windows; __getitem__ is pure tensor indexing."""

    def __init__(
        self,
        frame: pd.DataFrame,
        seq_len: int,
        pred_len: int,
        covariate_cols: list[str],
        target_col: str,
    ) -> None:
        past_targets: list[torch.Tensor] = []
        past_covs: list[torch.Tensor] = []
        future_covs: list[torch.Tensor] = []
        future_targets: list[torch.Tensor] = []

        for _, group in frame.groupby("series_id", sort=False):
            sorted_group = group.sort_values("timestamp")
            cov = torch.from_numpy(
                sorted_group[covariate_cols].astype(np.float32).fillna(0.0).to_numpy()
            )
            target = torch.from_numpy(
                sorted_group[[target_col]].astype(np.float32).fillna(0.0).to_numpy()
            ).squeeze(-1)

            n = target.shape[0]
            for start in range(0, n - seq_len - pred_len + 1):
                end = start + seq_len
                fut_end = end + pred_len
                past_targets.append(target[start:end])
                past_covs.append(cov[start:end])
                future_covs.append(cov[end:fut_end])
                future_targets.append(target[end:fut_end])

        if not past_targets:
            raise ValueError("No training windows could be created; check seq_len/pred_len and data size.")

        self.past_target = torch.stack(past_targets)
        self.past_cov = torch.stack(past_covs)
        self.future_cov = torch.stack(future_covs)
        self.future_target = torch.stack(future_targets)

    def __len__(self) -> int:
        return self.past_target.shape[0]

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.past_target[idx],
            self.past_cov[idx],
            self.future_cov[idx],
            self.future_target[idx],
        )


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by global timestamp order (no shuffle)."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    times = np.sort(df["timestamp"].unique())
    cut = times[int(len(times) * train_ratio) - 1]
    train_df = df[df["timestamp"] <= cut]
    val_df = df[df["timestamp"] > cut]
    return train_df, val_df


def build_loss_fn(loss_name: str) -> nn.Module:
    if loss_name == "l1":
        return nn.L1Loss()
    if loss_name == "smooth_l1":
        return nn.SmoothL1Loss()
    raise ValueError(f"Unsupported loss: {loss_name!r}")


@torch.no_grad()
def evaluate(
    model: ForecastModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total, n = 0.0, 0
    for past_target, past_cov, future_cov, future_target in loader:
        past_target = past_target.to(device)
        past_cov = past_cov.to(device)
        future_cov = future_cov.to(device)
        future_target = future_target.to(device)
        pred = model(past_target, past_cov, future_cov)
        loss = loss_fn(pred, future_target)
        total += loss.item() * past_target.size(0)
        n += past_target.size(0)
    return total / max(n, 1)


def build_config(args: argparse.Namespace) -> dict:
    return {
        "model_type": "tide",
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "num_covariates": len(COVARIATE_COLS),
        "hidden_dim": args.hidden_dim,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "dropout": args.dropout,
        "use_revin": args.use_revin,
        "covariate_cols": COVARIATE_COLS,
        "target_col": TARGET_COL,
        "feature_cols": FEATURE_COLS,
        "loss": args.loss,
        "weight_decay": args.weight_decay,
        "lr": args.lr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=Path, default=Path("../../data/train.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint.pt"))
    parser.add_argument("--seq_len", type=int, default=336, choices=[168, 336, 504])
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--num_decoder_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["smooth_l1", "l1"], default="smooth_l1")
    parser.add_argument("--use_revin", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"--> 当前训练使用的计算设备: {device}")

    raw = pd.read_csv(args.train_csv)
    train_df, val_df = chronological_split(raw, 0.8)
    print(f"train rows={len(train_df)} val rows={len(val_df)}")

    print("materializing training windows into memory tensors...")
    train_ds = SlidingWindowDataset(
        train_df, args.seq_len, args.pred_len, COVARIATE_COLS, TARGET_COL
    )
    val_ds = SlidingWindowDataset(
        val_df, args.seq_len, args.pred_len, COVARIATE_COLS, TARGET_COL
    )
    print(f"train windows={len(train_ds)} val windows={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    config = build_config(args)
    model = ForecastModel(
        seq_len=config["seq_len"],
        pred_len=config["pred_len"],
        num_covariates=config["num_covariates"],
        hidden_dim=config["hidden_dim"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        dropout=config["dropout"],
        use_revin=config["use_revin"],
    ).to(device)

    loss_fn = build_loss_fn(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for past_target, past_cov, future_cov, future_target in train_loader:
            past_target = past_target.to(device)
            past_cov = past_cov.to(device)
            future_cov = future_cov.to(device)
            future_target = future_target.to(device)

            optimizer.zero_grad()
            pred = model(past_target, past_cov, future_cov)
            loss = loss_fn(pred, future_target)
            loss.backward()
            optimizer.step()
            running += loss.item() * past_target.size(0)
            n += past_target.size(0)

        scheduler.step()
        train_loss = running / max(n, 1)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"epoch {epoch:02d}  lr={current_lr:.2e}  "
            f"train_{args.loss}={train_loss:.6f}  val_{args.loss}={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": config,
                },
                args.checkpoint,
            )
            print(f"  saved best -> {args.checkpoint} (val_{args.loss}={best_val:.6f})")

    print(f"done. best_val_{args.loss}={best_val:.6f}")


if __name__ == "__main__":
    main()
