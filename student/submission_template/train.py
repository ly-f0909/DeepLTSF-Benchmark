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
from torch.utils.data import DataLoader

from src.dataset import TiDEDataset, chronological_split
from src.features import COVARIATE_COLS, FEATURE_COLS, NUM_COVARIATES, NUM_FEATURES, TARGET_COL
from src.model import ForecastModel


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loss_fn(loss_name: str) -> nn.Module:
    if loss_name == "l1":
        return nn.L1Loss()
    if loss_name == "smooth_l1":
        return nn.SmoothL1Loss()
    if loss_name == "huber":
        return nn.HuberLoss()
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
    for x_past, x_future, y_target in loader:
        x_past = x_past.to(device)
        x_future = x_future.to(device)
        y_target = y_target.to(device)
        pred = model(x_past, x_future)
        loss = loss_fn(pred, y_target)
        total += loss.item() * x_past.size(0)
        n += x_past.size(0)
    return total / max(n, 1)


def build_config(args: argparse.Namespace) -> dict:
    return {
        "model_type": "tide",
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "num_features": NUM_FEATURES,
        "num_covariates": NUM_COVARIATES,
        "target_idx": FEATURE_COLS.index(TARGET_COL),
        "hidden_dim": args.hidden_dim,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "dropout": args.dropout,
        "use_revin": args.use_revin,
        "covariate_cols": COVARIATE_COLS,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "loss": args.loss,
        "weight_decay": args.weight_decay,
        "lr": args.lr,
        "min_lr": args.min_lr,
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
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--loss", choices=["smooth_l1", "l1", "huber"], default="smooth_l1")
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

    print("materializing training windows into memory tensors...", flush=True)
    train_ds = TiDEDataset(train_df, args.seq_len, args.pred_len)
    val_ds = TiDEDataset(val_df, args.seq_len, args.pred_len)
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
        num_features=config["num_features"],
        num_covariates=config["num_covariates"],
        target_idx=config["target_idx"],
        hidden_dim=config["hidden_dim"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        dropout=config["dropout"],
        use_revin=config["use_revin"],
    ).to(device)

    loss_fn = build_loss_fn(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for x_past, x_future, y_target in train_loader:
            x_past = x_past.to(device)
            x_future = x_future.to(device)
            y_target = y_target.to(device)

            optimizer.zero_grad()
            pred = model(x_past, x_future)
            loss = loss_fn(pred, y_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            running += loss.item() * x_past.size(0)
            n += x_past.size(0)

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
