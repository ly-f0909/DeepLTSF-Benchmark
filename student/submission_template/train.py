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

from src.data_io import resolve_train_csv
from src.dataset import TiDEDataset, chronological_split
from src.features import COVARIATE_COLS, FEATURE_COLS, NUM_COVARIATES, NUM_FEATURES, TARGET_COL
from src.model import ForecastModel


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    """Cloud/server friendly device selection."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
) -> tuple[float, dict[str, float]]:
    model.eval()
    total, n = 0.0, 0
    pred_all: list[np.ndarray] = []
    true_all: list[np.ndarray] = []
    for x_past, x_future, y_target in loader:
        x_past = x_past.to(device)
        x_future = x_future.to(device)
        y_target = y_target.to(device)
        pred = model(x_past, x_future)
        loss = loss_fn(pred, y_target)
        total += loss.item() * x_past.size(0)
        n += x_past.size(0)
        pred_all.append(pred.detach().cpu().numpy().reshape(-1))
        true_all.append(y_target.detach().cpu().numpy().reshape(-1))

    pred_vec = np.concatenate(pred_all)
    true_vec = np.concatenate(true_all)
    stats = {
        "pred_mean": float(pred_vec.mean()),
        "pred_min": float(pred_vec.min()),
        "pred_max": float(pred_vec.max()),
        "true_mean": float(true_vec.mean()),
        "true_min": float(true_vec.min()),
        "true_max": float(true_vec.max()),
    }
    return total / max(n, 1), stats


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
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing train.csv (auto-detected if omitted).",
    )
    parser.add_argument(
        "--train_csv",
        type=Path,
        default=None,
        help="Path to train.csv (overrides --data-dir).",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint.pt"))
    parser.add_argument("--seq_len", type=int, default=336, choices=[168, 336, 504])
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--num_decoder_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--loss", choices=["smooth_l1", "l1", "huber"], default="smooth_l1")
    parser.add_argument("--use_revin", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    set_seed(42)
    device = resolve_device()
    print(f"--> training device: {device}")

    train_csv = resolve_train_csv(args.train_csv, args.data_dir)
    print(f"using train_csv: {train_csv}")
    raw = pd.read_csv(train_csv)
    train_df, val_df = chronological_split(raw, 0.8)
    print(f"train rows={len(train_df)} val rows={len(val_df)}")

    print("materializing training windows into memory tensors...", flush=True)
    train_ds = TiDEDataset(train_df, args.seq_len, args.pred_len)
    val_ds = TiDEDataset(val_df, args.seq_len, args.pred_len)
    print(f"train windows={len(train_ds)} val windows={len(val_ds)}")

    loader_kwargs: dict = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

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
            x_past = x_past.to(device, non_blocking=True)
            x_future = x_future.to(device, non_blocking=True)
            y_target = y_target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x_past, x_future)
            loss = loss_fn(pred, y_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            running += loss.item() * x_past.size(0)
            n += x_past.size(0)

        scheduler.step()
        train_loss = running / max(n, 1)
        val_loss, val_stats = evaluate(model, val_loader, loss_fn, device)
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"epoch {epoch:02d}  lr={current_lr:.2e}  "
            f"train_{args.loss}={train_loss:.6f}  val_{args.loss}={val_loss:.6f}",
            flush=True,
        )
        print(
            f"  val scale check: pred mean={val_stats['pred_mean']:.3f} "
            f"[{val_stats['pred_min']:.3f}, {val_stats['pred_max']:.3f}]  "
            f"true mean={val_stats['true_mean']:.3f} "
            f"[{val_stats['true_min']:.3f}, {val_stats['true_max']:.3f}]",
            flush=True,
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"state_dict": model.state_dict(), "config": config}, args.checkpoint)
            print(f"  saved best -> {args.checkpoint} (val_{args.loss}={best_val:.6f})", flush=True)

    print(f"done. best_val_{args.loss}={best_val:.6f}")


if __name__ == "__main__":
    main()
