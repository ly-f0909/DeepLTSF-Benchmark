"""Train TiDE with engineered features, longer lookback, warmup LR, and multi-seed support."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.data_io import resolve_train_csv
from src.dataset import TiDEDataset, chronological_split
from src.features import COVARIATE_COLS, FEATURE_COLS, NUM_COVARIATES, NUM_FEATURES, TARGET_COL
from src.model import ForecastModel

DEFAULT_SEEDS = (42, 2024, 777)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_seeds(raw: str | None) -> list[int]:
    if raw is None or raw.strip() == "":
        return [42]
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("No valid seeds provided.")
    return seeds


def build_loss_fn(loss_name: str) -> nn.Module:
    if loss_name == "l1":
        return nn.L1Loss()
    if loss_name == "smooth_l1":
        return nn.SmoothL1Loss()
    if loss_name == "huber":
        return nn.HuberLoss()
    raise ValueError(f"Unsupported loss: {loss_name!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str,
    epochs: int,
    warmup_epochs: int,
    min_lr: float,
    lr: float,
    restart_period: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    warmup_epochs = max(0, min(warmup_epochs, max(epochs - 1, 0)))
    if scheduler_name == "cosine_warm_restarts":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(restart_period, 1),
            T_mult=2,
            eta_min=min_lr,
        )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(epochs - warmup_epochs, 1),
        eta_min=min_lr,
    )
    if warmup_epochs <= 0:
        return cosine

    warmup = LinearLR(
        optimizer,
        start_factor=min(1.0, min_lr / max(lr, 1e-12)),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


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


def build_config(args: argparse.Namespace, seed: int) -> dict:
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
        "seed": seed,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
    }


def checkpoint_path_for_seed(base: Path, seed: int, multi_seed: bool) -> Path:
    if not multi_seed:
        return base
    return base.with_name(f"{base.stem}_seed{seed}{base.suffix}")


def train_one_seed(
    args: argparse.Namespace,
    *,
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    checkpoint_path: Path,
) -> float:
    set_seed(seed)
    config = build_config(args, seed)
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
    scheduler = build_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        lr=args.lr,
        restart_period=args.restart_period,
    )

    best_val = float("inf")
    print(f"\n===== training seed={seed} -> {checkpoint_path} =====", flush=True)
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
            f"[seed {seed}] epoch {epoch:02d}  lr={current_lr:.2e}  "
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
            torch.save({"state_dict": model.state_dict(), "config": config}, checkpoint_path)
            print(
                f"  saved best -> {checkpoint_path} (val_{args.loss}={best_val:.6f})",
                flush=True,
            )

    print(f"[seed {seed}] done. best_val_{args.loss}={best_val:.6f}", flush=True)
    return best_val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--train_csv", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint.pt"))
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated seeds. Example: 42,2024,777 for seed averaging.",
    )
    parser.add_argument("--seq_len", type=int, default=504, choices=[168, 336, 504, 672])
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--num_decoder_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument(
        "--scheduler",
        choices=["cosine_warmup", "cosine_warm_restarts"],
        default="cosine_warmup",
        help="cosine_warmup: Linear warmup then CosineAnnealingLR; "
        "cosine_warm_restarts: CosineAnnealingWarmRestarts.",
    )
    parser.add_argument("--restart_period", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--loss", choices=["smooth_l1", "l1", "huber"], default="smooth_l1")
    parser.add_argument("--use_revin", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    multi_seed = len(seeds) > 1
    device = resolve_device()
    print(f"--> training device: {device}")
    print(f"--> seeds: {seeds}")
    print(f"--> feature dims: num_features={NUM_FEATURES} num_covariates={NUM_COVARIATES}")

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

    results: list[tuple[int, float, Path]] = []
    for seed in seeds:
        ckpt = checkpoint_path_for_seed(args.checkpoint, seed, multi_seed)
        best_val = train_one_seed(
            args,
            seed=seed,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            checkpoint_path=ckpt,
        )
        results.append((seed, best_val, ckpt))

    print("\n===== seed summary =====")
    for seed, best_val, ckpt in results:
        print(f"seed={seed} best_val_{args.loss}={best_val:.6f} checkpoint={ckpt}")
    if multi_seed:
        print(
            "ensemble predict example:\n"
            f"  python predict.py --checkpoints "
            + ",".join(str(path) for _, _, path in results)
            + " --output_file predictions.csv"
        )


if __name__ == "__main__":
    main()
