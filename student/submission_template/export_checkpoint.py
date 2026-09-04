"""Pack TiDE weights and a LightGBM booster into one official checkpoint.pt.

Example:
  python export_checkpoint.py \\
    --tide checkpoint_tide.pt \\
    --lgb lgb_model.txt \\
    --output_file checkpoint.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from predict import build_model_from_checkpoint, load_checkpoint_dict, resolve_device
from src.lgb_infer import infer_feature_columns


def _import_lightgbm():
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit("lightgbm is required. Install with: pip install lightgbm") from exc
    return lgb


def load_tide_bundle(path: Path, device: torch.device) -> tuple[dict, dict]:
    raw = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported TiDE checkpoint: {path}")
    if "tide_state_dict" in raw:
        return raw["tide_state_dict"], raw.get("tide_config", {})
    loaded = load_checkpoint_dict(path, device)
    return loaded["state_dict"], loaded.get("config", {})


def load_lgb_booster(path: Path):
    lgb = _import_lightgbm()
    suffix = path.suffix.lower()
    if suffix in {".txt", ".model"}:
        return lgb.Booster(model_file=str(path))
    if suffix in {".pkl", ".pickle", ".joblib"}:
        import pickle

        with path.open("rb") as handle:
            obj = pickle.load(handle)
        if hasattr(obj, "booster_"):
            return obj.booster_
        if hasattr(obj, "predict"):
            return obj
        raise ValueError(f"Pickle at {path} is not a LightGBM model.")
    # Fall back: try booster text, then pickle.
    try:
        return lgb.Booster(model_file=str(path))
    except Exception:
        import pickle

        with path.open("rb") as handle:
            obj = pickle.load(handle)
        return obj.booster_ if hasattr(obj, "booster_") else obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a packed official checkpoint.pt.")
    parser.add_argument("--tide", type=Path, default=Path("checkpoint_tide.pt"))
    parser.add_argument("--lgb", type=Path, default=Path("lgb_model.txt"))
    parser.add_argument("--output_file", type=Path, default=Path("checkpoint.pt"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional train.csv directory used to record LightGBM feature columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.tide.exists():
        raise FileNotFoundError(f"Missing TiDE checkpoint: {args.tide}")
    if not args.lgb.exists():
        raise FileNotFoundError(f"Missing LightGBM model: {args.lgb}")

    device = resolve_device()
    state_dict, config = load_tide_bundle(args.tide, device)
    model = build_model_from_checkpoint({"config": config, "state_dict": state_dict})
    model.load_state_dict(state_dict)
    model.eval()

    lgb_booster = load_lgb_booster(args.lgb)

    lgb_feature_cols = None
    if args.data_dir is not None:
        train_csv = args.data_dir / "train.csv"
        if train_csv.exists():
            import pandas as pd

            train = pd.read_csv(train_csv)
            train["timestamp"] = pd.to_datetime(train["timestamp"])
            lgb_feature_cols = infer_feature_columns(train)

    payload = {
        "tide_state_dict": model.state_dict(),
        "tide_config": config,
        "lgb_model": lgb_booster,
        "lgb_feature_cols": lgb_feature_cols,
        "blend_weights": {"tide": 0.7, "lgb": 0.3},
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_file)
    print(
        f"wrote packed checkpoint -> {args.output_file} "
        f"(tide_config keys={sorted(config.keys())}, "
        f"lgb_feature_cols={len(lgb_feature_cols) if lgb_feature_cols else 'infer-at-runtime'})",
        flush=True,
    )


if __name__ == "__main__":
    main()
