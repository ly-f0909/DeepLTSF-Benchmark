"""Weighted average of TiDE and LightGBM forecasts.

Default blend: 0.7 * TiDE + 0.3 * LGB

Example:
  python ensemble.py \\
    --tide predictions_tide_best.csv \\
    --lgb predictions_lgb.csv \\
    --output_file predictions_final_ensemble.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLS = ("series_id", "timestamp", "prediction")
UPPER_SCALE = 1.1


def load_prediction_file(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} file: {path}")
    frame = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLS if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    out = frame[list(REQUIRED_COLS)].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["prediction"] = pd.to_numeric(out["prediction"], errors="coerce")
    if out["prediction"].isna().any():
        raise ValueError(f"{path} contains NaN predictions.")
    if out.duplicated(["series_id", "timestamp"]).any():
        raise ValueError(f"{path} has duplicate (series_id, timestamp) rows.")
    return out


def load_series_max_targets(train_csv: Path) -> pd.Series:
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train.csv: {train_csv}")
    train = pd.read_csv(train_csv, usecols=["series_id", "target"])
    train["target"] = pd.to_numeric(train["target"], errors="coerce")
    max_target = train.groupby("series_id", sort=False)["target"].max()
    if max_target.isna().any():
        raise ValueError("Some series have no valid historical target in train.csv.")
    return max_target


def print_array_stats(name: str, values: np.ndarray) -> None:
    print(
        f"{name}: n={values.size} mean={float(np.mean(values)):.6f} "
        f"std={float(np.std(values)):.6f} min={float(np.min(values)):.6f} "
        f"max={float(np.max(values)):.6f} nan={int(np.isnan(values).sum())} "
        f"inf={int(np.isinf(values).sum())}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend TiDE and LightGBM predictions.")
    parser.add_argument("--tide", type=Path, default=Path("predictions_tide_best.csv"))
    parser.add_argument("--lgb", type=Path, default=Path("predictions_lgb.csv"))
    parser.add_argument("--output_file", type=Path, default=Path("predictions_final_ensemble.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("../../data"))
    parser.add_argument("--tide-weight", type=float, default=0.7)
    parser.add_argument("--lgb-weight", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tide_w = float(args.tide_weight)
    lgb_w = float(args.lgb_weight)
    if tide_w < 0 or lgb_w < 0:
        raise ValueError("Weights must be non-negative.")
    weight_sum = tide_w + lgb_w
    if weight_sum <= 0:
        raise ValueError("At least one weight must be > 0.")
    tide_w /= weight_sum
    lgb_w /= weight_sum

    tide = load_prediction_file(args.tide, "TiDE")
    lgb = load_prediction_file(args.lgb, "LightGBM")

    merged = tide.merge(
        lgb,
        on=["series_id", "timestamp"],
        how="inner",
        suffixes=("_tide", "_lgb"),
        validate="one_to_one",
    )
    if len(merged) != len(tide) or len(merged) != len(lgb):
        raise ValueError(
            f"Key mismatch: tide={len(tide)} lgb={len(lgb)} inner_join={len(merged)}. "
            "Both files must share the same (series_id, timestamp) rows."
        )

    tide_vals = merged["prediction_tide"].to_numpy(dtype=np.float64)
    lgb_vals = merged["prediction_lgb"].to_numpy(dtype=np.float64)
    blended = tide_w * tide_vals + lgb_w * lgb_vals

    print("=== pre-fusion inputs ===")
    print_array_stats("TiDE", tide_vals)
    print_array_stats("LGB ", lgb_vals)
    print("=== after weighted fusion (before clip) ===")
    print_array_stats("blend", blended)

    train_csv = args.data_dir / "train.csv"
    max_target = load_series_max_targets(train_csv)
    upper = merged["series_id"].map(max_target) * UPPER_SCALE
    if upper.isna().any():
        missing = sorted(merged.loc[upper.isna(), "series_id"].astype(str).unique())
        raise ValueError(f"No historical max_target for series: {missing}")
    upper_vals = upper.to_numpy(dtype=np.float64)
    clipped = np.minimum(blended, upper_vals)
    clipped = np.maximum(clipped, 0.0)

    n_upper = int(np.sum(blended > upper_vals))
    n_lower = int(np.sum(blended < 0.0))
    print("=== after physical clip [0, max_target * 1.1] ===")
    print_array_stats("final", clipped)
    print(f"weights: tide={tide_w:.3f} lgb={lgb_w:.3f}")
    print(f"clipped_to_upper={n_upper} clipped_to_zero={n_lower}")
    print(f"series_caps={len(max_target)} train_csv={train_csv}")

    if np.isnan(clipped).any():
        raise ValueError("Ensemble produced NaN predictions after clipping.")
    if np.isinf(clipped).any():
        raise ValueError("Ensemble produced Inf predictions after clipping.")
    if float(np.max(clipped)) > float(np.max(upper_vals)) + 1e-9:
        raise ValueError("Ensemble still contains values above the physical cap.")

    output = merged[["series_id", "timestamp"]].copy()
    output["prediction"] = clipped
    output["timestamp"] = pd.to_datetime(output["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_file, index=False)
    print(f"wrote {len(output)} rows -> {args.output_file}")


if __name__ == "__main__":
    main()
