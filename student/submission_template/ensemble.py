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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend TiDE and LightGBM predictions.")
    parser.add_argument("--tide", type=Path, default=Path("predictions_tide_best.csv"))
    parser.add_argument("--lgb", type=Path, default=Path("predictions_lgb.csv"))
    parser.add_argument("--output_file", type=Path, default=Path("predictions_final_ensemble.csv"))
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

    blended = tide_w * merged["prediction_tide"].to_numpy(dtype=np.float64) + lgb_w * merged[
        "prediction_lgb"
    ].to_numpy(dtype=np.float64)
    blended = np.clip(blended, 0.0, None)

    output = merged[["series_id", "timestamp"]].copy()
    output["prediction"] = blended
    output["timestamp"] = pd.to_datetime(output["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    print("=== ensemble stats ===")
    print(f"rows={len(output)}")
    print(f"weights: tide={tide_w:.3f} lgb={lgb_w:.3f}")
    print(f"min={blended.min():.6f}")
    print(f"max={blended.max():.6f}")
    print(f"mean={blended.mean():.6f}")
    print(f"negatives={(blended < 0).sum()}")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_file, index=False)
    print(f"wrote {len(output)} rows -> {args.output_file}")


if __name__ == "__main__":
    main()
