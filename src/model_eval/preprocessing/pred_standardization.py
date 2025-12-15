# src/model_eval/preprocessing/pred_standardization.py

from pathlib import Path
from typing import Tuple
import pandas as pd

from model_eval.config import (
    TARGET_WIDTH,
    TARGET_HEIGHT,
    get_paths_for_version,
    ensure_dir,
)

def parse_original_size(line: str) -> Tuple[float, float]:
    """
    Parse the first line 'original_size = 3840, 2160;' and return (width, height).
    """
    line = line.strip()
    try:
        size_part = line.split("=", 1)[1].strip().rstrip(";")
        width_str, height_str = [s.strip() for s in size_part.split(",")]
        return float(width_str), float(height_str)
    except Exception as e:
        raise ValueError(f"Cannot parse original size from line: {line}") from e


def standardize_predictions_for_series(
        video_series: str,
        pred_version: str | None = None,
) -> Path:
    """
    Standardize predictions for one video series for a given pred version.
    """
    paths = get_paths_for_version(pred_version)

    raw_path = paths.raw_pred_dir / f"{video_series}.txt"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw predictions file not found: {raw_path}")

    # First line: original size
    with raw_path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    orig_w, orig_h = parse_original_size(first_line)

    # Remaining lines: predictions
    df = pd.read_csv(
        raw_path,
        sep=";",
        skiprows=1,
        header=None,
        names=["frame", "confidence", "track_id", "bbox"],
    )

    if df.empty:
        raise ValueError(f"Raw predictions file empty for series: {video_series}")

    bbox_cols = df["bbox"].str.split(",", expand=True)
    if bbox_cols.shape[1] != 4:
        raise ValueError(
            f"Unexpected bbox format in {raw_path}. "
            f"Expected 4 values, got {bbox_cols.shape[1]}."
        )
    df[["x_min", "y_min", "x_max", "y_max"]] = bbox_cols.astype(float)
    df = df.drop(columns=["bbox"])

    df["frame"] = df["frame"].astype(int)

    # Scale
    sx = TARGET_WIDTH / orig_w
    sy = TARGET_HEIGHT / orig_h
    df["x_min"] *= sx
    df["y_min"] *= sy
    df["x_max"] *= sx
    df["y_max"] *= sy

    df["video_series"] = video_series

    df = df[["video_series", "frame", "confidence", "track_id", "x_min", "y_min", "x_max", "y_max"]]

    # Save
    out_dir = paths.std_pred_dir
    ensure_dir(out_dir)
    out_path = out_dir / f"{video_series}.csv"

    df.to_csv(out_path, index=False)
    print(f"[PRED] Saved standardized predictions to {out_path}")


def standardize_all_predictions(pred_version: str | None = None) -> None:
    """
    Standardize predictions for all video series found in:
        VersionPaths.raw_pred_dir = data/raw/predictions/<pred_version>/*.txt.
    """
    paths = get_paths_for_version(pred_version)
    raw_path = paths.raw_pred_dir
    if not raw_path.exists():
        raise FileNotFoundError(f"Predictions raw directory not found: {raw_path}.")

    for txt_path in sorted(raw_path.glob("*.txt")):
        video_series = txt_path.stem
        print(f"[PRED] Standardizing series: {video_series} (version={pred_version})")
        standardize_predictions_for_series(video_series, pred_version)