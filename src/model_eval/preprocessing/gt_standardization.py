# src/model_eval/preprocessing/gt_standardization.py

from pathlib import Path
from typing import List
import pandas as pd

from model_eval.config import DATA_RAW_GT, DATA_STD_GT


def parse_gt_txt_file(path: Path, video_series: str, segment_id: str) -> pd.DataFrame:
    """
     Parse a GT txt file with lines:
        frame x_min y_min x_max y_max
     Return a DataFrame:
        video_series, segment_id, frame, x_min, y_min, x_max, y_max
     """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["frame", "x_min", "y_min", "x_max", "y_max"],
    )

    df["video_series"] = video_series
    df["segment_id"] = segment_id
    df = df[["video_series", "segment_id", "frame", "x_min", "y_min", "x_max", "y_max"]]
    return df


def standardize_gt_for_series(video_series: str) -> None:
    """
    Standardize all GT segments for a video series:
    - Convert each .txt to .csv
    - Create an aggregates *_all.csv
    """
    raw_series_dir = DATA_RAW_GT / video_series
    std_series_dir = DATA_STD_GT / video_series

    if not raw_series_dir.exists():
        raise FileNotFoundError(f"Raw GT directory not found: {raw_series_dir}")

    std_series_dir.mkdir(parents=True, exist_ok=True)
    all_segments: List[pd.DataFrame] = []

    for txt_path in sorted(raw_series_dir.glob("*.txt")):
        segment_id = txt_path.stem
        df_segment = parse_gt_txt_file(txt_path, video_series, segment_id)
        segment_path = std_series_dir / f"{segment_id}.csv"
        df_segment.to_csv(segment_path, index=False)
        print(f"[GT] Saved standardized GT series segment to: {segment_path}")

        all_segments.append(df_segment)

    if all_segments:
        df_all = pd.concat(all_segments, ignore_index=True)
        all_csv_path = std_series_dir / f"{video_series}_all.csv"
        df_all.to_csv(all_csv_path, index=False)
        print(f"[GT] Saved standardized GT series concatenation to: {all_csv_path}")

    else:
        print(f"No GT txt files found for series {video_series}")


def standardize_all_gt() -> None:
    """
    Standardize all the raw GT video series.
    """
    if not DATA_RAW_GT.exists():
        raise FileNotFoundError(f"Raw GT directory not found: {DATA_RAW_GT}")

    for series_dir in sorted(DATA_RAW_GT.iterdir()):
        if series_dir.is_dir():
            video_series = series_dir.name
            print(f"[GT] Standardizing series: {video_series}")
            standardize_gt_for_series(video_series)