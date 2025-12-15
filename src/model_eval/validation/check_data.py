# src/model_eval/validation/check_data.py

from pathlib import Path
import pandas as pd

from model_eval.data.loaders import (
    load_gt_pred_all,
    load_std_gt_all_series,
)
from model_eval.config import (
    DATA_RAW_VIDEOS,
    GT_SOURCE,
    PRED_SOURCE,
)


def require_columns(df: pd.DataFrame, cols: set, name: str) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def check_series_have_gt_and_pred(pred_version: str | None = None) -> None:
    """
    Check that every video_series that appears in GT also appears in PRED,
    and vice versa.

    Uses simple sets of series names per source and compares them.
    Prints what is missing and raises ValueError if anything is wrong.
    """
    df = load_gt_pred_all(pred_version)

    if df.empty:
        raise ValueError(f"Processed GT+PRED table is empty ({pred_version}/gt_pred_all.csv).")

    required_cols = {
        "video_series",
        "segment_id",
        "source",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "confidence",
        "track_id",
    }

    require_columns(df, required_cols, "gt_pred_all.csv")

    gt_series = set(df.loc[df["source"] == GT_SOURCE, "video_series"].unique())
    pred_series = set(df.loc[df["source"] == PRED_SOURCE, "video_series"].unique())

    only_gt = gt_series - pred_series
    only_pred = pred_series - gt_series

    if not gt_series:
        raise ValueError("No GT rows found in gt_pred_all.csv.")
    if not pred_series:
        raise ValueError("No PRED rows found in gt_pred_all.csv.")

    if only_gt or only_pred:
        print("[CHECK] Mismatch between GT and PRED series:")
        if only_gt:
            print("  Series present in GT but missing in PRED:")
            for s in sorted(only_gt):
                print(f"    - {s}")
        if only_pred:
            print("  Series present in PRED but missing in GT:")
            for s in sorted(only_pred):
                print(f"    - {s}")
        raise ValueError("GT and PRED series sets do not match.")
    else:
        print("[CHECK] OK: GT and PRED have the same set of video_series.")


def check_gt_segments_have_video_folders() -> None:
    """
    Check that for each (video_series, segment_id) in standardized GT,
    there is a corresponding video folder:
        DATA_RAW_VIDEO / <video_series> / <segment_id>

    Prints missing folders and raises FileNotFoundError if any are missing.
    """
    df_gt = load_std_gt_all_series()

    if df_gt.empty:
        raise FileNotFoundError("Standardized GT data is empty (no GT segments).")

    require_columns(df_gt, {"video_series", "segment_id"}, "gt_pred_all.csv")

    pairs = (
        df_gt[["video_series", "segment_id"]]
        .drop_duplicates()
        .sort_values(["video_series", "segment_id"])
    )

    missing = []

    for _, row in pairs.iterrows():
        series = str(row["video_series"])
        segment = str(row["segment_id"])
        expected = DATA_RAW_VIDEOS / series / segment
        if not expected.is_dir():
            missing.append(expected)

    if missing:
        print("[CHECK] ERRORS: Some GT segments do not have matching video folders:")
        for p in missing:
            print(f"  - {p}")
        raise FileNotFoundError("Missing video folders for some GT segments.")
    else:
        print("[CHECK] OK: All GT segments have matching video folders.")