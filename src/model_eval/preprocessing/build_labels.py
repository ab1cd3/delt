# src/model_eval/preprocessing/build_labels.py

import pandas as pd

from model_eval.data.loaders import load_std_gt_all_series
from model_eval.config import DATA_METADATA


def build_labels_from_std_gt() -> pd.DataFrame:
    """
    Build series and segment labels from the processed GT+PRED joined csv file.
    Series labels: V1, V2...
    Event/Segment labels: E1, E2...
    Full labels: V1E1, V1E2...
    """
    df_gt = load_std_gt_all_series()

    if df_gt.empty:
        raise ValueError("Standardized GT data is empty.")

    required_cols = {"video_series", "segment_id"}
    if not required_cols.issubset(df_gt.columns):
        raise ValueError(f"Standardized GT must contain columns: {required_cols}.")

    # Unique (video_series, segment_id) pairs
    df_gt["segment_p1"] = pd.to_numeric(df_gt["segment_id"].str.split("_").str[0], errors="coerce")
    df_gt["segment_p2"] = pd.to_numeric(df_gt["segment_id"].str.split("_").str[1], errors="coerce")

    pairs = (
        df_gt[["video_series", "segment_id", "segment_p1", "segment_p2"]]
        .drop_duplicates()
        .sort_values(["video_series", "segment_p1", "segment_p2"], na_position="last")
        .reset_index(drop=True)
        .drop(columns=["segment_p1", "segment_p2"])
    )

    if pairs.empty:
        raise ValueError("No (video_series, segment_id) pairs found in GT data.")

    # Series label: V1, V2...
    pairs["series_idx"] = pairs.groupby("video_series", sort=True).ngroup() + 1
    pairs["series_label"] = "V" + pairs["series_idx"].astype(str)
    # Event label per series: E1, E2...
    pairs["event_idx"] = pairs.groupby("video_series", sort=True).cumcount() + 1
    pairs["event_label"] = "E" + pairs["event_idx"].astype(str)

    # Full label: V1E1, V1E2...
    pairs["full_label"] = pairs["series_label"] + pairs["event_label"]

    # Save metadata CSV
    DATA_METADATA.mkdir(parents=True, exist_ok=True)
    out_path = DATA_METADATA / "series_event_labels.csv"
    pairs.to_csv(out_path, index=False)

    print(f"[LABELS] Saved metadata label map to {out_path}")

    return pairs