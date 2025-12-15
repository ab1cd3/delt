# src/model_eval/preprocessing/build_gt_pred_all.py

import pandas as pd

from model_eval.data.loaders import (
    load_std_gt_all_series,
    load_std_pred_all_series
)
from model_eval.config import (
    get_paths_for_version,
    ensure_dir,
    GT_SOURCE,
    PRED_SOURCE,
)


def build_gt_pred_all(pred_version: str | None = None) -> pd.DataFrame:
    """
    Build processed dataframe combining standardized GT and Pred files.
    Adds source labels but does NOT add derived metrics.
    Saves to data/processed/<pred_version>/gt_pred_all.csv.
    """
    # Paths
    paths = get_paths_for_version(pred_version)

    df_gt = load_std_gt_all_series().copy()
    df_pred = load_std_pred_all_series(pred_version).copy()

    if df_gt.empty:
        raise ValueError("No standardized GT data loaded.")

    if df_pred.empty:
        raise ValueError(f"No standardized prediction data loaded for version {pred_version}.")

    df_gt["source"] = GT_SOURCE
    df_pred["source"] = PRED_SOURCE

    df_all = pd.concat([df_gt, df_pred], ignore_index=True)

    # Add is_event at frame level
    event_frames = (
        df_gt[df_gt["segment_id"].notna()][["video_series", "frame"]]
        .drop_duplicates()
    )
    event_frames["is_event"] = True

    df_all = df_all.merge(
        event_frames,
        on=["video_series", "frame"],
        how="left",
    )

    df_all["is_event"] = df_all["is_event"].fillna(False)

    # Add event_type label:
    # - frames with GT (is_event=True) -> event
    # - frames with no GT (is_event=False) -> background
    df_all["event_type"] = df_all["is_event"].map(
        {True: "event", False: "background"}
    )

    ensure_dir(paths.processed_dir)
    out_path = paths.gt_pred_all_path
    df_all.to_csv(out_path, index=False)

    print(f"[PROCESSED] Saved combined GT+PRED dataset to {out_path}")
    return df_all