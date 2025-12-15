# src/model_eval/interactive/timeline_data.py

from __future__ import annotations

from typing import Optional, Tuple
import pandas as pd

from model_eval.config import (
    GT_SOURCE,
    PRED_SOURCE,
    IOU_THRESHOLD,
    METRIC_CONFIDENCE,
    METRIC_AREA,
    METRIC_MOVE_DIST,
    METRIC_MOVE_IOU,
)
from model_eval.data.loaders import load_gt_pred_all_features
from model_eval.data.metadata import apply_series_event_labels
from model_eval.utils.movement import pick_one_box_per_frame, add_movement_features


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_timeline_base_df(
        iou_thresh: float = IOU_THRESHOLD,
        pred_version: str | None = None,
) -> pd.DataFrame:
    """
    Load the whole dataset from:
        data/processed/<version>/gt_pred_all_features_iou_<iou>.csv
    """
    df = load_gt_pred_all_features(iou_thresh=iou_thresh, pred_version=pred_version)

    required = {
        "video_series",
        "segment_id",
        "frame",
        "source",
        "confidence",
        "area_px",
        "cx",
        "cy",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "error_type",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in features CSV: {sorted(missing)}"
        )

    return df


# ----------------------------------------------------------------------
# Metadata helpers
# ----------------------------------------------------------------------
def list_video_series(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique video_series in the dataframe."""
    return sorted(df["video_series"].astype(str).unique().tolist())


def get_series_frame_bounds(
        df: pd.DataFrame,
        video_series: str,
) -> Tuple[int, int]:
    """
    Returns (min_frame, max_frame) for a given video series.
    """
    df_s = df[df["video_series"] == video_series]
    if df_s.empty:
        raise ValueError(f"No rows found for video_series={video_series}")

    return int(df_s["frame"].min()), int(df_s["frame"].max())


def get_events_for_series(
        df: pd.DataFrame,
        video_series: str,
) -> pd.DataFrame:
    """
    Returns event metadata for a given series, with columns:
        video_series, segment_id, frame_min, frame_max, n_gt_boxes, full_label
    """
    df_gt = df[
        (df["video_series"] == video_series)
        & (df["source"] == GT_SOURCE)
        & (df["segment_id"].notna())
    ].copy()

    cols = ["video_series", "segment_id", "frame_min", "frame_max", "n_gt_boxes", "full_label"]

    if df_gt.empty:
        return pd.DataFrame(columns=cols)

    events = (
        df_gt.groupby(["video_series", "segment_id"], as_index=False)
        .agg(
            frame_min=("frame", "min"),
            frame_max=("frame", "max"),
            n_gt_boxes=("frame", "size"),
        )
        .sort_values(["frame_min", "segment_id"])
    )
    events = apply_series_event_labels(events)

    # Ensure all expected columns exist
    for c in cols:
        if c not in events.columns:
            events[c] = None

    return events[cols]


def get_event_frame_bounds(
        df: pd.DataFrame,
        video_series: str,
        segment_id: str,
) -> Tuple[int, int]:
    """
    Return (frame_min, frame_max) for a specific GT event (series, segment_id).
    """
    df_gt = df[
        (df["video_series"] == video_series)
        & (df["segment_id"] == segment_id)
        & (df["source"] == GT_SOURCE)
    ]

    if df_gt.empty:
        raise ValueError(
            f"No GT rows found for series={video_series}, segment_id={segment_id}"
        )

    return int(df_gt["frame"].min()), int(df_gt["frame"].max())


# ----------------------------------------------------------------------
# Filtering
# ----------------------------------------------------------------------
def filter_series_and_range(
        df: pd.DataFrame,
        video_series: str,
        frame_start: int,
        frame_end: int,
) -> pd.DataFrame:
    """
    Filter DataFrame to rows in given series and frame range inclusive.
    """
    mask = (
        (df["video_series"] == video_series)
        & (df["frame"].between(frame_start, frame_end))
    )
    return df[mask].copy()


# ----------------------------------------------------------------------
# Main prep for interactive app
# ----------------------------------------------------------------------
def prepare_timeline_views(
        df: pd.DataFrame,
        video_series: str,
        frame_start: int,
        frame_end: int,
        metric: str,
        segment_id: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main helper to prepare interactive app data.

    Returns:
        - df_pred_plot: PRED filtered (possibly with movement features)
        - df_gt_plot:   GT filtered (possibly with movement features)
    """
    metric_norm = metric.lower()
    needs_movement = metric_norm in {METRIC_MOVE_DIST, METRIC_MOVE_IOU}

    # Base filtering by series + frame
    df_range = filter_series_and_range(df, video_series, frame_start, frame_end)

    # PRED
    df_pred = df_range[df_range["source"] == PRED_SOURCE].copy()

    # GT
    df_gt = df_range[df_range["source"] == GT_SOURCE].copy()
    if segment_id is not None:
        df_gt = df_gt[df_gt["segment_id"] == segment_id]

    if needs_movement:
        # For preds: group by series, optionally dedup boxes per frame inside add_movement_features
        df_pred = add_movement_features(df_pred, group_cols=["video_series"])
        # For GT: movement per (series, segment_id)
        df_gt = add_movement_features(df_gt, group_cols=["video_series", "segment_id"])

    return df_pred, df_gt