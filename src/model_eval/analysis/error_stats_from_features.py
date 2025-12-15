# src/model_eval/analysis/error_stats_from_features.py

from typing import Iterable
import pandas as pd
import numpy as np

from model_eval.data.loaders import load_gt_pred_all_features
from model_eval.data.metadata import apply_series_labels, apply_series_event_labels
from model_eval.config import GT_SOURCE, PRED_SOURCE, IOU_THRESHOLD
from model_eval.utils.metrics import add_pr_columns, normalize_filter_arg

def compute_error_counts_per_series(
        iou_thresh: float = IOU_THRESHOLD,
        video_series: str | Iterable[str] | None = None,
        pred_version: str | None = None,
) -> pd.DataFrame:
    """
    Compute the error type counts for given series.
    Uses data/processed/<pred_version>/gt_pred_all_features_iou_<iou_thresh>.csv.
    """
    df = load_gt_pred_all_features(iou_thresh, pred_version)

    vs_filter = normalize_filter_arg(video_series)
    if vs_filter is not None:
        df = df[df["video_series"].isin(vs_filter)]

    # Remove the GT 'tp' to avoid duplicating
    mask_gt_tp = (df["source"] == GT_SOURCE) & (df["error_type"].str.lower() == "tp")
    df.loc[mask_gt_tp, "error_type"] = np.nan

    # Count error types
    df_counts = (
        df
        .groupby(["video_series", "error_type"])
        .size()
        .unstack("error_type", fill_value=0)
        .rename_axis(None, axis="columns")
        .reset_index()
    )
    df_series = apply_series_labels(df_counts)
    df_series =  add_pr_columns(df_series)
    return df_series


def compute_error_counts_per_event(
        iou_thresh: float = IOU_THRESHOLD,
        video_series: str | Iterable[str] | None = None,
        segment_id: str | Iterable[str] | None = None,
        pred_version: str | None = None,
) -> pd.DataFrame:
    """
    Compute the error type counts for given events (video_series/segment_id).
    Uses data/processed/<pred_version>/gt_pred_all_features_iou_<iou_thresh>.csv.
    """
    df = load_gt_pred_all_features(iou_thresh, pred_version)

    vs_filter = normalize_filter_arg(video_series)
    seg_filter = normalize_filter_arg(segment_id)

    if vs_filter is not None:
        df = df[df["video_series"].isin(vs_filter)]

    if seg_filter is not None:
        df = df[df["segment_id"].isin(seg_filter)]

    # Remove the GT 'tp' to avoid duplicating
    mask_gt_tp = (df["source"] == GT_SOURCE) & (df["error_type"].str.lower() == "tp")
    df.loc[mask_gt_tp, "error_type"] = np.nan

    df_counts = (
        df
        .groupby(["video_series", "segment_id", "error_type"], as_index=False)
        .size()
        .unstack("error_type", fill_value=0)
        .rename_axis(None, axis="columns")
        .reset_index()
    )
    df_events = apply_series_event_labels(df_counts)
    df_events = add_pr_columns(df_events)
    return df_events
