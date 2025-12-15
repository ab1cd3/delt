# src/model_eval/experiments/dataset_utils.py

from dataclasses import dataclass
from typing import List, Tuple, Optional

import pandas as pd

from model_eval.config import IOU_THRESHOLD, TP_THRESHOLD, PRED_SOURCE
from model_eval.data.loaders import load_gt_pred_all_features
from model_eval.utils.metrics import normalize_filter_arg
from model_eval.utils.movement import add_movement_features, pick_one_box_per_frame


@dataclass
class WindowConfig:
    """
    Configuration for building window-level and sequence-level datasets.

    label_mode:
        - "tp_count"   -> use TP count in window (error_type == 'tp')
        - "event_frac" -> use fraction of frames with is_event == 1 in window
    """
    window_size: int = 120              # frames per window (4s at 30fps)
    step_size: int = 60                 # stride between window starts

    iou_thresh: float = IOU_THRESHOLD
    pred_version: Optional[str] = None
    label_mode: str = "tp_count"  # "tp_count" or "event_frac"

    # threshold for label_mode=tp_count
    tp_count_thresh: int = TP_THRESHOLD

    # threshold for label_mode=event_frac
    event_frac_thresh: float = 0.3

    extra_features: Optional[list[str]] = None
    include_track_ids: bool = False


# Base per-frame numeric features (track-agnostic, GT-free)
BASE_FRAME_FEATURES = [
    "confidence",
    "box_w_rel",
    "box_h_rel",
    "area_rel",
    "cx_rel",
    "cy_rel",
    "aspect_ratio",
    "dx",
    "dy",
    "move_dist",
    "move_iou",
    "frame_rel",
]


def get_frame_features(df_pred: pd.DataFrame, cfg: WindowConfig) -> list[str]:
    """
    Return the list of per-frame feature columns to use for modeling.

    Includes:
      - BASE_FRAME_FEATURES
      - optional track_id columns if cfg.include_track_ids=True
      - optional cfg.extra_features that exist in df_pred
    """
    features = list(BASE_FRAME_FEATURES)

    # Optional: include tracker cols
    if cfg.include_track_ids:
        for col in df_pred.columns:
            if col.startswith("track_id") and col not in features:
                features.append(col)

    # Optional: add extra features
    if cfg.extra_features:
        for f in cfg.extra_features:
            if f in df_pred.columns and f not in features:
                features.append(f)

    return features


def _ensure_movement_features(df_pred: pd.DataFrame, pred_version: str) -> pd.DataFrame:
    """
    Ensure movement features (dx, dy, move_dist, move_iou) exist in df_pred.

    IMPORTANT:
    - Do NOT drop duplicate boxes here.
    - Compute movement per frame using a representative box (min area),
      then map it to all boxes in that frame.
    """
    required = {"dx", "dy", "move_dist", "move_iou"}
    if required.issubset(df_pred.columns):
        return df_pred

    if "area_px" not in df_pred.columns:
        raise ValueError("Expected 'area_px' in df_pred to compute per-frame movement.")

    df_pred = df_pred.copy()

    # Representative box per (series, frame): smallest area
    idx_rep = df_pred.groupby(["video_series", "frame"])["area_px"].idxmin()
    df_rep = df_pred.loc[idx_rep].copy()

    # Compute movement on the representative sequence
    df_rep = add_movement_features(
        df_rep,
        group_cols=["video_series"],
        cx_col="cx",
        cy_col="cy",
        drop_duplicate_boxes=False,
    )

    # Map per-frame movement back to all boxes in that frame
    mov_cols = ["video_series", "frame", "dx", "dy", "move_dist", "move_iou"]
    df_pred = df_pred.merge(df_rep[mov_cols], on=["video_series", "frame"], how="left")

    # Fill
    df_pred[["dx", "dy", "move_dist", "move_iou"]] = df_pred[["dx", "dy", "move_dist", "move_iou"]].fillna(0.0)

    return df_pred


def _add_frame_rel(df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Add a normalized frame index per (video_series), in [0, 1]:
        frame_rel = (frame - frame_min_series) / (frame_max_series - frame_min_series)
    """
    if "frame_rel" in df_pred.columns:
        return df_pred

    df_pred = df_pred.copy()
    frame_min = df_pred.groupby("video_series")["frame"].transform("min")
    frame_max = df_pred.groupby("video_series")["frame"].transform("max")
    denom = (frame_max - frame_min).replace(0, 1)
    df_pred["frame_rel"] = (df_pred["frame"] - frame_min) / denom
    return df_pred


def ensure_pred_features(
        cfg: WindowConfig,
        video_series: str | List[str] | None = None,
) -> pd.DataFrame:
    """
    Load gt_pred_all_features_iou_<iou>.csv and return only PRED rows,
    with movement and frame_rel added.
    """
    df = load_gt_pred_all_features(
        iou_thresh=cfg.iou_thresh,
        pred_version=cfg.pred_version,
    )

    vs_filter = normalize_filter_arg(video_series)
    if vs_filter is not None:
        df = df[df["video_series"].isin(vs_filter)]

    if df.empty:
        raise ValueError(
            f"No rows left after filtering for video_series={video_series}"
        )

    df_pred = df[df["source"] == PRED_SOURCE].copy()
    if df_pred.empty:
        raise ValueError("No PRED rows found in features CSV.")

    df_pred = _ensure_movement_features(df_pred, cfg.pred_version)
    df_pred = _add_frame_rel(df_pred)
    return df_pred


def generate_windows_for_series(
        df_series: pd.DataFrame,
        cfg: WindowConfig,
) -> List[Tuple[int, int]]:
    """
    Given predictions for a single video_series (all frames),
    generate a list of (frame_start, frame_end) windows.

    Windows are defined based on frames (not rows):
        start in {frame_min, frame_min + step, ...}
        end = min(start + window_size - 1, frame_max)
    """
    frames = df_series["frame"].unique()
    if len(frames) == 0:
        return []

    f_min = int(frames.min())
    f_max = int(frames.max())

    windows: List[Tuple[int, int]] = []
    start = f_min
    while start <= f_max:
        end = min(start + cfg.window_size - 1, f_max)
        windows.append((start, end))
        start += cfg.step_size

    return windows


def _label_window_by_tp(df_window: pd.DataFrame, tp_thresh: int) -> int:
    """
    Label a window as event (1) or non-event (0) based on the number of TP predictions.
    Uses error_type, so it is used ONLY for training.
    """
    if df_window.empty:
        return 0
    n_tp = int((df_window["error_type"] == "tp").sum())
    return int(n_tp >= tp_thresh)


def _label_window_by_event_frac(df_window: pd.DataFrame, frac_thresh: float) -> int:
    """
    Label a window as event (1) if the fraction of frames with is_event == 1
    in the window is >= frac_thresh.
    """
    if df_window.empty:
        return 0

    frac = float((df_window["is_event"] == 1).mean())
    return int(frac >= frac_thresh)


def label_window(
        df_window: pd.DataFrame,
        cfg: WindowConfig,
) -> int:
    """
    Generic function for window labels based on cfg.label_mode.
    """
    mode = cfg.label_mode.lower()

    if mode in {"tp_count", "tp_thresh", "tp"}:
        return _label_window_by_tp(df_window, cfg.tp_count_thresh)

    elif mode in {"event_frac", "is_event_frac", "gt_event"}:
        return _label_window_by_event_frac(df_window, cfg.event_frac_thresh)

    else:
        raise ValueError(f"Unknown label_mode='{cfg.label_mode}'")