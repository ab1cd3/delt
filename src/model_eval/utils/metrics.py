# src/model_eval/utils/metrics.py

from typing import Iterable
import numpy as np
import pandas as pd

from model_eval.utils.bbox import iou_xyxy
from model_eval.config import IOU_THRESHOLD
from model_eval.utils.constants import PRED_SOURCE


def match_frame_predictions(
        gt_boxes: np.ndarray,
        df_pred_f: pd.DataFrame,
        iou_thresh: float = IOU_THRESHOLD,
        return_gt_matched: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    """
    Given GT boxes and a predictions dataframe for a single frame,
    perform greedy one-to-one matching (sorted by confidence).

    Returns:
        If return_gt_matched=False (default):
            matched_pred_df: df_pred_f (reordered by confidence) with extra columns:
                - best_iou
                - is_tp

        If return_gt_matched=True:
            (matched_pred_df, gt_matched)
              - matched_pred_df
              - gt_matched: bool array of shape (n_gt,),
                            True if that GT box was matched by some pred
    """
    n_gt = gt_boxes.shape[0]

    # No predictions in this frame
    if df_pred_f.empty:
        out = df_pred_f.copy()
        out["best_iou"] = np.nan
        out["is_tp"] = False
        if return_gt_matched:
            gt_matched = np.zeros(n_gt, dtype=bool)
            return out, gt_matched
        return out

    # Sort preds by confidence descending, but KEEP original index
    df_pred_sorted = df_pred_f.sort_values("confidence", ascending=False)
    pred_boxes = df_pred_sorted[["x_min", "y_min", "x_max", "y_max"]].to_numpy()
    n_pred = pred_boxes.shape[0]

    gt_matched = np.zeros(n_gt, dtype=bool)
    best_ious = np.zeros(n_pred, dtype=float)
    is_tp = np.zeros(n_pred, dtype=bool)

    for p_idx in range(n_pred):
        p_box = pred_boxes[p_idx]
        best_iou = 0.0
        best_gt_idx = -1

        for g_idx in range(n_gt):
            if gt_matched[g_idx]:
                continue
            g_box = gt_boxes[g_idx]
            iou = iou_xyxy(g_box, p_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        best_ious[p_idx] = best_iou
        if best_iou >= iou_thresh and best_gt_idx >= 0:
            is_tp[p_idx] = True
            gt_matched[best_gt_idx] = True

    # Attach results directly to the sorted DF
    out = df_pred_sorted.copy()
    out["best_iou"] = best_ious
    out["is_tp"] = is_tp

    if return_gt_matched:
        return out, gt_matched
    return out


def normalize_filter_arg(arg: str | Iterable[str] | None) -> list[str] | None:
    """
    Helper to allow a single string or list as filter argument.
    """
    if arg is None:
        return None
    if isinstance(arg, str):
        return [arg]
    return list(arg)


def add_pr_columns(df: pd.DataFrame, include_bg_as_fp: bool = True) -> pd.DataFrame:
    """
    Add precision and recall columns to a TP/FP/FN table.
    If include_bg_as_fp=True: fp = fp + bg
    precision = tp / (tp + fp)  (NaN if tp+fp == 0)
    recall  = tp / (tp + fn)  (NaN if tp+fn == 0)
    """
    df = df.copy()

    tp = df["tp"].astype(float)
    if "source" in df.columns:
        tp = df[df["source"] == PRED_SOURCE]["tp"].astype(float)

    fp = df["fp"].astype(float)
    if include_bg_as_fp:
        fp += df["bg"].astype(float)
    fn = df["fn"].astype(float)

    prec_den = tp + fp
    rec_den = tp + fn

    df["precision"] = np.where(prec_den > 0, tp / prec_den, np.nan)
    df["recall"] = np.where(rec_den > 0, tp / rec_den, np.nan)

    return df