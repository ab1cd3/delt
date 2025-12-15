# src/model_eval/analysis/build_gt_pred_all_features.py

import pandas as pd
import numpy as np

from model_eval.data.loaders import load_gt_pred_all
from model_eval.utils.bbox import add_static_geometry_features
from model_eval.utils.metrics import match_frame_predictions
from model_eval.config import (
    get_paths_for_version,
    ensure_dir,
    GT_SOURCE,
    PRED_SOURCE,
    IOU_THRESHOLD,
)


def add_error_labels(
        df: pd.DataFrame,
        iou_thresh: float = IOU_THRESHOLD,
) -> pd.DataFrame:
    """
    Add per-box error labels to a gt_pred_all dataframe.

    Added columns:
        - error_type (for both GT and PRED):
            - 'tp': prediction matched a GT (IoU >= iou_thresh)
            - 'fp': prediction in event frame but not matched
            - 'bg': prediction in frame with no GT (background)
            - 'fn': GT box not matched by any prediction
        - best_iou (for PRED rows):
            - max IoU vs any GT in that frame (0.0 for 'bg')
    """
    out = df.copy()

    if "error_type" not in out.columns:
        out["error_type"] = None
    if "best_iou" not in out.columns:
        out["best_iou"] = np.nan

    # Single pass over all (series, frame)
    for (series, frame), df_frame in out.groupby(["video_series", "frame"], sort=False):
        gt_mask = df_frame["source"] == GT_SOURCE
        pred_mask = df_frame["source"] == PRED_SOURCE

        df_gt_f = df_frame[gt_mask]
        df_pred_f = df_frame[pred_mask]

        gt_idx = df_gt_f.index.to_numpy()
        pred_idx = df_pred_f.index.to_numpy()

        # Case 1: no preds -> all GT are FN
        if df_pred_f.empty:
            if df_gt_f.shape[0] > 0:
                out.loc[gt_idx, "error_type"] = "fn"
            continue

        # Case 2: preds but no GT -> all preds = BG
        if df_gt_f.empty:
            out.loc[pred_idx, "error_type"] = "bg"
            out.loc[pred_idx, "best_iou"] = 0.0
            continue

        # Case 3: both present -> greedy one-to-one matching
        gt_boxes = df_gt_f[["x_min", "y_min", "x_max", "y_max"]].to_numpy()

        matched_pred, gt_matched = match_frame_predictions(
            gt_boxes=gt_boxes,
            df_pred_f=df_pred_f,
            iou_thresh=iou_thresh,
            return_gt_matched=True,
        )

        tp_mask = matched_pred["is_tp"]
        fp_mask = ~matched_pred["is_tp"]

        # Update PRED rows
        out.loc[matched_pred.index[tp_mask], "error_type"] = "tp"
        out.loc[matched_pred.index[fp_mask], "error_type"] = "fp"
        out.loc[matched_pred.index, "best_iou"] = matched_pred["best_iou"].to_numpy()

        # Update GT rows: unmatched GT -> fn, matched GT -> tp
        out.loc[gt_idx[~gt_matched], "error_type"] = "fn"
        out.loc[gt_idx[gt_matched], "error_type"] = "tp"

    return out


def build_gt_pred_all_features(
        iou_thresh: float = IOU_THRESHOLD,
        pred_version: str | None = None,
) -> pd.DataFrame:
    """
    Build enriched GT+PRED dataset with static geometry + error labels.

    Saves:
        data/processed/<pred_version>/gt_pred_all_features_iou_<iou>.csv
    """
    df = load_gt_pred_all(pred_version).copy()

    if df.empty:
        raise ValueError(
            f"No data loaded from gt_pred_all.csv (pred_version={pred_version})."
        )

    required = {
        "video_series",
        "segment_id",
        "frame",
        "source",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "is_event",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in gt_pred_all.csv: {sorted(missing)}")

    # 1) Static geometry (cx, cy, w, h, area, etc.)
    df = add_static_geometry_features(df)

    # 2) Error labels (tp/fp/bg/fn + best_iou)
    df = add_error_labels(df, iou_thresh)

    # 3) Save
    paths = get_paths_for_version(pred_version)
    save_path = paths.gt_pred_features_path(iou_thresh=iou_thresh)
    ensure_dir(save_path.parent)
    df.to_csv(save_path, index=False)

    print(
        f"[FEATURES] Saved GT+PRED features (pred_version={pred_version}, "
        f"iou_thresh={iou_thresh}) to {save_path}"
    )

    return df