# src/model_eval/analysis/build_pred_iou.py

from typing import Iterable

import numpy as np
import pandas as pd

from model_eval.data.loaders import load_gt_pred_all
from model_eval.utils.bbox import iou_xyxy
from model_eval.config import (
    GT_SOURCE,
    PRED_SOURCE,
    get_paths_for_version,
    ensure_dir,
)


def _normalize_series_filter(
        video_series: str | Iterable[str] | None
) -> list[str] | None:
    if video_series is None:
        return None
    if isinstance(video_series, str):
        return [video_series]
    return list(video_series)


def compute_pred_iou_table(
        video_series: str | Iterable[str] | None = None,
        pred_version: str | None = None,
) -> pd.DataFrame:
    """
    For each prediction (PRED row) compute best IoU with any GT box
    in the same (video_series, frame).

    Returns DataFrame with columns:
      video_series, frame, track_id, confidence, iou (best IoU vs any GT in that frame)
    """
    df = load_gt_pred_all(pred_version)
    if df.empty:
        raise ValueError(f"No data loaded from gt_pred_all.csv (pred_version={pred_version}).")

    vs_filter = _normalize_series_filter(video_series)
    if vs_filter is not None:
        df = df[df["video_series"].isin(vs_filter)]

    required = {
        "video_series",
        "frame",
        "source",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "confidence",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {pred_version}/gt_pred_all.csv: {sorted(missing)}")

    df_gt = df[df["source"] == GT_SOURCE]
    df_pred = df[df["source"] == PRED_SOURCE]

    records: list[dict] = []

    # Per (series, frame) to handle frames with only GT / only PRED
    for (series, frame), df_pred_frame in df_pred.groupby(["video_series", "frame"]):
        df_gt_frame = df_gt[
            (df_gt["video_series"] == series)
            & (df_gt["frame"] == frame)
        ]

        if df_gt_frame.empty:
            # No GT in frame -> all IoU = 0.0
            for _, r in df_pred_frame.iterrows():
                records.append(
                    {
                        "video_series": series,
                        "frame": frame,
                        "track_id": r.get("track_id", None),
                        "confidence": r.get("confidence", None),
                        "iou": 0.0,
                    }
                )
            continue

        gt_boxes = df_gt_frame[["x_min", "y_min", "x_max", "y_max"]].to_numpy()

        for _, r in df_pred_frame.iterrows():
            pred_box = np.array(
                [r["x_min"], r["y_min"], r["x_max"], r["y_max"]],
                dtype=float,
            )

            best_iou = 0.0
            for g in gt_boxes:
                iou = iou_xyxy(g, pred_box)
                if iou > best_iou:
                    best_iou = iou

            records.append(
                {
                    "video_series": series,
                    "frame": frame,
                    "track_id": r.get("track_id", None),
                    "confidence": r.get("confidence", None),
                    "iou": best_iou,
                }
            )

    if not records:
        return pd.DataFrame(
            columns=["video_series", "frame", "track_id", "confidence", "iou"]
        )

    return pd.DataFrame(records)


def build_pred_iou_table(
        video_series: str | Iterable[str] | None = None,
        pred_version: str | None = None
) -> pd.DataFrame:
    """
    Compute IoU per prediction and save to:
        VersionPaths.raw_pred_dir = data/processed/<pred_version>/pred_iou.csv.
    """
    df_pred_iou = compute_pred_iou_table(
        video_series=video_series,
        pred_version=pred_version
    )

    paths = get_paths_for_version(pred_version)
    ensure_dir(paths.processed_dir)
    out_path = paths.pred_iou_path

    df_pred_iou.to_csv(out_path, index=False)
    print(f"[IOU] Saved prediction IoU table to {out_path}")

    return df_pred_iou