# src/model_eval/utils/movement.py

from __future__ import annotations

import numpy as np
import pandas as pd

from model_eval.utils.bbox import iou_xyxy


def pick_one_box_per_frame(
        df: pd.DataFrame,
        group_cols: list[str] | None = None,
        by_col: str = "area_px",
        lowest: bool = True,
) -> pd.DataFrame:
    """
    For each group + frame, keep a single box (row) by given column and lowest.

    Returns a subset of df with at most 1 row per (group_cols, frame_col).
    """
    if df.empty:
        return df.copy()

    if by_col not in df.columns:
        raise ValueError(f"pick_one_box_per_frame: missing column {by_col}")

    if group_cols is None:
        group_cols = ["video_series"]

    df = df.copy()

    sort_cols = group_cols + ["frame", by_col, "confidence"]
    ascending = [True] * len(group_cols) + [True, lowest, False]
    df = df.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    return df.drop_duplicates(subset=group_cols + ["frame"], keep="first")


def add_movement_features(
        df: pd.DataFrame,
        group_cols: list[str] | None = None,
        cx_col: str = "cx",
        cy_col: str = "cy",
        drop_duplicate_boxes: bool = True,
) -> pd.DataFrame:
    """
    Add movement features within sequences defined by group_cols.

    For each group (e.g. GT (video_series, segment_id), PRED (video_series))
    sort frames by 'frame' and compute:
        - dx: cx - cx_prev
        - dy: cy - cy_prev
        - move_dist: sqrt(dx^2 + dy^2)
        - move_iou: IoU(box_t, box_{t-1})
    """
    if df.empty:
        return df.copy()

    if group_cols is None:
        group_cols = ["video_series"]

    if drop_duplicate_boxes:
        df = pick_one_box_per_frame(df, group_cols=group_cols)

    out = df.copy()

    # Initialize
    out["dx"] = 0.0
    out["dy"] = 0.0
    out["move_dist"] = 0.0
    out["move_iou"] = 0.0

    box_cols = ["x_min", "y_min", "x_max", "y_max"]

    for _, idx in out.groupby(group_cols).groups.items():
        g = out.loc[idx].sort_values("frame").copy()

        # dx, dy, dist
        dx = g[cx_col].diff().fillna(0.0)
        dy = g[cy_col].diff().fillna(0.0)
        dist = np.sqrt(dx ** 2 + dy ** 2)

        # IoU with previous box
        boxes = g[box_cols].to_numpy(dtype=float)  # (N, 4)
        mov_iou = np.zeros(len(g), dtype=float)
        for i in range(1, len(g)):
            mov_iou[i] = float(iou_xyxy(boxes[i], boxes[i - 1]))

        # save
        out.loc[g.index, "dx"] = dx.to_numpy()
        out.loc[g.index, "dy"] = dy.to_numpy()
        out.loc[g.index, "move_dist"] = dist.to_numpy()
        out.loc[g.index, "move_iou"] = mov_iou

    return out