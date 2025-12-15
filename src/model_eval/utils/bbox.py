# src/model_eval/utils/bbox.py

import pandas as pd
import numpy as np

from model_eval.config import TARGET_WIDTH, TARGET_HEIGHT


def add_bbox_area(
        df: pd.DataFrame,
        x_min: str = "x_min",
        y_min: str = "y_min",
        x_max: str = "x_max",
        y_max: str = "y_max",
        area_col: str = "area",
) -> pd.DataFrame:
    """
    Add a bbox area column (in pixels^2) to a DataFrame with xyxy columns.
    Area = (x_max - x_min) * (y_max - y_min)
    """
    df = df.copy()
    required_cols = {x_min, y_min, x_max, y_max}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in dataframe: {missing_cols}")

    df[area_col] = (df[x_max] - df[x_min]) * (df[y_max] - df[y_min])

    return df


def iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Intersection-over-Union for boxes in (x_min, y_min, x_max, y_max) format.
    """
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)

    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0

    return inter_area / denom


def add_static_geometry_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-box static geometry features:
        - raw: box_w, box_h, area_px, cs, cy
        - normalized: box_w_rel, box_h_rel, area_rel, cx_rel, cy_rel
        - aspect_ratio
    """
    df = df.copy()

    # raw geometry (pixels)
    df["box_w"] = df["x_max"] - df["x_min"]
    df["box_h"] = df["y_max"] - df["y_min"]
    df["area_px"] = df["box_w"] * df["box_h"]
    df["cx"] = (df["x_min"] + df["x_max"]) / 2.0
    df["cy"] = (df["y_min"] + df["y_max"]) / 2.0

    w = float(TARGET_WIDTH)
    h = float(TARGET_HEIGHT)
    img_area = w * h

    # normalized geometry
    df["box_w_rel"] = df["box_w"] / w
    df["box_h_rel"] = df["box_h"] / h
    df["area_rel"] = df["area_px"] / img_area
    df["cx_rel"] = df["cx"] / w
    df["cy_rel"] = df["cy"] / h

    # aspect ratio (eps to avoid division issues)
    eps = 1e-6
    df["aspect_ratio"] = df["box_w"] / (df["box_h"] + eps)

    return df