# src/model_eval/experiments/bytetrack_bt.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import supervision as sv

from model_eval.config import get_paths_for_version, ensure_dir


BT_COLS = [
    "video_series",
    "frame",
    "confidence",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "track_id_bt",
]


def add_bytetrack_for_series(df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Run ByteTrack on predictions for a single video_series and return ONLY
    the tracked boxes as a new DataFrame.

    Expects df_pred columns:
        video_series, frame, confidence, x_min, y_min, x_max, y_max
    Returns a DataFrame with same columns + track_id_bt
    """
    if df_pred.empty:
        return pd.DataFrame(columns=BT_COLS)

    df = df_pred.sort_values("frame").copy()
    series_name = df["video_series"].iloc[0]

    tracker = sv.ByteTrack()
    out_rows: List[dict] = []

    # Process frame by frame so ByteTrack can maintain internal state
    for frame, g in df.groupby("frame", sort=True):
        xyxy = g[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=float)
        confidences = g["confidence"].to_numpy(dtype=float)
        class_id = np.zeros(len(g), dtype=int)  # single class

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidences,
            class_id=class_id,
        )

        tracked = tracker.update_with_detections(detections)

        if len(tracked) == 0:
            continue  # nothing tracked in this frame

        for box, conf, tid in zip(
            tracked.xyxy,
            tracked.confidence,
            tracked.tracker_id,
        ):
            x_min, y_min, x_max, y_max = box.tolist()
            out_rows.append(
                {
                    "video_series": series_name,
                    "frame": int(frame),
                    "confidence": float(conf),
                    "x_min": float(x_min),
                    "y_min": float(y_min),
                    "x_max": float(x_max),
                    "y_max": float(y_max),
                    "track_id_bt": int(tid),
                }
            )

    if not out_rows:
        # No tracks (filtered predictions) for this series
        return pd.DataFrame(columns=BT_COLS)

    df_bt = pd.DataFrame(out_rows, columns=BT_COLS)
    return df_bt


def add_bytetrack_to_all_series(
        base_pred_version: str,
        bt_pred_version: Optional[str] = None,
) -> None:
    """
    For each per-series standardized predictions file in:
        data/standardized/predictions/<base_pred_version>/*.csv

    Run ByteTrack and save a new file with tracked predictions to:
        data/standardized/predictions/<bt_pred_version>/*.csv

    If bt_pred_version is not given:
        bt_pred_version = base_pred_version + "_bt" ("v0" -> "v0_bt")
    """
    paths_in = get_paths_for_version(base_pred_version)
    in_dir = paths_in.std_pred_dir

    if bt_pred_version is None:
        bt_pred_version = f"{base_pred_version}_bt"

    paths_out = get_paths_for_version(bt_pred_version)
    out_dir = paths_out.std_pred_dir
    ensure_dir(out_dir)

    for csv_path in sorted(in_dir.glob("*.csv")):
        series = csv_path.stem
        print(f"[BYTETRACK] Processing series: {series} (base={base_pred_version} -> bt={bt_pred_version})")

        df_pred = pd.read_csv(csv_path)
        df_bt = add_bytetrack_for_series(df_pred)

        out_path = out_dir / f"{series}.csv"
        df_bt.to_csv(out_path, index=False)
        print(f"[BYTETRACK] Saved ByteTrack predictions -> {out_path}")