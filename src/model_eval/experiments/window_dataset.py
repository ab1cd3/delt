# src/model_eval/experiments/window_dataset.py

from __future__ import annotations

from typing import List, Dict

import pandas as pd

from model_eval.experiments.dataset_utils import (
    WindowConfig,
    ensure_pred_features,
    generate_windows_for_series,
    label_window,
    get_frame_features,
)


def build_window_dataset(
        cfg: WindowConfig,
        video_series: str | List[str] | None = None,
) -> pd.DataFrame:
    """
    Build a window-level dataset from gt_pred_all_features_iou_<iou>.csv.

    Each row in the returned DataFrame is one window with:
        - video_series
        - frame_start, frame_end
        - label (1=event, 0=non-event, depending on cfg.label_mode)
        - n_pred
        - aggregated numerical features (mean, std, min, max) for:
          base features + optional track_id + optional extra features.
    """
    # Load full PRED features
    df_pred = ensure_pred_features(cfg, video_series)

    # Decide frame-level feature columns (base + optional track/extra features)
    feature_cols = get_frame_features(df_pred, cfg)

    missing = [c for c in feature_cols if c not in df_pred.columns]
    if missing:
        raise ValueError(f"Missing expected features in df_pred: {missing}")

    records: List[Dict] = []

    for series, df_s in df_pred.groupby("video_series"):
        df_s = df_s.sort_values("frame")

        # Precompute per-frame table ONCE per series
        # per-frame aggregates
        agg_map = {c: "mean" for c in feature_cols}

        # Override for the ones where max/min matter
        if "confidence" in agg_map:
            agg_map["confidence"] = "max"  # max conf per frame
        if "area_px" in agg_map:
            agg_map["area_px"] = "min"  # min area per frame

        df_frame = (
            df_s.groupby("frame", as_index=False)
            .agg(agg_map)
        )

        # Density features
        per_frame_counts = df_s.groupby("frame").size()
        df_frame["n_boxes_frame"] = df_frame["frame"].map(per_frame_counts).fillna(0).astype(int)

        track_cols = [c for c in df_s.columns if c.startswith("track_id")]
        if track_cols:
            tcol = track_cols[0]
            per_frame_tracks = df_s.groupby("frame")[tcol].nunique()
            df_frame["n_tracks_frame"] = df_frame["frame"].map(per_frame_tracks).fillna(0).astype(int)
        else:
            df_frame["n_tracks_frame"] = 0

        # Add max area_px
        if "area_px" in df_s.columns:
            per_frame_area_max = df_s.groupby("frame")["area_px"].max()
            df_frame["area_px_max"] = df_frame["frame"].map(per_frame_area_max)

        # Index
        df_frame = df_frame.set_index("frame").sort_index()

        frame_feature_cols = feature_cols + ["n_boxes_frame", "n_tracks_frame"]

        if "area_px_max" in df_frame.columns and "area_px_max" not in frame_feature_cols:
            frame_feature_cols.append("area_px_max")

        windows = generate_windows_for_series(df_s, cfg)
        if not windows:
            continue

        for (f_start, f_end) in windows:
            # RAW rows (for label + counts)
            df_w = df_s[df_s["frame"].between(f_start, f_end)]
            if df_w.empty:
                continue

            label = label_window(df_w, cfg)

            # FRAME rows (for aggregation)
            df_fw = df_frame.loc[f_start:f_end]
            if df_fw.empty:
                continue

            agg = df_fw[frame_feature_cols].agg(["mean", "std", "min", "max"])

            rec: Dict = {
                "video_series": series,
                "frame_start": int(f_start),
                "frame_end": int(f_end),
                "label": label,
                "n_pred": int(len(df_w)),  # raw boxes
                "n_frames": int(df_fw.shape[0]),  # unique frames
            }

            for col in frame_feature_cols:
                for stat in ["mean", "std", "min", "max"]:
                    rec[f"{col}_{stat}"] = float(agg.loc[stat, col])

            records.append(rec)

    if not records:
        raise ValueError(
            "No windows generated (check cfg.window_size/step_size and filters)."
        )

    return pd.DataFrame(records)