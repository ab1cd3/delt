# src/model_eval/analysis/top_iou_frames.py

import pandas as pd
from typing import Iterable

from model_eval.data.loaders import load_pred_iou


def _normalize_series_filter(
        video_series: str | Iterable[str] | None,
) -> list[str] | None:
    if video_series is None:
        return None
    if isinstance(video_series, str):
        return [video_series]
    return list(video_series)


def get_top_iou_per_series(
        top_k: int = 5,
        video_series: str | Iterable[str] | None = None,
        pred_version: str | None = None,
) -> pd.DataFrame:
    """
    Top-K predictions by IoU for each series from <pred_version>/pred_iou.csv.
    """
    df = load_pred_iou(pred_version)

    vs_filter = _normalize_series_filter(video_series)
    if vs_filter is not None:
        df = df[df["video_series"].isin(vs_filter)]

    if df.empty:
        return df

    # sort by series, IoU desc, confidence desc
    df = df.sort_values(
        ["video_series", "iou", "confidence"],
        ascending=[True, False, False],
    )

    # top-k per series
    df_top = (
        df.groupby("video_series", as_index=False, group_keys=False)
        .head(top_k)
    )

    return df_top.reset_index(drop=True)


def get_top_iou_global(
        top_k: int = 20,
        video_series: str | Iterable[str] | None = None,
        pred_version: str | None = None,
) -> pd.DataFrame:
    """
    Top-K predictions by IoU across all series (optionally filtered) from <pred_version>/pred_iou.csv.
    """
    df = load_pred_iou(pred_version)

    vs_filter = _normalize_series_filter(video_series)
    if vs_filter is not None:
        df = df[df["video_series"].isin(vs_filter)]

    if df.empty:
        return df

    df = df.sort_values(
        ["iou", "confidence"],
        ascending=[False, False],
    )

    return df.head(top_k).reset_index(drop=True)