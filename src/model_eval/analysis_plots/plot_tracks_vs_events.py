# src/model_eval/analysis/plot_tracks_vs_events.py

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from model_eval.data.loaders import load_gt_pred_all
from model_eval.config import (
    get_paths_for_version,
    ensure_dir,
    GT_SOURCE,
    PRED_SOURCE,
    GT_COLOR,
    PRED_COLOR,
)
from model_eval.data.metadata import apply_series_labels
from model_eval.utils.constants import PRED_SOURCE


def compute_tracks_vs_events_counts(pred_version: str | None = None) -> pd.DataFrame:
    """
    Compute per video series:
        - n_events: number of distinct GT segment_id (true tracks)
        - n_tracks: number of distinct PRED track_id
    """
    df = load_gt_pred_all(pred_version).copy()
    if df.empty:
        raise ValueError(
            f"No data loaded from gt_pred_all.csv (pred_version={pred_version})."
        )

    # GT: count distinct segment_id
    df_gt = df[df["source"] == GT_SOURCE].copy()
    gt_counts = (
        df_gt[df_gt["segment_id"].notna()]
        .groupby("video_series")["segment_id"]
        .nunique()
        .astype(int)
        .reset_index(name="n_events")
    )

    # PRED: count distinct track_id
    df_pred = df[df["source"] == PRED_SOURCE].copy()
    pred_counts = (
        df_pred[df_pred["track_id_bt"].notna()]
        .groupby("video_series")["track_id_bt"]
        .nunique()
        .astype(int)
        .reset_index(name="n_tracks")
    )

    summary = gt_counts.merge(pred_counts, on="video_series", how="outer").fillna(0)

    return summary


def plot_tracks_vs_events(
        rename_labels: bool = True,
        save_path: Path | None = None,
        pred_version: str | None = None,
) -> None:
    """
    Scatter plot per series:
        - x-axis: number of GT events (distinct segment_id)
        - y-axis: number of prediction tracks (distinct track_id)
        - point label: series_label (V1, V2, ...)
    """
    df = compute_tracks_vs_events_counts(pred_version)

    # Add series labels (series_idx, series_label)
    if rename_labels:
        df = apply_series_labels(df)
        x_col = "series_label"
        order_col = "series_idx"
    else:
        x_col = "video_series"
        order_col = x_col


    x = range(len(df))
    width = 0.4

    # Plot
    fig, ax = plt.subplots(figsize=(max(10, len(x) * 0.8), 6))

    ax.bar(
        [i - width / 2 for i in x],
        df["n_events"],
        width=width,
        color=GT_COLOR,
        alpha=0.8,
        label="GT events (true sightings)",
    )
    ax.bar(
        [i + width / 2 for i in x],
        df["n_tracks"],
        width=width,
        alpha=0.8,
        label="ByteTrack tracks (track_id_bt)",
    )

    x_rotation = 0 if rename_labels else 90
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col], rotation=x_rotation)
    ax.set_ylabel("Count")
    ax.set_xlabel("Video series")
    ax.set_title(
        f"GT events vs ByteTrack predicted tracks per series "
        f"(real turtle sightings vs ByteTrack unique track_id_bt counts)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    # Save
    paths = get_paths_for_version(pred_version)
    filename = "tracks_vs_events_bars_per_series.png"

    if save_path is None:
        save_path = paths.reports_analysis_dir / filename
    else:
        if save_path.is_dir():
            save_path = save_path / filename

    ensure_dir(save_path.parent)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"[PLOT] Saved GT vs PRED tracks barplot to {save_path}")


if __name__ == "__main__":
    plot_tracks_vs_events(pred_version="v0_bt")