# src/model_eval/analysis/plot_event_confidence_timeline.py

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from model_eval.data.loaders import load_gt_pred_all_features
from model_eval.data.metadata import apply_series_event_labels
from model_eval.config import (
    get_paths_for_version,
    ensure_dir,
    ERROR_COLORS,
    IOU_THRESHOLD,
    PRED_SOURCE,
)


ERRORS = ["tp", "fp", "bg"]


def plot_event_confidence_timeline(
        video_series: str,
        segment_id: str,
        iou_thresh: float = IOU_THRESHOLD,
        rename_labels: bool = True,
        save_path: Path | None = None,
        pred_version: str | None = None,
) -> Path:
    """
    Predictions confidence timeline for one event (video_series, segment_id).
      - x-axis: frame number
      - y-axis: prediction confidence
      - color: error_type (tp/fp/bg)
    """
    df = load_gt_pred_all_features(iou_thresh=iou_thresh, pred_version=pred_version)
    df_series = df[df["video_series"] == video_series].copy()
    if df_series.empty:
        raise ValueError(f"No rows found for video_series={video_series}")

    # Event window (min_f, max_f)
    seg_rows = df_series[df_series["segment_id"] == segment_id]
    if seg_rows.empty:
        raise ValueError(f"No rows found for segment_id={segment_id} in series={video_series}")

    min_f = int(seg_rows["frame"].min())
    max_f = int(seg_rows["frame"].max())

    # Predictions within the event window
    df_plot = df_series[
        (df_series["source"] == PRED_SOURCE)
        & (df_series["frame"].between(min_f, max_f))
    ].copy()

    if df_plot.empty:
        raise ValueError(
            f"No PRED rows found in window [{min_f}, {max_f}] for series={video_series}, segment_id={segment_id}"
        )

    # Title
    title_label = f"{video_series} / {segment_id}"
    if rename_labels:
        meta = apply_series_event_labels(
            df_series[["video_series", "segment_id"]].drop_duplicates()
        )
        # Use "full_label" (e.g., V1E1)
        if "full_label" in meta.columns:
            m = meta[meta["segment_id"] == segment_id]
            if not m.empty:
                title_label = str(m["full_label"].iloc[0])

    # Colors
    df_plot["error_type"] = df_plot["error_type"].astype(str).str.lower()
    colors = df_plot["error_type"].map(ERROR_COLORS).fillna("gray")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(
        df_plot["frame"],
        df_plot["confidence"],
        width=1,
        color=colors,
        alpha=0.8,
    )

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Frame number")
    ax.set_ylabel("Confidence")
    ax.set_title(f"Event timeline: {title_label} (IoU ≥ {iou_thresh:.2f})")
    ax.grid(True, axis="y", alpha=0.3)

    # legend
    legend_handles = [
        Patch(color=ERROR_COLORS[e], label=e.upper())
        for e in ERRORS
        if e in ERROR_COLORS
    ]
    ax.legend(handles=legend_handles, title="Error type", frameon=False, ncol=len(legend_handles))

    # save
    paths = get_paths_for_version(pred_version)
    iou_str = f"{iou_thresh:.2f}".replace(".", "_")
    filename = f"event_conf_timeline_{video_series}_{segment_id}_iou_{iou_str}.png"

    if save_path is None:
        save_path = paths.reports_analysis_dir / filename
    else:
        save_path = Path(save_path)
        if save_path.is_dir():
            save_path = save_path / filename
    ensure_dir(save_path.parent)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"[PLOT] Saved -> {save_path}")
    return save_path