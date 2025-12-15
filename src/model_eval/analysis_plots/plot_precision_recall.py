# src/model_eval/analysis/plot_precision_recall.py

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from model_eval.config import (
    get_paths_for_version,
    ensure_dir,
    IOU_THRESHOLD,
)
from model_eval.analysis.error_stats_from_features import compute_error_counts_per_series
from model_eval.utils.metrics import add_pr_columns


def plot_precision_recall_lines_by_series(
        iou_thresh: float = IOU_THRESHOLD,
        rename_labels: bool = True,
        save_path: Path | None = None,
        pred_version: str | None = None,
) -> None:
    """
    Line plot of precision and recall per video series.

    X-axis: video series (or series_label if available).
    Y-axis: precision and recall (two lines, different colors).
    """
    df = compute_error_counts_per_series(iou_thresh=iou_thresh, pred_version=pred_version)
    if df.empty:
        raise ValueError("No data loaded from gt_pred_all_features.")
    df = add_pr_columns(df)
    print(df.to_string())
    # Choose label col and sort order
    if rename_labels and "series_label" in df.columns and "series_idx" in df.columns:
        df = df.sort_values("series_idx")
        x_labels = df["series_label"].tolist()
    else:
        df = df.sort_values("video_series")
        x_labels = df["video_series"].tolist()

    precision = df["precision"].astype(float)
    recall = df["recall"].astype(float)

    x = range(len(x_labels))

    fig, ax = plt.subplots(figsize=(max(8, int(len(x_labels) * 0.6)), 5))

    ax.plot(x, precision, marker="o", label="Precision")
    ax.plot(x, recall, marker="s", label="Recall")

    x_rotation = 0 if rename_labels else 90
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels, rotation=x_rotation, ha="right")

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_xlabel("Video series")
    ax.set_title(f"Precision and Recall per series including background errors as false positives (IoU = {iou_thresh:.2f})")

    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    # Save
    paths = get_paths_for_version(pred_version)
    iou_str = f"{iou_thresh:.2f}".replace(".", "_")
    filename = f"precision_recall_by_series_iou_{iou_str}.png"

    if save_path is None:
        save_path = paths.reports_analysis_dir / filename
    else:
        if save_path.is_dir():
            save_path = save_path / filename

    ensure_dir(save_path.parent)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"[PLOT] Saved precision/recall-by-series line plot to {save_path}")

if __name__ == "__main__":
    plot_precision_recall_lines_by_series(pred_version="v0")