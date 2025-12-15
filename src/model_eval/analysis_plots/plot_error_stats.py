# src/model_eval/analysis/plot_error_stats.py

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from model_eval.config import (
    get_paths_for_version,
    ensure_dir,
    ERROR_COLORS,
    IOU_THRESHOLD,
    PRED_SOURCE
)
from model_eval.data.loaders import load_gt_pred_all_features
from model_eval.data.metadata import apply_series_labels


def plot_error_type_stacked_by_series(
        rename_labels: bool = True,
        iou_thresh: float = IOU_THRESHOLD,
        save_path: Path | None = None,
        pred_version: str | None = None,
) -> None:
    """
    Stacked barplot of normalized TP/FP/BG per series.

    X-axis: series (raw or series_label if rename_labels=True)
    Y-axis: proportion (TP+FP+FN = 1 per series)
    """
    df = load_gt_pred_all_features(pred_version=pred_version)
    df = df[df["source"] == PRED_SOURCE].copy()
    df = apply_series_labels(df)

    if rename_labels and "series_label" in df.columns:
        x_col = "series_label"
        order_col = "series_idx" if "series_idx" in df.columns else x_col
    else:
        x_col = "video_series"
        order_col = x_col

    # Sort and set index to the label column
    df = df.sort_values(order_col)

    # Keep only tp/fp/bg and fill missing with 0
    ERRORS = ["tp", "fp", "bg"]
    counts_norm = pd.crosstab(
        index=df[x_col],
        columns=df["error_type"],
        normalize="index",
    )

    counts_norm.columns.name = None

    order = (
        df.sort_values(order_col)[x_col]
        .drop_duplicates()
        .tolist()
    )
    counts_norm = counts_norm.reindex(order)

    # Colors in fixed order
    color_list = [ERROR_COLORS[k] for k in ERRORS]

    ax = counts_norm[ERRORS].plot(
        kind="bar",
        stacked=True,
        color=color_list,
        figsize=(max(6, int(len(counts_norm) * 0.8)), 5),
    )

    x_rotation = 0 if rename_labels else 90
    ax.set_xticklabels(ax.get_xticklabels(), rotation=x_rotation, ha="right")
    ax.set_xlabel("Video series")
    ax.set_ylabel("Proportion of detections")
    ax.set_ylim(0, 1)
    ax.set_title(f"Proportion of prediction errors per video series (IoU ≥ {iou_thresh})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend([e.upper() for e in ERRORS], title="Type", loc="upper right")

    # Save
    paths = get_paths_for_version(pred_version)
    iou_str = f"{iou_thresh:.2f}".replace(".", "_")
    filename = f"pred_error_types_by_series_iou_{iou_str}.png"

    if save_path is None:
        save_path = paths.reports_analysis_dir / filename
    else:
        if save_path.is_dir():
            save_path = save_path / filename
    ensure_dir(save_path.parent)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved error type stacked barplot to {save_path}")

if __name__ == "__main__":
    plot_error_type_stacked_by_series(pred_version="v0")