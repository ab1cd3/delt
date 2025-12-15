# src/model_eval/analysis/plot_errors_gt_vs_pred.py

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model_eval.data.loaders import load_gt_pred_all_features
from model_eval.data.metadata import apply_series_labels
from model_eval.config import (
    get_paths_for_version,
    ensure_dir,
    GT_SOURCE,
    PRED_SOURCE,
    ERROR_COLORS,
)

GT_ERRORS = ["tp", "fn"]
PRED_ERRORS = ["tp", "fp", "bg"]


def plot_total_frames_gt_vs_pred_by_series(
        rename_labels: bool = True,
        save_path: Optional[Path] = None,
        pred_version: Optional[str] = None,
) -> None:
    """
    Side-by-side stacked bars per series:
      - left bar: GT (tp/fn)
      - right bar: PRED (tp/fp/bg)
    """
    df = load_gt_pred_all_features(pred_version=pred_version)
    if df.empty:
        raise ValueError("No data loaded from gt_pred_all_features.")

    df = df.copy()
    df["error_type"] = df["error_type"].astype("string").str.lower()

    # Labels
    if rename_labels:
        df = apply_series_labels(df)
        x_col = "series_label"
        order_col = "series_idx"
    else:
        x_col = "video_series"
        order_col = x_col

    # Order by order_col (if rename_labels: e.g., V1, V2, ... V17)
    order = (
        df[[x_col, order_col]]
        .drop_duplicates()
        .sort_values(order_col)[x_col]
        .tolist()
    )

    # Build counts per (series, error_type) separate by source
    gt = df[df["source"] == GT_SOURCE]
    pred = df[df["source"] == PRED_SOURCE]

    gt_counts = (
        pd.crosstab(index=gt[x_col], columns=gt["error_type"])
        .reindex(columns=GT_ERRORS, fill_value=0)
        .reindex(order, fill_value=0)
    )

    pred_counts = (
        pd.crosstab(index=pred[x_col], columns=pred["error_type"])
        .reindex(columns=PRED_ERRORS, fill_value=0)
        .reindex(order, fill_value=0)
    )

    if gt_counts.empty and pred_counts.empty:
        raise ValueError("After filtering, both GT and PRED counts are empty.")

    # ------------------------------------------------------------
    # Plot grouped stacked bars
    # ------------------------------------------------------------
    idx = gt_counts.index  # series labels after filtering
    x = np.arange(len(idx))
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 8))

    # GT bar (left)
    bottom = np.zeros(len(idx))
    for e in GT_ERRORS:
        vals = gt_counts[e].to_numpy()
        ax.bar(
            x - width / 2,
            vals,
            width=width,
            bottom=bottom,
            color=ERROR_COLORS[e],
            label=f"GT {e.upper()}",
        )
        bottom += vals

    # PRED bar (right)
    bottom = np.zeros(len(idx))
    for e in PRED_ERRORS:
        vals = pred_counts[e].to_numpy()
        ax.bar(
            x + width / 2,
            vals,
            width=width,
            bottom=bottom,
            color=ERROR_COLORS[e],
            label=f"PRED {e.upper()}",
        )
        bottom += vals

    # Formatting
    rot = 0 if rename_labels else 90
    ax.set_xticks(x)
    ax.set_xticklabels(idx, rotation=rot, ha="right")
    ax.set_xlabel("Video series")
    ax.set_ylabel("Box count", rotation=0)
    ax.yaxis.set_label_coords(-0.05, 0.95)

    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    for s in ax.spines.values():
        s.set_visible(False)

    fig.suptitle("Absolute Error Counts per Video Series", fontsize=15, weight="bold")
    ax.set_title("Ground Truth (left bar) vs Predictions (right bar)")

    # Save
    paths = get_paths_for_version(pred_version)
    filename = "total_errors_gt_vs_pred_by_series.png"

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
    print(f"[PLOT] Total error count GT vs PRED saved to {save_path}")


if __name__ == "__main__":
    plot_total_frames_gt_vs_pred_by_series(pred_version="v0")