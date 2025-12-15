from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from IPython.core.pylabtools import figsize

from model_eval.experiments.results_utils import load_all_tree_runs
from model_eval.config import get_paths_for_version, ensure_dir, ERROR_COLORS, PRED_SOURCE
from model_eval.data.loaders import load_gt_pred_all_features


def plot_bytetrack_ablation(
        pred_version_base: str = "v0",
        pred_version_bt: str = "v0_bt",
        pred_labels: Optional[Sequence[str]] = ("YOLO (baseline)", "YOLO + ByteTrack"),
        label_mode: str = "tp_count",
        window_size: int = 150,
        window_step_ratio: int = 2,
        tp_count_thresh: Optional[int] = None,
        metrics: Sequence[str] = ("pr_auc", "recall"),
        add_display_names: bool = True,
        save_path: Optional[Path] = None,
) -> None:
    """
    Bar plots for ByteTrack ablation study (performance contribution).
    - Fix configuration: label_mode, window_size, step_size.
    - For each (model, pred_version, include_track_ids): keep the BEST run by pr_auc.
    - Plot bars for metrics (e.g. pr_auc, recall) grouped by model and version/track.

    Bars are grouped by model and colored by a version_track label:
        v0 no-id, v0 id, v0_bt no-id, v0_bt id
    """

    # Load + filter base dataframe
    pred_versions = [pred_version_base, pred_version_bt]

    df = load_all_tree_runs(
        pred_versions,
        add_display_names=add_display_names,
        pred_labels=pred_labels,
    ).copy()

    step_size = window_size // window_step_ratio

    df = df[
        (df["label_mode"] == label_mode)
        & (df["window_size"] == window_size)
        & (df["step_size"] == step_size)
        & (df["pred_version"].isin(pred_versions))
        & (df["pr_auc"].notna())
        & (df["recall"].notna())
    ].copy()

    if tp_count_thresh is not None:
        df = df[df["tp_count_thresh"] == tp_count_thresh]

    if df.empty:
        print("[PLOT] No rows left after filtering – nothing to plot.")
        return

    # Keep best config per (model, pred_version, include_track_ids)
    model_col = "model_label" if add_display_names else "model"
    group_cols = [model_col, "pred_version", "include_track_ids"]

    df_best = (
        df.sort_values(metrics[0], ascending=False)
        .groupby(group_cols, as_index=False)
        .head(1)
    )

    # -------------------------------------------------------
    # Plot: one subplot per metric (e.g. pr_auc, recall)
    # -------------------------------------------------------
    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(6 * n_metrics, 6),
        sharex=True,
    )

    if n_metrics == 1:
        axes = [axes]

    sns.set_style("whitegrid")
    sns.set_palette("Paired")

    for ax, metric in zip(axes, metrics):
        sns.barplot(
            data=df_best,
            x=model_col,
            y=metric,
            hue="pred_label_track" if add_display_names else "pred_version_track",
            ax=ax,
        )
        ax.set_ylabel(metric.replace("_", " ").upper())
        ax.set_xlabel("Model")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(
            f"{metric.upper()} – label_mode={label_mode}, "
            f"ws={window_size}, step={step_size}"
            + (f", tp_count_thresh={tp_count_thresh}" if tp_count_thresh is not None else "")
        )
        ax.legend(title="Version + track", loc="best")

    plt.tight_layout()

    # Save
    if save_path is None:
        # ByteTrack version as base for reports dir
        paths = get_paths_for_version(pred_version_bt)
        out_dir = paths.reports_experiments_dir / "tree"
        ensure_dir(out_dir)
        fname = (
            f"bytetrack_vs_base_ws{window_size}_step{step_size}_"
            f"{label_mode}_tp{tp_count_thresh if tp_count_thresh is not None else 'any'}.png"
        )
        save_path = out_dir / fname
    else:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)

    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[PLOT] Saved bytetrack bar plot to {save_path}")