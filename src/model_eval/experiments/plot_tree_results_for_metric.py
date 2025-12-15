# src/model_eval/experiments/plot_tree_results_for_metric.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from model_eval.experiments.results_utils import load_all_tree_runs
from model_eval.config import get_paths_for_version, ensure_dir


def plot_tree_results_for_metric(
        pred_version_1: str = "v0",
        pred_version_2: str = "v0_bt",
        pred_labels: Optional[Sequence[str]] = ("YOLO (baseline)", "YOLO + ByteTrack"),
        x_axis: str = "pr_auc",
        y_axis: str = "recall",
        color_col: str = "model_label",
        label_mode: str = "tp_count",
        size_col: str = "tp_count_thresh",
        window_step_ratio: int = 2,
        add_display_names: bool = True,
        save_path: Optional[Path] = None,
) -> None:
    """
    Scatter plot comparing two prediction versions:

        - x axis: x_axis  (e.g., pr_auc)
        - y axis: y_axis  (e.g., recall)
        - color: color_col (e.g., model)
        - marker SHAPE: pred_version + include_track_ids:
              v0_noid  -> 'o'
              v0_id    -> '*'
              v0_bt_noid -> 'v'
              v0_bt_id   -> '^'
        - size: size_col (e.g., tp_count_thresh)
    """
    # --------------------------------------------------
    # Load + filter
    # --------------------------------------------------
    pred_versions = [pred_version_1, pred_version_2]
    df = load_all_tree_runs(
        pred_versions,
        add_display_names=add_display_names,
        pred_labels=pred_labels,
    ).copy()

    df = df[
        (df["pred_version"].isin(pred_versions))
        & (df["label_mode"] == label_mode)
        & (df["step_size"] == df["window_size"] // window_step_ratio)
        & (df[x_axis].notna())
        & (df[y_axis].notna())
    ]

    # Top ranking groupby model + window_size
    df_rank1 = (
        df.sort_values([x_axis, y_axis], ascending=[False, False])
          .groupby(["model", "window_size"])
          .head(5)
    )
    df_rank2 = (
        df.sort_values([y_axis, x_axis], ascending=[False, False])
          .groupby(["model", "window_size"])
          .head(5)
    )
    df = pd.concat([df_rank1, df_rank2]).drop_duplicates().reset_index(drop=True)

    if df.empty:
        print("[PLOT] No data after filtering, nothing to plot.")
        return


    # Marker shapes per (version, track flag)

    version_track = "pred_label_track" if add_display_names else "pred_version_track"
    cats = list(df[version_track].cat.categories)

    markers = {
        cats[0]: "o",  # YOLO (baseline) no-id
        cats[1]: "*",  # YOLO (baseline) id
        cats[2]: "v",  # YOLO + ByteTrack no-id
        cats[3]: "^",  # YOLO + ByteTrack id
    }

    # Drop results not matching
    df = df[df[version_track].isin(markers.keys())]
    if df.empty:
        print("[PLOT] No rows matching version/track combinations; nothing to plot.")
        return

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure(figsize=(12, 10))

    ax = sns.scatterplot(
        data=df,
        x=x_axis,
        y=y_axis,
        hue=color_col,
        style=version_track,
        markers=markers,
        size=size_col,
        sizes=(150, 300),
        alpha=0.7,
        legend="full",
    )

    ax.set_xlabel(x_axis.upper())
    ax.set_ylabel(y_axis.upper())
    ax.set_title(
        f"Top-5 Configurations per Model: "
        f"{x_axis.replace('_', ' ').upper()} vs {y_axis.replace('_', ' ').title()}\n"
        f"Labeling: {label_mode}, Window/Step Ratio: {window_step_ratio}",
        fontsize=15,
    )
    for s in ax.spines.values():
        s.set_visible(False)

    # Save
    paths = get_paths_for_version(pred_version_2)
    filename = f"tree_results_{x_axis}_vs_{y_axis}_{size_col}_{pred_version_1}_{pred_version_2}_label.png"

    if save_path is None:
        save_path = paths.reports_experiments_dir / filename
    else:
        save_path = Path(save_path)
        if save_path.is_dir():
            save_path = save_path / filename

    ensure_dir(save_path.parent)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[PLOT] Saved results scatterplot to {save_path}")
