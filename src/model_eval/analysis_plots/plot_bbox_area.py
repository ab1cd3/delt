# src/model_eval/analysis/plot_bbox_area.py

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from model_eval.data.loaders import load_gt_pred_all
from model_eval.data.metadata import apply_series_event_labels
from model_eval.utils.bbox import add_bbox_area
from model_eval.config import (
    get_paths_for_version,
    ensure_dir,
    GT_SOURCE,
    PRED_SOURCE,
    GT_COLOR,
    PRED_COLOR,
)


def plot_bbox_area_violin_by_series(
        rename_labels: bool = True,
        save_path: Path | None = None,
        log_scale: bool = True,
        pred_version: str | None = None,
) -> None:
    """
    Create a violin plot comparing bbox area distributions between GT and PRED
    per video series.

    X-axis: video series (or series_label if rename_labels=True).
    For each series: two overlapped violins (GT, PRED) with transparency.
    """
    df = load_gt_pred_all(pred_version)
    if df.empty:
        raise ValueError(f"No data loaded from gt_pred_all.csv (pred_version={pred_version}).")

    # Optional sanity check on source values
    print("Unique source values in data:", df["source"].unique())
    print("GT_SOURCE:", GT_SOURCE, "PRED_SOURCE:", PRED_SOURCE)

    # Label metadata
    if rename_labels:
        df = apply_series_event_labels(df)
        label_col = "series_label"
    else:
        label_col = "video_series"

    # Add bbox area
    df = add_bbox_area(df, area_col="area")

    # Determine label order
    if rename_labels and "series_idx" in df.columns:
        order_df = (
            df[[label_col, "series_idx"]]
            .drop_duplicates()
            .sort_values("series_idx")
        )
        labels = list(order_df[label_col])
    else:
        labels = sorted(df[label_col].dropna().unique())

    if not labels:
        raise ValueError(f"No non-null values found in '{label_col}'.")

    # Helper: draw one violin
    def draw_violin(ax, data, x_pos, color, median_color):
        if data.empty:
            return
        parts = ax.violinplot(
            [data.values],
            positions=[x_pos],
            widths=0.5,
            showmedians=True,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.5)
            body.set_edgecolor(median_color)
        if "cmedians" in parts:
            parts["cmedians"].set_color(median_color)

    # Figure size scales with number of series
    fig, ax = plt.subplots(figsize=(max(6, int(len(labels) * 0.8)), 5))

    x_positions = range(1, len(labels) + 1)

    for x, label in zip(x_positions, labels):
        df_series = df[df[label_col] == label]

        gt_area = df_series.loc[df_series["source"] == GT_SOURCE, "area"].dropna()
        pred_area = df_series.loc[df_series["source"] == PRED_SOURCE, "area"].dropna()

        # Overlapped violins at same x position
        draw_violin(ax, gt_area, x, GT_COLOR, "darkgreen")
        draw_violin(ax, pred_area, x, PRED_COLOR, "darkblue")

    # X-axis labels
    x_rotation = 0 if rename_labels else 90
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=x_rotation, ha="right")

    ax.set_ylabel("Bounding box area (pixels²)")
    ax.set_title("BBox area distribution per series: GT vs PRED")

    if log_scale:
        ax.set_yscale("log")  # usually helpful for area distributions

    ax.grid(True, axis="y", alpha=0.3)

    # Legend
    legend_handles = [
        Patch(facecolor=GT_COLOR, alpha=0.5, label="GT"),
        Patch(facecolor=PRED_COLOR, alpha=0.5, label="PRED"),
    ]
    ax.legend(handles=legend_handles, title="Source", loc="upper right")

    # Save
    paths = get_paths_for_version(pred_version)
    ensure_dir(paths.reports_analysis_dir)
    filename = "bbox_area_violin_by_series.png"
    if save_path is None:
        save_path = paths.reports_analysis_dir / filename
    else:
        if save_path.is_dir():
            save_path = save_path / filename
    ensure_dir(save_path.parent)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"[PLOT] Saved bbox area violin plot to {save_path}")