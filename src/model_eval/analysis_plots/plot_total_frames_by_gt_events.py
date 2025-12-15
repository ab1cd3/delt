# src/model_eval/analysis/plot_total_frames_by_gt_events.py

from pathlib import Path
import matplotlib.pyplot as plt

from model_eval.data.loaders import load_std_gt_all_series
from model_eval.data.metadata import apply_series_event_labels
from model_eval.config import (
    get_paths_for_version,
    ensure_dir,
    GT_COLOR,
)


def plot_total_frames_by_gt_events(
        rename_labels: bool = True,
        save_path: Path | None = None,
        pred_version: str | None = None,
) -> None:
    """
    Scatter plot of GT event lengths (frames per segment).

    X-axis: video series (or series_label if rename_labels=True),
            one x position per series.
    Y-axis: number of GT frames for each event.
    Each point labeled with event_label (E1, E2...) or segment_id.
    """
    df_gt = load_std_gt_all_series()

    # Count frames per event (video_series + segment_id)
    event_lengths = (
        df_gt.groupby(["video_series", "segment_id"], as_index=False)["frame"]
        .count()
        .rename(columns={"frame": "frame_count"})
    )

    # 2) Label renaming (series/event labels)
    if rename_labels:
        event_lengths = apply_series_event_labels(event_lengths)
        x_col = "series_label"
        label_col = "event_label"
        sort_col = "series_idx"
    else:
        x_col = "video_series"
        label_col = "segment_id"
        sort_col = "video_series"

    # Sort by series (numeric idx if available)
    event_lengths = event_lengths.sort_values(sort_col)

    # Build unique x labels per series (NOT per event)
    if rename_labels and "series_idx" in event_lengths.columns:
        # Ensure ordering by series_idx, then take unique series_label
        series_order = (
            event_lengths[[x_col, "series_idx"]]
            .drop_duplicates()
            .sort_values("series_idx")
        )
        x_labels = list(series_order[x_col])
    else:
        x_labels = sorted(event_lengths[x_col].unique())

    # Map each series label to a fixed x position
    x_pos_map = {label: i for i, label in enumerate(x_labels)}
    event_lengths["x_pos"] = event_lengths[x_col].map(x_pos_map)

    # Plot
    fig, ax = plt.subplots(figsize=(max(6, len(x_labels) * 0.8), 5))

    ax.scatter(
        event_lengths["x_pos"],
        event_lengths["frame_count"],
        color=GT_COLOR,
        s=60,
        alpha=0.8,
    )

    # Label each point
    for _, row in event_lengths.iterrows():
        ax.text(
            row["x_pos"],
            row["frame_count"],
            str(row[label_col]),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # X-axis: one tick per series
    x_rotation = 0 if rename_labels else 90
    ax.set_xticks(list(x_pos_map.values()))
    ax.set_xticklabels(x_labels, rotation=x_rotation)

    ax.set_xlabel("Video Series")
    ax.set_ylabel("GT Frames per Event")
    ax.set_title("GT Event Lengths (frames per segment/event)")

    ax.grid(True, axis="y", alpha=0.3)

    # Save
    paths = get_paths_for_version(pred_version)
    filename = "total_frames_by_gt_events.png"

    if save_path is None:
        save_path = paths.reports_analysis_dir / filename
    else:
        if save_path.is_dir():
            save_path = save_path / filename

    ensure_dir(save_path.parent)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"[PLOT] Saved GT event frame count scatter plot to {save_path}")