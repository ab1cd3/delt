# src/model_eval/analysis/plot_error_box_area_vs_confidence.py

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from model_eval.data.loaders import load_gt_pred_all_features
from model_eval.config import (
    get_paths_for_version,
    ensure_dir,
    ERROR_COLORS,
    IOU_THRESHOLD,
)
from model_eval.utils.constants import PRED_SOURCE


def plot_error_box_area_vs_confidence(
        iou_thresh: float = IOU_THRESHOLD,
        save_path: Path | None = None,
        pred_version: str | None = None,
        log_area: bool = True,
) -> None:
    """
    Scatter plot for predictions:
        - x-axis: confidence (0–1)
        - y-axis: box area (pixels, optionally log-scaled)
        - color: error_type (tp / fp / bg / fn)
    """
    # Load enriched features (includes area_px + error_type)
    df = load_gt_pred_all_features(iou_thresh=iou_thresh, pred_version=pred_version)

    # Predictions only
    df_pred = df[df["source"] == PRED_SOURCE].copy()

    # Drop rows without valid confidence or area
    df_pred = df_pred.dropna(subset=["confidence", "area_px", "error_type"])
    if df_pred.empty:
        raise ValueError(
            f"No prediction rows with features for pred_version={pred_version}, "
            f"iou_thresh={iou_thresh}"
        )

    # Optionally log-scale area so very large boxes don't dominate
    if log_area:
        df_pred = df_pred[df_pred["area_px"] > 0]
        df_pred["area_metric"] = df_pred["area_px"].apply(lambda a: a ** 0.5)
        y_label = "sqrt(box area) (pixels^0.5)"
    else:
        df_pred["area_metric"] = df_pred["area_px"]
        y_label = "Box area (pixels)"

    # Map error_type -> color
    colors = df_pred["error_type"].str.lower().map(ERROR_COLORS).fillna("gray")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        df_pred["confidence"],
        df_pred["area_metric"],
        c=colors,
        alpha=0.4,
        s=10,
    )

    ax.set_xlabel("Confidence")
    ax.set_ylabel(y_label)
    ax.set_title(f"Box area vs confidence by error type (IoU ≥ {iou_thresh:.2f})")

    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    for s in ax.spines.values():
        s.set_visible(False)

    # Legend
    legend_handles = [
        Patch(color=ERROR_COLORS[error], label=error.upper()) for error in df_pred["error_type"].unique()
    ]
    ax.legend(handles=legend_handles, title="Error type")

    # Save
    paths = get_paths_for_version(pred_version)
    iou_str = f"{iou_thresh:.2f}".replace(".", "_")
    filename = f"error_box_area_vs_confidence_iou_{iou_str}.png"

    if save_path is None:
        save_path = paths.reports_analysis_dir / filename
    else:
        if save_path.is_dir():
            save_path = save_path / filename

    ensure_dir(save_path.parent)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"[PLOT] Saved box area vs confidence by error type scatterplot to {save_path}")