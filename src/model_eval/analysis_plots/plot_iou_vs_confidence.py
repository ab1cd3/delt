# src/model_eval/analysis/plot_iou_vs_confidence.py

from pathlib import Path

import matplotlib.pyplot as plt

from model_eval.data.loaders import load_pred_iou
from model_eval.config import get_paths_for_version, ensure_dir


def plot_iou_vs_confidence(
        save_path: Path | None = None,
        drop_zero_iou: bool = False,
        pred_version: str | None = None,
) -> None:
    """
    Scatterplot of IoU vs confidence for all predictions in pred_iou.csv.

    X-axis: confidence
    Y-axis: IoU (best IoU vs any GT in that frame)

    If drop_zero_iou=True, filter out rows with iou == 0.0
    (frames with no overlap / no GT).
    """
    df = load_pred_iou(pred_version)
    if df.empty:
        raise ValueError("pred_iou.csv is empty or not found. Build it first.")

    df = df[["confidence", "iou"]].dropna()

    if drop_zero_iou:
        df = df[df["iou"] > 0.0]

    if df.empty:
        raise ValueError("No valid (confidence, iou) pairs after filtering.")

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        df["confidence"],
        df["iou"],
        s=10,
        alpha=0.3,
    )

    ax.set_xlabel("Confidence")
    ax.set_ylabel("IoU (best vs GT per frame)")
    ax.set_title("IoU vs confidence for all predictions")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    # Save
    paths = get_paths_for_version(pred_version)
    filename = "iou_vs_confidence.png"

    if save_path is None:
        save_path = paths.reports_analysis_dir / filename
    else:
        if save_path.is_dir():
            save_path = save_path / filename

    ensure_dir(save_path.parent)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"[PLOT] Saved IoU vs confidence scatterplot to {save_path}")