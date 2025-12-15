# scripts/plot_bytetrack.py

import argparse

from model_eval.experiments.plot_bytetrack_filtering import plot_bytetrack_filtering
from model_eval.experiments.plot_bytetrack_ablation import plot_bytetrack_ablation

def main():
    parser = argparse.ArgumentParser(
        description="Plot ByteTrack performance for a given base prediction version.",
    )
    parser.add_argument(
        "--bt_pred_version",
        type=str,
        default="v0_bt",
        help="Pred version for ByteTrack (default: v0_bt)"
    )

    args = parser.parse_args()
    # Plot: filtered prediction count by ByteTrack (barchart)
    plot_bytetrack_filtering(
        bt_version=args.bt_pred_version,
    )

    # Plot: best model metric difference base vs bt
    base = args.bt_pred_version.replace("_bt", "")
    plot_bytetrack_ablation(
        pred_version_base=base,
        pred_version_bt=args.bt_pred_version,
        label_mode="tp_count",
        window_size=150,
        window_step_ratio=2,
        tp_count_thresh=None,
        metrics=("pr_auc", "recall"),
    )


if __name__ == "__main__":
    main()