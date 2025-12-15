# scripts/build_pred_iou.py

import argparse

from model_eval.analysis.build_pred_iou import build_pred_iou_table
from model_eval.config import LATEST_PRED_VERSION


def main():
    parser = argparse.ArgumentParser(
        description="Compute IoU for each prediction and save as pred_iou.csv.",
    )
    parser.add_argument(
        "--series",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of video_series to restrict (e.g. DC13112018-B).",
    )
    parser.add_argument(
        "--pred_version",
        type=str,
        default=None,
        help=(
            "Prediction version to use (e.g. 'v0', 'v1'). "
            "If omitted uses LATEST_PRED_VERSION from config.py "
            f"(currently '{LATEST_PRED_VERSION})."
        ),
    )
    args = parser.parse_args()

    version = args.pred_version or LATEST_PRED_VERSION
    print(f"[INFO] Building pred_iou.csv for pred_version='{version}'")
    df = build_pred_iou_table(
        video_series=args.series,
        pred_version=version,
    )
    print(f"[INFO] Done. Combined rows: {len(df)}")


if __name__ == "__main__":
    main()