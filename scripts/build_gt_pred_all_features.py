# scripts/build_gt_pred_all_features.py

import argparse

from model_eval.analysis.build_gt_pred_all_features import build_gt_pred_all_features
from model_eval.config import IOU_THRESHOLD, LATEST_PRED_VERSION

def main():
    parser = argparse.ArgumentParser(
        description="Build gt_pred_all_features.csv (static geometry + pred error labels).",
    )
    parser.add_argument(
        "--pred_version",
        type=str,
        default=None,
        help=(
            "Prediction version to use (e.g. 'v0', 'v1'). "
            f"If omitted uses LATEST_PRED_VERSION (currently '{LATEST_PRED_VERSION}')."
        ),
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=IOU_THRESHOLD,
        help="IoU threshold for error_type (tp/fp/bg).",
    )

    args = parser.parse_args()

    version = args.pred_version or LATEST_PRED_VERSION
    print(f"[INFO] Building gt_pred_features.csv for pred_version={version}, iou={args.iou}")

    df = build_gt_pred_all_features(pred_version=version, iou_thresh=args.iou)
    print(f"[INFO] Done. Rows: {len(df)}")


if __name__ == "__main__":
    main()