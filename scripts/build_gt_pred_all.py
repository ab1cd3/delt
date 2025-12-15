# scripts/build_gt_pred_all.py

import argparse

from model_eval.preprocessing.build_gt_pred_all import build_gt_pred_all
from model_eval.config import LATEST_PRED_VERSION

def main():
    parser = argparse.ArgumentParser(
        description="Build combined GT+PRED csv file (gt_pred_all.csv) for a given pred version.",
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
    print(f"[INFO] Building gt_pred_all.csv for pred_version='{version}'")
    df = build_gt_pred_all(pred_version=version)
    print(f"[INFO] Done. Combined rows: {len(df)}")


if __name__ == "__main__":
    main()