# scripts/validate_data_checks.py

import argparse

from model_eval.validation.check_data import (
    check_series_have_gt_and_pred,
    check_gt_segments_have_video_folders
)
from model_eval.config import LATEST_PRED_VERSION


def main():
    parser = argparse.ArgumentParser(
        description="Runs data validation for a given pred version."
    )
    parser.add_argument(
        "--pred_version",
        type=str,
        default=None,
        help=(
            "Prediction version to use (e.g. 'v0', 'v1'). "
            "If omitted uses LATEST_PRED_VERSION from config.py "
            f"(currently '{LATEST_PRED_VERSION})."
        )
    )
    args = parser.parse_args()
    version = args.pred_version or LATEST_PRED_VERSION
    print(f"[INFO] Running data checks for pred_version='{version}'\n")

    check_series_have_gt_and_pred(pred_version=version)
    check_gt_segments_have_video_folders()

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()