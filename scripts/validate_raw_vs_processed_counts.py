# scripts/validate_raw_vs_processed_counts.py

import argparse

from model_eval.validation.check_raw_vs_processed_counts import compare_raw_and_processed_counts
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
    print(f"[INFO] Running raw vs processed count validation for pred_version='{version}'\n")

    compare_raw_and_processed_counts(pred_version=version)

    print("\nAll checks passed.")

if __name__ == "__main__":
    main()