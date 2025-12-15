# scripts/standardize_pred.py

import argparse

from model_eval.preprocessing.pred_standardization import (
    standardize_predictions_for_series,
    standardize_all_predictions
)
from model_eval.config import LATEST_PRED_VERSION


def main():
    parser = argparse.ArgumentParser(
        description="Standardize prediction txt files into CSV per video series."
    )
    parser.add_argument(
        "--series",
        type=str,
        default=None,
        help="Video series name (e.g. DC13112018-B). If omitted, process all.",
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
    if args.series:
        print(f"[INFO] Standardizing PRED series:{args.series} (pred_version='{version}')")
        standardize_predictions_for_series(args.series, pred_version=version)
    else:
        print(f"[INFO] Standardizing all PRED files (pred_version='{version}')")
        standardize_all_predictions(pred_version=version)


if __name__ == "__main__":
    main()