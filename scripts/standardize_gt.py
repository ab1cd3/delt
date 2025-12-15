# scripts/standardize_gt.py

import argparse

from model_eval.preprocessing.gt_standardization import (
    standardize_gt_for_series,
    standardize_all_gt,
)


def main():
    parser = argparse.ArgumentParser(
        description="Standardize GT txt files into CSV per segment and per series.",
    )
    parser.add_argument(
        "--series",
        type=str,
        default=None,
        help="Video series name (e.g. DC13112018-B). If omitted, process all.",
    )
    args = parser.parse_args()

    if args.series:
        standardize_gt_for_series(args.series)
    else:
        standardize_all_gt()


if __name__ == "__main__":
    main()