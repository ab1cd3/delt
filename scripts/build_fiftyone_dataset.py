# scripts/build_fiftyone_dataset.py

import argparse

from model_eval.analysis.build_fiftyone_dataset import build_fiftyone_dataset
from model_eval.config import LATEST_PRED_VERSION

def main():

    parser = argparse.ArgumentParser(
        description="Build a FiftyOne dataset.",
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
    print(f"[INFO] Building FiftyOne dataset for pred_version='{version}'")
    ds = build_fiftyone_dataset(pred_version=version)
    print(ds)


if __name__ == "__main__":
    main()