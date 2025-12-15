# scripts/build_bytetrack.py

from __future__ import annotations
import argparse

from model_eval.experiments.bytetrack_bt import add_bytetrack_to_all_series
from model_eval.config import LATEST_PRED_VERSION


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create ByteTrack predictions for a given base prediction version.",
    )
    parser.add_argument(
        "--pred_version",
        type=str,
        default=LATEST_PRED_VERSION,
        help="Base prediction version (e.g. v0)"
    )
    parser.add_argument(
        "--bt_pred_version",
        type=str,
        default=None,
        help="Output pred version for ByteTrack (default: <base>_bt)"
    )

    args = parser.parse_args()

    add_bytetrack_to_all_series(
        base_pred_version=args.pred_version,
        bt_pred_version=args.bt_pred_version,
    )


if __name__ == "__main__":
    main()