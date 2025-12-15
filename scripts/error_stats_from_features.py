# scripts/error_stats_from_features.py

import argparse

from model_eval.analysis.error_stats_from_features import (
    compute_error_counts_per_event,
    compute_error_counts_per_series,
)
from model_eval.config import IOU_THRESHOLD, LATEST_PRED_VERSION


def main():
    parser = argparse.ArgumentParser(
        description="Compute TP/FP/FN error counts for MODEL PRED vs GT."
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=IOU_THRESHOLD,
        help=f"IoU threshold for TP (default: {IOU_THRESHOLD})"
    )
    parser.add_argument(
        "--series",
        type=str,
        nargs="*",
        default=None,
        help="Optional video series name(s) to filter (e.g. DC13112018-B)."
    )
    parser.add_argument(
        "--segment",
        type=str,
        nargs="*",
        default=None,
        help="Optional segment_id(s) to filter when level='event' (e.g. 1470_0)."
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["event", "series"],
        default="series",
        help="Aggregation level: 'event' or 'series' (default: series)."
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
    print(f"[INFO] Computing error stats for pred_version='{version}'")

    if args.level == "event":
        df = compute_error_counts_per_event(
            iou_thresh=args.iou,
            video_series=args.series,
            segment_id=args.segment,
            pred_version=version,
        )
    else:
        df = compute_error_counts_per_series(
            iou_thresh=args.iou,
            video_series=args.series,
            pred_version=version,
        )

    print(f"\nERROR STATS (IoU = {args.iou})")
    print(df)


if __name__ == "__main__":
    main()