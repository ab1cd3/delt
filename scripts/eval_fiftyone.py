# scripts/eval_fiftyone.py

import argparse
from typing import Tuple, List, Optional

from model_eval.analysis.eval_fiftyone import (
    eval_fiftyone_global,
    eval_fiftyone_per_series,
    eval_fiftyone_per_event,
    launch_fiftyone_app
)
from model_eval.config import IOU_THRESHOLD, LATEST_PRED_VERSION


def parse_event_arg(ev: str) -> Tuple[str, str]:
    """
    Parse an event string into (video_series, segment_id).

    Example:
        'DC27092018-B/1470_0' -> ('DC27092018-B', '1470_0')
    """
    parts = ev.split("/", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid --event format: {ev}. Expected '<video_series>/<segment_id>'"
        )
    return parts[0], parts[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FiftyOne COCO-style evaluation (global / series / event).",
    )

    parser.add_argument(
        "--level",
        type=str,
        choices=["global", "series", "event"],
        default="global",
        required=True,
        help="Evaluation level: 'global', 'series', or 'event'.",
    )

    parser.add_argument(
        "--series",
        type=str,
        nargs="*",
        default=None,
        help="Optional video series name(s) to evaluate "
             "(used for level=series and as a filter for level=event).",
    )

    parser.add_argument(
        "--event",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional event ID(s) for level=event, format '<video_series>/<segment_id>', "
            "e.g. 'DC27092018-B/1470_0'. "
            "If omitted, all events are evaluated (optionally filtered by --series)."
        ),
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=IOU_THRESHOLD,
        help=f"IoU threshold for evaluation (default: {IOU_THRESHOLD}).",
    )

    parser.add_argument(
        "--launch_app",
        action="store_true",
        help="Launch the FiftyOne App after evaluation.",
    )

    # Print / no-print flags (default: print)
    parser.add_argument(
        "--no_print_reports",
        dest="print_reports",
        action="store_false",
        help="Disable printing textual evaluation reports.",
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
    print(f"[INFO] Computing FiftyOne eval stats for pred_version='{version}'")

    # LEVELS
    if args.level == "global":
        df = eval_fiftyone_global(
            iou_thresh=args.iou,
            print_reports=args.print_reports,
            pred_version=version,
        )

    elif args.level == "series":
        # Only evaluate the series provided (or all if None)
        df = eval_fiftyone_per_series(
            video_series=args.series,
            iou_thresh=args.iou,
            print_reports=args.print_reports,
            pred_version=version,
        )

    else:  # level == "event"
        events: Optional[List[Tuple[str, str]]]

        if args.event:
            # If --event is provided (S1/seg1) -> restrict to those events
            events = [parse_event_arg(ev) for ev in args.event]
            video_series_filter = None  # no extra series filter
        else:
            # No explicit events (--event --series not used) -> evaluate all events
            # If --series is provided -> restrict to all events of those series
            events = None
            video_series_filter = args.series

        df = eval_fiftyone_per_event(
            events=events,
            video_series=video_series_filter,
            iou_thresh=args.iou,
            print_reports=args.print_reports,
            pred_version=version,
        )

    print("\n=== SUMMARY DATAFRAME ===")
    print(df)

    if args.launch_app:
        launch_fiftyone_app(pred_version=version)

if __name__ == "__main__":
    main()