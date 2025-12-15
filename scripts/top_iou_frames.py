# scripts/top_iou_frames.py

import argparse

from model_eval.analysis.top_iou_frames import (
    get_top_iou_per_series,
    get_top_iou_global,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Show top-K predictions by IoU, either per series or globally. "
            "Uses data/processed/pred_iou.csv (built by build_pred_iou.py)."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["series", "global"],
        default="series",
        help="Aggregation mode: 'series' (top-K per series) or 'global' (top-K overall). "
             "Default: series.",
    )
    parser.add_argument(
        "--series",
        type=str,
        nargs="*",
        default=None,
        help="Optional video series name(s) to filter, e.g. --series DC13112018-B HS04062019-B. "
             "If omitted and mode=series, all series are included.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of predictions to keep (top-K). Default: 5.",
    )

    args = parser.parse_args()

    if args.mode == "series":
        df = get_top_iou_per_series(
            top_k=args.top,
            video_series=args.series,
        )
    else:  # global
        df = get_top_iou_global(
            top_k=args.top,
            video_series=args.series,
        )

    if df.empty:
        print("No predictions found (check pred_iou.csv or filters).")
        return

    # Nice console print
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()