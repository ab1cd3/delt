# scripts/tree_training.py

import argparse

from model_eval.experiments.dataset_utils import WindowConfig
from model_eval.experiments.tree_training import run_tree_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train tree-based baselines on window-level dataset.",
    )

    # Windowing
    parser.add_argument("--window_size", type=int, default=150,
                        help="Window size in frames (e.g. 150 = 5s at 30fps).")
    parser.add_argument("--step_size", type=int, default=60,
                        help="Stride between window starts in frames.")

    # Labeling mode
    parser.add_argument(
        "--label_mode",
        type=str,
        choices=["tp_count", "event_frac"],
        required=True,
        help=(
            "How to label a window as event (1) or non-event (0): "
            "'tp_count' -> uses --tp_count_thresh on TP count in the window; "
            "'event_frac' -> uses --event_frac_thresh on fraction of frames with is_event=1."
        ),
    )
    parser.add_argument(
        "--tp_count_thresh",
        type=int,
        default=3,
        help="Min TP count in the window to label it as event (used when label_mode='tp_count').",
    )
    parser.add_argument(
        "--event_frac_thresh",
        type=float,
        default=0.3,
        help="Min fraction of frames with is_event=1 to label window as event (label_mode='event_frac').",
    )

    # Predictions version (orig vs ByteTrack, etc.)
    parser.add_argument(
        "--pred_version",
        type=str,
        default=None,
        help="Prediction version (e.g. 'v0', 'v0_bt'). If None, uses latest default.",
    )

    # Train/val + evaluation
    parser.add_argument("--val_frac", type=float, default=0.3,
                        help="Fraction of windows (by series) used for validation.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Fallback decision threshold for y_prob if threshold search is disabled. "
            "If fit_and_eval_single uses best-threshold search, this is only used as default."
        ),
    )

    parser.add_argument(
        "--search_best_threshold",
        action="store_true",
        help="If set, searches the best metric evaluation threshold.",
    )

    # Hyperparameter search
    parser.add_argument(
        "--tune",
        action="store_true",
        help="If set, use RandomizedSearchCV to tune hyperparameters.",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=20,
        help="RandomizedSearchCV n_iter per model when --tune is enabled.",
    )

    # Saving
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="If set, append metrics to reports/.../<model_type>_all_runs.csv.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validation for event_frac_thresh
    if args.label_mode == "event_frac":
        if not (0.0 < args.event_frac_thresh <= 1.0):
            raise ValueError(
                f"event_frac_thresh must be in (0, 1], got {args.event_frac_thresh}"
            )

    cfg = WindowConfig(
        window_size=args.window_size,
        step_size=args.step_size,
        label_mode=args.label_mode,
        tp_count_thresh=args.tp_count_thresh,
        event_frac_thresh=args.event_frac_thresh,
        pred_version=args.pred_version,
    )

    print("[CONFIG] Using WindowConfig:")
    print(cfg)

    metrics_df = run_tree_baselines(
        cfg=cfg,
        val_frac=args.val_frac,
        threshold=args.threshold,
        search_best_threshold=args.search_best_threshold,
        tune=args.tune,
        n_iter=args.n_iter,
        save_results=args.save_results,
    )

    print("\n=== Summary metrics (validation) ===")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()