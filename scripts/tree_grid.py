# scripts/tree_grid.py

from __future__ import annotations

from model_eval.experiments.dataset_utils import WindowConfig
from model_eval.experiments.tree_training import run_tree_baselines


def main() -> None:
    # Global defaults for the sweep
    val_frac = 0.3
    n_iter = 20                     # RandomizedSearchCV iterations
    base_threshold = 0.5            # only used if search_best_threshold=False
    search_best_threshold = True    # let each model pick its best metric threshold
    tune = True                     # set False if too slow
    save_results = True             # append to tree_all_runs.csv

    # Grid options
    pred_versions = ["v0", "v0_bt"]              # orig vs ByteTrack
    include_track_options = [False, True]        # track-agnostic vs using track_id*
    window_sizes = [60, 90, 120, 150]
    label_modes = ["tp_count", "event_frac"]     # labeling strategies
    thresholds = {
        "tp_count" : [1, 2, 3, 5],
        "event_frac": [0.1, 0.2, 0.3, 0.5]
    }

    # ---------------------------
    # Grid loop
    # ---------------------------
    # tune
    for pred_version in pred_versions:
        for include_track in include_track_options:
            for label_mode in label_modes:
                for label_thresh in thresholds[label_mode]:
                    for ws in window_sizes:
                        step = ws // 2  # stride

                        cfg = WindowConfig(
                            window_size=ws,
                            step_size=step,
                            label_mode=label_mode,
                            tp_count_thresh=label_thresh if label_mode == "tp_count" else None,
                            event_frac_thresh=label_thresh if label_mode == "event_frac" else None,
                            pred_version=pred_version,
                            include_track_ids=include_track,
                        )

                        print(
                            "\n=== RUN ===\n"
                            f"  pred_version      = {pred_version}\n"
                            f"  include_track_ids = {include_track}\n"
                            f"  label_mode        = {label_mode}\n"
                            f"  label_thresh      = {label_thresh}\n"
                            f"  window_size       = {ws}\n"
                            f"  step_size         = {step}\n"
                            f"  val_frac          = {val_frac}\n"
                            f"  tune              = {tune}\n"
                            f"  search_best_thr   = {search_best_threshold}\n"
                        )

                        metrics_df = run_tree_baselines(
                            cfg=cfg,
                            val_frac=val_frac,
                            threshold=base_threshold,
                            search_best_threshold=search_best_threshold,
                            tune=tune,
                            n_iter=n_iter,
                            save_results=save_results,
                            return_models=False,
                        )

                        print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()