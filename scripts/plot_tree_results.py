# scripts/plot_tree_results.py

import argparse

from model_eval.experiments.plot_tree_results_for_metric import plot_tree_results_for_metric


def main():
    parser = argparse.ArgumentParser(
        description="Plot decision results for two given prediction versions.",
    )
    parser.add_argument(
        "--pred_version_1",
        type=str,
        default="v0",
        help="Pred version option that has a tree_all_runs.csv results file."
    )

    parser.add_argument(
        "--pred_version_2",
        type=str,
        default="v0_bt",
        help="Pred version option that has a tree_all_runs.csv results file."
    )

    args = parser.parse_args()
    plot_tree_results_for_metric(
        pred_version_1=args.pred_version_1,
        pred_version_2=args.pred_version_2,
        add_display_names=True,
    )


if __name__ == "__main__":
    main()