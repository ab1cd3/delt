# scripts/plot_analysis.py

import argparse

from model_eval.analysis_plots.plot_bbox_area import plot_bbox_area_violin_by_series
from model_eval.analysis_plots.plot_total_frames_by_gt_events import plot_total_frames_by_gt_events
from model_eval.analysis_plots.plot_errors_gt_vs_pred import plot_total_frames_gt_vs_pred_by_series
from model_eval.analysis_plots.plot_error_stats import plot_error_type_stacked_by_series
from model_eval.analysis_plots.plot_iou_vs_confidence import plot_iou_vs_confidence
from model_eval.analysis_plots.plot_precision_recall import plot_precision_recall_lines_by_series
from model_eval.analysis_plots.plot_event_confidence_timeline import plot_event_confidence_timeline
from model_eval.analysis_plots.plot_error_box_area_vs_confidence import plot_error_box_area_vs_confidence
from model_eval.analysis_plots.plot_tracks_vs_events import plot_tracks_vs_events
from model_eval.config import LATEST_PRED_VERSION, IOU_THRESHOLD

def main():
    parser = argparse.ArgumentParser(
        description="Plot the analysis visualizations for given prediction version.",
    )
    parser.add_argument(
        "--type",
        choices=[""],
        default=None,
        help=(
            "Prediction version to use (e.g. 'v0', 'v1'). "
            "If omitted uses LATEST_PRED_VERSION from config.py "
            f"(currently '{LATEST_PRED_VERSION})."
        ),
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
    print(f"[INFO] Building all analysis plots for pred_version='{version}'")

    plot_bbox_area_violin_by_series(pred_version=version)
    plot_total_frames_by_gt_events(pred_version=version)
    plot_total_frames_gt_vs_pred_by_series(pred_version=version)
    plot_error_type_stacked_by_series(pred_version=version)
    plot_iou_vs_confidence(pred_version=version)
    plot_precision_recall_lines_by_series(pred_version=version)
    plot_iou_vs_confidence_by_error(pred_version=version)
    plot_error_box_area_vs_confidence(pred_version=version)
    plot_tracks_vs_events(pred_version=version)

    # Timelines
    plot_event_confidence_timeline(
        video_series="SC01032019-A",
        segment_id="2880_0",
        iou_thresh=IOU_THRESHOLD,
        pred_version=version,
    )
    plot_event_confidence_timeline(
        video_series="DC20190403_B",
        segment_id="2580_0",
        iou_thresh=IOU_THRESHOLD,
        pred_version=version,
    )
    plot_event_confidence_timeline(
        video_series="SC01032019-A",
        segment_id="3480_0",
        iou_thresh=IOU_THRESHOLD,
        pred_version=version,
    )
    plot_event_confidence_timeline(
        video_series="DC20190403_B",
        segment_id="810_0",
        iou_thresh=IOU_THRESHOLD,
        pred_version=version,
    )
    plot_event_confidence_timeline(
        video_series="SC01032019-A",
        segment_id="5460_0",
        iou_thresh=IOU_THRESHOLD,
        pred_version=version,
    )
    plot_event_confidence_timeline(
        video_series="DC20190403_B",
        segment_id="450_0",
        iou_thresh=IOU_THRESHOLD,
        pred_version=version,
    )
    plot_event_confidence_timeline(
        video_series="RCE21072018-A",
        segment_id="450_0",
        iou_thresh=IOU_THRESHOLD,
        pred_version=version,
    )
    plot_event_confidence_timeline(
        video_series="RCE21072018-A",
        segment_id="7380_0",
        iou_thresh=IOU_THRESHOLD,
        pred_version=version,
    )
    plot_event_confidence_timeline(
        video_series="RCE21072018-A",
        segment_id="8190_0",
        iou_thresh=IOU_THRESHOLD,
        pred_version=version,
    )

if __name__ == "__main__":
    main()