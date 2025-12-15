# src/model_eval/analysis/eval_fiftyone.py

from __future__ import annotations

import re
from typing import List, Iterable, Tuple

import fiftyone as fo
from fiftyone import ViewField as F
import pandas as pd

from model_eval.analysis.build_fiftyone_dataset import build_fiftyone_dataset
from model_eval.config import (
    FIFTYONE_DATASET_NAME,
    FIFTYONE_CLASS_NAME,
    IOU_THRESHOLD,
    LATEST_PRED_VERSION,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def launch_fiftyone_app(
        dataset: str | None = None,
        pred_version: str | None = None,
) -> None:
    """Launches the FiftyOne App in the browser using the full dataset."""
    ds = get_dataset(dataset, pred_version)
    ds.create_index([("video_series", 1), ("image_segment_id", 1)])
    ds.create_index([("event_id", 1)])
    ds.create_index([("event_label", 1)])
    session = fo.launch_app(ds)
    session.wait()


def _clean_eval_key_component(s: str) -> str:
    """Replace non-alphanumeric with '_' to safely use as a FiftyOne eval_key."""
    s = re.sub(r"\W+", "_", str(s))
    if not re.match(r"[A-Za-z_]", s):
        s = "k_" + s
    return s


def make_eval_key(prefix: str, iou_thresh: float) -> str:
    """
    Build a stable eval_key name from a prefix and an IoU threshold.

    Example:
      prefix="global" -> "global_coco_0_30" for iou_thresh=0.30
    """
    iou_str = f"{iou_thresh:.2f}".replace(".", "_")  # 0.3 -> "0_30"
    prefix = _clean_eval_key_component(prefix)
    return f"{prefix}_coco_{iou_str}"


def get_dataset(
        name: str | None = None,
        pred_version: str | None = None,
) -> fo.Dataset:
    """
    Load an existing FiftyOne dataset, or build it if missing.
    """
    dataset_name = name or FIFTYONE_DATASET_NAME
    version = pred_version or LATEST_PRED_VERSION
    dataset = f"{dataset_name}_{version}"

    try:
        return fo.load_dataset(dataset)
    except fo.core.dataset.DatasetNotFoundError:
        print(f"[INFO] Dataset '{name}' not found, building it...")
        return build_fiftyone_dataset(name=name, pred_version=pred_version)


def make_unique_frame_view(view: fo.DatasetView) -> fo.DatasetView:
    """
    Return a view with at most one sample per (video_series, frame),
    to avoid double-counting frames that appear under multiple segment_ids.
    The chosen sample per (series, frame) is first in sorted order.
    """
    seen: set[tuple[str, int]] = set()
    keep_ids: list[str] = []

    for sample in view:
        key = (sample.video_series, int(sample.frame))
        if key in seen:
            continue
        seen.add(key)
        keep_ids.append(sample.id)

    return view.select(keep_ids)


def evaluate_view(
        view: fo.DatasetView,
        eval_key: str,
        iou_thresh: float,
        class_name: str = FIFTYONE_CLASS_NAME,
) -> tuple[dict, fo.utils.eval.detection.DetectionResults]:
    """
    Run COCO evaluation on a view and return:
      - summary: dict with counts + metrics for 'class_name'.
      - results: the FiftyOne DetectionResults object.
    """
    view = make_unique_frame_view(view)
    results = view.evaluate_detections(
        "predictions",
        gt_field="gt",
        eval_key=eval_key,
        method="coco",
        iou=iou_thresh,
        classes=[class_name],
    )

    # report is a dict of dicts ({"turtle": {...}, "micro avg": {...}, ...})
    report = results.report()
    metrics = report.get(class_name, next(iter(report.values())))

    tp = int(view.sum(f"{eval_key}_tp"))
    fp = int(view.sum(f"{eval_key}_fp"))
    fn = int(view.sum(f"{eval_key}_fn"))

    summary = {
        "label": class_name,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1-score"]),
        "support": int(metrics["support"]),
        "eval_key": eval_key,
    }

    return summary, results


# -------------------------------------------------------------------
# Global / per-series / per-event evaluations
# -------------------------------------------------------------------

def eval_fiftyone_global(
        dataset: str | None = None,
        iou_thresh: float = IOU_THRESHOLD,
        print_reports: bool = True,
        pred_version: str | None = None,
) -> pd.DataFrame:
    """
    Run COCO-style evaluation on the full dataset and return a one-row DataFrame.

    Returns DataFrame with columns:
        label, tp, fp, fn, precision, recall, f1, support, eval_key
    """
    ds = get_dataset(dataset, pred_version)

    eval_key = make_eval_key("global", iou_thresh)
    summary, results = evaluate_view(
        view=ds,
        eval_key=eval_key,
        iou_thresh=iou_thresh,
        class_name=FIFTYONE_CLASS_NAME,
    )

    if print_reports:
        print(f"=== GLOBAL METRICS (pred_version={pred_version}) ===")
        results.print_report(classes=[FIFTYONE_CLASS_NAME])

    ds.save()

    return pd.DataFrame([summary])


def eval_fiftyone_per_series(
        dataset: str | None = None,
        iou_thresh: float = IOU_THRESHOLD,
        video_series: Iterable[str] | None = None,
        print_reports: bool = True,
        pred_version: str | None = None,
) -> pd.DataFrame:
    """
    Run separate COCO evaluations for each video_series.

    If 'video_series' is None -> evaluate all series in the dataset.

    Returns DataFrame with columns:
      video_series, label, tp, fp, fn, precision, recall, f1, support, eval_key
    """
    ds = get_dataset(dataset, pred_version)

    if video_series is None:
        series_values = sorted(ds.distinct("video_series"))
    else:
        series_values = sorted(set(video_series))

    rows: List[dict] = []

    if print_reports:
        print(f"\n=== PER VIDEO_SERIES METRICS (pred_version={pred_version}) ===")

    for series in series_values:
        view = ds.match(F("video_series") == series)
        eval_key = make_eval_key(f"series_{series}", iou_thresh)

        summary, results = evaluate_view(
            view=view,
            eval_key=eval_key,
            iou_thresh=iou_thresh,
            class_name=FIFTYONE_CLASS_NAME,
        )
        summary["video_series"] = series
        rows.append(summary)

        if print_reports:
            print(f"\n--- Series: {series} ---")
            results.print_report(classes=[FIFTYONE_CLASS_NAME])

    ds.save()

    df = pd.DataFrame(rows)
    return df[
        ["video_series", "label", "tp", "fp", "fn",
         "precision", "recall", "f1", "support", "eval_key"]
    ]


def eval_fiftyone_per_event(
        dataset: str | None = None,
        iou_thresh: float = IOU_THRESHOLD,
        video_series: Iterable[str] | None = None,
        events: Iterable[Tuple[str, str]] | None = None,
        print_reports: bool = True,
        pred_version: str | None = None,
) -> pd.DataFrame:
    """
    Run separate COCO evaluations for each event (video_series, segment_id).

    Priority of what to evaluate:
      - If 'events' is not None: evaluate exactly those (series, segment_id) pairs.
      - Else if 'video_series' is not None: evaluate all events for those series.
      - Else: evaluate all events in the dataset.

    Returns DataFrame with:
      video_series, segment_id, label, tp, fp, fn,
      precision, recall, f1, support, eval_key
    """
    ds = get_dataset(dataset, pred_version)

    # Decide which events to evaluate
    eval_events: List[Tuple[str, str]] = []

    if events is not None:
        eval_events = list(events)
    else:
        # All events in dataset
        if video_series is None:
            series_values = sorted(ds.distinct("video_series"))
        # All segments/events in those series
        else:
            series_values = sorted(set(video_series))

        for series in series_values:
            series_view = ds.match(F("video_series") == series)
            segment_ids = sorted(series_view.distinct("segment_id"))
            for seg in segment_ids:
                eval_events.append((series, seg))

    rows: List[dict] = []

    if print_reports:
        print(f"\n=== PER EVENT METRICS (pred_version={pred_version}) ===")

    for series, seg in eval_events:
        view = ds.match((F("video_series") == series) & (F("segment_id") == seg))
        eval_key = make_eval_key(f"event_{series}_{seg}", iou_thresh)

        summary, results = evaluate_view(
            view=view,
            eval_key=eval_key,
            iou_thresh=iou_thresh,
            class_name=FIFTYONE_CLASS_NAME,
        )
        summary["video_series"] = series
        summary["segment_id"] = seg
        rows.append(summary)

        if print_reports:
            print(f"\n--- Series: {series} | Segment: {seg} ---")
            results.print_report(classes=[FIFTYONE_CLASS_NAME])

    ds.save()

    df = pd.DataFrame(rows)
    return df[
        ["video_series", "segment_id", "label",
         "tp", "fp", "fn", "precision", "recall", "f1", "support", "eval_key"]
    ]