# src/validation/check_raw_vs_processed_counts.py

from pathlib import Path
import pandas as pd

from model_eval.config import (
    DATA_RAW_GT,
    get_paths_for_version,
    GT_SOURCE,
    PRED_SOURCE,
)
from model_eval.data.loaders import load_gt_pred_all


def count_raw_gt_lines() -> pd.DataFrame:
    """
    Count number of lines in raw GT txt files, per (video_series, segment_id).
    Assumes path structure:
        DATA_RAW_GT / <video_series> / <segment_id>.txt
    """
    records: list[dict] = []

    for series_dir in sorted(DATA_RAW_GT.iterdir()):
        if not series_dir.is_dir():
            continue
        video_series = series_dir.name

        for txt_path in sorted(series_dir.glob("*.txt")):
            segment_id = txt_path.stem  # e.g. "1470_0"
            with txt_path.open("r", encoding="utf-8") as f:
                n_lines = sum(1 for _ in f)

            records.append(
                {
                    "video_series": video_series,
                    "segment_id": segment_id,
                    "raw_gt_lines": n_lines,
                }
            )

    return pd.DataFrame(records)


def count_raw_pred_lines(pred_version: str | None = None) -> pd.DataFrame:
    """
    Count number of prediction lines in raw PRED txt files, per video_series.
    First line = original_size, so subtract 1.
    Assumes path structure:
        VersionPaths.raw_pred_dir = data/raw/predictions/<pred_version>/*.txt
    """
    records: list[dict] = []
    paths = get_paths_for_version(pred_version)
    for txt_path in sorted(paths.raw_pred_dir.glob("*.txt")):
        video_series = txt_path.stem
        with txt_path.open("r", encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)

        pred_lines = max(0, n_lines - 1)

        records.append(
            {
                "video_series": video_series,
                "raw_pred_lines": pred_lines,
            }
        )

    return pd.DataFrame(records)


def compare_raw_and_processed_counts(pred_version: str | None = None) -> None:
    """
    Compare raw line counts (GT + PRED) with processed/<pred_version>/gt_pred_all.csv.

    Prints ONLY mismatches:
      - events where raw_gt_lines != proc_gt_rows
      - series where raw_pred_lines != proc_pred_rows
    """
    df_raw_gt = count_raw_gt_lines()
    df_raw_pred = count_raw_pred_lines(pred_version)

    df = load_gt_pred_all(pred_version)
    if df.empty:
        raise ValueError(f"No data loaded from gt_pred_all.csv (pred_version={pred_version}.")

    # Processed GT rows per (video_series, segment_id)
    df_proc_gt = (
        df[df["source"] == GT_SOURCE]
        .groupby(["video_series", "segment_id"])
        .size()
        .reset_index(name="proc_gt_rows")
    )

    # Processed PRED rows per series
    df_proc_pred = (
        df[df["source"] == PRED_SOURCE]
        .groupby("video_series")
        .size()
        .reset_index(name="proc_pred_rows")
    )

    # GT: compare per event
    df_gt_compare = df_raw_gt.merge(
        df_proc_gt,
        on=["video_series", "segment_id"],
        how="outer",
    ).sort_values(["video_series", "segment_id"])

    df_gt_compare["gt_diff"] = (
        df_gt_compare["raw_gt_lines"].fillna(0)
        - df_gt_compare["proc_gt_rows"].fillna(0)
    )

    df_gt_mismatch = df_gt_compare[df_gt_compare["gt_diff"] != 0]

    #PRED: compare per series
    df_pred_compare = df_raw_pred.merge(
        df_proc_pred,
        on="video_series",
        how="outer",
    ).sort_values("video_series")

    df_pred_compare["pred_diff"] = (
        df_pred_compare["raw_pred_lines"].fillna(0)
        - df_pred_compare["proc_pred_rows"].fillna(0)
    )

    df_pred_mismatch = df_pred_compare[df_pred_compare["pred_diff"] != 0]

    # Print results
    if df_gt_mismatch.empty:
        print("[CHECK] GT raw vs processed: all events match.")
    else:
        print("\n[CHECK] GT raw vs processed: mismatches found (raw - processed):")
        print(df_gt_mismatch)

    if df_pred_mismatch.empty:
        print("[CHECK] PRED raw vs processed: all series match.")
    else:
        print("\n[CHECK] PRED raw vs processed: mismatches found (raw - processed):")
        print(df_pred_mismatch)