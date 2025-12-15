# src/model_eval/data/metadata.py

from pathlib import Path
import pandas as pd

from model_eval.config import DATA_METADATA


def load_series_event_labels() -> pd.DataFrame:
    """
    Load mapping from (video_series, segment_id) -> labels.
    """
    path = DATA_METADATA / "series_event_labels.csv"
    if not path.exists():
        raise FileNotFoundError(f"Metadata label file not found: {path}")
    return pd.read_csv(path)


def apply_series_labels(
        df: pd.DataFrame,
        labels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Add series-level labels (series_idx, series_label) to a DataFrame
    that has a 'video_series' column.
    """
    if "video_series" not in df.columns:
        raise ValueError("DataFrame must contain 'video_series' to apply series labels.")

    if labels is None:
        labels = load_series_event_labels()

    series_labels = (
        labels[["video_series", "series_idx", "series_label"]]
        .drop_duplicates(subset=["video_series"])
    )

    return df.merge(series_labels, on="video_series", how="left")


def apply_series_event_labels(
        df: pd.DataFrame,
        labels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Add both series-level and event-level labels to a DataFrame
    that has 'video_series' and 'segment_id' columns.
    """
    if labels is None:
        labels = load_series_event_labels()

    # Add series-level labels
    df = apply_series_labels(df, labels=labels)

    # Add event-level labels
    event_labels = labels[
        ["video_series", "segment_id", "event_idx", "event_label", "full_label"]
    ].drop_duplicates(subset=["video_series", "segment_id"])

    df = df.merge(event_labels, on=["video_series", "segment_id"], how="left")

    return df