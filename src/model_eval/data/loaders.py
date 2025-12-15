# src/model_eval/data/loaders.py

from pathlib import Path
from typing import List
import pandas as pd

from model_eval.config import DATA_STD_GT, get_paths_for_version, IOU_THRESHOLD

# --------------------------
# GT
# --------------------------
def get_std_gt_series_list() -> List[str]:
    """
    Return list of video_series names that have standardized GT data.
    Assumes one folder per series under DATA_STD_GT.
    """
    if not DATA_STD_GT.exists():
        return []

    return sorted(
        d.name for d in DATA_STD_GT.iterdir() if d.is_dir()
    )


def load_std_gt_for_series(video_series: str) -> pd.DataFrame:
    """
    Load standardized GT for one series from:
        DATA_STD_GT / <series> / <series>_all.csv
    """
    series_dir = DATA_STD_GT / video_series
    csv_path = series_dir / f"{video_series}_all.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Standardized GT file not found: {csv_path}")

    return pd.read_csv(csv_path)


def load_std_gt_all_series() -> pd.DataFrame:
    """
    Load standardized GT for all series and concatenate.
    """
    series_list = get_std_gt_series_list()
    dfs = []
    for series in series_list:
        df = load_std_gt_for_series(series)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


# --------------------------
# PRED
# --------------------------
def get_std_pred_series_list(pred_version: str | None = None) -> List[str]:
    """
    Return list of video_series names that have standardized predictions.
    Assumes one <series>.csv per series under:
        VersionPaths.std_pred_dir = data/standardized/predictions/<version>/
    """
    paths = get_paths_for_version(pred_version)
    std_pred_dir = paths.std_pred_dir

    if not std_pred_dir.exists():
        return []

    return sorted(p.stem for p in std_pred_dir.glob("*.csv"))


def load_std_pred_for_series(
        video_series: str,
        pred_version: str | None = None,
) -> pd.DataFrame:
    """
    Load standardized predictions for one series from:
        VersionPaths.std_pred_dir = data/standardized/predictions/<version>/
    """
    paths = get_paths_for_version(pred_version)
    csv_path = paths.std_pred_dir / f"{video_series}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Standardized predictions file not found: {csv_path}")

    return pd.read_csv(csv_path)


def load_std_pred_all_series(pred_version: str | None = None) -> pd.DataFrame:
    """
    Load standardized predictions for all series and concatenate.
    """
    series_list = get_std_pred_series_list(pred_version=pred_version)
    dfs = [load_std_pred_for_series(s, pred_version=pred_version) for s in series_list]

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


# --------------------------
# GT + PRED
# --------------------------
def load_gt_pred_all(pred_version: str | None = None) -> pd.DataFrame:
    """
    Load processed combined GT+PRED csv file from:
        VersionPaths.gt_pred_all_path = data/processed/<version>/gt_pred_all.csv
    """
    paths = get_paths_for_version(pred_version)
    csv_path = paths.gt_pred_all_path

    if not csv_path.exists():
        raise FileNotFoundError(f"gt_pred_all.csv not found at: {csv_path}")

    return pd.read_csv(csv_path)


def load_pred_iou(pred_version: str | None = None) -> pd.DataFrame:
    """
    Load precomputed IoU table for predictions from:
        VersionPaths.gt_pred_iou_path = data/processed/<version>/pred_iou.csv
    """
    paths = get_paths_for_version(pred_version)
    path = paths.pred_iou_path

    if not path.exists():
        raise FileNotFoundError(
            f"pred_iou.csv not found at {path}. "
            "Run scripts/build_pred_iou.py for this version first."
        )
    return pd.read_csv(path)


def load_gt_pred_all_features(
        iou_thresh: float = IOU_THRESHOLD,
        pred_version: str | None = None,
) -> pd.DataFrame:
    """
    Load processed combined GT+PRED csv file from:
        VersionPaths.gt_pred_all_path =
            data/processed/<version>/gt_pred_all_features_iou_<iou_thresh>.csv
    """
    paths = get_paths_for_version(pred_version)
    csv_path = paths.gt_pred_features_path(iou_thresh=iou_thresh)

    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path.stem} not found at: {csv_path}")

    return pd.read_csv(
        csv_path,
        dtype={
            "segment_id": "string",
        }
    )