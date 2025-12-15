# src/model_eval/utils/paths.py

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_ROOT = PROJECT_ROOT / "data"
DATA_RAW = DATA_ROOT / "raw"
DATA_STD = DATA_ROOT / "standardized"
DATA_PROCESSED = DATA_ROOT / "processed"
DATA_METADATA = DATA_ROOT / "metadata"

# Raw paths
DATA_RAW_GT = DATA_RAW / "gt"
DATA_RAW_PRED = DATA_RAW / "predictions"
DATA_RAW_VIDEOS = DATA_RAW / "videos"

# Std paths
DATA_STD_GT = DATA_STD / "gt"
DATA_STD_PRED = DATA_STD / "predictions"

# Reports (outputs)
REPORTS_ROOT = PROJECT_ROOT / "reports"
REPORTS_ANALYSIS_ROOT = REPORTS_ROOT / "analysis"
REPORTS_EXPERIMENTS_ROOT = REPORTS_ROOT / "experiments"

# Versioned filenames
GT_PRED_ALL_NAME = "gt_pred_all.csv"
PRED_IOU_NAME = "pred_iou.csv"

# Versioned template
GT_PRED_FEATURES_TEMPLATE = "gt_pred_all_features_iou_{iou_str}.csv"

LATEST_PRED_VERSION = "v0"

@dataclass(frozen=True)
class VersionPaths:
    """
    Collection of paths for a given prediction version.
    Example layout (for version='v0'):
        data/raw/predictions/v0/
        data/std/predictions/v0
        data/processed/v0/gt_pred_all.csv
        reports/analysis/v0
    """
    version: str
    raw_pred_dir: Path
    std_pred_dir: Path
    processed_dir: Path

    reports_analysis_dir: Path
    reports_experiments_dir: Path

    gt_pred_all_path: Path
    pred_iou_path: Path

    def gt_pred_features_path(self, iou_thresh: float) -> Path:
        iou_str = f"{iou_thresh:.2f}".replace(".", "_")
        filename = GT_PRED_FEATURES_TEMPLATE.format(iou_str=iou_str)
        return self.processed_dir / filename


def get_paths_for_version(pred_version: str | None = None) -> VersionPaths:
    """
    Resolve all versioned paths for a given prediction version.
    If 'pred_version' is None, uses LATEST_PRED_VERSION.
    """
    version = pred_version or LATEST_PRED_VERSION

    raw_pred_dir = DATA_RAW_PRED / version
    std_pred_dir = DATA_STD_PRED / version
    processed_dir = DATA_PROCESSED / version

    reports_analysis_dir = REPORTS_ANALYSIS_ROOT / version
    reports_experiments_dir = REPORTS_EXPERIMENTS_ROOT / version

    gt_pred_all_path = processed_dir / GT_PRED_ALL_NAME
    pred_iou_path = processed_dir / PRED_IOU_NAME

    return VersionPaths(
        version=version,
        raw_pred_dir=raw_pred_dir,
        std_pred_dir=std_pred_dir,
        processed_dir=processed_dir,
        reports_analysis_dir=reports_analysis_dir,
        reports_experiments_dir=reports_experiments_dir,
        gt_pred_all_path=gt_pred_all_path,
        pred_iou_path=pred_iou_path,
    )

def ensure_dir(path: Path):
    """Create path (and parents) if missing."""
    path.mkdir(parents=True, exist_ok=True)



