# src/model_eval/experiments/results_utils.py

from __future__ import annotations
from typing import Sequence, Optional
import pandas as pd
from pathlib import Path

from model_eval.config import get_paths_for_version

MODEL_LABELS = {
    "logreg": "LogisticRegression",
    "rf": "RandomForest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
}

TRACK_ORDER = ("no-id", "id")


def _make_order(items: Sequence[str]) -> list[str]:
    # For each item -> ["<item> no-id", "<item> id"]
    return [f"{it} {t}" for it in items for t in TRACK_ORDER]


def load_all_tree_runs(
        pred_versions: Sequence[str] = ("v0", "v0_bt"),
        add_display_names: bool = False,
        pred_labels: Optional[Sequence[str]] = ("YOLO (baseline)", "YOLO + ByteTrack"),
) -> pd.DataFrame:
    """
    Concatenate tree_all_runs.csv files for the given pred_versions.
    Always adds:
      - track_flag            ("id"/"no-id")
      - pred_version_track    (e.g. "v0 no-id")

    If add_display_names=True also adds:
      - model_label
      - pred_label
      - pred_label_track      (e.g. "YOLO + ByteTrack id")
    """
    dfs = []
    for version in pred_versions:
        paths = get_paths_for_version(version)
        csv_path = paths.reports_experiments_dir / "tree" / "tree_all_runs.csv"
        if csv_path.exists():
            print(f"[LOAD] OK for {csv_path}")
            dfs.append(pd.read_csv(csv_path))
        else:
            print(f"[ERROR] Missing tree_all_runs.csv for pred_version={version}")

    if not dfs:
        raise ValueError("No tree_all_runs.csv files found for given versions.")

    df = pd.concat(dfs, ignore_index=True)

    # -------------------------
    # Always
    # -------------------------
    df["track_flag"] = df["include_track_ids"].map({True: "id", False: "no-id"}).fillna("no-id")
    df["pred_version"] = df["pred_version"].astype(str)

    df["pred_version_track"] = df["pred_version"] + " " + df["track_flag"]

    # Ordered categories for consistent legends
    pv_order = _make_order([str(v) for v in pred_versions])
    pv_order = [x for x in pv_order if x in set(df["pred_version_track"])]
    df["pred_version_track"] = pd.Categorical(
        df["pred_version_track"], categories=pv_order, ordered=True
    )

    # -------------------------
    # Pretty labels (optional)
    # -------------------------
    if add_display_names:
        if pred_labels is None:
            raise ValueError("pred_labels must be provided when add_display_names=True")
        if len(pred_labels) != len(pred_versions):
            raise ValueError(
                f"pred_labels length ({len(pred_labels)}) must match pred_versions length ({len(pred_versions)})"
            )

        version_label_map = {str(v): str(l) for v, l in zip(pred_versions, pred_labels)}

        df["pred_label"] = df["pred_version"].map(version_label_map).fillna(df["pred_version"])
        df["pred_label_track"] = df["pred_label"] + " " + df["track_flag"]

        label_order = _make_order([str(l) for l in pred_labels])
        label_order = [x for x in label_order if x in set(df["pred_label_track"])]
        df["pred_label_track"] = pd.Categorical(
            df["pred_label_track"], categories=label_order, ordered=True
        )

        df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"].astype(str))

    return df