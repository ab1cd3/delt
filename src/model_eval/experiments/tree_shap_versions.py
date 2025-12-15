# src/model_eval/experiments/tree_shap_versions.py

from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import shap

from lightgbm import LGBMClassifier
from pandas.core.computation.ops import isnumeric

from model_eval.experiments.dataset_utils import WindowConfig
from model_eval.experiments.window_dataset import build_window_dataset
from model_eval.experiments.training_utils import (
    SEED,
    train_val_split_by_series,
    get_window_Xy,
)
from model_eval.experiments.results_utils import load_all_tree_runs
from model_eval.config import get_paths_for_version, ensure_dir


# -------------------------------------------------------
# Pick best LGBM
# -------------------------------------------------------
def pick_global_best_lgbm(df_all: pd.DataFrame) -> pd.Series:
    """Pick the best LGBM by highest pr_auc followed by highest recall."""
    df_lgbm = df_all[df_all["model"] == "lgbm"].copy()
    if df_lgbm.empty:
        raise ValueError("No LGBM rows in given dataframe.")

    df_lgbm = df_lgbm.sort_values(["pr_auc", "recall"], ascending=[False, False])
    best = df_lgbm.iloc[0]
    print("\n[GLOBAL BEST LGBM]")
    print(best.to_string())
    return best


def pick_best_like(
        df_all: pd.DataFrame,
        base_row: pd.Series,
        pred_version: str | None = None,
        include_track_ids: bool | None = None,
) -> pd.Series:
    """
    From df_all, find the best row that matches base_row on:
      - model
      - label_mode
      - window_size
      - step_size
      - tp_count_thresh
      - event_frac_thresh
    but with given pred_version and include_track_ids options.
    """
    if pred_version is None:
        pred_version = base_row["pred_version"]
    if include_track_ids is None:
        include_track_ids = base_row["include_track_ids"]

    mask = (
        (df_all["model"] == base_row["model"])
        & (df_all["label_mode"] == base_row["label_mode"])
        & (df_all["window_size"] == base_row["window_size"])
        & (df_all["step_size"] == base_row["step_size"])
        & (df_all["pred_version"] == pred_version)
        & (df_all["include_track_ids"] == include_track_ids)
    )

    mode = str(base_row["label_mode"]).lower()
    if mode == "tp_count":
        mask &= df_all["tp_count_thresh"] == base_row["tp_count_thresh"]
    elif mode == "event_frac":
        mask &= df_all["event_frac_thresh"] == base_row["event_frac_thresh"]

    df_f = df_all[mask].copy()
    if df_f.empty:
        raise ValueError(
            f"No matching runs for "
            f"pred_version={pred_version}, include_track_ids={include_track_ids}"
        )

    df_f = df_f.sort_values(["pr_auc", "recall"], ascending=[False, False])
    best = df_f.iloc[0]
    print("\n[BEST MATCH]")
    print(best.to_string())
    return best


# -------------------------------------------------------
# Rebuild config & model from a row
# -------------------------------------------------------

def cfg_from_row(row: pd.Series) -> WindowConfig:
    return WindowConfig(
        window_size=int(row["window_size"]),
        step_size=int(row["step_size"]),
        iou_thresh=float(row["iou_thresh"]),
        pred_version=str(row["pred_version"]),
        label_mode=str(row["label_mode"]),
        tp_count_thresh=int(row["tp_count_thresh"]),
        event_frac_thresh=float(row["event_frac_thresh"]),
        extra_features=None,
        include_track_ids=bool(row["include_track_ids"]),
    )


def build_lgbm_from_row(row: pd.Series) -> LGBMClassifier:
    """
    Build an LGBMClassifier from param_* columns in row.
    """
    param_cols = [
        c for c in row.index
        if c.startswith("param_") and pd.notna(row[c])
    ]

    params = {}

    for c in param_cols:
        p_name =  c.replace("param_", "")
        v = row[c]
        if isnumeric(v):
            v = v if v < 1.0 else int(v)
        params[p_name] = v

    params.setdefault("random_state", SEED)

    model = LGBMClassifier(**params)
    print("[LGBM PARAMS]", params)
    return model


# -------------------------------------------------------
# SHAP for a given row (one model/config)
# -------------------------------------------------------

def shap_for_lgbm_row(
        row: pd.Series,
        tag: str = "",
        max_background: int = 500,
        save: bool = True,
        show: bool = True,
) -> None:
    """
    - Build cfg + dataset
    - Split train/val by series
    - Build LGBM from row
    - Fit on train
    - Plot SHAP summary + bar on a background sample
    """
    cfg = cfg_from_row(row)
    val_frac = float(row.get("val_frac", 0.3))

    print("\n[SHAP] Using config:")
    print(cfg)

    # Build window dataset and split
    df_win = build_window_dataset(cfg)
    print(f"[DATA] Window dataset shape: {df_win.shape}")

    df_train, df_val = train_val_split_by_series(df_win, val_frac=val_frac, seed=SEED)
    print(f"[SPLIT] Train windows: {len(df_train)}, Val windows: {len(df_val)}")

    # Use the same feature selection logic
    X_train, y_train, feature_cols = get_window_Xy(df_train)

    # Wrap in DataFrame for SHAP (for feature names)
    X_train_df = pd.DataFrame(X_train, columns=feature_cols)

    # Background sample
    if len(X_train_df) > max_background:
        background = X_train_df.sample(n=max_background, random_state=SEED)
    else:
        background = X_train_df

    # Build & fit model
    model = build_lgbm_from_row(row)
    model.fit(X_train_df, y_train)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(background)

    pred_version = str(row["pred_version"])
    include_track = bool(row["include_track_ids"])
    include_str = "track_ids" if include_track else "no_track_ids"
    tag_str = tag or "base"

    base_title = (
        f"LGBM SHAP [{tag_str}] "
        f"pred={pred_version}, {include_str}, "
        f"ws={row['window_size']}, step={row['step_size']}"
    )

    if save:
        paths = get_paths_for_version(pred_version)
        out_dir = paths.reports_experiments_dir / "tree" / "shap"
        ensure_dir(out_dir)

    # SHAP Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        background,
        feature_names=feature_cols,
        show=False,
        max_display=20,
    )
    plt.title(base_title)
    plt.tight_layout()

    if save:
        summary_name = f"shap_lgbm_{tag_str}_{pred_version}_{include_str}_summary.png"
        summary_path = out_dir / summary_name
        plt.savefig(summary_path, dpi=200)
        print(f"[PLOT] Saved SHAP summary to {summary_path}")

    if show:
        plt.show()
    else:
        plt.close()

    # SHAP Bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        background,
        feature_names=feature_cols,
        show=False,
        plot_type="bar",
        max_display=20,
    )
    plt.title(base_title)
    plt.tight_layout()

    if save:
        bar_name = f"shap_lgbm_{tag_str}_{pred_version}_{include_str}_bar.png"
        bar_path = out_dir / bar_name
        plt.savefig(bar_path, dpi=200)
        print(f"[PLOT] Saved SHAP bar to {bar_path}")

    if show:
        plt.show()
    else:
        plt.close()


# -------------------------------------------------------
# MAIN: base + three version comparisons
# -------------------------------------------------------

def shap_for_all_versions(
        pred_version_1: str = "v0",
        pred_version_2: str = "v0_bt",
):
    """
    Run and save shap summary and bar plot for all configurations
    given 2 pred versions.
    """

    df_all = load_all_tree_runs([pred_version_1, pred_version_2])

    # Global best LGBM (most common model in top 15 ranking)
    base_row = pick_global_best_lgbm(df_all)
    base_pred = str(base_row["pred_version"])
    base_track = bool(base_row["include_track_ids"])

    # Choose the other pred_version (v0 <-> v0_bt)
    if base_pred == pred_version_1:
        other_pred = pred_version_2
    elif base_pred == pred_version_2:
        other_pred = pred_version_1
    else:
        raise ValueError(
            f"Unexpected base pred_version={base_pred}, expected: {"v0" or "v0_bt"}"
        )

    rows: Dict[str, pd.Series] = {}
    rows["base"] = base_row

    # Flip pred_version, same include_track_ids
    rows["flip_pred"] = pick_best_like(
        df_all, base_row,
        pred_version=other_pred,
        include_track_ids=base_track,
    )

    # Flip include_track_ids, same pred_version
    rows["flip_track"] = pick_best_like(
        df_all, base_row,
        pred_version=base_pred,
        include_track_ids=not base_track,
    )

    # Flip both options
    rows["flip_both"] = pick_best_like(
        df_all, base_row,
        pred_version=other_pred,
        include_track_ids=not base_track,
    )

    # Run SHAP for all 4 configs
    for tag, row in rows.items():
        print(f"\n================ SHAP for {tag} ================")
        shap_for_lgbm_row(row, tag=tag)