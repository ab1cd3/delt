# src/model_eval/experiments/tree_shap_models.py

from __future__ import annotations

from typing import Dict
import pandas as pd
import shap
import matplotlib.pyplot as plt

from model_eval.experiments.dataset_utils import WindowConfig
from model_eval.experiments.window_dataset import build_window_dataset
from model_eval.experiments.training_utils import (
    SEED,
    train_val_split_by_series,
    get_window_Xy,
)
from model_eval.experiments.tree_models import build_base_tree_models
from model_eval.config import get_paths_for_version, ensure_dir


def shap_all_tree_models_for_config(
        pred_version: str = "v0_bt",
        include_track_ids: bool = False,
        label_mode: str = "tp_count",
        tp_count_thresh: int = 1,
        window_size: int = 150,
        step_size: int | None = None,
        val_frac: float = 0.3,
        max_background: int = 300,
        save: bool = True,
        show: bool = False,
) -> None:
    """
    Compare SHAP feature importance for tree-based models:
        RandomForest, XGBoost, LightGBM

    under a single configuration (pred_version, track ids, label_mode, window).
    """

    if step_size is None:
        step_size = window_size // 2

    cfg = WindowConfig(
        window_size=window_size,
        step_size=step_size,
        pred_version=pred_version,
        label_mode=label_mode,
        tp_count_thresh=tp_count_thresh,
        include_track_ids=include_track_ids,
    )

    print("[CONFIG] SHAP tree-model comparison on:")
    print(cfg)

    # ------------------------------------------------------------------
    # Build window dataset and train/val split (by series)
    # ------------------------------------------------------------------
    df_win = build_window_dataset(cfg)
    print(f"[DATA] Window dataset shape: {df_win.shape}")

    df_train, df_val = train_val_split_by_series(df_win, val_frac=val_frac, seed=SEED)
    print(f"[SPLIT] Train windows: {len(df_train)}, Val windows: {len(df_val)}")

    X_train, y_train, feature_cols = get_window_Xy(df_train)
    X_train_df = pd.DataFrame(X_train, columns=feature_cols)

    print(f"[FEATURES] Using {len(feature_cols)} features")

    # Background sample for SHAP
    if len(X_train_df) > max_background:
        background = X_train_df.sample(n=max_background, random_state=SEED)
    else:
        background = X_train_df

    # ------------------------------------------------------------------
    # Build models (rf, xgb, lgbm)
    # ------------------------------------------------------------------
    all_models: Dict[str, object] = build_base_tree_models()
    models = {k: all_models[k] for k in ["logreg", "rf", "xgb", "lgbm"]}

    # Output directory
    paths = get_paths_for_version(pred_version)
    out_dir = paths.reports_experiments_dir / "tree" / "shap"
    ensure_dir(out_dir)

    include_str = "track_ids" if include_track_ids else "no_track_ids"

    for name, model in models.items():
        print(f"\n[SHAP MODELS] Fitting {name}")
        model.fit(X_train_df, y_train)

        explainer = shap.Explainer(
            model.predict_proba,
            background,
            feature_names=feature_cols,
        )
        shap_values = explainer(background)

        print(
            f"[DEBUG] {name} – shap_values shape: {shap_values.values.shape}, "
            f"data shape: {shap_values.data.shape}"
        )

        if shap_values.values.ndim == 3:
            shap_for_plot = shap_values[:, :, 1]  # class 1
        else:
            shap_for_plot = shap_values

        base_title = (
            f"SHAP – {name}, pred={pred_version}, {include_str}, "
            f"ws={window_size}, step={step_size}"
        )

        # SHAP Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_for_plot,
            show=False,
            max_display=20,
        )
        plt.title(base_title)
        plt.tight_layout()

        if save:
            fname = f"shap_{name}_{pred_version}_{include_str}_ws{window_size}_summary.png"
            fpath = out_dir / fname
            plt.savefig(fpath, dpi=200)
            print(f"[PLOT] Saved SHAP summary to {fpath}")

        if show:
            plt.show()
        else:
            plt.close()

        # SHAP Bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_for_plot,
            show=False,
            plot_type="bar",
            max_display=20,
        )
        plt.title(base_title)
        plt.tight_layout()

        if save:
            fname = f"shap_{name}_{pred_version}_{include_str}_ws{window_size}_bar.png"
            fpath = out_dir / fname
            plt.savefig(fpath, dpi=200)
            print(f"[PLOT] Saved SHAP bar to {fpath}")

        if show:
            plt.show()
        else:
            plt.close()