# src/model_eval/experiments/tree_training.py

from __future__ import annotations

from typing import Dict, List, Tuple
import pandas as pd

from model_eval.experiments.dataset_utils import WindowConfig
from model_eval.experiments.window_dataset import build_window_dataset
from model_eval.experiments.tree_models import (
    build_base_tree_models,
    get_param_distributions,
)
from model_eval.experiments.training_utils import (
    SEED,
    train_val_split_by_series,
    get_window_Xy,
    fit_and_eval_single,
    describe_labels,
    save_metrics_df,
    tune_with_random_search,
)


def run_tree_baselines(
        cfg: WindowConfig,
        val_frac: float = 0.3,
        threshold: float = 0.5,
        search_best_threshold: bool = True,
        tune: bool = False,
        n_iter: int = 20,
        save_results: bool = True,
        return_models: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, object]]:
    """
    End-to-end:
      - build window dataset
      - split train/val by series
      - (optional) tune hyperparameters with RandomizedSearchCV
      - train and evaluate models
      - return metrics (and optional fitted models)
    """
    # -------------------------------------------------
    # Data
    # -------------------------------------------------
    df_win = build_window_dataset(cfg)
    print(f"[DATA] Window dataset shape: {df_win.shape}")
    print("[DATA] label counts:\n", df_win["label"].value_counts(dropna=False))

    df_train, df_val = train_val_split_by_series(df_win, val_frac=val_frac, seed=SEED)
    print(f"[SPLIT] Train windows: {len(df_train)}, Val windows: {len(df_val)}")

    X_train, y_train, feature_cols = get_window_Xy(df_train)
    X_val, y_val, _ = get_window_Xy(df_val, feature_cols=feature_cols)

    describe_labels("TRAIN", y_train)
    describe_labels("VAL", y_val)
    print(f"[FEATURES] Using {len(feature_cols)} features")

    # -------------------------------------------------
    # Models + optional tuning
    # -------------------------------------------------
    base_models = build_base_tree_models()
    param_dists = get_param_distributions()

    metrics_list: List[dict] = []
    fitted_models: Dict[str, object] = {}

    for name, base_model in base_models.items():
        model = base_model
        best_params = None

        # Hyperparameter search (optional)
        if tune and name in param_dists:
            model, best_params = tune_with_random_search(
                name=name,
                base_estimator=base_model,
                param_distributions=param_dists[name],
                X_train=X_train,
                y_train=y_train,
                n_iter=n_iter,
                cv=3,
                scoring="average_precision",
                n_jobs=-1,
                random_state=SEED,
                verbose=True,
            )

        # Final fit + val metrics
        metrics = fit_and_eval_single(
            name=name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            threshold=threshold,
            search_best_threshold=search_best_threshold,
            verbose=True,
        )

        # Add best_params to results
        if best_params is not None:
            for k, v in best_params.items():
                metrics[f"param_{k}"] = v

        metrics_list.append(metrics)
        fitted_models[name] = model

    # -------------------------------------------------
    # Metrics DataFrame
    # -------------------------------------------------
    metrics_df = pd.DataFrame(metrics_list)
    cols = ["model"] + [c for c in metrics_df.columns if c != "model"]
    metrics_df = metrics_df[cols]

    metrics_df["tuned"] = tune

    if save_results:
        save_metrics_df(
            metrics_df, cfg,
            val_frac=val_frac, threshold=threshold, model_type="tree"
        )

    if return_models:
        return metrics_df, fitted_models
    return metrics_df
