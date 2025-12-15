# src/model_eval/experiments/training_utils.py

from __future__ import annotations

from typing import Dict, Tuple, List, Optional
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV

from model_eval.config import get_paths_for_version, ensure_dir, LATEST_PRED_VERSION
from model_eval.experiments.dataset_utils import WindowConfig


SEED = 42

WINDOW_META_COLS = [
    "video_series",
    "frame_start",
    "frame_end",
    "label",
    "n_pred",
]


def describe_labels(name, y):
    """Print the class label counts."""
    vals, counts = np.unique(y, return_counts=True)
    print(f"\n[{name}] label distribution: ")
    for v, c in zip(vals, counts):
        print(f"\tclass {v}: {c} ({c/len(y):.2f})")


def train_val_split_by_series(
        df: pd.DataFrame,
        val_frac: float = 0.3,
        seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split windows into train/val sets by video_series.
    Uses GroupShuffleSplit on the video_series column.
    """
    groups = df["video_series"].to_numpy()
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_frac,
        random_state=seed,
    )
    (train_idx, val_idx), = splitter.split(df, groups=groups)

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    return df_train, df_val


def get_window_Xy(
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Given a window-level dataframe, return (X, y, feature_cols).
    By default uses all columns except WINDOW_META_COLS as features.
    """
    if feature_cols is None:
        feature_cols = [
            c for c in df.columns
            if c not in WINDOW_META_COLS
        ]

    X = df[feature_cols].copy().fillna(0.0).to_numpy(dtype="float32")
    y = df["label"].to_numpy(dtype="int64")
    return X, y, feature_cols


def tune_with_random_search(
        name: str,
        base_estimator,
        param_distributions: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_iter: int = 20,
        cv: int = 3,
        scoring: str = "average_precision",
        n_jobs: int = -1,
        random_state: int = SEED,
        verbose: bool = True,
):
    """
    Run RandomizedSearchCV for one model and return the best estimator + best_params.
    """
    if verbose:
        print(f"\n[{name}] RandomizedSearchCV: n_iter={n_iter}, cv={cv}, scoring={scoring}")

    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=n_jobs,
        cv=cv,
        verbose=1 if verbose else 0,
        random_state=random_state,
        refit=True,
    )
    search.fit(X_train, y_train)

    if verbose:
        print(f"[{name}] best score (cv {scoring}): {search.best_score_:.4f}")
        print(f"[{name}] best params: {search.best_params_}")

    best_estimator = search.best_estimator_
    best_params = search.best_params_
    return best_estimator, best_params


def _get_probas(model, X_val: np.ndarray) -> np.ndarray:
    """
    Return probability for positive class if possible.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_val)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_val)
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            return (scores - s_min) / (s_max - s_min)
        return np.zeros_like(scores)
    preds = model.predict(X_val)
    return preds.astype(float)


def compute_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute a set of scalar metrics for binary classification.
    """
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    try:
        roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc = float("nan")

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = float("nan")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr_auc,
    }


def find_best_threshold(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        metric: str = "f1",
        thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Scan a grid of thresholds and return (best_threshold, metrics_dict_at_best),
    based on given metric ("f1") to optimize.
    """
    if thresholds is None:
        # Create thresholds
        thresholds = np.arange(0.05, 1.0, 0.05)

    best_thr = 0.5
    best_score = -1.0
    best_metrics: Dict[str, float] = {}

    for thr in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=thr)
        score = m.get(metric, float("nan"))
        if np.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_metrics = m

    # Save
    best_metrics["threshold"] = best_thr
    return best_thr, best_metrics


def fit_and_eval_single(
        name: str,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        threshold: float = 0.5,
        search_best_threshold: bool = True,
        threshold_metric: str = "f1",
        verbose: bool = True,
) -> Dict[str, float]:
    """
    Fit a model and return metrics dict with model name included.

    If search_best_threshold=True,scan a grid of thresholds on y_prob
    and pick the one that maximizes threshold_metric (e.g. "f1").
    """
    if verbose:
        print(f"\n=== Training {name} ===")

    model.fit(X_train, y_train)
    y_prob = _get_probas(model, X_val)

    if search_best_threshold:
        best_thr, metrics = find_best_threshold(
            y_true=y_val,
            y_prob=y_prob,
            metric=threshold_metric,
        )
        if verbose:
            print(f"[{name}] best threshold by {threshold_metric}: {best_thr:.3f}")
        metrics["threshold"] = best_thr
    else:
        metrics = compute_metrics(y_val, y_prob, threshold=threshold)
        metrics["threshold"] = threshold

    metrics["model"] = name

    if verbose:
        print(f"[{name}] metrics (threshold={metrics['threshold']:.3f}):")
        for k, v in metrics.items():
            if k in ("model", "threshold"):
                continue
            print(f"  {k:10s}: {v:.4f}")

    return metrics


def save_metrics_df(
        metrics_df: pd.DataFrame,
        cfg: WindowConfig,
        val_frac: float,
        threshold: float,
        model_type: str,  # e.g. "tree" or "sequence"
) -> Path:
    """
    Append metrics_df to a single CSV:

        reports/analysis/<version>/experiments/<model_type>/<model_type>_all_runs.csv

    Adds:
      - all fields from WindowConfig
      - SEED, model_type, val_frac, decision_threshold
      - run_ts: timestamp string for this run (same for all rows in metrics_df)
    """
    df = metrics_df.copy()

    # 1) Add config info (flatten WindowConfig)
    cfg_dict = asdict(cfg)
    for k, v in cfg_dict.items():
        df[k] = v

    # 2) Extra run metadata
    df["seed"] = SEED
    df["model_type"] = model_type
    df["val_frac"] = val_frac
    df["decision_threshold"] = threshold

    # One timestamp for the whole run
    df["run_ts"] = datetime.now()

    # Get output path based on pred_version
    version = cfg.pred_version or LATEST_PRED_VERSION
    paths = get_paths_for_version(version)

    exp_dir = paths.reports_experiments_dir / model_type
    ensure_dir(exp_dir)

    all_csv = exp_dir / f"{model_type}_all_runs.csv"

    # Align columns if file exists
    if all_csv.exists():
        orig_df = pd.read_csv(all_csv)

        # Join columns (in case new columns added)
        all_cols = sorted(set(orig_df.columns) | set(df.columns))
        orig_df = orig_df.reindex(columns=all_cols)
        df = df.reindex(columns=all_cols)

        combined = pd.concat([orig_df, df], ignore_index=True)
        combined.to_csv(all_csv, index=False)
    else:
        df.to_csv(all_csv, index=False)

    print(f"[SAVE] Appended {len(df)} rows to: {all_csv}")
    return all_csv