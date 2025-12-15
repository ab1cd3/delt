# src/model_eval/experiments/tree_models.py

from __future__ import annotations

from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from model_eval.experiments.training_utils import SEED


def build_base_tree_models() -> Dict[str, object]:
    """
    Base (not-tuned) decision classifier models:
        - LogisticRegresion
        - RandomForest
        - XGBoost
        - LightGBM
    """
    models: Dict[str, object] = {}

    # Logistic Regression (with scaling)
    models["logreg"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                class_weight="balanced",
                max_iter=1000,
                random_state=SEED,
            )),
        ]
    )

    # Random Forest
    models["rf"] = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    )

    # XGBoost
    models["xgb"] = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=SEED,
    )

    # LightGBM
    models["lgbm"] = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        class_weight="balanced",
        n_jobs=-1,
    )

    return models


def get_param_distributions() -> Dict[str, Dict[str, Any]]:
    """
    Parameter distributions for RandomizedSearchCV per model name.
    """
    param_dists: Dict[str, Dict[str, Any]] = {}

    # Logistic regression (C only, rest fixed)
    param_dists["logreg"] = {
        "clf__C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
    }

    # Random Forest
    param_dists["rf"] = {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
    }

    # XGBoost
    param_dists["xgb"] = {
        "n_estimators": [200, 300, 500],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.02, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "scale_pos_weight": [3.0, 5.0, 7.0, 9.0],
    }

    # LightGBM
    param_dists["lgbm"] = {
        "n_estimators": [200, 300, 500],
        "learning_rate": [0.01, 0.02, 0.05, 0.1],
        "num_leaves": [15, 31, 63],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

    return param_dists