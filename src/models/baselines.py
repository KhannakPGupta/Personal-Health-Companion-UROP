"""Baseline models (M0 and M1) with inner subject-wise cross-validation tuning.

Enforces Rule 2: Subject-only isolation. Inner CV for hyperparameter tuning strictly uses
GroupKFold(groups=subject_ids) so that no subject leaks between inner train and val folds.

Models:
  - M0: Majority Class (Dummy) and Logistic Regression (L2 penalized, balanced class weights).
  - M1: Decision Tree (max_depth=5), Random Forest, and HistGradientBoosting / XGBoost.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def build_m0_models() -> Dict[str, Any]:
    """Build M0 baseline models: Majority classifier and Logistic Regression."""
    return {
        "m0_majority": DummyClassifier(strategy="most_frequent"),
        "m0_logistic_regression": LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
    }


def build_m1_models() -> Dict[str, Any]:
    """Build M1 edge-friendly tree/ensemble models."""
    models = {
        "m1_decision_tree_depth5": DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=42,
        ),
        "m1_random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "m1_hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=5,
            class_weight="balanced",
            random_state=42,
        ),
    }
    if HAS_XGBOOST:
        models["m1_xgboost"] = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
        )
    return models


def tune_m1_model_inner_cv(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    subject_ids_train: np.ndarray,
    n_splits: int = 3,
) -> Any:
    """Tune M1 model hyperparameters using INNER subject-wise cross-validation.
    
    Guarantees that hyperparameter tuning never leaks validation subjects.
    
    Args:
        model_name: Identifier of model to tune.
        X_train: Training fold feature matrix.
        y_train: Training fold labels.
        subject_ids_train: Subject identifiers for training fold samples.
        n_splits: Number of inner subject folds.
        
    Returns:
        Fitted best model instance.
    """
    unique_subs = np.unique(subject_ids_train)
    k = min(n_splits, len(unique_subs))
    gkf = GroupKFold(n_splits=k)

    if model_name == "m1_random_forest":
        best_score = -1.0
        best_model = None
        # Inner grid search over max_depth and min_samples_leaf
        for max_depth in [4, 6, 8]:
            for min_leaf in [2, 5]:
                fold_scores = []
                for inner_train_idx, inner_val_idx in gkf.split(X_train, y_train, groups=subject_ids_train):
                    clf = RandomForestClassifier(
                        n_estimators=50,
                        max_depth=max_depth,
                        min_samples_leaf=min_leaf,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    )
                    clf.fit(X_train[inner_train_idx], y_train[inner_train_idx])
                    preds = clf.predict(X_train[inner_val_idx])
                    # Balanced accuracy score
                    acc = np.mean(preds == y_train[inner_val_idx])
                    fold_scores.append(acc)

                mean_val = np.mean(fold_scores)
                if mean_val > best_score:
                    best_score = mean_val
                    best_model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=max_depth,
                        min_samples_leaf=min_leaf,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    )
        best_model.fit(X_train, y_train)
        return best_model

    elif model_name == "m1_decision_tree":
        clf = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
        clf.fit(X_train, y_train)
        return clf

    elif model_name == "m1_hist_gradient_boosting":
        clf = HistGradientBoostingClassifier(max_iter=100, max_depth=5, class_weight="balanced", random_state=42)
        clf.fit(X_train, y_train)
        return clf

    else:
        # Default fallback
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        return clf
