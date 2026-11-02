"""Leave-One-Subject-Out (LOSO) evaluation engine for Evaluation E1.

Strict Anti-Leakage Protocol:
  1. Outer Split: Exactly 1 subject in test fold; remaining subjects in train fold.
  2. Normalizers / Scalers fitted strictly on train fold.
  3. assert_no_subject_leakage executed on every fold.
  4. Models evaluated across WESAD, PhysioNet, and Pooled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch

from src.features.normalizer import GlobalZScoreNormalizer
from src.eval.metrics import compute_classification_metrics, bootstrap_subject_confidence_intervals
from src.eval.calibration import SubjectCalibrator
from tests.test_leakage_assertion import assert_no_subject_leakage


def run_loso_evaluation(
    df_features: pd.DataFrame,
    model_name: str,
    feature_cols: List[str],
    dataset_name: str = "WESAD",
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute LOSO cross-validation across all unique subjects in df_features.
    
    Args:
        df_features: Tabular DataFrame containing metadata and features.
        model_name: Identifier of model (e.g. 'm0_logistic_regression', 'm1_random_forest').
        feature_cols: Names of numeric feature columns.
        dataset_name: Label for tracking ('WESAD', 'PhysioNet', 'Pooled').
        random_seed: Seed for stochastic classifiers.
        
    Returns:
        Tuple of (per_subject_results_df, overall_summary_dict).
    """
    subjects = df_features["subject_id"].unique()
    per_subject_rows = []

    for test_sub in subjects:
        train_mask = df_features["subject_id"] != test_sub
        test_mask = df_features["subject_id"] == test_sub

        train_subs = df_features.loc[train_mask, "subject_id"].unique()
        test_subs = [test_sub]

        # Enforce Rule 2: Automated assert against subject leakage
        assert_no_subject_leakage(train_subs, test_subs)

        df_train = df_features[train_mask]
        df_test = df_features[test_mask]

        if len(df_train) == 0 or len(df_test) == 0:
            continue

        # Fit normalizer strictly on train fold
        normalizer = GlobalZScoreNormalizer(feature_cols=feature_cols)
        normalizer.fit(df_train)

        norm_train = normalizer.transform(df_train)
        norm_test = normalizer.transform(df_test)

        X_train = norm_train[feature_cols].values
        y_train = norm_train["label"].values.astype(int)
        X_test = norm_test[feature_cols].values
        y_test = norm_test["label"].values.astype(int)

        # Skip test folds that contain only 1 class (cannot compute AUC)
        if len(np.unique(y_test)) < 2 or len(np.unique(y_train)) < 2:
            continue

        # Train model
        if model_name == "m0_majority":
            from sklearn.dummy import DummyClassifier
            clf = DummyClassifier(strategy="most_frequent")
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1] if clf.classes_.size > 1 else np.zeros(len(X_test))

        elif model_name == "m0_logistic_regression":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_seed)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]

        elif model_name == "m1_random_forest":
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=random_seed, n_jobs=-1)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]

        elif model_name == "m1_decision_tree":
            from sklearn.tree import DecisionTreeClassifier
            clf = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=random_seed)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]

        else:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(random_state=random_seed)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]

        # Compute metrics on test subject
        metrics = compute_classification_metrics(y_test, probs)
        metrics["subject_id"] = test_sub
        metrics["dataset"] = dataset_name
        metrics["model"] = model_name
        metrics["n_windows"] = len(y_test)
        per_subject_rows.append(metrics)

    res_df = pd.DataFrame(per_subject_rows)
    if res_df.empty:
        return res_df, {}

    # Compile aggregated statistics
    summary = {
        "dataset": dataset_name,
        "model": model_name,
        "macro_f1_mean": round(float(res_df["macro_f1"].mean()), 4),
        "macro_f1_std": round(float(res_df["macro_f1"].std()), 4),
        "macro_f1_worst": round(float(res_df["macro_f1"].min()), 4),
        "strain_f1_mean": round(float(res_df["strain_f1"].mean()), 4),
        "balanced_acc_mean": round(float(res_df["balanced_accuracy"].mean()), 4),
        "roc_auc_mean": round(float(res_df["roc_auc"].mean()), 4),
        "ece_mean": round(float(res_df["ece"].mean()), 4),
        "false_alerts_persisted_mean": round(float(res_df["false_alerts_persisted_per_hr"].mean()), 2),
    }

    # 1,000 bootstrap confidence intervals over subject scores
    mean_f1, ci_low, ci_high = bootstrap_subject_confidence_intervals(per_subject_rows, metric_name="macro_f1")
    summary["macro_f1_95ci_lower"] = round(ci_low, 4)
    summary["macro_f1_95ci_upper"] = round(ci_high, 4)

    return res_df, summary
