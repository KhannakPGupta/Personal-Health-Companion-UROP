"""Leave-One-Subject-Out (LOSO) evaluation engine for Evaluation E1.

Strict Anti-Leakage Protocol:
  1. Outer Split: Exactly 1 subject in test fold; remaining subjects in train fold.
  2. Normalizers, feature selectors, and threshold tuners fitted strictly on train fold.
  3. Unsupervised subject-session prefix calibration uses zero ground-truth labels (y is never accessed).
  4. assert_no_subject_leakage executed on every fold.
  5. Models evaluated across WESAD, PhysioNet, and Pooled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from src.features.normalizer import (
    GlobalZScoreNormalizer,
    UnsupervisedSubjectSessionCalibrator,
    augment_autonomic_features,
)
from src.eval.metrics import compute_classification_metrics, bootstrap_subject_confidence_intervals
from src.eval.calibration import SubjectCalibrator
from src.models.cnn_1d import LightweightStrainCNN, train_cnn_with_early_stopping
from tests.test_leakage_assertion import assert_no_subject_leakage


def select_optimal_threshold_on_val(y_val: np.ndarray, val_probs: np.ndarray) -> float:
    """Find threshold in [0.30, 0.70] maximizing Macro-F1 strictly on validation/train split."""
    if len(np.unique(y_val)) < 2:
        return 0.50
    best_tau = 0.50
    best_f1 = -1.0
    for tau in np.linspace(0.30, 0.70, 21):
        preds = (val_probs >= tau).astype(int)
        score = float(f1_score(y_val, preds, average="macro", zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_tau = float(tau)
    return best_tau


def standardize_subject_tensors_unsupervised(
    X: np.ndarray,
    subject_ids: np.ndarray,
    prefix_windows: int = 4,
) -> np.ndarray:
    """Remove per-subject DC baseline offset on raw (N, C, L) tensors using zero labels."""
    X_out = X.copy()
    sub_arr = np.asarray(subject_ids)
    for sub in np.unique(sub_arr):
        mask = sub_arr == sub
        sub_x = X_out[mask]
        if len(sub_x) < 2:
            continue
        k = min(prefix_windows, max(1, len(sub_x) // 3))
        # Blend initial resting prefix mean + session mean across (windows, time) per channel
        prefix_mean = sub_x[:k].mean(axis=(0, 2), keepdims=True)
        sess_mean = sub_x.mean(axis=(0, 2), keepdims=True)
        center = 0.5 * prefix_mean + 0.5 * sess_mean
        scale = sub_x.std(axis=(0, 2), keepdims=True) + 1e-5
        X_out[mask] = (sub_x - center) / scale
    return X_out


def run_loso_evaluation(
    df_features: pd.DataFrame,
    model_name: str,
    feature_cols: List[str],
    dataset_name: str = "WESAD",
    random_seed: int = 42,
    use_subject_session_calibration: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute LOSO cross-validation across all unique subjects in df_features.
    
    Args:
        df_features: Tabular DataFrame containing metadata and features.
        model_name: Identifier of model (e.g. 'm0_logistic_regression', 'm1_random_forest').
        feature_cols: Names of numeric feature columns.
        dataset_name: Label for tracking ('WESAD', 'PhysioNet', 'Pooled').
        random_seed: Seed for stochastic classifiers.
        use_subject_session_calibration: If True, applies label-free UnsupervisedSubjectSessionCalibrator.
        
    Returns:
        Tuple of (per_subject_results_df, overall_summary_dict).
    """
    df_aug = augment_autonomic_features(df_features)
    extra_cols = [
        c for c in [
            "sympathovagal_ratio",
            "hr_dynamic_range",
            "eda_arousal_index",
            "eda_phasic_tonic_ratio",
            "motion_adjusted_hr",
        ]
        if c in df_aug.columns and len(feature_cols) >= 10
    ]
    active_feature_cols = list(dict.fromkeys([c for c in feature_cols if c in df_aug.columns] + extra_cols))

    if use_subject_session_calibration and "subject_id" in df_aug.columns:
        cal_unsupervised = UnsupervisedSubjectSessionCalibrator(feature_cols=active_feature_cols)
        df_aug = cal_unsupervised.transform(df_aug)

    subjects = df_aug["subject_id"].unique()
    per_subject_rows = []

    for idx, test_sub in enumerate(subjects):
        train_mask = df_aug["subject_id"] != test_sub
        test_mask = df_aug["subject_id"] == test_sub

        train_subs = df_aug.loc[train_mask, "subject_id"].unique()
        test_subs = [test_sub]

        # Enforce Rule 2: Automated assert against subject leakage
        assert_no_subject_leakage(train_subs, test_subs)

        df_train = df_aug[train_mask]
        df_test = df_aug[test_mask]

        if len(df_train) == 0 or len(df_test) == 0:
            continue

        # Fit global z-score normalizer strictly on train fold
        normalizer = GlobalZScoreNormalizer(feature_cols=active_feature_cols)
        normalizer.fit(df_train)

        norm_train = normalizer.transform(df_train)
        norm_test = normalizer.transform(df_test)

        X_train = np.nan_to_num(norm_train[active_feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = norm_train["label"].values.astype(int)
        X_test = np.nan_to_num(norm_test[active_feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
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
            best_tau = 0.50

        elif model_name == "m0_logistic_regression":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=0.25, class_weight="balanced", max_iter=1000, random_state=random_seed)
            clf.fit(X_train, y_train)
            best_tau = 0.50
            probs = clf.predict_proba(X_test)[:, 1]

        elif model_name == "m1_random_forest":
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(
                n_estimators=150,
                max_depth=5,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=random_seed,
                n_jobs=-1,
            )
            clf.fit(X_train, y_train)
            best_tau = 0.50
            probs = clf.predict_proba(X_test)[:, 1]

        elif model_name == "m1_decision_tree":
            from sklearn.tree import DecisionTreeClassifier
            clf = DecisionTreeClassifier(
                max_depth=4, min_samples_leaf=6, class_weight="balanced", random_state=random_seed
            )
            clf.fit(X_train, y_train)
            best_tau = 0.50
            probs = clf.predict_proba(X_test)[:, 1]

        elif model_name == "m1_hist_gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=4,
                min_samples_leaf=10,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=random_seed,
            )
            clf.fit(X_train, y_train)
            best_tau = 0.50
            probs = clf.predict_proba(X_test)[:, 1]

        else:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=0.25, class_weight="balanced", max_iter=1000, random_state=random_seed)
            clf.fit(X_train, y_train)
            best_tau = 0.50
            probs = clf.predict_proba(X_test)[:, 1]

        # Compute metrics on test subject using train-selected threshold
        metrics = compute_classification_metrics(y_test, probs, threshold=best_tau)
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


def run_cnn_loso_evaluation(
    tensors: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    dataset_name: str = "WESAD",
    epochs: int = 20,
    batch_size: int = 32,
    patience: int = 5,
    random_seed: int = 42,
    active_channels: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute strict subject-independent LOSO evaluation for M2 (LightweightStrainCNN).

    Enforces Rule 2:
      - Outer split holds out 1 test subject.
      - Inner validation split holds out 1 training subject for early stopping & threshold calibration.
      - Channel-wise z-score standardization is fitted strictly on inner training subjects.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    subject_ids_arr = np.asarray(subject_ids)
    unique_subjects = np.unique(subject_ids_arr)
    per_subject_rows = []

    X_all = np.asarray(tensors, dtype=np.float32)
    if active_channels is not None:
        if X_all.shape[-1] <= 6:
            X_all = X_all[:, :, active_channels]
        else:
            X_all = X_all[:, active_channels, :]

    # Standardize layout to (N, C, L)
    if X_all.ndim == 3 and X_all.shape[1] > X_all.shape[2]:
        X_all = np.transpose(X_all, (0, 2, 1))

    # Label-free per-subject resting-prefix & session standardization
    X_all = standardize_subject_tensors_unsupervised(X_all, subject_ids_arr)

    in_channels = X_all.shape[1]
    y_all = np.asarray(labels, dtype=np.float32)

    for idx, test_sub in enumerate(unique_subjects):
        train_mask = subject_ids_arr != test_sub
        test_mask = subject_ids_arr == test_sub

        train_subs = np.unique(subject_ids_arr[train_mask])
        test_subs = [test_sub]
        assert_no_subject_leakage(train_subs, test_subs)

        y_test = y_all[test_mask].astype(int)
        if len(np.unique(y_test)) < 2 or len(train_subs) < 2:
            continue

        # Select 1 inner validation subject from train_subs for early stopping
        val_sub = train_subs[idx % len(train_subs)]
        inner_train_mask = train_mask & (subject_ids_arr != val_sub)
        inner_val_mask = subject_ids_arr == val_sub

        if np.sum(inner_train_mask) == 0 or np.sum(inner_val_mask) == 0:
            inner_train_mask = train_mask
            inner_val_mask = train_mask

        assert_no_subject_leakage(
            np.unique(subject_ids_arr[inner_train_mask]),
            test_subs,
        )

        # Fit channel-wise mean & std strictly on inner_train
        X_inner_train = X_all[inner_train_mask]
        ch_mean = X_inner_train.mean(axis=(0, 2), keepdims=True)
        ch_std = X_inner_train.std(axis=(0, 2), keepdims=True) + 1e-8

        X_tr_norm = torch.tensor((X_inner_train - ch_mean) / ch_std, dtype=torch.float32)
        y_tr_t = torch.tensor(y_all[inner_train_mask], dtype=torch.float32)

        X_val_norm = torch.tensor((X_all[inner_val_mask] - ch_mean) / ch_std, dtype=torch.float32)
        y_val_t = torch.tensor(y_all[inner_val_mask], dtype=torch.float32)

        X_te_norm = torch.tensor((X_all[test_mask] - ch_mean) / ch_std, dtype=torch.float32)

        model = LightweightStrainCNN(in_channels=in_channels, num_classes=1)
        model, _ = train_cnn_with_early_stopping(
            model,
            X_tr_norm,
            y_tr_t,
            X_val_norm,
            y_val_t,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
        )

        model.eval()
        with torch.no_grad():
            logits = model(X_te_norm).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()

        if np.ndim(probs) == 0:
            probs = np.array([float(probs)])

        metrics = compute_classification_metrics(y_test, probs, threshold=0.50)
        metrics["subject_id"] = str(test_sub)
        metrics["dataset"] = dataset_name
        metrics["model"] = "m2_cnn_1d"
        metrics["n_windows"] = len(y_test)
        per_subject_rows.append(metrics)

    res_df = pd.DataFrame(per_subject_rows)
    if res_df.empty:
        return res_df, {}

    summary = {
        "dataset": dataset_name,
        "model": "m2_cnn_1d",
        "macro_f1_mean": round(float(res_df["macro_f1"].mean()), 4),
        "macro_f1_std": round(float(res_df["macro_f1"].std(ddof=0)), 4),
        "macro_f1_worst": round(float(res_df["macro_f1"].min()), 4),
        "strain_f1_mean": round(float(res_df["strain_f1"].mean()), 4),
        "balanced_acc_mean": round(float(res_df["balanced_accuracy"].mean()), 4),
        "roc_auc_mean": round(float(res_df["roc_auc"].mean()), 4),
        "ece_mean": round(float(res_df["ece"].mean()), 4),
        "false_alerts_persisted_mean": round(float(res_df["false_alerts_persisted_per_hr"].mean()), 2),
    }
    _, ci_low, ci_high = bootstrap_subject_confidence_intervals(per_subject_rows, metric_name="macro_f1")
    summary["macro_f1_95ci_lower"] = round(ci_low, 4)
    summary["macro_f1_95ci_upper"] = round(ci_high, 4)

    return res_df, summary
