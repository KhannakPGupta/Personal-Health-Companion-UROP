"""Phase 1E — Cross-Dataset Generalization & Transfer Evaluation (Evaluation E2).

Evaluates directional transfer between heterogeneous wearable protocols:
  - Direction A: Train on WESAD (psychological stress) -> Test on PhysioNet (stress & exercise)
  - Direction B: Train on PhysioNet -> Test on WESAD

Strict Anti-Leakage Rules:
  1. assert_no_subject_leakage enforced between source and target cohorts.
  2. GlobalZScoreNormalizer and channel statistics fitted strictly on source training data.
  3. Optional Platt probability calibration fitted strictly on held-out source validation subjects.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

from src.features.normalizer import GlobalZScoreNormalizer
from src.eval.metrics import compute_classification_metrics, bootstrap_subject_confidence_intervals
from src.eval.calibration import SubjectCalibrator
from src.models.cnn_1d import LightweightStrainCNN, train_cnn_with_early_stopping
from tests.test_leakage_assertion import assert_no_subject_leakage


def _instantiate_tabular_model(model_name: str, random_seed: int = 42) -> Any:
    """Instantiate a tabular classifier by identifier."""
    if model_name == "m0_majority":
        from sklearn.dummy import DummyClassifier
        return DummyClassifier(strategy="most_frequent")
    elif model_name == "m0_logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_seed)
    elif model_name == "m1_decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=random_seed)
    elif model_name == "m1_random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=100, max_depth=6, class_weight="balanced", random_state=random_seed, n_jobs=-1
        )
    elif model_name == "m1_hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=100, max_depth=5, class_weight="balanced", random_state=random_seed
        )
    else:
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_seed)


def run_cross_dataset_tabular_transfer(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    model_name: str,
    feature_cols: List[str],
    source_name: str = "WESAD",
    target_name: str = "PhysioNet",
    calibrate_on_source_val: bool = True,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Train a tabular model on df_source and evaluate per-subject on df_target."""
    src_subjects = df_source["subject_id"].unique()
    tgt_subjects = df_target["subject_id"].unique()

    # Enforce zero overlap between source and target subject IDs
    assert_no_subject_leakage(src_subjects, tgt_subjects)

    normalizer = GlobalZScoreNormalizer(feature_cols=feature_cols)
    normalizer.fit(df_source)

    norm_source = normalizer.transform(df_source)
    norm_target = normalizer.transform(df_target)

    calibrator: Optional[SubjectCalibrator] = None
    if calibrate_on_source_val and len(src_subjects) >= 3:
        # Hold out 1 source subject strictly for Platt scaling calibration
        val_sub = src_subjects[-1]
        inner_train_df = norm_source[norm_source["subject_id"] != val_sub]
        inner_val_df = norm_source[norm_source["subject_id"] == val_sub]

        if len(inner_val_df["label"].unique()) >= 2 and len(inner_train_df["label"].unique()) >= 2:
            clf_inner = _instantiate_tabular_model(model_name, random_seed=random_seed)
            clf_inner.fit(inner_train_df[feature_cols].values, inner_train_df["label"].values.astype(int))
            val_probs = (
                clf_inner.predict_proba(inner_val_df[feature_cols].values)[:, 1]
                if clf_inner.classes_.size > 1
                else np.zeros(len(inner_val_df))
            )
            calibrator = SubjectCalibrator(method="sigmoid").fit(val_probs, inner_val_df["label"].values.astype(int))

    # Fit final model on full source dataset
    X_src = norm_source[feature_cols].values
    y_src = norm_source["label"].values.astype(int)
    clf = _instantiate_tabular_model(model_name, random_seed=random_seed)
    clf.fit(X_src, y_src)

    per_subject_rows = []
    for test_sub in tgt_subjects:
        sub_df = norm_target[norm_target["subject_id"] == test_sub]
        y_test = sub_df["label"].values.astype(int)
        if len(np.unique(y_test)) < 2:
            continue

        X_test = sub_df[feature_cols].values
        raw_probs = clf.predict_proba(X_test)[:, 1] if clf.classes_.size > 1 else np.zeros(len(X_test))
        probs = calibrator.predict_proba(raw_probs) if calibrator is not None else raw_probs

        metrics = compute_classification_metrics(y_test, probs)
        metrics["subject_id"] = str(test_sub)
        metrics["direction"] = f"{source_name}->{target_name}"
        metrics["model"] = model_name
        metrics["n_windows"] = len(y_test)
        per_subject_rows.append(metrics)

    res_df = pd.DataFrame(per_subject_rows)
    if res_df.empty:
        return res_df, {}

    summary = {
        "direction": f"{source_name}->{target_name}",
        "source_dataset": source_name,
        "target_dataset": target_name,
        "model": model_name,
        "calibrated": calibrate_on_source_val,
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


def run_cross_dataset_cnn_transfer(
    src_tensors: np.ndarray,
    src_labels: np.ndarray,
    src_subjects: np.ndarray,
    tgt_tensors: np.ndarray,
    tgt_labels: np.ndarray,
    tgt_subjects: np.ndarray,
    source_name: str = "WESAD",
    target_name: str = "PhysioNet",
    epochs: int = 15,
    batch_size: int = 32,
    patience: int = 5,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Train M2 1D-CNN on source dataset and evaluate per-subject on target dataset."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    src_subs_arr = np.asarray(src_subjects)
    tgt_subs_arr = np.asarray(tgt_subjects)
    unique_src = np.unique(src_subs_arr)
    unique_tgt = np.unique(tgt_subs_arr)

    assert_no_subject_leakage(unique_src, unique_tgt)

    X_src = np.asarray(src_tensors, dtype=np.float32)
    X_tgt = np.asarray(tgt_tensors, dtype=np.float32)
    if X_src.ndim == 3 and X_src.shape[1] > X_src.shape[2]:
        X_src = np.transpose(X_src, (0, 2, 1))
    if X_tgt.ndim == 3 and X_tgt.shape[1] > X_tgt.shape[2]:
        X_tgt = np.transpose(X_tgt, (0, 2, 1))

    y_src = np.asarray(src_labels, dtype=np.float32)
    y_tgt = np.asarray(tgt_labels, dtype=np.float32)

    # Split source into inner train and 1 source validation subject for early stopping
    val_sub = unique_src[-1]
    inner_train_mask = src_subs_arr != val_sub
    inner_val_mask = src_subs_arr == val_sub
    if np.sum(inner_train_mask) == 0 or np.sum(inner_val_mask) == 0:
        inner_train_mask = np.ones(len(src_subs_arr), dtype=bool)
        inner_val_mask = np.ones(len(src_subs_arr), dtype=bool)

    # Fit channel-wise statistics strictly on source inner train
    ch_mean = X_src[inner_train_mask].mean(axis=(0, 2), keepdims=True)
    ch_std = X_src[inner_train_mask].std(axis=(0, 2), keepdims=True) + 1e-8

    X_tr_t = torch.tensor((X_src[inner_train_mask] - ch_mean) / ch_std, dtype=torch.float32)
    y_tr_t = torch.tensor(y_src[inner_train_mask], dtype=torch.float32)
    X_val_t = torch.tensor((X_src[inner_val_mask] - ch_mean) / ch_std, dtype=torch.float32)
    y_val_t = torch.tensor(y_src[inner_val_mask], dtype=torch.float32)

    model = LightweightStrainCNN(in_channels=X_src.shape[1], num_classes=1)
    model, _ = train_cnn_with_early_stopping(
        model, X_tr_t, y_tr_t, X_val_t, y_val_t, epochs=epochs, batch_size=batch_size, patience=patience
    )
    model.eval()

    per_subject_rows = []
    for test_sub in unique_tgt:
        test_mask = tgt_subs_arr == test_sub
        y_test = y_tgt[test_mask].astype(int)
        if len(np.unique(y_test)) < 2:
            continue

        X_te_t = torch.tensor((X_tgt[test_mask] - ch_mean) / ch_std, dtype=torch.float32)
        with torch.no_grad():
            logits = model(X_te_t).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        if np.ndim(probs) == 0:
            probs = np.array([float(probs)])

        metrics = compute_classification_metrics(y_test, probs)
        metrics["subject_id"] = str(test_sub)
        metrics["direction"] = f"{source_name}->{target_name}"
        metrics["model"] = "m2_cnn_1d"
        metrics["n_windows"] = len(y_test)
        per_subject_rows.append(metrics)

    res_df = pd.DataFrame(per_subject_rows)
    if res_df.empty:
        return res_df, {}

    summary = {
        "direction": f"{source_name}->{target_name}",
        "source_dataset": source_name,
        "target_dataset": target_name,
        "model": "m2_cnn_1d",
        "calibrated": False,
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
