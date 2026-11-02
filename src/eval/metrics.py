"""Comprehensive evaluation metrics for physiological strain classification.

Calculates:
  - macro_f1, strain_f1, balanced_accuracy
  - roc_auc, pr_auc, brier_score
  - expected_calibration_error (ECE)
  - false_alerts_per_hour:
      * Raw window false alerts / hour
      * 3-of-5 persistence rule false alerts / hour
  - Bootstrap 95% confidence intervals (1,000 resamples over subjects)
  - Mean +/- SD across subjects, worst-subject performance
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    brier_score_loss,
)


def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE) across n_bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0

    for b in range(n_bins):
        mask = bin_indices == b
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_count / n_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_false_alerts_per_hour(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window_step_sec: float = 30.0,
    persistence_rule: bool = True,
) -> float:
    """Calculate false alert rate per hour of actual baseline data.
    
    Args:
        y_true: Ground truth binary labels (0 = baseline).
        y_pred: Binary model predictions.
        window_step_sec: Window stride in seconds (30s).
        persistence_rule: If True, triggers alert only if 3 of last 5 windows are positive.
        
    Returns:
        False alerts per hour of baseline recording.
    """
    baseline_mask = y_true == 0
    baseline_count = np.sum(baseline_mask)
    if baseline_count == 0:
        return 0.0

    total_baseline_hours = (baseline_count * window_step_sec) / 3600.0

    if not persistence_rule:
        # Raw false positives in baseline windows
        false_alerts = np.sum((y_pred == 1) & baseline_mask)
        return float(false_alerts / total_baseline_hours)

    # 3-of-5 persistence rule:
    # A rolling window over predictions: alert triggered if >= 3 of the last 5 windows are positive
    k = 5
    min_pos = 3
    persisted_preds = np.zeros(len(y_pred), dtype=int)
    for i in range(len(y_pred)):
        start = max(0, i - k + 1)
        sub_window = y_pred[start : i + 1]
        if np.sum(sub_window == 1) >= min_pos:
            persisted_preds[i] = 1

    false_alerts = np.sum((persisted_preds == 1) & baseline_mask)
    return float(false_alerts / total_baseline_hours)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    window_step_sec: float = 30.0,
) -> Dict[str, float]:
    """Compute primary evaluation metrics for a single prediction array."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    strain_f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    # ROC-AUC & PR-AUC
    try:
        if len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = float(auc(recall, precision))
        else:
            roc_auc = 0.5
            pr_auc = 0.0
    except Exception:
        roc_auc = 0.5
        pr_auc = 0.0

    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_expected_calibration_error(y_true, y_prob)

    fa_raw = compute_false_alerts_per_hour(y_true, y_pred, window_step_sec=window_step_sec, persistence_rule=False)
    fa_persisted = compute_false_alerts_per_hour(y_true, y_pred, window_step_sec=window_step_sec, persistence_rule=True)

    return {
        "macro_f1": macro_f1,
        "strain_f1": strain_f1,
        "balanced_accuracy": balanced_acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "ece": ece,
        "false_alerts_raw_per_hr": fa_raw,
        "false_alerts_persisted_per_hr": fa_persisted,
    }


def bootstrap_subject_confidence_intervals(
    per_subject_metrics: List[Dict[str, float]],
    metric_name: str = "macro_f1",
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Calculate mean and bootstrap 95% CI over subject scores."""
    rng = np.random.default_rng(seed)
    scores = np.array([m[metric_name] for m in per_subject_metrics])
    if len(scores) == 0:
        return 0.0, 0.0, 0.0

    boot_means = []
    n = len(scores)
    for _ in range(n_resamples):
        sample = rng.choice(scores, size=n, replace=True)
        boot_means.append(np.mean(sample))

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_means, 100.0 * alpha))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    mean_val = float(np.mean(scores))

    return mean_val, ci_lower, ci_upper
