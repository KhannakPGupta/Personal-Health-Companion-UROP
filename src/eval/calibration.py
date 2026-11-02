"""Probability calibration (Platt scaling / Isotonic) and Selective Prediction.

Rules:
  - Calibrator is fitted strictly on held-out validation subjects (never test subjects).
  - Selective prediction abstains when calibrated probability falls in uncertainty band [0.5 - delta, 0.5 + delta].
  - Evaluates the Coverage versus F1 trade-off.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score


class SubjectCalibrator:
    """Calibrator fitted on held-out validation fold probabilities."""

    def __init__(self, method: str = "sigmoid"):
        self.method = method  # 'sigmoid' (Platt) or 'isotonic'
        self.calibrator = None
        self.is_fitted = False

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray) -> "SubjectCalibrator":
        """Fit calibration mapping strictly on validation split."""
        probs = np.asarray(val_probs, dtype=float).reshape(-1, 1)
        y = np.asarray(y_val, dtype=int)

        if self.method == "sigmoid":
            # Platt scaling: LogisticRegression on log-odds / probabilities
            self.calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            self.calibrator.fit(probs, y)
        elif self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(probs.flatten(), y)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_fitted = True
        return self

    def predict_proba(self, test_probs: np.ndarray) -> np.ndarray:
        """Apply fitted calibration to test probabilities."""
        if not self.is_fitted:
            return test_probs

        probs = np.asarray(test_probs, dtype=float)
        if self.method == "sigmoid":
            return self.calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            return self.calibrator.predict(probs.flatten())


def evaluate_selective_prediction(
    y_true: np.ndarray,
    calibrated_probs: np.ndarray,
    delta_steps: int = 20,
) -> pd.DataFrame:
    """Evaluate Coverage vs Macro-F1 across uncertainty rejection bands.
    
    Abstains when |prob - 0.5| < delta.
    
    Args:
        y_true: Ground truth binary labels.
        calibrated_probs: Calibrated probabilities for strain class.
        delta_steps: Number of uncertainty threshold steps between 0.0 and 0.45.
        
    Returns:
        DataFrame with columns: [delta, coverage, macro_f1, strain_f1, abstained_fraction].
    """
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(calibrated_probs, dtype=float)
    deltas = np.linspace(0.0, 0.45, delta_steps)
    results = []

    for delta in deltas:
        # Abstain if probability falls into uncertainty band [0.5 - delta, 0.5 + delta]
        confident_mask = np.abs(probs - 0.5) >= delta
        coverage = float(np.mean(confident_mask))

        if coverage > 0.05 and np.sum(confident_mask) >= 5:
            filtered_y = y_true[confident_mask]
            filtered_pred = (probs[confident_mask] >= 0.5).astype(int)
            m_f1 = float(f1_score(filtered_y, filtered_pred, average="macro", zero_division=0))
            s_f1 = float(f1_score(filtered_y, filtered_pred, pos_label=1, zero_division=0))
        else:
            m_f1 = 0.0
            s_f1 = 0.0

        results.append({
            "delta": round(float(delta), 3),
            "coverage": round(coverage, 4),
            "macro_f1": round(m_f1, 4),
            "strain_f1": round(s_f1, 4),
            "abstained_fraction": round(1.0 - coverage, 4),
        })

    return pd.DataFrame(results)
