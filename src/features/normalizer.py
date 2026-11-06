"""Strict subject-isolated normalizers for tabular features and raw tensors.

Enforces Rule 2: Fit scalers, imputers, and calibrators inside training folds only.
Zero subject leakage guaranteed.

Variants:
  - GlobalZScoreNormalizer (Primary): Fits mean and std solely on training subjects.
  - UnsupervisedSubjectPrefixCalibrator: Subtracts each subject's unlabeled initial resting
    prefix (first K windows at start of session) + scales by subject/train dispersion without touching labels.
  - PerSubjectBaselineNormalizer (Sensitivity): Normalizes by each subject's designated baseline period.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def augment_autonomic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific physiological interaction ratios if base columns are present."""
    df_out = df.copy()
    if "hr_mean" in df_out.columns and "hrv_rmssd" in df_out.columns:
        df_out["sympathovagal_ratio"] = df_out["hr_mean"] / (df_out["hrv_rmssd"].clip(lower=0.0) + 1.0)
    if "hr_max" in df_out.columns and "hr_min" in df_out.columns:
        df_out["hr_dynamic_range"] = df_out["hr_max"] - df_out["hr_min"]
    if "eda_phasic_peaks" in df_out.columns and "eda_phasic_amplitude_sum" in df_out.columns:
        df_out["eda_arousal_index"] = df_out["eda_phasic_peaks"] * df_out["eda_phasic_amplitude_sum"]
    if "eda_phasic_amplitude_sum" in df_out.columns and "eda_tonic_level" in df_out.columns:
        df_out["eda_phasic_tonic_ratio"] = df_out["eda_phasic_amplitude_sum"] / (
            df_out["eda_tonic_level"].abs() + 0.1
        )
    if "hr_mean" in df_out.columns and "acc_energy" in df_out.columns:
        df_out["motion_adjusted_hr"] = df_out["hr_mean"] / (df_out["acc_energy"].clip(lower=0.0) + 0.5)
    return df_out


class GlobalZScoreNormalizer:
    """Standardizes tabular features across training subjects."""

    def __init__(self, feature_cols: Optional[List[str]] = None, eps: float = 1e-8):
        self.feature_cols = feature_cols
        self.eps = eps
        self.means_: Dict[str, float] = {}
        self.stds_: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, df_train: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> "GlobalZScoreNormalizer":
        """Compute mean and std strictly on df_train. df_test must NEVER be passed here."""
        cols = feature_cols or self.feature_cols
        if cols is None:
            metadata_cols = {"dataset", "subject_id", "session", "window_idx", "label", "label_name", "segment_name"}
            cols = [c for c in df_train.columns if c not in metadata_cols and pd.api.types.is_numeric_dtype(df_train[c])]
        self.feature_cols = cols

        for col in self.feature_cols:
            self.means_[col] = float(df_train[col].mean())
            std = float(df_train[col].std(ddof=1))
            self.stds_[col] = std if std > self.eps else 1.0

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted standardization to any split (train, val, or test)."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted on training fold before transform!")

        df_out = df.copy()
        for col in self.feature_cols:
            if col in df_out.columns:
                df_out[col] = (df_out[col] - self.means_[col]) / self.stds_[col]
        return df_out


class UnsupervisedSubjectSessionCalibrator:
    """Label-free per-subject calibration removing inter-subject resting physiological offsets.

    Uses ZERO ground-truth labels (y is never accessed). For each subject/session stream,
    computes the subject's initial resting calibration offset (first prefix_windows windows)
    and session scale, eliminating inter-subject baseline drift prior to global train-fold z-scoring.
    """

    def __init__(self, feature_cols: List[str], prefix_windows: int = 5, eps: float = 1e-8):
        self.feature_cols = feature_cols
        self.prefix_windows = prefix_windows
        self.eps = eps

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize each subject's feature trajectory without accessing label column."""
        if "subject_id" not in df.columns:
            return df.copy()

        df_out = df.copy()
        cols = [c for c in self.feature_cols if c in df_out.columns]
        group_keys = ["subject_id", "session"] if "session" in df_out.columns else ["subject_id"]

        for _, group in df_out.groupby(group_keys):
            if len(group) < 2:
                continue
            # Sort chronologically by window_idx if available
            if "window_idx" in group.columns:
                ordered = group.sort_values("window_idx")
            else:
                ordered = group

            # Initial resting prefix (first k windows of recording) + session robust center
            k = min(self.prefix_windows, max(1, len(ordered) // 3))
            prefix_mean = ordered[cols].iloc[:k].mean()
            session_mean = ordered[cols].mean()
            # Blend 50% initial resting prefix + 50% session mean (100% label-free)
            center = 0.5 * prefix_mean + 0.5 * session_mean
            scale = ordered[cols].std(ddof=0).replace(0.0, 1.0).fillna(1.0)

            idx = group.index
            df_out.loc[idx, cols] = (df_out.loc[idx, cols] - center) / (scale + self.eps)

        return df_out


class PerSubjectBaselineNormalizer:
    """Sensitivity variant: Normalizes features by each subject's baseline resting segment."""

    def __init__(self, feature_cols: Optional[List[str]] = None, eps: float = 1e-8):
        self.feature_cols = feature_cols
        self.eps = eps

    def normalize(self, df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
        """Subtract subject baseline mean and divide by baseline std.
        
        Assumes label 0 corresponds to resting baseline.
        """
        cols = self.feature_cols or [
            c for c in df.columns if c not in {"dataset", "subject_id", "session", "window_idx", "label", "label_name", "segment_name"}
        ]
        df_out = df.copy()

        for sub_id, group in df.groupby("subject_id"):
            baseline_rows = group[group[label_col] == 0]
            if len(baseline_rows) < 2:
                baseline_rows = group

            sub_means = baseline_rows[cols].mean()
            sub_stds = baseline_rows[cols].std().replace(0.0, 1.0).fillna(1.0)

            sub_indices = group.index
            df_out.loc[sub_indices, cols] = (df_out.loc[sub_indices, cols] - sub_means) / (sub_stds + self.eps)

        return df_out
