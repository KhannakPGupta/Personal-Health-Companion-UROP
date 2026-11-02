"""Strict subject-isolated normalizers for tabular features and raw tensors.

Enforces Rule 2: Fit scalers, imputers, and calibrators inside training folds only.
Zero subject leakage guaranteed.

Variants:
  - GlobalZScoreNormalizer (Primary): Fits mean and std solely on training subjects.
  - PerSubjectBaselineNormalizer (Sensitivity): Normalizes by each subject's designated baseline period.
    Note: Requires an explicit resting baseline calibration period that on-device real deployment may not have.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


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
            # Autodetect numeric columns excluding metadata
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
                # If no baseline rows, use group mean
                baseline_rows = group

            sub_means = baseline_rows[cols].mean()
            sub_stds = baseline_rows[cols].std().replace(0.0, 1.0).fillna(1.0)

            sub_indices = group.index
            df_out.loc[sub_indices, cols] = (df_out.loc[sub_indices, cols] - sub_means) / (sub_stds + self.eps)

        return df_out
