"""Phase 2 — CPCB India Ambient Air Quality Index (AQI) Pipeline & Hazard Model.

Supports the CPCB (Central Pollution Control Board) India dataset schema:
  Columns: [City, Date, PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene, AQI, AQI_Bucket]

Implements:
  1. Official CPCB National Air Quality Index (NAQI) 6-category classification:
       - Good (0-50), Satisfactory (51-100), Moderate (101-200),
         Poor (201-300), Very Poor (301-400), Severe (401+)
  2. Sub-index calculation from PM2.5 and PM10 CPCB linear breakpoints when AQI column is missing.
  3. Chronological (non-leaky) imputation and ambient hazard classifier predicting
     elevated environmental hazard (AQI > 200: Poor / Very Poor / Severe).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score


class NAQITier(IntEnum):
    """CPCB National Air Quality Index (NAQI) 6-level hazard tiers."""
    GOOD = 0           # AQI 0 - 50
    SATISFACTORY = 1   # AQI 51 - 100
    MODERATE = 2       # AQI 101 - 200
    POOR = 3           # AQI 201 - 300
    VERY_POOR = 4      # AQI 301 - 400
    SEVERE = 5         # AQI > 400


NAQI_LABELS: Dict[NAQITier, str] = {
    NAQITier.GOOD: "Good",
    NAQITier.SATISFACTORY: "Satisfactory",
    NAQITier.MODERATE: "Moderate",
    NAQITier.POOR: "Poor",
    NAQITier.VERY_POOR: "Very Poor",
    NAQITier.SEVERE: "Severe",
}

POLLUTANT_COLS = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]


def map_aqi_to_naqi_tier(aqi_value: float) -> Tuple[NAQITier, str]:
    """Map numeric CPCB AQI value to its NAQI tier and category label."""
    if np.isnan(aqi_value) or aqi_value < 0:
        return NAQITier.MODERATE, NAQI_LABELS[NAQITier.MODERATE]
    if aqi_value <= 50:
        return NAQITier.GOOD, NAQI_LABELS[NAQITier.GOOD]
    elif aqi_value <= 100:
        return NAQITier.SATISFACTORY, NAQI_LABELS[NAQITier.SATISFACTORY]
    elif aqi_value <= 200:
        return NAQITier.MODERATE, NAQI_LABELS[NAQITier.MODERATE]
    elif aqi_value <= 300:
        return NAQITier.POOR, NAQI_LABELS[NAQITier.POOR]
    elif aqi_value <= 400:
        return NAQITier.VERY_POOR, NAQI_LABELS[NAQITier.VERY_POOR]
    else:
        return NAQITier.SEVERE, NAQI_LABELS[NAQITier.SEVERE]


def compute_cpcb_pm25_subindex(pm25_ug_m3: float) -> float:
    """Compute official CPCB 24-hour PM2.5 sub-index from concentration in ug/m3."""
    if np.isnan(pm25_ug_m3) or pm25_ug_m3 < 0:
        return np.nan
    # CPCB breakpoints: (C_low, C_high, I_low, I_high)
    breakpoints = [
        (0.0, 30.0, 0.0, 50.0),
        (30.0, 60.0, 51.0, 100.0),
        (60.0, 90.0, 101.0, 200.0),
        (90.0, 120.0, 201.0, 300.0),
        (120.0, 250.0, 301.0, 400.0),
        (250.0, 500.0, 401.0, 500.0),
    ]
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if pm25_ug_m3 <= c_hi:
            return i_lo + (i_hi - i_lo) * (pm25_ug_m3 - c_lo) / (c_hi - c_lo)
    return 500.0


def compute_cpcb_pm10_subindex(pm10_ug_m3: float) -> float:
    """Compute official CPCB 24-hour PM10 sub-index from concentration in ug/m3."""
    if np.isnan(pm10_ug_m3) or pm10_ug_m3 < 0:
        return np.nan
    breakpoints = [
        (0.0, 50.0, 0.0, 50.0),
        (50.0, 100.0, 51.0, 100.0),
        (100.0, 250.0, 101.0, 200.0),
        (250.0, 350.0, 201.0, 300.0),
        (350.0, 430.0, 301.0, 400.0),
        (430.0, 600.0, 401.0, 500.0),
    ]
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if pm10_ug_m3 <= c_hi:
            return i_lo + (i_hi - i_lo) * (pm10_ug_m3 - c_lo) / (c_hi - c_lo)
    return 500.0


def estimate_aqi_from_pollutants(row: pd.Series) -> float:
    """Estimate AQI as the dominant (maximum) sub-index across available pollutants."""
    if "AQI" in row and pd.notna(row["AQI"]):
        return float(row["AQI"])
    sub_indices = []
    if "PM2.5" in row and pd.notna(row["PM2.5"]):
        sub_indices.append(compute_cpcb_pm25_subindex(float(row["PM2.5"])))
    if "PM10" in row and pd.notna(row["PM10"]):
        sub_indices.append(compute_cpcb_pm10_subindex(float(row["PM10"])))
    valid = [s for s in sub_indices if pd.notna(s)]
    return float(max(valid)) if valid else 120.0


class AmbientAQIPredictor:
    """Train-split-enclosed CPCB AQI hazard classifier (predicts AQI > 200: Poor to Severe)."""

    def __init__(self, feature_cols: Optional[List[str]] = None, random_state: int = 42):
        self.feature_cols = feature_cols or POLLUTANT_COLS
        self.random_state = random_state
        self.medians_: Dict[str, float] = {}
        self.clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            class_weight="balanced",
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame) -> "AmbientAQIPredictor":
        """Fit medians and classifier strictly on chronological training split."""
        df_proc = df_train.copy()
        for col in self.feature_cols:
            if col in df_proc.columns:
                med = float(df_proc[col].median())
                self.medians_[col] = med if pd.notna(med) else 0.0
                df_proc[col] = df_proc[col].fillna(self.medians_[col])
            else:
                self.medians_[col] = 0.0
                df_proc[col] = 0.0

        aqi_vals = df_proc.apply(estimate_aqi_from_pollutants, axis=1)
        # Binary hazard target: 1 if AQI > 200 (Poor, Very Poor, Severe), else 0
        y_train = (aqi_vals > 200.0).astype(int).values
        X_train = df_proc[self.feature_cols].values.astype(np.float32)

        self.clf.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict_hazard_proba(self, df_test: pd.DataFrame) -> np.ndarray:
        """Predict probability of elevated ambient air hazard (AQI > 200)."""
        if not self.is_fitted:
            raise RuntimeError("AmbientAQIPredictor must be fitted before calling predict_hazard_proba.")

        df_proc = df_test.copy()
        for col in self.feature_cols:
            if col in df_proc.columns:
                df_proc[col] = df_proc[col].fillna(self.medians_.get(col, 0.0))
            else:
                df_proc[col] = self.medians_.get(col, 0.0)

        X_test = df_proc[self.feature_cols].values.astype(np.float32)
        if self.clf.classes_.size > 1:
            return self.clf.predict_proba(X_test)[:, 1]
        return np.zeros(len(df_proc), dtype=float)

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, float]:
        """Evaluate ambient hazard classification metrics on chronological test split."""
        aqi_vals = df_test.apply(estimate_aqi_from_pollutants, axis=1)
        y_true = (aqi_vals > 200.0).astype(int).values
        probs = self.predict_hazard_proba(df_test)
        preds = (probs >= 0.5).astype(int)

        macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_true, preds))
        roc_auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.5
        return {
            "aqi_hazard_macro_f1": round(macro_f1, 4),
            "aqi_hazard_balanced_accuracy": round(bal_acc, 4),
            "aqi_hazard_roc_auc": round(roc_auc, 4),
        }


def generate_synthetic_cpcb_aqi_data(n_days: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generate realistic multi-city CPCB India daily air quality data for testing and simulation."""
    rng = np.random.default_rng(seed)
    cities = ["Delhi", "Mumbai", "Chennai", "Bengaluru"]
    rows = []

    for city in cities:
        base_pm25 = 115.0 if city == "Delhi" else (55.0 if city == "Mumbai" else 40.0)
        for day_idx in range(n_days):
            seasonal_wave = 45.0 * np.sin(2 * np.pi * day_idx / 60.0)
            pm25 = max(8.0, base_pm25 + seasonal_wave + rng.normal(0, 25.0))
            pm10 = max(15.0, pm25 * 1.85 + rng.normal(0, 30.0))
            no2 = max(5.0, 22.0 + 0.25 * pm25 + rng.normal(0, 6.0))
            co = max(0.1, 0.6 + 0.008 * pm25 + rng.normal(0, 0.2))
            so2 = max(2.0, 10.0 + 0.08 * pm25 + rng.normal(0, 3.0))
            o3 = max(5.0, 35.0 + rng.normal(0, 10.0))

            aqi = max(compute_cpcb_pm25_subindex(pm25), compute_cpcb_pm10_subindex(pm10))
            _, bucket = map_aqi_to_naqi_tier(aqi)
            rows.append({
                "City": city,
                "DayIndex": day_idx,
                "PM2.5": round(pm25, 2),
                "PM10": round(pm10, 2),
                "NO2": round(no2, 2),
                "CO": round(co, 2),
                "SO2": round(so2, 2),
                "O3": round(o3, 2),
                "AQI": round(aqi, 1),
                "AQI_Bucket": bucket,
            })

    return pd.DataFrame(rows)

