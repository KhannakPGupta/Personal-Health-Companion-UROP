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
    """Chronological Next-Day (t+1) CPCB AQI Hazard Forecaster (predicts AQI_{t+1} > 200 from day-t pollutants)."""

    def __init__(self, feature_cols: Optional[List[str]] = None, random_state: int = 42):
        self.base_cols = feature_cols or POLLUTANT_COLS
        self.feature_cols = list(self.base_cols) + ["PM2.5_roll3", "PM10_roll3", "PM2.5_diff1"]
        self.random_state = random_state
        self.medians_: Dict[str, float] = {}
        self.clf = RandomForestClassifier(
            n_estimators=120,
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fitted = False

    def _prepare_causal_features_and_next_day_target(
        self, df: pd.DataFrame, is_train: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute causal day-t features (with 3-day rolling lags) and next-day (t+1) binary hazard target."""
        df_proc = df.copy()
        sort_cols = [c for c in ["City", "Date", "DayIndex"] if c in df_proc.columns]
        if sort_cols:
            df_proc = df_proc.sort_values(sort_cols).copy()

        for col in self.base_cols:
            if col in df_proc.columns:
                if is_train:
                    med = float(df_proc[col].median())
                    self.medians_[col] = med if pd.notna(med) else 0.0
                df_proc[col] = df_proc[col].fillna(self.medians_.get(col, 0.0))
            else:
                if is_train:
                    self.medians_[col] = 0.0
                df_proc[col] = self.medians_.get(col, 0.0)

        # Causal rolling 3-day averages and 1-day delta per city (uses only <= t)
        if "City" in df_proc.columns:
            df_proc["PM2.5_roll3"] = df_proc.groupby("City")["PM2.5"].transform(
                lambda s: s.rolling(3, min_periods=1).mean()
            )
            df_proc["PM10_roll3"] = df_proc.groupby("City")["PM10"].transform(
                lambda s: s.rolling(3, min_periods=1).mean()
            )
            df_proc["PM2.5_diff1"] = df_proc.groupby("City")["PM2.5"].transform(
                lambda s: s.diff().fillna(0.0)
            )
        else:
            df_proc["PM2.5_roll3"] = df_proc["PM2.5"].rolling(3, min_periods=1).mean()
            df_proc["PM10_roll3"] = df_proc["PM10"].rolling(3, min_periods=1).mean()
            df_proc["PM2.5_diff1"] = df_proc["PM2.5"].diff().fillna(0.0)

        # Estimate today's AQI_t and next-day AQI_{t+1} per city
        if "AQI" in df_proc.columns and df_proc["AQI"].notna().any():
            aqi_today = df_proc["AQI"].copy()
            missing_aqi = aqi_today.isna()
            if missing_aqi.any():
                aqi_today.loc[missing_aqi] = df_proc.loc[missing_aqi].apply(estimate_aqi_from_pollutants, axis=1)
        else:
            aqi_today = df_proc.apply(estimate_aqi_from_pollutants, axis=1)

        df_proc["_aqi_today"] = aqi_today.astype(float)
        if "City" in df_proc.columns:
            df_proc["_aqi_next_day"] = df_proc.groupby("City")["_aqi_today"].shift(-1)
        else:
            df_proc["_aqi_next_day"] = df_proc["_aqi_today"].shift(-1)

        # Fill last date of each city with today's AQI so array length matches len(df)
        df_proc["_aqi_next_day"] = df_proc["_aqi_next_day"].fillna(df_proc["_aqi_today"])

        # Restore original row order of df
        df_proc = df_proc.loc[df.index]

        X = np.nan_to_num(df_proc[self.feature_cols].values.astype(np.float32), nan=0.0)
        y_next = (df_proc["_aqi_next_day"].values > 200.0).astype(int)
        y_today = (df_proc["_aqi_today"].values > 200.0).astype(int)
        return X, y_next, y_today

    def fit(self, df_train: pd.DataFrame) -> "AmbientAQIPredictor":
        """Fit medians and next-day hazard forecaster strictly on chronological training split."""
        X_train, y_next_train, _ = self._prepare_causal_features_and_next_day_target(df_train, is_train=True)
        self.clf.fit(X_train, y_next_train)
        self.is_fitted = True
        return self

    def predict_hazard_proba(self, df_test: pd.DataFrame) -> np.ndarray:
        """Predict next-day (t+1) probability of elevated ambient air hazard (AQI_{t+1} > 200)."""
        if not self.is_fitted:
            raise RuntimeError("AmbientAQIPredictor must be fitted before calling predict_hazard_proba.")

        X_test, _, _ = self._prepare_causal_features_and_next_day_target(df_test, is_train=False)
        if self.clf.classes_.size > 1:
            return self.clf.predict_proba(X_test)[:, 1]
        return np.zeros(len(df_test), dtype=float)

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, float]:
        """Evaluate next-day (t+1) hazard forecasting metrics and persistence baseline on test split."""
        _, y_next_true, y_today = self._prepare_causal_features_and_next_day_target(df_test, is_train=False)
        probs = self.predict_hazard_proba(df_test)
        preds = (probs >= 0.5).astype(int)

        macro_f1 = float(f1_score(y_next_true, preds, average="macro", zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_next_true, preds))
        roc_auc = float(roc_auc_score(y_next_true, probs)) if len(np.unique(y_next_true)) > 1 else 0.5

        # Naive persistence baseline: predicting tomorrow's hazard = today's hazard (y_hat_{t+1} = y_t)
        persist_f1 = float(f1_score(y_next_true, y_today, average="macro", zero_division=0))
        persist_bal = float(balanced_accuracy_score(y_next_true, y_today))

        return {
            "aqi_hazard_macro_f1": round(macro_f1, 4),
            "aqi_hazard_balanced_accuracy": round(bal_acc, 4),
            "aqi_hazard_roc_auc": round(roc_auc, 4),
            "persistence_baseline_macro_f1": round(persist_f1, 4),
            "persistence_baseline_balanced_acc": round(persist_bal, 4),
        }


def load_real_or_synthetic_cpcb_aqi_data(
    csv_path: Path = Path("data/raw/india_aqi/city_day.csv"),
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load real CPCB India city_day.csv (chronological 80/20 split per city), or fallback to synthetic."""
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Keep rows with valid Date and at least PM2.5, PM10, or AQI
        df = df.dropna(subset=["City", "Date"]).copy()
        df = df[df[["PM2.5", "PM10", "AQI"]].notna().any(axis=1)].copy()
        df = df.sort_values(["City", "Date"]).reset_index(drop=True)

        train_parts, test_parts = [], []
        for _, grp in df.groupby("City", sort=False):
            if len(grp) < 20:
                continue
            split_idx = int(0.80 * len(grp))
            train_parts.append(grp.iloc[:split_idx])
            test_parts.append(grp.iloc[split_idx:])

        train_df = pd.concat(train_parts, ignore_index=True)
        test_df = pd.concat(test_parts, ignore_index=True)
        meta = {
            "aqi_data_source": "Real_CPCB_city_day.csv",
            "aqi_task": "Next_Day_Forecast_t_plus_1 (AQI_{t+1} > 200)",
            "n_cities": int(df["City"].nunique()),
            "n_train_days": int(len(train_df)),
            "n_test_days": int(len(test_df)),
        }
        return train_df, test_df, meta

    df_syn = generate_synthetic_cpcb_aqi_data(n_days=100, seed=42)
    train_df = df_syn[df_syn["DayIndex"] < 80].copy()
    test_df = df_syn[df_syn["DayIndex"] >= 80].copy()
    meta = {
        "aqi_data_source": "Synthetic_Fallback",
        "aqi_task": "Next_Day_Forecast_t_plus_1",
        "n_cities": int(df_syn["City"].nunique()),
        "n_train_days": int(len(train_df)),
        "n_test_days": int(len(test_df)),
    }
    return train_df, test_df, meta


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

