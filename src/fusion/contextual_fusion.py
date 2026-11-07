"""Phase 2 — Contextual Physiological-Environmental Rule-Matrix & Fusion Engine.

Fuses:
  1. On-device calibrated physiological strain probability P(strain) in [0, 1].
  2. Temporal 3-of-5 persistence state (suppresses isolated single-window false alarms).
  3. Wrist motion context (high_motion_fraction) to distinguish active physical exercise
     from non-exertional sympathetic/cardiovascular strain at rest.
  4. Ambient CPCB Air Quality Index (NAQI 6-tier hazard level) and optional ambient heat proxy.

Outputs a deterministic, auditable ContextualAlertDecision with explicit research proxy disclaimer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from enum import IntEnum
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fusion.aqi_pipeline import (
    NAQITier,
    map_aqi_to_naqi_tier,
    AmbientAQIPredictor,
    generate_synthetic_cpcb_aqi_data,
    load_real_or_synthetic_cpcb_aqi_data,
)


class AlertLevel(IntEnum):
    """Four-stage actionable companion alert hierarchy."""
    NORMAL = 0
    ADVISORY = 1
    WARNING = 2
    CRITICAL_ALERT = 3


ALERT_NAMES: Dict[AlertLevel, str] = {
    AlertLevel.NORMAL: "NORMAL",
    AlertLevel.ADVISORY: "ADVISORY",
    AlertLevel.WARNING: "WARNING",
    AlertLevel.CRITICAL_ALERT: "CRITICAL_ALERT",
}


@dataclass
class ContextualAlertDecision:
    """Structured output of the Phase 2 Contextual Risk Fusion Engine."""
    alert_level: int
    alert_name: str
    composite_risk_score: float
    strain_probability: float
    persisted_strain: bool
    motion_context: str
    ambient_aqi: float
    naqi_tier: int
    naqi_category: str
    rationale: str
    proxy_disclaimer: str = (
        "PROXY NOTICE: Experimental wrist-signal strain + ambient AQI fusion; not a clinical diagnosis."
    )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class ContextualRiskEngine:
    """Rule-Matrix & Weighted Probabilistic Fusion Engine for Personal Health Companion."""

    def __init__(
        self,
        strain_threshold: float = 0.50,
        high_strain_threshold: float = 0.75,
        motion_exertion_threshold: float = 0.35,
        persistence_window: int = 5,
        persistence_min_positives: int = 3,
        physio_weight: float = 0.60,
        env_weight: float = 0.40,
    ):
        self.strain_threshold = strain_threshold
        self.high_strain_threshold = high_strain_threshold
        self.motion_exertion_threshold = motion_exertion_threshold
        self.persistence_window = persistence_window
        self.persistence_min_positives = persistence_min_positives
        self.physio_weight = physio_weight
        self.env_weight = env_weight

    def check_persistence(self, recent_strain_probs: Sequence[float]) -> bool:
        """Return True if at least 3 of the last 5 windows exceed strain_threshold."""
        if not recent_strain_probs:
            return False
        tail = list(recent_strain_probs)[-self.persistence_window :]
        pos_count = sum(1 for p in tail if p >= self.strain_threshold)
        return pos_count >= self.persistence_min_positives

    def evaluate_window(
        self,
        strain_prob: float,
        recent_strain_probs: Sequence[float],
        high_motion_fraction: float,
        ambient_aqi: float,
        ambient_temp_c: Optional[float] = None,
    ) -> ContextualAlertDecision:
        """Fuse wearable physiological strain, persistence, motion context, and ambient AQI."""
        strain_prob = float(np.clip(strain_prob, 0.0, 1.0))
        high_motion_fraction = float(np.clip(high_motion_fraction, 0.0, 1.0))
        persisted = self.check_persistence(recent_strain_probs)

        naqi_tier, naqi_cat = map_aqi_to_naqi_tier(ambient_aqi)
        env_norm = float(np.clip(ambient_aqi / 400.0, 0.0, 1.0))
        is_high_heat = ambient_temp_c is not None and ambient_temp_c >= 38.0
        temp_note = f", Temp={ambient_temp_c:.1f}C" if ambient_temp_c is not None else ""
        if is_high_heat:
            # Boost environmental hazard score under high ambient heat proxy (>= 38C)
            heat_boost = min(0.25, (ambient_temp_c - 38.0) * 0.04)
            env_norm = float(np.clip(env_norm + heat_boost, 0.0, 1.0))

        is_exertional = high_motion_fraction >= self.motion_exertion_threshold
        motion_context = "EXERTIONAL_ACTIVITY" if is_exertional else "NON_EXERTIONAL_REST_OR_LIGHT"

        composite_score = round(self.physio_weight * strain_prob + self.env_weight * env_norm, 4)

        # Contextual Rule Matrix (incorporates both NAQI tier and ambient heat threshold >= 38.0C)
        if not persisted and strain_prob < self.strain_threshold:
            if naqi_tier >= NAQITier.VERY_POOR or is_high_heat:
                level = AlertLevel.ADVISORY
                rationale = (
                    f"Physiological strain is within baseline (P={strain_prob:.2f}), but ambient environment "
                    f"is hazardous ({naqi_cat}, AQI={ambient_aqi:.0f}{temp_note}). Limit prolonged outdoor exposure."
                )
            else:
                level = AlertLevel.NORMAL
                rationale = (
                    f"Physiological signals at baseline (P={strain_prob:.2f}) and ambient AQI is {naqi_cat} "
                    f"({ambient_aqi:.0f}{temp_note}). No intervention required."
                )

        elif not persisted and strain_prob >= self.strain_threshold:
            # Isolated single-window spike suppressed by 3-of-5 persistence gate
            if naqi_tier >= NAQITier.POOR or is_high_heat:
                level = AlertLevel.ADVISORY
                rationale = (
                    f"Transient physiological elevation (P={strain_prob:.2f}, persistence 3-of-5 not yet met) "
                    f"under {naqi_cat} air quality (AQI={ambient_aqi:.0f}{temp_note}). Monitoring closely."
                )
            else:
                level = AlertLevel.NORMAL
                rationale = (
                    f"Transient single-window strain spike (P={strain_prob:.2f}) suppressed by 3-of-5 "
                    f"persistence rule under {naqi_cat} AQI ({ambient_aqi:.0f}{temp_note})."
                )

        else:
            # Persisted physiological strain (>= 3 of last 5 windows positive)
            if is_exertional:
                if naqi_tier >= NAQITier.POOR or is_high_heat:
                    level = (
                        AlertLevel.CRITICAL_ALERT
                        if (naqi_tier >= NAQITier.VERY_POOR or (naqi_tier >= NAQITier.POOR and is_high_heat))
                        else AlertLevel.WARNING
                    )
                    rationale = (
                        f"Sustained exertional strain (P={strain_prob:.2f}, high motion={high_motion_fraction:.2f}) "
                        f"coinciding with {naqi_cat} ambient conditions (AQI={ambient_aqi:.0f}{temp_note}). "
                        f"Reduce outdoor physical exertion immediately."
                    )
                else:
                    level = AlertLevel.ADVISORY
                    rationale = (
                        f"Sustained physiological strain detected during active motion "
                        f"(P={strain_prob:.2f}, motion={high_motion_fraction:.2f}) under {naqi_cat} AQI{temp_note}. "
                        f"Consistent with structured exercise; stay hydrated."
                    )
            else:
                # Non-exertional persisted strain at rest / low activity
                if (
                    naqi_tier >= NAQITier.POOR
                    or is_high_heat
                    or (strain_prob >= self.high_strain_threshold and naqi_tier >= NAQITier.MODERATE)
                ):
                    level = AlertLevel.CRITICAL_ALERT
                    rationale = (
                        f"Sustained NON-EXERTIONAL physiological strain at rest (P={strain_prob:.2f}, "
                        f"low motion={high_motion_fraction:.2f}) combined with {naqi_cat} ambient environment "
                        f"(AQI={ambient_aqi:.0f}{temp_note}). Immediate rest in a clean/cooled indoor space advised."
                    )
                elif strain_prob >= self.high_strain_threshold:
                    level = AlertLevel.WARNING
                    rationale = (
                        f"Sustained high non-exertional physiological strain (P={strain_prob:.2f}) despite "
                        f"{naqi_cat} ambient AQI ({ambient_aqi:.0f}{temp_note}). Check hydration and stress recovery."
                    )
                else:
                    level = AlertLevel.ADVISORY
                    rationale = (
                        f"Moderate persisted non-exertional strain (P={strain_prob:.2f}) under "
                        f"{naqi_cat} AQI ({ambient_aqi:.0f}{temp_note}). Recommend brief recovery break."
                    )

        return ContextualAlertDecision(
            alert_level=int(level),
            alert_name=ALERT_NAMES[level],
            composite_risk_score=composite_score,
            strain_probability=round(strain_prob, 4),
            persisted_strain=persisted,
            motion_context=motion_context,
            ambient_aqi=round(float(ambient_aqi), 1),
            naqi_tier=int(naqi_tier),
            naqi_category=naqi_cat,
            rationale=rationale,
        )


def run_phase2_fusion_demo(out_dir: Path = Path("results/tables")) -> pd.DataFrame:
    """Run Phase 2 Next-Day CPCB AQI Forecaster (on real city_day.csv) + Contextual Risk Fusion verification."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train & evaluate Next-Day (t+1) CPCB AQI Hazard Forecaster on real city_day.csv (80% train / 20% test per city)
    train_df, test_df, aqi_meta = load_real_or_synthetic_cpcb_aqi_data()
    predictor = AmbientAQIPredictor().fit(train_df)
    aqi_metrics = {**aqi_meta, **predictor.evaluate(test_df)}

    # Extract real AQI readings from Delhi and Mumbai test splits if available
    delhi_test = test_df[test_df["City"] == "Delhi"]["AQI"].dropna()
    mumbai_test = test_df[test_df["City"] == "Mumbai"]["AQI"].dropna()
    real_delhi_poor_aqi = float(delhi_test[delhi_test > 250].iloc[0]) if len(delhi_test[delhi_test > 250]) > 0 else 285.0
    real_mumbai_good_aqi = float(mumbai_test[mumbai_test <= 90].iloc[0]) if len(mumbai_test[mumbai_test <= 90]) > 0 else 68.0

    # 2. Evaluate canonical rule-verification scenarios + real out-of-fold wearable window streams
    engine = ContextualRiskEngine()
    scenarios = [
        {
            "scenario_id": "S1_Resting_CleanAir",
            "source_type": "Rule_Verification_Scenario",
            "strain_prob": 0.18,
            "history": [0.15, 0.20, 0.12, 0.19, 0.18],
            "motion": 0.04,
            "aqi": 42.0,
            "temp_c": 26.0,
        },
        {
            "scenario_id": "S2_Transient_Spike_Suppressed",
            "source_type": "Rule_Verification_Scenario",
            "strain_prob": 0.68,
            "history": [0.15, 0.22, 0.18, 0.25, 0.68],
            "motion": 0.08,
            "aqi": 85.0,
            "temp_c": 29.0,
        },
        {
            "scenario_id": "S3_Exercise_GoodAir",
            "source_type": "Rule_Verification_Scenario",
            "strain_prob": 0.82,
            "history": [0.65, 0.74, 0.79, 0.81, 0.82],
            "motion": 0.62,
            "aqi": 64.0,
            "temp_c": 28.0,
        },
        {
            "scenario_id": "S4_Exercise_SeverePollution",
            "source_type": "Rule_Verification_Scenario",
            "strain_prob": 0.84,
            "history": [0.60, 0.72, 0.78, 0.80, 0.84],
            "motion": 0.58,
            "aqi": 345.0,
            "temp_c": 34.0,
        },
        {
            "scenario_id": "S5_NonExertional_Strain_ModerateAir",
            "source_type": "Rule_Verification_Scenario",
            "strain_prob": 0.62,
            "history": [0.55, 0.58, 0.52, 0.60, 0.62],
            "motion": 0.05,
            "aqi": 140.0,
            "temp_c": 31.0,
        },
        {
            "scenario_id": "S6_NonExertional_HighStrain_PoorAir_Heat",
            "source_type": "Rule_Verification_Scenario",
            "strain_prob": 0.86,
            "history": [0.71, 0.78, 0.82, 0.85, 0.86],
            "motion": 0.06,
            "aqi": 265.0,
            "temp_c": 40.5,
        },
    ]

    # Add real out-of-fold LOSO wearable predictions from PhysioNet_STRESS (S01) & PhysioNet_EXERCISE (S01)
    feat_path = Path("data/labels/real_physionet_features_60s.csv")
    if feat_path.exists():
        from sklearn.linear_model import LogisticRegression
        from src.features.normalizer import (
            GlobalZScoreNormalizer,
            UnsupervisedSubjectSessionCalibrator,
            augment_autonomic_features,
        )
        from src.eval.run_phase1_experiments import TABULAR_FEATURE_COLS

        df_real = pd.read_csv(feat_path)
        for sess_name, sub_target, city_aqi, temp_val, sc_id in [
            ("STRESS", "S01", real_delhi_poor_aqi, 33.0, "S7_Real_OOF_STRESS_S01_Delhi_PoorAQI"),
            ("AEROBIC", "S01", real_mumbai_good_aqi, 27.5, "S8_Real_OOF_AEROBIC_S01_Mumbai_SatisfactoryAQI"),
        ]:
            df_sess = df_real[df_real["session"] == sess_name].reset_index(drop=True)
            if sub_target in df_sess["subject_id"].values:
                df_aug = augment_autonomic_features(df_sess)
                extra = ["sympathovagal_ratio", "hr_dynamic_range", "eda_arousal_index", "eda_phasic_tonic_ratio", "motion_adjusted_hr"]
                fcols = list(dict.fromkeys([c for c in TABULAR_FEATURE_COLS if c in df_aug.columns] + extra))
                df_cal = UnsupervisedSubjectSessionCalibrator(feature_cols=fcols).transform(df_aug)
                tr_m = df_cal["subject_id"] != sub_target
                te_m = df_cal["subject_id"] == sub_target
                norm = GlobalZScoreNormalizer(feature_cols=fcols).fit(df_cal[tr_m])
                X_tr = np.nan_to_num(norm.transform(df_cal[tr_m])[fcols].values, nan=0.0)
                y_tr = df_cal.loc[tr_m, "label"].values.astype(int)
                X_te = np.nan_to_num(norm.transform(df_cal[te_m])[fcols].values, nan=0.0)
                clf = LogisticRegression(C=0.25, class_weight="balanced", max_iter=1000, random_state=42).fit(X_tr, y_tr)
                oof_probs = clf.predict_proba(X_te)[:, 1]
                strain_idx = np.where(df_cal.loc[te_m, "label"].values == 1)[0]
                if len(strain_idx) >= 5:
                    idx5 = strain_idx[:5]
                    hist5 = [round(float(p), 4) for p in oof_probs[idx5]]
                    mot_val = float(df_sess.loc[te_m, "high_motion_fraction"].iloc[idx5[-1]])
                    scenarios.append({
                        "scenario_id": sc_id,
                        "source_type": "Real_OOF_LOSO_Wearable_Plus_Real_CPCB_AQI",
                        "strain_prob": hist5[-1],
                        "history": hist5,
                        "motion": round(mot_val, 4),
                        "aqi": round(city_aqi, 1),
                        "temp_c": temp_val,
                    })

    decision_rows = []
    for sc in scenarios:
        dec = engine.evaluate_window(
            strain_prob=sc["strain_prob"],
            recent_strain_probs=sc["history"],
            high_motion_fraction=sc["motion"],
            ambient_aqi=sc["aqi"],
            ambient_temp_c=sc["temp_c"],
        )
        row = {
            "scenario_id": sc["scenario_id"],
            "source_type": sc.get("source_type", "Rule_Verification_Scenario"),
            "ambient_temp_c": sc["temp_c"],
            **dec.to_dict(),
            **aqi_metrics,
        }
        decision_rows.append(row)

    df_out = pd.DataFrame(decision_rows)
    out_csv = out_dir / "phase2_contextual_fusion_demo.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"[*] Saved Phase 2 Contextual Fusion table to {out_csv}")
    return df_out


if __name__ == "__main__":
    run_phase2_fusion_demo()

