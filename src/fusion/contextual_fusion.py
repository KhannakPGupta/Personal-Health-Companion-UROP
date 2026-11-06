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
        if ambient_temp_c is not None and ambient_temp_c >= 38.0:
            # Boost environmental hazard score under high ambient heat index proxy
            heat_boost = min(0.25, (ambient_temp_c - 38.0) * 0.04)
            env_norm = float(np.clip(env_norm + heat_boost, 0.0, 1.0))

        is_exertional = high_motion_fraction >= self.motion_exertion_threshold
        motion_context = "EXERTIONAL_ACTIVITY" if is_exertional else "NON_EXERTIONAL_REST_OR_LIGHT"

        composite_score = round(self.physio_weight * strain_prob + self.env_weight * env_norm, 4)

        # Contextual Rule Matrix
        if not persisted and strain_prob < self.strain_threshold:
            if naqi_tier >= NAQITier.VERY_POOR:
                level = AlertLevel.ADVISORY
                rationale = (
                    f"Physiological strain is within baseline (P={strain_prob:.2f}), but ambient air quality "
                    f"is {naqi_cat} (AQI={ambient_aqi:.0f}). Recommend limiting prolonged outdoor exposure."
                )
            else:
                level = AlertLevel.NORMAL
                rationale = (
                    f"Physiological signals at baseline (P={strain_prob:.2f}) and ambient AQI is {naqi_cat} "
                    f"({ambient_aqi:.0f}). No intervention required."
                )

        elif not persisted and strain_prob >= self.strain_threshold:
            # Isolated single-window spike suppressed by 3-of-5 persistence gate
            if naqi_tier >= NAQITier.POOR:
                level = AlertLevel.ADVISORY
                rationale = (
                    f"Transient physiological elevation (P={strain_prob:.2f}, persistence 3-of-5 not yet met) "
                    f"under {naqi_cat} air quality (AQI={ambient_aqi:.0f}). Monitoring closely."
                )
            else:
                level = AlertLevel.NORMAL
                rationale = (
                    f"Transient single-window strain spike (P={strain_prob:.2f}) suppressed by 3-of-5 "
                    f"persistence rule under {naqi_cat} AQI ({ambient_aqi:.0f})."
                )

        else:
            # Persisted physiological strain (>= 3 of last 5 windows positive)
            if is_exertional:
                if naqi_tier >= NAQITier.POOR:
                    level = AlertLevel.CRITICAL_ALERT if naqi_tier >= NAQITier.VERY_POOR else AlertLevel.WARNING
                    rationale = (
                        f"Sustained exertional strain (P={strain_prob:.2f}, high motion={high_motion_fraction:.2f}) "
                        f"coinciding with {naqi_cat} ambient pollution (AQI={ambient_aqi:.0f}). High inhaled dose risk; "
                        f"reduce outdoor physical exertion immediately."
                    )
                else:
                    level = AlertLevel.ADVISORY
                    rationale = (
                        f"Sustained physiological strain detected during active motion "
                        f"(P={strain_prob:.2f}, motion={high_motion_fraction:.2f}) under {naqi_cat} AQI. "
                        f"Consistent with structured exercise; stay hydrated."
                    )
            else:
                # Non-exertional persisted strain at rest / low activity
                if naqi_tier >= NAQITier.POOR or (strain_prob >= self.high_strain_threshold and naqi_tier >= NAQITier.MODERATE):
                    level = AlertLevel.CRITICAL_ALERT
                    rationale = (
                        f"Sustained NON-EXERTIONAL physiological strain at rest (P={strain_prob:.2f}, "
                        f"low motion={high_motion_fraction:.2f}) combined with {naqi_cat} ambient environment "
                        f"(AQI={ambient_aqi:.0f}). Immediate rest in a clean/cooled indoor space advised."
                    )
                elif strain_prob >= self.high_strain_threshold:
                    level = AlertLevel.WARNING
                    rationale = (
                        f"Sustained high non-exertional physiological strain (P={strain_prob:.2f}) despite "
                        f"{naqi_cat} ambient AQI ({ambient_aqi:.0f}). Check hydration and stress recovery."
                    )
                else:
                    level = AlertLevel.ADVISORY
                    rationale = (
                        f"Moderate persisted non-exertional strain (P={strain_prob:.2f}) under "
                        f"{naqi_cat} AQI ({ambient_aqi:.0f}). Recommend brief recovery break."
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
    """Run Phase 2 CPCB AQI hazard classifier + Contextual Risk Fusion matrix benchmark."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train & evaluate CPCB AQI Hazard Predictor on chronological split (first 80% train, last 20% test)
    df_aqi = generate_synthetic_cpcb_aqi_data(n_days=100, seed=42)
    train_df = df_aqi[df_aqi["DayIndex"] < 80].copy()
    test_df = df_aqi[df_aqi["DayIndex"] >= 80].copy()

    predictor = AmbientAQIPredictor().fit(train_df)
    aqi_metrics = predictor.evaluate(test_df)

    # 2. Evaluate canonical physiological + environmental scenarios through ContextualRiskEngine
    engine = ContextualRiskEngine()
    scenarios = [
        {
            "scenario_id": "S1_Resting_CleanAir",
            "strain_prob": 0.18,
            "history": [0.15, 0.20, 0.12, 0.19, 0.18],
            "motion": 0.04,
            "aqi": 42.0,
            "temp_c": 26.0,
        },
        {
            "scenario_id": "S2_Transient_Spike_Suppressed",
            "strain_prob": 0.68,
            "history": [0.15, 0.22, 0.18, 0.25, 0.68],
            "motion": 0.08,
            "aqi": 85.0,
            "temp_c": 29.0,
        },
        {
            "scenario_id": "S3_Exercise_GoodAir",
            "strain_prob": 0.82,
            "history": [0.65, 0.74, 0.79, 0.81, 0.82],
            "motion": 0.62,
            "aqi": 64.0,
            "temp_c": 28.0,
        },
        {
            "scenario_id": "S4_Exercise_SeverePollution",
            "strain_prob": 0.84,
            "history": [0.60, 0.72, 0.78, 0.80, 0.84],
            "motion": 0.58,
            "aqi": 345.0,
            "temp_c": 34.0,
        },
        {
            "scenario_id": "S5_NonExertional_Strain_ModerateAir",
            "strain_prob": 0.62,
            "history": [0.55, 0.58, 0.52, 0.60, 0.62],
            "motion": 0.05,
            "aqi": 140.0,
            "temp_c": 31.0,
        },
        {
            "scenario_id": "S6_NonExertional_HighStrain_PoorAir_Heat",
            "strain_prob": 0.86,
            "history": [0.71, 0.78, 0.82, 0.85, 0.86],
            "motion": 0.06,
            "aqi": 265.0,
            "temp_c": 40.5,
        },
    ]

    decision_rows = []
    for sc in scenarios:
        dec = engine.evaluate_window(
            strain_prob=sc["strain_prob"],
            recent_strain_probs=sc["history"],
            high_motion_fraction=sc["motion"],
            ambient_aqi=sc["aqi"],
            ambient_temp_c=sc["temp_c"],
        )
        row = {"scenario_id": sc["scenario_id"], **dec.to_dict(), **aqi_metrics}
        decision_rows.append(row)

    df_out = pd.DataFrame(decision_rows)
    out_csv = out_dir / "phase2_contextual_fusion_demo.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"[*] Saved Phase 2 Contextual Fusion table to {out_csv}")
    return df_out


if __name__ == "__main__":
    run_phase2_fusion_demo()
