"""Unit tests for Phase 2 CPCB Ambient AQI Pipeline & Contextual Fusion Engine."""

import unittest
import numpy as np
import pandas as pd

from src.fusion.aqi_pipeline import (
    NAQITier,
    map_aqi_to_naqi_tier,
    compute_cpcb_pm25_subindex,
    compute_cpcb_pm10_subindex,
    estimate_aqi_from_pollutants,
    AmbientAQIPredictor,
    generate_synthetic_cpcb_aqi_data,
)
from src.fusion.contextual_fusion import (
    AlertLevel,
    ContextualRiskEngine,
    run_phase2_fusion_demo,
)


class TestPhase2AQIFusion(unittest.TestCase):
    def test_cpcb_naqi_tiers_and_subindices(self):
        """Verify CPCB NAQI 6-tier mapping and PM2.5/PM10 sub-index breakpoints."""
        self.assertEqual(map_aqi_to_naqi_tier(35.0), (NAQITier.GOOD, "Good"))
        self.assertEqual(map_aqi_to_naqi_tier(75.0), (NAQITier.SATISFACTORY, "Satisfactory"))
        self.assertEqual(map_aqi_to_naqi_tier(150.0), (NAQITier.MODERATE, "Moderate"))
        self.assertEqual(map_aqi_to_naqi_tier(250.0), (NAQITier.POOR, "Poor"))
        self.assertEqual(map_aqi_to_naqi_tier(350.0), (NAQITier.VERY_POOR, "Very Poor"))
        self.assertEqual(map_aqi_to_naqi_tier(450.0), (NAQITier.SEVERE, "Severe"))

        # PM2.5 = 30 ug/m3 -> sub-index = 50; PM2.5 = 90 ug/m3 -> sub-index = 200
        self.assertAlmostEqual(compute_cpcb_pm25_subindex(30.0), 50.0, places=1)
        self.assertAlmostEqual(compute_cpcb_pm25_subindex(90.0), 200.0, places=1)
        # PM10 = 100 ug/m3 -> sub-index = 100
        self.assertAlmostEqual(compute_cpcb_pm10_subindex(100.0), 100.0, places=1)

    def test_ambient_aqi_predictor_chronological_split(self):
        """Test AmbientAQIPredictor fits on chronological train split and predicts hazard on test."""
        df = generate_synthetic_cpcb_aqi_data(n_days=60, seed=42)
        train_df = df[df["DayIndex"] < 45]
        test_df = df[df["DayIndex"] >= 45]

        predictor = AmbientAQIPredictor().fit(train_df)
        probs = predictor.predict_hazard_proba(test_df)
        self.assertEqual(len(probs), len(test_df))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

        metrics = predictor.evaluate(test_df)
        self.assertGreaterEqual(metrics["aqi_hazard_roc_auc"], 0.75)

    def test_contextual_risk_engine_rules_and_persistence(self):
        """Test ContextualRiskEngine rule matrix across motion, persistence, and AQI tiers."""
        engine = ContextualRiskEngine()

        # 1. Baseline resting in clean air -> NORMAL
        d1 = engine.evaluate_window(
            strain_prob=0.15,
            recent_strain_probs=[0.12, 0.18, 0.14, 0.16, 0.15],
            high_motion_fraction=0.05,
            ambient_aqi=40.0,
        )
        self.assertEqual(d1.alert_level, int(AlertLevel.NORMAL))
        self.assertFalse(d1.persisted_strain)

        # 2. Isolated single-window spike (1 of 5 positive) in Satisfactory AQI -> Suppressed to NORMAL
        d2 = engine.evaluate_window(
            strain_prob=0.72,
            recent_strain_probs=[0.15, 0.20, 0.18, 0.22, 0.72],
            high_motion_fraction=0.05,
            ambient_aqi=80.0,
        )
        self.assertEqual(d2.alert_level, int(AlertLevel.NORMAL))
        self.assertFalse(d2.persisted_strain)

        # 3. Persisted strain during active exercise in Good AQI -> ADVISORY (not false critical alarm)
        d3 = engine.evaluate_window(
            strain_prob=0.80,
            recent_strain_probs=[0.65, 0.72, 0.75, 0.78, 0.80],
            high_motion_fraction=0.60,
            ambient_aqi=45.0,
        )
        self.assertEqual(d3.alert_level, int(AlertLevel.ADVISORY))
        self.assertTrue(d3.persisted_strain)
        self.assertEqual(d3.motion_context, "EXERTIONAL_ACTIVITY")

        # 4. Persisted NON-EXERTIONAL strain at rest in Poor AQI -> CRITICAL_ALERT
        d4 = engine.evaluate_window(
            strain_prob=0.82,
            recent_strain_probs=[0.68, 0.74, 0.79, 0.81, 0.82],
            high_motion_fraction=0.04,
            ambient_aqi=275.0,
            ambient_temp_c=40.0,
        )
        self.assertEqual(d4.alert_level, int(AlertLevel.CRITICAL_ALERT))
        self.assertTrue(d4.persisted_strain)
        self.assertEqual(d4.motion_context, "NON_EXERTIONAL_REST_OR_LIGHT")
        self.assertIn("PROXY NOTICE", d4.proxy_disclaimer)


if __name__ == "__main__":
    unittest.main()
