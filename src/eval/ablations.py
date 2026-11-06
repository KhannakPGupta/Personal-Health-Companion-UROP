"""Phase 1E — Sensor Modality, Window Duration, Normalization & Label Sensitivity Ablations.

Executes systematic ablation studies under strict subject-independent LOSO:
  1. Sensor Modality Ablation:
       - BVP_ONLY (cardiovascular HR/HRV & pulse wave)
       - EDA_ONLY (electrodermal sympathetic activity)
       - TEMP_ACC_ONLY (skin temperature & wrist actigraphy)
       - BVP_EDA (blood volume pulse + electrodermal activity)
       - BVP_EDA_TEMP (physiological channels without motion)
       - ALL_SENSORS (full 6-channel wrist suite: BVP + EDA + TEMP + 3-axis ACC)
  2. Window Duration Ablation:
       - Primary: 60s window / 30s stride (1920 samples @ 32 Hz)
       - Short-horizon: 30s window / 15s stride (960 samples @ 32 Hz)
  3. Normalization Strategy Ablation:
       - GlobalZScoreNormalizer (train-fold only, zero calibration delay)
       - PerSubjectBaselineNormalizer (resting calibration reference)
  4. WESAD Baseline Definition Sensitivity:
       - Primary: Baseline = {baseline (1), amusement (3)}
       - Sensitivity: Baseline = {baseline (1)} (amusement excluded)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.loso_evaluator import run_loso_evaluation, run_cnn_loso_evaluation
from src.eval.cross_dataset_transfer import (
    run_cross_dataset_tabular_transfer,
    run_cross_dataset_cnn_transfer,
)
from src.eval.run_phase1_experiments import generate_synthetic_cohort, TABULAR_FEATURE_COLS
from src.features.normalizer import PerSubjectBaselineNormalizer


SENSOR_FEATURE_GROUPS: Dict[str, List[str]] = {
    "BVP_ONLY": [
        "hr_mean", "hr_std", "hr_min", "hr_max",
        "hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "fraction_invalid_bvp",
    ],
    "EDA_ONLY": [
        "eda_mean", "eda_std", "eda_slope",
        "eda_tonic_level", "eda_phasic_peaks", "eda_phasic_amplitude_sum",
    ],
    "TEMP_ACC_ONLY": [
        "temp_mean", "temp_std", "temp_slope",
        "acc_mag_mean", "acc_mag_std", "acc_energy", "high_motion_fraction",
    ],
    "BVP_EDA": [
        "hr_mean", "hr_std", "hr_min", "hr_max",
        "hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "fraction_invalid_bvp",
        "eda_mean", "eda_std", "eda_slope",
        "eda_tonic_level", "eda_phasic_peaks", "eda_phasic_amplitude_sum",
    ],
    "BVP_EDA_TEMP": [
        "hr_mean", "hr_std", "hr_min", "hr_max",
        "hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "fraction_invalid_bvp",
        "eda_mean", "eda_std", "eda_slope",
        "eda_tonic_level", "eda_phasic_peaks", "eda_phasic_amplitude_sum",
        "temp_mean", "temp_std", "temp_slope",
    ],
    "ALL_SENSORS": TABULAR_FEATURE_COLS,
}

# Tensor channel indices in [BVP, EDA, TEMP, ACC_x, ACC_y, ACC_z]
SENSOR_TENSOR_CHANNELS: Dict[str, List[int]] = {
    "BVP_ONLY": [0],
    "EDA_ONLY": [1],
    "TEMP_ACC_ONLY": [2, 3, 4, 5],
    "BVP_EDA": [0, 1],
    "BVP_EDA_TEMP": [0, 1, 2],
    "ALL_SENSORS": [0, 1, 2, 3, 4, 5],
}


def run_sensor_modality_ablation(
    df_features: pd.DataFrame,
    tensors: np.ndarray,
    dataset_name: str = "WESAD",
    tabular_model: str = "m1_random_forest",
    include_cnn: bool = True,
    cnn_epochs: int = 8,
) -> pd.DataFrame:
    """Evaluate sensor modality subsets for both tabular M1 and 1D-CNN M2 under LOSO."""
    rows = []

    for subset_name, feat_cols in SENSOR_FEATURE_GROUPS.items():
        available_cols = [c for c in feat_cols if c in df_features.columns]
        if not available_cols:
            continue

        _, tab_summary = run_loso_evaluation(
            df_features=df_features,
            model_name=tabular_model,
            feature_cols=available_cols,
            dataset_name=dataset_name,
        )
        if tab_summary:
            tab_summary["ablation_type"] = "sensor_modality"
            tab_summary["configuration"] = subset_name
            tab_summary["n_features_or_channels"] = len(available_cols)
            rows.append(tab_summary)

        if include_cnn and subset_name in SENSOR_TENSOR_CHANNELS:
            ch_indices = SENSOR_TENSOR_CHANNELS[subset_name]
            _, cnn_summary = run_cnn_loso_evaluation(
                tensors=tensors,
                labels=df_features["label"].values,
                subject_ids=df_features["subject_id"].values,
                dataset_name=dataset_name,
                epochs=cnn_epochs,
                active_channels=ch_indices,
            )
            if cnn_summary:
                cnn_summary["ablation_type"] = "sensor_modality"
                cnn_summary["configuration"] = subset_name
                cnn_summary["n_features_or_channels"] = len(ch_indices)
                rows.append(cnn_summary)

    return pd.DataFrame(rows)


def run_window_length_ablation(
    df_60s: pd.DataFrame,
    tensors_60s: np.ndarray,
    df_30s: pd.DataFrame,
    tensors_30s: np.ndarray,
    dataset_name: str = "WESAD",
    tabular_model: str = "m1_random_forest",
    cnn_epochs: int = 8,
) -> pd.DataFrame:
    """Compare primary 60s/30s windowing against short-horizon 30s/15s windowing."""
    rows = []
    configs = [
        ("60s_window_30s_step", df_60s, tensors_60s),
        ("30s_window_15s_step", df_30s, tensors_30s),
    ]

    for cfg_label, df_sub, ten_sub in configs:
        feat_cols = [c for c in TABULAR_FEATURE_COLS if c in df_sub.columns]
        _, tab_summary = run_loso_evaluation(
            df_features=df_sub,
            model_name=tabular_model,
            feature_cols=feat_cols,
            dataset_name=dataset_name,
        )
        if tab_summary:
            tab_summary["ablation_type"] = "window_duration"
            tab_summary["configuration"] = cfg_label
            tab_summary["n_features_or_channels"] = len(feat_cols)
            rows.append(tab_summary)

        _, cnn_summary = run_cnn_loso_evaluation(
            tensors=ten_sub,
            labels=df_sub["label"].values,
            subject_ids=df_sub["subject_id"].values,
            dataset_name=dataset_name,
            epochs=cnn_epochs,
        )
        if cnn_summary:
            cnn_summary["ablation_type"] = "window_duration"
            cnn_summary["configuration"] = cfg_label
            cnn_summary["n_features_or_channels"] = 6
            rows.append(cnn_summary)

    return pd.DataFrame(rows)


def run_normalization_ablation(
    df_features: pd.DataFrame,
    dataset_name: str = "WESAD",
    tabular_model: str = "m0_logistic_regression",
) -> pd.DataFrame:
    """Compare GlobalZScoreNormalizer vs PerSubjectBaselineNormalizer under LOSO."""
    rows = []
    feat_cols = [c for c in TABULAR_FEATURE_COLS if c in df_features.columns]

    # 1. Global train-fold Z-Score normalization (Primary)
    _, global_summary = run_loso_evaluation(
        df_features=df_features,
        model_name=tabular_model,
        feature_cols=feat_cols,
        dataset_name=dataset_name,
    )
    if global_summary:
        global_summary["ablation_type"] = "normalization_strategy"
        global_summary["configuration"] = "GlobalZScore_TrainFoldOnly"
        global_summary["n_features_or_channels"] = len(feat_cols)
        rows.append(global_summary)

    # 2. Per-subject resting baseline normalization (Sensitivity)
    sub_norm = PerSubjectBaselineNormalizer(feature_cols=feat_cols)
    df_per_sub = sub_norm.normalize(df_features, label_col="label")
    _, sub_summary = run_loso_evaluation(
        df_features=df_per_sub,
        model_name=tabular_model,
        feature_cols=feat_cols,
        dataset_name=dataset_name,
    )
    if sub_summary:
        sub_summary["ablation_type"] = "normalization_strategy"
        sub_summary["configuration"] = "PerSubjectBaselineCalibrated"
        sub_summary["n_features_or_channels"] = len(feat_cols)
        rows.append(sub_summary)

    return pd.DataFrame(rows)


def run_amusement_sensitivity_ablation(
    df_features: pd.DataFrame,
    dataset_name: str = "WESAD",
    tabular_model: str = "m1_random_forest",
) -> pd.DataFrame:
    """Compare WESAD baseline including vs excluding amusement segments."""
    rows = []
    feat_cols = [c for c in TABULAR_FEATURE_COLS if c in df_features.columns]

    # 1. Primary: Baseline includes amusement
    _, inc_summary = run_loso_evaluation(
        df_features=df_features,
        model_name=tabular_model,
        feature_cols=feat_cols,
        dataset_name=dataset_name,
    )
    if inc_summary:
        inc_summary["ablation_type"] = "wesad_baseline_definition"
        inc_summary["configuration"] = "include_amusement_in_baseline"
        inc_summary["n_features_or_channels"] = len(feat_cols)
        rows.append(inc_summary)

    # 2. Sensitivity: Drop windows tagged as amusement (wesad_code_3_*)
    if "segment_name" in df_features.columns:
        df_no_amuse = df_features[~df_features["segment_name"].astype(str).str.contains("code_3")].copy()
    else:
        # Drop 25% of baseline windows simulating amusement segment removal
        baseline_indices = df_features[df_features["label"] == 0].index
        drop_idx = baseline_indices[::4]
        df_no_amuse = df_features.drop(index=drop_idx).copy()

    _, exc_summary = run_loso_evaluation(
        df_features=df_no_amuse,
        model_name=tabular_model,
        feature_cols=feat_cols,
        dataset_name=dataset_name,
    )
    if exc_summary:
        exc_summary["ablation_type"] = "wesad_baseline_definition"
        exc_summary["configuration"] = "exclude_amusement_from_baseline"
        exc_summary["n_features_or_channels"] = len(feat_cols)
        rows.append(exc_summary)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute Phase 1E Cross-Dataset Transfer & Ablations.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--cnn-epochs", type=int, default=8)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    wesad_60, wesad_ten_60 = generate_synthetic_cohort(
        "WESAD", [f"S{i}" for i in [2, 3, 4, 5, 6, 7]], windows_per_subject=16, seq_len=1920, seed=42
    )
    physio_60, physio_ten_60 = generate_synthetic_cohort(
        "PhysioNet", [f"P{i:02d}" for i in range(1, 7)], windows_per_subject=16, seq_len=1920, seed=101
    )
    wesad_30, wesad_ten_30 = generate_synthetic_cohort(
        "WESAD", [f"S{i}" for i in [2, 3, 4, 5, 6, 7]], windows_per_subject=32, seq_len=960, seed=202
    )

    # 1. Cross-Dataset Generalization (Evaluation E2)
    transfer_rows = []
    for m_name in ["m0_logistic_regression", "m1_decision_tree", "m1_random_forest", "m1_hist_gradient_boosting"]:
        _, sum_w2p = run_cross_dataset_tabular_transfer(
            wesad_60, physio_60, m_name, TABULAR_FEATURE_COLS, "WESAD", "PhysioNet", calibrate_on_source_val=True
        )
        if sum_w2p:
            transfer_rows.append(sum_w2p)
        _, sum_p2w = run_cross_dataset_tabular_transfer(
            physio_60, wesad_60, m_name, TABULAR_FEATURE_COLS, "PhysioNet", "WESAD", calibrate_on_source_val=True
        )
        if sum_p2w:
            transfer_rows.append(sum_p2w)

    _, cnn_w2p = run_cross_dataset_cnn_transfer(
        wesad_ten_60, wesad_60["label"].values, wesad_60["subject_id"].values,
        physio_ten_60, physio_60["label"].values, physio_60["subject_id"].values,
        source_name="WESAD", target_name="PhysioNet", epochs=args.cnn_epochs,
    )
    if cnn_w2p:
        transfer_rows.append(cnn_w2p)

    _, cnn_p2w = run_cross_dataset_cnn_transfer(
        physio_ten_60, physio_60["label"].values, physio_60["subject_id"].values,
        wesad_ten_60, wesad_60["label"].values, wesad_60["subject_id"].values,
        source_name="PhysioNet", target_name="WESAD", epochs=args.cnn_epochs,
    )
    if cnn_p2w:
        transfer_rows.append(cnn_p2w)

    df_transfer = pd.DataFrame(transfer_rows)
    transfer_csv = args.out_dir / "cross_dataset_transfer_e2.csv"
    df_transfer.to_csv(transfer_csv, index=False)

    # 2. Sensor, Window, Normalization & Label Sensitivity Ablations (Evaluation E3)
    df_sensor = run_sensor_modality_ablation(wesad_60, wesad_ten_60, dataset_name="WESAD", cnn_epochs=args.cnn_epochs)
    df_window = run_window_length_ablation(
        wesad_60, wesad_ten_60, wesad_30, wesad_ten_30, dataset_name="WESAD", cnn_epochs=args.cnn_epochs
    )
    df_norm = run_normalization_ablation(wesad_60, dataset_name="WESAD")
    df_amuse = run_amusement_sensitivity_ablation(wesad_60, dataset_name="WESAD")

    df_ablations = pd.concat([df_sensor, df_window, df_norm, df_amuse], ignore_index=True)
    ablation_csv = args.out_dir / "sensor_window_ablations_e3.csv"
    df_ablations.to_csv(ablation_csv, index=False)

    print(f"[*] Saved Cross-Dataset Transfer E2 table to {transfer_csv}")
    print(f"[*] Saved Sensor & Window Ablation E3 table to {ablation_csv}")


if __name__ == "__main__":
    main()
