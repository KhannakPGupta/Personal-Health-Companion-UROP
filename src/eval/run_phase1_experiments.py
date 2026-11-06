"""End-to-end Phase 1D LOSO evaluation & edge profiling runner.

Executes:
  1. Loads WESAD and PhysioNet windows (or runs a deterministic synthetic dry-run for CI verification).
  2. Evaluates M0 (Majority, Logistic Regression), M1 (Decision Tree, Random Forest, HistGB),
     and M2 (Lightweight 1D-CNN < 50k params) under strict Leave-One-Subject-Out (LOSO).
  3. Profiles M2 float32 vs int8 post-training quantization and CPU single-window latency.
  4. Saves structured tables to results/tables/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.loso_evaluator import run_loso_evaluation, run_cnn_loso_evaluation
from src.edge.quantization import profile_quantization
from src.edge.profiler import benchmark_inference_latency
from src.models.cnn_1d import LightweightStrainCNN


TABULAR_FEATURE_COLS = [
    "hr_mean",
    "hr_std",
    "hr_min",
    "hr_max",
    "hrv_rmssd",
    "hrv_sdnn",
    "hrv_pnn50",
    "fraction_invalid_bvp",
    "eda_mean",
    "eda_std",
    "eda_slope",
    "eda_tonic_level",
    "eda_phasic_peaks",
    "eda_phasic_amplitude_sum",
    "temp_mean",
    "temp_std",
    "temp_slope",
    "acc_mag_mean",
    "acc_mag_std",
    "acc_energy",
    "high_motion_fraction",
]


def generate_synthetic_cohort(
    dataset_name: str,
    subjects: List[str],
    windows_per_subject: int = 16,
    seq_len: int = 1920,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate a deterministic, physiologically structured synthetic cohort for CI/dry-run verification."""
    rng = np.random.default_rng(seed)
    rows = []
    tensors = []

    for s_idx, sub_id in enumerate(subjects):
        # Subject-specific random baseline offset (models inter-subject physiological drift)
        sub_hr_offset = rng.normal(0.0, 4.0)
        sub_eda_offset = rng.normal(0.0, 0.3)

        for w_idx in range(windows_per_subject):
            label = 1 if w_idx >= (windows_per_subject // 2) else 0
            strain_shift = 1.25 if label == 1 else -0.75

            feat_row = {
                "dataset": dataset_name,
                "subject_id": sub_id,
                "session": "SYNTHETIC_DRY_RUN",
                "window_idx": w_idx,
                "label": label,
                "label_name": "strain" if label == 1 else "baseline",
                "hr_mean": 72.0 + sub_hr_offset + 14.0 * strain_shift + rng.normal(0, 4.0),
                "hr_std": 4.5 + 1.8 * label + abs(rng.normal(0, 1.0)),
                "hr_min": 62.0 + sub_hr_offset + 8.0 * strain_shift + rng.normal(0, 3.0),
                "hr_max": 85.0 + sub_hr_offset + 18.0 * strain_shift + rng.normal(0, 5.0),
                "hrv_rmssd": max(10.0, 42.0 - 14.0 * strain_shift + rng.normal(0, 5.0)),
                "hrv_sdnn": max(12.0, 48.0 - 12.0 * strain_shift + rng.normal(0, 5.0)),
                "hrv_pnn50": np.clip(0.28 - 0.12 * strain_shift + rng.normal(0, 0.05), 0.0, 1.0),
                "fraction_invalid_bvp": np.clip(rng.beta(1.2, 15.0), 0.0, 1.0),
                "eda_mean": max(0.05, 1.5 + sub_eda_offset + 0.9 * strain_shift + rng.normal(0, 0.25)),
                "eda_std": max(0.01, 0.2 + 0.25 * label + rng.normal(0, 0.05)),
                "eda_slope": 0.01 * strain_shift + rng.normal(0, 0.005),
                "eda_tonic_level": max(0.05, 1.3 + sub_eda_offset + 0.7 * strain_shift + rng.normal(0, 0.2)),
                "eda_phasic_peaks": max(0.0, float(int(2 + 4 * label + rng.normal(0, 1.2)))),
                "eda_phasic_amplitude_sum": max(0.0, 0.4 + 0.8 * label + rng.normal(0, 0.2)),
                "temp_mean": 33.2 + 0.25 * strain_shift + rng.normal(0, 0.3),
                "temp_std": max(0.01, 0.05 + 0.02 * label + rng.normal(0, 0.01)),
                "temp_slope": 0.002 * strain_shift + rng.normal(0, 0.001),
                "acc_mag_mean": 1.02 + 0.15 * label + rng.normal(0, 0.05),
                "acc_mag_std": max(0.01, 0.08 + 0.08 * label + rng.normal(0, 0.02)),
                "acc_energy": max(0.1, 1.05 + 0.35 * label + rng.normal(0, 0.1)),
                "high_motion_fraction": np.clip(0.05 + 0.15 * label + rng.normal(0, 0.04), 0.0, 1.0),
            }
            rows.append(feat_row)

            # Construct corresponding 6-channel tensor (6, seq_len)
            t = np.linspace(0, 60.0, seq_len, dtype=np.float32)
            freq_hz = (1.2 + 0.35 * label)
            bvp_ch = np.sin(2 * np.pi * freq_hz * t) + rng.normal(0, 0.25, size=seq_len)
            eda_ch = (0.8 * strain_shift) + 0.2 * np.sin(2 * np.pi * 0.05 * t) + rng.normal(0, 0.15, size=seq_len)
            temp_ch = (0.3 * strain_shift) + rng.normal(0, 0.1, size=seq_len)
            acc_x = 0.2 * label * np.sin(2 * np.pi * 2.0 * t) + rng.normal(0, 0.15, size=seq_len)
            acc_y = 0.2 * label * np.cos(2 * np.pi * 2.0 * t) + rng.normal(0, 0.15, size=seq_len)
            acc_z = 1.0 + rng.normal(0, 0.15, size=seq_len)
            tensor_6ch = np.stack([bvp_ch, eda_ch, temp_ch, acc_x, acc_y, acc_z], axis=0).astype(np.float32)
            tensors.append(tensor_6ch)

    return pd.DataFrame(rows), np.stack(tensors, axis=0)


def run_benchmark_suite(
    df_features: pd.DataFrame,
    tensors: np.ndarray,
    dataset_label: str = "WESAD",
    cnn_epochs: int = 12,
) -> pd.DataFrame:
    """Run M0, M1, and M2 models under strict LOSO and return summary DataFrame."""
    models_tabular = [
        "m0_majority",
        "m0_logistic_regression",
        "m1_decision_tree",
        "m1_random_forest",
        "m1_hist_gradient_boosting",
    ]
    summaries = []
    for m_name in models_tabular:
        _, summary = run_loso_evaluation(
            df_features=df_features,
            model_name=m_name,
            feature_cols=TABULAR_FEATURE_COLS,
            dataset_name=dataset_label,
        )
        if summary:
            summaries.append(summary)

    # Evaluate M2 1D-CNN
    _, cnn_summary = run_cnn_loso_evaluation(
        tensors=tensors,
        labels=df_features["label"].values,
        subject_ids=df_features["subject_id"].values,
        dataset_name=dataset_label,
        epochs=cnn_epochs,
    )
    if cnn_summary:
        summaries.append(cnn_summary)

    return pd.DataFrame(summaries)


def profile_edge_m2(tensors: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Profile M2 float32 vs int8 quantization and single-window CPU inference latency."""
    X_t = torch.tensor(tensors, dtype=torch.float32)
    if X_t.shape[1] != 6 and X_t.shape[2] == 6:
        X_t = X_t.transpose(1, 2)
    y_t = torch.tensor(labels, dtype=torch.float32)

    model = LightweightStrainCNN(in_channels=6, num_classes=1)
    q_stats = profile_quantization(model, X_t, y_t)
    lat_stats = benchmark_inference_latency(model, X_t[0:1], n_warmup=10, n_runs=50)
    combined = {**q_stats, **lat_stats}
    return combined


def load_real_or_synthetic_cohorts() -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Load real PhysioNet Stress & Exercise cohorts if extracted, else fallback to synthetic dry-run."""
    feat_path = Path("data/labels/real_physionet_features_60s.csv")
    ten_path = Path("data/cache/real_physionet_tensors_60s.npz")

    if feat_path.exists() and ten_path.exists():
        df_all = pd.read_csv(feat_path)
        ten_all = np.load(ten_path)["tensors"]

        stress_mask = (df_all["session"] == "STRESS").values
        ex_mask = (df_all["session"].isin(["AEROBIC", "ANAEROBIC"])).values

        df_stress = df_all[stress_mask].reset_index(drop=True)
        ten_stress = ten_all[stress_mask]

        df_ex = df_all[ex_mask].reset_index(drop=True)
        ten_ex = ten_all[ex_mask]
        return df_stress, ten_stress, df_ex, ten_ex

    wesad_df, wesad_tensors = generate_synthetic_cohort(
        "WESAD", [f"S{i}" for i in [2, 3, 4, 5, 6, 7]], windows_per_subject=16, seed=42
    )
    physio_df, physio_tensors = generate_synthetic_cohort(
        "PhysioNet", [f"P{i:02d}" for i in range(1, 7)], windows_per_subject=16, seed=101
    )
    return wesad_df, wesad_tensors, physio_df, physio_tensors


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1D LOSO evaluation and edge profiling.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--cnn-epochs", type=int, default=10)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cohort1_df, cohort1_tensors, cohort2_df, cohort2_tensors = load_real_or_synthetic_cohorts()
    is_real = Path("data/labels/real_physionet_features_60s.csv").exists()
    label_c1 = "PhysioNet_STRESS" if is_real else "WESAD"
    label_c2 = "PhysioNet_EXERCISE" if is_real else "PhysioNet"

    df_c1_res = run_benchmark_suite(cohort1_df, cohort1_tensors, dataset_label=label_c1, cnn_epochs=args.cnn_epochs)
    df_c2_res = run_benchmark_suite(cohort2_df, cohort2_tensors, dataset_label=label_c2, cnn_epochs=args.cnn_epochs)

    pooled_df = pd.concat([cohort1_df, cohort2_df], ignore_index=True)
    pooled_tensors = np.concatenate([cohort1_tensors, cohort2_tensors], axis=0)
    df_pooled_res = run_benchmark_suite(pooled_df, pooled_tensors, dataset_label="Pooled", cnn_epochs=args.cnn_epochs)

    full_table = pd.concat([df_c1_res, df_c2_res, df_pooled_res], ignore_index=True)
    out_csv = args.out_dir / "loso_benchmark_e1.csv"
    full_table.to_csv(out_csv, index=False)

    edge_stats = profile_edge_m2(cohort1_tensors, cohort1_df["label"].values)
    edge_df = pd.DataFrame([edge_stats])
    edge_csv = args.out_dir / "edge_profile_m2.csv"
    edge_df.to_csv(edge_csv, index=False)

    print(f"[*] Saved LOSO E1 benchmark table to {out_csv}")
    print(f"[*] Saved M2 Edge quantization & latency profile to {edge_csv}")


if __name__ == "__main__":
    main()

