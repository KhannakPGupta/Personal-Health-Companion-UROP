"""Phase 3 — Subject-Independent Federated LOSO Evaluator & Centralized Comparison.

Evaluates:
  1. Centralized LOSO M2 (upper-bound pooled training on train subjects)
  2. Federated FedAvg (mu = 0.0, privacy-preserving per-subject clients)
  3. Federated FedProx (mu = 0.05, proximal regularization against non-IID drift)
  4. Federated Partial Client Participation (C = 0.6)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.metrics import compute_classification_metrics, bootstrap_subject_confidence_intervals
from src.eval.loso_evaluator import (
    run_cnn_loso_evaluation,
    select_optimal_threshold_on_val,
    standardize_subject_tensors_unsupervised,
)
from src.eval.run_phase1_experiments import generate_synthetic_cohort
from src.federated.fedavg import EdgeClientData, run_federated_training
from tests.test_leakage_assertion import assert_no_subject_leakage


def run_federated_loso_evaluation(
    tensors: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    dataset_name: str = "WESAD",
    algorithm_name: str = "FedAvg",
    rounds: int = 6,
    local_epochs: int = 2,
    client_fraction: float = 1.0,
    fedprox_mu: float = 0.0,
    batch_size: int = 16,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run Leave-One-Subject-Out where remaining N-1 subjects train a global model via FedAvg/FedProx."""
    subject_ids_arr = np.asarray(subject_ids)
    unique_subjects = np.unique(subject_ids_arr)

    X_all = np.asarray(tensors, dtype=np.float32)
    if X_all.ndim == 3 and X_all.shape[1] > X_all.shape[2]:
        X_all = np.transpose(X_all, (0, 2, 1))
    X_all = standardize_subject_tensors_unsupervised(X_all, subject_ids_arr)
    y_all = np.asarray(labels, dtype=np.float32)

    per_subject_rows = []
    last_round_logs: List[Dict[str, float]] = []

    for test_sub in unique_subjects:
        train_mask = subject_ids_arr != test_sub
        test_mask = subject_ids_arr == test_sub

        train_subs = np.unique(subject_ids_arr[train_mask])
        test_subs = [str(test_sub)]
        assert_no_subject_leakage(train_subs, test_subs)

        y_test = y_all[test_mask].astype(int)
        if len(np.unique(y_test)) < 2 or len(train_subs) < 2:
            continue

        # Compute channel-wise standardization across training clients only (via sufficient stats)
        X_train_all = X_all[train_mask]
        ch_mean = X_train_all.mean(axis=(0, 2), keepdims=True)
        ch_std = X_train_all.std(axis=(0, 2), keepdims=True) + 1e-8

        # Build isolated EdgeClientData for each training subject
        clients: List[EdgeClientData] = []
        for c_sub in train_subs:
            c_mask = subject_ids_arr == c_sub
            X_c = torch.tensor((X_all[c_mask] - ch_mean) / ch_std, dtype=torch.float32)
            y_c = torch.tensor(y_all[c_mask], dtype=torch.float32)
            clients.append(EdgeClientData(client_id=str(c_sub), X=X_c, y=y_c))

        global_model, round_logs = run_federated_training(
            clients=clients,
            test_subjects=test_subs,
            in_channels=X_all.shape[1],
            rounds=rounds,
            local_epochs=local_epochs,
            client_fraction=client_fraction,
            batch_size=batch_size,
            fedprox_mu=fedprox_mu,
            random_seed=random_seed,
        )
        last_round_logs = round_logs

        global_model.eval()
        X_te_t = torch.tensor((X_all[test_mask] - ch_mean) / ch_std, dtype=torch.float32)
        with torch.no_grad():
            logits = global_model(X_te_t).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        if np.ndim(probs) == 0:
            probs = np.array([float(probs)])

        metrics = compute_classification_metrics(y_test, probs)
        metrics["subject_id"] = str(test_sub)
        metrics["dataset"] = dataset_name
        metrics["algorithm"] = algorithm_name
        metrics["n_windows"] = len(y_test)
        per_subject_rows.append(metrics)

    res_df = pd.DataFrame(per_subject_rows)
    if res_df.empty:
        return res_df, {}

    comm_f32_kb = last_round_logs[-1]["cumulative_comm_float32_kb"] if last_round_logs else 0.0
    comm_int8_kb = last_round_logs[-1]["cumulative_comm_int8_kb"] if last_round_logs else 0.0

    summary = {
        "dataset": dataset_name,
        "algorithm": algorithm_name,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "client_fraction": client_fraction,
        "fedprox_mu": fedprox_mu,
        "macro_f1_mean": round(float(res_df["macro_f1"].mean()), 4),
        "macro_f1_std": round(float(res_df["macro_f1"].std(ddof=0)), 4),
        "macro_f1_worst": round(float(res_df["macro_f1"].min()), 4),
        "strain_f1_mean": round(float(res_df["strain_f1"].mean()), 4),
        "balanced_acc_mean": round(float(res_df["balanced_accuracy"].mean()), 4),
        "roc_auc_mean": round(float(res_df["roc_auc"].mean()), 4),
        "ece_mean": round(float(res_df["ece"].mean()), 4),
        "false_alerts_persisted_mean": round(float(res_df["false_alerts_persisted_per_hr"].mean()), 2),
        "total_comm_float32_kb": comm_f32_kb,
        "total_comm_int8_kb": comm_int8_kb,
    }
    _, ci_low, ci_high = bootstrap_subject_confidence_intervals(per_subject_rows, metric_name="macro_f1")
    summary["macro_f1_95ci_lower"] = round(ci_low, 4)
    summary["macro_f1_95ci_upper"] = round(ci_high, 4)

    return res_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 Federated Edge Learning Simulation.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--local-epochs", type=int, default=2)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    feat_60_path = Path("data/labels/real_physionet_features_60s.csv")
    ten_60_path = Path("data/cache/real_physionet_tensors_60s.npz")
    if feat_60_path.exists() and ten_60_path.exists():
        df_all = pd.read_csv(feat_60_path)
        ten_all = np.load(ten_60_path)["tensors"]
        mask = (df_all["session"] == "STRESS").values
        df_stress = df_all[mask].reset_index(drop=True)
        ten_stress = ten_all[mask]
        selected_subs = sorted(df_stress["subject_id"].unique())[:12]
        sub_mask = df_stress["subject_id"].isin(selected_subs).values
        df_wesad = df_stress[sub_mask].reset_index(drop=True)
        ten_wesad = ten_stress[sub_mask]
        ds_name = "PhysioNet_STRESS"
    else:
        df_wesad, ten_wesad = generate_synthetic_cohort(
            "WESAD", [f"S{i}" for i in [2, 3, 4, 5, 6, 7]], windows_per_subject=16, seq_len=512, seed=42
        )
        ds_name = "WESAD"

    labels = df_wesad["label"].values
    subjects = df_wesad["subject_id"].values

    rows = []

    # 1. Centralized LOSO baseline
    _, cent_sum = run_cnn_loso_evaluation(
        tensors=ten_wesad,
        labels=labels,
        subject_ids=subjects,
        dataset_name=ds_name,
        epochs=args.rounds * args.local_epochs,
    )
    if cent_sum:
        rows.append({
            "dataset": ds_name,
            "algorithm": "Centralized_LOSO_M2",
            "rounds": 1,
            "local_epochs": args.rounds * args.local_epochs,
            "client_fraction": 1.0,
            "fedprox_mu": 0.0,
            "macro_f1_mean": cent_sum["macro_f1_mean"],
            "macro_f1_std": cent_sum["macro_f1_std"],
            "macro_f1_worst": cent_sum["macro_f1_worst"],
            "strain_f1_mean": cent_sum["strain_f1_mean"],
            "balanced_acc_mean": cent_sum["balanced_acc_mean"],
            "roc_auc_mean": cent_sum["roc_auc_mean"],
            "ece_mean": cent_sum["ece_mean"],
            "false_alerts_persisted_mean": cent_sum["false_alerts_persisted_mean"],
            "total_comm_float32_kb": 0.0,
            "total_comm_int8_kb": 0.0,
            "macro_f1_95ci_lower": cent_sum["macro_f1_95ci_lower"],
            "macro_f1_95ci_upper": cent_sum["macro_f1_95ci_upper"],
        })

    # 2. FedAvg (Full Client Participation, mu = 0.0)
    _, fedavg_sum = run_federated_loso_evaluation(
        tensors=ten_wesad,
        labels=labels,
        subject_ids=subjects,
        dataset_name=ds_name,
        algorithm_name="FedAvg_FullParticipation",
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        client_fraction=1.0,
        fedprox_mu=0.0,
    )
    if fedavg_sum:
        rows.append(fedavg_sum)

    # 3. FedProx (Full Client Participation, mu = 0.05 for non-IID physiological drift)
    _, fedprox_sum = run_federated_loso_evaluation(
        tensors=ten_wesad,
        labels=labels,
        subject_ids=subjects,
        dataset_name=ds_name,
        algorithm_name="FedProx_mu0.05",
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        client_fraction=1.0,
        fedprox_mu=0.05,
    )
    if fedprox_sum:
        rows.append(fedprox_sum)

    # 4. FedAvg Partial Participation (60% clients per round)
    _, fedpart_sum = run_federated_loso_evaluation(
        tensors=ten_wesad,
        labels=labels,
        subject_ids=subjects,
        dataset_name=ds_name,
        algorithm_name="FedAvg_PartialClients_0.6",
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        client_fraction=0.6,
        fedprox_mu=0.0,
    )
    if fedpart_sum:
        rows.append(fedpart_sum)

    df_out = pd.DataFrame(rows)
    out_csv = args.out_dir / "phase3_federated_benchmark.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"[*] Saved Phase 3 Federated Learning benchmark table to {out_csv}")


if __name__ == "__main__":
    main()

