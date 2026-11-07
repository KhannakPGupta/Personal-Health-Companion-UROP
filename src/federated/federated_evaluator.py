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
    clients_per_rd = int(last_round_logs[-1]["participating_clients"]) if last_round_logs else 0

    summary = {
        "dataset": dataset_name,
        "algorithm": algorithm_name,
        "n_subjects": int(len(res_df)),
        "clients_per_round": clients_per_rd,
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
        "false_alarm_episodes_mean": round(float(res_df["false_alarm_episodes_per_hr"].mean()), 2),
        "total_comm_float32_kb": comm_f32_kb,
        "total_comm_int8_kb": comm_int8_kb,
    }
    _, ci_low, ci_high = bootstrap_subject_confidence_intervals(per_subject_rows, metric_name="macro_f1")
    summary["macro_f1_95ci_lower"] = round(ci_low, 4)
    summary["macro_f1_95ci_upper"] = round(ci_high, 4)

    return res_df, summary


def run_federated_tabular_logreg_loso(
    df_features: pd.DataFrame,
    dataset_name: str = "PhysioNet_STRESS",
    algorithm_name: str = "FedAvg_Tabular_LogReg",
    rounds: int = 8,
    local_epochs: int = 5,
    client_fraction: float = 1.0,
    fedprox_mu: float = 0.0,
    lr: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run subject-isolated FedAvg/FedProx for a 26-feature Logistic Regression edge classifier."""
    from src.eval.run_phase1_experiments import TABULAR_FEATURE_COLS
    from src.features.normalizer import (
        GlobalZScoreNormalizer,
        UnsupervisedSubjectSessionCalibrator,
        augment_autonomic_features,
    )
    from src.federated.fedavg import compute_communication_cost_kb

    rng = np.random.default_rng(random_seed)
    df_aug = augment_autonomic_features(df_features)
    extra = [
        "sympathovagal_ratio",
        "hr_dynamic_range",
        "eda_arousal_index",
        "eda_phasic_tonic_ratio",
        "motion_adjusted_hr",
    ]
    feat_cols = list(dict.fromkeys([c for c in TABULAR_FEATURE_COLS if c in df_aug.columns] + extra))
    df_cal = UnsupervisedSubjectSessionCalibrator(feature_cols=feat_cols).transform(df_aug)

    subjects = df_cal["subject_id"].unique()
    per_subject_rows = []
    n_params = len(feat_cols) + 1  # 26 weights + 1 bias = 27 parameters
    m_select = max(1, int(np.ceil(client_fraction * (len(subjects) - 1))))

    for test_sub in subjects:
        tr_mask = df_cal["subject_id"] != test_sub
        te_mask = df_cal["subject_id"] == test_sub
        train_subs = df_cal.loc[tr_mask, "subject_id"].unique()
        assert_no_subject_leakage(train_subs, [test_sub])

        df_tr = df_cal[tr_mask]
        df_te = df_cal[te_mask]
        y_te = df_te["label"].values.astype(int)
        if len(np.unique(y_te)) < 2 or len(train_subs) < 2:
            continue

        norm = GlobalZScoreNormalizer(feature_cols=feat_cols).fit(df_tr)
        df_tr_norm = norm.transform(df_tr)
        df_te_norm = norm.transform(df_te)

        # Global class balance weight on train subjects
        y_tr_all = df_tr_norm["label"].values.astype(float)
        pos_w = float(np.sum(y_tr_all == 0) / max(1.0, np.sum(y_tr_all == 1)))

        # Initialize global linear weights w (D,) and bias b
        D = len(feat_cols)
        w_global = np.zeros(D, dtype=np.float64)
        b_global = 0.0

        # Prepare per-subject client arrays
        client_arrays = []
        for c_sub in train_subs:
            c_df = df_tr_norm[df_tr_norm["subject_id"] == c_sub]
            Xc = np.nan_to_num(c_df[feat_cols].values.astype(np.float64), nan=0.0)
            yc = c_df["label"].values.astype(np.float64)
            client_arrays.append((Xc, yc, len(yc)))

        for _ in range(rounds):
            sel_idx = rng.choice(len(client_arrays), size=min(m_select, len(client_arrays)), replace=False)
            w_updates, b_updates, counts = [], [], []
            for idx_c in sel_idx:
                Xc, yc, nc = client_arrays[idx_c]
                w_loc = w_global.copy()
                b_loc = b_global
                sample_w = np.where(yc == 1.0, pos_w, 1.0)
                sample_w = sample_w / np.mean(sample_w)
                for _ in range(local_epochs):
                    logits = np.clip(Xc @ w_loc + b_loc, -20.0, 20.0)
                    probs_c = 1.0 / (1.0 + np.exp(-logits))
                    err = (probs_c - yc) * sample_w
                    grad_w = (Xc.T @ err) / nc + 0.05 * w_loc + fedprox_mu * (w_loc - w_global)
                    grad_b = float(np.mean(err)) + fedprox_mu * (b_loc - b_global)
                    w_loc -= lr * grad_w
                    b_loc -= lr * grad_b
                w_updates.append(w_loc)
                b_updates.append(b_loc)
                counts.append(nc)

            total_n = float(sum(counts))
            w_global = sum(w_u * (c_n / total_n) for w_u, c_n in zip(w_updates, counts))
            b_global = sum(b_u * (c_n / total_n) for b_u, c_n in zip(b_updates, counts))

        X_te = np.nan_to_num(df_te_norm[feat_cols].values.astype(np.float64), nan=0.0)
        te_logits = np.clip(X_te @ w_global + b_global, -20.0, 20.0)
        te_probs = 1.0 / (1.0 + np.exp(-te_logits))

        metrics = compute_classification_metrics(y_te, te_probs, threshold=0.50)
        metrics["subject_id"] = str(test_sub)
        metrics["dataset"] = dataset_name
        metrics["algorithm"] = algorithm_name
        metrics["n_windows"] = len(y_te)
        per_subject_rows.append(metrics)

    res_df = pd.DataFrame(per_subject_rows)
    if res_df.empty:
        return res_df, {}

    comm_f32 = compute_communication_cost_kb(n_params, rounds, m_select, quantized_int8=False)
    comm_int8 = compute_communication_cost_kb(n_params, rounds, m_select, quantized_int8=True)
    _, ci_low, ci_high = bootstrap_subject_confidence_intervals(per_subject_rows, metric_name="macro_f1")

    summary = {
        "dataset": dataset_name,
        "algorithm": algorithm_name,
        "n_subjects": int(len(res_df)),
        "clients_per_round": int(m_select),
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
        "false_alarm_episodes_mean": round(float(res_df["false_alarm_episodes_per_hr"].mean()), 2),
        "total_comm_float32_kb": comm_f32["total_communication_kb"],
        "total_comm_int8_kb": comm_int8["total_communication_kb"],
        "macro_f1_95ci_lower": round(ci_low, 4),
        "macro_f1_95ci_upper": round(ci_high, 4),
    }
    return res_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 Federated Edge Learning Simulation.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--local-epochs", type=int, default=1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    feat_60_path = Path("data/labels/real_physionet_features_60s.csv")
    ten_60_path = Path("data/cache/real_physionet_tensors_60s.npz")
    if feat_60_path.exists() and ten_60_path.exists():
        df_all = pd.read_csv(feat_60_path)
        ten_all = np.load(ten_60_path)["tensors"]
        mask = (df_all["session"] == "STRESS").values
        # Evaluate across ALL 36 valid PhysioNet_STRESS subjects (35 training clients per LOSO fold)
        df_wesad = df_all[mask].reset_index(drop=True)
        ten_wesad = ten_all[mask][:, :, ::16]
        ds_name = "PhysioNet_STRESS"
    else:
        df_wesad, ten_wesad = generate_synthetic_cohort(
            "WESAD", [f"S{i}" for i in [2, 3, 4, 5, 6, 7]], windows_per_subject=16, seq_len=512, seed=42
        )
        ds_name = "WESAD"

    labels = df_wesad["label"].values
    subjects = df_wesad["subject_id"].values

    rows = []

    # A. Tabular Logistic Regression Centralized vs Federated (All 36 subjects)
    from src.eval.loso_evaluator import run_loso_evaluation
    from src.eval.run_phase1_experiments import TABULAR_FEATURE_COLS

    _, cent_lr = run_loso_evaluation(df_wesad, "m0_logistic_regression", TABULAR_FEATURE_COLS, dataset_name=ds_name)
    if cent_lr:
        rows.append({
            "dataset": ds_name,
            "algorithm": "Centralized_LOSO_LogReg",
            "n_subjects": cent_lr["n_subjects"],
            "clients_per_round": cent_lr["n_subjects"] - 1,
            "rounds": 1,
            "local_epochs": 1,
            "client_fraction": 1.0,
            "fedprox_mu": 0.0,
            "macro_f1_mean": cent_lr["macro_f1_mean"],
            "macro_f1_std": cent_lr["macro_f1_std"],
            "macro_f1_worst": cent_lr["macro_f1_worst"],
            "strain_f1_mean": cent_lr["strain_f1_mean"],
            "balanced_acc_mean": cent_lr["balanced_acc_mean"],
            "roc_auc_mean": cent_lr["roc_auc_mean"],
            "ece_mean": cent_lr["ece_mean"],
            "false_alerts_persisted_mean": cent_lr["false_alerts_persisted_mean"],
            "false_alarm_episodes_mean": cent_lr["false_alarm_episodes_mean"],
            "total_comm_float32_kb": 0.0,
            "total_comm_int8_kb": 0.0,
            "macro_f1_95ci_lower": cent_lr["macro_f1_95ci_lower"],
            "macro_f1_95ci_upper": cent_lr["macro_f1_95ci_upper"],
        })

    _, fed_lr_avg = run_federated_tabular_logreg_loso(
        df_wesad, dataset_name=ds_name, algorithm_name="FedAvg_Tabular_LogReg", rounds=8, local_epochs=5, client_fraction=1.0, fedprox_mu=0.0
    )
    if fed_lr_avg:
        rows.append(fed_lr_avg)

    _, fed_lr_prox = run_federated_tabular_logreg_loso(
        df_wesad, dataset_name=ds_name, algorithm_name="FedProx_Tabular_LogReg_mu0.05", rounds=8, local_epochs=5, client_fraction=1.0, fedprox_mu=0.05
    )
    if fed_lr_prox:
        rows.append(fed_lr_prox)

    # B. 1D-CNN (M2) Centralized vs Federated (All 36 subjects)
    _, cent_sum = run_cnn_loso_evaluation(
        tensors=ten_wesad,
        labels=labels,
        subject_ids=subjects,
        dataset_name=ds_name,
        epochs=max(6, args.rounds * args.local_epochs),
        batch_size=64,
    )
    if cent_sum:
        rows.append({
            "dataset": ds_name,
            "algorithm": "Centralized_LOSO_M2_CNN",
            "n_subjects": cent_sum["n_subjects"],
            "clients_per_round": cent_sum["n_subjects"] - 1,
            "rounds": 1,
            "local_epochs": max(6, args.rounds * args.local_epochs),
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
            "false_alarm_episodes_mean": cent_sum["false_alarm_episodes_mean"],
            "total_comm_float32_kb": 0.0,
            "total_comm_int8_kb": 0.0,
            "macro_f1_95ci_lower": cent_sum["macro_f1_95ci_lower"],
            "macro_f1_95ci_upper": cent_sum["macro_f1_95ci_upper"],
        })

    # 2. FedAvg 1D-CNN (Full Client Participation, mu = 0.0, all 35 training clients/round)
    _, fedavg_sum = run_federated_loso_evaluation(
        tensors=ten_wesad,
        labels=labels,
        subject_ids=subjects,
        dataset_name=ds_name,
        algorithm_name="FedAvg_M2_CNN_FullClients",
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        client_fraction=1.0,
        fedprox_mu=0.0,
        batch_size=128,
    )
    if fedavg_sum:
        rows.append(fedavg_sum)

    # 3. FedProx 1D-CNN (Full Client Participation, mu = 0.05)
    _, fedprox_sum = run_federated_loso_evaluation(
        tensors=ten_wesad,
        labels=labels,
        subject_ids=subjects,
        dataset_name=ds_name,
        algorithm_name="FedProx_M2_CNN_mu0.05",
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        client_fraction=1.0,
        fedprox_mu=0.05,
        batch_size=128,
    )
    if fedprox_sum:
        rows.append(fedprox_sum)

    # 4. FedAvg 1D-CNN Partial Participation (60% clients per round = 21 clients/round)
    _, fedpart_sum = run_federated_loso_evaluation(
        tensors=ten_wesad,
        labels=labels,
        subject_ids=subjects,
        dataset_name=ds_name,
        algorithm_name="FedAvg_M2_CNN_PartialClients_0.6",
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        client_fraction=0.6,
        fedprox_mu=0.0,
        batch_size=128,
    )
    if fedpart_sum:
        rows.append(fedpart_sum)

    df_out = pd.DataFrame(rows)
    out_csv = args.out_dir / "phase3_federated_benchmark.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"[*] Saved Phase 3 Federated Learning benchmark table to {out_csv}")


if __name__ == "__main__":
    main()

