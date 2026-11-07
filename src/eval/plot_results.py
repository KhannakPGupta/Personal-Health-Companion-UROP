"""Generate 7 publication-ready figures (300 DPI PNGs) from real benchmark tables in results/tables/.

Saves to results/figures/:
  1. fig1_loso_model_comparison.png
  2. fig2_sensor_and_normalization_ablations.png
  3. fig3_contextual_fusion_matrix.png
  4. fig4_federated_comm_vs_f1.png
  5. fig5_cross_protocol_transfer_heatmap.png
  6. fig6_edge_quantization_pareto.png
  7. fig7_system_architecture_overview.png
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


PALETTE = {
    "PhysioNet_STRESS": "#1f77b4",
    "PhysioNet_EXERCISE": "#2ca02c",
    "Pooled": "#ff7f0e",
}


def plot_fig1_loso_comparison(tables_dir: Path, fig_dir: Path) -> None:
    """Plot LOSO E1 Macro-F1 (with 95% CI error bars), ROC-AUC, and False Alarm Episodes/hr."""
    df = pd.read_csv(tables_dir / "loso_benchmark_e1.csv")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=300)

    datasets = ["PhysioNet_STRESS", "PhysioNet_EXERCISE", "Pooled"]
    models = [
        "m0_logistic_regression",
        "m1_decision_tree",
        "m1_random_forest",
        "m1_hist_gradient_boosting",
        "m2_cnn_1d",
    ]
    model_labels = [
        "M0: Logistic\nRegression (26f)",
        "M1: Decision\nTree (26f)",
        "M1: Random\nForest (26f)",
        "M1: HistGrad\nBoosting (26f)",
        "M2: 1D-CNN\nGroupNorm (6ch)",
    ]
    x = np.arange(len(models))
    width = 0.25

    for idx, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds].set_index("model").reindex(models)
        offset = (idx - 1) * width
        color = PALETTE.get(ds, "#333333")

        # Panel 0: Macro-F1 with 95% Bootstrap CI error bars
        f1_vals = sub["macro_f1_mean"].values
        ci_low = f1_vals - sub["macro_f1_95ci_lower"].values
        ci_high = sub["macro_f1_95ci_upper"].values - f1_vals
        axes[0].bar(
            x + offset,
            f1_vals,
            width,
            label=ds,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            yerr=[np.maximum(0, ci_low), np.maximum(0, ci_high)],
            capsize=2.5,
            error_kw={"elinewidth": 0.9, "ecolor": "#222222"},
        )

        # Panel 1: ROC-AUC
        axes[1].bar(
            x + offset,
            sub["roc_auc_mean"].values,
            width,
            label=ds,
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )

        # Panel 2: False Alarm Onset Episodes per Baseline Hour
        fa_col = "false_alarm_episodes_mean" if "false_alarm_episodes_mean" in sub.columns else "false_alerts_per_hr_mean"
        axes[2].bar(
            x + offset,
            sub[fa_col].values,
            width,
            label=ds,
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )

    for ax, title, ylabel, ylim in zip(
        axes,
        [
            "(A) Macro-F1 Score (Mean ± 95% Bootstrap CI)",
            "(B) Area Under ROC Curve (ROC-AUC)",
            "(C) Persisted False-Alarm Onset Episodes / hr",
        ],
        ["Macro-F1 (LOSO)", "ROC-AUC (LOSO)", "Distinct Alarm Onsets / Baseline hr"],
        [(0.0, 1.05), (0.0, 1.08), (0.0, 8.0)],
    ):
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=8)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.45)
        ax.legend(fontsize=8, loc="upper left" if ylim[1] > 2 else "lower right")

    fig.suptitle(
        "Evaluation E1: Subject-Independent LOSO Benchmark Across 37 Real Subjects (68 Wearable Sessions)\n"
        "[PhysioNet_STRESS n=36 | PhysioNet_EXERCISE n=32 | Pooled n=37]",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_loso_model_comparison.png", dpi=300)
    plt.close(fig)


def plot_fig2_ablations(tables_dir: Path, fig_dir: Path) -> None:
    """Plot E3 36-Subject Sensor Modality (with 95% CI), Normalization, Window & Cohort Ablations."""
    df = pd.read_csv(tables_dir / "sensor_window_ablations_e3.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), dpi=300)

    # Panel A: Sensor modality ablation for M1 Random Forest (with 95% Bootstrap CIs)
    df_sens = df[(df["ablation_type"] == "sensor_modality") & (df["model"] == "m1_random_forest")]
    if not df_sens.empty:
        y_pos = np.arange(len(df_sens))
        f1 = df_sens["macro_f1_mean"].values
        err_low = np.maximum(0, f1 - df_sens["macro_f1_95ci_lower"].values)
        err_high = np.maximum(0, df_sens["macro_f1_95ci_upper"].values - f1)

        bars1 = axes[0].barh(
            y_pos - 0.18,
            f1,
            height=0.35,
            xerr=[err_low, err_high],
            capsize=3,
            label="Macro-F1 (±95% CI)",
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.6,
        )
        axes[0].barh(
            y_pos + 0.18,
            df_sens["roc_auc_mean"].values,
            height=0.35,
            label="ROC-AUC",
            color="#ff7f0e",
            edgecolor="black",
            linewidth=0.6,
        )
        for bar, val in zip(bars1, f1):
            axes[0].text(val + 0.065, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=7.5, fontweight="bold")

        labels_a = [
            f"{row['configuration']}\n({int(row['n_features'])} feats, n={int(row['n_subjects'])})"
            for _, row in df_sens.iterrows()
        ]
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(labels_a, fontsize=8)
        axes[0].set_xlim(0.0, 0.95)
        axes[0].set_xlabel("Score (Full 36-Subject PhysioNet_STRESS LOSO)", fontsize=9, fontweight="bold")
        axes[0].set_title("(A) Sensor Subset & 21-vs-26 Feature Ablation\n[Note Overlapping 95% CIs & Minimal TEMP/ACC Gain]", fontsize=9.5, fontweight="bold")
        axes[0].grid(axis="x", linestyle="--", alpha=0.45)
        axes[0].legend(fontsize=8, loc="lower right")

    # Panel B: Window Duration, Normalization Strategy & Cohort Sensitivity (M1 Random Forest)
    df_other = df[
        (df["ablation_type"].isin(["window_duration", "normalization_strategy", "protocol_cohort_sensitivity"]))
        & (df["model"] == "m1_random_forest")
    ]
    if not df_other.empty:
        labels_b = [f"{r['ablation_type'].split('_')[0].upper()}: {r['configuration']} (n={int(r['n_subjects'])})" for _, r in df_other.iterrows()]
        y_pos2 = np.arange(len(df_other))
        f1_b = df_other["macro_f1_mean"].values
        bars2 = axes[1].barh(y_pos2 - 0.18, f1_b, height=0.35, label="Macro-F1", color="#2ca02c", edgecolor="black", linewidth=0.6)
        axes[1].barh(y_pos2 + 0.18, df_other["roc_auc_mean"].values, height=0.35, label="ROC-AUC", color="#9467bd", edgecolor="black", linewidth=0.6)
        for bar, val in zip(bars2, f1_b):
            axes[1].text(val + 0.015, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=7.5, fontweight="bold")

        axes[1].set_yticks(y_pos2)
        axes[1].set_yticklabels(labels_b, fontsize=7.8)
        axes[1].set_xlim(0.0, 0.95)
        axes[1].set_xlabel("Score (PhysioNet_STRESS LOSO)", fontsize=9, fontweight="bold")
        axes[1].set_title("(B) Window Length, Subject Calibration & Protocol Cohort Ablations\n[Resting-Prefix=0.6544 vs Oracle=0.7025 | S* Cohort=0.6776 vs f* Cohort=0.6347]", fontsize=9, fontweight="bold")
        axes[1].grid(axis="x", linestyle="--", alpha=0.45)
        axes[1].legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "Evaluation E3: Sensor Modality, Feature Interaction (21 vs 26), Calibration & Cohort Ablations",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_sensor_and_normalization_ablations.png", dpi=300)
    plt.close(fig)


def plot_fig3_contextual_fusion(tables_dir: Path, fig_dir: Path) -> None:
    """Plot Phase 2 Next-Day CPCB AQI Forecast Benchmark + Contextual Risk Fusion Scenarios."""
    df = pd.read_csv(tables_dir / "phase2_contextual_fusion_demo.csv")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2), dpi=300, gridspec_kw={"width_ratios": [1.0, 1.85]})

    # Panel A: Next-Day (t+1) AQI Forecast vs Same-Day Persistence Baseline on Real city_day.csv
    row0 = df.iloc[0]
    metrics_names = ["Macro-F1\n(Test n=5,161d)", "Balanced Acc\n(Test n=5,161d)", "ROC-AUC\n(Test n=5,161d)"]
    hgb_scores = [
        float(row0["aqi_hazard_macro_f1"]),
        float(row0["aqi_hazard_balanced_accuracy"]),
        float(row0["aqi_hazard_roc_auc"]),
    ]
    pers_scores = [
        float(row0["persistence_baseline_macro_f1"]),
        float(row0["persistence_baseline_balanced_acc"]),
        float(row0["persistence_baseline_balanced_acc"]),
    ]
    x_a = np.arange(len(metrics_names))
    b1 = axes[0].bar(x_a - 0.18, hgb_scores, width=0.34, label="HistGradBoost Next-Day (t+1) Forecast", color="#00796b", edgecolor="black")
    b2 = axes[0].bar(x_a + 0.18, pers_scores, width=0.34, label="Naive Persistence Baseline (y_hat_t+1 = y_t)", color="#b0bec5", edgecolor="black")
    for bar in list(b1) + list(b2):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{bar.get_height():.4f}", ha="center", fontsize=7.5, fontweight="bold")
    axes[0].set_xticks(x_a)
    axes[0].set_xticklabels(metrics_names, fontsize=8.5)
    axes[0].set_ylim(0.75, 1.03)
    axes[0].set_ylabel("Chronological Test Score (26 Indian Cities)", fontsize=9, fontweight="bold")
    axes[0].set_title("(A) Real CPCB India city_day.csv (2015-2020)\nNext-Day Hazard (AQI_t+1 > 200) Forecast", fontsize=9.5, fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.45)
    axes[0].legend(fontsize=7.8, loc="lower right")

    # Panel B: Contextual Fusion Scenarios (including Ambient Heat >= 38C and Real OOF S01 Scenarios)
    x = np.arange(len(df))
    colors = [
        "#4caf50" if lvl == 0 else ("#ffb300" if lvl == 1 else ("#fb8c00" if lvl == 2 else "#e53935"))
        for lvl in df["alert_level"]
    ]

    bars = axes[1].bar(x, df["composite_risk_score"], color=colors, width=0.58, edgecolor="black", linewidth=0.8)
    for bar, (_, row) in zip(bars, df.iterrows()):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.018,
            f"{row['alert_name']}\n(AQI={int(row['ambient_aqi'])}, T={row['ambient_temp_c']:.0f}°C)",
            ha="center",
            va="bottom",
            fontsize=6.8,
            fontweight="bold",
        )

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([s.replace("_", "\n") for s in df["scenario_id"]], fontsize=7.2)
    axes[1].set_ylim(0.0, 1.12)
    axes[1].set_ylabel("Composite Risk Score [0 - 1]", fontsize=9, fontweight="bold")
    axes[1].set_title(
        "(B) Multimodal Contextual Risk Engine (Wearable Strain + Next-Day AQI + Ambient Heat >= 38°C)\n"
        "[S7 & S8 Driven by Real Out-of-Fold LOSO Subject S01 Predictions + Real Delhi/Mumbai Test Split]",
        fontsize=9.2,
        fontweight="bold",
    )
    axes[1].grid(axis="y", linestyle="--", alpha=0.45)

    fig.suptitle("Phase 2: Causal Next-Day Environmental Hazard Forecasting & Multimodal Alert Escalation", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_contextual_fusion_matrix.png", dpi=300)
    plt.close(fig)


def plot_fig4_federated(tables_dir: Path, fig_dir: Path) -> None:
    """Plot Phase 3 Federated Learning accuracy vs float32/int8 communication cost."""
    df = pd.read_csv(tables_dir / "phase3_federated_benchmark.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    x = np.arange(len(df))
    b1 = axes[0].bar(x - 0.18, df["macro_f1_mean"], width=0.35, label="Macro-F1", color="#1976d2", edgecolor="black", linewidth=0.6)
    b2 = axes[0].bar(x + 0.18, df["roc_auc_mean"], width=0.35, label="ROC-AUC", color="#00897b", edgecolor="black", linewidth=0.6)
    for bar in list(b1) + list(b2):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{bar.get_height():.3f}", ha="center", fontsize=6.8, fontweight="bold")

    labels = [f"{r['algorithm'].replace('_', ' ')}\n(n={int(r['n_subjects'])}, c/r={int(r['clients_per_round'])})" for _, r in df.iterrows()]
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=7, rotation=18, ha="right")
    axes[0].set_ylim(0.0, 0.95)
    axes[0].set_ylabel("Subject-Independent Score", fontsize=9, fontweight="bold")
    axes[0].set_title("(A) 36-Subject Tabular LogReg vs. 12-Subject 1D-CNN\n[Tabular FL Matches Centralized; 12-Sub CNN Fails to Converge]", fontsize=9.2, fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.45)
    axes[0].legend(fontsize=8, loc="upper right")

    df_comm = df[df["total_comm_float32_kb"] > 0].copy()
    xc = np.arange(len(df_comm))
    c1 = axes[1].bar(xc - 0.18, df_comm["total_comm_float32_kb"], width=0.35, label="Float32 Traffic (KB)", color="#e64a19", edgecolor="black", linewidth=0.6)
    c2 = axes[1].bar(xc + 0.18, df_comm["total_comm_int8_kb"], width=0.35, label="Int8 Quantized Traffic (KB)", color="#512da8", edgecolor="black", linewidth=0.6)
    for bar in list(c1) + list(c2):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.12, f"{bar.get_height():.1f} KB", ha="center", fontsize=7, fontweight="bold")

    axes[1].set_yscale("log")
    axes[1].set_xticks(xc)
    axes[1].set_xticklabels([a.replace("_", "\n") for a in df_comm["algorithm"]], fontsize=7.5)
    axes[1].set_ylim(1.0, 25000.0)
    axes[1].set_ylabel("Total Uplink + Downlink Traffic (KB, Log Scale)", fontsize=9, fontweight="bold")
    axes[1].set_title("(B) Total 20-Round Communication Overhead\n[Tabular LogReg Int8 Requires Only 8.86 - 14.77 KB]", fontsize=9.2, fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.45)
    axes[1].legend(fontsize=8, loc="upper left")

    fig.suptitle("Phase 3: Privacy-Preserving Federated Learning Benchmark on PhysioNet_STRESS", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_federated_comm_vs_f1.png", dpi=300)
    plt.close(fig)


def plot_fig5_cross_protocol_transfer(tables_dir: Path, fig_dir: Path) -> None:
    """Plot E2 Cross-Protocol Domain Generalization vs Within-Dataset LOSO."""
    df_e1 = pd.read_csv(tables_dir / "loso_benchmark_e1.csv")
    df_e2 = pd.read_csv(tables_dir / "cross_dataset_transfer_e2.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), dpi=300)
    models = ["m0_logistic_regression", "m1_decision_tree", "m1_random_forest", "m1_hist_gradient_boosting", "m2_cnn_1d"]
    short_names = ["M0 LogReg", "M1 DecTree", "M1 RandForest", "M1 HistGB", "M2 1D-CNN"]
    x = np.arange(len(models))
    width = 0.34

    # Panel A: Target = EXERCISE (Within-Dataset LOSO vs Cross-Protocol STRESS -> EXERCISE)
    e1_ex = df_e1[df_e1["dataset"] == "PhysioNet_EXERCISE"].set_index("model").reindex(models)
    e2_s2e = df_e2[df_e2["direction"] == "PhysioNet_STRESS->PhysioNet_EXERCISE"].set_index("model").reindex(models)

    b1 = axes[0].bar(x - width / 2, e1_ex["macro_f1_mean"], width, label="Within-Cohort LOSO (EXERCISE -> EXERCISE)", color="#2ca02c", edgecolor="black")
    b2 = axes[0].bar(x + width / 2, e2_s2e["macro_f1_mean"], width, label="Zero-Shot Transfer (STRESS -> EXERCISE)", color="#d62728", edgecolor="black")
    for bar in list(b1) + list(b2):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{bar.get_height():.3f}", ha="center", fontsize=7.5, fontweight="bold")
    axes[0].axhline(0.2024, color="#d62728", linestyle=":", linewidth=1.2, label="All-Baseline Collapse (0.2024)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, fontsize=8.5)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Target Macro-F1", fontsize=9, fontweight="bold")
    axes[0].set_title("(A) Evaluating on PhysioNet_EXERCISE (n=32)\n[Zero-Shot Models Trained on Stress Collapse to Baseline]", fontsize=9.5, fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.45)
    axes[0].legend(fontsize=8, loc="upper right")

    # Panel B: Target = STRESS (Within-Dataset LOSO vs Cross-Protocol EXERCISE -> STRESS)
    e1_st = df_e1[df_e1["dataset"] == "PhysioNet_STRESS"].set_index("model").reindex(models)
    e2_e2s = df_e2[df_e2["direction"] == "PhysioNet_EXERCISE->PhysioNet_STRESS"].set_index("model").reindex(models)

    b3 = axes[1].bar(x - width / 2, e1_st["macro_f1_mean"], width, label="Within-Cohort LOSO (STRESS -> STRESS)", color="#1f77b4", edgecolor="black")
    b4 = axes[1].bar(x + width / 2, e2_e2s["macro_f1_mean"], width, label="Zero-Shot Transfer (EXERCISE -> STRESS)", color="#ff7f0e", edgecolor="black")
    for bar in list(b3) + list(b4):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{bar.get_height():.3f}", ha="center", fontsize=7.5, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, fontsize=8.5)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Target Macro-F1", fontsize=9, fontweight="bold")
    axes[1].set_title("(B) Evaluating on PhysioNet_STRESS (n=36)\n[Exertion-Trained Models Reach Only 0.45 - 0.52 Macro-F1]", fontsize=9.5, fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.45)
    axes[1].legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Evaluation E2: Cross-Protocol Domain Shift — Psychological Stress vs. Physical Exertion Are Distinct Signals",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_cross_protocol_transfer_heatmap.png", dpi=300)
    plt.close(fig)


def plot_fig6_edge_quantization(tables_dir: Path, fig_dir: Path) -> None:
    """Plot Multi-Model Edge Quantization Profile (Float32 vs Int8 Size, Held-Out Macro-F1, and Latency)."""
    df = pd.read_csv(tables_dir / "edge_profile_m2.csv")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=300)

    x = np.arange(len(df))
    labels = [m.replace("_", "\n") for m in df["model"]]

    # Panel A: Float32 vs Int8 Model Footprint (KB, Log Scale)
    b1 = axes[0].bar(x - 0.18, df["float_size_kb"], width=0.35, label="Float32 Footprint (KB)", color="#fb8c00", edgecolor="black")
    b2 = axes[0].bar(x + 0.18, df["int8_size_kb"], width=0.35, label="Int8 Footprint (KB)", color="#1e88e5", edgecolor="black")
    for bar, (_, r) in zip(b2, df.iterrows()):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.18, f"{r['int8_size_kb']:.2f}K\n({r['compression_ratio']:.1f}x)", ha="center", fontsize=6.8, fontweight="bold")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylim(0.02, 2000.0)
    axes[0].set_ylabel("Memory Footprint (KB, Log Scale)", fontsize=9, fontweight="bold")
    axes[0].set_title("(A) Model Footprint & Compression Ratio\n[M0=0.05 KB (18.2x) | M2 CNN=12.88 KB (4.3x)]", fontsize=9.2, fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.45)
    axes[0].legend(fontsize=8, loc="upper left")

    # Panel B: Held-Out Subject Split Accuracy (Float32 vs Int8 Macro-F1)
    b3 = axes[1].bar(x - 0.18, df["float_macro_f1"], width=0.35, label="Float32 Held-Out Macro-F1", color="#43a047", edgecolor="black")
    b4 = axes[1].bar(x + 0.18, df["int8_macro_f1"], width=0.35, label="Int8 Held-Out Macro-F1", color="#00acc1", edgecolor="black")
    for bar in b4:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{bar.get_height():.4f}", ha="center", fontsize=7.2, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylim(0.50, 1.02)
    axes[1].set_ylabel("Held-Out Test Macro-F1 (9 Subjects, 758 Win)", fontsize=9, fontweight="bold")
    axes[1].set_title("(B) Post-Training Quantization Fidelity\n[Zero/Negligible F1 Drop (|ΔF1| <= 0.0010)]", fontsize=9.2, fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.45)
    axes[1].legend(fontsize=8, loc="lower right")

    # Panel C: Desktop x86_64 Single-Thread Proxy Latency (p50 & p95 ms)
    b5 = axes[2].bar(x - 0.18, df["latency_p50_ms"], width=0.35, label="p50 Median Latency (ms)", color="#5e35b1", edgecolor="black")
    b6 = axes[2].bar(x + 0.18, df["latency_p95_ms"], width=0.35, label="p95 Tail Latency (ms)", color="#e53935", edgecolor="black")
    for bar in b5:
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.18, f"{bar.get_height():.2f}ms", ha="center", fontsize=7.2, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].set_ylim(0.0, 10.2)
    axes[2].set_ylabel("Inference Latency per Window (ms)", fontsize=9, fontweight="bold")
    axes[2].set_title("(C) Desktop x86_64 CPU Proxy Latency\n[Single-Thread Proxy — Not Physical MCU]", fontsize=9.2, fontweight="bold")
    axes[2].grid(axis="y", linestyle="--", alpha=0.45)
    axes[2].legend(fontsize=8, loc="upper left")

    fig.suptitle("Phase 1D: Multi-Model Edge Deployment & Int8 Quantization Benchmark (Trained on 28 Subjects, Tested on 9 Subjects)", fontsize=10.8, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig6_edge_quantization_pareto.png", dpi=300)
    plt.close(fig)


def plot_fig7_architecture_diagram(fig_dir: Path) -> None:
    """Render a crisp 300-DPI System Architecture & Dataflow Diagram as PNG."""
    fig, ax = plt.subplots(figsize=(15, 6.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 48)
    ax.axis("off")

    def draw_box(x: float, y: float, w: float, h: float, title: str, body: str, facecolor: str, edgecolor: str = "#1a237e") -> None:
        box = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.6,rounding_size=1.2",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 2.3, title, ha="center", va="top", fontsize=9.2, fontweight="bold", color="#0d1b2a")
        ax.text(x + w / 2, y + h / 2 - 1.2, body, ha="center", va="center", fontsize=7.6, color="#1b263b")

    def draw_arrow(x1: float, y1: float, x2: float, y2: float, label: str = "") -> None:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>,head_width=0.6,head_length=0.8", lw=1.6, color="#37474f"),
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.1, label, ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#263238")

    # Title Banner
    ax.text(
        50, 45.8,
        "Edge-Intelligent Personal Health Companion — Three-Phase System Architecture",
        ha="center", va="center", fontsize=12.5, fontweight="bold", color="#0d1b2a"
    )
    ax.text(
        50, 43.2,
        "Strict Subject-Independent LOSO (37 Real Subjects, 68 Wearable Sessions) + Real CPCB India Next-Day AQI Forecast (29,531 Daily Records)",
        ha="center", va="center", fontsize=8.8, color="#37474f"
    )

    # Row 1: Phase 1 Wearable Edge Pipeline
    draw_box(
        2, 24, 19, 15.5,
        "1. Multimodal Wearable I/O",
        "Empatica E4 Wrist Sensors\n• BVP (64 Hz)  • EDA (4 Hz)\n• TEMP (4 Hz)  • ACC (32 Hz)\n• 60s Window, 15s Stride\n• Transition Buffer: ±30s",
        "#e3f2fd", "#1565c0"
    )
    draw_arrow(21.6, 31.8, 25.4, 31.8, "60s Windows")

    draw_box(
        26, 24, 22, 15.5,
        "2. Feature & Tensor Engine",
        "• Tabular: 26 Features\n  (21 Base + 5 Autonomic Ratios)\n  Resting-Prefix Z-Score Calib.\n• Raw CNN Tensor: (6, 1920)\n  Resampled at 32 Hz",
        "#e8f5e9", "#2e7d32"
    )
    draw_arrow(48.6, 31.8, 52.4, 31.8, "Zero-Leakage LOSO")

    draw_box(
        53, 24, 21, 15.5,
        "3. Edge Strain Models",
        "• M0 LogReg: 0.05 KB (int8)\n  STRESS F1=0.6658 | EX=0.8881\n• M1 HistGB: 103.6 KB (int8)\n  STRESS F1=0.6616 | EX=0.8941\n• M2 1D-CNN: 12.88 KB (int8)",
        "#fff3e0", "#ef6c00"
    )
    draw_arrow(74.6, 31.8, 78.4, 31.8, "p_phys + K=2 Persist")

    # Row 2: Phase 2 Environmental + Phase 3 Federated
    draw_box(
        2, 4, 34, 15.5,
        "Phase 3: Privacy-Preserving Federated Learning",
        "• 36-Subject Tabular LogReg (35 Clients/Round)\n  FedAvg (60% Partial): F1=0.6588, AUC=0.8062 (8.86 KB int8)\n  FedProx (μ=0.05): F1=0.6584, AUC=0.8040 (14.77 KB int8)\n• 12-Subject M2 1D-CNN Baseline: F1=0.4361 (Fails to Converge)",
        "#f3e5f5", "#6a1b9a"
    )
    draw_arrow(36.6, 11.8, 53.0, 23.4, "Global Aggregated Weights")

    draw_box(
        40, 4, 34, 15.5,
        "Phase 2A: Real CPCB India Next-Day (t+1) AQI Forecast",
        "• Real city_day.csv (26 Indian Cities, 29,531 Daily Rows)\n• Chronological Split: 20,591 Train Days | 5,161 Test Days\n• Target: Next-Day Severe/Poor Hazard (AQI_{t+1} > 200)\n• Test F1 = 0.9094 | Bal Acc = 0.9315 (+4.18% vs Persistence)",
        "#e0f2f1", "#00695c"
    )
    draw_arrow(74.6, 11.8, 78.4, 18.0, "p_env(t+1) + Temp°C")

    # Right Output Column: Contextual Risk Fusion Engine
    draw_box(
        79, 8, 19, 31.5,
        "4. Contextual Risk\nFusion Engine",
        "Inputs:\n• Wearable p_phys\n• Persisted Alarm (K=2)\n• Next-Day Hazard p_env\n• Ambient Heat (T >= 38°C)\n\n4-Tier Alert Escalation:\n0: LOW_NORMAL\n1: MODERATE_ADVISORY\n2: HIGH_RISK_ALERT\n3: CRITICAL_EMERGENCY\n\nValidated on Real S01\nOOF + Delhi/Mumbai AQI",
        "#ffebee", "#c62828"
    )

    fig.tight_layout()
    fig.savefig(fig_dir / "fig7_system_architecture_overview.png", dpi=300)
    plt.close(fig)


def main() -> None:
    tables_dir = Path("results/tables")
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_fig1_loso_comparison(tables_dir, fig_dir)
    plot_fig2_ablations(tables_dir, fig_dir)
    plot_fig3_contextual_fusion(tables_dir, fig_dir)
    plot_fig4_federated(tables_dir, fig_dir)
    plot_fig5_cross_protocol_transfer(tables_dir, fig_dir)
    plot_fig6_edge_quantization(tables_dir, fig_dir)
    plot_fig7_architecture_diagram(fig_dir)
    print(f"[*] Generated 7 publication figures in {fig_dir}")


if __name__ == "__main__":
    main()
