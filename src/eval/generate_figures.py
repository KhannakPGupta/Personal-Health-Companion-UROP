"""Generate publication-ready figures (300 DPI PNGs) from real benchmark tables in results/tables/.

Saves to results/figures/:
  1. fig1_loso_model_comparison.png
  2. fig2_sensor_and_normalization_ablations.png
  3. fig3_contextual_fusion_matrix.png
  4. fig4_federated_comm_vs_f1.png
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_fig1_loso_comparison(tables_dir: Path, fig_dir: Path) -> None:
    """Plot LOSO E1 Macro-F1 and ROC-AUC across M0, M1, and M2 models."""
    df = pd.read_csv(tables_dir / "loso_benchmark_e1.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    datasets = df["dataset"].unique()
    models = df["model"].unique()
    x = np.arange(len(models))
    width = 0.25

    for idx, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds].set_index("model").reindex(models)
        offset = (idx - 1) * width
        axes[0].bar(x + offset, sub["macro_f1_mean"], width, label=ds)
        axes[1].bar(x + offset, sub["roc_auc_mean"], width, label=ds)

    for ax, metric_title in zip(axes, ["Macro-F1 (LOSO)", "ROC-AUC (LOSO)"]):
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=8)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(metric_title)
        ax.set_title(f"Evaluation E1: {metric_title} across Cohorts")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Wearable Physiological Strain Proxy Detection (Strict Subject-Independent LOSO)\n"
        "[Disclaimer: Experimental Strain Proxy Only — Non-Clinical]",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_loso_model_comparison.png", dpi=300)
    plt.close(fig)


def plot_fig2_ablations(tables_dir: Path, fig_dir: Path) -> None:
    """Plot Sensor Modality and Normalization Strategy Ablation results."""
    df = pd.read_csv(tables_dir / "sensor_window_ablations_e3.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Panel A: Sensor modality ablation for M1 Random Forest
    df_sens = df[(df["ablation_type"] == "sensor_modality") & (df["model"] == "m1_random_forest")]
    if not df_sens.empty:
        y_pos = np.arange(len(df_sens))
        axes[0].barh(y_pos - 0.18, df_sens["macro_f1_mean"], height=0.35, label="Macro-F1", color="#2b5c8f")
        axes[0].barh(y_pos + 0.18, df_sens["roc_auc_mean"], height=0.35, label="ROC-AUC", color="#d95f02")
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(df_sens["configuration"], fontsize=8)
        axes[0].set_xlim(0.0, 0.95)
        axes[0].set_xlabel("Score")
        axes[0].set_title("Sensor Modality Ablation (M1 Random Forest)")
        axes[0].grid(axis="x", linestyle="--", alpha=0.5)
        axes[0].legend(fontsize=8)

    # Panel B: Window Duration & Normalization Strategy
    df_other = df[df["ablation_type"].isin(["window_duration", "normalization_strategy"])]
    if not df_other.empty:
        labels = [f"{r['configuration']}\n({r['model']})" for _, r in df_other.iterrows()]
        y_pos2 = np.arange(len(df_other))
        axes[1].barh(y_pos2 - 0.18, df_other["macro_f1_mean"], height=0.35, label="Macro-F1", color="#388e3c")
        axes[1].barh(y_pos2 + 0.18, df_other["roc_auc_mean"], height=0.35, label="ROC-AUC", color="#7b1fa2")
        axes[1].set_yticks(y_pos2)
        axes[1].set_yticklabels(labels, fontsize=8)
        axes[1].set_xlim(0.0, 0.95)
        axes[1].set_xlabel("Score")
        axes[1].set_title("Window Length & Subject Normalization Ablations")
        axes[1].grid(axis="x", linestyle="--", alpha=0.5)
        axes[1].legend(fontsize=8)

    fig.suptitle(
        "Phase 1E: Sensor Modality, Window Duration & Normalization Ablations (PhysioNet E4)",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_sensor_and_normalization_ablations.png", dpi=300)
    plt.close(fig)


def plot_fig3_contextual_fusion(tables_dir: Path, fig_dir: Path) -> None:
    """Plot Phase 2 Contextual Risk Fusion scores and alert tiers across scenarios."""
    df = pd.read_csv(tables_dir / "phase2_contextual_fusion_demo.csv")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    x = np.arange(len(df))
    colors = ["#4caf50" if lvl == 0 else ("#ffb300" if lvl == 1 else ("#fb8c00" if lvl == 2 else "#e53935"))
              for lvl in df["alert_level"]]

    bars = ax.bar(x, df["composite_risk_score"], color=colors, width=0.55, edgecolor="black")
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{row['alert_name']}\n(AQI={int(row['ambient_aqi'])})",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in df["scenario_id"]], fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Composite Risk Score [0 - 1]")
    ax.set_title(
        "Phase 2: Contextual Physiological-Environmental Risk Fusion (Wearable + CPCB AQI)\n"
        "[Green=NORMAL, Yellow=ADVISORY, Orange=WARNING, Red=CRITICAL_ALERT]",
        fontsize=9.5,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_contextual_fusion_matrix.png", dpi=300)
    plt.close(fig)


def plot_fig4_federated(tables_dir: Path, fig_dir: Path) -> None:
    """Plot Phase 3 Federated Learning accuracy vs float32/int8 communication cost."""
    df = pd.read_csv(tables_dir / "phase3_federated_benchmark.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

    x = np.arange(len(df))
    axes[0].bar(x - 0.18, df["macro_f1_mean"], width=0.35, label="Macro-F1", color="#1976d2")
    axes[0].bar(x + 0.18, df["roc_auc_mean"], width=0.35, label="ROC-AUC", color="#00897b")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([a.replace("_", "\n") for a in df["algorithm"]], fontsize=7)
    axes[0].set_ylim(0.0, 0.90)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Centralized vs. Subject-Isolated Federated LOSO")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=8)

    axes[1].bar(x - 0.18, df["total_comm_float32_kb"], width=0.35, label="Float32 Traffic (KB)", color="#e64a19")
    axes[1].bar(x + 0.18, df["total_comm_int8_kb"], width=0.35, label="Int8 Quantized Traffic (KB)", color="#512da8")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([a.replace("_", "\n") for a in df["algorithm"]], fontsize=8)
    axes[1].set_ylabel("Total Uplink + Downlink Traffic (KB)")
    axes[1].set_title("Federated Communication Overhead (4x Reduction via int8)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=8)

    fig.suptitle("Phase 3: Privacy-Preserving Federated Edge Learning Benchmark", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_federated_comm_vs_f1.png", dpi=300)
    plt.close(fig)


def main() -> None:
    tables_dir = Path("results/tables")
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_fig1_loso_comparison(tables_dir, fig_dir)
    plot_fig2_ablations(tables_dir, fig_dir)
    plot_fig3_contextual_fusion(tables_dir, fig_dir)
    plot_fig4_federated(tables_dir, fig_dir)
    print(f"[*] Generated 4 publication figures in {fig_dir}")


if __name__ == "__main__":
    main()

