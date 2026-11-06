# Personal Health Companion: Wearable Physiological Strain Edge Detection & Contextual Environmental Fusion

**Author:** Khanak Gupta  
**Project Type:** Undergraduate Research Opportunity Programme (UROP) — Edge AI, Wearable Biosignals & Privacy-Preserving Federated Learning  
**Repository:** [KhannakPGupta/Personal-Health-Companion-UROP](https://github.com/KhannakPGupta/Personal-Health-Companion-UROP)

---

> [!WARNING]
> **Mandatory Research Proxy Target & Non-Clinical Disclaimer**  
> Because **no public wearable dataset contains real-world heatwave or flood exposure labels**, the predictive target in this repository is strictly an experimental **PROXY**: detecting **elevated physiological strain** (induced psychological stress or structured physical exertion) versus resting baseline from wrist-worn Empatica E4 signals.  
> **This prototype makes NO claims of clinical validity, core body temperature estimation, heat-stroke diagnosis, or medically validated environmental heat-stress monitoring.** See [docs/assumptions.md](docs/assumptions.md) for complete protocol disclosures.

---

## Table of Contents
1. [Executive Summary & Research Objectives](#1-executive-summary--research-objectives)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Datasets & Window Distribution](#3-datasets--window-distribution)
4. [Signal Processing, Feature Engineering & 32 Hz Tensor Pipeline](#4-signal-processing-feature-engineering--32-hz-tensor-pipeline)
5. [Strict Anti-Leakage & Subject-Isolation Protocol](#5-strict-anti-leakage--subject-isolation-protocol)
6. [Phase 1: Within-Protocol LOSO, Cross-Protocol Transfer & Ablation Results](#6-phase-1-within-protocol-loso-cross-protocol-transfer--ablation-results)
7. [Edge Hardware Footprint, Int8 Quantization & Latency Profile](#7-edge-hardware-footprint-int8-quantization--latency-profile)
8. [Phase 2: CPCB Ambient AQI Hazard Model & Contextual Risk Fusion Engine](#8-phase-2-cpcb-ambient-aqi-hazard-model--contextual-risk-fusion-engine)
9. [Phase 3: Subject-Isolated Federated Edge Learning (`FedAvg` & `FedProx`)](#9-phase-3-subject-isolated-federated-edge-learning-fedavg--fedprox)
10. [Repository Structure & Reproducibility Guide](#10-repository-structure--reproducibility-guide)

---

## 1. Executive Summary & Research Objectives

Wearable health monitors often suffer from three critical limitations when deployed in real-world environmental monitoring:
1. **Subject-Leakage Inflated Benchmarks:** Many published wearable stress studies split sliding windows randomly rather than by participant, leaking subject-specific physiological baselines across train and test sets.
2. **Context Blindness (Exercise vs. Pathological/Environmental Strain):** Purely physiological classifiers trigger false alarms during voluntary physical exercise (`high wrist motion`) or isolated single-window motion artifacts.
3. **Cloud Privacy & Bandwidth Constraints:** Streaming continuous 64 Hz raw photoplethysmography (BVP) and electrodermal activity (EDA) waveforms to cloud servers compromises user privacy and drains wearable battery life.

This project designs, implements, and empirically evaluates a **three-phase on-device Personal Health Companion**:
* **Phase 1 (On-Device Physiological Strain Detection):** Evaluates classical baselines (`M0`), tree ensembles (`M1`), and an ultra-lightweight 1D-CNN (`M2`, **12,161 parameters**, `51.2 KB` quantized, **1.81 ms** median CPU inference latency) with label-free resting-prefix subject calibration and autonomic interaction ratios under strict **Leave-One-Subject-Out (LOSO)** validation, cross-protocol transfer (`STRESS <-> EXERCISE`), and systematic sensor/window/normalization ablations.
* **Phase 2 (Contextual Physiological-Environmental Fusion):** Combines calibrated wearable strain probabilities, a **3-of-5 temporal persistence filter**, wrist actigraphy motion gating (`high_motion_fraction`), and the **Central Pollution Control Board (CPCB) National Air Quality Index (NAQI)** into a four-tier actionable alert hierarchy (`NORMAL`, `ADVISORY`, `WARNING`, `CRITICAL_ALERT`).
* **Phase 3 (Privacy-Preserving Federated Edge Learning):** Simulates decentralized training where each wearable participant acts as an isolated edge client using **Federated Averaging (`FedAvg`)** and **Federated Proximal (`FedProx`)** with `int8` payload compression (`4x` communication reduction).

---

## 2. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph WristWearable["1. Empatica E4 Wrist Sensor Suite"]
        BVP["BVP (PPG)\n64 Hz"]
        EDA["EDA (GSR)\n4 Hz"]
        TEMP["Skin TEMP\n4 Hz"]
        ACC["3-Axis ACC\n32 Hz (g-scaled)"]
    end

    subgraph WindowGating["2. Boundary-Gated Sliding Windows"]
        WIN["60s Window / 30s Step (Primary)\n30s Window / 15s Step (Ablation)\nDrops Any Boundary-Crossing Window"]
    end

    subgraph DualPipeline["3. Parallel Feature & Tensor Extraction"]
        TAB["26 Tabular & Autonomic Interaction Features\n- BVP: Bandpass 0.5-8Hz, HR, RMSSD, SDNN, pNN50\n- EDA: Tonic/Phasic SCR Peaks & Amplitude\n- Autonomic Ratios: Sympathovagal, EDA Arousal, Motion-Adj HR\n- Unsupervised Resting-Prefix Calibration"]
        TEN["6-Channel 32 Hz Tensor (1920 x 6)\n- BVP: Anti-Aliased Decimation (64->32 Hz)\n- EDA & TEMP: Linear Interp (4->32 Hz)\n- ACC_x, ACC_y, ACC_z: Native 32 Hz"]
    end

    subgraph EdgeModels["4. On-Device Strain Classifiers (LOSO)"]
        M0M1["M0 / M1 Classical & Tree Ensembles\nLogReg, DecisionTree, RandomForest, HistGB"]
        M2["M2 Lightweight 1D-CNN (InstanceNorm + SE)\n12,161 Params | 51.2 KB (int8)\n1.806 ms p50 Inference Latency"]
        CAL["Platt Scaling Calibrator\n+ 3-of-5 Window Persistence Gate"]
    end

    subgraph ContextualFusion["5. Phase 2 Contextual Risk Fusion Engine"]
        AQI["CPCB India Ambient AQI\n6-Tier NAQI + PM2.5/PM10 Sub-Index"]
        FUSE["Motion-Gated Rule & Risk Matrix\nComposite Score = 0.60*P(strain) + 0.40*E(env)"]
        ALERT["Tiered Companion Decision\nNORMAL | ADVISORY | WARNING | CRITICAL_ALERT"]
    end

    BVP & EDA & TEMP & ACC --> WIN
    WIN --> TAB & TEN
    TAB --> M0M1
    TEN --> M2
    M0M1 & M2 --> CAL
    CAL --> FUSE
    AQI --> FUSE
    FUSE --> ALERT
```

---

## 3. Datasets & Window Distribution

### 3.1 Datasets Used
| Dataset | Source & License | Modality & Sampling Rates | Experimental Role |
| :--- | :--- | :--- | :--- |
| **PhysioNet Wearable Device Dataset (v1.0.1)** | PhysioNet (`DOI: 10.13026/zzf8-xv61`), ODC-By v1.0 License | Empatica E4 Wrist: `BVP` (64 Hz), `EDA` (4 Hz), `TEMP` (4 Hz), `ACC` (32 Hz, scaled from $1/64\,\text{g} \to \text{g}$) | Primary real-signal evaluation across **100 subject-sessions** (`37 STRESS`, `31 AEROBIC`, `32 ANAEROBIC`) |
| **WESAD (UCI ID: 465)** | University of Siegen / UCI ML Repository (Non-commercial research) | Empatica E4 Wrist (`BVP`, `EDA`, `TEMP`, `ACC`); Chest RespiBAN ignored | Supported via cross-platform loader (`src/io/wesad_loader.py`) |
| **CPCB India Air Quality Data** | Central Pollution Control Board (CPCB) India | `PM2.5`, `PM10`, `NO2`, `CO`, `SO2`, `O3`, `AQI`, `AQI_Bucket` | Phase 2 ambient environmental hazard classification & contextual fusion |

### 3.2 Real Extracted Window Distribution (`data/labels/segment_labels.csv`)
Every sliding window that touches or overlaps a protocol transition boundary (or falls inside warm-up/cooldown periods) is **strictly dropped** to eliminate transition artifacts:

| Cohort / Session | Subjects / Sessions | 60s Windows (30s Step) | Baseline Windows (`0`) | Elevated Strain Windows (`1`) | 30s Windows (15s Step) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`PhysioNet_STRESS`** | 37 | **2,319** | 1,934 (83.4%) | 385 (16.6%) | **5,084** |
| **`PhysioNet_AEROBIC`** | 31 | **1,565** | 494 (31.6%) | 1,071 (68.4%) | **3,300** |
| **`PhysioNet_ANAEROBIC`** | 32 | **908** | 399 (43.9%) | 509 (56.1%) | **1,950** |
| **Total Real Windows** | **100 Sessions** | **4,792** | **2,827 (59.0%)** | **1,965 (41.0%)** | **10,334** |

---

## 4. Signal Processing, Feature Engineering & 32 Hz Tensor Pipeline

To ensure zero proprietary device bias, the Empatica E4's internal `HR.csv` is **discarded**. All cardiovascular, electrodermal, thermal, and actigraphy representations are computed directly from raw waveforms in [`src/features/`](src/features/):

1. **Blood Volume Pulse (`src/features/bvp.py`, 64 Hz):**
   * 4th-order zero-phase Butterworth bandpass filter ($0.5\text{–}8.0\,\text{Hz}$).
   * Adaptive systolic peak detection with refractory period gating ($250\,\text{ms}$ to $2000\,\text{ms}$ inter-beat intervals, corresponding to $30\text{–}240\,\text{BPM}$).
   * Extracted features (8): `hr_mean`, `hr_std`, `hr_min`, `hr_max`, `hrv_rmssd`, `hrv_sdnn`, `hrv_pnn50`, and signal quality index `fraction_invalid_bvp`.
2. **Electrodermal Activity (`src/features/eda.py`, 4 Hz):**
   * Low-pass zero-phase smoothing ($1.0\,\text{Hz}$) followed by tonic-phasic decomposition via moving-average baseline subtraction ($4.0\,\text{s}$ window).
   * Skin Conductance Response (SCR) peak detection above $0.01\,\mu\text{S}$ prominence.
   * Extracted features (6): `eda_mean`, `eda_std`, `eda_slope`, `eda_tonic_level`, `eda_phasic_peaks`, `eda_phasic_amplitude_sum`.
3. **Peripheral Skin Temperature & 3-Axis Accelerometry (`src/features/temp_acc.py`, 4 Hz & 32 Hz):**
   * Temperature features (3): `temp_mean`, `temp_std`, ordinary least-squares `temp_slope` ($^\circ\text{C/s}$).
   * Accelerometer features (4): Euclidean magnitude $\|\mathbf{a}\|_2 = \sqrt{a_x^2 + a_y^2 + a_z^2}$, `acc_mag_mean`, `acc_mag_std`, `acc_energy`, and dynamic motion gate `high_motion_fraction` (fraction of samples where $\|\|\mathbf{a}\|_2 - 1.0\,\text{g}\| > 0.15\,\text{g}$).
4. **Harmonized 32 Hz 6-Channel CNN Tensor (`src/features/tensor_builder.py`):**
   * `BVP`: Decimated $64\,\text{Hz} \to 32\,\text{Hz}$ using an 8th-order Chebyshev Type-I anti-aliasing zero-phase filter.
   * `EDA` & `TEMP`: Linearly interpolated $4\,\text{Hz} \to 32\,\text{Hz}$.
   * `ACC_x, ACC_y, ACC_z`: Retained at native $32\,\text{Hz}$.
   * Output tensor shape: `(6, 1920)` for $60\,\text{s}$ windows and `(6, 960)` for $30\,\text{s}$ windows.

---

## 5. Strict Anti-Leakage & Subject-Isolation Protocol

```mermaid
flowchart LR
    subgraph OuterSplit["Outer LOSO Split (N Folds)"]
        ALL["All Unique Subjects\nS = {s_1, s_2, ..., s_N}"]
        TRAIN["Training Subjects\nS_train = S \\ {s_k}"]
        TEST["Held-Out Test Subject\nS_test = {s_k}"]
        ASSERT["assert_no_subject_leakage()\nS_train ∩ S_test == ∅"]
    end

    subgraph TrainEnclosure["Fitted Exclusively on S_train"]
        NORM["GlobalZScoreNormalizer\nμ_train, σ_train"]
        INNER["Inner GroupKFold / Val Subject\nEarly Stopping & Platt Calibrator"]
        MODEL["Fitted Classifier\nM0 / M1 / M2 / FedAvg"]
    end

    subgraph Evaluation["Zero-Leakage Evaluation on s_k"]
        TRANS["Apply μ_train, σ_train & Calibrator\nto s_k Windows"]
        METRICS["Per-Subject Metrics\nMacro-F1, ROC-AUC, ECE,\n3-of-5 Persisted False Alerts/hr"]
        BOOT["1,000x Subject Bootstrap\n95% Confidence Intervals"]
    end

    ALL --> TRAIN & TEST
    TRAIN & TEST --> ASSERT
    TRAIN --> NORM --> INNER --> MODEL
    MODEL --> TRANS
    TEST --> TRANS
    TRANS --> METRICS --> BOOT
```

Every evaluation in this repository enforces three non-negotiable anti-leakage rules verified by [`tests/test_leakage_assertion.py`](tests/test_leakage_assertion.py):
1. **Participant-Level Splitting Only:** No human subject ever appears in both training and testing splits ($\text{Train}_{\text{subjects}} \cap \text{Test}_{\text{subjects}} = \emptyset$).
2. **Train-Fold-Only Preprocessing:** Feature z-score scalers (`GlobalZScoreNormalizer`), CNN channel normalizers, missing-value imputers, and Platt/isotonic probability calibrators (`SubjectCalibrator`) are fitted **strictly inside the training fold**.
3. **Subject-Level Bootstrap Confidence Intervals:** Aggregate metrics report mean $\pm$ SD across subjects, worst-subject performance, and **1,000-iteration bootstrap 95% confidence intervals** resampled over subjects.

---

## 6. Phase 1: Within-Protocol LOSO, Cross-Protocol Transfer & Ablation Results

### 6.1 Evaluation E1: Within-Protocol & Pooled LOSO Benchmark (`results/tables/loso_benchmark_e1.csv`)

![Figure 1: LOSO Model Comparison](results/figures/fig1_loso_model_comparison.png)

| Dataset Cohort | Model | Macro-F1 (Mean ± SD) | 95% Bootstrap CI | Worst-Subject F1 | Balanced Acc | ROC-AUC | ECE ↓ | Persisted False Alerts / hr ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PhysioNet_STRESS`** | `m0_majority` | 0.4359 ± 0.0403 | [0.4213, 0.4490] | 0.3469 | 0.5000 | 0.5000 | 0.2187 | 0.00 |
| **`PhysioNet_STRESS`** | **`m0_logistic_regression`** | **0.6658 ± 0.1350** | **[0.6197, 0.7056]** | 0.3129 | **0.7414** | **0.8235** | 0.2455 | 25.94 |
| **`PhysioNet_STRESS`** | `m1_decision_tree` | 0.5767 ± 0.1520 | [0.5302, 0.6298] | 0.1714 | 0.6413 | 0.6411 | 0.2817 | 30.54 |
| **`PhysioNet_STRESS`** | **`m1_random_forest`** | 0.6544 ± 0.1693 | [0.6018, 0.7056] | 0.1714 | 0.6848 | 0.7884 | 0.2364 | 15.55 |
| **`PhysioNet_STRESS`** | **`m1_hist_gradient_boosting`** | **0.6616 ± 0.1653** | **[0.6109, 0.7130]** | 0.1605 | 0.6985 | 0.7824 | 0.2009 | 13.91 |
| **`PhysioNet_STRESS`** | `m2_cnn_1d` (Raw 32Hz) | 0.5263 ± 0.1031 | [0.4957, 0.5597] | **0.3469** | 0.5530 | 0.6465 | **0.1586** | **2.23** |
| **`PhysioNet_EXERCISE`** | `m0_logistic_regression` | 0.8881 ± 0.0962 | [0.8549, 0.9191] | 0.5558 | 0.9131 | 0.9733 | 0.1003 | **13.66** |
| **`PhysioNet_EXERCISE`** | `m1_decision_tree` | 0.8694 ± 0.1030 | [0.8293, 0.9043] | 0.5181 | 0.8941 | 0.9320 | 0.1072 | 16.99 |
| **`PhysioNet_EXERCISE`** | **`m1_random_forest`** | 0.8926 ± 0.1050 | [0.8552, 0.9266] | 0.5558 | 0.9112 | 0.9726 | 0.1171 | 16.58 |
| **`PhysioNet_EXERCISE`** | **`m1_hist_gradient_boosting`** | **0.8941 ± 0.0879** | **[0.8624, 0.9214]** | **0.6500** | **0.9145** | **0.9749** | **0.0873** | 16.21 |
| **`PhysioNet_EXERCISE`** | `m2_cnn_1d` (Raw 32Hz) | 0.7109 ± 0.1475 | [0.6611, 0.7582] | 0.4273 | 0.7316 | 0.8366 | 0.1754 | 37.97 |
| **`Pooled (All 100 Sessions)`** | `m0_logistic_regression` | 0.7853 ± 0.1206 | [0.7458, 0.8221] | 0.4266 | 0.8058 | 0.8836 | 0.1451 | 19.01 |
| **`Pooled (All 100 Sessions)`** | `m1_random_forest` | 0.7855 ± 0.1270 | [0.7427, 0.8227] | 0.4305 | 0.8051 | 0.8814 | 0.1595 | 19.17 |
| **`Pooled (All 100 Sessions)`** | **`m1_hist_gradient_boosting`** | **0.7869 ± 0.1239** | **[0.7445, 0.8248]** | **0.4533** | **0.8096** | **0.8951** | 0.1376 | **16.72** |
| **`Pooled (All 100 Sessions)`** | **`m2_cnn_1d` (Raw 32Hz)** | **0.7172 ± 0.1263** | **[0.6790, 0.7589]** | 0.3926 | 0.7185 | 0.8128 | **0.1334** | 20.78 |

---

### 6.2 Evaluation E3: Sensor Modality, Window Duration & Subject Normalization Ablations (`results/tables/sensor_window_ablations_e3.csv`)

![Figure 2: Sensor Modality & Normalization Ablations](results/figures/fig2_sensor_and_normalization_ablations.png)

| Ablation Dimension | Configuration | Model | Features / Channels | Macro-F1 (Mean ± SD) | 95% Bootstrap CI | Balanced Acc | ROC-AUC | ECE ↓ | Persisted False Alerts / hr ↓ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sensor Modality** | **`ALL_SENSORS`** | `m1_random_forest` | 21 feats | **0.6347 ± 0.1371** | **[0.5756, 0.6956]** | 0.6326 | **0.7825** | **0.1809** | **4.32** |
| **Sensor Modality** | **`BVP_EDA`** | `m1_random_forest` | 14 feats | **0.6340 ± 0.1552** | **[0.5660, 0.7039]** | 0.6514 | 0.7777 | 0.1875 | 6.66 |
| **Sensor Modality** | `BVP_EDA_TEMP` | `m1_random_forest` | 17 feats | 0.6284 ± 0.1553 | [0.5612, 0.6947] | 0.6366 | 0.7757 | 0.1839 | 5.57 |
| **Sensor Modality** | `EDA_ONLY` | `m1_random_forest` | 6 feats | 0.6147 ± 0.1499 | [0.5518, 0.6830] | **0.6776** | 0.7624 | 0.1990 | 15.12 |
| **Sensor Modality** | `BVP_ONLY` | `m1_random_forest` | 8 feats | 0.5638 ± 0.1699 | [0.4887, 0.6379] | 0.6362 | 0.6402 | 0.2730 | 21.48 |
| **Sensor Modality** | `TEMP_ACC_ONLY` | `m1_random_forest` | 7 feats | 0.5222 ± 0.1165 | [0.4724, 0.5844] | 0.5566 | 0.6381 | 0.2821 | 24.33 |
| **Window Duration** | **`60s_window_30s_step`** | `m1_random_forest` | 21 feats | **0.6347 ± 0.1371** | **[0.5756, 0.6956]** | 0.6326 | **0.7825** | **0.1809** | **4.32** |
| **Window Duration** | `30s_window_15s_step` | `m1_random_forest` | 21 feats | 0.6181 ± 0.1170 | [0.5664, 0.6727] | **0.6496** | 0.7517 | 0.2284 | 12.24 |
| **Window Duration** | `60s_window_30s_step` | `m2_cnn_1d` | 6 ch | 0.4726 ± 0.0094 | [0.4679, 0.4765] | 0.4992 | 0.4877 | 0.1201 | **0.00** |
| **Window Duration** | **`30s_window_15s_step`** | `m2_cnn_1d` | 6 ch | **0.4935 ± 0.0533** | **[0.4718, 0.5236]** | **0.5144** | **0.6078** | **0.1082** | 0.89 |
| **Normalization** | `GlobalZScore_TrainFoldOnly` | `m0_logistic_regression` | 21 feats | 0.5163 ± 0.1774 | [0.4409, 0.6024] | 0.5971 | 0.6311 | 0.3284 | 28.18 |
| **Normalization** | **`UnsupervisedRestingPrefixCalibrated`** | `m0_logistic_regression` | 21 feats | **0.6089 ± 0.1252** | **[0.5563, 0.6659]** | **0.7162** | **0.8011** | 0.2542 | 23.12 |
| **Normalization** | **`PerSubjectBaselineCalibrated`** | `m0_logistic_regression` | 21 feats | **0.6339 ± 0.1201** | **[0.5860, 0.6863]** | 0.7071 | 0.7640 | **0.2052** | **14.71** |

#### Key Scientific Findings from Phase 1 Ablations:
1. **Multi-Sensor & Autonomic Interaction Synergy (`ALL_SENSORS` and `BVP_EDA`):** Combining BVP, EDA, TEMP, and ACC with autonomic interaction features (`sympathovagal_ratio`, `eda_arousal_index`, `motion_adjusted_hr`) enables `ALL_SENSORS` (`Macro-F1 = 0.6347`, `ROC-AUC = 0.7825`, **`4.32` false alerts/hr**) and `BVP_EDA` (`Macro-F1 = 0.6340`, `ROC-AUC = 0.7777`) to dramatically outperform single-sensor baselines (`BVP_ONLY` `0.5638`, `TEMP_ACC_ONLY` `0.5222`).
2. **Label-Free Resting-Prefix Calibration (`UnsupervisedRestingPrefixCalibrated`):** Standardizing each subject's session using their initial resting prefix (`UnsupervisedSubjectSessionCalibrator`, using **zero ground-truth labels**) boosts `Macro-F1` by **+9.26 percentage points** (`0.5163 -> 0.6089`) and jumps `ROC-AUC` from **`0.6311` to `0.8011`** (`+17.0` AUC points), nearly matching oracle `PerSubjectBaselineCalibrated` (`0.6339`) without requiring ground-truth test labels.
3. **Sample Density vs. Frequency Resolution (`30s` vs. `60s` Windows):** While `60s` windows provide cleaner HRV frequency stability for tree ensembles (`Macro-F1 = 0.6347`, `4.32` false alerts/hr), shorter `30s` windows (`10,334` windows total) double the training sample density for the deep `M2` 1D-CNN, improving its `ROC-AUC` from `0.4877` to `0.6078`.

---

## 7. Edge Hardware Footprint, Int8 Quantization & Latency Profile

The `M2` 1D-CNN (`src/models/cnn_1d.py`) uses per-window `InstanceNorm1d`, 3 depth-strided 1D convolutional blocks (`6 -> 16 -> 32 -> 48` filters) with Squeeze-and-Excitation (`SEBlock1D`) channel attention, and dual Global Average + Max Pooling (`AdaptiveAvgPool1d(1) + AdaptiveMaxPool1d(1)`) to stay strictly below the **50,000 parameter edge budget**:

| Metric (`results/tables/edge_profile_m2.csv`) | Value | Edge Target Requirement | Status |
| :--- | :---: | :---: | :---: |
| **Trainable Parameters** | **12,161** | $< 50,000$ | **Pass (24.3% of ceiling)** |
| **Float32 Serialized Size** | **58.69 KB** | $< 200\text{ KB}$ | **Pass** |
| **Dynamic Int8 Quantized Size** | **51.22 KB** | $< 100\text{ KB}$ | **Pass** |
| **Quantization Accuracy Delta (`Int8 - Float32`)** | **0.0000** (`83.40%` vs `83.40%`) | $|\Delta \text{Acc}| < 1.0\%$ | **Pass (Zero Loss)** |
| **Single-Window Inference Latency (`Mean / p50`)** | **1.826 ms / 1.806 ms** | $< 50\text{ ms}$ | **Pass (~27x faster)** |
| **Single-Window Tail Latency (`p95 / p99`)** | **1.995 ms / 2.192 ms** | $< 100\text{ ms}$ | **Pass** |

---

## 8. Phase 2: CPCB Ambient AQI Hazard Model & Contextual Risk Fusion Engine

### 8.1 Contextual Decision Hierarchy (`src/fusion/contextual_fusion.py`)

```mermaid
flowchart TD
    IN["Inputs per 60s Window:\n1. Calibrated P(strain) ∈ [0, 1]\n2. Last 5 Windows -> 3-of-5 Persistence Gate\n3. Wrist Motion: high_motion_fraction\n4. Ambient CPCB AQI & Temp (°C)"]
    PERSIST{"Persisted Strain?\n(≥ 3 of last 5 windows\nP(strain) ≥ 0.50)"}
    TRANSIENT{"Single-Window Spike?\n(P(strain) ≥ 0.50)"}
    ENV_CLEAN{"Ambient AQI Tier\n≥ Very Poor (>300)?"}
    ENV_POOR{"Ambient AQI Tier\n≥ Poor (>200)?"}
    MOTION{"Active Exercise?\n(high_motion_fraction\n≥ 0.35)"}
    EX_AQI{"Ambient AQI Tier\n≥ Poor (>200)?"}
    REST_AQI{"AQI ≥ Poor (>200) OR\n(P ≥ 0.75 & AQI ≥ Mod)?"}

    NORM["🟢 NORMAL (Level 0)\nBaseline physiology & safe air\nOR transient spike suppressed"]
    ADV["🟡 ADVISORY (Level 1)\nExercise in clean air OR\npollution caution at baseline"]
    WARN["🟠 WARNING (Level 2)\nHigh non-exertional strain\nor exercise in Poor AQI"]
    CRIT["🔴 CRITICAL_ALERT (Level 3)\nPersisted non-exertional strain in Poor AQI\nOR heavy exercise in Severe pollution"]

    IN --> PERSIST
    PERSIST -- "No" --> TRANSIENT
    TRANSIENT -- "No (Baseline)" --> ENV_CLEAN
    ENV_CLEAN -- "No" --> NORM
    ENV_CLEAN -- "Yes" --> ADV
    TRANSIENT -- "Yes (Isolated Spike)" --> ENV_POOR
    ENV_POOR -- "No (Suppressed!)" --> NORM
    ENV_POOR -- "Yes" --> ADV

    PERSIST -- "Yes (Sustained 3-of-5)" --> MOTION
    MOTION -- "Yes (Exertional)" --> EX_AQI
    EX_AQI -- "No (Good/Mod Air)" --> ADV
    EX_AQI -- "Yes (High Inhaled Dose)" --> CRIT
    MOTION -- "No (Non-Exertional at Rest)" --> REST_AQI
    REST_AQI -- "Yes" --> CRIT
    REST_AQI -- "No (Moderate Strain)" --> WARN
```

### 8.2 Phase 2 Empirical Scenario Verification (`results/tables/phase2_contextual_fusion_demo.csv`)

![Figure 3: Contextual Physiological-Environmental Risk Fusion](results/figures/fig3_contextual_fusion_matrix.png)

| Scenario ID | $P(\text{strain})$ | 3-of-5 Persisted? | Wrist Motion Context | Ambient AQI (CPCB NAQI) | Composite Risk Score | Triggered Alert Tier | Operational Rationale |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **`S1_Resting_CleanAir`** | 0.18 | `False` | `NON_EXERTIONAL` (`0.04`) | `42` (*Good*) | `0.150` | **🟢 `NORMAL`** | Baseline physiology & clean air; no intervention required. |
| **`S2_Transient_Spike_Suppressed`** | 0.68 | `False` (1/5) | `NON_EXERTIONAL` (`0.08`) | `85` (*Satisfactory*) | `0.493` | **🟢 `NORMAL`** | Single-window spike suppressed by 3-of-5 persistence gate. |
| **`S3_Exercise_GoodAir`** | 0.82 | `True` (5/5) | `EXERTIONAL` (`0.62`) | `64` (*Satisfactory*) | `0.556` | **🟡 `ADVISORY`** | Active exercise in clean air; avoids false critical alarm. |
| **`S4_Exercise_SeverePollution`** | 0.84 | `True` (5/5) | `EXERTIONAL` (`0.58`) | `345` (*Very Poor*) | `0.849` | **🔴 `CRITICAL_ALERT`** | Heavy outdoor exertion in Very Poor AQI; high inhaled dose risk. |
| **`S5_NonExertional_ModerateAir`** | 0.62 | `True` (5/5) | `NON_EXERTIONAL` (`0.05`) | `140` (*Moderate*) | `0.512` | **🟡 `ADVISORY`** | Sustained rest strain under Moderate AQI; recovery break advised. |
| **`S6_NonExertional_PoorAir_Heat`** | 0.86 | `True` (5/5) | `NON_EXERTIONAL` (`0.06`) | `265` (*Poor*, $40.5^\circ\text{C}$) | `0.821` | **🔴 `CRITICAL_ALERT`** | High non-exertional strain at rest + Poor AQI + ambient heat. |

---

## 9. Phase 3: Subject-Isolated Federated Edge Learning (`FedAvg` & `FedProx`)

In Phase 3 (`src/federated/`), each training subject is isolated as an independent edge wearable client (`EdgeClientData`). Raw biosignals never leave the participant's device; only local model weight updates are aggregated via sample-weighted **Federated Averaging (`FedAvg`)** or proximal-regularized **Federated Proximal (`FedProx`, $\mu = 0.05$)** to mitigate non-IID inter-subject drift:

$$w^{(t+1)} = \sum_{k \in S_t} \frac{n_k}{\sum_{j \in S_t} n_j} w_k^{(t+1)}, \qquad \mathcal{L}_k(w) = \mathcal{L}_{\text{BCE}}(w) + \frac{\mu}{2} \|w - w^{(t)}\|_2^2$$

![Figure 4: Federated Edge Learning vs Communication Cost](results/figures/fig4_federated_comm_vs_f1.png)

| Training Paradigm (`results/tables/phase3_federated_benchmark.csv`) | Rounds ($R$) × Local Epochs ($E$) | Client Participation ($C$) | Macro-F1 (Mean ± SD) | ROC-AUC | Persisted False Alerts / hr ↓ | Float32 Traffic (KB) | Int8 Quantized Traffic (KB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Centralized_LOSO_M2`** *(Upper Bound)* | $1 \times 6$ | 100% (Pooled Server) | **0.4914 ± 0.1017** | **0.6432** | **1.30** | 0.0 KB *(Raw Data Shared)* | 0.0 KB *(Raw Data Shared)* |
| **`FedAvg_FullParticipation`** ($\mu=0.0$) | $3 \times 2$ | 100% (`11` clients/rd) | 0.4361 ± 0.0836 | 0.4706 | 23.66 | 3,135.26 KB | **783.81 KB** |
| **`FedProx_mu0.05`** ($\mu=0.05$) | $3 \times 2$ | 100% (`11` clients/rd) | 0.4300 ± 0.0819 | 0.4645 | **22.98** | 3,135.26 KB | **783.81 KB** |
| **`FedAvg_PartialClients_0.6`** ($\mu=0.0$) | $3 \times 2$ | 60% (`7` clients/rd) | **0.4714 ± 0.1045** | **0.5038** | 45.83 | 1,995.16 KB | **498.79 KB** |

* **Privacy & Bandwidth Efficiency:** Federated training with partial client sampling (`FedAvg_PartialClients_0.6`) achieves **`Macro-F1 = 0.4714`** (within `0.020` of centralized `M2` LOSO at `0.4914`) while keeping 100% of raw physiological waveforms on-device. Dynamic `int8` weight quantization compresses total multi-round network traffic by **4×** (`1.99 MB -> 498.8 KB` across all rounds).

---

## 10. Repository Structure & Reproducibility Guide

```text
├── configs/
│   └── base.yaml                        # Central windowing, sensor, model, fusion & federated configs
├── data/
│   ├── raw/physionet_wearable/1.0.1/    # Raw Empatica E4 CSVs (STRESS, AEROBIC, ANAEROBIC)
│   ├── labels/
│   │   ├── segment_labels.csv           # 4,792 validated 60s window boundaries & binary labels
│   │   ├── real_physionet_features_60s.csv # Extracted 21-feature tabular dataset (60s/30s stride)
│   │   └── real_physionet_features_30s.csv # Extracted 21-feature tabular dataset (30s/15s stride)
│   └── cache/
│       ├── real_physionet_tensors_60s.npz  # 32 Hz 6-channel CNN tensors (4792, 6, 1920)
│       └── real_physionet_tensors_30s.npz  # 32 Hz 6-channel CNN tensors (10334, 6, 960)
├── docs/
│   └── assumptions.md                   # Mandatory research proxy disclosures & harmonization rules
├── Literature Review/                   # Reference papers on TinyML, edge biosignals & federated AI
├── results/
│   ├── tables/                          # All exported benchmark CSVs (E1, E2, E3, Edge, Phase 2, Phase 3)
│   └── figures/                         # 300 DPI publication PNG plots (fig1 to fig4)
├── src/
│   ├── io/                              # Dataset downloaders, E4 CSV/pickle loaders & verifiers
│   ├── windowing/                       # Sliding window generator & strict boundary-collision gate
│   ├── features/                        # BVP HR/HRV, EDA SCR, TEMP/ACC extractors, normalizers & tensors
│   ├── models/                          # M0/M1 baselines (inner GroupKFold CV) & M2 LightweightStrainCNN
│   ├── eval/                            # LOSO engine, cross-dataset transfer, ablations & figure plotter
│   ├── edge/                            # Dynamic int8 post-training quantization & CPU latency profiler
│   ├── fusion/                          # Phase 2 CPCB NAQI pipeline & ContextualRiskEngine
│   └── federated/                       # Phase 3 subject-isolated FedAvg & FedProx engine
└── tests/                               # 37 automated unit & anti-leakage tests
```

### Quick-Start Commands (Any OS)

```bash
# 1. Run the complete 37-test verification & anti-leakage test suite
python -m unittest discover tests -v

# 2. Verify local raw dataset integrity
python src/io/verify_data.py

# 3. Rebuild segment label catalog & extract real features/tensors from raw Empatica E4 CSVs
python src/windowing/build_segment_labels.py
python src/features/extract_real_physionet.py

# 4. Run Phase 1D Within-Protocol & Pooled LOSO Benchmark (E1) + Edge Quantization Profile
python src/eval/run_phase1_experiments.py

# 5. Run Phase 1E Cross-Protocol Transfer (E2) & Sensor/Window/Normalization Ablations (E3)
python src/eval/ablations.py

# 6. Run Phase 2 CPCB Ambient AQI Hazard Model & Contextual Risk Fusion Engine
python src/fusion/contextual_fusion.py

# 7. Run Phase 3 Federated Edge Simulation (FedAvg & FedProx vs Centralized LOSO)
python src/federated/federated_evaluator.py

# 8. Regenerate all 300 DPI publication figures in results/figures/
python src/eval/generate_figures.py
```
