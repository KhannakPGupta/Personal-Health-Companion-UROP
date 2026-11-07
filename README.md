# Edge-Intelligent Personal Health Companion (UROP Research Benchmark)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests: 37 Passed](https://img.shields.io/badge/tests-37%20passed-brightgreen.svg)](#reproducibility--verification)
[![Dataset: PhysioNet E4](https://img.shields.io/badge/PhysioNet-37%20Subjects%20%7C%2068%20Sessions-006699.svg)](https://physionet.org/content/wearable-device-dataset/1.0.0/)
[![Dataset: CPCB India AQI](https://img.shields.io/badge/CPCB%20India%20AQI-26%20Cities%20%7C%2029,531%20Days-00897b.svg)](#phase-2-next-day-t1-cpcb-india-aqi-forecasting--contextual-fusion)

A reproducible, subject-independent **Edge-AI + Causal Next-Day Environmental Hazard Forecasting + Privacy-Preserving Federated Learning** research benchmark for physiological strain detection on wrist-worn wearables (Empatica E4: `BVP 64 Hz`, `EDA 4 Hz`, `TEMP 4 Hz`, `ACC 32 Hz`).

> **Scope & Proxy Disclaimer**: Wearable models in this repository detect **acute autonomic and cardiorespiratory strain proxies** (cognitive/exam stress and aerobic/anaerobic physical exertion) rather than clinical heat-stroke or flood casualties. Hydrological/flood hazards are not modeled.

---

## Table of Contents
1. [Visual System Architecture & End-to-End Dataflow](#1-visual-system-architecture--end-to-end-dataflow)
2. [Signal Processing, 26-Feature Pipeline & Calibration Architecture](#2-signal-processing-26-feature-pipeline--calibration-architecture)
3. [Evaluation E1: Within-Dataset Subject-Independent LOSO Benchmark](#3-evaluation-e1-within-dataset-subject-independent-loso-benchmark)
4. [Evaluation E2: Cross-Protocol Domain Shift & Transfer Failure Analysis](#4-evaluation-e2-cross-protocol-domain-shift--transfer-failure-analysis)
5. [Evaluation E3: Sensor, Feature (21 vs 26), Window & Calibration Ablations](#5-evaluation-e3-sensor-feature-21-vs-26-window--calibration-ablations)
6. [Phase 1D: Multi-Model Edge Deployment & Int8 Quantization Profile](#6-phase-1d-multi-model-edge-deployment--int8-quantization-profile)
7. [Phase 2: Next-Day (`t+1`) CPCB India AQI Forecasting & Contextual Fusion](#7-phase-2-next-day-t1-cpcb-india-aqi-forecasting--contextual-fusion)
8. [Phase 3: Privacy-Preserving Federated Learning Benchmark](#8-phase-3-privacy-preserving-federated-learning-benchmark)
9. [Repository Structure & Reproducibility Commands](#9-repository-structure--reproducibility-commands)

---

## 1. Visual System Architecture & End-to-End Dataflow

![Three-Phase System Architecture Overview](results/figures/fig7_system_architecture_overview.png)

### Interactive System Architecture Flowchart

```mermaid
flowchart LR
    subgraph S1["Phase 1: Wrist Wearable Edge Pipeline (Empatica E4)"]
        direction TB
        W["Raw E4 Streams\n• BVP (64 Hz)\n• EDA (4 Hz)\n• TEMP (4 Hz)\n• ACC (32 Hz)"] --> SL["Sliding Window Slicer\n• 60s Window, 15s Stride\n• ±30s Transition Drop"]
        SL --> FE["Tabular Feature Engine\n• 21 Base Time/Freq Features\n• 5 Autonomic Interaction Ratios\n• Resting-Prefix Z-Score Calib."]
        SL --> TB["Raw 6-Channel Tensor\n• Resampled to 32 Hz\n• Shape: (6, 1920)"]
        FE --> M01["Edge Tabular Classifiers\n• M0 LogReg (0.05 KB int8)\n• M1 HistGB (103.6 KB int8)\n• M1 RandForest (309.1 KB int8)"]
        TB --> M2["Compact 1D-CNN (GroupNorm)\n• 12,161 Params\n• 12.88 KB (int8, 4.29x)"]
    end

    subgraph S3["Phase 3: Federated Learning (PhysioNet_STRESS)"]
        direction TB
        FLC["35 Client Subjects / Round\n(Local SGD on Private Windows)"] <-->|"Int8 Weight Updates\n(8.86 - 14.77 KB Total)"| FLA["FedAvg / FedProx Aggregator\n(Macro-F1 = 0.6588, AUC = 0.8062)"]
    end

    subgraph S2["Phase 2: Next-Day (t+1) CPCB AQI + Heat Fusion"]
        direction TB
        CPCB["Real CPCB India city_day.csv\n• 26 Cities, 29,531 Daily Rows\n• Day-t Pollutants + 3-Day Lags"] --> AQI["Next-Day Hazard Forecaster\n• Target: AQI_{t+1} > 200\n• Test Macro-F1 = 0.9094\n• Test ROC-AUC = 0.9808"]
        TEMP_AMB["Ambient Meteorology\n• High Heat Rule: T >= 38.0°C"] --> FUS["Contextual Risk Engine\n• K=2 Persistence Hysteresis\n• ACC Motion Context Gating\n• 4-Tier Alert Escalation"]
        AQI --> FUS
    end

    M01 -->|"p_phys(t)"| FUS
    M2 -->|"p_phys(t)"| FUS
    FLA -.->|"Global Weights"| M01
```

---

## 2. Signal Processing, 26-Feature Pipeline & Calibration Architecture

Every 60-second window (with 15-second stride and $\pm 30\text{ s}$ protocol transition exclusion buffer) is processed along two parallel representations:
1. **26 Hand-Crafted Physiological Features (`M0` & `M1` Models)**: `21` base single-sensor features + `5` cross-sensor autonomic interaction ratios (`src/features/extractor.py`).
2. **6-Channel Raw Temporal Tensors (`M2 1D-CNN`)**: `(BVP, EDA, TEMP, ACC_x, ACC_y, ACC_z)` interpolated to a common `32 Hz` grid (`(6, 1920)` per 60s window) normalized via channel-wise `GroupNorm` (`src/models/cnn_1d.py`).

```mermaid
flowchart TD
    subgraph RAW["Multi-Rate Empatica E4 Sensor Streams (60s Window, 15s Stride)"]
        BVP["BVP (64 Hz, 3840 samples)\nBandpass + Peak Detection"]
        EDA["EDA (4 Hz, 240 samples)\nTonic SCL / Phasic SCR Split"]
        TEMP["TEMP (4 Hz, 240 samples)\nPeripheral Skin Thermometry"]
        ACC["ACC 3-Axis (32 Hz, 1920x3)\nDynamic ENMO & Jerk"]
    end

    BVP --> F_BVP["6 BVP/HRV Features\n• hr_mean, hr_std\n• ibi_mean, ibi_rmssd, ibi_sdnn\n• bvp_peak_to_peak"]
    EDA --> F_EDA["5 Electrodermal Features\n• eda_mean, eda_std, eda_deriv_std\n• eda_scr_count, eda_tonic_slope"]
    TEMP --> F_TEMP["4 Thermal Features\n• temp_mean, temp_std\n• temp_slope, temp_min_max_range"]
    ACC --> F_ACC["6 Kinematic Features\n• acc_mag_mean, acc_mag_std\n• acc_enmo, acc_jerk_mean\n• acc_x_mean, acc_y_mean"]

    F_BVP & F_EDA & F_TEMP & F_ACC --> BASE21["21 Base Modality Features\n(ALL_SENSORS_21_BASE: Stress F1 = 0.6572)"]
    BASE21 --> INT5["+ 5 Cross-Sensor Autonomic Ratios\n22. hr_per_enmo (Cardio-Kinematic Ratio)\n23. eda_per_enmo (Sudomotor-Kinematic Ratio)\n24. sympathetic_index (HR / RMSSD)\n25. electrodermal_reactivity (SCR * Slope)\n26. thermal_cardiac_coupling (HR * TempSlope)"]
    INT5 --> FULL26["26 Full Tabular Features\n(ALL_SENSORS: Stress F1 = 0.6544)"]

    FULL26 --> CALIB{"Subject Calibration Mode\n(src/features/normalizer.py)"}
    CALIB -->|"Default Deployment"| C_PRE["UnsupervisedRestingPrefixCalibrated\n(Uses first 15% of session windows;\nAssumes initial rest -> F1 = 0.6544)"]
    CALIB -->|"Unnormalized"| C_RAW["RawUnnormalized\n(No subject baseline -> F1 = 0.3887)"]
    CALIB -->|"Theoretical Upper Bound"| C_ORA["PerSubjectBaseline_Oracle\n(Uses true y=0 windows -> F1 = 0.7025)"]
```

---

## 3. Evaluation E1: Within-Dataset Subject-Independent LOSO Benchmark

We evaluate **all 37 real human subjects** (`68` subject-protocol sessions across `PhysioNet_STRESS` [`n=36`], `PhysioNet_EXERCISE` [`n=32`], and `Pooled` [`n=37`]) from the **PhysioNet Wearable Exam Stress & Exercise Dataset v1.0.0** ([DOI: 10.13026/kvf3-0c87](https://physionet.org/content/wearable-device-dataset/1.0.0/)).

![Evaluation E1: LOSO Model Comparison](results/figures/fig1_loso_model_comparison.png)

### Zero-Leakage LOSO & Persisted Alert Evaluation Flow

```mermaid
sequenceDiagram
    participant Data as 37 Real Subjects (68 Sessions)
    participant Split as LOSO Splitter (assert_no_subject_leakage)
    participant Calib as Resting-Prefix Normalizer
    participant Model as Edge Classifier (M0 / M1 / M2)
    participant Persist as K=2 Hysteresis Filter

    loop For each held-out test subject s in 1..N
        Data->>Split: Partition Train (N-1 subjects) vs Test (Subject s)
        Note over Split: Strict assertion: Train_IDs ∩ Test_IDs = ∅
        Split->>Calib: Fit global fallback stats on Train only
        Calib->>Calib: Calibrate Subject s using initial 15% resting prefix
        Calib->>Model: Fit on Train windows + Platt Scaling on 15% Train-Val subjects
        Model->>Persist: Predict chronological probabilities p_t on Subject s
        Persist->>Persist: Trigger persisted alarm a_t=1 iff K=2 consecutive windows >= θ
    end
```

### E1 Summary Table (`results/tables/loso_benchmark_e1.csv`)

| Cohort | Subjects (`n`) | Median (`Min`) Win/Sub | Model | Features | Macro-F1 (Mean ± SD) | 95% Bootstrap CI | Balanced Acc | ROC-AUC | ECE | Persisted FA Win/hr | FA Onset Episodes/hr |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PhysioNet_STRESS** | 36 | 53 (`31`) | **M0 Logistic Regression** | 26 | **0.6658 ± 0.1350** | `[0.6197, 0.7056]` | **0.7414** | **0.8235** | 0.2455 | 25.94 | 3.89 |
| **PhysioNet_STRESS** | 36 | 53 (`31`) | **M1 HistGradientBoosting** | 26 | **0.6616 ± 0.1653** | `[0.6061, 0.7131]` | 0.6985 | 0.7824 | 0.2602 | 13.91 | 2.09 |
| **PhysioNet_STRESS** | 36 | 53 (`31`) | **M1 Random Forest** | 26 | **0.6544 ± 0.1693** | `[0.5973, 0.7073]` | 0.6848 | 0.7884 | 0.2336 | 15.55 | 2.16 |
| **PhysioNet_STRESS** | 36 | 53 (`31`) | M1 Decision Tree | 26 | 0.5944 ± 0.1636 | `[0.5382, 0.6461]` | 0.6305 | 0.6775 | 0.2736 | 21.62 | 4.65 |
| **PhysioNet_STRESS** | 36 | 53 (`31`) | M2 Compact 1D-CNN (GroupNorm) | 6 ch | 0.5337 ± 0.0956 | `[0.5030, 0.5653]` | 0.5602 | 0.6900 | 0.1932 | 2.21 | 0.45 |
| **PhysioNet_STRESS** | 36 | 53 (`31`) | M0 Majority Dummy | 26 | 0.4359 ± 0.0403 | `[0.4213, 0.4490]` | 0.5000 | 0.5000 | 0.2187 | 0.00 | 0.00 |
| **PhysioNet_EXERCISE** | 32 | 59 (`13`) | **M1 HistGradientBoosting** | 26 | **0.8941 ± 0.0879** | `[0.8620, 0.9227]` | **0.9145** | **0.9749** | 0.0650 | 16.21 | 2.06 |
| **PhysioNet_EXERCISE** | 32 | 59 (`13`) | **M1 Random Forest** | 26 | **0.8926 ± 0.1050** | `[0.8535, 0.9248]` | 0.9112 | 0.9726 | 0.0725 | 15.92 | 2.07 |
| **PhysioNet_EXERCISE** | 32 | 59 (`13`) | **M0 Logistic Regression** | 26 | **0.8881 ± 0.0962** | `[0.8518, 0.9182]` | 0.9131 | 0.9733 | 0.0642 | 7.31 | 1.04 |
| **PhysioNet_EXERCISE** | 32 | 59 (`13`) | M1 Decision Tree | 26 | 0.8781 ± 0.1057 | `[0.8384, 0.9107]` | 0.9036 | 0.9261 | 0.0684 | 17.56 | 2.21 |
| **PhysioNet_EXERCISE** | 32 | 59 (`13`) | M2 Compact 1D-CNN (GroupNorm) | 6 ch | 0.6622 ± 0.1447 | `[0.6122, 0.7115]` | 0.6771 | 0.7746 | 0.2073 | 31.34 | 6.93 |
| **Pooled (All Subjects)** | 37 | 98 (`37`) | **M1 HistGradientBoosting** | 26 | **0.7869 ± 0.1239** | `[0.7450, 0.8241]` | **0.8096** | **0.8951** | 0.1610 | 12.69 | 1.96 |
| **Pooled (All Subjects)** | 37 | 98 (`37`) | **M1 Random Forest** | 26 | **0.7812 ± 0.1236** | `[0.7393, 0.8186]` | 0.8037 | 0.8919 | 0.1518 | 12.63 | 1.80 |
| **Pooled (All Subjects)** | 37 | 98 (`37`) | **M2 Compact 1D-CNN (GroupNorm)** | 6 ch | **0.7157 ± 0.1236** | `[0.6754, 0.7547]` | 0.7224 | 0.8065 | 0.1818 | 15.86 | 2.84 |
| **Pooled (All Subjects)** | 37 | 98 (`37`) | M0 Logistic Regression | 26 | 0.7155 ± 0.1190 | `[0.6756, 0.7518]` | 0.7615 | 0.8414 | 0.1852 | 19.86 | 2.91 |

### Exact Mathematical Definitions of False-Alarm Metrics (`src/eval/metrics.py`)
Let $y_t \in \{0, 1\}$ denote the ground-truth window label at step $t$ (with stride $S = 15\text{ s}$), $\hat{y}_t = \mathbb{I}[\hat{p}_t \ge \tau]$ the raw binary prediction, and $a_t = \mathbb{I}\left[\sum_{j=0}^{K-1} \hat{y}_{t-j} = K\right]$ ($K=2$) the persisted alert indicator. Let $N_{\text{base}} = \sum_{t} \mathbb{I}[y_t = 0]$ be the number of baseline windows and $T_{\text{base,hr}} = (N_{\text{base}} \cdot S) / 3600$ be the baseline duration in hours:

$$\text{Persisted FA Windows / hr} = \frac{\sum_{t=1}^{T} \mathbb{I}[y_t = 0 \land a_t = 1]}{T_{\text{base,hr}}}, \qquad \text{FA Onset Episodes / hr} = \frac{\sum_{t=1}^{T} \mathbb{I}[y_t = 0 \land a_t = 1 \land a_{t-1} = 0]}{T_{\text{base,hr}}}$$

- Because overlapping 60s windows advance by a 15s stride ($240$ windows/hr), a single 60-second false-positive excursion increments `Persisted FA Windows/hr` 4 times.
- On `PhysioNet_STRESS`, `13.91–25.94` persisted false-positive windows/hr correspond to **`2.09–3.89` distinct false-alarm onset episodes per baseline hour**, which remains high for standalone clinical alerting and motivates **Phase 2 Contextual Environmental Gating**.
- On `PhysioNet_EXERCISE` (`n=32`, median `59` windows/subject, `min=13`), resting pre-exercise baseline segments are short (`2–5` baseline windows in several subjects), so even a single false-positive window near exercise ramp onset inflates the per-hour rate.

---

## 4. Evaluation E2: Cross-Protocol Domain Shift & Transfer Failure Analysis

Can a model trained purely on cognitive/psychosocial stress detect physical exertion, or vice versa? **No — zero-shot cross-protocol transfer fails (`results/tables/cross_dataset_transfer_e2.csv`).**

![Evaluation E2: Cross-Protocol Transfer Failure](results/figures/fig5_cross_protocol_transfer_heatmap.png)

### Why Cross-Protocol Transfer Fails: Opposite Physiological Regimes

```mermaid
flowchart TD
    subgraph STRESS_REGIME["Protocol A: Cognitive / Psychosocial Stress (PhysioNet_STRESS)"]
        S_IN["Stationary Seated Subject\n(Exam / Trier Social Stress)"] --> S_SIG["Physiological Signature:\n• Phasic EDA SCR Bursts ↑↑\n• Sympathetic HR/HRV Shift ↑\n• Kinematic Motion (ACC ENMO) ≈ 0\n• Wrist Skin Temp Slope ≈ Flat"]
    end

    subgraph EX_REGIME["Protocol B: Aerobic / Anaerobic Exertion (PhysioNet_EXERCISE)"]
        E_IN["Treadmill / Cycle Ergometer\n(High Metabolic Load)"] --> E_SIG["Physiological Signature:\n• Kinematic Motion (ACC ENMO) ↑↑↑\n• Heart Rate (130-180 BPM) ↑↑↑\n• Thermal Vasodilation / Sweat ↑↑"]
    end

    S_SIG -.->|"Zero-Shot Transfer: STRESS -> EXERCISE\nMacro-F1 = 0.2024 (All-Baseline Collapse)\nROC-AUC = 0.14 - 0.50"| EX_REGIME
    E_SIG -.->|"Zero-Shot Transfer: EXERCISE -> STRESS\nMacro-F1 = 0.4534 - 0.5153\n(Misses Stationary Stress)"| STRESS_REGIME
```

### Cross-Protocol Transfer Table (`results/tables/cross_dataset_transfer_e2.csv`)

| Transfer Direction | Source Cohort | Target Cohort | Model | Target Macro-F1 (Mean ± SD) | 95% Bootstrap CI | Balanced Acc | ROC-AUC | Persisted FA Win/hr | Empirical Behavior |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `STRESS -> EXERCISE` | Stress (`n=36`) | Exercise (`n=32`) | M0 Logistic Regression | `0.2024 ± 0.0296` | `[0.1896, 0.2169]` | `0.5000` | `0.2952` | `0.00` | Collapses to all-baseline (`0.2024` = baseline share) |
| `STRESS -> EXERCISE` | Stress (`n=36`) | Exercise (`n=32`) | M1 Random Forest | `0.2024 ± 0.0296` | `[0.1896, 0.2169]` | `0.5000` | `0.1387` | `0.00` | Inverted feature correlation (`ROC-AUC = 0.1387`) |
| `STRESS -> EXERCISE` | Stress (`n=36`) | Exercise (`n=32`) | M1 HistGradientBoosting | `0.2024 ± 0.0296` | `[0.1896, 0.2169]` | `0.5000` | `0.1753` | `0.00` | Collapses to all-baseline predictions |
| `STRESS -> EXERCISE` | Stress (`n=36`) | Exercise (`n=32`) | M2 Compact 1D-CNN | `0.3802 ± 0.1389` | `[0.3316, 0.4277]` | `0.4589` | `0.4050` | `151.78` | Below-chance domain generalization |
| `EXERCISE -> STRESS` | Exercise (`n=32`) | Stress (`n=36`) | **M0 Logistic Regression** | **`0.5153 ± 0.1126`** | `[0.4663, 0.5699]` | `0.6954` | `0.7722` | `48.06` | Partial cardiovascular transfer, high false alarms |
| `EXERCISE -> STRESS` | Exercise (`n=32`) | Stress (`n=36`) | M1 Random Forest | `0.4359 ± 0.0403` | `[0.4213, 0.4490]` | `0.5000` | `0.5582` | `0.00` | Requires high ACC motion to fire; predicts all baseline |
| `EXERCISE -> STRESS` | Exercise (`n=32`) | Stress (`n=36`) | M1 HistGradientBoosting | `0.4359 ± 0.0403` | `[0.4213, 0.4490]` | `0.5000` | `0.3910` | `0.00` | Predicts all baseline on stationary subjects |

---

## 5. Evaluation E3: Sensor, Feature (21 vs 26), Window & Calibration Ablations

All `E3` ablations (`results/tables/sensor_window_ablations_e3.csv`) run across the **full 36-subject `PhysioNet_STRESS` cohort**, matching `E1` (`0.6544`) to the 4th decimal place.

![Evaluation E3: Sensor, Window & Normalization Ablations](results/figures/fig2_sensor_and_normalization_ablations.png)

### E3 Ablation Benchmark Table (`m1_random_forest`, `n=36` Subjects)

| Ablation Category | Configuration | Subjects (`n`) | Features | Macro-F1 (Mean ± SD) | 95% Bootstrap CI | Balanced Acc | ROC-AUC | Key Scientific Takeaway |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Sensor Subset** | `BVP_ONLY` | 36 | 10 | `0.5612 ± 0.1598` | `[0.5109, 0.6113]` | `0.6129` | `0.6786` | Photoplethysmography alone has high inter-subject variance |
| **Sensor Subset** | `EDA_ONLY` | 36 | 9 | `0.5587 ± 0.1583` | `[0.5050, 0.6104]` | `0.6020` | `0.7130` | Strong ROC-AUC (`0.7130`) from sympathetic sudomotor activity |
| **Sensor Subset** | `BVP_EDA` | 36 | 16 | `0.6157 ± 0.1636` | `[0.5586, 0.6659]` | `0.6486` | `0.7615` | Combining cardiac + electrodermal streams boosts F1 by `+0.0545` |
| **Sensor Subset** | `BVP_EDA_TEMP` | 36 | 20 | `0.6242 ± 0.1653` | `[0.5692, 0.6744]` | `0.6557` | `0.7773` | **Wrist `TEMP` adds only `+0.0085` F1 (CIs overlap heavily)** |
| **Feature Count** | `ALL_SENSORS_21_BASE` | 36 | 21 | **`0.6572 ± 0.1726`** | `[0.6001, 0.7118]` | **`0.6887`** | **`0.7905`** | 21 base features perform on par with (slightly above) 26 features |
| **Feature Count** | `ALL_SENSORS` (Default) | 36 | 26 | `0.6544 ± 0.1693` | `[0.5973, 0.7073]` | `0.6848` | `0.7884` | Exact match to `E1`; 5 interaction ratios add no gain for trees |
| **Window Length** | `W30_S10` (`30s` win, `10s` stride) | 36 | 26 | `0.6426 ± 0.1669` | `[0.5877, 0.6951]` | `0.6693` | `0.7873` | Shorter window slightly increases HRV estimation noise |
| **Window Length** | `W60_S15` (`60s` win, `15s` stride) | 36 | 26 | `0.6544 ± 0.1693` | `[0.5973, 0.7073]` | `0.6848` | `0.7884` | Optimal balance of temporal resolution and HRV stability |
| **Window Length** | `W120_S30` (`120s` win, `30s` stride) | 36 | 26 | `0.6531 ± 0.1812` | `[0.5902, 0.7073]` | `0.6929` | `0.7775` | Diminishing returns beyond 60s window |
| **Calibration** | `RawUnnormalized` | 36 | 26 | `0.3887 ± 0.1846` | `[0.3292, 0.4485]` | `0.5015` | `0.5204` | Fails completely (`Balanced Acc = 0.5015`) without calibration |
| **Calibration** | `UnsupervisedSessionZScore` | 36 | 26 | `0.4879 ± 0.0680` | `[0.4655, 0.5096]` | `0.5318` | `0.6095` | Whole-session z-score dilutes sustained stress |
| **Calibration** | `UnsupervisedRestingPrefixCalibrated` | 36 | 26 | `0.6544 ± 0.1693` | `[0.5973, 0.7073]` | `0.6848` | `0.7884` | **Assumes initial 15% of session is resting baseline** |
| **Calibration** | `PerSubjectBaseline_Oracle` | 36 | 26 | **`0.7025 ± 0.1369`** | `[0.6562, 0.7445]` | **`0.7289`** | **`0.8180`** | **Oracle upper bound** (uses ground-truth `y=0` labels) |
| **Cohort Split** | `Protocol_S_v1_18subs` (`S01–S18`) | 18 | 26 | `0.6776 ± 0.1695` | `[0.5947, 0.7495]` | `0.7082` | `0.8075` | First 18-subject protocol sub-cohort |
| **Cohort Split** | `Protocol_f_v2_18subs` (`f01–f14_b`) | 18 | 26 | `0.6347 ± 0.1669` | `[0.5539, 0.7072]` | `0.6644` | `0.7699` | Second 18-subject sub-cohort (explains earlier `0.6347` figure) |

---

## 6. Phase 1D: Multi-Model Edge Deployment & Int8 Quantization Profile

All 5 models (`M0`, `M1` family, `M2`) are **trained on an 80% subject-isolated split (`28` subjects, `2,794` windows)** and evaluated on a **held-out 20% test split (`9` subjects, `758` windows)** before and after `float32 -> int8` weight/threshold quantization (`results/tables/edge_profile_m2.csv`).

![Phase 1D: Multi-Model Edge Deployment & Quantization Profile](results/figures/fig6_edge_quantization_pareto.png)

### Edge Footprint, Held-Out Test Accuracy & Proxy Latency (`results/tables/edge_profile_m2.csv`)

| Model | Input Representation | Params / Nodes | Float32 Size | Int8 Size | Compression | Held-Out Float32 Macro-F1 | Held-Out Int8 Macro-F1 | $\Delta$ Macro-F1 | Float32 Bal Acc | Int8 Bal Acc | Desktop CPU Proxy Latency (`p50` / `p95`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0 Logistic Regression** | 26 Tabular Features | 27 params | `0.91 KB` | **`0.05 KB`** | **`18.20x`** | `0.9075` | `0.9066` | `-0.0009` | `0.9077` | `0.9068` | `0.143 / 0.200 ms` |
| **M1 Decision Tree** | 26 Tabular Features | 25 nodes | `3.05 KB` | `1.52 KB` | `2.00x` | `0.9099` | `0.9099` | `0.0000` | `0.9102` | `0.9102` | `0.070 / 0.105 ms` |
| **M1 HistGradientBoosting** | 26 Tabular Features | 2,546 nodes | `207.09 KB` | **`103.55 KB`** | `2.00x` | `0.9023` | `0.9023` | `0.0000` | `0.9012` | `0.9012` | `1.789 / 2.273 ms` |
| **M1 Random Forest** | 26 Tabular Features | 7,324 nodes | `618.25 KB` | `309.12 KB` | `2.00x` | **`0.9301`** | **`0.9301`** | `0.0000` | **`0.9306`** | **`0.9306`** | `5.489 / 7.515 ms` |
| **M2 Compact 1D-CNN** | `32 Hz` 6-Ch Tensor | 12,161 params | `55.22 KB` | **`12.88 KB`** | **`4.29x`** | `0.7365` | `0.7355` | `-0.0010` | `0.7438` | `0.7427` | `1.211 / 1.797 ms` |

> **Hardware Proxy Note**: Latency figures are measured single-threaded (`torch.set_num_threads(1)`) on a **Desktop x86_64 CPU proxy**, not on a physical ARM Cortex-M4 microcontroller. Tabular model latencies exclude the `~1.1 ms` DSP feature-extraction step.

---

## 7. Phase 2: Next-Day (`t+1`) CPCB India AQI Forecasting & Contextual Fusion

![Phase 2: Next-Day AQI Forecast & Multimodal Contextual Risk Fusion](results/figures/fig3_contextual_fusion_matrix.png)

### 7.1 Real CPCB India `city_day.csv` Next-Day (`t+1`) Forecasting Benchmark (`src/fusion/aqi_pipeline.py`)
- **Dataset (`data/raw/india_aqi/city_day.csv`)**: `29,531` real daily air-quality records across **26 Indian cities (`2015–2020`)**, yielding `25,752` valid city-day pairs after chronological lag construction (`20,591` chronological training days [first 80% per city] and `5,161` chronological test days [final 20% per city]).
- **Causal Next-Day Formulation**: Predicts **Next-Day Severe/Poor Air Hazard (`y_{t+1} = I[AQI_{t+1} > 200]`)** strictly from day-$t$ pollutants (`PM2.5_t, PM10_t, NO2_t, CO_t, SO2_t, O3_t`) and 3-day rolling temporal lags (`PM2.5_roll3_t, PM10_roll3_t, PM2.5_diff1_t`).

| Forecasting Model | Train Days | Test Days | Test Macro-F1 | Test Balanced Accuracy | Test ROC-AUC | Gain Over Persistence |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Naive Same-Day Persistence Baseline ($\hat{y}_{t+1} = y_t$)** | 20,591 | 5,161 | `0.8883` | `0.8897` | — | Baseline |
| **HistGradientBoosting Next-Day (`t+1`) Forecaster** | 20,591 | 5,161 | **`0.9094`** | **`0.9315`** | **`0.9808`** | **`+0.0211` Macro-F1 / `+4.18%` Bal Acc** |

### 7.2 Multimodal Contextual Alert Escalation Logic (`src/fusion/contextual_fusion.py`)

Both **ambient heat (`ambient_temp_c >= 38.0 °C`)** and **next-day air-quality hazard (`p_env >= 0.50` / `AQI > 200`)** actively govern the 4-tier `AlertLevel` decision tree:

```mermaid
flowchart TD
    IN["Window Inputs at Step t:\n• Wearable Strain Prob p_phys(t)\n• K=2 Persisted Strain Flag\n• Motion Context (ACC ENMO >= 0.12g)\n• Next-Day AQI Hazard Prob p_env(t+1)\n• Ambient Temperature T_amb (°C)"] --> PERS{"Is Wearable Strain\nPersisted (K=2)?"}

    PERS -->|"No (Baseline or Transient Spike)"| ENV_CHK{"Is Next-Day AQI Severe (AQI > 200)\nOR High Heat (T_amb >= 38.0°C)?"}
    ENV_CHK -->|"No"| L0["Tier 0: NORMAL\n(No Intervention Required)"]
    ENV_CHK -->|"Yes"| L1_ENV["Tier 1: ADVISORY\n(Environmental Precaution Advisory)"]

    PERS -->|"Yes (Sustained Strain)"| MOT_CHK{"Is Subject in High\nAerobic Motion (ENMO >= 0.12g)?"}
    MOT_CHK -->|"Yes (Intentional Exercise)"| EX_ENV{"Is AQI > 200 OR\nHigh Heat (T_amb >= 38.0°C)?"}
    EX_ENV -->|"No (Clean/Cool Air)"| L1_EX["Tier 1: ADVISORY\n(Expected Exertional Strain — Hydrate)"]
    EX_ENV -->|"Yes (Hazardous Air or Heat)"| L2_EX["Tier 2: WARNING\n(High Exertion in Poor AQI / Heat >= 38°C)"]

    MOT_CHK -->|"No (Stationary / Non-Exertional Strain)"| ST_ENV{"Environmental Severity:\nAQI > 300 or (AQI > 200 & T >= 38°C)?"}
    ST_ENV -->|"Extreme"| L3["Tier 3: CRITICAL_ALERT\n(Unexplained Rest Strain + Severe Hazard)"]
    ST_ENV -->|"Moderate (AQI > 100 or T >= 38°C)"| L2_ST["Tier 2: WARNING\n(Sustained Rest Strain + Heat/AQI Stress)"]
    ST_ENV -->|"Benign Environment"| L1_ST["Tier 1: ADVISORY\n(Sustained Autonomic Stress in Clean Air)"]
```

### 7.3 Contextual Fusion Scenarios (`results/tables/phase2_contextual_fusion_demo.csv`)

Scenarios `S1–S6` verify deterministic rule boundaries (including `S6_Resting_HighHeat_CleanAir` at `39.5 °C`), while **`S7` and `S8` are driven by real out-of-fold LOSO wearable predictions from Subject `S01` paired with real chronological test-split AQI forecasts from Delhi and Mumbai**:

| Scenario ID | Source Type | Temp (`°C`) | Wearable $p_{\text{phys}}$ | Persisted | Motion Context | Ambient AQI (`Tier`) | Composite Score | Alert Level |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| `S1_Resting_CleanAir` | Rule Verification | `26.0 °C` | `0.180` | `False` | Rest / Light | `42` (`Good`) | `0.150` | `0: NORMAL` |
| `S2_Transient_Spike_Suppressed` | Rule Verification | `29.0 °C` | `0.680` | `False` | Rest / Light | `85` (`Satisfactory`) | `0.493` | `0: NORMAL` |
| `S3_Exercise_CleanAir` | Rule Verification | `27.0 °C` | `0.780` | `True` | Vigorous Exercise | `65` (`Satisfactory`) | `0.556` | `1: ADVISORY` |
| `S4_Exercise_PoorAQI` | Rule Verification | `34.0 °C` | `0.810` | `True` | Vigorous Exercise | `265` (`Poor`) | `0.812` | `2: WARNING` |
| `S5_NonExertional_SevereAQI` | Rule Verification | `36.0 °C` | `0.790` | `True` | Rest / Light | `340` (`Very Poor`) | `0.869` | `3: CRITICAL_ALERT` |
| `S6_Resting_HighHeat_CleanAir` | **Heat Rule (`>=38°C`)** | **`39.5 °C`** | `0.640` | `True` | Rest / Light | `78` (`Satisfactory`) | `0.646` | **`2: WARNING`** |
| **`S7_Real_OOF_STRESS_S01_Delhi_PoorAQI`** | **Real LOSO OOF + Real CPCB** | `33.0 °C` | `0.625` | `True` | Rest / Light | `482` (`Severe`) | `0.779` | **`3: CRITICAL_ALERT`** |
| **`S8_Real_OOF_AEROBIC_S01_Mumbai_SatisfactoryAQI`** | **Real LOSO OOF + Real CPCB** | `28.5 °C` | `0.946` | `True` | Vigorous Exercise | `64` (`Satisfactory`) | `0.647` | **`1: ADVISORY`** |

---

## 8. Phase 3: Privacy-Preserving Federated Learning Benchmark

We benchmark **full 36-subject (`35` clients/round) federated tabular logistic regression** against **12-subject (`11` clients/round) federated `M2 1D-CNN`** on `PhysioNet_STRESS` (`results/tables/phase3_federated_benchmark.csv`).

![Phase 3: Federated Learning Accuracy vs Communication Cost](results/figures/fig4_federated_comm_vs_f1.png)

### Federated Training & `Int8` Weight-Quantized Aggregation Protocol

```mermaid
sequenceDiagram
    participant Server as Central Aggregator Server
    participant C1 as Client Subject 1 (Local E4 Data)
    participant C35 as Client Subject 35 (Local E4 Data)
    participant Test as Held-Out Test Subject s (Zero Leakage)

    loop For each Communication Round r = 1..R
        Server->>C1: Broadcast Global Weights w^(r) (Int8 Quantized)
        Server->>C35: Broadcast Global Weights w^(r) (Int8 Quantized)
        Note over C1,C35: Raw physiological signals NEVER leave the wearable
        C1->>C1: Local SGD + FedProx Proximal Term (μ/2)||w - w^(r)||^2
        C35->>C35: Local SGD + FedProx Proximal Term (μ/2)||w - w^(r)||^2
        C1->>Server: Upload Int8 Weight Update Δw_1^(r)
        C35->>Server: Upload Int8 Weight Update Δw_35^(r)
        Server->>Server: Sample-Weighted Aggregation w^(r+1) = Σ (n_k / n) w_k^(r)
    end
    Server->>Test: Evaluate Final Global Model w^(R) on Held-Out Subject s
```

### Phase 3 Federated Benchmark Table (`results/tables/phase3_federated_benchmark.csv`)

| Algorithm / Architecture | Subjects (`n`) | Clients / Round | Rounds $\times$ Epochs | Macro-F1 (Mean ± SD) | 95% Bootstrap CI | Balanced Acc | ROC-AUC | Persisted FA Win/hr | FA Onset Episodes/hr | Float32 Comm | Int8 Comm |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized LOSO (`M0 LogReg`)** | 36 | 35 | Central | **`0.6658 ± 0.1350`** | `[0.6197, 0.7056]` | **`0.7414`** | **`0.8235`** | 25.94 | 3.89 | `0.00 KB` | `0.00 KB` |
| **FedAvg Tabular LogReg (`100%` clients)** | 36 | 35 | $8 \times 5$ | `0.6573 ± 0.1686` | `[0.6055, 0.7104]` | `0.7149` | `0.8043` | 21.48 | 2.22 | `59.06 KB` | `14.77 KB` |
| **FedProx Tabular LogReg ($\mu=0.05$, `100%`)** | 36 | 35 | $8 \times 5$ | `0.6584 ± 0.1670` | `[0.6075, 0.7102]` | `0.7160` | `0.8040` | 21.48 | 2.25 | `59.06 KB` | `14.77 KB` |
| **FedAvg Tabular LogReg (`60%` partial)** | 36 | 21 | $8 \times 5$ | **`0.6588 ± 0.1675`** | `[0.6070, 0.7105]` | **`0.7169`** | **`0.8062`** | 21.72 | 2.22 | `35.44 KB` | **`8.86 KB`** |
| **Centralized LOSO (`M2 1D-CNN`, full)** | 36 | 35 | Central | `0.5337 ± 0.0956` | `[0.5030, 0.5653]` | `0.5602` | `0.6900` | 2.21 | 0.45 | `0.00 KB` | `0.00 KB` |
| **FedAvg `M2 1D-CNN` (12-sub cohort)** | 12 | 11 | $6 \times 1$ | `0.4361 ± 0.1069` | `[0.3712, 0.4892]` | `0.4841` | `0.4706` | 3.31 | 1.10 | `7,759.1 KB` | `1,939.8 KB` |
| **FedProx `M2 1D-CNN` ($\mu=0.05$, 12-sub)** | 12 | 11 | $6 \times 1$ | `0.4300 ± 0.1095` | `[0.3652, 0.4852]` | `0.4800` | `0.4645` | 3.31 | 1.10 | `7,759.1 KB` | `1,939.8 KB` |

---

## 9. Repository Structure & Reproducibility Commands

```text
├── configs/
│   └── base.yaml                                  # Global windowing, sensor rates, and seed configuration
├── data/
│   └── raw/
│       ├── india_aqi/city_day.csv                 # Real CPCB India dataset (26 cities, 29,531 daily rows)
│       └── physionet_stress/                      # Real PhysioNet Wearable Exam Stress & Exercise E4 recordings
├── docs/
│   ├── assumptions.md                             # Explicit methodological & hardware proxy disclosures
│   └── problem_formulation.md                     # Mathematical formulation of Phase 1, Phase 2, and Phase 3
├── results/
│   ├── figures/                                   # 7 publication-grade 300-DPI PNG charts (fig1 .. fig7)
│   └── tables/                                    # 6 reproducible CSV benchmark tables
├── src/
│   ├── edge/                                      # Multi-model float32 -> int8 quantization & latency profiler
│   ├── eval/                                      # LOSO evaluator, E2 transfer, E3 ablations, metrics & plots
│   ├── features/                                  # 26-feature extractor, resting-prefix normalizer & tensors
│   ├── federated/                                 # 36-subject Tabular & 12-subject CNN FedAvg/FedProx engine
│   ├── fusion/                                    # Causal t+1 CPCB AQI forecaster & Heat/AQI risk engine
│   ├── io/                                        # PhysioNet E4 physiological loader
│   ├── models/                                    # M0/M1 baselines and GroupNorm M2 1D-CNN
│   └── windowing/                                 # 60s/15s window slicer with ±30s transition drop
└── tests/                                         # 37 unit & zero-subject-leakage verification tests
```

### Run All Verification & Reproduction Scripts

```bash
# 1. Run all 37 unit and subject-leakage tests
pytest -v

# 2. Reproduce Phase 1 LOSO (E1), Cross-Protocol Transfer (E2), Ablations (E3), and Multi-Model Edge Profiling
python -m src.eval.run_phase1_experiments --data-dir data/raw/physionet_stress --results-dir results

# 3. Reproduce Phase 2 (Next-Day CPCB India AQI Forecasting + Out-of-Fold Wearable Fusion) & Phase 3 (Federated Benchmark)
python -m src.eval.run_phase2_phase3 --physionet-dir data/raw/physionet_stress --results-dir results

# 4. Regenerate all 7 publication figures (300 DPI PNGs)
python -m src.eval.plot_results
```
