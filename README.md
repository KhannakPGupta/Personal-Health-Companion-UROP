# Edge-Intelligent Personal Health Companion (UROP Research Benchmark)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests: 37 Passed](https://img.shields.io/badge/tests-37%20passed-brightgreen.svg)](#reproducibility--verification)

A reproducible, subject-independent **Edge-AI + Contextual Environmental Fusion + Federated Learning** research repository for physiological strain detection on wrist-worn wearables (Empatica E4: BVP `64 Hz`, EDA `4 Hz`, TEMP `4 Hz`, ACC `32 Hz`).

---

## Key Empirical Findings & Methodological Disclosures

This benchmark evaluates **all 37 real human subjects** (`68` subject-protocol recordings across `PhysioNet_STRESS` [`n=36`], `PhysioNet_EXERCISE` [`n=32`], and `Pooled` [`n=37`]) from the **PhysioNet Wearable Exam Stress & Exercise Dataset v1.0.0** ([DOI: 10.13026/kvf3-0c87](https://physionet.org/content/wearable-device-dataset/1.0.0/)) alongside **29,531 real daily air-quality records across 26 Indian cities (`2015–2020`)** from the Central Pollution Control Board (`data/raw/india_aqi/city_day.csv`).

### 1. Within-Dataset Subject-Independent LOSO (`E1`: `results/tables/loso_benchmark_e1.csv`)
All tabular models use **26 features** (`21` base time/frequency features + `5` cross-sensor autonomic interaction ratios); the `M2 1D-CNN` operates directly on **6 raw sensor channels** (`BVP, EDA, TEMP, ACC_x, ACC_y, ACC_z`) resampled to `32 Hz`.

| Cohort | Subjects (`n`) | Median (`Min`) Win/Sub | Model | Features | Macro-F1 (Mean ± SD) | 95% Bootstrap CI | Balanced Acc | ROC-AUC | PR-AUC | Persisted FA Win/hr | FA Onset Episodes/hr |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PhysioNet_STRESS** | 36 | 53 (`31`) | **M0 Logistic Regression** | 26 | **0.6658 ± 0.1350** | `[0.6197, 0.7056]` | **0.7414** | **0.8235** | **0.8759** | 25.94 | 3.89 |
| **PhysioNet_STRESS** | 36 | 53 (`31`) | **M1 HistGradientBoosting** | 26 | **0.6616 ± 0.1653** | `[0.6061, 0.7131]` | 0.6985 | 0.7824 | 0.8588 | 13.91 | 2.09 |
| **PhysioNet_STRESS** | 36 | 53 (`31`) | **M1 Random Forest** | 26 | **0.6544 ± 0.1693** | `[0.5973, 0.7073]` | 0.6848 | 0.7884 | 0.8617 | 15.55 | 2.16 |
| **PhysioNet_STRESS** | 36 | 53 (`31`) | M1 Decision Tree | 26 | 0.5944 ± 0.1636 | `[0.5382, 0.6461]` | 0.6305 | 0.6775 | 0.7998 | 21.62 | 4.65 |
| **PhysioNet_STRESS** | 36 | 53 (`31`) | M2 Compact 1D-CNN (GroupNorm) | 6 ch | 0.5337 ± 0.0956 | `[0.5030, 0.5653]` | 0.5602 | 0.6900 | 0.8087 | 2.21 | 0.45 |
| **PhysioNet_EXERCISE** | 32 | 59 (`13`) | **M1 HistGradientBoosting** | 26 | **0.8941 ± 0.0879** | `[0.8620, 0.9227]` | **0.9145** | **0.9749** | **0.9862** | 16.21 | 2.06 |
| **PhysioNet_EXERCISE** | 32 | 59 (`13`) | **M1 Random Forest** | 26 | **0.8926 ± 0.1050** | `[0.8535, 0.9248]` | 0.9112 | 0.9726 | 0.9854 | 15.92 | 2.07 |
| **PhysioNet_EXERCISE** | 32 | 59 (`13`) | **M0 Logistic Regression** | 26 | **0.8881 ± 0.0962** | `[0.8518, 0.9182]` | 0.9131 | 0.9733 | 0.9865 | 7.31 | 1.04 |
| **PhysioNet_EXERCISE** | 32 | 59 (`13`) | M1 Decision Tree | 26 | 0.8781 ± 0.1057 | `[0.8384, 0.9107]` | 0.9036 | 0.9261 | 0.9465 | 17.56 | 2.21 |
| **PhysioNet_EXERCISE** | 32 | 59 (`13`) | M2 Compact 1D-CNN (GroupNorm) | 6 ch | 0.6622 ± 0.1447 | `[0.6122, 0.7115]` | 0.6771 | 0.7746 | 0.8489 | 31.34 | 6.93 |
| **Pooled (All Subjects)** | 37 | 98 (`37`) | **M1 HistGradientBoosting** | 26 | **0.7869 ± 0.1239** | `[0.7450, 0.8241]` | **0.8096** | **0.8951** | **0.9035** | 12.69 | 1.96 |
| **Pooled (All Subjects)** | 37 | 98 (`37`) | **M1 Random Forest** | 26 | **0.7812 ± 0.1236** | `[0.7393, 0.8186]` | 0.8037 | 0.8919 | 0.9032 | 12.63 | 1.80 |
| **Pooled (All Subjects)** | 37 | 98 (`37`) | **M2 Compact 1D-CNN (GroupNorm)** | 6 ch | **0.7157 ± 0.1236** | `[0.6754, 0.7547]` | 0.7224 | 0.8065 | 0.8434 | 15.86 | 2.84 |
| **Pooled (All Subjects)** | 37 | 98 (`37`) | M0 Logistic Regression | 26 | 0.7155 ± 0.1190 | `[0.6756, 0.7518]` | 0.7615 | 0.8414 | 0.8758 | 19.86 | 2.91 |

> **Exact False-Alarm Metric Definitions (`src/eval/metrics.py`)**:
> 1. **`false_alerts_persisted_per_hr`**: Number of persisted false-positive windows ($y_t = 0 \land a_t = 1$, after $K=2$ consecutive raw detections) normalized per hour of true baseline time ($T_{\text{base,hr}} = N_{\text{base}} \times 15\text{ s} / 3600$). Because windows overlap with a 15-second stride, a single 60-second false-positive excursion increments this counter 4 times.
> 2. **`false_alarm_episodes_per_hr`**: Number of distinct persisted alarm **onset transitions** ($a_{t-1} = 0 \to a_t = 1$ while $y_t = 0$) per baseline hour. On `PhysioNet_STRESS`, `13.91–25.94` persisted false-positive windows/hr correspond to **`2.09–3.89` distinct false-alarm onset episodes per baseline hour**, which remains elevated for continuous clinical alerting without contextual gating. Furthermore, while `PhysioNet_EXERCISE` has a median of `59` windows/subject (`min=13`), resting baseline periods prior to treadmill/ergometer ramp onset are brief (`2–5` baseline windows in several subjects), inflating per-subject false-alert variance.

---

### 2. Cross-Protocol Transfer Failure Analysis (`E2`: `results/tables/cross_dataset_transfer_e2.csv`)

Training on cognitive/psychosocial stress (`PhysioNet_STRESS`) and testing zero-shot on physical exertion (`PhysioNet_EXERCISE`), or vice versa, **fails to generalize**:

| Transfer Direction | Model | Target Macro-F1 | Target Balanced Acc | Target ROC-AUC | Finding |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `STRESS -> EXERCISE` | M0 Logistic Regression | `0.2140 ± 0.0882` | `0.4832` | `0.1404` | Severe domain shift (inverted ACC correlation) |
| `STRESS -> EXERCISE` | M1 Random Forest | `0.2024 ± 0.0297` | `0.5000` | `0.3690` | Collapses to all-baseline predictions (`0.2024` matches baseline class share) |
| `STRESS -> EXERCISE` | M1 HistGradientBoosting | `0.2059 ± 0.0442` | `0.5007` | `0.2896` | Near-all-baseline predictions |
| `EXERCISE -> STRESS` | M1 HistGradientBoosting | `0.5153 ± 0.1656` | `0.5624` | `0.6252` | Near-chance transfer |
| `EXERCISE -> STRESS` | M1 Random Forest | `0.4534 ± 0.1684` | `0.5432` | `0.6502` | Fails to detect stationary cognitive stress |

**Core Scientific Finding**: Psychological stress (stationary sympathetic arousal during an exam or interview) and physical exertion (high kinematic movement + metabolic cardiovascular demand) produce **fundamentally distinct physiological signatures**. Treating both as a single interchangeable proxy for "elevated physiological strain" does not hold under zero-shot cross-protocol transfer; multi-protocol pooled training (`Macro-F1 = 0.7869`) or protocol-specific models are required.

---

### 3. Sensor, Window, Feature & Normalization Ablations (`E3`: `results/tables/sensor_window_ablations_e3.csv`)

All `E3` ablations evaluate `m1_random_forest` across the **full 36-subject `PhysioNet_STRESS` cohort**, matching `E1` (`0.6544`) to the 4th decimal place:

| Ablation Type | Variant | Subjects (`n`) | Features | Macro-F1 (Mean ± SD) | 95% Bootstrap CI | Balanced Acc | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sensor Subset** | `BVP_ONLY` | 36 | 6 | `0.4820 ± 0.1190` | `[0.4437, 0.5206]` | `0.5210` | `0.5608` |
| **Sensor Subset** | `EDA_ONLY` | 36 | 5 | `0.6196 ± 0.1657` | `[0.5630, 0.6708]` | `0.6683` | `0.7852` |
| **Sensor Subset** | `BVP_EDA` | 36 | 11 | `0.6157 ± 0.1726` | `[0.5572, 0.6688]` | `0.6512` | `0.7793` |
| **Sensor Subset** | `BVP_EDA_TEMP` | 36 | 15 | `0.6242 ± 0.1665` | `[0.5673, 0.6754]` | `0.6584` | `0.7834` |
| **Sensor Subset** | `ALL_SENSORS_21_BASE` (no interactions) | 36 | 21 | **`0.6572 ± 0.1604`** | `[0.6022, 0.7069]` | `0.6884` | **`0.7931`** |
| **Sensor Subset** | `ALL_SENSORS` (`21` base + `5` interactions) | 36 | 26 | `0.6544 ± 0.1693` | `[0.5973, 0.7073]` | `0.6848` | `0.7884` |
| **Window Length** | `W30_S15` (`30s` window, `15s` stride) | 36 | 26 | `0.6406 ± 0.1670` | `[0.5841, 0.6921]` | `0.6694` | `0.7798` |
| **Window Length** | `W60_S15` (`60s` window, `15s` stride) | 36 | 26 | `0.6544 ± 0.1693` | `[0.5973, 0.7073]` | `0.6848` | `0.7884` |
| **Window Length** | `W120_S30` (`120s` window, `30s` stride) | 36 | 26 | `0.6596 ± 0.1782` | `[0.5988, 0.7145]` | `0.6929` | `0.7867` |
| **Normalization** | `RawUnnormalized` (uncalibrated across subjects) | 36 | 26 | `0.3887 ± 0.1846` | `[0.3292, 0.4485]` | `0.5015` | `0.5204` |
| **Normalization** | `UnsupervisedSessionZScore` | 36 | 26 | `0.4879 ± 0.0680` | `[0.4655, 0.5096]` | `0.5318` | `0.6095` |
| **Normalization** | `UnsupervisedRestingPrefixCalibrated` *(Default)* | 36 | 26 | `0.6544 ± 0.1693` | `[0.5973, 0.7073]` | `0.6848` | `0.7884` |
| **Normalization** | `PerSubjectBaseline_Oracle` *(Oracle Upper Bound)* | 36 | 26 | **`0.7025 ± 0.1369`** | `[0.6562, 0.7445]` | `0.7289` | `0.8180` |
| **Cohort Split** | `Protocol_S_v1_18subs` (`S01–S18`) | 18 | 26 | `0.6776 ± 0.1695` | `[0.5947, 0.7495]` | `0.7082` | `0.8075` |
| **Cohort Split** | `Protocol_f_v2_18subs` (`f01–f14_b`) | 18 | 26 | `0.6347 ± 0.1669` | `[0.5539, 0.7072]` | `0.6644` | `0.7699` |

**Key Ablation Takeaways**:
- **Overlapping Confidence Intervals & Marginal `TEMP`/`ACC` Gain**: Across `EDA_ONLY` (`0.6196`, CI `[0.5630, 0.6708]`), `BVP_EDA` (`0.6157`), `BVP_EDA_TEMP` (`0.6242`), and `ALL_SENSORS` (`0.6544`, CI `[0.5973, 0.7073]`), the 95% bootstrap confidence intervals overlap substantially. Adding peripheral wrist skin temperature (`TEMP`) increases Macro-F1 by only `+0.0085` over `BVP_EDA`, and adding `ACC` adds `+0.0302`. This is a critical domain finding for heat-health wearables: during indoor cognitive stress in a thermally controlled room, wrist skin temperature and movement contribute almost no discriminative signal compared to electrodermal activity (`EDA`).
- **Base `21` vs Extended `26` Features**: The `21` base features (`ALL_SENSORS_21_BASE`, `Macro-F1 = 0.6572`) and `26` features (`ALL_SENSORS`, `Macro-F1 = 0.6544`) perform identically within statistical noise; the 5 hand-crafted autonomic interaction ratios do not improve tree-based LOSO generalization over the 21 base features.
- **Calibration Assumptions**: `UnsupervisedRestingPrefixCalibrated` (`0.6544`) requires the practical assumption that a wearable session begins with a resting prefix (`first 15%` of windows). `PerSubjectBaseline_Oracle` (`0.7025`) uses ground-truth baseline labels (`y == 0`) across the entire recording and is reported strictly as a theoretical **oracle upper bound**.
- **WESAD Clarification**: Only the **PhysioNet Wearable Exam Stress & Exercise Dataset** (`68` subject-protocol sessions) is evaluated in this repository; the **WESAD** dataset is **not** evaluated here due to licensing/redistribution constraints.

---

### 4. Multi-Model Edge Deployment & Quantization Profile (`results/tables/edge_profile_m2.csv`)

All 5 models (`M0`, `M1` family, `M2`) are **trained on an 80% subject-isolated training split (`28` subjects, `2,794` windows)** and evaluated on a **held-out 20% subject test split (`9` subjects, `758` windows)** before and after `float32 -> int8` quantization:

| Model | Params / Nodes | Float32 Size (KB) | Int8 Size (KB) | Compression Ratio | Held-Out Float32 Macro-F1 | Held-Out Int8 Macro-F1 | $\Delta$ Macro-F1 | Float32 Bal Acc | Int8 Bal Acc | Desktop x86_64 CPU Proxy Latency (`p50` / `p95` ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0 Logistic Regression** | 27 params | `0.91 KB` | **`0.05 KB`** | **`18.20x`** | `0.9075` | `0.9066` | `-0.0009` | `0.8925` | `0.8906` | `0.143 / 0.258 ms` |
| **M1 Decision Tree** | 25 nodes | `3.05 KB` | `1.52 KB` | `2.00x` | `0.9099` | `0.9099` | `0.0000` | `0.9036` | `0.9036` | `0.070 / 0.119 ms` |
| **M1 HistGradientBoosting** | 2,546 nodes | `207.09 KB` | `103.55 KB` | `2.00x` | `0.9023` | `0.9023` | `0.0000` | `0.8938` | `0.8938` | `1.789 / 2.868 ms` |
| **M1 Random Forest** | 7,324 nodes | `618.25 KB` | `309.12 KB` | `2.00x` | `0.9301` | `0.9301` | `0.0000` | `0.9182` | `0.9182` | `5.489 / 8.711 ms` |
| **M2 Compact 1D-CNN** | 12,161 params | `55.22 KB` | **`12.88 KB`** | **`4.29x`** | `0.7365` | `0.7355` | `-0.0010` | `0.7368` | `0.7349` | `1.211 / 1.986 ms` |

- **Hardware Disclosure**: Latency numbers are measured single-sample on a **Desktop x86_64 CPU proxy** (`torch.set_num_threads(1)`), **not** on a physical Cortex-M4 wearable MCU.
- **Best Edge Trade-off**: `M0 Logistic Regression` achieves higher accuracy than `M2 1D-CNN` while requiring only **`0.05 KB` (`53 bytes`)** in `int8` (`18.2x` compression) and `0.143 ms` proxy inference latency. For tree ensembles, `M1 HistGradientBoosting` (`103.55 KB` quantized) fits comfortably inside a `256 KB` Cortex-M4F flash budget, whereas `M1 Random Forest` (`618.25 KB` float32 / `309.12 KB` quantized) exceeds tight MCU flash budgets.

---

### 5. Phase 2: Next-Day (`t+1`) CPCB India AQI Forecasting & Contextual Fusion (`results/tables/phase2_contextual_fusion_demo.csv`)

1. **Real CPCB India `city_day.csv` Dataset (`data/raw/india_aqi/city_day.csv`)**:
   - **Source**: `29,531` daily air-quality records across **26 Indian cities (`2015–2020`)**; `25,752` valid city-day pairs after chronological lag alignment (`20,591` chronological training days [first 80% per city] and `5,161` chronological test days [final 20% per city]).
   - **Next-Day (`t+1`) Forecasting Formulation (`src/fusion/aqi_pipeline.py`)**: Rather than classifying same-day `AQI_t > 200` from same-day pollutants (which would be a deterministic formula lookup), `AmbientAQIPredictor` forecasts **Next-Day Severe/Poor Air Hazard (`y_{t+1} = I[AQI_{t+1} > 200]`)** strictly from day-$t$ pollutants (`PM2.5_t, PM10_t, NO2_t, CO_t, SO2_t, O3_t`) and 3-day rolling temporal lags (`PM2.5_roll3_t, PM10_roll3_t, PM2.5_diff1_t`).
   - **Held-Out Test Performance (`5,161` chronological test days)**:
     - **HistGradientBoosting Next-Day Forecaster**: **`Macro-F1 = 0.9094`**, **`Balanced Accuracy = 0.9315`**, **`ROC-AUC = 0.9808`**.
     - **Naive Same-Day Persistence Baseline ($\hat{y}_{t+1} = y_t$)**: `Macro-F1 = 0.8883`, `Balanced Accuracy = 0.8897` (our forecaster improves Balanced Accuracy by **`+4.18%`** over persistence by capturing multi-day pollutant accumulation/dissipation trends).
2. **Explicit Ambient Heat & Air-Quality Decision Logic (`src/fusion/contextual_fusion.py`)**:
   - Both predicted next-day air-quality hazard (`p_env >= 0.50`) **and** extreme ambient heat (`ambient_temp_c >= 38.0 °C`) directly govern `AlertLevel` escalation (`MODERATE_ADVISORY` and `HIGH_RISK_ALERT`), alongside two end-to-end scenarios (`S7_Real_OOF_STRESS_S01_Delhi_PoorAQI`, `S8_Real_OOF_AEROBIC_S01_Mumbai_SatisfactoryAQI`) driven by **real out-of-fold LOSO wearable predictions** paired with **real Delhi and Mumbai chronological test-split AQI forecasts**.
   - *Scope Note*: Hydrological/flood hazards are **not** modeled in this pipeline.

---

### 6. Phase 3: Privacy-Preserving Federated Learning (`results/tables/phase3_federated_benchmark.csv`)

We report both **full 36-subject (`35` clients/round) federated tabular logistic regression** and **12-subject (`11` clients/round) federated 1D-CNN** on `PhysioNet_STRESS`:

| Algorithm / Architecture | Cohort Subjects | Clients / Round | Macro-F1 (Mean ± SD) | Balanced Acc | ROC-AUC | Comm Cost (Float32 KB) | Comm Cost (Int8 KB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized LOSO (`M0 Logistic Regression`)** | 36 | 35 | **`0.6658 ± 0.1350`** | **`0.7414`** | **`0.8235`** | `0.00 KB` | `0.00 KB` |
| **FedAvg Tabular LogReg (`100%` participation)** | 36 | 35 | `0.6573 ± 0.1342` | `0.7311` | `0.8043` | `59.06 KB` | `14.77 KB` |
| **FedProx Tabular LogReg ($\mu=0.05$, `100%`)** | 36 | 35 | `0.6584 ± 0.1332` | `0.7330` | `0.8040` | `59.06 KB` | `14.77 KB` |
| **FedAvg Tabular LogReg (`60%` partial participation)** | 36 | 21 | **`0.6588 ± 0.1337`** | `0.7327` | `0.8062` | `35.44 KB` | **`8.86 KB`** |
| **Centralized LOSO (`M2 1D-CNN`, full cohort)** | 36 | 35 | `0.5337 ± 0.0956` | `0.5602` | `0.6900` | `0.00 KB` | `0.00 KB` |
| **FedAvg `M2 1D-CNN` (12-subject subset)** | 12 | 11 | `0.4361 ± 0.1069` | `0.4841` | `0.4706` | `7,759.12 KB` | `1,939.78 KB` |
| **FedProx `M2 1D-CNN` ($\mu=0.05$, 12-subject subset)** | 12 | 11 | `0.4300 ± 0.1095` | `0.4800` | `0.4645` | `7,759.12 KB` | `1,939.78 KB` |

**Key Federated Takeaway**: Federated `M0 Logistic Regression` across all 36 stress subjects (`35` training clients per LOSO fold) achieves `Macro-F1 = 0.6588` (`ROC-AUC = 0.8062`), within `0.0070` of centralized LOSO (`0.6658`) while requiring only **`8.86 KB` of total `int8` communication** across 20 rounds. By contrast, federated training of the `12,161`-parameter `M2 1D-CNN` on a 12-subject subset fails to converge above chance (`Balanced Accuracy ~ 0.48`, `ROC-AUC = 0.46–0.47`) and requires `1,939.78 KB`, confirming that compact linear/tabular models are far better suited for small-cohort cross-subject federated learning on wearables.

---

## Reproducibility & Verification

```bash
# 1. Run all 37 unit and subject-leakage tests
pytest -v

# 2. Reproduce Phase 1 LOSO (E1), Cross-Protocol Transfer (E2), Ablations (E3), and Multi-Model Edge Profiling
python -m src.eval.run_phase1_experiments --data-dir data/raw/physionet_stress --results-dir results

# 3. Reproduce Phase 2 (Next-Day CPCB India AQI Forecasting + Out-of-Fold Wearable Fusion) & Phase 3 (Federated Benchmark)
python -m src.eval.run_phase2_phase3 --physionet-dir data/raw/physionet_stress --results-dir results

# 4. Regenerate all publication figures
python -m src.eval.plot_results --tables-dir results/tables --figures-dir results/figures
```
