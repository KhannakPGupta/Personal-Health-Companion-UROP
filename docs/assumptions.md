# Research Assumptions, Protocol Specifications & Limitations

## 1. Project Context & Non-Negotiable Proxy Limitation
- **Target Definition**: This research prototype targets an on-device "Personal Health Companion" designed to provide early warnings during environmental hazards (heatwaves, floods, pollution). Because **no public wearable dataset contains real-world heat-wave exposure labels**, the predictive target is strictly a **PROXY**: detecting elevated physiological strain (psychological stress or physical exertion) versus baseline from wrist-wearable signals.
- **Explicit Clinical & Heat Disclaimer**: 
  > **WARNING / DISCLAIMER**: This system does **NOT** claim clinical validity, core body temperature estimation, heat-stroke diagnosis, or validated environmental heat-stress monitoring. It evaluates solely whether wrist physiological signals reflect general heightened sympathetic / cardiovascular exertion under laboratory protocols. This limitation must accompany all published figures, tables, and reports.

---

## 2. Dataset Licences and Citations
1. **WESAD (Wearable Stress and Affect Detection)**:
   - **Source**: UCI Machine Learning Repository (ID: 465) / University of Siegen.
   - **Citation**: Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Stockhardt, and Kristof Van Laerhoven. 2018. *Introducing WESAD, a multimodal dataset for wearable stress and affect detection*. In Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI '18).
   - **License**: Strictly non-commercial research use. Credit provided to the original authors.
   - **Device Used**: Empatica E4 wrist signals (BVP @ 64 Hz, EDA @ 4 Hz, TEMP @ 4 Hz, ACC @ 32 Hz). Chest RespiBAN signals are deliberately ignored.

2. **PhysioNet Wearable Device Dataset (Stress & Structured Exercise)**:
   - **Source**: PhysioNet (DOI: 10.13026/zzf8-xv61, version 1.0.1).
   - **Citation**: Hongn et al., *Wearable Device Dataset from Induced Stress and Structured Exercise Sessions*, PhysioNet (2023).
   - **License**: Open Data Commons Attribution License (ODC-By v1.0).
   - **Device Used**: Empatica E4 wrist signals (BVP, EDA, TEMP, ACC, tags). Exclusions defined in `data_constraints.txt` are strictly enforced.

3. **CPCB Air Quality Data in India (2015–2020) (Phase 2)**:
   - **Source**: Kaggle (rohanrao/air-quality-data-in-india), Central Pollution Control Board (CPCB) data.
   - **License**: Recorded per Kaggle dataset specification (public government data).

---

## 3. Strict Subject-Isolation & Leakage Prevention Rules
1. **Subject-Only Splitting**: No participant / subject identifier may ever appear in both training and testing folds simultaneously.
2. **Train-Fold Enclosure**: All transformations—including global z-score scalers, missing-value imputers, feature selectors, and probability calibrators (Platt scaling / isotonic regression)—must be fitted **exclusively inside the training fold** and merely transformed on the test fold.
3. **Automated Leakage Assertion**: Automated test suites in `tests/` verify that $Train_{subjects} \cap Test_{subjects} = \emptyset$ across all cross-validation folds.

---

## 4. Signal Standardization & Harmonization Assumptions
- **Wrist Accelerometer Units**:
  - PhysioNet E4 accelerometer data is sampled in units of $1/64\,\text{g}$.
  - WESAD E4 accelerometer is verified and scaled to standard $\text{g}$ units so both datasets share the exact same physical acceleration units.
- **BVP Processing**:
  - The E4's proprietary on-board `HR.csv` calculation is discarded. Both datasets extract heart rate (HR) and heart rate variability (HRV) identically from raw BVP using a 0.5–8.0 Hz bandpass filter and systolic peak detection.
- **Deep Learning Tensor Resampling**:
  - All signals for the 1D CNN are resampled to a common $32\,\text{Hz}$ rate (BVP decimated from $64\,\text{Hz}$ with an anti-aliasing filter; EDA and TEMP linearly interpolated from $4\,\text{Hz}$ to $32\,\text{Hz}$; ACC maintained at $32\,\text{Hz}$).
- **Segment Boundary Rule**:
  - Any time window that crosses or overlaps a protocol segment transition boundary is dropped to eliminate transition artifacts.

---

## 5. Key Methodological Disclosures & Experimental Scope Clarifications
1. **WESAD Dataset Availability (Rule 1 Compliance)**:
   - While the WESAD parser (`src/io/wesad_loader.py`) and label mapper (`src/windowing/label_mapper.py`) are fully implemented and unit-tested, the official University of Siegen Sciebo mirror (`uni-siegen.sciebo.de` / `ubicomp.eti.uni-siegen.de`) returns `HTTP 404`. In strict compliance with **Rule 1 (never fabricate or simulate missing datasets)**, WESAD was not evaluated in the empirical tables. All real-signal wearable benchmarks evaluate the downloaded **PhysioNet Wearable Device Dataset v1.0.1** (`100` Empatica E4 sessions: `37 STRESS`, `31 AEROBIC`, `32 ANAEROBIC`; `4,792` 60s windows and `10,334` 30s windows).
2. **Non-Equivalence of Psychological Stress and Physical Exertion (`E2` Transfer Finding)**:
   - Cross-protocol transfer between `PhysioNet_STRESS` and `PhysioNet_EXERCISE` fails (`STRESS -> EXERCISE` Macro-F1 = `0.2024` [all-baseline predictions, ROC-AUC = `0.14–0.50`]; `EXERCISE -> STRESS` Macro-F1 = `0.3562–0.5153`). Cognitive stress (subtle electrodermal SCR peaks at near-resting HR ~81–84 BPM with near-zero wrist motion) and physical exertion (HR > 105–130 BPM, large tonic sweat EDA elevation, and high wrist acceleration) are physiologically distinct regimes. Consequently, "elevated physiological strain" is **not** a single unified concept, and neither laboratory stress nor exercise can be treated as a direct physiological equivalent of environmental heat stress.
3. **Subject Calibration Assumptions (`UnsupervisedRestingPrefixCalibrated` vs. `PerSubjectBaseline_Oracle`)**:
   - **Resting-Prefix Calibration (`UnsupervisedSubjectSessionCalibrator`)**: Uses zero ground-truth labels (`y` is never accessed), standardizing each session using a blend of its initial windows and overall session statistics. **Crucial Assumption**: This assumes the user begins each recording session in a resting baseline state (which holds by construction in the PhysioNet laboratory protocols, where every session begins with seated rest, but would require a dedicated rest-onboarding prompt in real-world wear).
   - **Per-Subject Baseline Calibration (`PerSubjectBaseline_Oracle`)**: Standardizes features using the mean and standard deviation of ground-truth `label == 0` windows for each subject. Because it accesses `label == 0` on test subjects, it is strictly reported as an **Oracle Upper Bound** in the normalization ablation (`E3`).
4. **Wrist Skin Temperature (`TEMP`) & Accelerometry (`ACC`) Ablation Finding**:
   - In the sensor modality ablation (`E3`), 95% bootstrap confidence intervals across `EDA_ONLY`, `BVP_EDA`, `BVP_EDA_TEMP`, and `ALL_SENSORS` overlap substantially, and adding peripheral wrist skin temperature (`TEMP`) or raw accelerometry (`ACC`) yields no statistically significant improvement over `BVP_EDA` (`Macro-F1 = 0.6534` vs `0.6544`). For a heat-monitoring project, this is an important negative finding: short-term wrist skin temperature during indoor laboratory stress does not provide discriminative signal beyond cardiovascular and electrodermal activity.
5. **Exact Definition of False-Alert Metrics & Per-Subject Window Counts**:
   - **Persisted False-Positive Window Rate (`false_alerts_persisted_per_hr`)**: With a $60\,\text{s}$ window and $\Delta t = 30\,\text{s}$ stride, there are $120$ overlapping evaluation steps per hour of baseline recording. At each step $i$, the 3-of-5 persistence gate outputs $p_i = \mathbb{I}(\sum_{j=\max(0, i-4)}^{i} \hat{y}_j \ge 3)$. The metric $\text{FA}_{\text{win/hr}} = \frac{\sum_{i: y_i=0} p_i}{N_{\text{baseline}} \cdot (30 / 3600)}$ counts how many of the $120$ steps/hr remain in an active alarm state during ground-truth baseline (`13.9–25.9/hr` corresponds to `11.6%–21.6%` of baseline windows).
   - **Distinct False-Alarm Onset Episodes per Hour (`false_alarm_episodes_per_hr`)**: Counts distinct $0 \to 1$ onset transitions ($p_{i-1}=0 \land p_i=1 \land y_i=0$) per hour of baseline data (`~1.8–3.6` onset episodes/hr).
   - **Per-Subject Window Sample Size**: `PhysioNet_STRESS` has a median of `63.5` windows/subject (`min = 37`), whereas `PhysioNet_EXERCISE` has a median of `42.5` windows/subject (`min = 10` on short anaerobic sessions), which increases per-subject variance in the exercise cohort.
6. **Phase 2 CPCB AQI Next-Day Forecasting & Heuristic Rule Matrix**:
   - **Real CPCB Dataset & Next-Day (`t+1`) Forecasting**: `AmbientAQIPredictor` is trained and evaluated on the real **CPCB India `data/raw/india_aqi/city_day.csv`** (`29,531` daily records across `26` cities, `2015–2020`, split chronologically 80% train / 20% test per city). Rather than circular same-day lookup (`PM2.5_t -> AQI_t`), the model forecasts **next-day (`t + 1`) hazard (`AQI_{t+1} > 200`)** from causal day-$t$ pollutants and 3-day rolling lags, benchmarked against a naive same-day persistence baseline ($\hat{y}_{t+1} = y_t$).
   - **Heuristic Fusion Rules**: The Phase 2 `ContextualRiskEngine` combines physiological strain probability ($w_{\text{phys}}=0.60$), environmental hazard ($w_{\text{env}}=0.40$), wrist motion gating (`high_motion_fraction >= 0.35`), and an ambient temperature escalation threshold ($\ge 38.0^\circ\text{C}$). These weights and thresholds are **heuristic engineering rules**, not clinically fitted parameters, and **flood hazards are not modeled**.
7. **Edge Profiling Hardware Scope**:
   - Latency measurements (`latency_mean_ms`, `latency_p50_ms`, `latency_p95_ms`, `latency_p99_ms`) in `results/tables/edge_profile_m2.csv` are recorded on a **Desktop x86_64 CPU proxy**, not on a microcontroller (e.g., ARM Cortex-M4), and are reported across all trained model families (`M0`, `M1`, and `M2`).
