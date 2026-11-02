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
