# Personal Health Companion: Wearable Physiological Strain Edge Detection

Research prototype for an on-device, secure Personal Health Companion designed to detect elevated physiological strain from wrist-worn wearable signals.

> [!WARNING]
> **Research Proxy Target Limitation**: No public dataset contains real-world heat-wave or flood exposure labels. Therefore, the target here is strictly an experimental **PROXY**: detecting elevated physiological strain (psychological stress or physical exertion) versus baseline from wrist signals. **This prototype makes NO claims of clinical validity or validated environmental heat-stress monitoring.** See [assumptions.md](docs/assumptions.md) for complete details.

---

## 1. Datasets Used
- **WESAD (UCI id 465 / Univ. of Siegen)**: Empatica E4 wrist signals (BVP @ 64 Hz, EDA @ 4 Hz, TEMP @ 4 Hz, ACC @ 32 Hz). Non-commercial research license.
- **PhysioNet Wearable Device Dataset (v1.0.1)**: Empatica E4 stress and exercise sessions (BVP, EDA, TEMP, ACC). Open Data Commons Attribution License (ODC-By).
- **CPCB Air Quality Data in India (Phase 2)**: City-level ambient AQI for contextual environmental fusion.

---

## 2. Repository Layout

```text
├── configs/
│   └── base.yaml              # Central experiment, windowing, and sensor configs
├── data/
│   ├── raw/                   # Raw sensor datasets (gitignored)
│   └── labels/                # Segment boundary definitions and window distributions
├── docs/
│   ├── assumptions.md         # Mandatory research disclosures, unit conversions & rules
│   └── literature_review/     # Literature review papers and benchmark matrices
├── results/
│   ├── tables/                # LOSO metric evaluation tables (macro-F1, ECE, latency)
│   └── figures/               # Confusion matrices, calibration & ROC curves
├── src/
│   ├── io/                    # Dataset loaders, integrity validators & manifests
│   ├── windowing/             # 60s/30s sliding windows with segment boundary gating
│   ├── features/              # Identical BVP HR/HRV, EDA, TEMP, and ACC feature pipelines
│   ├── models/                # Baselines (M0, M1) and edge 1D-CNN (M2 < 50k params)
│   ├── eval/                  # Strict subject-independent LOSO evaluation metrics
│   ├── edge/                  # TFLite post-training int8 quantization & latency profiling
│   ├── fusion/                # Phase 2 AQI contextual rule-matrix engine
│   └── federated/             # Phase 3 simulated FedAvg on edge clients
└── tests/
    └── test_leakage_assertion.py # Automated tests ensuring zero train/test subject leakage
```

---

## 3. Strict Validation & Anti-Leakage Protocol
- **Subject-Only Splitting**: No participant appears in both training and test sets.
- **Enclosed Preprocessing**: Imputers, scalers, selectors, and calibrators are fitted strictly within training folds.
- **Automated Verification**: Run `python -m unittest discover tests` to assert subject separation.

---

## 4. Dataset Acquisition & Verification (Any Platform)

The project includes platform-independent loaders and verification utilities:

```bash
# 1. Verify tests & leakage assertions
python -m unittest discover tests

# 2. Verify dataset availability and integrity
python src/io/verify_data.py

# 3. Automatically download WESAD (UCI / Siegen mirror)
python src/io/download.py --dataset wesad

# 4. Sync PhysioNet Wearable v1.0.1 (via wget or AWS CLI)
# Using AWS Open Data (fastest):
aws s3 sync --no-sign-request s3://physionet-open/wearable-device-dataset/1.0.1/ data/raw/physionet_wearable/1.0.1
# Or via wget:
wget -r -N -c -np https://physionet.org/files/wearable-device-dataset/1.0.1/ -P data/raw/physionet_wearable/

# 5. Run Phase 1D LOSO E1 Benchmark & Edge Quantization Profile
python src/eval/run_phase1_experiments.py

# 6. Run Phase 1E Cross-Dataset Transfer (E2) & Sensor/Window Ablations (E3)
python src/eval/ablations.py

# 7. Run Phase 2 CPCB Ambient AQI Hazard Model & Contextual Risk Fusion Engine
python src/fusion/contextual_fusion.py

# 8. Run Phase 3 Federated Edge Simulation (FedAvg & FedProx vs Centralized LOSO)
python src/federated/federated_evaluator.py
```

---

## 5. Execution Roadmap
- **Phase 0**: Repository scaffolding, experiment configuration & leakage assertions *(Completed)*.
- **Phase 1A**: Cross-platform dataset loaders, integrity verification & acquisition tools *(Completed)*.
- **Phase 1B**: Unified segment labeling & Checkpoint 1 sign-off (`data/labels/segment_labels.csv`) *(Completed)*.
- **Phase 1C**: Signal preprocessing, quality gating, normalizers & 32 Hz CNN tensor pipelines *(Completed)*.
- **Phase 1D**: Baseline models (M0, M1), edge 1D-CNN (M2 < 50k params), int8 quantizer, and LOSO evaluation engine *(Completed)*.
- **Phase 1E**: Cross-dataset transfer (`WESAD <-> PhysioNet`) & sensor/window/normalization ablations *(Completed)*.
- **Phase 2**: CPCB Ambient AQI hazard model & motion-gated contextual fusion engine *(Completed)*.
- **Phase 3**: Subject-isolated Federated Edge Learning simulation (`FedAvg` & `FedProx`) *(Completed)*.