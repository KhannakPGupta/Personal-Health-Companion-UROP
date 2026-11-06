"""Extract real tabular features and 32 Hz 6-channel tensors from PhysioNet Wearable v1.0.1.

Processes all valid non-boundary-overlapping windows across STRESS, AEROBIC, and ANAEROBIC
Empatica E4 wrist sessions, saving:
  - data/labels/real_physionet_features_60s.csv
  - data/labels/real_physionet_features_30s.csv
  - data/cache/real_physionet_tensors_60s.npz
  - data/cache/real_physionet_tensors_30s.npz
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Tuple
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io.physionet_loader import parse_data_constraints, load_physionet_session
from src.windowing.label_mapper import (
    BinaryStrainLabel,
    get_physionet_stress_segments,
    get_physionet_exercise_segments,
)
from src.windowing.window_generator import (
    find_enclosing_segment,
    generate_sliding_windows,
    extract_signal_slice,
)
from src.features.extractor import extract_window_features_and_tensor


def extract_physionet_cohort(
    physionet_dir: Path,
    window_sec: float = 60.0,
    step_sec: float = 30.0,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Extract real features and 32Hz 6-channel CNN tensors from PhysioNet v1.0.1."""
    expected_len = int(round(window_sec * 32))
    constraints = parse_data_constraints(physionet_dir / "data_constraints.txt")
    rows = []
    tensors = []

    for sess_type in ["STRESS", "AEROBIC", "ANAEROBIC"]:
        sess_dir = physionet_dir / sess_type
        if not sess_dir.exists():
            continue

        for sub_dir in sorted(sess_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            sub_id = sub_dir.name
            try:
                sess_data = load_physionet_session(sub_dir, sub_id, sess_type, constraints=constraints)
                if sess_data is None or len(sess_data.tags) == 0:
                    continue

                rec_start = sess_data.bvp.start_time
                if sess_type == "STRESS":
                    segments = get_physionet_stress_segments(
                        sess_data.tags, sess_data.protocol_type, rec_start_sec=rec_start
                    )
                else:
                    segments = get_physionet_exercise_segments(
                        sess_data.tags, sess_type, rec_start_sec=rec_start
                    )

                total_duration = len(sess_data.bvp.data) / sess_data.bvp.sample_rate_hz
                windows = generate_sliding_windows(
                    total_duration, window_sec=window_sec, step_sec=step_sec, start_offset_sec=rec_start
                )

                for w_idx, (w_start, w_end) in enumerate(windows):
                    seg = find_enclosing_segment(w_start, w_end, segments)
                    if seg is None or seg.label is None:
                        continue

                    bvp_slice = extract_signal_slice(
                        sess_data.bvp.data, sess_data.bvp.sample_rate_hz, w_start, w_end, sess_data.bvp.start_time
                    )
                    eda_slice = extract_signal_slice(
                        sess_data.eda.data, sess_data.eda.sample_rate_hz, w_start, w_end, sess_data.eda.start_time
                    )
                    temp_slice = extract_signal_slice(
                        sess_data.temp.data, sess_data.temp.sample_rate_hz, w_start, w_end, sess_data.temp.start_time
                    )
                    acc_slice = extract_signal_slice(
                        sess_data.acc.data, sess_data.acc.sample_rate_hz, w_start, w_end, sess_data.acc.start_time
                    )

                    if len(bvp_slice) < 64 or len(eda_slice) < 4 or len(temp_slice) < 4 or len(acc_slice) < 32:
                        continue

                    feats, tensor_lx6 = extract_window_features_and_tensor(
                        bvp_64hz=bvp_slice,
                        eda_4hz=eda_slice,
                        temp_4hz=temp_slice,
                        acc_32hz=acc_slice,
                        expected_tensor_length=expected_len,
                    )

                    row = {
                        "dataset": f"PhysioNet_{sess_type}",
                        "subject_id": sub_id,
                        "session": sess_type,
                        "window_idx": w_idx,
                        "start_sec": round(w_start - rec_start, 2),
                        "end_sec": round(w_end - rec_start, 2),
                        "label": int(seg.label),
                        "label_name": "strain" if seg.label == BinaryStrainLabel.ELEVATED_STRAIN else "baseline",
                        "segment_name": seg.segment_name,
                        **feats,
                    }
                    rows.append(row)
                    # Store as (6, expected_len)
                    tensors.append(np.transpose(tensor_lx6, (1, 0)))

            except Exception as err:
                print(f"[!] Skipping {sess_type}/{sub_id}: {err}")
                continue

    df = pd.DataFrame(rows)
    tensor_arr = np.stack(tensors, axis=0) if tensors else np.empty((0, 6, expected_len), dtype=np.float32)
    return df, tensor_arr


def main() -> None:
    physio_dir = Path("data/raw/physionet_wearable/1.0.1")
    labels_dir = Path("data/labels")
    cache_dir = Path("data/cache")
    labels_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Extracting 60s/30s windows from real PhysioNet Wearable v1.0.1...")
    df_60, ten_60 = extract_physionet_cohort(physio_dir, window_sec=60.0, step_sec=30.0)
    df_60.to_csv(labels_dir / "real_physionet_features_60s.csv", index=False)
    np.savez_compressed(cache_dir / "real_physionet_tensors_60s.npz", tensors=ten_60)
    print(f"[*] Saved {len(df_60)} real 60s windows ({ten_60.shape}) to data/labels/real_physionet_features_60s.csv")

    print("[*] Extracting 30s/15s windows from real PhysioNet Wearable v1.0.1...")
    df_30, ten_30 = extract_physionet_cohort(physio_dir, window_sec=30.0, step_sec=15.0)
    df_30.to_csv(labels_dir / "real_physionet_features_30s.csv", index=False)
    np.savez_compressed(cache_dir / "real_physionet_tensors_30s.npz", tensors=ten_30)
    print(f"[*] Saved {len(df_30)} real 30s windows ({ten_30.shape}) to data/labels/real_physionet_features_30s.csv")


if __name__ == "__main__":
    main()

