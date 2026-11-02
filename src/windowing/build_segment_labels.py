"""Build segment labels catalog and window distribution summary.

Executes Checkpoint 1 preparation:
  - Generates data/labels/segment_labels.csv
  - Computes window distribution per class, per subject, and per dataset.
  - Drops all windows overlapping protocol boundaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io.wesad_loader import WESAD_VALID_SUBJECTS, load_wesad_subject
from src.io.physionet_loader import parse_data_constraints, load_physionet_session
from src.windowing.label_mapper import (
    BinaryStrainLabel,
    SegmentBoundary,
    map_wesad_label,
    get_physionet_stress_segments,
    get_physionet_exercise_segments,
)
from src.windowing.window_generator import (
    WindowMetadata,
    find_enclosing_segment,
    generate_sliding_windows,
)


def extract_wesad_segments(labels_700hz: np.ndarray, fs_hz: float = 700.0, include_amusement: bool = True) -> list[SegmentBoundary]:
    """Convert continuous WESAD 700 Hz label series into discrete SegmentBoundary intervals."""
    segments = []
    if len(labels_700hz) == 0:
        return segments

    # Identify continuous runs of identical label codes
    diff_indices = np.where(np.diff(labels_700hz) != 0)[0]
    segment_starts = np.insert(diff_indices + 1, 0, 0)
    segment_ends = np.append(diff_indices, len(labels_700hz) - 1)

    for seg_idx, (start_idx, end_idx) in enumerate(zip(segment_starts, segment_ends)):
        raw_code = int(labels_700hz[start_idx])
        bin_label = map_wesad_label(raw_code, include_amusement_as_baseline=include_amusement)
        start_sec = start_idx / fs_hz
        end_sec = (end_idx + 1) / fs_hz
        segments.append(
            SegmentBoundary(
                segment_name=f"wesad_code_{raw_code}_{seg_idx}",
                start_sec=start_sec,
                end_sec=end_sec,
                label=bin_label,
            )
        )
    return segments


def build_wesad_windows(
    wesad_dir: Path,
    window_sec: float = 60.0,
    step_sec: float = 30.0,
    include_amusement: bool = True,
) -> pd.DataFrame:
    """Process WESAD dataset and produce window catalog."""
    rows = []
    if not wesad_dir.exists():
        return pd.DataFrame()

    print("[*] Processing WESAD subjects for segment labeling...")
    for sub_id in WESAD_VALID_SUBJECTS:
        try:
            data = load_wesad_subject(wesad_dir, sub_id)
        except Exception as err:
            continue

        total_duration = len(data.labels_700hz) / 700.0
        segments = extract_wesad_segments(data.labels_700hz, fs_hz=700.0, include_amusement=include_amusement)
        windows = generate_sliding_windows(total_duration, window_sec, step_sec)

        for w_idx, (w_start, w_end) in enumerate(windows):
            seg = find_enclosing_segment(w_start, w_end, segments)
            if seg is not None and seg.label is not None:
                rows.append({
                    "dataset": "WESAD",
                    "subject_id": sub_id,
                    "session": "WRIST",
                    "window_idx": w_idx,
                    "start_sec": round(w_start, 2),
                    "end_sec": round(w_end, 2),
                    "label": int(seg.label),
                    "label_name": "strain" if seg.label == BinaryStrainLabel.ELEVATED_STRAIN else "baseline",
                    "segment_name": seg.segment_name,
                })

    return pd.DataFrame(rows)


def build_physionet_windows(
    physionet_dir: Path,
    window_sec: float = 60.0,
    step_sec: float = 30.0,
) -> pd.DataFrame:
    """Process PhysioNet dataset and produce window catalog."""
    rows = []
    if not physionet_dir.exists():
        return pd.DataFrame()

    constraints = parse_data_constraints(physionet_dir / "data_constraints.txt")
    print("[*] Processing PhysioNet subjects for segment labeling...")

    for sess_type in ["STRESS", "AEROBIC", "ANAEROBIC"]:
        sess_dir = physionet_dir / sess_type
        if not sess_dir.exists():
            continue

        for sub_dir in sess_dir.iterdir():
            if not sub_dir.is_dir():
                continue
            sub_id = sub_dir.name
            try:
                sess_data = load_physionet_session(sub_dir, sub_id, sess_type, constraints=constraints)
                if sess_data is None or len(sess_data.tags) == 0:
                    continue

                if sess_type == "STRESS":
                    segments = get_physionet_stress_segments(sess_data.tags, sess_data.protocol_type)
                else:
                    segments = get_physionet_exercise_segments(sess_data.tags, sess_type)

                total_duration = len(sess_data.bvp.data) / sess_data.bvp.sample_rate_hz
                recording_start = sess_data.bvp.start_time

                # Generate windows aligned with recording epoch
                windows = generate_sliding_windows(total_duration, window_sec, step_sec, start_offset_sec=recording_start)

                for w_idx, (w_start, w_end) in enumerate(windows):
                    seg = find_enclosing_segment(w_start, w_end, segments)
                    if seg is not None and seg.label is not None:
                        rows.append({
                            "dataset": "PhysioNet",
                            "subject_id": sub_id,
                            "session": sess_type,
                            "window_idx": w_idx,
                            "start_sec": round(w_start - recording_start, 2),
                            "end_sec": round(w_end - recording_start, 2),
                            "label": int(seg.label),
                            "label_name": "strain" if seg.label == BinaryStrainLabel.ELEVATED_STRAIN else "baseline",
                            "segment_name": seg.segment_name,
                        })
            except Exception:
                continue

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate segment labels catalog.")
    parser.add_argument("--wesad-dir", type=Path, default=Path("data/raw/WESAD"))
    parser.add_argument("--physionet-dir", type=Path, default=Path("data/raw/physionet_wearable/1.0.1"))
    parser.add_argument("--out-csv", type=Path, default=Path("data/labels/segment_labels.csv"))
    parser.add_argument("--window-sec", type=float, default=60.0)
    parser.add_argument("--step-sec", type=float, default=30.0)
    parser.add_argument("--exclude-amusement", action="store_true", help="Exclude amusement from baseline (sensitivity analysis)")
    args = parser.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    wesad_df = build_wesad_windows(args.wesad_dir, args.window_sec, args.step_sec, include_amusement=not args.exclude_amusement)
    physio_df = build_physionet_windows(args.physionet_dir, args.window_sec, args.step_sec)

    combined_df = pd.concat([wesad_df, physio_df], ignore_index=True)

    if combined_df.empty:
        print("\n[!] No raw data found in data/raw. Creating template segment_labels.csv structure.")
        empty_df = pd.DataFrame(columns=[
            "dataset", "subject_id", "session", "window_idx", "start_sec", "end_sec", "label", "label_name", "segment_name"
        ])
        empty_df.to_csv(args.out_csv, index=False)
        print(f"[*] Template created at {args.out_csv}")
    else:
        combined_df.to_csv(args.out_csv, index=False)
        print(f"\n[*] Saved {len(combined_df)} valid non-boundary-overlapping windows to {args.out_csv}")
        print("\n=== Window Distribution Summary (Checkpoint 1) ===")
        summary = combined_df.groupby(["dataset", "session", "label_name"]).size().unstack(fill_value=0)
        print(summary)


if __name__ == "__main__":
    main()
