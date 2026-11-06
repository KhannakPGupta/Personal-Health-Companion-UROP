"""Unified binary label mapping for WESAD and PhysioNet datasets.

Unified Target Definition:
  - 1 = Elevated Physiological Strain (psychological stress or physical exertion)
  - 0 = Baseline (resting baseline or amusement)
  - None / Dropped = Transitional periods, warm-ups, cool-downs, meditation, undefined codes.

Rules:
  - Any window overlapping a segment boundary must be strictly dropped.
  - Supports sensitivity analysis variant with amusement excluded from baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple
import numpy as np


class BinaryStrainLabel(IntEnum):
    BASELINE = 0
    ELEVATED_STRAIN = 1


@dataclass(frozen=True)
class SegmentBoundary:
    """Represents a protocol segment with start and end times in epoch or relative seconds."""
    segment_name: str
    start_sec: float
    end_sec: float
    label: Optional[BinaryStrainLabel]  # None indicates transitional/to be dropped


# WESAD Protocol label codes from README (stored at 700 Hz)
# 0 = not defined / transient
# 1 = baseline
# 2 = stress
# 3 = amusement
# 4 = meditation
# 5, 6, 7 = ignore / study pauses
WESAD_RAW_LABELS = {
    0: "not_defined",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
    5: "ignore_5",
    6: "ignore_6",
    7: "ignore_7",
}


def map_wesad_label(raw_code: int, include_amusement_as_baseline: bool = True) -> Optional[BinaryStrainLabel]:
    """Map raw WESAD 700 Hz label integer to unified binary strain label.
    
    Args:
        raw_code: Integer label code (0 to 7).
        include_amusement_as_baseline: If True, code 3 (amusement) is mapped to 0 (baseline).
                                       If False, code 3 is dropped (sensitivity variant).
                                       
    Returns:
        BinaryStrainLabel.ELEVATED_STRAIN (1), BinaryStrainLabel.BASELINE (0), or None (dropped).
    """
    if raw_code == 2:
        return BinaryStrainLabel.ELEVATED_STRAIN
    elif raw_code == 1:
        return BinaryStrainLabel.BASELINE
    elif raw_code == 3:
        return BinaryStrainLabel.BASELINE if include_amusement_as_baseline else None
    else:
        # Codes 0, 4, 5, 6, 7 are dropped
        return None


def get_physionet_stress_segments(
    tags: np.ndarray,
    protocol_type: str = "S",
    rec_start_sec: Optional[float] = None,
) -> List[SegmentBoundary]:
    """Build protocol segment boundaries for PhysioNet STRESS sessions using tags.csv.
    
    PhysioNet Stress Protocol (Hong et al. / Empatica E4 v1.0.1):
      - Protocol 'S' (v1, S01..S18, ~12 tags):
        * [rec_start, tags[0]] & [tags[0], tags[1]]: Initial Baseline Rest -> Baseline (0)
        * [tags[1], tags[3]]: Stroop + Trier Mental Challenge Test (TMCT) -> Strain (1)
        * [tags[3], tags[4]]: First 5-min Recovery Rest -> Baseline (0)
        * [tags[4], tags[5]]: Public Speaking / Opposite Opinion Stress Task -> Strain (1)
        * [tags[5], tags[6]]: Second 5-min Recovery Rest -> Baseline (0)
        * [tags[6], tags[-1]]: Mental Arithmetic (Subtracting 13 countdown blocks) -> Strain (1)
      - Protocol 'f' (v2, f01..f18, ~8 tags):
        * [tags[0], tags[1]]: Initial Baseline Rest -> Baseline (0)
        * [tags[1], tags[2]]: Stress Task Block 1 (TMCT / Stroop) -> Strain (1)
        * [tags[2], tags[3]]: First Recovery Rest -> Baseline (0)
        * [tags[3], tags[6]]: Stress Task Block 2 (Speaking + Mental Arithmetic) -> Strain (1)
        * [tags[6], tags[7]]: Second Recovery Rest -> Baseline (0)
    """
    segments: List[SegmentBoundary] = []
    if len(tags) < 2:
        return segments

    if rec_start_sec is not None and (tags[0] - rec_start_sec) >= 60.0:
        segments.append(SegmentBoundary("pre_tag_baseline", float(rec_start_sec), float(tags[0]), BinaryStrainLabel.BASELINE))

    if str(protocol_type).startswith("S") and len(tags) >= 7:
        segments.append(SegmentBoundary("rest_block_0", float(tags[0]), float(tags[1]), BinaryStrainLabel.BASELINE))
        segments.append(SegmentBoundary("stress_stroop_tmct", float(tags[1]), float(tags[3]), BinaryStrainLabel.ELEVATED_STRAIN))
        segments.append(SegmentBoundary("rest_recovery_1", float(tags[3]), float(tags[4]), BinaryStrainLabel.BASELINE))
        segments.append(SegmentBoundary("stress_speaking", float(tags[4]), float(tags[5]), BinaryStrainLabel.ELEVATED_STRAIN))
        segments.append(SegmentBoundary("rest_recovery_2", float(tags[5]), float(tags[6]), BinaryStrainLabel.BASELINE))
        if tags[-1] > tags[6]:
            segments.append(SegmentBoundary("stress_arithmetic", float(tags[6]), float(tags[-1]), BinaryStrainLabel.ELEVATED_STRAIN))
        return segments

    if str(protocol_type).startswith("f") and len(tags) >= 7:
        segments.append(SegmentBoundary("rest_block_0", float(tags[0]), float(tags[1]), BinaryStrainLabel.BASELINE))
        segments.append(SegmentBoundary("stress_task_1", float(tags[1]), float(tags[2]), BinaryStrainLabel.ELEVATED_STRAIN))
        segments.append(SegmentBoundary("rest_recovery_1", float(tags[2]), float(tags[3]), BinaryStrainLabel.BASELINE))
        segments.append(SegmentBoundary("stress_task_2", float(tags[3]), float(tags[6]), BinaryStrainLabel.ELEVATED_STRAIN))
        segments.append(SegmentBoundary("rest_recovery_2", float(tags[6]), float(tags[7]), BinaryStrainLabel.BASELINE))
        return segments

    # Fallback alternating task vs rest blocks
    for i in range(len(tags) - 1):
        start = float(tags[i])
        end = float(tags[i + 1])
        if i % 2 == 0:
            segments.append(SegmentBoundary(f"rest_block_{i}", start, end, BinaryStrainLabel.BASELINE))
        else:
            segments.append(SegmentBoundary(f"stress_task_{i}", start, end, BinaryStrainLabel.ELEVATED_STRAIN))

    return segments


def get_physionet_exercise_segments(
    tags: np.ndarray,
    session_type: str = "AEROBIC",
    rec_start_sec: Optional[float] = None,
) -> List[SegmentBoundary]:
    """Build protocol segment boundaries for PhysioNet AEROBIC / ANAEROBIC exercise sessions.
    
    Protocol:
      - Pre-tag & initial seated rest -> Baseline (0)
      - Warm-up -> Transitional (None / dropped)
      - Incremental loaded pedalling & maximal sprints -> Elevated Strain (1)
      - Cooldown -> Transitional (None / dropped)
    """
    segments: List[SegmentBoundary] = []
    if len(tags) < 4:
        return segments

    if rec_start_sec is not None and (tags[0] - rec_start_sec) >= 60.0:
        segments.append(SegmentBoundary("pre_exercise_baseline", float(rec_start_sec), float(tags[0]), BinaryStrainLabel.BASELINE))

    segments.append(SegmentBoundary("initial_baseline", float(tags[0]), float(tags[1]), BinaryStrainLabel.BASELINE))
    segments.append(SegmentBoundary("warmup_transitional", float(tags[1]), float(tags[2]), None))

    if rec_start_sec is not None and len(tags) >= 5:
        # In real PhysioNet Stationary Bike protocol, tags[2]..tags[-2] are incremental RPM / sprint stages
        strain_end = float(tags[-2]) if len(tags) >= 6 else float(tags[-1])
        segments.append(SegmentBoundary("exercise_strain", float(tags[2]), strain_end, BinaryStrainLabel.ELEVATED_STRAIN))
        if len(tags) >= 6:
            segments.append(SegmentBoundary("cooldown_transitional", float(tags[-2]), float(tags[-1]), None))
        return segments

    segments.append(SegmentBoundary("exercise_strain", float(tags[2]), float(tags[3]), BinaryStrainLabel.ELEVATED_STRAIN))

    if len(tags) >= 5:
        segments.append(SegmentBoundary("cooldown_transitional", float(tags[3]), float(tags[4]), None))
        if len(tags) >= 6:
            segments.append(SegmentBoundary("final_recovery", float(tags[4]), float(tags[5]), BinaryStrainLabel.BASELINE))

    return segments
