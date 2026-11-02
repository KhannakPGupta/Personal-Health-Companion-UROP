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


def get_physionet_stress_segments(tags: np.ndarray, protocol_type: str = "S") -> List[SegmentBoundary]:
    """Build protocol segment boundaries for PhysioNet STRESS sessions using tags.csv.
    
    PhysioNet Stress Protocol:
      - Initial Baseline / Rest -> Baseline (0)
      - Stressor Blocks (Stroop, Mental Arithmetic, Cold Pressor / Public Speaking) -> Strain (1)
      - Intermediate Rest Periods -> Baseline (0)
      
    Args:
        tags: Timestamps of tag markers pressed during experiment.
        protocol_type: 'S' (standard) or 'f' (variant protocol).
        
    Returns:
        List of SegmentBoundary objects with assigned binary labels.
    """
    segments: List[SegmentBoundary] = []
    if len(tags) < 2:
        return segments

    # Protocol segment partitioning based on tag markers
    # Tag 0: start baseline; Tag 1: start stress task 1; etc.
    for i in range(len(tags) - 1):
        start = tags[i]
        end = tags[i + 1]
        # Alternating task vs rest blocks typical in the Trier Social Stress Test (TSST) structure
        # Even segments (0, 2, ...): baseline/rest; Odd segments (1, 3, ...): stressors
        if i % 2 == 0:
            segments.append(SegmentBoundary(f"rest_block_{i}", start, end, BinaryStrainLabel.BASELINE))
        else:
            segments.append(SegmentBoundary(f"stress_task_{i}", start, end, BinaryStrainLabel.ELEVATED_STRAIN))

    return segments


def get_physionet_exercise_segments(tags: np.ndarray, session_type: str = "AEROBIC") -> List[SegmentBoundary]:
    """Build protocol segment boundaries for PhysioNet AEROBIC / ANAEROBIC exercise sessions.
    
    Protocol:
      - Baseline seated rest -> Baseline (0)
      - Warm-up -> Transitional (None / dropped)
      - Pedalling with load & sprints -> Elevated Strain (1)
      - No-resistance cooldown -> Transitional (None / dropped)
      - Final seated recovery rest -> Baseline (0)
      
    Args:
        tags: Event tag timestamps.
        session_type: 'AEROBIC' or 'ANAEROBIC'.
        
    Returns:
        List of SegmentBoundary objects.
    """
    segments: List[SegmentBoundary] = []
    if len(tags) < 4:
        return segments

    # Minimum 4 tags expected:
    # [t0, t1]: Initial baseline rest -> Baseline (0)
    # [t1, t2]: Warm-up -> Dropped (None)
    # [t2, t3]: Loaded exercise / sprints -> Strain (1)
    # [t3, t4]: Cooldown -> Dropped (None)
    # [t4, ...]: Final seated recovery -> Baseline (0)
    
    segments.append(SegmentBoundary("initial_baseline", tags[0], tags[1], BinaryStrainLabel.BASELINE))
    segments.append(SegmentBoundary("warmup_transitional", tags[1], tags[2], None))
    segments.append(SegmentBoundary("exercise_strain", tags[2], tags[3], BinaryStrainLabel.ELEVATED_STRAIN))

    if len(tags) >= 5:
        segments.append(SegmentBoundary("cooldown_transitional", tags[3], tags[4], None))
        if len(tags) >= 6:
            segments.append(SegmentBoundary("final_recovery", tags[4], tags[5], BinaryStrainLabel.BASELINE))

    return segments
