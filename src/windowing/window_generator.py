"""Sliding window generator with strict segment boundary collision gating.

Rules:
  - Fixed-duration windows (default: 60s window, 30s step; ablatable to 30s/15s).
  - Any window that crosses, intersects, or touches a segment transition boundary is dropped.
  - Returns timestamp-synchronized slices for multi-frequency Empatica E4 wrist signals:
      - BVP (64 Hz)
      - EDA (4 Hz)
      - TEMP (4 Hz)
      - ACC (32 Hz)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple
import numpy as np
from src.windowing.label_mapper import BinaryStrainLabel, SegmentBoundary


@dataclass
class WindowMetadata:
    subject_id: str
    dataset: str
    window_idx: int
    start_sec: float
    end_sec: float
    label: int  # 0: baseline, 1: strain
    segment_name: str


def find_enclosing_segment(
    win_start: float,
    win_end: float,
    segments: List[SegmentBoundary],
    boundary_tolerance_sec: float = 0.0
) -> Optional[SegmentBoundary]:
    """Check if a time window is strictly and entirely contained within a single labeled segment.
    
    If the window crosses any segment boundary, it is dropped to prevent boundary transition artifacts.
    
    Args:
        win_start: Window start timestamp in seconds.
        win_end: Window end timestamp in seconds.
        segments: List of protocol SegmentBoundary intervals.
        boundary_tolerance_sec: Safety buffer around boundaries.
        
    Returns:
        SegmentBoundary if window is strictly inside and has a valid label; None if it crosses boundaries or is transitional.
    """
    for seg in segments:
        # Check strict containment inside this segment
        if (win_start >= seg.start_sec + boundary_tolerance_sec) and (win_end <= seg.end_sec - boundary_tolerance_sec):
            if seg.label is not None:
                return seg
            else:
                # Inside a transitional/dropped segment (e.g. warm-up)
                return None

    # If it falls between segments or spans across a boundary
    return None


def generate_sliding_windows(
    total_duration_sec: float,
    window_sec: float = 60.0,
    step_sec: float = 30.0,
    start_offset_sec: float = 0.0
) -> List[Tuple[float, float]]:
    """Generate (start_sec, end_sec) tuples for uniform sliding windows.
    
    Args:
        total_duration_sec: Total duration of the recording in seconds.
        window_sec: Window length in seconds (default: 60.0).
        step_sec: Step / stride length in seconds (default: 30.0).
        start_offset_sec: Starting timestamp offset.
        
    Returns:
        List of (win_start, win_end) timestamp pairs.
    """
    windows = []
    current_start = start_offset_sec
    while current_start + window_sec <= start_offset_sec + total_duration_sec:
        windows.append((current_start, current_start + window_sec))
        current_start += step_sec
    return windows


def extract_signal_slice(
    signal: np.ndarray,
    fs_hz: float,
    win_start_sec: float,
    win_end_sec: float,
    recording_start_sec: float = 0.0
) -> np.ndarray:
    """Extract slice of sensor samples corresponding to a time window.
    
    Args:
        signal: Array of sensor samples (1D or 2D).
        fs_hz: Sampling frequency in Hz.
        win_start_sec: Window start relative to epoch or recording.
        win_end_sec: Window end relative to epoch or recording.
        recording_start_sec: Timestamp offset of the first sample in `signal`.
        
    Returns:
        Sliced numpy array of sensor samples.
    """
    start_idx = int(round((win_start_sec - recording_start_sec) * fs_hz))
    end_idx = int(round((win_end_sec - recording_start_sec) * fs_hz))

    start_idx = max(0, start_idx)
    end_idx = min(len(signal), end_idx)

    return signal[start_idx:end_idx]
