"""Unit tests for windowing, segment boundary collision drop rule, and label mappings."""

import unittest
import numpy as np
from src.windowing.label_mapper import (
    BinaryStrainLabel,
    SegmentBoundary,
    map_wesad_label,
    get_physionet_exercise_segments,
)
from src.windowing.window_generator import (
    find_enclosing_segment,
    generate_sliding_windows,
    extract_signal_slice,
)


class TestWindowingAndLabels(unittest.TestCase):
    def test_wesad_label_mapping(self):
        """Test standard and sensitivity label mappings for WESAD."""
        # Baseline = 1 -> 0
        self.assertEqual(map_wesad_label(1), BinaryStrainLabel.BASELINE)
        # Stress = 2 -> 1
        self.assertEqual(map_wesad_label(2), BinaryStrainLabel.ELEVATED_STRAIN)
        # Amusement = 3 -> 0 (default)
        self.assertEqual(map_wesad_label(3, include_amusement_as_baseline=True), BinaryStrainLabel.BASELINE)
        # Amusement = 3 -> None (sensitivity variant: excluded)
        self.assertIsNone(map_wesad_label(3, include_amusement_as_baseline=False))
        # Non-target codes (meditation, undefined) are dropped
        self.assertIsNone(map_wesad_label(0))
        self.assertIsNone(map_wesad_label(4))
        self.assertIsNone(map_wesad_label(5))

    def test_sliding_window_counts(self):
        """Test number of generated sliding windows."""
        # 120 seconds total, 60s window, 30s step -> starts at 0, 30, 60 -> 3 windows
        windows = generate_sliding_windows(total_duration_sec=120.0, window_sec=60.0, step_sec=30.0)
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0], (0.0, 60.0))
        self.assertEqual(windows[1], (30.0, 90.0))
        self.assertEqual(windows[2], (60.0, 120.0))

    def test_boundary_collision_dropped(self):
        """Verify strict rule: Any window overlapping a segment boundary must be dropped."""
        # Segment 1: [0, 100] -> Baseline (0)
        # Segment 2: [100, 200] -> Strain (1)
        segments = [
            SegmentBoundary("baseline_seg", 0.0, 100.0, BinaryStrainLabel.BASELINE),
            SegmentBoundary("strain_seg", 100.0, 200.0, BinaryStrainLabel.ELEVATED_STRAIN),
        ]

        # Window [20, 80] is strictly inside Segment 1 -> Kept as Baseline
        seg_inside = find_enclosing_segment(20.0, 80.0, segments)
        self.assertIsNotNone(seg_inside)
        self.assertEqual(seg_inside.label, BinaryStrainLabel.BASELINE)

        # Window [80, 140] spans across boundary at t=100 -> MUST BE DROPPED
        seg_crossing = find_enclosing_segment(80.0, 140.0, segments)
        self.assertIsNone(seg_crossing)

    def test_exercise_transitional_periods_dropped(self):
        """Verify warm-up and cooldown periods in exercise sessions have label=None (dropped)."""
        tags = np.array([0.0, 120.0, 240.0, 600.0, 720.0, 900.0])
        segments = get_physionet_exercise_segments(tags, "AEROBIC")

        # Segment 0: Baseline (0)
        self.assertEqual(segments[0].label, BinaryStrainLabel.BASELINE)
        # Segment 1: Warmup -> Dropped
        self.assertIsNone(segments[1].label)
        # Segment 2: Exercise -> Strain (1)
        self.assertEqual(segments[2].label, BinaryStrainLabel.ELEVATED_STRAIN)
        # Segment 3: Cooldown -> Dropped
        self.assertIsNone(segments[3].label)

    def test_signal_slicing(self):
        """Test precise multi-rate signal slicing."""
        fs = 32.0 # 32 Hz (ACC)
        # 10 seconds of signal = 320 samples
        sig = np.arange(320)
        # Extract window from 2.0s to 5.0s (3.0s * 32 Hz = 96 samples)
        window_slice = extract_signal_slice(sig, fs_hz=fs, win_start_sec=2.0, win_end_sec=5.0)
        self.assertEqual(len(window_slice), 96)
        self.assertEqual(window_slice[0], 64)
        self.assertEqual(window_slice[-1], 159)


if __name__ == "__main__":
    unittest.main()
