import unittest
import numpy as np
import pandas as pd


def assert_no_subject_leakage(train_subjects, test_subjects, val_subjects=None):
    """Enforces zero subject intersection between train, val, and test splits.
    
    Raises:
        AssertionError: If any subject ID appears in multiple splits.
    """
    train_set = set(train_subjects)
    test_set = set(test_subjects)
    val_set = set(val_subjects) if val_subjects is not None else set()

    train_test_overlap = train_set.intersection(test_set)
    assert len(train_test_overlap) == 0, (
        f"CRITICAL DATA LEAKAGE: Subject(s) {train_test_overlap} found in both train and test sets!"
    )

    if val_set:
        train_val_overlap = train_set.intersection(val_set)
        test_val_overlap = test_set.intersection(val_set)
        assert len(train_val_overlap) == 0, (
            f"CRITICAL DATA LEAKAGE: Subject(s) {train_val_overlap} found in both train and val sets!"
        )
        assert len(test_val_overlap) == 0, (
            f"CRITICAL DATA LEAKAGE: Subject(s) {test_val_overlap} found in both test and val sets!"
        )


class TestSubjectLeakage(unittest.TestCase):
    def test_subject_leakage_detector_catches_leak(self):
        """Verify that the leakage assertion correctly identifies illegal overlaps."""
        train_subs = ["S2", "S3", "S4", "S5"]
        test_subs = ["S5", "S6"] # S5 leaked

        with self.assertRaises(AssertionError) as ctx:
            assert_no_subject_leakage(train_subs, test_subs)
        self.assertIn("CRITICAL DATA LEAKAGE", str(ctx.exception))

    def test_subject_leakage_detector_passes_valid(self):
        """Verify that disjoint subject sets pass without error."""
        train_subs = ["S2", "S3", "S4"]
        val_subs = ["S5"]
        test_subs = ["S6", "S7"]

        # Should pass cleanly without raising
        assert_no_subject_leakage(train_subs, test_subs, val_subs)


if __name__ == "__main__":
    unittest.main()

