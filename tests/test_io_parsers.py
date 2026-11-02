"""Unit tests for IO loaders and E4 signal parsing."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.io.physionet_loader import read_e4_csv, parse_data_constraints


class TestIOParsers(unittest.TestCase):
    def test_e4_csv_parsing_and_sampling_rate(self):
        """Test parsing Empatica E4 CSV header and sampling rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "BVP.csv"
            start_ts = 1590000000.0
            fs_hz = 64.0
            
            # Format: row 0 = start timestamp, row 1 = sample rate, row 2+ = samples
            lines = [f"{start_ts}\n", f"{fs_hz}\n", "0.5\n", "-0.2\n", "1.1\n", "0.0\n"]
            with open(csv_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            ts_data = read_e4_csv(csv_path)
            self.assertEqual(ts_data.start_time, start_ts)
            self.assertEqual(ts_data.sample_rate_hz, fs_hz)
            self.assertEqual(len(ts_data.data), 4)
            self.assertAlmostEqual(float(ts_data.data[0][0]), 0.5)

    def test_parse_data_constraints(self):
        """Test parsing data_constraints.txt to exclude flagged sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            c_file = Path(tmpdir) / "data_constraints.txt"
            content = (
                "# Contaminated sessions\n"
                "S1, STRESS\n"
                "f3, AEROBIC\n"
            )
            with open(c_file, "w", encoding="utf-8") as f:
                f.write(content)

            excluded = parse_data_constraints(c_file)
            self.assertIn(("S1", "STRESS"), excluded)
            self.assertIn(("f3", "AEROBIC"), excluded)
            self.assertNotIn(("S2", "STRESS"), excluded)


if __name__ == "__main__":
    unittest.main()
