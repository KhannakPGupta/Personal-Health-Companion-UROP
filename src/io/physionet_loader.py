"""Loader and parser for PhysioNet 'Wearable Device Dataset from Induced Stress and Structured Exercise Sessions' (v1.0.1).

Handles:
- Parsing row 0 (start timestamp) and row 1 (sample rate in Hz) of Empatica E4 CSVs.
- Converting ACC units from 1/64 g to standard g.
- Parsing data_constraints.txt and excluding contaminated subjects/sessions.
- Both protocol ID formats ('Sxx' and 'fxx').
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


@dataclass
class E4TimeSeries:
    start_time: float
    sample_rate_hz: float
    data: np.ndarray


@dataclass
class PhysioNetSessionData:
    subject_id: str
    session_type: str  # 'STRESS', 'AEROBIC', or 'ANAEROBIC'
    protocol_type: str # 'S' or 'f'
    bvp: E4TimeSeries
    eda: E4TimeSeries
    temp: E4TimeSeries
    acc: E4TimeSeries  # Converted to standard g
    tags: np.ndarray   # Event timestamps (epoch seconds)


def parse_data_constraints(constraints_file: Path) -> Set[Tuple[str, str]]:
    """Parse data_constraints.txt to extract flagged (subject, session) tuples to exclude.
    
    Args:
        constraints_file: Path to data_constraints.txt.
        
    Returns:
        Set of (subject_id, session_name) tuples to exclude.
    """
    excluded = set()
    if not constraints_file.exists():
        return excluded

    with open(constraints_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.replace(",", " ").split()]
            if len(parts) >= 2:
                sub_id, sess = parts[0], parts[1].upper()
                excluded.add((sub_id, sess))
    return excluded


def read_e4_csv(csv_path: Path) -> E4TimeSeries:
    """Read an Empatica E4 CSV file.
    
    Row 0: Start timestamp (UTC epoch seconds).
    Row 1: Sample frequency in Hz.
    Rows 2+: Sensor values.
    
    Args:
        csv_path: Path to CSV file.
        
    Returns:
        E4TimeSeries containing start_time, sample_rate_hz, and data array.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        second_line = f.readline().strip()

    start_time = float(first_line.split(",")[0])
    sample_rate = float(second_line.split(",")[0])

    # Load sensor samples skipping header rows
    data = pd.read_csv(csv_path, skiprows=2, header=None).values.astype(np.float32)

    return E4TimeSeries(start_time=start_time, sample_rate_hz=sample_rate, data=data)


def load_physionet_session(
    session_dir: Path,
    subject_id: str,
    session_type: str,
    constraints: Optional[Set[Tuple[str, str]]] = None,
) -> Optional[PhysioNetSessionData]:
    """Load a single PhysioNet E4 session.
    
    Args:
        session_dir: Directory containing the session CSVs.
        subject_id: Subject identifier (e.g. 'S1' or 'f1').
        session_type: 'STRESS', 'AEROBIC', or 'ANAEROBIC'.
        constraints: Set of excluded (subject_id, session_type) pairs.
        
    Returns:
        PhysioNetSessionData or None if flagged for exclusion.
    """
    if constraints and (subject_id, session_type.upper()) in constraints:
        print(f"[*] Excluding flagged session: {subject_id} - {session_type}")
        return None

    bvp = read_e4_csv(session_dir / "BVP.csv")
    eda = read_e4_csv(session_dir / "EDA.csv")
    temp = read_e4_csv(session_dir / "TEMP.csv")
    acc = read_e4_csv(session_dir / "ACC.csv")

    # Standardize ACC units: PhysioNet raw ACC is in 1/64 g units -> scale to standard g
    acc.data = acc.data / 64.0

    tags_path = session_dir / "tags.csv"
    if tags_path.exists() and tags_path.stat().st_size > 0:
        tags = pd.read_csv(tags_path, header=None).values.flatten().astype(np.float64)
    else:
        tags = np.array([], dtype=np.float64)

    protocol_type = "f" if subject_id.startswith("f") else "S"

    return PhysioNetSessionData(
        subject_id=subject_id,
        session_type=session_type.upper(),
        protocol_type=protocol_type,
        bvp=bvp,
        eda=eda,
        temp=temp,
        acc=acc,
        tags=tags,
    )
