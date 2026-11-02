"""Loader and validator for the WESAD dataset (UCI ML Repository id 465).

Extracts wrist Empatica E4 signals (BVP 64Hz, EDA 4Hz, TEMP 4Hz, ACC 32Hz).
Deliberately ignores chest RespiBAN signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


WESAD_VALID_SUBJECTS = [
    "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
    "S10", "S11", "S13", "S14", "S15", "S16", "S17"
] # Note: S12 was excluded by original authors due to sensor malfunction


@dataclass
class WesadWristSubjectData:
    subject_id: str
    bvp: np.ndarray        # 64 Hz (N_bvp, 1)
    eda: np.ndarray        # 4 Hz (N_eda, 1)
    temp: np.ndarray       # 4 Hz (N_temp, 1)
    acc: np.ndarray        # 32 Hz (N_acc, 3) in g
    labels_700hz: np.ndarray # 700 Hz ground truth label sequence


def load_wesad_subject(subject_dir: Path, subject_id: str) -> WesadWristSubjectData:
    """Load a single WESAD subject's wrist data from pickle file.
    
    Args:
        subject_dir: Path to directory containing subject folder or pickle (e.g. data/raw/WESAD/S2).
        subject_id: Identifier string (e.g. 'S2').
        
    Returns:
        WesadWristSubjectData with validated wrist channels and ground truth labels.
        
    Raises:
        FileNotFoundError: If the subject pickle cannot be located.
        KeyError: If wrist signals are missing from dictionary.
    """
    pkl_path = subject_dir / f"{subject_id}.pkl"
    if not pkl_path.exists():
        # Try nested structure: subject_dir / subject_id / f"{subject_id}.pkl"
        nested_path = subject_dir / subject_id / f"{subject_id}.pkl"
        if nested_path.exists():
            pkl_path = nested_path
        else:
            raise FileNotFoundError(f"WESAD pickle file not found for {subject_id} at {pkl_path}")

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    if "signal" not in raw or "wrist" not in raw["signal"]:
        raise KeyError(f"Corrupted WESAD pickle for {subject_id}: missing 'signal' -> 'wrist'")

    wrist = raw["signal"]["wrist"]
    labels = raw["label"].flatten()

    # Wrist E4 channels:
    bvp = np.asarray(wrist["BVP"], dtype=np.float32)
    eda = np.asarray(wrist["EDA"], dtype=np.float32)
    temp = np.asarray(wrist["TEMP"], dtype=np.float32)
    acc = np.asarray(wrist["ACC"], dtype=np.float32)

    # Note on WESAD ACC units: Empatica E4 measures in [-2g, +2g] with range [-64, 64] -> 1/32 or 1/64 g.
    # WESAD authors document ACC units as [g] or scaled raw integers. We standardize ACC to standard g:
    # If values exceed 4g, they are raw counts scaled by 64.
    if np.max(np.abs(acc)) > 16.0:
        acc = acc / 64.0

    return WesadWristSubjectData(
        subject_id=subject_id,
        bvp=bvp,
        eda=eda,
        temp=temp,
        acc=acc,
        labels_700hz=labels
    )
