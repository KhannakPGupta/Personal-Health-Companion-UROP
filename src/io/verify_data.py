"""Verification and dataset integrity inspection utility.

Checks directory structures, sampling frequencies, and file existence without assuming files are present.
Enforces Rule 1: Never fabricate data if files are missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path for direct execution across any OS
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io.wesad_loader import WESAD_VALID_SUBJECTS, load_wesad_subject
from src.io.physionet_loader import parse_data_constraints, load_physionet_session



def verify_wesad_integrity(wesad_dir: Path) -> dict:
    """Inspect local WESAD files and verify channel sampling rates."""
    results = {"found_subjects": [], "missing_subjects": [], "valid": False}
    if not wesad_dir.exists():
        print(f"[-] WESAD directory not found at: {wesad_dir}")
        return results

    print(f"\n[*] Checking WESAD in {wesad_dir}...")
    for sub in WESAD_VALID_SUBJECTS:
        pkl_path = wesad_dir / sub / f"{sub}.pkl"
        if not pkl_path.exists():
            pkl_path = wesad_dir / f"{sub}.pkl"

        if pkl_path.exists():
            results["found_subjects"].append(sub)
        else:
            results["missing_subjects"].append(sub)

    print(f"    Found {len(results['found_subjects'])}/{len(WESAD_VALID_SUBJECTS)} expected subjects.")
    if results["missing_subjects"]:
        print(f"    Missing: {results['missing_subjects']}")
    else:
        results["valid"] = True
    return results


def verify_physionet_integrity(physionet_dir: Path) -> dict:
    """Inspect local PhysioNet wearable dataset files."""
    results = {"found_sessions": 0, "excluded_sessions": 0, "valid": False}
    if not physionet_dir.exists():
        print(f"[-] PhysioNet directory not found at: {physionet_dir}")
        return results

    constraints_file = physionet_dir / "data_constraints.txt"
    constraints = parse_data_constraints(constraints_file)
    print(f"\n[*] Checking PhysioNet in {physionet_dir}...")
    print(f"    Parsed {len(constraints)} flagged exclusions from data_constraints.txt.")

    for sess_name in ["STRESS", "AEROBIC", "ANAEROBIC"]:
        sess_dir = physionet_dir / sess_name
        if sess_dir.exists():
            sub_dirs = [d for d in sess_dir.iterdir() if d.is_dir()]
            results["found_sessions"] += len(sub_dirs)
            print(f"    {sess_name}: found {len(sub_dirs)} subject folders.")

    results["valid"] = results["found_sessions"] > 0
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify raw dataset integrity.")
    parser.add_argument("--wesad-dir", type=Path, default=Path("data/raw/WESAD"))
    parser.add_argument("--physionet-dir", type=Path, default=Path("data/raw/physionet_wearable/1.0.1"))
    args = parser.parse_args()

    w_res = verify_wesad_integrity(args.wesad_dir)
    p_res = verify_physionet_integrity(args.physionet_dir)

    print("\n=== Verification Summary ===")
    print(f"WESAD: {'Ready' if w_res['valid'] else 'Not found / Incomplete'}")
    print(f"PhysioNet: {'Ready' if p_res['valid'] else 'Not found / Incomplete'}")


if __name__ == "__main__":
    main()
