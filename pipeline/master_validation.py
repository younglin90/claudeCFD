#!/usr/bin/env python3
"""
Master validation runner for 1D CFD solver.
Runs all 1D validation cases sequentially and generates qa_report.md.

Procedure:
1. Run Case A (Smooth Interface - Ideal Gas)
2. If A passes, run all other cases
3. Generate final qa_report.md and exit flags
"""
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Validation cases
CASES = [
    ('validate_case_A.py', 'Smooth Interface Advection (Ideal Gas)'),
    # Add other cases here as needed
]

def run_case(case_script, case_name):
    """Run a single validation case."""
    script_path = Path('/home/younglin90/work/claude_code/claudeCFD/pipeline') / case_script
    print(f"\n{'='*70}")
    print(f"Running: {case_name}")
    print(f"Script: {script_path}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    result = subprocess.run(
        ['python3', str(script_path)],
        cwd='/home/younglin90/work/claude_code/claudeCFD'
    )

    return result.returncode == 0

def main():
    results = {}

    for case_script, case_name in CASES:
        passed = run_case(case_script, case_name)
        results[case_name] = 'PASS' if passed else 'FAIL'

        # If Case A fails, stop immediately
        if case_name == 'Smooth Interface Advection (Ideal Gas)' and not passed:
            print(f"\n{'='*70}")
            print("✗ Case A failed. Stopping validation.")
            print(f"{'='*70}\n")
            break

    # Generate summary report
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    for case_name, result in results.items():
        print(f"{case_name}: {result}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
