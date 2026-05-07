#!/usr/bin/env python3
"""Regression guard for dense two-phase cases already accepted by the user."""
from __future__ import annotations

import json
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE_SCRIPTS = [
    ("01_A", "results/1D/cases/01_A_PE_static_interface.py"),
    ("02_A", "results/1D/cases/02_A_PE_advection_unified.py"),
    ("07_B", "results/1D/cases/07_B_acoustic_reflection_transmission.py"),
    ("13_E", "results/1D/cases/13_E_shocktube_hp_air_lp_water.py"),
    ("25_H", "results/1D/cases/25_H_hypersonic_mach10_air_water.py"),
]


def main():
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    rows = []
    for name, rel in CASE_SCRIPTS:
        proc = subprocess.run(
            [sys.executable, os.path.join(ROOT, rel)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        case_json = None
        for line in proc.stdout.splitlines():
            if line.startswith("CASE_JSON "):
                try:
                    case_json = json.loads(line[len("CASE_JSON "):])
                except json.JSONDecodeError:
                    case_json = None
        rows.append({
            "case": name,
            "returncode": proc.returncode,
            "pass": proc.returncode == 0 and bool(case_json.get("pass") if case_json else False),
            "case_json": case_json,
        })
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    failures = int(sum(0 if row["pass"] else 1 for row in rows))
    result = {"guard_failures": failures, "pass": failures == 0, "cases": rows}
    print("GUARD_JSON " + json.dumps(result, sort_keys=True))
    print(failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
