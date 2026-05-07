#!/usr/bin/env python3
"""Run the T-MLP-u target validation set.

The final stdout line is JSON for codex-autoresearch.  The primary metric is
``failures``.  The script also reports whether the active run requested the
T-MLP-u primitive path, so a zero-failure baseline without T-MLP-u cannot be
mistaken for the requested end state.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ("01", "results/1D/cases/01_A_PE_static_interface.py"),
    ("02", "results/1D/cases/02_A_PE_advection_unified.py"),
    ("04", "results/1D/cases/04_B_acoustic_sinusoidal_air_2000Hz.py"),
    ("05", "results/1D/cases/05_B_acoustic_sinusoidal_water_6000Hz.py"),
    ("07", "results/1D/cases/07_B_acoustic_reflection_transmission.py"),
    ("13", "results/1D/cases/13_E_shocktube_hp_air_lp_water.py"),
    ("14", "results/1D/cases/14_E_shocktube_hp_water_lp_air.py"),
    ("15", "results/1D/cases/15_E_cavitation.py"),
    ("24", "results/1D/cases/24_H_hypersonic_mixture_ms10.py"),
    ("25", "results/1D/cases/25_H_hypersonic_mach10_air_water.py"),
]


def _extract_json(text: str) -> dict | None:
    out = None
    for line in text.splitlines():
        if line.startswith("CASE_JSON "):
            try:
                out = json.loads(line[len("CASE_JSON "):])
            except json.JSONDecodeError:
                pass
        elif line.startswith("ACCEPTANCE_JSON "):
            try:
                out = json.loads(line[len("ACCEPTANCE_JSON "):])
            except json.JSONDecodeError:
                pass
    return out


def main() -> int:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    scheme = env.get("FIVE_EQ_IMEX_PRIMITIVE_SCHEME", "")
    tmlpu_requested = scheme.lower() in {"tmlpu", "t-mlp-u", "t_mlp_u"}
    rows = []
    start = time.time()
    for case_id, script in TARGETS:
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(ROOT / script)],
            cwd=str(ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        payload = _extract_json(proc.stdout) or {}
        passed = proc.returncode == 0 and bool(
            payload.get("pass", payload.get("goal_reached", False))
        )
        rows.append({
            "case": case_id,
            "pass": bool(passed),
            "returncode": int(proc.returncode),
            "wall": float(time.time() - t0),
            "json": payload,
            "tail": "\n".join(proc.stdout.splitlines()[-8:]),
        })
        print(
            f"TARGET_CASE {case_id} pass={passed} rc={proc.returncode} "
            f"wall={rows[-1]['wall']:.2f}s",
            flush=True,
        )

    failures = int(sum(0 if row["pass"] else 1 for row in rows))
    result = {
        "failures": failures,
        "pass": bool(failures == 0 and tmlpu_requested),
        "case_pass_count": int(len(rows) - failures),
        "case_total": int(len(rows)),
        "tmlpu_requested": bool(tmlpu_requested),
        "primitive_scheme": scheme,
        "wall": float(time.time() - start),
        "cases": rows,
    }
    print("TARGET_JSON " + json.dumps(result, sort_keys=True, default=str))
    print(json.dumps({
        "failures": failures,
        "case_pass_count": int(len(rows) - failures),
        "tmlpu_requested": 1.0 if tmlpu_requested else 0.0,
        "wall": float(result["wall"]),
    }, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
