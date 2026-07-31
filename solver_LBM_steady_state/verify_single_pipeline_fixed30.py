"""Strict fixed30 verifier plus proposed-solver single-pipeline checks.

This wrapper keeps the existing fixed30 benchmark and reference caches intact,
then adds a static audit for reviewer-risk case-specific proposed logic.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROPOSED = ROOT / "solver_proposed_single.py"


FORBIDDEN_SINGLE_PIPELINE_PATTERNS = {
    "class-name dispatch": "case.__class__.__name__",
    "kolmogorov-specific branch": "KolmogorovCase",
    "channel-specific branch": "ChannelCase",
    "couette-specific branch": "CouetteCase",
    "cavity-specific branch": "LBMCavity",
    "plbe cavity branch": "PLBECavity",
    "stiff cavity warm-start": "stiff_cavity_warm",
    "preconditioned warm-start call": "solve_preconditioned_lbm(",
    "large mask helper": "_large_mask_warm_safe_nn(",
    "porosity threshold": "porosity <",
    "case-specific Re branch": 'getattr(case, "Re"',
}


def run_fixed30() -> dict:
    proc = subprocess.run(
        [sys.executable, "verify_fixed30_scaling_strict.py"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return {
            "score": 0.0,
            "all_pass": 0,
            "verify_failed": 1,
            "verify_returncode": proc.returncode,
            "verify_error": "no stdout from fixed30 verifier",
        }
    try:
        data = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        return {
            "score": 0.0,
            "all_pass": 0,
            "verify_failed": 1,
            "verify_returncode": proc.returncode,
            "verify_error": f"failed to parse fixed30 JSON: {exc}",
        }
    data["verify_failed"] = int(proc.returncode != 0)
    data["verify_returncode"] = int(proc.returncode)
    return data


def audit_single_pipeline() -> dict:
    text = PROPOSED.read_text(encoding="utf-8")
    hits = [
        {"name": name, "pattern": pattern}
        for name, pattern in FORBIDDEN_SINGLE_PIPELINE_PATTERNS.items()
        if pattern in text
    ]
    return {
        "case_specific_conflicts": len(hits),
        "single_pipeline_pass": int(len(hits) == 0),
        "case_specific_hits": hits,
    }


def main() -> int:
    metrics = run_fixed30()
    metrics.update(audit_single_pipeline())
    metrics["all_pass_single_pipeline"] = int(
        metrics.get("all_pass") == 1 and metrics.get("single_pipeline_pass") == 1
    )
    metrics["pipeline_score"] = float(metrics.get("score", 0.0)) - 250.0 * float(
        metrics["case_specific_conflicts"]
    ) + 1000.0 * float(metrics["single_pipeline_pass"])
    print(json.dumps(metrics, allow_nan=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
