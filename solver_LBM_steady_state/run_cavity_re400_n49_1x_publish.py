#!/usr/bin/env python3
"""Run lid-driven cavity Re=400, N=49 (1x) and export paper-ready artifacts.

This is a thin wrapper around the existing Re=100 cavity production runner.
It rebinds the case parameters to Re=400 / N=49 while preserving the same
artifact protocol:
- Picard reference for full-field comparison
- Ghia 1982 centerline comparison
- residual_vs_iteration / residual_vs_wall_seconds / normalized history
- VTK / CSV / PNG outputs
"""

from __future__ import annotations

import sys
from pathlib import Path

import run_cavity_re100_n33_1x_publish as base


def configure():
    base.CASE_ID = "cavity_re400_n49"
    base.CASE_LABEL = "Lid-driven cavity Re=400 N=49"
    base.RE = 400
    base.N = 49
    base.U_WALL = 0.1
    base.TOL = 5e-7

    base.OUT_ROOT = Path("papers_data") / "lid_driven_Re400_N49__1x"
    base.FIELD_DIR = base.OUT_ROOT / "fields"
    base.FIG_DIR = base.OUT_ROOT / "figure"
    base.HIST_DIR = base.OUT_ROOT / "histories"
    base.VTK_DIR = base.OUT_ROOT / "vtk"


def main(argv=None):
    configure()
    methods = base.parse_methods(",".join(base.METHODS))
    base.run_case(methods=methods, do_clean=True)

    plan_text = (
        "# Proposed Solver Optimization Plan (60/10/30)\n\n"
        "1. Wall-time first (60%): keep the single SafeNN pipeline, reduce late-stage expensive checks, and reject unstable candidates early.\n"
        "2. Simplicity (10%): preserve one algorithmic path with fixed default coefficients and only grid-based scaling.\n"
        "3. Accuracy (30%): keep monotone polish and reject candidates that worsen centerline wake error relative to the tight reference.\n"
        "4. Data integrity: use only hash-matched caches, strict-monotone wall histories, and summary-final-point consistency checks.\n"
    )
    (base.OUT_ROOT / "proposed_optimization_plan.md").write_text(plan_text, encoding="utf-8")
    print(f"[saved] {base.OUT_ROOT / 'proposed_optimization_plan.md'}")


if __name__ == "__main__":
    main(sys.argv[1:])
