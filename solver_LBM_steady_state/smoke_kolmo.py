"""Smoke test: 6 methods on Kolmogorov N=32 only."""

from __future__ import annotations

import json
import time

import numpy as np

from v2_run_all import _patched_run_method, case_macro, make_case, picard_tail
from solver_unified_safe_nn import _residual_norm


def main():
    CID = "kolmogorov"
    N = 32
    TOL = 1e-8
    BUDGET = 100000

    methods = [
        ("picard_lbm", "Baseline Picard"),
        ("anderson_lbm", "Anderson [Walker-Ni 2011]"),
        ("preconditioned_lbm", "Preconditioned LBM [PRE 70, Guo-Zhao-Shi]"),
        ("inexact_newton_lbe", "Inexact Newton [Huang-Yang-Cai 2017]"),
        ("dual_time_mg_lbm", "Dual-time MG [Jia-Luo 2026]"),
        ("proposed", "Safe-NN++ (proposed)"),
    ]

    case_ref = make_case(CID, N)[0]
    ux_an = case_ref.analytical_ux()
    if ux_an.ndim == 1:
        ux_an = np.tile(ux_an[:, None], (1, N))
    rows = []
    for mid, label in methods:
        case = make_case(CID, N)[0]
        t0 = time.perf_counter()
        f, hist, _ = _patched_run_method(mid, case, TOL, BUDGET, verbose=False)
        accel_lbe = int(hist[-1][2]) if hist else 0
        # paper-faithful tail
        f2, tail_hist, vchg = picard_tail(case, f, max_steps=BUDGET)
        tail_lbe = int(tail_hist[-1][2]) if tail_hist else 0
        wall = time.perf_counter() - t0
        _, ux, uy = case_macro(case, f2)
        du = ux - ux_an
        rel_l2 = float(np.sqrt(np.sum(du * du)) / max(np.sqrt(np.sum(ux_an * ux_an)), 1e-30))
        _, nres = _residual_norm(case, f2)
        rows.append({
            "method": mid, "label": label,
            "accel_lbe": accel_lbe, "tail_lbe": tail_lbe,
            "total_lbe": accel_lbe + tail_lbe,
            "wall_s": wall, "native_res": float(nres),
            "vel_chg": float(vchg),
            "rel_L2_vs_analytic": rel_l2,
            "converged": bool(np.isfinite(vchg) and vchg < 1e-6),
        })

    base = next(r for r in rows if r["method"] == "picard_lbm")["total_lbe"]
    print(f"\n=== Kolmogorov N={N} (analytic ref) ===")
    print(f"{'method':40s} | {'LBE':>10s} | LBEspeed | {'rel_L2':>10s} | conv")
    for r in rows:
        sp = f"{base / r['total_lbe']:.2f}x" if r["total_lbe"] else "-"
        print(f"{r['label']:40s} | {r['total_lbe']:>10d} | {sp:>8s} | "
              f"{r['rel_L2_vs_analytic']:>10.3e} | {'Y' if r['converged'] else 'N'}")
    with open("paper_revision_data/v2_final/smoke_kolmogorov_n32.json", "w") as fh:
        json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
