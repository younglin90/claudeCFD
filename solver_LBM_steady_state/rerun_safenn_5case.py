"""Re-run Safe-NN-SCMK (paper-aligned defaults) on 5 cases.

Compares to baselines already stored in compare_5case_11method.json (if present)
or just reports Safe-NN standalone.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from v2_run_all import case_macro, make_case, picard_tail
from solver_safe_nn import solve_safe_nn
from solver_baseline import solve_baseline
from solver_unified_safe_nn import _residual_norm


TOL = 1e-7
BUDGET = 200000

CASES = [
    ("kolmogorov", 32),
    ("channel", 32),
    ("couette", 32),
    ("cavity_re100", 33),
    ("multi_cylinder", 32),
]


def analytic_or_picard(cid, N):
    c = make_case(cid, N)[0]
    if hasattr(c, "analytical_ux"):
        ux_an = c.analytical_ux()
        if ux_an.ndim == 1:
            ux_an = (np.tile(ux_an[:, None], (1, N))
                     if cid == "channel"
                     else np.tile(ux_an[None, :], (N, 1)))
        return c, "analytic", (ux_an, np.zeros_like(ux_an))
    f_p, _ = solve_baseline(c, max_steps=200000, tol=1e-10, check_every=500, verbose=False)
    _, ux, uy = case_macro(c, f_p)
    return c, "picard_1e-10", (ux, uy)


def score(case, f, ref):
    _, ux, uy = case_macro(case, f)
    ux_r, uy_r = ref
    mask = (case.chi > 0) if hasattr(case, "chi") else np.ones_like(ux_r, dtype=bool)
    du = ux[mask] - ux_r[mask]; dv = uy[mask] - uy_r[mask]
    den = max(float(np.sqrt(np.sum(ux_r[mask] ** 2 + uy_r[mask] ** 2))), 1e-30)
    return {
        "rel_L2": float(np.sqrt(np.sum(du * du + dv * dv)) / den),
        "Linf": float(max(np.max(np.abs(du)) if du.size else 0.0,
                          np.max(np.abs(dv)) if dv.size else 0.0)),
    }


def main():
    rows = []
    for cid, N in CASES:
        print(f"\n=== {cid} N={N} ===", flush=True)
        ref_case, ref_kind, ref_field = analytic_or_picard(cid, N)
        print(f"  reference: {ref_kind}", flush=True)
        case = make_case(cid, N)[0]
        t0 = time.perf_counter()
        f, hist = solve_safe_nn(
            case, max_outer=300, tol=TOL,
            krylov_max=10, krylov_tol=1e-3,
            kinetic_substeps=15, beta_max=0.7, eps_accept=0.10,
            line_search=False, verbose=False,
        )
        accel_lbe = int(hist[-1][2]) if hist else 0
        if np.all(np.isfinite(f)):
            f2, tail_hist, vchg = picard_tail(case, f, max_steps=BUDGET)
            tail_lbe = int(tail_hist[-1][2]) if tail_hist else 0
        else:
            f2 = f; tail_lbe = 0; vchg = float("nan")
        wall = time.perf_counter() - t0
        if not np.all(np.isfinite(f2)):
            sc = {"rel_L2": float("nan"), "Linf": float("nan")}
            nres = float("nan"); conv = False
        else:
            _, nres = _residual_norm(case, f2)
            sc = score(ref_case, f2, ref_field)
            conv = bool(np.isfinite(vchg) and vchg < 1e-6)
        row = {
            "case": cid, "N": N, "ref": ref_kind,
            "accel_lbe": accel_lbe, "tail_lbe": tail_lbe,
            "total_lbe": accel_lbe + tail_lbe,
            "wall_s": float(wall),
            "native_residual": float(nres),
            "tail_vchg": float(vchg),
            "converged": conv,
            **sc,
        }
        rows.append(row)
        print(f"  Safe-NN  total_lbe={row['total_lbe']:>7d}  wall={wall:>6.1f}s  "
              f"native_res={nres:.3e}  rel_L2={sc['rel_L2']:.3e}  conv={'Y' if conv else 'N'}",
              flush=True)
    out = Path("paper_revision_data/v2_final/safenn_only_5case.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n\n=== Safe-NN-SCMK (paper-aligned) summary ===")
    header = f"{'case':18s} {'N':>4s} {'total LBE':>10s} {'wall(s)':>8s} {'native_res':>12s} {'rel_L2':>10s} conv"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['case']:18s} {r['N']:>4d} {r['total_lbe']:>10d} "
              f"{r['wall_s']:>8.1f} {r['native_residual']:>12.3e} "
              f"{r['rel_L2']:>10.3e} {'Y' if r['converged'] else 'N'}")


if __name__ == "__main__":
    main()
