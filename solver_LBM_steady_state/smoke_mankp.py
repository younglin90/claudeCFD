"""Smoke test MANK-P on 5 cases, compare to Iter 0 grand table."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from v2_run_all import case_macro, make_case, picard_tail
from solver_baseline import solve_baseline
from solver_unified_safe_nn import _residual_norm
from solver_novel5 import solve_mankp

TOL = 1e-7
BUDGET = 200000

CASES = [
    ("kolmogorov", 32, "analytic"),
    ("channel", 32, "analytic"),
    ("couette", 32, "analytic"),
    ("cavity_re100", 33, "picard"),
    ("multi_cylinder", 32, "picard"),
]


def get_ref(cid, N, kind):
    c = make_case(cid, N)[0]
    if kind == "analytic" and hasattr(c, "analytical_ux"):
        ux = c.analytical_ux()
        if ux.ndim == 1:
            ux = np.tile(ux[:, None], (1, N)) if cid == "channel" else np.tile(ux[None, :], (N, 1))
        return c, (ux, np.zeros_like(ux))
    f_p, _ = solve_baseline(c, max_steps=200000, tol=1e-10, check_every=500, verbose=False)
    _, ux, uy = case_macro(c, f_p)
    return c, (ux, uy)


def score(ref_case, f, ref):
    _, ux, uy = case_macro(ref_case, f)
    ux_r, uy_r = ref
    mask = (ref_case.chi > 0) if hasattr(ref_case, "chi") else np.ones_like(ux_r, dtype=bool)
    du = ux[mask] - ux_r[mask]; dv = uy[mask] - uy_r[mask]
    den = max(float(np.sqrt(np.sum(ux_r[mask] ** 2 + uy_r[mask] ** 2))), 1e-30)
    return {
        "rel_L2": float(np.sqrt(np.sum(du * du + dv * dv)) / den),
        "Linf": float(max(np.max(np.abs(du)) if du.size else 0.0,
                          np.max(np.abs(dv)) if dv.size else 0.0)),
    }


def main():
    rows = []
    for cid, N, kind in CASES:
        print(f"\n=== {cid} N={N} ===", flush=True)
        ref_case, ref_field = get_ref(cid, N, kind)
        case = make_case(cid, N)[0]
        t0 = time.perf_counter()
        f, hist = solve_mankp(case, max_outer=200, tol=TOL,
                               anderson_m=5, beta=0.8, safeguard=True,
                               K_polish=10, warmup=5, verbose=False)
        accel_lbe = int(hist[-1][2]) if hist else 0
        f2, tail_hist, vchg = picard_tail(case, f, max_steps=BUDGET)
        tail_lbe = int(tail_hist[-1][2]) if tail_hist else 0
        wall = time.perf_counter() - t0
        _, nres = _residual_norm(case, f2)
        sc = score(ref_case, f2, ref_field)
        conv = bool(np.isfinite(vchg) and vchg < 1e-6)
        row = {"case": cid, "N": N, "accel_lbe": accel_lbe, "tail_lbe": tail_lbe,
               "total_lbe": accel_lbe + tail_lbe, "wall_s": float(wall),
               "native_residual": float(nres), "tail_vchg": float(vchg),
               "converged": conv, **sc}
        rows.append(row)
        print(f"  MANK-P  total_lbe={row['total_lbe']:>7d}  wall={wall:>6.1f}s  "
              f"native_res={nres:.3e}  rel_L2={sc['rel_L2']:.3e}  conv={'Y' if conv else 'N'}",
              flush=True)
    Path("paper_revision_data/v2_final/iter1_mankp.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    print("\n=== Iter1 MANK-P vs Iter0 best ===")
    iter0_best = {"kolmogorov": ("Anderson", 128),
                  "channel": ("Anderson", 4720),
                  "couette": ("Safe-NN", 12731),
                  "cavity_re100": ("MS-NK-EHI", 6450),
                  "multi_cylinder": ("Safe-NN", 4772)}
    for r in rows:
        bm, blbe = iter0_best[r["case"]]
        ratio = blbe / r["total_lbe"] if r["total_lbe"] else 0
        verdict = "WIN" if r["total_lbe"] < blbe and r["converged"] else "LOSE"
        print(f"  {r['case']:18s} MANK-P={r['total_lbe']:>7d}  best={bm:15s} ({blbe:>7d})  "
              f"ratio={ratio:.2f}x  {verdict}")


if __name__ == "__main__":
    main()
