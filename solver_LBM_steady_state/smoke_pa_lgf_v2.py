"""Iter 4: PA-LGF v2 on 5 cases."""

from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from v2_run_all import case_macro, make_case, picard_tail
from solver_baseline import solve_baseline
from solver_unified_safe_nn import _residual_norm
from solver_novel5 import solve_pa_lgf_v2

CASES = [("kolmogorov", 32, "analytic"), ("channel", 32, "analytic"),
         ("couette", 32, "analytic"), ("cavity_re100", 33, "picard"),
         ("multi_cylinder", 32, "picard")]
BUDGET = 200000
ITER0 = {"kolmogorov": 128, "channel": 4720, "couette": 12731,
         "cavity_re100": 6450, "multi_cylinder": 4772}


def get_ref(cid, N, kind):
    c = make_case(cid, N)[0]
    if kind == "analytic" and hasattr(c, "analytical_ux"):
        ux = c.analytical_ux()
        if ux.ndim == 1:
            ux = (np.tile(ux[:, None], (1, N)) if cid == "channel"
                  else np.tile(ux[None, :], (N, 1)))
        return c, (ux, np.zeros_like(ux))
    f_p, _ = solve_baseline(c, max_steps=200000, tol=1e-10, check_every=500, verbose=False)
    _, ux, uy = case_macro(c, f_p)
    return c, (ux, uy)


def score(case, f, ref):
    _, ux, uy = case_macro(case, f)
    ux_r, uy_r = ref
    mask = (case.chi > 0) if hasattr(case, "chi") else np.ones_like(ux_r, dtype=bool)
    du = ux[mask] - ux_r[mask]; dv = uy[mask] - uy_r[mask]
    den = max(float(np.sqrt(np.sum(ux_r[mask] ** 2 + uy_r[mask] ** 2))), 1e-30)
    return {"rel_L2": float(np.sqrt(np.sum(du * du + dv * dv)) / den)}


def main():
    rows = []
    for cid, N, kind in CASES:
        ref_case, ref_field = get_ref(cid, N, kind)
        case = make_case(cid, N)[0]
        t0 = time.perf_counter()
        f, hist = solve_pa_lgf_v2(case, max_outer=2000, vchg_tol=1e-6,
                                    residual_tol=1e-7, anderson_m=10,
                                    check_every=10, fallback_picard_steps=20,
                                    verbose=False)
        accel_lbe = int(hist[-1][2]) if hist else 0
        f2, tail_hist, vchg = picard_tail(case, f, max_steps=BUDGET)
        tail_lbe = int(tail_hist[-1][2]) if tail_hist else 0
        wall = time.perf_counter() - t0
        if np.all(np.isfinite(f2)):
            _, nres = _residual_norm(case, f2)
            sc = score(ref_case, f2, ref_field)
            conv = bool(np.isfinite(vchg) and vchg < 1e-6)
        else:
            nres = float("nan"); sc = {"rel_L2": float("nan")}; conv = False
        total = accel_lbe + tail_lbe
        rows.append({"case": cid, "accel": accel_lbe, "tail": tail_lbe,
                     "total": total, "wall": wall, "nres": float(nres),
                     **sc, "conv": conv})
        best = ITER0[cid]
        v = "WIN" if total < best and conv else "LOSE"
        print(f"  PA-LGF-v2 {cid:18s} accel={accel_lbe:>6d} tail={tail_lbe:>6d} "
              f"total={total:>7d} rel_L2={sc['rel_L2']:.2e} conv={'Y' if conv else 'N'}  "
              f"best={best:>6d}  {v}")
    Path("paper_revision_data/v2_final/iter4_pa_lgf_v2.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
