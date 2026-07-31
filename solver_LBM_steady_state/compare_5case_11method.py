"""5 case × 11 method comparison (5 baselines + Safe-NN + 5 novel).

Cases:
  kolmogorov N=32, channel N=32, couette N=32, cavity_re100 N=33, multi_cylinder N=32

Methods:
  picard_lbm, anderson_lbm, preconditioned_lbm, inexact_newton_lbe, dual_time_mg_lbm,
  safe_nn, ms_nk_ehi, lgf_lbm, dhh_lbm, apix_lbm, elgf_lbm

For each (case, method): run accelerator + paper-faithful Picard tail,
report total LBE-call, wall-time, accuracy vs analytic/Ghia/picard reference,
convergence.
"""

from __future__ import annotations

import json
import math
import time
import traceback
from pathlib import Path

import numpy as np

from v2_run_all import (
    _patched_run_method, case_macro, make_case, picard_tail,
)
from solver_unified_safe_nn import _residual_norm
from solver_safe_nn import solve_safe_nn
from solver_novel5 import (
    solve_apix_lbm, solve_dhh_lbm, solve_elgf_lbm,
    solve_lgf_lbm, solve_ms_nk_ehi,
)

OUT = Path("paper_revision_data/v2_final/compare_5case_11method.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

TOL = 1e-7
BUDGET = 200000

CASES = [
    ("kolmogorov", 32),
    ("channel", 32),
    ("couette", 32),
    ("cavity_re100", 33),
    ("multi_cylinder", 32),
]

METHODS = [
    "picard_lbm", "anderson_lbm", "preconditioned_lbm",
    "inexact_newton_lbe", "dual_time_mg_lbm",
    "safe_nn",
    "ms_nk_ehi", "lgf_lbm", "dhh_lbm", "apix_lbm", "elgf_lbm",
]

LABELS = {
    "picard_lbm": "Picard",
    "anderson_lbm": "Anderson",
    "preconditioned_lbm": "PLBE [PRE70]",
    "inexact_newton_lbe": "InexNewton",
    "dual_time_mg_lbm": "DT-MG",
    "safe_nn": "Safe-NN",
    "ms_nk_ehi": "MS-NK-EHI (NEW)",
    "lgf_lbm": "LGF-LBM (NEW)",
    "dhh_lbm": "DHH-LBM (NEW)",
    "apix_lbm": "APIX-LBM (NEW)",
    "elgf_lbm": "ELGF-LBM (NEW)",
}


def dispatch(method, case):
    """Returns (f, hist) for the chosen method."""
    if method == "safe_nn":
        return solve_safe_nn(case, max_outer=300, tol=TOL,
                              krylov_max=10, krylov_tol=1e-3,
                              kinetic_substeps=15, beta_max=0.7, eps_accept=0.05,
                              line_search=True, line_search_max=5, verbose=False)
    if method == "ms_nk_ehi":
        return solve_ms_nk_ehi(case, max_outer=80, tol=TOL,
                                krylov_max=10, line_search_max=5, verbose=False)
    if method == "lgf_lbm":
        return solve_lgf_lbm(case, max_outer=200, tol=TOL,
                              line_search_max=5, beta_max=0.7, verbose=False)
    if method == "dhh_lbm":
        return solve_dhh_lbm(case, max_outer=BUDGET, tol=TOL,
                              N_hh=50, check_every=200, verbose=False)
    if method == "apix_lbm":
        return solve_apix_lbm(case, max_outer=BUDGET, tol=TOL,
                               dt0=1.0, dt_max=1e3, check_every=20, verbose=False)
    if method == "elgf_lbm":
        return solve_elgf_lbm(case, max_outer=400, tol=TOL,
                               line_search_max=8, beta_max=0.85, verbose=False)
    # baseline methods through patched dispatcher
    f, h, _ = _patched_run_method(method, case, TOL, BUDGET, verbose=False)
    return f, h


def analytic_or_picard_ref(cid, N):
    """Build a reference field for accuracy scoring."""
    c = make_case(cid, N)[0]
    if hasattr(c, "analytical_ux"):
        ux_an = c.analytical_ux()
        if ux_an.ndim == 1:
            if cid == "channel":
                ux_an = np.tile(ux_an[:, None], (1, N))
            else:
                ux_an = np.tile(ux_an[None, :], (N, 1))
        return c, "analytic", (ux_an, np.zeros_like(ux_an))
    if cid == "cavity_re100":
        # use picard ground truth (run picard to tight tol)
        from solver_baseline import solve_baseline
        f_p, _ = solve_baseline(c, max_steps=200000, tol=1e-10, check_every=500, verbose=False)
        _, ux, uy = case_macro(c, f_p)
        return c, "picard", (ux, uy)
    if cid == "multi_cylinder":
        from solver_baseline import solve_baseline
        f_p, _ = solve_baseline(c, max_steps=200000, tol=1e-10, check_every=500, verbose=False)
        _, ux, uy = case_macro(c, f_p)
        return c, "picard", (ux, uy)
    raise ValueError(cid)


def score(cid, case, f, ref):
    _, ux, uy = case_macro(case, f)
    ux_ref, uy_ref = ref
    fluid = (case.chi > 0) if hasattr(case, "chi") else np.ones_like(ux_ref, dtype=bool)
    du = ux[fluid] - ux_ref[fluid]; dv = uy[fluid] - uy_ref[fluid]
    den = max(float(np.sqrt(np.sum(ux_ref[fluid] ** 2 + uy_ref[fluid] ** 2))), 1e-30)
    return {
        "rel_L2": float(np.sqrt(np.sum(du * du + dv * dv)) / den),
        "Linf": float(max(np.max(np.abs(du)) if du.size else 0.0,
                         np.max(np.abs(dv)) if dv.size else 0.0)),
    }


def run_method(cid, N, method, ref_case, ref_field):
    case = make_case(cid, N)[0]
    t0 = time.perf_counter()
    try:
        f, hist = dispatch(method, case)
        accel_lbe = int(hist[-1][2]) if hist else 0
        accel_wall = time.perf_counter() - t0
        # paper-faithful tail
        if np.all(np.isfinite(f)):
            f2, tail_hist, vchg = picard_tail(case, f, max_steps=BUDGET)
            tail_lbe = int(tail_hist[-1][2]) if tail_hist else 0
        else:
            f2 = f; tail_lbe = 0; vchg = float("nan")
        total_wall = time.perf_counter() - t0
        if not np.all(np.isfinite(f2)):
            return {"method": method, "label": LABELS[method],
                    "accel_lbe": accel_lbe, "tail_lbe": tail_lbe,
                    "total_lbe": accel_lbe + tail_lbe,
                    "wall_s": total_wall, "converged": False,
                    "error": "non-finite", "rel_L2": float("nan"), "Linf": float("nan"),
                    "tail_vchg": float("nan")}
        _, nres = _residual_norm(case, f2)
        acc = score(cid, ref_case, f2, ref_field)
        return {"method": method, "label": LABELS[method],
                "accel_lbe": accel_lbe, "tail_lbe": tail_lbe,
                "total_lbe": accel_lbe + tail_lbe,
                "wall_s": float(total_wall),
                "native_residual": float(nres),
                "tail_vchg": float(vchg),
                "converged": bool(np.isfinite(vchg) and vchg < 1e-6),
                "error": None,
                **acc}
    except Exception as exc:
        traceback.print_exc()
        return {"method": method, "label": LABELS[method],
                "accel_lbe": 0, "tail_lbe": 0, "total_lbe": 0,
                "wall_s": float(time.perf_counter() - t0),
                "converged": False, "error": f"{type(exc).__name__}: {exc}",
                "rel_L2": float("nan"), "Linf": float("nan"), "tail_vchg": float("nan")}


def main():
    big = {}
    for cid, N in CASES:
        print(f"\n=== {cid} N={N} ===", flush=True)
        ref_case, ref_kind, ref_field = analytic_or_picard_ref(cid, N)
        print(f"  reference: {ref_kind}", flush=True)
        rows = []
        for m in METHODS:
            r = run_method(cid, N, m, ref_case, ref_field)
            rows.append(r)
            print(f"  {LABELS[m]:22s} lbe={r['total_lbe']:>7d} wall={r['wall_s']:>6.1f}s "
                  f"rel_L2={r.get('rel_L2', float('nan')):.3e} conv={'Y' if r['converged'] else 'N'} "
                  f"{('err='+str(r['error'])) if r.get('error') else ''}",
                  flush=True)
        big[cid] = {"N": N, "ref": ref_kind, "rows": rows}
        OUT.write_text(json.dumps(big, indent=2), encoding="utf-8")

    print("\n\n=== GRAND TABLE (LBE-call) ===")
    header = f"{'Method':22s} | " + " | ".join(f"{cid:>16s}" for cid, _ in CASES)
    print(header)
    print("-" * len(header))
    for m in METHODS:
        row = f"{LABELS[m]:22s} | "
        for cid, _ in CASES:
            r = next((x for x in big[cid]["rows"] if x["method"] == m), {})
            lbe = r.get("total_lbe", 0)
            mark = "" if r.get("converged") else "*"
            row += f"{lbe:>15d}{mark} | "
        print(row)
    print("\n* = did not satisfy paper-faithful velocity-change criterion")


if __name__ == "__main__":
    main()
