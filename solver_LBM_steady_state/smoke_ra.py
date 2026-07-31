"""Iter 9: Robust Anderson smoke. NO external tail picard."""
import time
import numpy as np
from v2_run_all import case_macro, make_case
from solver_baseline import solve_baseline
from solver_unified_safe_nn import _residual_norm
from solver_novel5 import solve_robust_anderson

CASES = [("kolmogorov",32,"analytic"),("channel",32,"analytic"),("couette",32,"analytic"),
         ("cavity_re100",33,"picard"),("multi_cylinder",32,"picard")]
BASELINE_BEST = {"kolmogorov":128,"channel":4720,"couette":17118,
                 "cavity_re100":11799,"multi_cylinder":7381}


def get_ref(cid, N, kind):
    c = make_case(cid, N)[0]
    if kind=="analytic" and hasattr(c,"analytical_ux"):
        ux = c.analytical_ux()
        if ux.ndim==1:
            ux = np.tile(ux[:,None],(1,N)) if cid=="channel" else np.tile(ux[None,:],(N,1))
        return c, (ux, np.zeros_like(ux))
    f_p, _ = solve_baseline(c, max_steps=200000, tol=1e-10, check_every=500, verbose=False)
    _, ux, uy = case_macro(c, f_p)
    return c, (ux, uy)


def score(case, f, ref):
    _, ux, uy = case_macro(case, f)
    ux_r, uy_r = ref
    mask = (case.chi > 0) if hasattr(case,"chi") else np.ones_like(ux_r, dtype=bool)
    du = ux[mask] - ux_r[mask]; dv = uy[mask] - uy_r[mask]
    den = max(float(np.sqrt(np.sum(ux_r[mask]**2 + uy_r[mask]**2))), 1e-30)
    return float(np.sqrt(np.sum(du*du + dv*dv)) / den)


def main():
    for cid, N, kind in CASES:
        ref_case, ref_field = get_ref(cid, N, kind)
        case = make_case(cid, N)[0]
        t0 = time.perf_counter()
        f, hist = solve_robust_anderson(case, max_outer=500, vchg_tol=1e-6,
                                          m=5, beta=1.0, safeguard_ratio=1.0,
                                          vchg_check_outer=50, verbose=False)
        lbe = int(hist[-1][2]) if hist else 0
        wall = time.perf_counter() - t0
        if np.all(np.isfinite(f)):
            _, nres = _residual_norm(case, f)
            rL2 = score(ref_case, f, ref_field)
        else:
            nres=float('nan'); rL2=float('nan')
        best = BASELINE_BEST[cid]
        v = "WIN" if lbe < best and np.isfinite(rL2) and rL2 < 1e-2 else "LOSE"
        print(f"  RA {cid:18s} lbe={lbe:>6d} wall={wall:>5.1f}s nres={nres:.2e} "
              f"rel_L2={rL2:.2e}  best={best:>6d}  {v}")


if __name__ == "__main__":
    main()
