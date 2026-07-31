"""Benchmark Safe-NN-v2 on 5 cases. Compare to baselines + original Safe-NN."""
import time
import numpy as np
from v2_run_all import case_macro, make_case, picard_tail
from solver_baseline import solve_baseline
from solver_unified_safe_nn import _residual_norm
from solver_safe_nn_v2 import solve_safenn_v2

CASES = [("kolmogorov",32,"analytic"),("channel",32,"analytic"),("couette",32,"analytic"),
         ("cavity_re100",33,"picard"),("multi_cylinder",32,"picard")]

# Per-case best baseline reference
PCBEST = {"kolmogorov":("Anderson",128),"channel":("Anderson",4720),
          "couette":("PLBE",17118),"cavity_re100":("InexNewton",11799),
          "multi_cylinder":("InexNewton",7381)}


def get_ref(cid, N, kind):
    c = make_case(cid, N)[0]
    if kind=="analytic" and hasattr(c,"analytical_ux"):
        ux = c.analytical_ux()
        if ux.ndim==1:
            ux = np.tile(ux[:,None],(1,N)) if cid=="channel" else np.tile(ux[None,:],(N,1))
        return c, (ux, np.zeros_like(ux))
    f_p,_ = solve_baseline(c, max_steps=200000, tol=1e-10, check_every=500, verbose=False)
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
    wins = 0
    for cid, N, kind in CASES:
        ref_case, ref_field = get_ref(cid, N, kind)
        case = make_case(cid, N)[0]
        t0 = time.perf_counter()
        f, hist = solve_safenn_v2(case, max_outer=400, tol=1e-7,
                                    anderson_m=5, anderson_beta=1.0,
                                    safeguard_ratio=2.0,
                                    kinetic_substeps=15, kinetic_substeps_min=8,
                                    krylov_max=10,
                                    vchg_check_outer=50,
                                    internal_polish_max=20000,
                                    internal_polish_check=100,
                                    verbose=False)
        total_lbe = int(hist[-1][2]) if hist else 0
        accel = total_lbe; tail = 0; f2 = f
        # quick post-check: vchg under 1e-6 already?
        f_check = case.lbe_step(f); f_check_after = f_check.copy()
        for _ in range(99):
            f_check_after = case.lbe_step(f_check_after)
        if hasattr(case, "macro"):
            _, ux, uy = case.macro(f_check_after); _, uxp, uyp = case.macro(f_check)
        else:
            from lbm_core import moments
            _, ux, uy = moments(f_check_after); _, uxp, uyp = moments(f_check)
        num = float(np.sqrt(np.sum((ux - uxp) ** 2 + (uy - uyp) ** 2)))
        den = max(float(np.sqrt(np.sum(ux * ux + uy * uy))), 1e-30)
        vchg = num / den
        # add the 100 verification LBE
        accel += 100
        total_lbe = accel
        wall = time.perf_counter() - t0
        if np.all(np.isfinite(f)):
            _, nres = _residual_norm(case, f)
            rL2 = score(ref_case, f, ref_field)
            conv = bool(np.isfinite(vchg) and vchg < 1e-6)
        else:
            nres=float('nan'); rL2=float('nan'); conv=False
        total = total_lbe
        pcname, pcbest = PCBEST[cid]
        v = "WIN" if total < pcbest and conv else "LOSE"
        if v == "WIN": wins += 1
        print(f"  v2 {cid:18s} accel={accel:>6d} total={total:>7d} "
              f"wall={wall:>5.1f}s rel_L2={rL2:.2e} vchg={vchg:.2e} conv={'Y' if conv else 'N'}  "
              f"vs {pcname}({pcbest:>6d})  {v}")
    print(f"\nWINS: {wins}/5")


if __name__ == "__main__":
    main()
