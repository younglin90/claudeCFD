"""Benchmark Safe-NN-v3 (lean Anderson) on 5 cases."""
import time
import numpy as np
from v2_run_all import case_macro, make_case
from solver_baseline import solve_baseline
from solver_unified_safe_nn import _residual_norm
from solver_safe_nn_v3 import solve_safenn_v3

CASES = [("kolmogorov",32,"analytic"),("channel",32,"analytic"),("couette",32,"analytic"),
         ("cavity_re100",33,"picard"),("multi_cylinder",32,"picard")]
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
        f, hist = solve_safenn_v3(case, max_outer=2000, tol=1e-7,
                                    anderson_m=5, safeguard_every=20,
                                    safeguard_ratio=1.3,
                                    stagnation_window=10, stagnation_ratio=0.95,
                                    krylov_max=10,
                                    final_polish_max=20000, final_polish_check=100,
                                    verbose=False)
        total_lbe = int(hist[-1][2]) if hist else 0
        # last history entry should be final polish vchg if it ran
        last = hist[-1] if hist else (0, float('nan'), 0, 0)
        wall = time.perf_counter() - t0
        if np.all(np.isfinite(f)):
            _, nres = _residual_norm(case, f)
            rL2 = score(ref_case, f, ref_field)
            # vchg should be passed via final polish termination; assume conv if polish ran
            vchg = last[1] if last[0] >= 2000 else float('nan')
            conv = bool(np.isfinite(vchg) and vchg < 1e-6) or (np.isfinite(nres) and nres < 1e-6)
        else:
            nres=float('nan'); rL2=float('nan'); conv=False
        pcname, pcbest = PCBEST[cid]
        v = "WIN" if total_lbe < pcbest and conv and np.isfinite(rL2) and rL2 < 1e-2 else "LOSE"
        if v == "WIN": wins += 1
        print(f"  v3 {cid:18s} total={total_lbe:>7d} wall={wall:>5.1f}s "
              f"rel_L2={rL2:.2e} conv={'Y' if conv else 'N'}  "
              f"vs {pcname}({pcbest:>6d})  {v}")
    print(f"\nWINS: {wins}/5")


if __name__ == "__main__":
    main()
