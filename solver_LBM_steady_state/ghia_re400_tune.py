"""Re=400 Ghia tuning : optimize N + U_wall to match Ghia 1982 reference.

Loop variable :  (N, U_wall, max_steps) tuple
Metric        :  max(u_err, v_err) RMS vs Ghia 1982
Target        :  < 5e-3 to match standard LBM literature
"""

import os, time, sys, json
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from lbm_core import LBMCavity
from solver_baseline import solve_baseline
from solver_hybrid import solve_hybrid
from ghia_validation import compare_with_ghia


def run_config(N, U_wall, tol, max_steps, label, do_scmk=True):
    case = LBMCavity(N=N, Re=400, U_wall=U_wall)
    Ma = U_wall * np.sqrt(3)
    print(f"  {label} : N={N} U={U_wall} omega={case.omega:.4f} Ma={Ma:.4f} max_steps={max_steps}")
    t0 = time.perf_counter()
    f_b, hist_b = solve_baseline(case, max_steps=max_steps, tol=tol,
                                   check_every=2000, verbose=False)
    wall_b = time.perf_counter() - t0
    res_b = hist_b[-1][1]
    if not np.isfinite(res_b) or res_b > tol * 10:
        print(f"    baseline FAIL res={res_b:.2e}")
        return None
    print(f"    baseline : step {hist_b[-1][0]} lbe {hist_b[-1][2]} res {res_b:.2e} wall {wall_b:.1f}s")
    eu_b, ev_b = compare_with_ghia(f_b, case, 400,
                                     f"results_ghia/tune_{label}_baseline.png")
    print(f"    baseline Ghia err : u {eu_b:.3e}, v {ev_b:.3e}")

    eu_s, ev_s, su = None, None, None
    if do_scmk:
        case_s = LBMCavity(N=N, Re=400, U_wall=U_wall)
        t0 = time.perf_counter()
        f_s, hist_s = solve_hybrid(case_s, max_outer=300, tol=tol, krylov_max=10,
                                     kinetic_substeps=15, N_check=6, min_ratio=2.0,
                                     verbose=False)
        wall_s = time.perf_counter() - t0
        res_s = hist_s[-1][1]
        if np.isfinite(res_s) and res_s < tol * 10:
            su = hist_b[-1][2] / max(hist_s[-1][2], 1)
            print(f"    SCMK     : outer {hist_s[-1][0]} lbe {hist_s[-1][2]} res {res_s:.2e} wall {wall_s:.1f}s speedup {su:.1f}x")
            eu_s, ev_s = compare_with_ghia(f_s, case_s, 400,
                                             f"results_ghia/tune_{label}_SCMK.png")
            print(f"    SCMK     Ghia err : u {eu_s:.3e}, v {ev_s:.3e}")
        else:
            print(f"    SCMK FAIL res={res_s:.2e}")

    return {
        "label": label, "N": N, "U_wall": U_wall, "Ma": Ma, "omega": case.omega,
        "baseline_lbe": int(hist_b[-1][2]), "baseline_res": float(res_b),
        "baseline_eu": float(eu_b), "baseline_ev": float(ev_b),
        "scmk_eu": float(eu_s) if eu_s else None,
        "scmk_ev": float(ev_s) if ev_s else None,
        "scmk_speedup": float(su) if su else None,
    }


def main():
    os.makedirs("results_ghia", exist_ok=True)
    # tol scaled by U_wall (residual ~ U_wall * eq distribution magnitude)
    configs = [
        # (label, N, U_wall, tol, max_steps)
        ("N65_U10",   65,  0.10, 5e-8, 200000),
        ("N97_U10",   97,  0.10, 5e-8, 400000),
        ("N129_U075", 129, 0.075, 4e-8, 600000),
        ("N129_U05",  129, 0.05,  2e-8, 800000),
        ("N193_U05",  193, 0.05,  2e-8, 1000000),
    ]
    results = []
    for label, N, U, tol, mx in configs:
        print(f"\n=== {label} ===")
        try:
            r = run_config(N, U, tol, mx, label, do_scmk=True)
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"  FAIL: {e}")

    print("\n=== SUMMARY ===")
    print(f"{'Label':<12} {'N':>5} {'U':>6} {'Ma':>6} {'omega':>7} {'baseline u-err':>16} {'SCMK u-err':>12}")
    print("-" * 78)
    for r in results:
        scmk_eu = r["scmk_eu"]
        scmk_str = f"{scmk_eu:.3e}" if scmk_eu else "      —"
        print(f"{r['label']:<12} {r['N']:>5} {r['U_wall']:>6.3f} {r['Ma']:>6.4f} "
              f"{r['omega']:>7.4f} {r['baseline_eu']:>16.3e} {scmk_str:>12}")

    with open("results_ghia/tune_re400_summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
    # winning u-err (baseline)
    if results:
        best = min(results, key=lambda r: r["baseline_eu"])
        print(f"\nBEST baseline : {best['label']} u-err = {best['baseline_eu']:.3e}")


if __name__ == "__main__":
    main()
