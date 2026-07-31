from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import numba

    numba.set_num_threads(32)
except Exception:
    pass

from lbm_optimized_2d import OptimizedCavityCase, solve_baseline_fast
from paper_remaining_calculations import ghia_metrics, solve_safe_nn_stats, velocity_rel_l2


OUT = Path("paper_revision_data/cavity_oscillation_diagnostic.json")


def continue_picard(case, f, targets=(5e-7, 1e-7, 5e-8, 1e-8), max_steps=60000, check_every=500):
    targets = list(targets)
    reached = {}
    cur = f.copy()
    start = time.perf_counter()
    for step in range(1, max_steps + 1):
        cur = case.lbe_step(cur)
        if step % check_every == 0:
            r = case.res_norm(cur)
            for target in targets:
                key = f"{target:.0e}"
                if r < target and key not in reached:
                    reached[key] = {
                        "steps": step,
                        "residual": float(r),
                        "wall": float(time.perf_counter() - start),
                    }
            if len(reached) == len(targets):
                break
    final_res = case.res_norm(cur)
    return cur, reached, float(final_res)


def run_case(re=400, n=65):
    case_b = OptimizedCavityCase(N=n, Re=re, U_wall=0.1)
    case_s = OptimizedCavityCase(N=n, Re=re, U_wall=0.1)
    print(f"[baseline] cavity Re={re} N={n}", flush=True)
    f_b, h_b = solve_baseline_fast(case_b, max_steps=250000, tol=5e-8, check_every=500, verbose=True)
    print(f"[safe] cavity Re={re} N={n}", flush=True)
    f_s, h_s, stats_s = solve_safe_nn_stats(
        case_s,
        max_outer=420,
        tol=5e-7,
        kinetic_substeps=15,
        eps_accept=0.10,
        line_search=False,
        verbose=True,
    )
    before = {
        "safe_residual": float(h_s[-1][1]),
        "safe_lbe": int(h_s[-1][2]),
        "rel_l2_vs_tight_baseline": velocity_rel_l2(case_b, f_b, f_s),
        "ghia": ghia_metrics(f_s, case_s, re),
        "stats": stats_s,
    }
    print("[post-relax] continuing Safe-NN field by Picard", flush=True)
    f_pr, reached, final_res = continue_picard(case_s, f_s)
    after = {
        "post_relax_final_residual": final_res,
        "targets_reached": reached,
        "rel_l2_vs_tight_baseline": velocity_rel_l2(case_b, f_b, f_pr),
        "ghia": ghia_metrics(f_pr, case_s, re),
    }
    return {
        "case": f"Cavity Re={re} N={n}",
        "baseline_tight": {
            "tol": 5e-8,
            "lbe": int(h_b[-1][2]),
            "residual": float(h_b[-1][1]),
            "ghia": ghia_metrics(f_b, case_b, re),
        },
        "safe_before_post_relax": before,
        "safe_after_picard_post_relax": after,
    }


def main():
    OUT.parent.mkdir(exist_ok=True)
    payload = run_case()
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
