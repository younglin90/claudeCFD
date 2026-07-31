from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    import numba

    numba.set_num_threads(32)
except Exception:
    pass

from generate_complete_updated_figures import OUT, hist_to_json, plot_history_grid
from lbm_optimized_2d import OptimizedCavityCase, solve_baseline_fast
from paper_remaining_calculations import solve_safe_nn_stats


JSON_OUT = Path("paper_revision_data/cavity_re400_n49_production_history.json")
COMPLETE_JSON = Path("paper_revision_data/complete_updated_figure_runs.json")


def main() -> None:
    case_b = OptimizedCavityCase(N=49, Re=400, U_wall=0.1)
    case_s = OptimizedCavityCase(N=49, Re=400, U_wall=0.1)

    print("[production] Cavity Re=400 N=49 tol=5e-7", flush=True)
    t0 = time.perf_counter()
    f_b, h_b = solve_baseline_fast(case_b, max_steps=120000, tol=5e-7, check_every=200, verbose=True)
    wall_b = time.perf_counter() - t0

    t0 = time.perf_counter()
    f_s, h_s, stats = solve_safe_nn_stats(
        case_s,
        max_outer=220,
        tol=5e-7,
        kinetic_substeps=15,
        beta_max=0.7,
        eps_accept=0.10,
        verbose=True,
    )
    wall_s = time.perf_counter() - t0

    run = {
        "group": "core",
        "key": "cavity_Re400_N49",
        "label": "Cavity Re=400 N=49 production",
        "tol": 5e-7,
        "baseline_history": hist_to_json(h_b),
        "safe_history": hist_to_json(h_s),
        "baseline_lbe": int(h_b[-1][2]),
        "safe_lbe": int(h_s[-1][2]),
        "baseline_residual": float(h_b[-1][1]),
        "safe_residual": float(h_s[-1][1]),
        "baseline_wall": float(wall_b),
        "safe_wall": float(wall_s),
        "speedup_lbe": float(h_b[-1][2] / max(h_s[-1][2], 1)),
        "speedup_wall": float(wall_b / max(wall_s, 1e-12)),
        "safe_stats": stats,
    }
    JSON_OUT.write_text(json.dumps(run, indent=2), encoding="utf-8")

    payload = json.loads(COMPLETE_JSON.read_text(encoding="utf-8"))
    core_runs = [r for r in payload["runs"] if r["group"] == "core"]
    plot_runs = []
    for r in core_runs:
        if r["key"] == "cavity_Re400_N49":
            plot_runs.append(run)
        else:
            plot_runs.append(r)
    plot_history_grid(
        plot_runs,
        OUT / "figU1_core6_convergence_updated.png",
        "Core 6 production convergence histories",
    )
    print(JSON_OUT)
    print(OUT / "figU1_core6_convergence_updated.png")


if __name__ == "__main__":
    main()
