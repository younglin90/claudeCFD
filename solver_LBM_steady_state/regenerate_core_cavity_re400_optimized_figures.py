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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate_complete_updated_figures import OUT, hist_to_json, plot_cavity_centerlines, plot_history_grid
from ghia_validation import extract_centerline, get_ghia_data
from lbm_optimized_2d import OptimizedCavityCase, solve_baseline_fast
from paper_remaining_calculations import solve_safe_nn_stats


JSON_OUT = Path("paper_revision_data/complete_updated_figure_runs.json")


def rerun_n49():
    case_b = OptimizedCavityCase(N=49, Re=400, U_wall=0.1)
    case_s = OptimizedCavityCase(N=49, Re=400, U_wall=0.1)
    print("[rerun] optimized Cavity Re=400 N=49 polished", flush=True)
    t0 = time.perf_counter()
    f_b, h_b = solve_baseline_fast(case_b, max_steps=300000, tol=5e-8, check_every=500, verbose=True)
    wall_b = time.perf_counter() - t0
    t0 = time.perf_counter()
    f_s, h_s, stats = solve_safe_nn_stats(
        case_s,
        max_outer=420,
        tol=5e-7,
        kinetic_substeps=15,
        eps_accept=0.10,
        final_polish_tol=5e-8,
        final_polish_max_steps=120000,
        final_polish_check_every=500,
        verbose=True,
    )
    wall_s = time.perf_counter() - t0
    run = {
        "group": "core",
        "key": "cavity_Re400_N49",
        "label": "Cavity Re=400 N=49 polished",
        "tol": 5e-8,
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
    return run, (case_b, f_b, case_s, f_s)


def plot_centerlines_with_re100(existing_fields, n49_field):
    fields = {"cavity_Re100_N33": existing_fields["cavity_Re100_N33"], "cavity_Re400_N49": n49_field}
    return plot_cavity_centerlines(fields)


def save_n49_centerline_single(n49_field):
    OUT.mkdir(parents=True, exist_ok=True)
    case_b, f_b, case_s, f_s = n49_field
    y_g, u_g, x_g, v_g = get_ghia_data(400)
    y_b, u_b, x_b, v_b = extract_centerline(f_b, case_b, case_b.U_wall)
    y_s, u_s, x_s, v_s = extract_centerline(f_s, case_s, case_s.U_wall)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].plot(u_g, y_g, "ko", ms=4, label="Ghia")
    axes[0].plot(u_b, y_b, "--", color="0.45", label="Picard tight")
    axes[0].plot(u_s, y_s, "-", color="#b84040", label="Safe-NN + polish")
    axes[0].set_title("Cavity Re=400 N=49: u vertical")
    axes[0].set_xlabel("u/U_lid")
    axes[0].set_ylabel("y")
    axes[1].plot(x_g, v_g, "ko", ms=4, label="Ghia")
    axes[1].plot(x_b, v_b, "--", color="0.45", label="Picard tight")
    axes[1].plot(x_s, v_s, "-", color="#b84040", label="Safe-NN + polish")
    axes[1].set_title("Cavity Re=400 N=49: v horizontal")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("v/U_lid")
    axes[0].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    path = OUT / "figU3b_cavity_re400_n49_polished_centerline_updated.png"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(path)


def main():
    payload = json.loads(JSON_OUT.read_text(encoding="utf-8"))
    run, n49_field = rerun_n49()
    for i, old in enumerate(payload["runs"]):
        if old["key"] == "cavity_Re400_N49" and old["group"] == "core":
            payload["runs"][i] = run
            break
    else:
        payload["runs"].append(run)
    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    core_runs = [r for r in payload["runs"] if r["group"] == "core"]
    plot_history_grid(core_runs, OUT / "figU1_core6_convergence_updated.png", "Core 6 validation convergence histories")
    save_n49_centerline_single(n49_field)


if __name__ == "__main__":
    main()
