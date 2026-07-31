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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lbm_optimized_2d import OptimizedCavityCase, solve_baseline_fast
from paper_remaining_calculations import ghia_metrics, solve_safe_nn_stats, velocity_rel_l2


OUT = Path("paper_revision_data")
FIG_DIR = OUT / "figures"
JSON_OUT = OUT / "cavity_polished_results.json"


def res_norm(case, f):
    return case._fast_norm(case.residual(f)) / math.sqrt(case.dof)


def speed(case, f):
    _, ux, uy = case.macro(f)
    return np.sqrt(ux * ux + uy * uy)


def run_polished(re=400, n=65, tol=5e-7, polish_tol=5e-8):
    case_b = OptimizedCavityCase(N=n, Re=re, U_wall=0.1)
    case_s = OptimizedCavityCase(N=n, Re=re, U_wall=0.1)
    print(f"[baseline-tight] Cavity Re={re} N={n}", flush=True)
    t0 = time.perf_counter()
    f_b, h_b = solve_baseline_fast(case_b, max_steps=300000, tol=polish_tol, check_every=500, verbose=True)
    wall_b = time.perf_counter() - t0

    print(f"[safe+polish] Cavity Re={re} N={n}", flush=True)
    t0 = time.perf_counter()
    f_s, h_s, stats = solve_safe_nn_stats(
        case_s,
        max_outer=420,
        tol=tol,
        kinetic_substeps=15,
        eps_accept=0.10,
        line_search=False,
        final_polish_tol=polish_tol,
        final_polish_max_steps=60000,
        final_polish_check_every=500,
        verbose=True,
    )
    wall_s = time.perf_counter() - t0

    result = {
        "label": f"Cavity Re={re} N={n} polished",
        "Re": re,
        "N": n,
        "nu": case_s.nu,
        "omega": case_s.omega,
        "safe_stop_tol": tol,
        "polish_tol": polish_tol,
        "baseline_lbe_tight": int(h_b[-1][2]),
        "baseline_wall_tight": float(wall_b),
        "baseline_residual_tight": float(h_b[-1][1]),
        "safe_total_lbe": int(h_s[-1][2]),
        "safe_wall_total": float(wall_s),
        "safe_final_residual": float(h_s[-1][1]),
        "speedup_lbe_tight": float(h_b[-1][2] / max(h_s[-1][2], 1)),
        "speedup_wall_tight": float(wall_b / max(wall_s, 1e-12)),
        "rel_l2_vs_tight_baseline": velocity_rel_l2(case_b, f_b, f_s),
        "baseline_ghia": ghia_metrics(f_b, case_b, re),
        "safe_ghia": ghia_metrics(f_s, case_s, re),
        "safe_stats": stats,
    }
    return result, case_b, f_b, case_s, f_s


def plot_polished(case_b, f_b, case_s, f_s, result):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sb = speed(case_b, f_b)
    ss = speed(case_s, f_s)
    diff = np.abs(ss - sb)
    path = FIG_DIR / "fig_cavity_re400_n65_polished_field.png"
    vmax = max(float(sb.max()), float(ss.max()), 1e-16)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, arr, title in [
        (axes[0], sb, "Picard tight |u|"),
        (axes[1], ss, "Safe-NN + polish |u|"),
    ]:
        im = ax.imshow(arr, origin="lower", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    im = axes[2].imshow(diff, origin="lower", cmap="magma", vmin=0.0, vmax=max(float(diff.max()), 1e-16))
    axes[2].set_title("|Safe - Picard|")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03)
    fig.suptitle(
        f"{result['label']}: residual {result['safe_final_residual']:.2e}, "
        f"relL2 {result['rel_l2_vs_tight_baseline']:.2e}"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return str(path)


def main():
    OUT.mkdir(exist_ok=True)
    result, case_b, f_b, case_s, f_s = run_polished()
    result["figure"] = plot_polished(case_b, f_b, case_s, f_s, result)
    JSON_OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
