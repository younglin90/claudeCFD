"""Grid-scaling check for the two difficult 2D cases.

The purpose is diagnostic: determine whether the weak performance of the two
focus cases is primarily caused by under-resolution. We compare the current
unified proposed solver against Picard references at two residual-check
cadences so early-stopping artifacts are visible.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from lbm_core import LBMCavity
from lbm_voxel import VoxelCase
from paper_60case_benchmark import velocity_error, write_history_csv, write_vtk
from solver_scmk import solve_scmk
from solver_baseline import solve_baseline
from paper_extra_benchmarks import solve_baseline_generic
from lbm_periodic import build_spectral_schur


OUT = Path("paper_revision_data") / "bench2_grid_scaling"
HIST = OUT / "histories"
VTK = OUT / "vtk"


def t_junction_mask_scaled(n):
    chi = np.zeros((n, n), dtype=np.float64)
    mid = n // 2
    half = max(5, n // 12)
    margin = max(4, n // 16)
    chi[mid - half : mid + half + 1, margin : n - margin] = 1.0
    chi[margin : mid + half + 1, mid - half : mid + half + 1] = 1.0
    return chi


def make_t_junction_scaled(n):
    chi = t_junction_mask_scaled(n)
    case = VoxelCase(chi, nu=0.05, F0=0.0, kf=0)
    case.Fx = np.zeros((n, n), dtype=np.float64)
    case.Fy = np.zeros((n, n), dtype=np.float64)
    fluid = case.chi > 0
    case.Fx[fluid] = 8.0e-6
    half = max(5, n // 12)
    case.Fy[: n // 2, n // 2 - half : n // 2 + half + 1] = -8.0e-6
    case.Fx *= case.chi
    case.Fy *= case.chi
    return case


def cases():
    return [
        ("cavity_re1000_n129", "Cavity Re=1000 N=129", 5.0e-7, lambda: LBMCavity(N=129, Re=1000, U_wall=0.1), 900000),
        ("cavity_re1000_n257", "Cavity Re=1000 N=257", 5.0e-7, lambda: LBMCavity(N=257, Re=1000, U_wall=0.1), 1200000),
        ("t_junction_scaled_n64", "Scaled T-junction N=64", 1.0e-7, lambda: make_t_junction_scaled(64), 100000),
        ("t_junction_scaled_n128", "Scaled T-junction N=128", 1.0e-7, lambda: make_t_junction_scaled(128), 200000),
    ]


def run_picard(case, tol, max_steps, check_every, verbose=False):
    if isinstance(case, LBMCavity):
        return solve_baseline(case, max_steps=max_steps, tol=tol, check_every=check_every, verbose=verbose)
    return solve_baseline_generic(case, max_steps=max_steps, tol=tol, check_every=check_every, verbose=verbose)


def run_proposed(case, tol, verbose=False):
    s_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    return solve_scmk(
        case,
        s_inv,
        max_outer=360,
        tol=tol,
        krylov_max=10,
        krylov_tol=1.0e-3,
        kinetic_substeps=15,
        line_search_max=5,
        verbose=verbose,
    )


def row(case_id, label, method, case_ref, f_ref, case, f, hist, wall, tol, ref_lbe, ref_wall):
    fluid = getattr(case_ref, "chi", np.ones((case_ref.N, case_ref.N), dtype=np.float64)) > 0
    err = velocity_error(case_ref, f_ref, case, f, fluid_mask=fluid)
    final_res = float(hist[-1][1])
    lbe = int(hist[-1][2])
    return {
        "case_id": case_id,
        "case_label": label,
        "method": method,
        "N": case.N,
        "tol": tol,
        "lbe_calls": lbe,
        "wall_seconds": float(wall),
        "final_residual": final_res,
        "converged": bool(np.isfinite(final_res) and final_res < 5.0 * tol),
        "lbe_speedup_vs_picard500": float(ref_lbe / max(lbe, 1)),
        "wall_speedup_vs_picard500": float(ref_wall / max(wall, 1.0e-12)),
        "rel_l2_vs_picard500": err["rel_l2"],
        "linf_vs_picard500": err["linf"],
        "rms_vs_picard500": err["rms"],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    HIST.mkdir(parents=True, exist_ok=True)
    VTK.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()
    for case_id, label, tol, factory, max_steps in cases():
        print(f"[grid] {label}", flush=True)
        cref = factory()
        t0 = time.perf_counter()
        f_ref, h_ref = run_picard(cref, tol, max_steps, check_every=500, verbose=False)
        w_ref = time.perf_counter() - t0
        ref_lbe = int(h_ref[-1][2])
        ref_wall = float(w_ref)
        rows.append(row(case_id, label, "picard_check500", cref, f_ref, cref, f_ref, h_ref, w_ref, tol, ref_lbe, ref_wall))
        write_history_csv(HIST / f"{case_id}__picard_check500.csv", h_ref)
        write_vtk(VTK / f"{case_id}__picard_check500.vtk", cref, f_ref)
        print(f"  picard_check500 lbe={ref_lbe:8d} res={h_ref[-1][1]:.3e}", flush=True)

        c50 = factory()
        t0 = time.perf_counter()
        f50, h50 = run_picard(c50, tol, max_steps, check_every=50, verbose=False)
        w50 = time.perf_counter() - t0
        rows.append(row(case_id, label, "picard_check50", cref, f_ref, c50, f50, h50, w50, tol, ref_lbe, ref_wall))
        write_history_csv(HIST / f"{case_id}__picard_check50.csv", h50)
        print(f"  picard_check50  lbe={int(h50[-1][2]):8d} res={h50[-1][1]:.3e}", flush=True)

        cp = factory()
        t0 = time.perf_counter()
        fp, hp = run_proposed(cp, tol, verbose=False)
        wp = time.perf_counter() - t0
        rows.append(row(case_id, label, "proposed", cref, f_ref, cp, fp, hp, wp, tol, ref_lbe, ref_wall))
        write_history_csv(HIST / f"{case_id}__proposed.csv", hp)
        write_vtk(VTK / f"{case_id}__proposed.vtk", cp, fp)
        print(f"  proposed        lbe={int(hp[-1][2]):8d} res={hp[-1][1]:.3e}", flush=True)

    fields = list(rows[0].keys())
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    proposed = [r for r in rows if r["method"] == "proposed"]
    metrics = {
        "case_count": len(proposed),
        "strict_pass_count": int(sum(r["converged"] and r["lbe_speedup_vs_picard500"] > 1.0 and r["rel_l2_vs_picard500"] <= 0.05 for r in proposed)),
        "proposed_converged_count": int(sum(r["converged"] for r in proposed)),
        "worst_rel_l2": float(max(r["rel_l2_vs_picard500"] for r in proposed)),
        "mean_lbe_speedup_vs_picard500": float(np.mean([r["lbe_speedup_vs_picard500"] for r in proposed])),
        "elapsed_wall_seconds": float(time.perf_counter() - started),
    }
    metrics["all_pass"] = int(metrics["strict_pass_count"] == metrics["case_count"])
    (OUT / "summary.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
