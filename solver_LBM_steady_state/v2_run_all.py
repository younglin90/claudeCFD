"""V2 unified driver: 6 methods x 9 cases x multiple grid sizes.

For every (method, case, N):
  - construct case (with PLBE for cavity_re1000)
  - run accelerator
  - apply paper-faithful Picard tail until ||du||/||u|| over 100 steps < 1e-6
  - save VTK field, history CSV, centerline CSV (where defined)
  - score against analytic/Ghia/Picard reference

Outputs land in paper_revision_data/v2_final/.
"""

from __future__ import annotations

import json
import math
import time
import traceback
from pathlib import Path

import numpy as np

from ghia_validation import get_ghia_data
from lbm_channel import ChannelCase
from lbm_couette import CouetteCase
from lbm_periodic import KolmogorovCase, build_spectral_schur
from lbm_plbe_cavity import PLBECavity
from lbm_core import LBMCavity
from lbm_voxel import VoxelCase, build_cylinder_mask
from paper_60case_benchmark import (
    backward_step_mask,
    cylinder_wake_mask,
    make_multi_cylinder_mask,
    write_history_csv,
    write_vtk,
)
from solver_baseline import solve_baseline
from solver_unified_safe_nn import _residual_norm
from solver_safe_nn import solve_safe_nn
from lbm_core import moments as _cavity_moments
import paper_60case_benchmark as _bench
import time as _time

_orig_run_method = _bench.run_method


from paper_faithful_baselines import (
    solve_dual_time_mg,
    solve_inexact_newton_ne,
    solve_preconditioned_lbm,
)


def _patched_run_method(method, case, tol, max_steps, verbose=False):
    """Re-route accelerators to paper-faithful implementations.

    proposed         -> solve_unified_safe_nn (Safe-NN++)
    preconditioned_lbm -> Guo-Zhao-Shi PLBE EDF + Picard (PRE 70.066706)
    inexact_newton_lbe -> Huang-Yang-Cai 2017 NE-preconditioned JFNK
    dual_time_mg_lbm -> Jia-Luo 2026 dual-time 2-level V-cycle MG
    others (picard_lbm, anderson_lbm) -> original dispatch
    """
    t0 = _time.perf_counter()
    if method == "proposed":
        # paper-aligned: beta init=0.1, eps_accept=0.10 (paper Eq for eps_eff),
        # K=15, krylov_max=10, krylov_tol=1e-3, restart=20, no line search.
        f, hist = solve_safe_nn(
            case, max_outer=300, tol=tol,
            krylov_max=10, krylov_tol=1e-3,
            kinetic_substeps=15, beta_max=0.7, eps_accept=0.10,
            line_search=False,
            verbose=verbose,
        )
        return f, hist, _time.perf_counter() - t0
    if method == "preconditioned_lbm":
        # gamma = 0.5 default (paper recommends gamma close to 1 for stability,
        # smaller gamma -> faster convergence; PRE 70 uses gamma in [0.1, 1]).
        f, hist = solve_preconditioned_lbm(
            case, max_steps=max_steps, tol=tol, gamma=0.5,
            check_every=200, verbose=verbose,
        )
        return f, hist, _time.perf_counter() - t0
    if method == "inexact_newton_lbe":
        f, hist = solve_inexact_newton_ne(
            case, max_outer=200, tol=tol,
            krylov_max=10, krylov_tol=1e-3,
            K_ne=20, K_smooth=10, line_search_max=4,
            reynolds_continuation=False, verbose=verbose,
        )
        return f, hist, _time.perf_counter() - t0
    if method == "dual_time_mg_lbm":
        f, hist = solve_dual_time_mg(
            case, max_outer=500, tol=tol,
            K_pre=20, K_coarse=30, K_post=20, verbose=verbose,
        )
        return f, hist, _time.perf_counter() - t0
    return _orig_run_method(method, case, tol, max_steps, verbose=verbose)


_bench.run_method = _patched_run_method
from numba_kernels import enable_numba_kernels

# Patch every Case class to use parallel njit lbe_step (fair wall-time + speed)
enable_numba_kernels(verbose=True)


def case_macro(case, f):
    if hasattr(case, "macro"):
        return case.macro(f)
    return _cavity_moments(f)


def velocity_change_norm(case, f, f_prev):
    _, ux, uy = case_macro(case, f)
    _, ux_p, uy_p = case_macro(case, f_prev)
    num = float(np.sqrt(np.sum((ux - ux_p) ** 2 + (uy - uy_p) ** 2)))
    den = max(float(np.sqrt(np.sum(ux * ux + uy * uy))), 1e-30)
    return num / den


ROOT = Path("paper_revision_data/v2_final")
ROOT.mkdir(parents=True, exist_ok=True)

METHODS = [
    "picard_lbm",
    "anderson_lbm",
    "preconditioned_lbm",
    "inexact_newton_lbe",
    "dual_time_mg_lbm",
    "proposed",
]


def make_multi_cylinder(n):
    return make_multi_cylinder_mask(n)


def make_case(case_id: str, N: int):
    if case_id == "kolmogorov":
        return KolmogorovCase(N=N, nu=0.05, F0=2e-4, kf=1), {}
    if case_id == "channel":
        return ChannelCase(N=N, nu=0.05, F0=1e-5), {}
    if case_id == "couette":
        return CouetteCase(N=N, nu=0.05, U_wall=0.05), {}
    if case_id == "cavity_re100":
        return LBMCavity(N=N, Re=100, U_wall=0.1), {"ghia_Re": 100}
    if case_id == "cavity_re400":
        return LBMCavity(N=N, Re=400, U_wall=0.1), {"ghia_Re": 400}
    if case_id == "cavity_re1000":
        # paper-faithful PLBE low-Mach setup
        mach_star = 0.058
        ratio = 0.25
        gamma = ratio * ratio
        mach = mach_star * ratio
        u_wall = mach / math.sqrt(3.0)
        return PLBECavity(N=N, Re=1000, U_wall=u_wall, gamma=gamma), {"ghia_Re": 1000, "plbe": True}
    if case_id == "multi_cylinder":
        return VoxelCase(make_multi_cylinder(N), nu=0.05, F0=2e-4, kf=1), {}
    if case_id == "backward_step":
        return VoxelCase(backward_step_mask(N), nu=0.05, F0=1.5e-5, kf=0), {}
    if case_id == "cylinder_wake":
        return VoxelCase(cylinder_wake_mask(N), nu=0.04, F0=1.0e-5, kf=0), {}
    raise ValueError(case_id)


CASE_GRID = {
    "kolmogorov":    [32, 64, 128],
    "channel":       [32, 64, 128],
    "couette":       [32, 64],
    "cavity_re100":  [33, 49, 65],
    "cavity_re400":  [49, 65, 97],
    "cavity_re1000": [65, 97, 129],
    "multi_cylinder":[32, 64],
    "backward_step": [64, 96],
    "cylinder_wake": [64, 96],
}


def picard_tail(case, f, max_steps=200000, tol=1.0e-6, check_every=100):
    f = f.copy()
    prev = f.copy()
    hist = []
    t0 = time.perf_counter()
    lbe = 0
    last_change = float("inf")
    for step in range(1, max_steps + 1):
        f = case.lbe_step(f)
        lbe += 1
        if not np.all(np.isfinite(f)):
            hist.append((step, float("nan"), lbe, time.perf_counter() - t0))
            return f, hist, float("nan")
        if step % check_every == 0:
            last_change = velocity_change_norm(case, f, prev)
            hist.append((step, last_change, lbe, time.perf_counter() - t0))
            if last_change < tol:
                break
            prev = f.copy()
    return f, hist, last_change


def analytic_reference(case_id, case):
    """Return reference velocity field (ux, uy) where analytic exists, else None."""
    if hasattr(case, "analytical_ux"):
        ux = case.analytical_ux()
        # broadcast 1D into 2D
        if ux.ndim == 1:
            ux2 = np.tile(ux[:, None], (1, case.N)) if case_id == "channel" else np.tile(ux[None, :], (case.N, 1))
            return ux2, np.zeros_like(ux2)
        return ux, np.zeros_like(ux)
    return None


def velocity_error(case, f_ref, f_test, fluid_mask=None):
    _, ux_ref, uy_ref = case_macro(case, f_ref) if not isinstance(f_ref, tuple) else (None, *f_ref)
    _, ux, uy = case_macro(case, f_test)
    if fluid_mask is None:
        fluid_mask = np.ones_like(ux_ref, dtype=bool)
    du = ux[fluid_mask] - ux_ref[fluid_mask]
    dv = uy[fluid_mask] - uy_ref[fluid_mask]
    ref = np.sqrt(ux_ref[fluid_mask] ** 2 + uy_ref[fluid_mask] ** 2)
    den = max(float(np.sqrt(np.sum(ref * ref))), 1e-30)
    return {
        "rel_l2": float(np.sqrt(np.sum(du * du + dv * dv)) / den),
        "linf": float(max(np.max(np.abs(du)), np.max(np.abs(dv)))),
        "rms": float(np.sqrt(np.mean(du * du + dv * dv))),
    }


def ghia_score(case, f, Re):
    _, ux, uy = case_macro(case, f)
    n = case.N
    y_g, u_g, x_g, v_g = get_ghia_data(Re)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    u_c = ux[:, n // 2] / case.U_wall
    v_c = uy[n // 2, :] / case.U_wall
    u_i = np.interp(y_g, y, u_c)
    v_i = np.interp(x_g, x, v_c)
    return {
        "ghia_u_rms": float(np.sqrt(np.mean((u_i - u_g) ** 2))),
        "ghia_v_rms": float(np.sqrt(np.mean((v_i - v_g) ** 2))),
        "ghia_u_linf": float(np.max(np.abs(u_i - u_g))),
        "ghia_v_linf": float(np.max(np.abs(v_i - v_g))),
    }, (u_c, v_c, y_g, u_g, x_g, v_g)


def write_centerline_csv(path, x_arr, u_arr, v_arr, label_u="u", label_v="v"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"s,{label_u},{label_v}\n")
        for i in range(len(x_arr)):
            fh.write(f"{x_arr[i]:.6f},{u_arr[i]:.6e},{v_arr[i]:.6e}\n")


def write_ghia_csv(path, u_c, v_c, y_g, u_g, x_g, v_g):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("idx,y_ghia,u_solver_centerline,u_ghia,x_ghia,v_solver_centerline,v_ghia\n")
        # solver centerline length = N, ghia length = 17. Interpolate solver to ghia points.
        n = len(u_c)
        x = np.linspace(0, 1, n)
        u_i = np.interp(y_g, x, u_c)
        v_i = np.interp(x_g, x, v_c)
        for i in range(len(y_g)):
            fh.write(f"{i},{y_g[i]:.6f},{u_i[i]:.6e},{u_g[i]:.6e},{x_g[i]:.6f},{v_i[i]:.6e},{v_g[i]:.6e}\n")


def run_one(case_id, N, method, case_meta, max_steps_budget):
    case = make_case(case_id, N)[0]
    print(f"  [{method}] {case_id} N={N}", flush=True)
    t0 = time.perf_counter()
    try:
        # tolerance for inner solver: tight enough so accelerator doesn't shortcut
        tol = 1e-8
        f, hist, accel_wall = _patched_run_method(method, case, tol, max_steps_budget, verbose=False)
        accel_lbe = int(hist[-1][2]) if hist else 0
        # paper-faithful tail
        if np.all(np.isfinite(f)):
            f, tail_hist, tail_change = picard_tail(case, f, max_steps=max_steps_budget)
            for s, ch, lb, w in tail_hist:
                hist.append((s, ch, accel_lbe + lb, w))
            tail_lbe = int(tail_hist[-1][2]) if tail_hist else 0
        else:
            tail_lbe = 0
            tail_change = float("nan")
        wall = time.perf_counter() - t0
        return f, hist, wall, accel_lbe, tail_lbe, tail_change, None
    except Exception as exc:
        traceback.print_exc()
        return None, [], time.perf_counter() - t0, 0, 0, float("nan"), f"{type(exc).__name__}: {exc}"


def budget_for(case_id, N):
    if case_id == "cavity_re1000":
        return 400000
    if case_id.startswith("cavity"):
        return 200000
    if case_id in ("multi_cylinder", "backward_step", "cylinder_wake"):
        return 200000
    return 200000


def score(case_id, case, f, meta):
    out = {}
    fluid_mask = (case.chi > 0) if hasattr(case, "chi") else None
    # analytic reference?
    ref = analytic_reference(case_id, case)
    if ref is not None:
        ux_ref, uy_ref = ref
        _, ux, uy = case_macro(case, f)
        m = fluid_mask if fluid_mask is not None else np.ones_like(ux, dtype=bool)
        du = ux[m] - ux_ref[m]; dv = uy[m] - uy_ref[m]
        denom = max(float(np.sqrt(np.sum(ux_ref[m] ** 2 + uy_ref[m] ** 2))), 1e-30)
        out["analytic_rel_l2"] = float(np.sqrt(np.sum(du * du + dv * dv)) / denom)
        out["analytic_linf"] = float(max(np.max(np.abs(du)), np.max(np.abs(dv))))
    # ghia?
    if "ghia_Re" in meta:
        gm, _ = ghia_score(case, f, meta["ghia_Re"])
        out.update(gm)
    return out


def main():
    cases = list(CASE_GRID.keys())
    big_summary = []
    sj = ROOT / "summary.json"
    done_keys = set()
    if sj.exists():
        try:
            big_summary = json.loads(sj.read_text())
            done_keys = {(r["case_id"], r["N"], r["method"]) for r in big_summary}
            print(f"[resume] loaded {len(big_summary)} prior rows", flush=True)
        except Exception:
            big_summary = []
    for case_id in cases:
        for N in CASE_GRID[case_id]:
            case_meta = make_case(case_id, N)[1]
            # build picard reference (for non-analytic cases) at this N: reuse from method run
            ref_field = None
            for method in METHODS:
                if (case_id, N, method) in done_keys:
                    print(f"  [skip] {case_id} N={N} {method}", flush=True)
                    continue
                f, hist, wall, accel_lbe, tail_lbe, tail_change, err = run_one(
                    case_id, N, method, case_meta, budget_for(case_id, N),
                )
                row = {
                    "case_id": case_id, "N": N, "method": method,
                    "wall_seconds": wall,
                    "accel_lbe_calls": accel_lbe, "tail_lbe_calls": tail_lbe,
                    "total_lbe_calls": accel_lbe + tail_lbe,
                    "tail_velocity_change": tail_change,
                    "error_string": err,
                    "converged": False,
                }
                if f is not None and np.all(np.isfinite(f)):
                    case_eval = make_case(case_id, N)[0]
                    _, native_res = _residual_norm(case_eval, f)
                    row["native_residual"] = float(native_res)
                    row["converged"] = bool(np.isfinite(tail_change) and tail_change < 1e-6)
                    row.update(score(case_id, case_eval, f, case_meta))
                    # picard self-reference (for cases without analytic + non-ghia)
                    if method == "picard_lbm":
                        ref_field = f
                    if ref_field is not None and "analytic_rel_l2" not in row and "ghia_Re" not in case_meta:
                        fluid = (case_eval.chi > 0) if hasattr(case_eval, "chi") else None
                        ve = velocity_error(case_eval, ref_field, f, fluid)
                        row.update({"vs_picard_" + k: v for k, v in ve.items()})
                    # save artifacts
                    base = ROOT / case_id / f"N{N}" / method
                    write_vtk(base / "field.vtk", case_eval, f)
                    if hist:
                        write_history_csv(base / "history.csv", hist)
                    # centerlines / profiles
                    n = case_eval.N
                    if case_id.startswith("cavity"):
                        _, ux, uy = case_macro(case_eval, f)
                        x_arr = np.linspace(0, 1, n)
                        write_centerline_csv(
                            base / "centerline.csv", x_arr,
                            ux[:, n // 2] / case_eval.U_wall,
                            uy[n // 2, :] / case_eval.U_wall,
                            "u_vertical_over_ulid", "v_horizontal_over_ulid",
                        )
                        if "ghia_Re" in case_meta:
                            _, gh_tup = ghia_score(case_eval, f, case_meta["ghia_Re"])
                            write_ghia_csv(base / "ghia_compare.csv", *gh_tup)
                    elif case_id in ("kolmogorov", "channel", "couette"):
                        _, ux, uy = case_macro(case_eval, f)
                        x_arr = np.linspace(0, 1, n)
                        u_prof = ux.mean(axis=1) if case_id in ("channel", "couette") else ux[:, n // 2]
                        write_centerline_csv(
                            base / "profile.csv", x_arr, u_prof, np.zeros_like(u_prof),
                            "u_profile", "v_profile",
                        )
                print(f"    -> conv={row['converged']} lbe={row['total_lbe_calls']} "
                      f"wall={wall:.2f}s err={err}", flush=True)
                big_summary.append(row)
                (ROOT / "summary.json").write_text(json.dumps(big_summary, indent=2), encoding="utf-8")
    # final csv
    import csv as _csv
    keys = sorted({k for r in big_summary for k in r.keys()})
    with (ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = _csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in big_summary:
            wr.writerow(r)
    print(f"\n[done] rows={len(big_summary)} -> {ROOT/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
