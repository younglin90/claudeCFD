"""Additional 2D LBM benchmarks for the Safe-NN major revision.

Runs three square masked-flow analogues:
  - backward-facing step channel
  - cylinder wake / vortex-shedding analogue
  - T-junction channel network

The cases intentionally use N=64 square masks so the existing Fourier-moment
preconditioner can be reused. They are supplementary coverage tests, not
production validation against external benchmark data.
"""

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

import numpy as np

try:
    import numba

    numba.set_num_threads(32)
except Exception:
    numba = None

from lbm_voxel import VoxelCase
from paper_remaining_calculations import solve_safe_nn_stats


OUT = Path("paper_revision_data")
JSON_OUT = OUT / "extra_2d_benchmarks.json"
MD_OUT = OUT / "extra_2d_benchmarks.md"


def finite_res_norm(case, f) -> float:
    r = case.residual(f)
    chi = getattr(case, "chi", None)
    if chi is not None:
        fluid = chi > 0.0
        val = np.sqrt(np.mean(r[:, fluid] * r[:, fluid]))
    else:
        val = case._fast_norm(r) / math.sqrt(case.dof)
    return float(val) if np.isfinite(val) else float("inf")


def solve_baseline_generic(case, max_steps=60000, tol=1e-7, check_every=500, verbose=True):
    f = case.initial_field()
    history = []
    t0 = time.perf_counter()
    lbe = 0
    for step in range(1, max_steps + 1):
        f = case.lbe_step(f)
        lbe += 1
        if step % check_every == 0:
            res = finite_res_norm(case, f)
            lbe += 1
            wall = time.perf_counter() - t0
            history.append((step, res, lbe, wall))
            if verbose and (step == check_every or step % (10 * check_every) == 0):
                print(f"  base {step:7d} | res {res:.3e} | wall {wall:.1f}s", flush=True)
            if not np.isfinite(res) or res < tol:
                break
    return f, history


def velocity_metrics(case, f_ref, f_test):
    _, ux_ref, uy_ref = case.macro(f_ref)
    _, ux, uy = case.macro(f_test)
    fluid = case.chi > 0
    du = ux[fluid] - ux_ref[fluid]
    dv = uy[fluid] - uy_ref[fluid]
    mag_ref = np.sqrt(ux_ref[fluid] ** 2 + uy_ref[fluid] ** 2)
    num = np.sqrt(np.sum(du * du + dv * dv))
    den = max(float(np.sqrt(np.sum(mag_ref * mag_ref))), 1e-30)
    return {
        "rel_l2": float(num / den),
        "linf": float(max(np.max(np.abs(du)), np.max(np.abs(dv)))),
        "mean_speed_ref": float(np.mean(mag_ref)),
    }


def _cell_centers(n):
    y = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    x = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    return np.meshgrid(y, x, indexing="ij")


def backward_step_mask(n=64):
    """Physical backward-facing step rasterized on an n x n grid.

    The N=64 reference corresponds to wall thickness 6 cells and an upstream
    blocked lower half before x=1/3, but this implementation defines those
    features in unit-square coordinates so larger grids are true refinements.
    """
    yy, xx = _cell_centers(n)
    chi = np.ones((n, n), dtype=np.float64)
    wall = 6.0 / 64.0
    chi[(yy < wall) | (yy > 1.0 - wall)] = 0.0
    chi[(xx < 1.0 / 3.0) & (yy >= wall) & (yy < 0.5)] = 0.0
    return chi


def cylinder_wake_mask(n=64):
    """Physical cylinder wake analogue rasterized on an n x n grid."""
    yy, xx = _cell_centers(n)
    chi = np.ones((n, n), dtype=np.float64)
    radius = 6.0 / 64.0
    cx = 1.0 / 3.0
    cy = 0.5
    chi[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2] = 0.0
    return chi


def t_junction_mask(n=64):
    """Physical T-junction channel rasterized on an n x n grid."""
    yy, xx = _cell_centers(n)
    chi = np.zeros((n, n), dtype=np.float64)
    half_width = 5.5 / 64.0
    inlet_margin = 4.0 / 64.0
    horizontal = (
        (np.abs(yy - 0.5) <= half_width)
        & (xx >= inlet_margin)
        & (xx <= 1.0 - inlet_margin)
    )
    vertical = (
        (np.abs(xx - 0.5) <= half_width)
        & (yy >= inlet_margin)
        & (yy <= 0.5 + half_width)
    )
    chi[horizontal | vertical] = 1.0
    return chi


def make_case(name):
    n = 64
    if name == "backward_step":
        case = VoxelCase(backward_step_mask(n), nu=0.05, F0=1.5e-5, kf=0)
        label = "Backward-facing step mask N=64"
        forcing = "Fx=1.5e-5"
    elif name == "cylinder_wake":
        case = VoxelCase(cylinder_wake_mask(n), nu=0.04, F0=1.0e-5, kf=0)
        label = "Cylinder wake analogue N=64"
        forcing = "Fx=1.0e-5, cylinder radius=6"
    elif name == "t_junction":
        case = VoxelCase(t_junction_mask(n), nu=0.05, F0=0.0, kf=0)
        # Drive both main branch and vertical branch weakly.
        case.Fx = np.zeros((n, n), dtype=np.float64)
        case.Fy = np.zeros((n, n), dtype=np.float64)
        fluid = case.chi > 0
        case.Fx[fluid] = 8.0e-6
        case.Fy[: n // 2, n // 2 - 5 : n // 2 + 6] = -8.0e-6
        case.Fx *= case.chi
        case.Fy *= case.chi
        label = "T-junction mask N=64"
        forcing = "Fx=8e-6 plus vertical branch Fy=-8e-6"
    else:
        raise ValueError(name)
    return label, case, forcing


def run_case(name):
    label, case_b, forcing = make_case(name)
    _, case_s, _ = make_case(name)
    tol = 1e-7
    print(f"\n[extra] {label} ({forcing})", flush=True)
    t0 = time.perf_counter()
    f_b, h_b = solve_baseline_generic(case_b, max_steps=70000, tol=tol, check_every=500, verbose=True)
    wall_b = time.perf_counter() - t0
    t0 = time.perf_counter()
    f_s, h_s, stats = solve_safe_nn_stats(
        case_s,
        max_outer=220,
        tol=tol,
        kinetic_substeps=15,
        eps_accept=0.12,
        line_search=True,
        line_search_max=4,
        verbose=True,
    )
    wall_s = time.perf_counter() - t0
    metrics = velocity_metrics(case_b, f_b, f_s)
    result = {
        "label": label,
        "forcing": forcing,
        "N": case_b.N,
        "fluid_fraction": case_b.fluid_fraction,
        "tol": tol,
        "baseline_lbe": int(h_b[-1][2]),
        "baseline_wall": float(wall_b),
        "baseline_residual": float(h_b[-1][1]),
        "baseline_converged": bool(h_b[-1][1] < tol),
        "safe_lbe": int(h_s[-1][2]),
        "safe_wall": float(wall_s),
        "safe_residual": float(h_s[-1][1]),
        "safe_converged": bool(h_s[-1][1] < tol),
        "safe_speedup_lbe": float(h_b[-1][2] / max(h_s[-1][2], 1)),
        "safe_speedup_wall": float(wall_b / max(wall_s, 1e-12)),
        "velocity_metrics": metrics,
        "safe_stats": stats,
    }
    print(
        f"[done] {label}: LBE x{result['safe_speedup_lbe']:.2f}, "
        f"wall x{result['safe_speedup_wall']:.2f}, res {result['safe_residual']:.3e}, "
        f"conv {result['safe_converged']}",
        flush=True,
    )
    return result


def write_md(payload):
    lines = [
        "# Extra 2D LBM Benchmarks",
        "",
        "3D is excluded. Runs used at most 32 numba/OMP threads and one BLAS thread.",
        "",
        "| Case | fluid frac. | Picard LBE | Safe LBE | LBE x | wall x | Safe residual | rel L2 vs baseline | converged |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in payload["benchmarks"]:
        lines.append(
            f"| {r['label']} | {r['fluid_fraction']:.3f} | {r['baseline_lbe']} | {r['safe_lbe']} | "
            f"{r['safe_speedup_lbe']:.2f} | {r['safe_speedup_wall']:.2f} | "
            f"{r['safe_residual']:.3e} | {r['velocity_metrics']['rel_l2']:.3e} | {r['safe_converged']} |"
        )
    lines += [
        "",
        "Note: the cylinder case is a steady periodic-mask wake analogue. A true vortex-shedding benchmark is unsteady and should not be claimed as a steady-state convergence success.",
        "",
        "## Safeguard statistics",
        "",
        "| Case | Newton steps | lookahead eval | rejected | residual restarts | NaN fallback | line-search rejected | max beta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in payload["benchmarks"]:
        s = r["safe_stats"]
        lines.append(
            f"| {r['label']} | {s['newton_steps']} | {s['lookahead_evaluations']} | "
            f"{s['lookahead_rejections']} | {s['residual_increase_restarts']} | "
            f"{s['nan_fallbacks']} | {s.get('line_search_rejections', 0)} | {s['max_beta_used']:.2f} |"
        )
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    payload = {
        "thread_limit": 32,
        "benchmarks": [run_case(name) for name in ["backward_step", "cylinder_wake", "t_junction"]],
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_md(payload)
    print(f"[saved] {JSON_OUT}")
    print(f"[saved] {MD_OUT}")


if __name__ == "__main__":
    main()
