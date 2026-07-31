"""Run remaining 2D major-revision calculations with optimized LBM kernels.

Targets:
  - Kolmogorov/channel N=128 and N=256 scaling
  - Lid-driven cavity Re=400 and Re=1000
  - No 3D runs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from ghia_validation import extract_centerline, get_ghia_data
from lbm_optimized_2d import (
    OptimizedCavityCase,
    OptimizedChannelCase,
    OptimizedKolmogorovCase,
    solve_baseline_fast,
)
from lbm_periodic import apply_spectral_schur, build_spectral_schur


OUT = Path("paper_revision_data")
JSON_OUT = OUT / "remaining_2d_calculations.json"
MD_OUT = OUT / "remaining_2d_calculations.md"


def sci(x: float) -> str:
    return f"{x:.3e}"


def solve_safe_nn_stats(
    case,
    max_outer=260,
    tol=1e-7,
    krylov_max=10,
    krylov_tol=1e-3,
    kinetic_substeps=15,
    beta_max=0.7,
    eps_accept=0.10,
    line_search=False,
    line_search_max=4,
    final_polish_tol=None,
    final_polish_max_steps=0,
    final_polish_check_every=500,
    verbose=True,
):
    """Safe-NN with compact stats, using the optimized case API."""
    f_prev = case.initial_field()
    f = f_prev.copy()
    n_full = case.dof
    s_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    history = []
    t0 = time.perf_counter()
    lbe = 0
    beta = 0.0
    beta_cap = beta_max
    res_prev = np.inf
    streak_no_reject = 0
    stats = {
        "lookahead_evaluations": 0,
        "lookahead_rejections": 0,
        "residual_increase_restarts": 0,
        "nan_fallbacks": 0,
        "short_K_steps": 0,
        "line_search_rejections": 0,
        "final_polish_steps": 0,
        "final_polish_lbe": 0,
        "final_polish_residual": None,
        "newton_steps": 0,
        "gmres_probe_calls": 0,
        "max_beta_used": 0.0,
        "final_beta": 0.0,
    }

    for k in range(max_outer):
        r = case.residual(f)
        lbe += 1
        res = case._fast_norm(r) / math.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, res, lbe, wall))
        if verbose and (k < 5 or k % 10 == 0 or res < tol):
            print(
                f"  snn {k:4d} | res {res:.3e} | beta {beta:.3f} | "
                f"lbe {lbe:7d} | wall {wall:.1f}s",
                flush=True,
            )
        if not np.isfinite(res):
            stats["nan_fallbacks"] += 1
            break
        if res < tol:
            break

        if res > res_prev:
            beta *= 0.7
            beta_cap = beta_max
            streak_no_reject = 0
            stats["residual_increase_restarts"] += 1
        else:
            beta = min(beta_cap, beta + 0.15)
            if streak_no_reject >= 2:
                beta_cap = min(0.95, beta_max + 0.2)

        if beta > 0.3:
            y = f + beta * (f - f_prev)
            r_y = y - case.lbe_step(y)
            lbe += 1
            stats["lookahead_evaluations"] += 1
            eps_eff = eps_accept + 0.2 * beta
            if case._fast_norm(r_y) > (1.0 + eps_eff) * case._fast_norm(r) or not np.all(np.isfinite(r_y)):
                y = f
                r_y = r
                beta *= 0.7
                beta_cap = beta_max
                streak_no_reject = 0
                stats["lookahead_rejections"] += 1
            else:
                streak_no_reject += 1
        else:
            y = f
            r_y = r
            streak_no_reject += 1

        norm_y = case._fast_norm(y)
        probes = [0]

        def matvec(v_flat):
            probes[0] += 1
            return case.jvp(v_flat.reshape(case.shape), y, r_y, norm_f_cached=norm_y).ravel()

        def precond(r_flat):
            return apply_spectral_schur(case, r_flat.reshape(case.shape), s_inv).ravel()

        op = LinearOperator((n_full, n_full), matvec=matvec, dtype=np.float64)
        mop = LinearOperator((n_full, n_full), matvec=precond, dtype=np.float64)
        df, info = gmres(
            op,
            -r_y.ravel(),
            M=mop,
            rtol=krylov_tol,
            atol=krylov_tol * np.linalg.norm(r_y) * 1e-3,
            maxiter=1,
            restart=2 * krylov_max,
        )
        lbe += probes[0]
        stats["gmres_probe_calls"] += probes[0]
        stats["newton_steps"] += 1
        stats["max_beta_used"] = max(stats["max_beta_used"], float(beta))
        if info < 0 or not np.all(np.isfinite(df)):
            stats["nan_fallbacks"] += 1
            break

        k_eff = max(5, kinetic_substeps // 2) if res < 3e-5 and res < res_prev else kinetic_substeps
        if k_eff < kinetic_substeps:
            stats["short_K_steps"] += 1

        accepted = False
        alpha = 1.0
        f_new = None
        trials = line_search_max if line_search else 1
        for trial in range(trials):
            f_trial = y + alpha * df.reshape(case.shape)
            for _ in range(k_eff):
                f_trial = case.lbe_step(f_trial)
            lbe += k_eff
            if line_search:
                r_trial_arr = f_trial - case.lbe_step(f_trial)
                lbe += 1
                r_trial = case._fast_norm(r_trial_arr) / math.sqrt(n_full)
                if np.isfinite(r_trial) and r_trial < res:
                    f_new = f_trial
                    accepted = True
                    break
                stats["line_search_rejections"] += 1
                alpha *= 0.5
            else:
                f_new = f_trial
                accepted = True
                break

        if not accepted:
            f_new = f
            for _ in range(kinetic_substeps):
                f_new = case.lbe_step(f_new)
            lbe += kinetic_substeps
            beta = 0.0

        if not np.all(np.isfinite(f_new)):
            stats["nan_fallbacks"] += 1
            f_new = f
            for _ in range(kinetic_substeps):
                f_new = case.lbe_step(f_new)
            lbe += kinetic_substeps
            beta = 0.0

        f_prev = f
        f = f_new
        res_prev = res

    stats["final_beta"] = float(beta)
    if final_polish_tol is not None and final_polish_max_steps > 0:
        for step in range(1, final_polish_max_steps + 1):
            f = case.lbe_step(f)
            lbe += 1
            if step % final_polish_check_every == 0:
                r = case.residual(f)
                lbe += 1
                res = case._fast_norm(r) / math.sqrt(n_full)
                history.append((k + step / max(final_polish_max_steps, 1), res, lbe, time.perf_counter() - t0))
                stats["final_polish_steps"] = step
                stats["final_polish_lbe"] = step + step // final_polish_check_every
                stats["final_polish_residual"] = float(res)
                if verbose:
                    print(f"  polish {step:7d} | res {res:.3e} | lbe {lbe:7d}", flush=True)
                if not np.isfinite(res) or res < final_polish_tol:
                    break
    return f, history, stats


def velocity_rel_l2(case, f_ref, f_test) -> float:
    _, ux_ref, uy_ref = case.macro(f_ref)
    _, ux, uy = case.macro(f_test)
    num = np.linalg.norm(ux - ux_ref) + np.linalg.norm(uy - uy_ref)
    den = np.linalg.norm(ux_ref) + np.linalg.norm(uy_ref)
    return float(num / max(den, 1e-30))


def ghia_metrics(f, case, re):
    y_grid, u_vert, x_grid, v_horiz = extract_centerline(f, case, case.U_wall)
    y_g, u_g, x_g, v_g = get_ghia_data(re)
    u_interp = np.interp(y_g, y_grid, u_vert)
    v_interp = np.interp(x_g, x_grid, v_horiz)
    du = u_interp - u_g
    dv = v_interp - v_g
    return {
        "u_rms": float(np.sqrt(np.mean(du * du))),
        "v_rms": float(np.sqrt(np.mean(dv * dv))),
        "u_linf": float(np.max(np.abs(du))),
        "v_linf": float(np.max(np.abs(dv))),
        "centerline_max": float(max(np.max(np.abs(du)), np.max(np.abs(dv)))),
    }


def run_scaling_case(label, factory, tol=1e-7, max_steps=800000, check_every=1000):
    print(f"\n[scaling] {label}", flush=True)
    case_b = factory()
    case_s = factory()
    t0 = time.perf_counter()
    f_b, h_b = solve_baseline_fast(
        case_b, max_steps=max_steps, tol=tol, check_every=check_every, verbose=True
    )
    wall_b = time.perf_counter() - t0
    t0 = time.perf_counter()
    f_s, h_s, stats = solve_safe_nn_stats(case_s, max_outer=300, tol=tol, verbose=True)
    wall_s = time.perf_counter() - t0
    result = {
        "label": label,
        "N": case_b.N,
        "tol": tol,
        "baseline_step": int(h_b[-1][0]),
        "baseline_lbe": int(h_b[-1][2]),
        "baseline_wall": float(wall_b),
        "baseline_residual": float(h_b[-1][1]),
        "baseline_converged": bool(h_b[-1][1] < tol),
        "safe_outer": int(h_s[-1][0]),
        "safe_lbe": int(h_s[-1][2]),
        "safe_wall": float(wall_s),
        "safe_residual": float(h_s[-1][1]),
        "safe_converged": bool(h_s[-1][1] < tol),
        "safe_speedup_lbe": float(h_b[-1][2] / max(h_s[-1][2], 1)),
        "safe_speedup_wall": float(wall_b / max(wall_s, 1e-12)),
        "rel_l2_velocity_vs_baseline": velocity_rel_l2(case_b, f_b, f_s),
        "safe_stats": stats,
    }
    print(
        f"[done] {label}: LBE x{result['safe_speedup_lbe']:.2f}, "
        f"wall x{result['safe_speedup_wall']:.2f}, res {result['safe_residual']:.3e}",
        flush=True,
    )
    return result


def run_cavity_case(re, n, tol=5e-7, max_steps=500000, check_every=500):
    label = f"Cavity Re={re} N={n}"
    print(f"\n[cavity] {label}", flush=True)
    case_b = OptimizedCavityCase(N=n, Re=re, U_wall=0.1)
    case_s = OptimizedCavityCase(N=n, Re=re, U_wall=0.1)
    t0 = time.perf_counter()
    f_b, h_b = solve_baseline_fast(
        case_b, max_steps=max_steps, tol=tol, check_every=check_every, verbose=True
    )
    wall_b = time.perf_counter() - t0
    t0 = time.perf_counter()
    f_s, h_s, stats = solve_safe_nn_stats(
        case_s,
        max_outer=420,
        tol=tol,
        kinetic_substeps=20 if re >= 1000 else 15,
        eps_accept=0.15 if re >= 1000 else 0.10,
        line_search=re >= 1000,
        line_search_max=4,
        final_polish_tol=5e-8 if re == 400 else None,
        final_polish_max_steps=60000 if re == 400 else 0,
        final_polish_check_every=500,
        verbose=True,
    )
    wall_s = time.perf_counter() - t0
    result = {
        "label": label,
        "Re": re,
        "N": n,
        "nu": case_b.nu,
        "omega": case_b.omega,
        "tol": tol,
        "baseline_step": int(h_b[-1][0]),
        "baseline_lbe": int(h_b[-1][2]),
        "baseline_wall": float(wall_b),
        "baseline_residual": float(h_b[-1][1]),
        "baseline_converged": bool(h_b[-1][1] < tol),
        "safe_outer": int(h_s[-1][0]),
        "safe_lbe": int(h_s[-1][2]),
        "safe_wall": float(wall_s),
        "safe_residual": float(h_s[-1][1]),
        "safe_converged": bool(h_s[-1][1] < tol),
        "safe_speedup_lbe": float(h_b[-1][2] / max(h_s[-1][2], 1)),
        "safe_speedup_wall": float(wall_b / max(wall_s, 1e-12)),
        "rel_l2_velocity_vs_baseline": velocity_rel_l2(case_b, f_b, f_s),
        "baseline_ghia": ghia_metrics(f_b, case_b, re),
        "safe_ghia": ghia_metrics(f_s, case_s, re),
        "safe_stats": stats,
    }
    print(
        f"[done] {label}: LBE x{result['safe_speedup_lbe']:.2f}, "
        f"wall x{result['safe_speedup_wall']:.2f}, res {result['safe_residual']:.3e}, "
        f"Ghia max {result['safe_ghia']['centerline_max']:.3e}",
        flush=True,
    )
    return result


def run_smoke():
    return {
        "scaling": [
            run_scaling_case(
                "Kolmogorov N=32 smoke",
                lambda: OptimizedKolmogorovCase(N=32, nu=0.05, F0=2e-4, kf=1),
                max_steps=50000,
                check_every=200,
            )
        ],
        "cavity": [
            run_cavity_case(400, 49, max_steps=120000, check_every=500),
        ],
    }


def run_full():
    target_u = 0.05
    nu = 0.05

    def kol_factory(n):
        k_lat = 2.0 * math.pi / n
        f0 = target_u * nu * k_lat * k_lat
        return lambda: OptimizedKolmogorovCase(N=n, nu=nu, F0=f0, kf=1)

    def ch_factory(n):
        f0 = 8.0 * nu * target_u / ((n - 1.0) ** 2)
        return lambda: OptimizedChannelCase(N=n, nu=nu, F0=f0)

    scaling_tol = 1e-9
    scaling = []
    for n in [128, 256]:
        scaling.append(
            run_scaling_case(
                f"Kolmogorov N={n} Uamp=0.05",
                kol_factory(n),
                tol=scaling_tol,
                max_steps=900000,
                check_every=1000 if n == 128 else 2000,
            )
        )
        scaling.append(
            run_scaling_case(
                f"Channel N={n} Umax=0.05",
                ch_factory(n),
                tol=scaling_tol,
                max_steps=900000,
                check_every=1000 if n == 128 else 2000,
            )
        )

    cavity = [
        run_cavity_case(400, 65, max_steps=250000, check_every=500),
        run_cavity_case(1000, 129, max_steps=700000, check_every=1000),
    ]
    return {
        "settings": {
            "scaling_target_lattice_velocity": target_u,
            "scaling_nu": nu,
            "note": "Forcing is scaled with N to keep lattice velocity stable; 3D excluded by request.",
        },
        "scaling": scaling,
        "cavity": cavity,
    }


def write_md(payload):
    lines = [
        "# Remaining 2D Major-Revision Calculations",
        "",
        "3D runs are intentionally excluded. Optimized D2Q9 kernels stream into preallocated buffers for baseline runs and avoid np.roll/equilibrium/source temporary arrays in the LBE step.",
        "",
        "## N=128/256 scaling",
        "",
        "| Case | Picard LBE | Safe LBE | LBE x | wall x | Safe residual | rel L2 vs baseline | converged |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in payload["scaling"]:
        lines.append(
            f"| {r['label']} | {r['baseline_lbe']} | {r['safe_lbe']} | "
            f"{r['safe_speedup_lbe']:.2f} | {r['safe_speedup_wall']:.2f} | "
            f"{sci(r['safe_residual'])} | {sci(r['rel_l2_velocity_vs_baseline'])} | "
            f"{r['safe_converged']} |"
        )
    lines += [
        "",
        "## Cavity Re=400/1000",
        "",
        "| Case | nu | omega | Picard LBE | Safe LBE | LBE x | wall x | Safe residual | rel L2 vs baseline | Safe Ghia max | converged |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in payload["cavity"]:
        lines.append(
            f"| {r['label']} | {r['nu']:.5f} | {r['omega']:.5f} | {r['baseline_lbe']} | "
            f"{r['safe_lbe']} | {r['safe_speedup_lbe']:.2f} | {r['safe_speedup_wall']:.2f} | "
            f"{sci(r['safe_residual'])} | {sci(r['rel_l2_velocity_vs_baseline'])} | "
            f"{sci(r['safe_ghia']['centerline_max'])} | {r['safe_converged']} |"
        )
    lines += [
        "",
        "## Safeguard statistics",
        "",
        "| Case | Newton steps | lookahead eval | rejected | residual restarts | NaN fallback | short-K | max beta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ["scaling", "cavity"]:
        for r in payload[group]:
            s = r["safe_stats"]
            lines.append(
                f"| {r['label']} | {s['newton_steps']} | {s['lookahead_evaluations']} | "
                f"{s['lookahead_rejections']} | {s['residual_increase_restarts']} | "
                f"{s['nan_fallbacks']} | {s['short_K_steps']} | {s['max_beta_used']:.2f} |"
            )
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run short validation cases")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    payload = run_smoke() if args.smoke else run_full()
    payload["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_md(payload)
    print(f"[saved] {JSON_OUT}", flush=True)
    print(f"[saved] {MD_OUT}", flush=True)


if __name__ == "__main__":
    main()
