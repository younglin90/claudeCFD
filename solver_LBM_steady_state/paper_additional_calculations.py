"""Additional calculations requested for the Safe-NN-SCMK major revision.

Outputs:
  paper_revision_data/additional_calculations.json
  paper_revision_data/additional_calculations.md
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from ghia_validation import extract_centerline, get_ghia_data
from lbm_3d import apply_spectral_schur_3d, build_spectral_schur_3d, Kolmogorov3DCase
from lbm_channel import ChannelCase
from lbm_channel_3d import Channel3DCase
from lbm_core import LBMCavity, moments as cavity_moments
from lbm_periodic import KolmogorovCase
from solver_baseline import solve_baseline
from solver_safe_nn import solve_safe_nn
from solver_scmk import solve_baseline_periodic
from solver_scmk_3d import solve_baseline_3d


OUT = Path("paper_revision_data")


def safe_float(x):
    return float(x) if np.isfinite(x) else None


def ghia_metrics(f, case, Re):
    y_grid, u_vert, x_grid, v_horiz = extract_centerline(f, case, case.U_wall)
    y_g, u_g, x_g, v_g = get_ghia_data(Re)
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


def run_cavity_ghia(Re, N, max_steps):
    tol = 5e-7
    print(f"[ghia] Re={Re} N={N}", flush=True)
    case_b = LBMCavity(N=N, Re=Re, U_wall=0.1)
    case_s = LBMCavity(N=N, Re=Re, U_wall=0.1)
    t0 = time.perf_counter()
    f_b, h_b = solve_baseline(case_b, max_steps=max_steps, tol=tol, check_every=200, verbose=False)
    wall_b = time.perf_counter() - t0
    t0 = time.perf_counter()
    f_s, h_s = solve_safe_nn(
        case_s,
        max_outer=220,
        tol=tol,
        krylov_max=10,
        krylov_tol=1e-3,
        kinetic_substeps=15,
        beta_max=0.7,
        eps_accept=0.10,
        verbose=False,
    )
    wall_s = time.perf_counter() - t0
    return {
        "Re": Re,
        "N": N,
        "baseline_lbe": int(h_b[-1][2]),
        "baseline_wall": wall_b,
        "baseline_residual": float(h_b[-1][1]),
        "safe_lbe": int(h_s[-1][2]),
        "safe_wall": wall_s,
        "safe_residual": float(h_s[-1][1]),
        "safe_speedup_lbe": float(h_b[-1][2] / max(h_s[-1][2], 1)),
        "safe_speedup_wall": float(wall_b / max(wall_s, 1e-12)),
        "baseline_ghia": ghia_metrics(f_b, case_b, Re),
        "safe_ghia": ghia_metrics(f_s, case_s, Re),
    }


def solve_safe_nn_3d(
    case,
    max_outer=120,
    tol=1e-7,
    krylov_max=10,
    krylov_tol=1e-3,
    kinetic_substeps=15,
    beta_max=0.7,
    eps_accept=0.10,
):
    f_prev = case.initial_field()
    f = f_prev.copy()
    n_full = case.dof
    s_inv = build_spectral_schur_3d(case.N, omega=case.omega)
    history = []
    t0 = time.perf_counter()
    lbe = 0
    beta = 0.0
    res_prev = np.inf
    stats = {"lookahead_rejections": 0, "nan_fallbacks": 0, "residual_restarts": 0}
    for k in range(max_outer):
        r = case.residual(f)
        lbe += 1
        res = case._fast_norm(r) / math.sqrt(n_full)
        history.append((k, res, lbe, time.perf_counter() - t0))
        if res < tol:
            break
        if res > res_prev:
            beta *= 0.7
            stats["residual_restarts"] += 1
        else:
            beta = min(beta_max, beta + 0.15)
        if beta > 0.3:
            y = f + beta * (f - f_prev)
            r_y = y - case.lbe_step(y)
            lbe += 1
            if (
                case._fast_norm(r_y) > (1.0 + eps_accept + 0.2 * beta) * case._fast_norm(r)
                or not np.all(np.isfinite(r_y))
            ):
                y = f
                r_y = r
                beta *= 0.7
                stats["lookahead_rejections"] += 1
        else:
            y = f
            r_y = r
        norm_y = case._fast_norm(y)
        probes = [0]

        def matvec(v_flat):
            w = v_flat.reshape(case.shape)
            probes[0] += 1
            return case.jvp(w, y, r_y, norm_f_cached=norm_y).ravel()

        def precond(r_flat):
            return apply_spectral_schur_3d(case, r_flat.reshape(case.shape), s_inv).ravel()

        op = LinearOperator((n_full, n_full), matvec=matvec, dtype=np.float64)
        mop = LinearOperator((n_full, n_full), matvec=precond, dtype=np.float64)
        df, _ = gmres(
            op,
            -r_y.ravel(),
            M=mop,
            rtol=krylov_tol,
            atol=krylov_tol * np.linalg.norm(r_y) * 1e-3,
            maxiter=1,
            restart=2 * krylov_max,
        )
        lbe += probes[0]
        if not np.all(np.isfinite(df)):
            stats["nan_fallbacks"] += 1
            break
        f_new = y + df.reshape(case.shape)
        k_eff = max(5, kinetic_substeps // 2) if res < 3e-5 and res < res_prev else kinetic_substeps
        for _ in range(k_eff):
            f_new = case.lbe_step(f_new)
        lbe += k_eff
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
    return f, history, stats


def macro_of(case, f):
    m = case.macro(f)
    return m


def rel_ux_error(case, f_ref, f_test):
    ux_ref = macro_of(case, f_ref)[1]
    ux = macro_of(case, f_test)[1]
    return float(np.linalg.norm(ux - ux_ref) / max(np.linalg.norm(ux_ref), 1e-30))


def run_2d_scaling():
    out = []
    for label, factory, tol, max_b in [
        ("Kolmogorov N=64", lambda: KolmogorovCase(N=64, nu=0.05, F0=2e-4, kf=1), 1e-7, 100000),
        ("Channel N=64", lambda: ChannelCase(N=64, nu=0.05, F0=1e-5), 1e-7, 100000),
    ]:
        print(f"[scaling] {label}", flush=True)
        c_b = factory()
        c_s = factory()
        t0 = time.perf_counter()
        f_b, h_b = solve_baseline_periodic(c_b, max_steps=max_b, tol=tol, check_every=400, verbose=False)
        wall_b = time.perf_counter() - t0
        t0 = time.perf_counter()
        f_s, h_s = solve_safe_nn(
            c_s,
            max_outer=220,
            tol=tol,
            krylov_max=10,
            krylov_tol=1e-3,
            kinetic_substeps=15,
            beta_max=0.7,
            eps_accept=0.10,
            verbose=False,
        )
        wall_s = time.perf_counter() - t0
        out.append(
            {
                "label": label,
                "baseline_lbe": int(h_b[-1][2]),
                "baseline_wall": wall_b,
                "baseline_residual": float(h_b[-1][1]),
                "safe_lbe": int(h_s[-1][2]),
                "safe_wall": wall_s,
                "safe_residual": float(h_s[-1][1]),
                "safe_speedup_lbe": float(h_b[-1][2] / max(h_s[-1][2], 1)),
                "safe_speedup_wall": float(wall_b / max(wall_s, 1e-12)),
                "rel_ux_error": rel_ux_error(c_b, f_b, f_s),
            }
        )
    return out


def run_3d():
    out = []
    for label, factory, max_b in [
        ("3D Kolmogorov N=16", lambda: Kolmogorov3DCase(N=16, nu=0.05, F0=2e-4, kf=1), 20000),
        ("3D Channel N=16", lambda: Channel3DCase(N=16, nu=0.05, F0=1e-4), 20000),
    ]:
        print(f"[3d] {label}", flush=True)
        tol = 1e-7
        c_b = factory()
        c_s = factory()
        t0 = time.perf_counter()
        f_b, h_b = solve_baseline_3d(c_b, max_steps=max_b, tol=tol, check_every=200, verbose=False)
        wall_b = time.perf_counter() - t0
        t0 = time.perf_counter()
        f_s, h_s, stats = solve_safe_nn_3d(c_s, max_outer=160, tol=tol)
        wall_s = time.perf_counter() - t0
        out.append(
            {
                "label": label,
                "baseline_lbe": int(h_b[-1][2]),
                "baseline_wall": wall_b,
                "baseline_residual": float(h_b[-1][1]),
                "safe_lbe": int(h_s[-1][2]),
                "safe_wall": wall_s,
                "safe_residual": float(h_s[-1][1]),
                "safe_converged": bool(h_s[-1][1] < tol * 5),
                "safe_speedup_lbe": float(h_b[-1][2] / max(h_s[-1][2], 1)),
                "safe_speedup_wall": float(wall_b / max(wall_s, 1e-12)),
                "rel_ux_error": rel_ux_error(c_b, f_b, f_s),
                "stats": stats,
            }
        )
    return out


def write_md(payload, path):
    lines = [
        "# 추가 계산 결과: Ghia max deviation, N=64 scaling, 3D Safe-NN smoke",
        "",
        "## Ghia centerline metrics",
        "",
        "| Case | N | Safe LBE x | Safe wall x | Safe u RMS | Safe v RMS | Safe u L_inf | Safe v L_inf | Safe max | Baseline max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in payload["ghia"]:
        lines.append(
            f"| Re={r['Re']} | {r['N']} | {r['safe_speedup_lbe']:.2f} | {r['safe_speedup_wall']:.2f} | "
            f"{r['safe_ghia']['u_rms']:.3e} | {r['safe_ghia']['v_rms']:.3e} | "
            f"{r['safe_ghia']['u_linf']:.3e} | {r['safe_ghia']['v_linf']:.3e} | "
            f"{r['safe_ghia']['centerline_max']:.3e} | {r['baseline_ghia']['centerline_max']:.3e} |"
        )
    lines += [
        "",
        "## N=64 scaling additions",
        "",
        "| Case | Picard LBE | Safe LBE | LBE x | wall x | rel ux error | residual |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in payload["scaling_2d"]:
        lines.append(
            f"| {r['label']} | {r['baseline_lbe']} | {r['safe_lbe']} | {r['safe_speedup_lbe']:.2f} | "
            f"{r['safe_speedup_wall']:.2f} | {r['rel_ux_error']:.3e} | {r['safe_residual']:.3e} |"
        )
    lines += [
        "",
        "## 3D Safe-NN smoke test",
        "",
        "| Case | Picard LBE | Safe LBE | LBE x | wall x | rel ux error | residual | converged |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in payload["safe_3d"]:
        lines.append(
            f"| {r['label']} | {r['baseline_lbe']} | {r['safe_lbe']} | {r['safe_speedup_lbe']:.2f} | "
            f"{r['safe_speedup_wall']:.2f} | {r['rel_ux_error']:.3e} | {r['safe_residual']:.3e} | {r['safe_converged']} |"
        )
    lines += [
        "",
        "Caution: N=256 production scaling and Re=1000 Safe-NN remain long-run jobs and are not included in this quick revision package.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    payload = {
        "ghia": [
            run_cavity_ghia(100, 33, 100000),
            run_cavity_ghia(400, 49, 120000),
        ],
        "scaling_2d": run_2d_scaling(),
        "safe_3d": run_3d(),
    }
    json_path = OUT / "additional_calculations.json"
    md_path = OUT / "additional_calculations.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_md(payload, md_path)
    print(f"[saved] {json_path}")
    print(f"[saved] {md_path}")


if __name__ == "__main__":
    main()
