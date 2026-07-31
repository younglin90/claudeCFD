"""Generate quantitative revision data for the Safe-NN LBM manuscript.

This script is intentionally paper-facing rather than optimizer-facing. It
recomputes the six core cases with the grid sizes that should be used
consistently in the manuscript, and writes both machine-readable JSON and a
Korean markdown summary for revision work.
"""

from __future__ import annotations

import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from lbm_channel import ChannelCase
from lbm_core import LBMCavity, moments as cavity_moments
from lbm_couette import CouetteCase
from lbm_periodic import KolmogorovCase, apply_spectral_schur, build_spectral_schur
from lbm_voxel import VoxelCase, build_cylinder_mask
from solver_anderson import solve_anderson
from solver_baseline import solve_baseline
from solver_scmk import solve_baseline_periodic


OUT_DIR = Path("paper_revision_data")
ONEDRIVE_DIR = Path(
    "/mnt/c/Users/user/OneDrive/[논문투고]/할거_0_LBM_steady_state_가속화"
)
ONEDRIVE_OUT = ONEDRIVE_DIR / "SafeNN_LBM_revision_data_2026-05-22.md"


def macro_of(case, f):
    if hasattr(case, "macro"):
        return case.macro(f)
    return cavity_moments(f)


def mass_of(case, f) -> float:
    rho = macro_of(case, f)[0]
    return float(np.sum(rho))


def velocity_error(case, f_ref, f_test):
    _, ux_ref, uy_ref = macro_of(case, f_ref)
    _, ux, uy = macro_of(case, f_test)
    dux = ux - ux_ref
    duy = uy - uy_ref
    mag = np.sqrt(dux * dux + duy * duy)
    ref_mag = np.sqrt(ux_ref * ux_ref + uy_ref * uy_ref)
    denom = max(float(np.sqrt(np.sum(ref_mag * ref_mag))), 1e-30)
    return {
        "rms_abs": float(np.sqrt(np.mean(mag * mag))),
        "linf_abs": float(np.max(mag)),
        "rel_l2": float(np.sqrt(np.sum(mag * mag)) / denom),
        "rel_linf": float(np.max(mag) / max(float(np.max(ref_mag)), 1e-30)),
        "ux_rel_l2": float(np.linalg.norm(dux) / max(np.linalg.norm(ux_ref), 1e-30)),
    }


def solve_safe_nn_with_stats(
    case,
    max_outer=200,
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
    s_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0
    beta = 0.0
    res_prev = np.inf
    streak_no_reject = 0
    beta_cap = beta_max
    stats = {
        "lookahead_evaluations": 0,
        "lookahead_rejections": 0,
        "residual_increase_restarts": 0,
        "nan_fallbacks": 0,
        "short_K_steps": 0,
        "newton_steps": 0,
        "gmres_probe_calls": 0,
        "max_beta_used": 0.0,
    }

    for k in range(max_outer):
        r = case.residual(f)
        lbe_calls += 1
        res = case._fast_norm(r) / np.sqrt(n_full)
        history.append((k, res, lbe_calls, time.perf_counter() - t0))
        if res < tol:
            break

        if res > res_prev:
            beta *= 0.7
            streak_no_reject = 0
            beta_cap = beta_max
            stats["residual_increase_restarts"] += 1
        else:
            beta = min(beta_cap, beta + 0.15)
            if streak_no_reject >= 2:
                beta_cap = min(0.95, beta_max + 0.2)

        if beta > 0.3:
            y = f + beta * (f - f_prev)
            r_y = y - case.lbe_step(y)
            lbe_calls += 1
            stats["lookahead_evaluations"] += 1
            eps_eff = eps_accept + 0.2 * beta
            if (
                case._fast_norm(r_y) > (1.0 + eps_eff) * case._fast_norm(r)
                or not np.all(np.isfinite(r_y))
            ):
                y = f.copy()
                r_y = r
                beta *= 0.7
                streak_no_reject = 0
                beta_cap = beta_max
                stats["lookahead_rejections"] += 1
            else:
                streak_no_reject += 1
        else:
            y = f
            r_y = r
            streak_no_reject += 1

        stats["max_beta_used"] = max(stats["max_beta_used"], float(beta))
        norm_y = case._fast_norm(y)
        probes = [0]

        def matvec(v_flat):
            w = v_flat.reshape(case.shape)
            probes[0] += 1
            return case.jvp(w, y, r_y, norm_f_cached=norm_y).ravel()

        def precond(r_flat):
            return apply_spectral_schur(case, r_flat.reshape(case.shape), s_inv).ravel()

        a_op = LinearOperator((n_full, n_full), matvec=matvec, dtype=np.float64)
        m_op = LinearOperator((n_full, n_full), matvec=precond, dtype=np.float64)
        df, _ = gmres(
            a_op,
            -r_y.ravel(),
            M=m_op,
            rtol=krylov_tol,
            atol=krylov_tol * np.linalg.norm(r_y) * 1e-3,
            maxiter=1,
            restart=2 * krylov_max,
        )
        lbe_calls += probes[0]
        stats["gmres_probe_calls"] += probes[0]
        stats["newton_steps"] += 1

        if not np.all(np.isfinite(df)):
            stats["nan_fallbacks"] += 1
            break

        f_new = y + df.reshape(case.shape)
        if res < 3e-5 and res < res_prev:
            k_eff = max(5, kinetic_substeps // 2)
            stats["short_K_steps"] += 1
        else:
            k_eff = kinetic_substeps
        for _ in range(k_eff):
            f_new = case.lbe_step(f_new)
        lbe_calls += k_eff
        if not np.all(np.isfinite(f_new)):
            f_new = f
            for _ in range(kinetic_substeps):
                f_new = case.lbe_step(f_new)
            lbe_calls += kinetic_substeps
            beta = 0.0
            stats["nan_fallbacks"] += 1

        f_prev = f
        f = f_new
        res_prev = res

    stats["final_beta"] = float(beta)
    return f, history, stats


def build_cases():
    n = 32
    chi = np.ones((n, n))
    rng = np.random.RandomState(7)
    radius = max(2, n // 12)
    centers = []
    for _ in range(6):
        cx = int(rng.randint(radius, n - radius))
        cy = int(rng.randint(radius, n - radius))
        centers.append([cx, cy])
        chi *= build_cylinder_mask(n, cx, cy, radius)

    return [
        {
            "key": "kolmogorov_N32",
            "label": "Kolmogorov N=32",
            "factory": lambda: KolmogorovCase(N=32, nu=0.05, F0=2e-4, kf=1),
            "tol": 1e-7,
            "max_baseline": 50000,
            "baseline": "periodic",
            "params": {"N": 32, "nu": 0.05, "F0": 2e-4, "kf": 1},
        },
        {
            "key": "channel_N32",
            "label": "Channel N=32",
            "factory": lambda: ChannelCase(N=32, nu=0.05, F0=1e-5),
            "tol": 1e-7,
            "max_baseline": 50000,
            "baseline": "periodic",
            "params": {"N": 32, "nu": 0.05, "F0": 1e-5},
        },
        {
            "key": "couette_N32",
            "label": "Couette N=32",
            "factory": lambda: CouetteCase(N=32, nu=0.05, U_wall=0.05),
            "tol": 1e-7,
            "max_baseline": 50000,
            "baseline": "periodic",
            "params": {"N": 32, "nu": 0.05, "U_wall": 0.05},
        },
        {
            "key": "cavity_Re100_N33",
            "label": "Cavity Re=100 N=33",
            "factory": lambda: LBMCavity(N=33, Re=100, U_wall=0.1),
            "tol": 5e-7,
            "max_baseline": 100000,
            "baseline": "cavity",
            "params": {"N": 33, "Re": 100, "U_wall": 0.1},
        },
        {
            "key": "multi_cylinder_N32",
            "label": "Multi-cylinder N=32",
            "factory": lambda: VoxelCase(chi.copy(), nu=0.05, F0=2e-4, kf=1),
            "tol": 1e-7,
            "max_baseline": 50000,
            "baseline": "periodic",
            "params": {
                "N": 32,
                "nu": 0.05,
                "F0": 2e-4,
                "kf": 1,
                "cylinders": centers,
                "radius": radius,
                "fluid_fraction": float(chi.mean()),
            },
        },
        {
            "key": "cavity_Re400_N49",
            "label": "Cavity Re=400 N=49",
            "factory": lambda: LBMCavity(N=49, Re=400, U_wall=0.1),
            "tol": 5e-7,
            "max_baseline": 120000,
            "baseline": "cavity",
            "params": {"N": 49, "Re": 400, "U_wall": 0.1},
        },
    ]


def run_baseline(case, spec):
    if spec["baseline"] == "cavity":
        return solve_baseline(
            case,
            max_steps=spec["max_baseline"],
            tol=spec["tol"],
            check_every=200,
            verbose=False,
        )
    return solve_baseline_periodic(
        case,
        max_steps=spec["max_baseline"],
        tol=spec["tol"],
        check_every=200,
        verbose=False,
    )


def fmt(x, digits=3):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "nan"
    if abs(x) >= 100:
        return f"{x:.1f}"
    if abs(x) >= 10:
        return f"{x:.2f}"
    return f"{x:.{digits}e}"


def write_markdown(results, summary, path):
    lines = []
    lines.append("# Safe-NN-SCMK JCP Major Revision 보완 계산 요약")
    lines.append("")
    lines.append(f"Generated: 2026-05-22 KST")
    lines.append("")
    lines.append("## 1. 결론")
    lines.append("")
    lines.append(
        f"- 6개 핵심 case(Table 1 grid 통일: Re=100 N=33, Re=400 N=49)에서 "
        f"Safe-NN LBE-call speedup 산술평균은 **{summary['arith_speedup']:.2f}x**, "
        f"기하평균은 **{summary['geom_speedup']:.2f}x**입니다."
    )
    lines.append(
        f"- Re=400 stress test를 제외한 5개 표준 case 산술평균은 "
        f"**{summary['arith_speedup_5_standard']:.2f}x**입니다."
    )
    lines.append(
        f"- 최악 상대 L2 velocity error는 **{summary['worst_rel_l2']:.3e}**, "
        f"최악 RMS absolute velocity error는 **{summary['worst_rms_abs']:.3e}**입니다."
    )
    lines.append(
        f"- Safe-NN 수렴 case는 {summary['safe_converged_count']}/6, "
        f"Anderson 수렴 case는 {summary['anderson_converged_count']}/6입니다."
    )
    lines.append("")
    lines.append("원고에는 52x와 44.6x를 혼용하지 말고 아래처럼 분리해서 쓰는 것이 안전합니다.")
    lines.append("")
    lines.append(
        "> Across the five standard cases excluding the Re=400 stress test, "
        f"the arithmetic mean LBE-call speedup was {summary['arith_speedup_5_standard']:.1f}x. "
        f"Across all six cases, the arithmetic and geometric mean speedups were "
        f"{summary['arith_speedup']:.1f}x and {summary['geom_speedup']:.1f}x, respectively."
    )
    lines.append("")
    lines.append("## 2. Speedup / wall-clock / Anderson baseline")
    lines.append("")
    lines.append(
        "| Case | Picard LBE | Safe-NN LBE | Safe-NN LBE x | Safe-NN wall x | "
        "Anderson LBE x | Safe residual | Anderson residual |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['label']} | {r['baseline_lbe']} | {r['safe_lbe']} | "
            f"{r['safe_speedup_lbe']:.2f} | {r['safe_speedup_wall']:.2f} | "
            f"{r['anderson_speedup_lbe']:.2f} | {r['safe_residual']:.3e} | "
            f"{r['anderson_residual']:.3e} |"
        )
    lines.append("")
    lines.append("## 3. 정량 accuracy table")
    lines.append("")
    lines.append(
        "| Case | RMS abs velocity diff | L_inf abs velocity diff | Relative L2 | "
        "Relative L_inf | ux relative L2 | Safe mass drift | Baseline mass drift |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        e = r["velocity_error"]
        lines.append(
            f"| {r['label']} | {e['rms_abs']:.3e} | {e['linf_abs']:.3e} | "
            f"{e['rel_l2']:.3e} | {e['rel_linf']:.3e} | {e['ux_rel_l2']:.3e} | "
            f"{r['safe_mass_drift']:.3e} | {r['baseline_mass_drift']:.3e} |"
        )
    lines.append("")
    lines.append("## 4. Safeguard / fallback statistics")
    lines.append("")
    lines.append(
        "| Case | Newton steps | Lookahead evals | Rejected lookaheads | "
        "Residual-increase restarts | NaN fallbacks | Short-K steps | Max beta |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        s = r["safe_stats"]
        lines.append(
            f"| {r['label']} | {s['newton_steps']} | {s['lookahead_evaluations']} | "
            f"{s['lookahead_rejections']} | {s['residual_increase_restarts']} | "
            f"{s['nan_fallbacks']} | {s['short_K_steps']} | {s['max_beta_used']:.2f} |"
        )
    lines.append("")
    lines.append("## 5. Grid-size 정리 권고")
    lines.append("")
    lines.append("- 원고 전체에서 cavity Re=100은 `N=33`, cavity Re=400 stress는 `N=49`로 통일하십시오.")
    lines.append("- 기존 figure caption의 `N=65` 표기는 Table 1/Table 2와 충돌하므로 삭제하거나 별도 Ghia-validation appendix로 분리하십시오.")
    lines.append("- `N=65` Ghia figure를 유지하려면 main benchmark speedup table과 다른 목적의 문헌 검증 figure임을 caption 첫 문장에 명시해야 합니다.")
    lines.append("")
    lines.append("## 6. 원고 문장 교체안")
    lines.append("")
    lines.append("### Abstract/Results speedup")
    lines.append("")
    lines.append(
        "Across the five standard benchmarks, Safe-NN-SCMK achieved an arithmetic mean "
        f"LBE-call speedup of {summary['arith_speedup_5_standard']:.1f}x over Picard LBM. "
        f"Including the Re=400 cavity stress test, the arithmetic and geometric mean "
        f"speedups were {summary['arith_speedup']:.1f}x and {summary['geom_speedup']:.1f}x, "
        "respectively."
    )
    lines.append("")
    lines.append("### Novelty 완화")
    lines.append("")
    lines.append(
        "This work appears to be among the first attempts to combine Nesterov-type "
        "lookahead, AP-Schur moment preconditioning, and residual-monotone safeguarding "
        "for native-residual steady-state LBM acceleration."
    )
    lines.append("")
    lines.append("### Accuracy 표현")
    lines.append("")
    lines.append(
        "The Safe-NN solution was compared directly with the converged Picard-LBM "
        "reference. Table X reports RMS, L_inf, relative L2 velocity differences, "
        "final native residuals, and mass drift for each benchmark."
    )
    lines.append("")
    lines.append("## 7. 재현성 정보")
    lines.append("")
    lines.append("- Nonlinear tolerance: `1e-7` for Kolmogorov/channel/Couette/multi-cylinder; `5e-7` for cavity.")
    lines.append("- Krylov method: SciPy GMRES/FGMRES-style right-preconditioned solve via `scipy.sparse.linalg.gmres`.")
    lines.append("- GMRES setting: `maxiter=1`, `restart=2*krylov_max=20`, `krylov_max=10`, `krylov_tol=1e-3`.")
    lines.append("- Finite-difference JVP perturbation: `eps = 1e-7 * (||f|| + 1) / ||w||`.")
    lines.append("- Post-Newton LBM relaxation: base `K=15`, shortened to `K=7` near convergence by the current Safe-NN implementation.")
    lines.append("- Nesterov/safeguard: `beta_max=0.7`, `eps_accept=0.10`; effective threshold is `eps_accept + 0.2 beta`.")
    lines.append("- FFT backend: NumPy FFT through the AP-Schur preconditioner.")
    lines.append(f"- Python: `{platform.python_version()}`; platform: `{platform.platform()}`.")
    lines.append("")
    lines.append("## 8. 아직 보완해야 할 항목")
    lines.append("")
    lines.append("- JCP 제출 전에는 `N=64,128,256`까지 확장한 production grid scaling을 별도 실행해야 합니다. 현재 repo의 `results_scaling/summary.json`은 N=128까지 있지만 channel N=96/128에서 수렴 flag가 false입니다.")
    lines.append("- 3D Safe-NN 자체 검증은 아직 없습니다. 기존 `verify_final_log.json`의 3D 값은 SCMK-3D 자료이므로 Safe-NN claim에는 직접 쓰지 않는 편이 안전합니다.")
    lines.append("- Ghia centerline error는 기존 `results_ghia/summary.json` 값을 appendix 성격으로 쓰되, main benchmark grid와 혼동하지 않게 분리해야 합니다.")
    lines.append("- References 중복, `[48] Saad`, `[AUTHOR VERIFY]`, `[CITATION NEEDED]`는 제출 전 전부 제거해야 합니다.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    results = []
    for spec in build_cases():
        print(f"[run] {spec['label']}", flush=True)
        case_b = spec["factory"]()
        case_s = spec["factory"]()
        case_a = spec["factory"]()
        initial_mass = mass_of(case_b, case_b.initial_field())

        t0 = time.perf_counter()
        f_b, hist_b = run_baseline(case_b, spec)
        baseline_wall = time.perf_counter() - t0

        t0 = time.perf_counter()
        f_s, hist_s, safe_stats = solve_safe_nn_with_stats(
            case_s,
            max_outer=200,
            tol=spec["tol"],
            krylov_max=10,
            krylov_tol=1e-3,
            kinetic_substeps=15,
            beta_max=0.7,
            eps_accept=0.10,
        )
        safe_wall = time.perf_counter() - t0

        t0 = time.perf_counter()
        f_a, hist_a = solve_anderson(
            case_a,
            max_iter=max(1, spec["max_baseline"] // 2),
            tol=spec["tol"],
            m=5,
            beta=1.0,
            safeguard=False,
            verbose=False,
        )
        anderson_wall = time.perf_counter() - t0

        baseline_lbe = int(hist_b[-1][2])
        safe_lbe = int(hist_s[-1][2])
        anderson_lbe = int(hist_a[-1][2])
        result = {
            "key": spec["key"],
            "label": spec["label"],
            "params": spec["params"],
            "tol": spec["tol"],
            "baseline_lbe": baseline_lbe,
            "baseline_wall": float(baseline_wall),
            "baseline_residual": float(hist_b[-1][1]),
            "baseline_converged": bool(hist_b[-1][1] < spec["tol"] * 5),
            "safe_lbe": safe_lbe,
            "safe_wall": float(safe_wall),
            "safe_residual": float(hist_s[-1][1]),
            "safe_converged": bool(hist_s[-1][1] < spec["tol"] * 5),
            "safe_speedup_lbe": float(baseline_lbe / max(safe_lbe, 1)),
            "safe_speedup_wall": float(baseline_wall / max(safe_wall, 1e-12)),
            "anderson_lbe": anderson_lbe,
            "anderson_wall": float(anderson_wall),
            "anderson_residual": float(hist_a[-1][1]),
            "anderson_converged": bool(hist_a[-1][1] < spec["tol"] * 5),
            "anderson_speedup_lbe": float(baseline_lbe / max(anderson_lbe, 1)),
            "anderson_speedup_wall": float(baseline_wall / max(anderson_wall, 1e-12)),
            "velocity_error": velocity_error(case_b, f_b, f_s),
            "safe_mass_drift": float(abs(mass_of(case_s, f_s) - initial_mass) / max(abs(initial_mass), 1e-30)),
            "baseline_mass_drift": float(abs(mass_of(case_b, f_b) - initial_mass) / max(abs(initial_mass), 1e-30)),
            "safe_stats": safe_stats,
        }
        results.append(result)

    speedups = np.array([r["safe_speedup_lbe"] for r in results], dtype=float)
    standard_speedups = np.array([r["safe_speedup_lbe"] for r in results[:5]], dtype=float)
    summary = {
        "arith_speedup": float(np.mean(speedups)),
        "geom_speedup": float(np.exp(np.mean(np.log(speedups)))),
        "arith_speedup_5_standard": float(np.mean(standard_speedups)),
        "geom_speedup_5_standard": float(np.exp(np.mean(np.log(standard_speedups)))),
        "worst_rel_l2": float(max(r["velocity_error"]["rel_l2"] for r in results)),
        "worst_rms_abs": float(max(r["velocity_error"]["rms_abs"] for r in results)),
        "safe_converged_count": int(sum(r["safe_converged"] for r in results)),
        "anderson_converged_count": int(sum(r["anderson_converged"] for r in results)),
    }

    payload = {"summary": summary, "results": results}
    json_path = OUT_DIR / "safenn_revision_metrics.json"
    md_path = OUT_DIR / "SafeNN_LBM_revision_data_2026-05-22.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(results, summary, md_path)
    if ONEDRIVE_DIR.exists():
        write_markdown(results, summary, ONEDRIVE_OUT)
    print(f"[saved] {json_path}")
    print(f"[saved] {md_path}")
    if ONEDRIVE_DIR.exists():
        print(f"[saved] {ONEDRIVE_OUT}")
    print(f"{summary['arith_speedup']:.6f} {summary['geom_speedup']:.6f}")


if __name__ == "__main__":
    main()
