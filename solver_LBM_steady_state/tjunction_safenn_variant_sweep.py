"""Sweep SafeNN-family variants on the T-junction benchmark.

This is an experiment harness, not the public proposed solver. It compares
residual-only SafeNN, macro-guarded SafeNN, trust-region SafeNN,
pseudo-transient SafeNN, and a Koopman proposal guarded by native residual.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

import paper_60case_benchmark as bench
from lbm_periodic import apply_spectral_schur, build_spectral_schur
from solver_safe_nn import solve_safe_nn
from solver_unified_safe_nn import _picard_sweep, _project_state, _residual_norm, solve_unified_safe_nn


OUT = Path("paper_revision_data") / "bench2_focus" / "tjunction_safenn_sweep"


def macro_residual_norm(case, r):
    mr = case.project(r)
    return case._fast_norm(mr) / math.sqrt(case.macro_dof)


def velocity_rel_l2(case_ref, f_ref, case, f):
    fluid = getattr(case_ref, "chi", np.ones((case_ref.N, case_ref.N), dtype=bool)) > 0
    return bench.velocity_error(case_ref, f_ref, case, f, fluid_mask=fluid)["rel_l2"]


def finalize_row(name, family, case_ref, f_ref, case, f, hist, baseline_lbe, tol, extra=None):
    _, res = _residual_norm(case, f)
    rel = velocity_rel_l2(case_ref, f_ref, case, f)
    lbe = int(hist[-1][2]) if hist else 0
    row = {
        "name": name,
        "family": family,
        "lbe_calls": lbe,
        "speedup_vs_picard": baseline_lbe / max(lbe, 1),
        "final_residual": res,
        "rel_l2_vs_picard": rel,
        "converged": bool(np.isfinite(res) and res < 5.0 * tol),
        "accurate": bool(np.isfinite(rel) and rel <= 0.05),
        "strict_2x": bool(np.isfinite(res) and res < 5.0 * tol and np.isfinite(rel) and rel <= 0.05 and baseline_lbe / max(lbe, 1) >= 2.0),
    }
    if extra:
        row.update(extra)
    return row


def run_custom_safenn(
    case,
    tol,
    *,
    max_outer=24,
    min_outer=0,
    kinetic_substeps=8,
    beta_max=0.7,
    krylov_max=8,
    krylov_tol=1.0e-3,
    trust_radius=None,
    lambda_shift=0.0,
    macro_guard=False,
    line_search_max=3,
    post_picard=0,
):
    f_prev = _project_state(case, case.initial_field())
    f = f_prev.copy()
    s_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    hist = []
    t0 = time.perf_counter()
    lbe = 0
    beta = 0.0
    res_prev = float("inf")

    for k in range(max_outer):
        r, res = _residual_norm(case, f)
        mres = macro_residual_norm(case, r)
        lbe += 1
        hist.append((k, res, lbe, time.perf_counter() - t0))
        if k >= min_outer and res < tol:
            break

        beta = min(beta_max, beta + 0.12) if res <= res_prev else beta * 0.5
        y = f
        r_y = r
        res_y = res
        mres_y = mres
        if beta > 0.0:
            y_trial = _project_state(case, f + beta * (f - f_prev))
            r_trial, trial_res = _residual_norm(case, y_trial)
            trial_mres = macro_residual_norm(case, r_trial)
            lbe += 1
            ok = np.isfinite(trial_res) and trial_res <= 1.05 * res
            if macro_guard:
                ok = ok and trial_mres <= 1.05 * mres
            if ok:
                y, r_y, res_y, mres_y = y_trial, r_trial, trial_res, trial_mres
            else:
                beta *= 0.5

        norm_y = case._fast_norm(y)
        probes = [0]

        def matvec(v_flat):
            probes[0] += 1
            v = v_flat.reshape(case.shape)
            jv = case.jvp(v, y, r_y, norm_f_cached=norm_y)
            if lambda_shift:
                jv = jv + lambda_shift * v
            return jv.ravel()

        def precond(r_flat):
            return apply_spectral_schur(case, r_flat.reshape(case.shape), s_inv).ravel()

        op = LinearOperator((case.dof, case.dof), matvec=matvec, dtype=np.float64)
        mop = LinearOperator((case.dof, case.dof), matvec=precond, dtype=np.float64)
        df, info = gmres(
            op,
            -r_y.ravel(),
            M=mop,
            rtol=krylov_tol,
            atol=krylov_tol * np.linalg.norm(r_y) * 1.0e-3,
            maxiter=1,
            restart=2 * krylov_max,
        )
        lbe += probes[0]
        if info < 0 or not np.all(np.isfinite(df)):
            f_new = _picard_sweep(case, f, kinetic_substeps)
            lbe += kinetic_substeps
            f_prev, f = f, f_new
            res_prev = res
            continue

        df = df.reshape(case.shape)
        if trust_radius is not None:
            step_norm = case._fast_norm(df)
            base_norm = max(case._fast_norm(y), 1.0e-30)
            cap = trust_radius * base_norm
            if step_norm > cap:
                df *= cap / step_norm

        accepted = False
        alpha = 1.0
        f_new = None
        for _ in range(line_search_max):
            trial = _project_state(case, y + alpha * df)
            trial = _picard_sweep(case, trial, kinetic_substeps)
            lbe += kinetic_substeps
            r_trial, trial_res = _residual_norm(case, trial)
            trial_mres = macro_residual_norm(case, r_trial)
            lbe += 1
            ok = np.isfinite(trial_res) and trial_res <= max(1.05 * res_y, tol)
            if macro_guard:
                ok = ok and trial_mres <= max(1.10 * mres_y, tol)
            if ok:
                f_new = trial
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            f_new = _picard_sweep(case, f, kinetic_substeps)
            lbe += kinetic_substeps
            beta = 0.0

        f_prev, f = f, f_new
        res_prev = res

    if post_picard:
        f = _picard_sweep(case, f, post_picard)
        lbe += post_picard
        _, res = _residual_norm(case, f)
        lbe += 1
        hist.append((max_outer + post_picard, res, lbe, time.perf_counter() - t0))

    return f, hist


def dmd_predict(snaps, future_steps, stride, rank):
    x = np.stack([s.ravel() for s in snaps[:-1]], axis=1)
    y = np.stack([s.ravel() for s in snaps[1:]], axis=1)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    r = min(rank, len(s))
    u = u[:, :r]
    s = s[:r]
    v = vt[:r, :].T
    a = u.T @ y @ v @ np.diag(1.0 / s)
    z = u.T @ snaps[-1].ravel()
    for _ in range(max(0, int(round(future_steps / stride)))):
        z = a @ z
    return _project_state_like(np.nan_to_num((u @ z).reshape(snaps[-1].shape)), snaps[-1])


def _project_state_like(f, template):
    f = np.nan_to_num(f, nan=0.0, posinf=1.0, neginf=0.0)
    if template.shape == f.shape:
        return f
    return f


def run_dmd_safe(case, tol, sample_steps, stride, rank, future_steps, polish_steps):
    hist = []
    t0 = time.perf_counter()
    f = _project_state(case, case.initial_field())
    snaps = []
    lbe = 0
    for step in range(1, sample_steps + 1):
        f = case.lbe_step(f)
        lbe += 1
        if step % stride == 0:
            snaps.append(_project_state(case, f.copy()))
    pred = _project_state(case, dmd_predict(snaps, future_steps, stride, rank))
    if polish_steps:
        pred = _picard_sweep(case, pred, polish_steps)
        lbe += polish_steps
    _, res = _residual_norm(case, pred)
    lbe += len(snaps) + 1
    hist.append((sample_steps, res, lbe, time.perf_counter() - t0))
    return pred, hist


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    label, tol, factory = bench.case_factory("t_junction_n64")
    case_ref = factory()
    f_ref, hist_ref, _ = bench.run_method("picard_lbm", case_ref, tol, bench.max_steps_for("t_junction_n64"), verbose=False)
    baseline_lbe = int(hist_ref[-1][2])
    rows = []

    # Existing proposed route.
    case = factory()
    f, hist = solve_unified_safe_nn(case, tol=tol, verbose=False)
    rows.append(finalize_row("current_unified_anchor", "current", case_ref, f_ref, case, f, hist, baseline_lbe, tol))

    # Existing SafeNN implementation grid.
    for k in [4, 8, 12, 20]:
        for line in [False, True]:
            for polish in [0, 100, 300]:
                case = factory()
                f, hist = solve_safe_nn(
                    case,
                    max_outer=80,
                    tol=tol,
                    krylov_max=8,
                    krylov_tol=1.0e-3,
                    kinetic_substeps=k,
                    line_search=line,
                    final_polish_tol=tol if polish else None,
                    final_polish_max_steps=polish,
                    final_polish_check_every=max(polish, 1),
                    verbose=False,
                )
                rows.append(
                    finalize_row(
                        f"safenn_builtin_K{k}_line{int(line)}_polish{polish}",
                        "builtin_safenn",
                        case_ref,
                        f_ref,
                        case,
                        f,
                        hist,
                        baseline_lbe,
                        tol,
                    )
                )

    # Macro/trust/pseudo-transient variants.
    configs = []
    for k in [4, 8, 12]:
        configs.append((f"custom_residual_K{k}", dict(kinetic_substeps=k)))
        configs.append((f"custom_macro_K{k}", dict(kinetic_substeps=k, macro_guard=True, min_outer=3)))
        for tr in [0.02, 0.05, 0.10, 0.25]:
            configs.append((f"custom_trust{tr:g}_K{k}", dict(kinetic_substeps=k, trust_radius=tr, min_outer=3)))
            configs.append((f"custom_macro_trust{tr:g}_K{k}", dict(kinetic_substeps=k, trust_radius=tr, macro_guard=True, min_outer=3)))
        for lam in [0.01, 0.1, 1.0, 10.0]:
            configs.append((f"custom_pseudo_lam{lam:g}_K{k}", dict(kinetic_substeps=k, lambda_shift=lam, min_outer=3)))
    for name, kwargs in configs:
        case = factory()
        f, hist = run_custom_safenn(case, tol, max_outer=40, line_search_max=3, post_picard=0, **kwargs)
        rows.append(finalize_row(name, "custom_safenn", case_ref, f_ref, case, f, hist, baseline_lbe, tol))

    # Koopman/DMD proposal guarded by native residual; included as a comparator.
    for sample in [350, 400, 425, 450]:
        for rank in [8, 12, 16, 20]:
            for future in [500, 550, 600]:
                for polish in [0, 25, 50]:
                    case = factory()
                    f, hist = run_dmd_safe(case, tol, sample, 25, rank, future, polish)
                    rows.append(
                        finalize_row(
                            f"dmd_safe_sample{sample}_r{rank}_future{future}_polish{polish}",
                            "dmd_safe",
                            case_ref,
                            f_ref,
                            case,
                            f,
                            hist,
                            baseline_lbe,
                            tol,
                        )
                    )

    rows.sort(key=lambda r: (not r["strict_2x"], r["rel_l2_vs_picard"], r["lbe_calls"]))
    fields = [
        "name",
        "family",
        "lbe_calls",
        "speedup_vs_picard",
        "final_residual",
        "rel_l2_vs_picard",
        "converged",
        "accurate",
        "strict_2x",
    ]
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows({k: r[k] for k in fields} for r in rows)
    metrics = {
        "case": "t_junction_n64",
        "baseline_lbe": baseline_lbe,
        "variant_count": len(rows),
        "strict_2x_count": sum(int(r["strict_2x"]) for r in rows),
        "accurate_count": sum(int(r["accurate"]) for r in rows),
        "best_strict_2x": next((r for r in rows if r["strict_2x"]), None),
        "best_by_accuracy": rows[0],
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (OUT / "report.md").open("w", encoding="utf-8") as fh:
        fh.write("# T-junction SafeNN Variant Sweep\n\n")
        fh.write(f"Picard reference LBE: {baseline_lbe}\n\n")
        fh.write(f"Variants tested: {len(rows)}\n\n")
        fh.write(f"Strict 2x + relL2<=0.05 variants: {metrics['strict_2x_count']}\n\n")
        fh.write("## Top 15\n\n")
        fh.write("| name | family | LBE | speedup | residual | relL2 | strict |\n")
        fh.write("| --- | --- | ---: | ---: | ---: | ---: | --- |\n")
        for r in rows[:15]:
            fh.write(
                f"| {r['name']} | {r['family']} | {r['lbe_calls']} | "
                f"{r['speedup_vs_picard']:.3f} | {r['final_residual']:.3e} | "
                f"{r['rel_l2_vs_picard']:.3e} | {r['strict_2x']} |\n"
            )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
