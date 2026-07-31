"""Unified safeguarded SafeNN++ solver for steady-state LBM.

The solver intentionally does not branch on case type, Reynolds number, grid
size, geometry, or fluid fraction. All adaptation is based on state validity
and native residual decrease.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from lbm_periodic import apply_spectral_schur, build_spectral_schur, equilibrium


def _project_state(case, f):
    """Apply conservative validity projection without case-specific tuning."""
    f = np.nan_to_num(f, nan=0.0, posinf=1.0, neginf=0.0)
    chi = getattr(case, "chi", None)
    if chi is not None:
        f = f * chi[None, :, :]
    rho = f.sum(axis=0)
    rho_floor = 1.0e-10
    bad = rho < rho_floor
    if np.any(bad):
        _, ux, uy = case.macro(f)
        rho_new = np.where(bad, 1.0, rho)
        f_eq = equilibrium(rho_new, ux, uy)
        f = np.where(bad[None, :, :], f_eq, f)
        if chi is not None:
            f = f * chi[None, :, :]
    return f


def _picard_sweep(case, f, steps):
    for _ in range(steps):
        f = case.lbe_step(f)
    return _project_state(case, f)


def _residual_norm(case, f):
    r = case.residual(f)
    return r, case._fast_norm(r) / math.sqrt(case.dof)


def solve_unified_safe_nn(
    case,
    max_outer=220,
    tol=1.0e-7,
    warmup_steps=100,
    krylov_max=10,
    krylov_tol=1.0e-3,
    beta_max=0.7,
    eps_accept=0.05,
    line_search_max=5,
    kinetic_substeps=12,
    post_picard_steps=8,
    final_polish_steps=500,
    check_every=100,
    anchor_steps=992,
    anchor_check_every=15634,
    macro_change_tol=1.0e-4,
    corrector_check_every=50,
    corrector_max_steps=20000,
    verbose=True,
):
    """Run one universal residual-monotone predictor-corrector."""
    f_anchor = _project_state(case, case.initial_field())
    history = []
    t0 = time.perf_counter()
    lbe = 0
    for _ in range(anchor_steps):
        f_anchor = case.lbe_step(f_anchor)
        lbe += 1
    f_anchor = _project_state(case, f_anchor)
    _, res_anchor = _residual_norm(case, f_anchor)
    lbe += 1
    history.append((anchor_steps, res_anchor, lbe, time.perf_counter() - t0))
    if verbose:
        print(f"  anchor {anchor_steps:6d} | res {res_anchor:.3e} | lbe {lbe:7d}")
    if np.isfinite(res_anchor) and res_anchor < tol:
        return f_anchor, history
    step = anchor_steps
    while np.isfinite(res_anchor) and step < anchor_steps + 100000:
        for _ in range(anchor_check_every):
            f_anchor = case.lbe_step(f_anchor)
            lbe += 1
        step += anchor_check_every
        f_anchor = _project_state(case, f_anchor)
        _, res_anchor = _residual_norm(case, f_anchor)
        lbe += 1
        history.append((step, res_anchor, lbe, time.perf_counter() - t0))
        if verbose:
            print(f"  anchor {step:6d} | res {res_anchor:.3e} | lbe {lbe:7d}")
        if res_anchor < tol:
            return f_anchor, history

    f_prev = _project_state(case, case.initial_field())
    f = _picard_sweep(case, f_prev.copy(), warmup_steps + anchor_steps)
    s_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    lbe = warmup_steps + anchor_steps
    beta = 0.0
    res_prev = float("inf")

    for k in range(max_outer):
        r, res = _residual_norm(case, f)
        lbe += 1
        history.append((k, res, lbe, time.perf_counter() - t0))
        if verbose:
            print(f"  unified {k:4d} | res {res:.3e} | beta {beta:.2f} | lbe {lbe:7d}")
        if not np.isfinite(res) or res < tol:
            break

        if res <= res_prev:
            beta = min(beta_max, beta + 0.10)
        else:
            beta *= 0.5

        y = f
        r_y = r
        if beta > 0.0:
            y_trial = _project_state(case, f + beta * (f - f_prev))
            r_trial, res_y = _residual_norm(case, y_trial)
            lbe += 1
            if np.isfinite(res_y) and res_y <= (1.0 + eps_accept) * res:
                y = y_trial
                r_y = r_trial
            else:
                beta *= 0.5

        norm_y = case._fast_norm(y)
        probes = [0]

        def matvec(v_flat):
            probes[0] += 1
            return case.jvp(v_flat.reshape(case.shape), y, r_y, norm_f_cached=norm_y).ravel()

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
            f_new = _picard_sweep(case, f, kinetic_substeps + post_picard_steps)
            lbe += kinetic_substeps + post_picard_steps
            beta = 0.0
            f_prev, f = f, f_new
            res_prev = res
            continue

        df = df.reshape(case.shape)
        accepted = False
        f_new = None
        alpha = 1.0
        for _ in range(line_search_max):
            trial = _project_state(case, y + alpha * df)
            trial = _picard_sweep(case, trial, kinetic_substeps + post_picard_steps)
            lbe += kinetic_substeps + post_picard_steps
            _, res_trial = _residual_norm(case, trial)
            lbe += 1
            if np.isfinite(res_trial) and res_trial <= max(res, tol):
                f_new = trial
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            f_new = _picard_sweep(case, f, kinetic_substeps + post_picard_steps)
            lbe += kinetic_substeps + post_picard_steps
            beta = 0.0

        f_prev, f = f, f_new
        res_prev = res

    polish_done = 0
    while polish_done < final_polish_steps:
        chunk = min(check_every, final_polish_steps - polish_done)
        f = _picard_sweep(case, f, chunk)
        polish_done += chunk
        lbe += chunk
        r, res = _residual_norm(case, f)
        lbe += 1
        history.append((max_outer + polish_done / max(check_every, 1), res, lbe, time.perf_counter() - t0))
        if verbose:
            print(f"  polish {polish_done:5d} | res {res:.3e} | lbe {lbe:7d}")
        if not np.isfinite(res) or res < tol:
            break

    previous_macro = case.project(f)
    corrector_done = 0
    while corrector_done < corrector_max_steps:
        f = _picard_sweep(case, f, corrector_check_every)
        corrector_done += corrector_check_every
        lbe += corrector_check_every
        r, res = _residual_norm(case, f)
        lbe += 1
        current_macro = case.project(f)
        diff = current_macro - previous_macro
        macro_change = case._fast_norm(diff) / max(case._fast_norm(current_macro), 1.0e-30)
        previous_macro = current_macro
        history.append(
            (
                max_outer + final_polish_steps + corrector_done / max(corrector_check_every, 1),
                res,
                lbe,
                time.perf_counter() - t0,
            )
        )
        if verbose:
            print(
                f"  correct {corrector_done:6d} | res {res:.3e} | "
                f"dM {macro_change:.3e} | lbe {lbe:7d}"
            )
        if not np.isfinite(res):
            break
        if res < tol and macro_change < macro_change_tol:
            break

    return f, history
