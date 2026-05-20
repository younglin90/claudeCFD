"""VEF/QD-LBM — Variable Eddington tensor / Quasi-Diffusion closure.

Extract local closure tensor T_αβ(x) = π_αβ(x) / Σ_γ π_γγ(x) from f,
use as variable diffusion coefficient in macro Stokes correction.

Closure is more physically meaningful than fixed isotropic ν.
"""
import time
import numpy as np
from lbm_periodic import CX as _CX, CY as _CY
from macro_low_order import fft_stokes_inverse, hot_stress_from_fneq


def solve_vef(case, max_outer=200, tol=1e-7, ho_substeps=8, kinetic_substeps=5,
               relax=0.3, verbose=True):
    f = case.initial_field()
    n_full = case.dof
    nu = case.nu if hasattr(case, "nu") else 0.05
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0

    for k in range(max_outer):
        R = case.residual(f); lbe_calls += 1
        res = case._fast_norm(R) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, res, lbe_calls, wall))
        if verbose:
            print(f"  vef {k:3d} | res {res:.3e} | lbe {lbe_calls:5d}")
        if res < tol:
            if verbose: print(f"  CONVERGED at outer {k}")
            break

        for _ in range(ho_substeps):
            f = case.lbe_step(f)
        lbe_calls += ho_substeps

        U = case.project(f)
        f_neq = f - case.lift(U)
        pi_xx, pi_xy, pi_yy = hot_stress_from_fneq(f_neq, _CX, _CY)
        tr = pi_xx + pi_yy + 1e-30
        # Variable Eddington = anisotropy ratio (max 5x ν, min 0.2x ν)
        anis = np.sqrt(pi_xx * pi_xx + pi_yy * pi_yy + 2 * pi_xy * pi_xy) / tr
        nu_eff = float(nu * np.clip(1.0 + 2.0 * anis.mean(), 0.5, 5.0))

        R_macro = case.project(case.residual(f)); lbe_calls += 1
        try:
            dU = fft_stokes_inverse(R_macro, nu_eff)
        except Exception:
            dU = None
        if dU is not None and np.all(np.isfinite(dU)):
            f_trial = case.lift(U + relax * dU) + f_neq
            for _ in range(kinetic_substeps):
                f_trial = case.lbe_step(f_trial)
            lbe_calls += kinetic_substeps
            if np.all(np.isfinite(f_trial)):
                R_new = case.residual(f_trial); lbe_calls += 1
                res_new = case._fast_norm(R_new) / np.sqrt(n_full)
                if res_new < res:
                    f = f_trial
    return f, history
