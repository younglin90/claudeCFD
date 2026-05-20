"""NDA-LBM — Nonlinear Diffusion Acceleration.

HO step + nonlinear effective-viscosity macro Stokes:
    ν_eff(u) = ν · (1 + α · |∇u|)
    ν_eff ∇² u = - F_residual_macro
"""
import time
import numpy as np
from macro_low_order import fft_stokes_inverse


def solve_nda(case, max_outer=200, tol=1e-7, ho_substeps=8, kinetic_substeps=5,
               alpha=0.5, relax=0.3, verbose=True):
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
            print(f"  nda {k:3d} | res {res:.3e} | lbe {lbe_calls:5d}")
        if res < tol:
            if verbose: print(f"  CONVERGED at outer {k}")
            break

        # HO step
        for _ in range(ho_substeps):
            f = case.lbe_step(f)
        lbe_calls += ho_substeps

        # LO step: effective viscosity from |∇u|
        U = case.project(f)
        ux = U[1]; uy = U[2]
        dux_dx = (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1)) / 2
        duy_dy = (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)) / 2
        grad_mag = np.sqrt(dux_dx ** 2 + duy_dy ** 2)
        nu_eff = nu * (1 + alpha * grad_mag.mean())

        R_macro = case.project(case.residual(f)); lbe_calls += 1
        try:
            dU = fft_stokes_inverse(R_macro, nu_eff)
        except Exception:
            dU = None
        if dU is not None and np.all(np.isfinite(dU)):
            f_neq = f - case.lift(U)
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
