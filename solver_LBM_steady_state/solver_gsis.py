"""GSIS-LBM — General Synthetic Iterative Scheme for LBM.

Architecture:
    HO step  : K LBM substeps on full f
    Closure  : extract π_αβ = Σ c_α c_β f_neq  (HoT stress tensor)
    LO step  : solve macro Stokes equation
                 ν ∇²u = -F_total = ∂_β π_αβ - F_body
               in Fourier space (periodic)
    Update   : f ← T·(U + δU_LO) + f_neq

Synthetic acceleration of slow-diffusion macro modes via closed LO solve.
"""
import time
import numpy as np
from lbm_periodic import CX as _CX, CY as _CY
from macro_low_order import fft_stokes_inverse, hot_stress_from_fneq, divergence_2tensor


def solve_gsis(case, max_outer=200, tol=1e-7, ho_substeps=8, kinetic_substeps=5, verbose=True):
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
            print(f"  gsis {k:3d} | res {res:.3e} | lbe {lbe_calls:5d}")
        if res < tol:
            if verbose: print(f"  CONVERGED at outer {k}")
            break

        # HO step: K LBM substeps
        for _ in range(ho_substeps):
            f = case.lbe_step(f)
        lbe_calls += ho_substeps

        # Closure: extract HoT stress from f_neq
        U = case.project(f)
        f_eq_lift = case.lift(U)
        f_neq = f - f_eq_lift
        pi_xx, pi_xy, pi_yy = hot_stress_from_fneq(f_neq, _CX, _CY)

        # LO step: macro Stokes with stress divergence as forcing
        dpi_x, dpi_y = divergence_2tensor(pi_xx, pi_xy, pi_yy)
        R_macro = np.stack([np.zeros_like(dpi_x), -dpi_x, -dpi_y], axis=0)
        try:
            dU = fft_stokes_inverse(R_macro, nu)
        except Exception:
            dU = None
        if dU is not None and np.all(np.isfinite(dU)):
            # accept correction
            f_trial = case.lift(U + 0.3 * dU) + f_neq    # under-relax LO correction
            for _ in range(kinetic_substeps):
                f_trial = case.lbe_step(f_trial)
            lbe_calls += kinetic_substeps
            if np.all(np.isfinite(f_trial)):
                R_new = case.residual(f_trial); lbe_calls += 1
                res_new = case._fast_norm(R_new) / np.sqrt(n_full)
                if res_new < res:
                    f = f_trial
    return f, history
