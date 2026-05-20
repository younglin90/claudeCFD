"""IMM-LBM — Implicit Moment Method.

Solve moment equation implicitly via Picard + Stokes-projection,
with f_neq slaved by 1 LBM substep. Cheaper than HKR (no JVP×k).
"""
import time
import numpy as np
from macro_low_order import fft_stokes_inverse


def solve_imm(case, max_outer=200, tol=1e-7, picard_inner=3, kinetic_substeps=8,
               relax=0.5, verbose=True):
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
            print(f"  imm {k:3d} | res {res:.3e} | lbe {lbe_calls:5d}")
        if res < tol:
            if verbose: print(f"  CONVERGED at outer {k}")
            break

        # Implicit moment: Picard fixed point on macro
        U = case.project(f)
        f_neq = f - case.lift(U)
        for _ in range(picard_inner):
            f_picard = case.lift(U) + f_neq
            f_picard = case.lbe_step(f_picard); lbe_calls += 1
            R_U = case.project(f_picard - f_picard)  # placeholder
            R_macro = case.project(f_picard - case.lbe_step(f_picard)); lbe_calls += 1
            try:
                dU = fft_stokes_inverse(R_macro, nu)
            except Exception:
                dU = None
            if dU is None or not np.all(np.isfinite(dU)):
                break
            U = U + relax * dU
        f = case.lift(U) + f_neq
        for _ in range(kinetic_substeps):
            f = case.lbe_step(f)
        lbe_calls += kinetic_substeps
        if not np.all(np.isfinite(f)):
            break
    return f, history
