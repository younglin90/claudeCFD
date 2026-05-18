"""3D SCMK + hybrid baseline fallback for D3Q19 cases."""

import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from lbm_3d import build_spectral_schur_3d, apply_spectral_schur_3d


def solve_scmk_3d(case, max_outer=100, tol=1e-7, krylov_max=10, krylov_tol=1e-3,
                   kinetic_substeps=15, N_check=6, min_ratio=2.0, verbose=True):
    f = case.initial_field()
    n_full = case.dof
    S_inv = build_spectral_schur_3d(case.N, omega=case.omega)
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0
    phase_b = False
    res_history = []

    for k in range(max_outer):
        R_f = case.residual(f); lbe_calls += 1
        res_norm = case._fast_norm(R_f) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, res_norm, lbe_calls, wall))
        res_history.append(res_norm)
        if verbose:
            tag = "B" if phase_b else "A"
            print(f"  3d[{tag}] {k:3d} | res {res_norm:.3e} | lbe {lbe_calls:5d} | wall {wall:.2f}s")
        if res_norm < tol:
            if verbose: print(f"  CONVERGED at iter {k}")
            break

        if not phase_b and k == N_check and res_history[0] > 0:
            ratio = res_history[0] / res_norm
            if ratio < min_ratio:
                if verbose: print(f"  STAGNATED (ratio={ratio:.2f}), switch to baseline")
                phase_b = True

        if phase_b:
            for _ in range(50):
                f = case.lbe_step(f)
            lbe_calls += 50
            continue

        # Phase A : SCMK
        norm_f = case._fast_norm(f)
        probe = [0]

        def matvec(v_flat):
            w = v_flat.reshape(case.shape)
            probe[0] += 1
            return case.jvp(w, f, R_f, norm_f_cached=norm_f).ravel()

        def precond(r_flat):
            R = r_flat.reshape(case.shape)
            return apply_spectral_schur_3d(case, R, S_inv).ravel()

        Aop = LinearOperator((n_full, n_full), matvec=matvec, dtype=np.float64)
        Mop = LinearOperator((n_full, n_full), matvec=precond, dtype=np.float64)
        rhs = -R_f.ravel()
        df_flat, info = gmres(Aop, rhs, M=Mop, rtol=krylov_tol,
                              atol=krylov_tol * np.linalg.norm(rhs) * 1e-3,
                              maxiter=1, restart=2 * krylov_max)
        lbe_calls += probe[0]

        if not np.all(np.isfinite(df_flat)):
            if verbose: print("  GMRES NaN, switch to baseline")
            phase_b = True
            continue

        df = df_flat.reshape(case.shape)
        f = f + df
        for _ in range(kinetic_substeps):
            f = case.lbe_step(f)
        lbe_calls += kinetic_substeps

    return f, history


def solve_baseline_3d(case, max_steps=200000, tol=1e-7, check_every=200, verbose=True):
    f = case.initial_field()
    n_full = case.dof
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0
    for step in range(1, max_steps + 1):
        f = case.lbe_step(f); lbe_calls += 1
        if step % check_every == 0:
            R = f - case.lbe_step(f); lbe_calls += 1
            res = case._fast_norm(R) / np.sqrt(n_full)
            wall = time.perf_counter() - t0
            history.append((step, res, lbe_calls, wall))
            if verbose and step % 1000 == 0:
                print(f"  step {step:7d} | res {res:.3e} | wall {wall:.2f}s")
            if res < tol:
                if verbose: print(f"  CONVERGED at step {step}")
                break
    return f, history
