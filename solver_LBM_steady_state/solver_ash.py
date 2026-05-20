"""ASH-LBM — Adaptive Spectral-Schur Hybrid LBM solver.

Residual-spectrum-guided dispatch (no case-name lookup):

    Probe phase: K=5 LBM warm-up, compute spectral signature
    Classify regime: smooth_periodic / mild_wall / kinetic_heavy / stiff_broad
    Dispatch to specialized solver:
        smooth_periodic → SAN (Anderson+Newton)
        mild_wall       → Safe-NN (residual-monotone Nesterov)
        kinetic_heavy   → Lean SCMK (post-LBM smoothing dominant)
        stiff_broad     → Lean SCMK (default robust)
"""
import time
import numpy as np
from spectral_signature import compute_signature, classify_regime
from solver_lean import solve_lean
from solver_san import solve_san
from solver_safe_nn import solve_safe_nn


def solve_ash(case, max_outer=200, tol=1e-7, krylov_max=10, krylov_tol=1e-3,
               kinetic_substeps=15, verbose=True):
    t0 = time.perf_counter()
    # Probe phase
    f_probe = case.initial_field()
    lbe_probe = 0
    for _ in range(5):
        f_probe = case.lbe_step(f_probe); lbe_probe += 1
    R_probe = case.residual(f_probe); lbe_probe += 1
    sig = compute_signature(case, R_probe)
    regime = classify_regime(sig)

    if verbose:
        print(f"  ASH probe: chi_low={sig['chi_low']:.3f}  H_k={sig['H_k']:.3f}  "
              f"chi_kin={sig['chi_kin']:.3f}  regime={regime}")

    common = dict(max_outer=max_outer, tol=tol, krylov_max=krylov_max,
                   krylov_tol=krylov_tol, kinetic_substeps=kinetic_substeps,
                   verbose=verbose)

    # Wall detection: probe directional variance.
    # Pure periodic (Kolmogorov): residual energy concentrated at single kf, isotropic
    # in spectrum sense. Channel/Couette: anisotropic (y-dir wall-driven).
    R_U_for_zm = case.project(R_probe)
    R_hat = np.fft.fft2(R_U_for_zm, axes=(1, 2))
    e = np.sum(np.abs(R_hat) ** 2, axis=0)
    zero_mode_frac = float(e[0, 0] / (e.sum() + 1e-30))   # mean-flow energy fraction

    if verbose:
        print(f"  ASH zero_mode_frac={zero_mode_frac:.3f}")

    if regime == "smooth_periodic":
        if zero_mode_frac < 0.1:               # pure periodic (Kol-like, no mean flow)
            f, h = solve_san(case, anderson_m=5, newton_every=5, **common)
        else:                                  # mean-flow dominant (Channel)
            f, h = solve_safe_nn(case, beta_max=0.7, eps_accept=0.05, **common)
    elif regime == "mild_wall":
        f, h = solve_safe_nn(case, beta_max=0.7, eps_accept=0.05, **common)
    elif regime == "kinetic_heavy":
        # voxel/cavity. Safe-NN handles them with residual-monotone protection.
        f, h = solve_safe_nn(case, beta_max=0.7, eps_accept=0.05, **common)
    else:   # stiff_broad (Couette-like)
        f, h = solve_lean(case, **common)

    # Adjust history to include probe LBE
    adj_h = []
    for k, res, lbe, wall in h:
        adj_h.append((k, res, lbe + lbe_probe, wall))
    return f, adj_h
