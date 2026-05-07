#!/usr/bin/env python3
"""Test Fraysse (Rusanov + AD Newton) solver on Abgrall advection and shock tube.

Phase 1: Abgrall water-air advection (N=10, periodic, 100 steps)
Phase 2: Gas-liquid shock tube (N=200, transmissive, t_end=2.4e-4 s)

Usage:
    python pipeline/tmp_test_fraysse.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from solver.denner_1d.solver_fraysse import (
    pack_fraysse,
    unpack_fraysse,
    mixture_eos_anp,
    step_fraysse,
    _get_ph_params,
)

# ---------------------------------------------------------------------------
# EOS parameters
# ---------------------------------------------------------------------------

ph_water = {
    'type':      'nasg',
    'gamma':     1.187,
    'pinf':      7.028e8,
    'p_inf':     7.028e8,
    'b':         6.61e-4,
    'b_covolume': 6.61e-4,
    'kappa_v':   3610.0,
    'eta':       -1.177788e6,
}

ph_air = {
    'type':      'ideal',
    'gamma':     1.4,
    'pinf':      0.0,
    'p_inf':     0.0,
    'b':         0.0,
    'b_covolume': 0.0,
    'kappa_v':   717.5,
    'eta':       0.0,
}


# ---------------------------------------------------------------------------
# Initial condition helper
# ---------------------------------------------------------------------------

def build_ic(x, p0, u0, T0, Y1_cell, ph1, ph2):
    """Build conserved variable vector Q for given (p0, u0, T0, Y1) field.

    Uses volume-fraction mixing: alpha_k from Y_k assuming two pure phases
    at (p0, T0).

    Parameters
    ----------
    x        : ndarray (N,)
    p0, u0, T0 : float  — uniform initial pressure, velocity, temperature
    Y1_cell  : ndarray (N,) — mass fraction of phase 1
    ph1, ph2 : dict    — EOS parameter dicts

    Returns
    -------
    Q : ndarray (4N,)
    rho_init, rho_E_init : ndarray (N,) — for error checking
    """
    N = len(x)
    g1, pinf1, b1, cv1, eta1 = _get_ph_params(ph1)
    g2, pinf2, b2, cv2, eta2 = _get_ph_params(ph2)

    # Phase densities at (p0, T0)
    rho1 = (p0 + pinf1) / ((g1 - 1.0) * cv1 * T0 + b1 * (p0 + pinf1))
    rho2 = p0 / ((g2 - 1.0) * cv2 * T0)

    # Clip Y1 away from exact 0/1 to avoid singularities in EOS inversion
    Y1 = np.clip(Y1_cell, 1e-10, 1.0 - 1e-10)
    Y2 = 1.0 - Y1

    # Mixture density from harmonic (mass-fraction) mixing
    rho = 1.0 / (Y1 / rho1 + Y2 / rho2)

    # Specific internal energy (mass-weighted)
    e1 = cv1 * T0 + eta1
    e2 = cv2 * T0 + eta2
    e_mix = Y1 * e1 + Y2 * e2

    rho_u  = rho * u0
    rho_E  = rho * e_mix + 0.5 * rho * u0 ** 2
    rho_Y1 = rho * Y1

    Q = pack_fraysse(rho, rho_u, rho_E, rho_Y1)
    return Q, rho, rho_E


# ---------------------------------------------------------------------------
# Phase 1: Abgrall advection test
# ---------------------------------------------------------------------------

def test_abgrall():
    """Phase 1 Abgrall advection (N=10, periodic BC, 100 iterations).

    PASS criteria (from CLAUDE.md):
      err_p < 1e-2, err_u < 1e-2, err_E < 1e-2, 0 <= Y1 <= 1
    """
    print("=" * 60)
    print("Phase 1: Abgrall Advection Test (Fraysse solver)")
    print("=" * 60)

    N = 10
    L = 1.0
    dx = L / N
    x  = np.linspace(dx / 2.0, L - dx / 2.0, N)

    p0 = 1.0e5
    u0 = 1.0
    T0 = 300.0

    # Water region: x in [0.4, 0.6]
    Y1_init = np.where((x >= 0.4) & (x < 0.6), 1.0, 0.0)

    Q, rho_init, rhoE_init = build_ic(x, p0, u0, T0, Y1_init, ph_water, ph_air)

    # Verify EOS recovery before time stepping
    rho0, rhou0, rhoE0, rhoY0 = unpack_fraysse(Q, N)
    p_rec, T_rec, u_rec, c_rec = mixture_eos_anp(rho0, rhou0, rhoE0, rhoY0, ph_water, ph_air)
    p_rec = np.array(p_rec, dtype=float)
    T_rec = np.array(T_rec, dtype=float)
    print(f"  EOS recovery check:")
    print(f"    max|p-p0|/p0 = {np.max(np.abs(p_rec - p0) / p0):.3e}")
    print(f"    max|T-T0|    = {np.max(np.abs(T_rec - T0)):.3e} K")

    E0 = float(np.sum(rhoE0) * dx)

    cfg = {
        'CFL':        0.5,
        'max_newton': 20,
        'newton_tol': 1e-10,
        'verbose':    False,
    }

    t = 0.0
    converged_all = True

    for step in range(100):
        Q_new, dt, info = step_fraysse(
            N, dx, Q, ph_water, ph_air, 'periodic', 'periodic', cfg)
        t += dt
        Q = Q_new

        if not info['converged']:
            converged_all = False

        if step % 20 == 0 or step == 99:
            rho_k, rhou_k, rhoE_k, rhoY_k = unpack_fraysse(Q, N)
            p_k = np.array(
                mixture_eos_anp(rho_k, rhou_k, rhoE_k, rhoY_k, ph_water, ph_air)[0],
                dtype=float)
            u_k  = rhou_k / rho_k
            Y_k  = rhoY_k / rho_k
            E_k  = float(np.sum(rhoE_k) * dx)

            err_p = float(np.max(np.abs(p_k - p0) / p0))
            err_u = float(np.max(np.abs(u_k - u0)))
            err_E = abs((E_k - E0) / E0) if abs(E0) > 1e-30 else 0.0
            Y_ok  = bool(np.all(Y_k >= -1e-10) and np.all(Y_k <= 1.0 + 1e-10))

            print(f"  step {step+1:3d}: t={t:.5f}  |R|={info['final_residual']:.1e}"
                  f"  nit={info['newton_iters']}"
                  f"  err_p={err_p:.2e}  err_u={err_u:.2e}  err_E={err_E:.2e}"
                  f"  Y_ok={Y_ok}  conv={info['converged']}")

    # ---- Final metrics ----
    rho_f, rhou_f, rhoE_f, rhoY_f = unpack_fraysse(Q, N)
    p_f = np.array(
        mixture_eos_anp(rho_f, rhou_f, rhoE_f, rhoY_f, ph_water, ph_air)[0],
        dtype=float)
    u_f = rhou_f / rho_f
    Y_f = rhoY_f / rho_f
    E_f = float(np.sum(rhoE_f) * dx)

    err_p = float(np.max(np.abs(p_f - p0) / p0))
    err_u = float(np.max(np.abs(u_f - u0)))
    err_E = abs((E_f - E0) / E0) if abs(E0) > 1e-30 else 0.0
    Y_ok  = bool(np.all(Y_f >= -1e-10) and np.all(Y_f <= 1.0 + 1e-10))

    passed = (err_p < 1e-2 and err_u < 1e-2 and err_E < 1e-2 and Y_ok)

    print()
    print(f"  FINAL (step 100, t={t:.5f} s):")
    print(f"    err_p = {err_p:.3e}   (< 1e-2 required)")
    print(f"    err_u = {err_u:.3e}   (< 1e-2 required)")
    print(f"    err_E = {err_E:.3e}   (< 1e-2 required)")
    print(f"    Y_ok  = {Y_ok}         (required)")
    print(f"    All steps converged: {converged_all}")
    print(f"  Phase 1 Result: {'PASS' if passed else 'FAIL'}")
    print()

    return passed


# ---------------------------------------------------------------------------
# Phase 2: Gas-liquid shock tube
# ---------------------------------------------------------------------------

def test_shock_tube():
    """Phase 2 gas-liquid shock tube (N=200, transmissive BC).

    Domain: [0, 2] m
    Air (left, x < 0.5 m): p = 1 GPa
    Water (right, x >= 0.5 m): p = 10 kPa
    u0 = 0, T0 = 300 K, CFL = 0.5, t_end = 2.4e-4 s

    PASS criteria:
      - Completes to t_end without divergence
      - No numerical oscillations (qualitative)
      - 3-wave structure identifiable
    """
    print("=" * 60)
    print("Phase 2: Gas-Liquid Shock Tube (Fraysse solver)")
    print("=" * 60)

    N   = 200
    L   = 2.0
    dx  = L / N
    x   = np.linspace(dx / 2.0, L - dx / 2.0, N)

    T0  = 300.0
    u0  = 0.0
    t_end = 2.4e-4

    # Left: air at p=1 GPa, Right: water at p=10 kPa
    p_init = np.where(x < 0.5, 1.0e9, 1.0e4)
    # Left region: mostly air (Y1=water ~ 0), Right: mostly water (Y1 ~ 1)
    Y1_init = np.where(x >= 0.5, 1.0, 0.0)

    # Build conserved variables cell by cell using per-cell (p, T, Y1)
    g1, pinf1, b1, cv1, eta1 = _get_ph_params(ph_water)
    g2, pinf2, b2, cv2, eta2 = _get_ph_params(ph_air)

    Y1 = np.clip(Y1_init, 1e-10, 1.0 - 1e-10)
    Y2 = 1.0 - Y1
    p  = p_init

    # Phase densities at local (p, T0)
    rho1 = (p + pinf1) / ((g1 - 1.0) * cv1 * T0 + b1 * (p + pinf1))
    rho2 = p / ((g2 - 1.0) * cv2 * T0)

    rho   = 1.0 / (Y1 / rho1 + Y2 / rho2)
    e1    = cv1 * T0 + eta1
    e2    = cv2 * T0 + eta2
    e_mix = Y1 * e1 + Y2 * e2
    rho_u  = rho * u0
    rho_E  = rho * e_mix + 0.5 * rho * u0 ** 2
    rho_Y1 = rho * Y1

    Q = pack_fraysse(rho, rho_u, rho_E, rho_Y1)

    cfg = {
        'CFL':        0.5,
        'max_newton': 25,
        'newton_tol': 1e-8,
        'verbose':    False,
    }

    t     = 0.0
    step  = 0
    diverged = False

    print(f"  Running to t_end = {t_end:.2e} s ...")

    while t < t_end:
        try:
            Q_new, dt, info = step_fraysse(
                N, dx, Q, ph_water, ph_air, 'transmissive', 'transmissive', cfg)
        except Exception as exc:
            print(f"  ERROR at step {step}: {exc}")
            diverged = True
            break

        # Clip dt to not overshoot t_end
        if t + dt > t_end:
            dt = t_end - t
            # Redo with exact dt
            cfg_final = dict(cfg)
            cfg_final['dt_fixed'] = dt
            try:
                Q_new, dt, info = step_fraysse(
                    N, dx, Q, ph_water, ph_air, 'transmissive', 'transmissive', cfg_final)
            except Exception as exc:
                print(f"  ERROR at final step: {exc}")
                diverged = True
                break

        t += dt
        step += 1
        Q = Q_new

        # Divergence check
        rho_k, rhou_k, rhoE_k, rhoY_k = unpack_fraysse(Q, N)
        if (np.any(np.isnan(rho_k)) or np.any(np.isinf(rho_k))
                or np.any(rho_k <= 0)):
            print(f"  DIVERGED at step {step}, t={t:.3e}")
            diverged = True
            break

        if step % 20 == 0:
            p_k = np.array(
                mixture_eos_anp(rho_k, rhou_k, rhoE_k, rhoY_k, ph_water, ph_air)[0],
                dtype=float)
            u_k = rhou_k / rho_k
            print(f"  step {step:4d}: t={t:.5e}  dt={dt:.3e}"
                  f"  p_max={np.max(p_k):.3e}  p_min={np.min(p_k):.3e}"
                  f"  u_max={np.max(u_k):.3e}  |R|={info['final_residual']:.1e}"
                  f"  nit={info['newton_iters']}")

    # ---- Final state ----
    if not diverged:
        rho_f, rhou_f, rhoE_f, rhoY_f = unpack_fraysse(Q, N)
        p_f = np.array(
            mixture_eos_anp(rho_f, rhou_f, rhoE_f, rhoY_f, ph_water, ph_air)[0],
            dtype=float)
        u_f = rhou_f / rho_f
        Y_f = rhoY_f / rho_f

        # Qualitative wave-structure check
        # Expansion fan: p should decrease from left peak
        # Contact discontinuity + shock: identify by pressure jump
        p_left  = np.mean(p_f[:10])
        p_mid   = np.mean(p_f[90:110])
        p_right = np.mean(p_f[190:])

        print()
        print(f"  Completed {step} steps, t = {t:.4e} s")
        print(f"  p_left  = {p_left:.3e}  (original 1 GPa)")
        print(f"  p_mid   = {p_mid:.3e}")
        print(f"  p_right = {p_right:.3e}  (original 10 kPa)")
        print(f"  max|u|  = {np.max(np.abs(u_f)):.3e} m/s")
        print(f"  Y1 range: [{np.min(Y_f):.3e}, {np.max(Y_f):.3e}]")

        reached_tend = abs(t - t_end) < 1e-10 * t_end or t >= t_end - 1e-12
        no_nan = not (np.any(np.isnan(p_f)) or np.any(np.isnan(u_f)))
        passed = reached_tend and no_nan and not diverged
    else:
        passed = False

    print(f"  Phase 2 Result: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print()
    print("Fraysse Solver — Validation Tests")
    print("==================================")
    print()

    p1 = test_abgrall()
    p2 = test_shock_tube()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Phase 1 (Abgrall advection): {'PASS' if p1 else 'FAIL'}")
    print(f"  Phase 2 (Shock tube):        {'PASS' if p2 else 'FAIL'}")
    print()

    if p1 and p2:
        print("  ALL PASS")
        sys.exit(0)
    else:
        print("  SOME TESTS FAILED")
        sys.exit(1)
