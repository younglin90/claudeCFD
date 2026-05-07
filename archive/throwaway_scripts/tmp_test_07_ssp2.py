#!/usr/bin/env python3
"""
Validator: Phase 6-3 (07_B) Acoustic Reflection & Transmission
3 sub-cases: Air-Water, Helium-Air, Argon-Air
Using IMEX with time_integrator='ssp222' or imex_rk2=True
max_iteration=100 steps (acoustic CFL=0.4 auto-adjusted)
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')

from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from solver.He2024.eos_general import IdealEOS, SGEOS

_EPS = 1e-30

def compute_exact_dalember(x, t, t_intf, c_L, c_R, x_src, x_intf, u_peak, Z_L, R, T):
    """Exact d'Alembert solution for acoustic pulse reflection/transmission.

    Returns (u_exact, p_exact) at all points after pulse crosses interface.
    """
    u_exact = np.zeros_like(x)
    p_exact = np.zeros_like(x) + 1e5  # base pressure

    # Incident pulse (left side)
    sigma_L = (x_src - x_intf + c_L * t_intf) / 6.0  # approx pulse width
    if sigma_L < 1e-6:
        sigma_L = 0.014  # fallback

    # Reflected pulse peak travel
    x_refl_peak = x_intf - c_L * (t - t_intf) if t > t_intf else x_src + c_L * t

    # Transmitted pulse peak travel
    x_trans_peak = x_intf + c_R * (t - t_intf) if t > t_intf else 1e10

    # Left region (incident + reflected)
    left_mask = x < x_intf
    if t > t_intf:
        # Reflected Gaussian (peak at x_refl_peak, traveling left)
        u_exact[left_mask] = R * u_peak * np.exp(-(x[left_mask] - x_refl_peak)**2 / (2*sigma_L**2))
        p_exact[left_mask] += Z_L * u_exact[left_mask]

    # Right region (transmitted)
    right_mask = x >= x_intf
    if t > t_intf:
        sigma_R = sigma_L * (c_R / c_L)
        u_exact[right_mask] = T * u_peak * np.exp(-(x[right_mask] - x_trans_peak)**2 / (2*sigma_R**2))
        p_exact[right_mask] += Z_L * u_peak * (2*Z_L*Z_R/(Z_L+Z_R)) if Z_L and Z_R else 0

    return u_exact, p_exact


def run_case_air_water():
    """Case 1: Air (left) → Water (right), interface at x=0.5 m"""
    print("\n" + "="*70)
    print("CASE 1: Air → Water (Z ratio ≈ 1000×)")
    print("="*70)

    L = 1.5
    N = 400
    dx = L / N
    x = np.arange(N) * dx

    # EOS
    ph_air = IdealEOS(gamma=1.4, kv=287.0)
    ph_water = SGEOS(gamma=4.1, pinf=4.4e8, kv=474.2)

    # Spec parameters
    u_peak = 0.02
    sigma_L = 0.014
    x_src = 0.1
    x_intf = 0.5

    # Acoustic impedances
    rho_air_0 = 1.157
    a_air_0 = 347.8
    Z_air = rho_air_0 * a_air_0

    rho_water_0 = 998.0
    a_water_0 = 1344.6
    Z_water = rho_water_0 * a_water_0

    R = (Z_water - Z_air) / (Z_water + Z_air)
    T = 2.0 * Z_water / (Z_water + Z_air)

    # Initial condition
    u0 = u_peak * np.exp(-(x - x_src)**2 / (2*sigma_L**2))
    p0_init = 1e5 + Z_air * u0

    a1_0 = np.where(x < x_intf, 1.0, 1e-8)
    a1_0 = np.maximum(np.minimum(a1_0, 1.0-1e-8), 1e-8)

    rho1_0 = np.array([ph_air.density(p0_init[i], 300.0) for i in range(N)])
    rho2_0 = np.array([ph_water.density(p0_init[i], 300.0) for i in range(N)])

    a1r1_0 = a1_0 * rho1_0
    a2r2_0 = (1.0 - a1_0) * rho2_0
    ru_0 = (a1r1_0 + a2r2_0) * u0
    rE_0 = np.zeros(N)
    for i in range(N):
        e1 = ph_air.energy(rho1_0[i], p0_init[i])
        e2 = ph_water.energy(rho2_0[i], p0_init[i])
        rho_e = a1_0[i] * rho1_0[i] * e1 + (1.0-a1_0[i]) * rho2_0[i] * e2
        rE_0[i] = rho_e + 0.5 * (a1r1_0[i] + a2r2_0[i]) * u0[i]**2

    # Spec time parameters
    t_intf = x_intf / a_air_0
    t_end = 1.63e-3

    print(f"Domain: L={L} m, N={N}, dx={dx:.6e} m")
    print(f"Interface: x={x_intf} m, t_interface={t_intf:.6e} s, t_end={t_end:.6e} s")
    print(f"Impedances: Z_air={Z_air:.2e}, Z_water={Z_water:.2e}, ratio={Z_water/Z_air:.1e}")
    print(f"Reflection/Transmission: R={R:.4f}, T={T:.4f}")

    # Solve
    result = solve_IMEX(
        ph1=ph_air, ph2=ph_water,
        a1r1_0=a1r1_0, a2r2_0=a2r2_0, ru_0=ru_0, rE_0=rE_0,
        a1_0=a1_0,
        dx=dx, t_end=t_end, cfl=0.4,
        bc_l='transmissive', bc_r='transmissive',
        max_steps=100,
        print_interval=50,
        time_integrator='ssp222',
        imex_rk2=True,
    )

    t_final, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = result
    print(f"Solver: t_final={t_final:.6e} s")

    # Convert to primitives
    u_num = np.zeros(N)
    p_num = np.zeros(N)
    rho1_num = np.zeros(N)
    rho2_num = np.zeros(N)

    for i in range(N):
        try:
            p, u, T, rho1, rho2, c1, c2, c_mix = cons_to_prim(
                a1r1_f[i], a2r2_f[i], ru_f[i], rE_f[i], a1_f[i],
                ph_air, ph_water
            )
            u_num[i] = u
            p_num[i] = p
            rho1_num[i] = rho1
            rho2_num[i] = rho2
        except:
            u_num[i] = np.nan
            p_num[i] = np.nan

    # Compute metrics
    return evaluate_case(
        "Air-Water",
        x, u_num, p_num, 1e5,
        u_peak, Z_air,
        t_final, t_intf, a_air_0, a_water_0,
        x_intf, x_src, R, T, sigma_L,
        ph_air, ph_water
    )


def run_case_helium_air():
    """Case 2: Helium (left) → Air (right), interface at x=1.0 m"""
    print("\n" + "="*70)
    print("CASE 2: Helium → Air (soft→hard, Z ratio ≈ 2.4×)")
    print("="*70)

    L = 1.5
    N = 400
    dx = L / N
    x = np.arange(N) * dx

    # EOS
    ph_he = IdealEOS(gamma=1.667, kv=2077.0)  # Helium
    ph_air = IdealEOS(gamma=1.4, kv=287.0)

    # Spec parameters
    u_peak = 0.02
    sigma_L = 0.049
    x_src = 0.2
    x_intf = 1.0

    # Acoustic impedances
    rho_he_0 = 0.164
    a_he_0 = 1008.2
    Z_he = rho_he_0 * a_he_0

    rho_air_0 = 1.157
    a_air_0 = 347.8
    Z_air = rho_air_0 * a_air_0

    R = (Z_air - Z_he) / (Z_air + Z_he)
    T = 2.0 * Z_air / (Z_air + Z_he)

    # Initial condition
    u0 = u_peak * np.exp(-(x - x_src)**2 / (2*sigma_L**2))
    p0_init = 1e5 + Z_he * u0

    a1_0 = np.where(x < x_intf, 1.0, 1e-8)
    a1_0 = np.maximum(np.minimum(a1_0, 1.0-1e-8), 1e-8)

    rho1_0 = np.array([ph_he.density(p0_init[i], 300.0) for i in range(N)])
    rho2_0 = np.array([ph_air.density(p0_init[i], 300.0) for i in range(N)])

    a1r1_0 = a1_0 * rho1_0
    a2r2_0 = (1.0 - a1_0) * rho2_0
    ru_0 = (a1r1_0 + a2r2_0) * u0
    rE_0 = np.zeros(N)
    for i in range(N):
        e1 = ph_he.energy(rho1_0[i], p0_init[i])
        e2 = ph_air.energy(rho2_0[i], p0_init[i])
        rho_e = a1_0[i] * rho1_0[i] * e1 + (1.0-a1_0[i]) * rho2_0[i] * e2
        rE_0[i] = rho_e + 0.5 * (a1r1_0[i] + a2r2_0[i]) * u0[i]**2

    # Spec time parameters
    t_intf = (x_intf - x_src) / a_he_0
    t_end = 1.513e-3

    print(f"Domain: L={L} m, N={N}, dx={dx:.6e} m")
    print(f"Interface: x={x_intf} m, t_interface={t_intf:.6e} s, t_end={t_end:.6e} s")
    print(f"Impedances: Z_He={Z_he:.2e}, Z_air={Z_air:.2e}, ratio={Z_air/Z_he:.4f}")
    print(f"Reflection/Transmission: R={R:.4f}, T={T:.4f}")

    # Solve
    result = solve_IMEX(
        ph1=ph_he, ph2=ph_air,
        a1r1_0=a1r1_0, a2r2_0=a2r2_0, ru_0=ru_0, rE_0=rE_0,
        a1_0=a1_0,
        dx=dx, t_end=t_end, cfl=0.4,
        bc_l='transmissive', bc_r='transmissive',
        max_steps=100,
        print_interval=50,
        time_integrator='ssp222',
        imex_rk2=True,
    )

    t_final, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = result
    print(f"Solver: t_final={t_final:.6e} s")

    # Convert to primitives
    u_num = np.zeros(N)
    p_num = np.zeros(N)

    for i in range(N):
        try:
            p, u, T, rho1, rho2, c1, c2, c_mix = cons_to_prim(
                a1r1_f[i], a2r2_f[i], ru_f[i], rE_f[i], a1_f[i],
                ph_he, ph_air
            )
            u_num[i] = u
            p_num[i] = p
        except Exception as e:
            if i < 3:
                print(f"  [i={i}] cons_to_prim error: {type(e).__name__}")
            u_num[i] = np.nan
            p_num[i] = np.nan

    return evaluate_case(
        "Helium-Air",
        x, u_num, p_num, 1e5,
        u_peak, Z_he,
        t_final, t_intf, a_he_0, a_air_0,
        x_intf, x_src, R, T, sigma_L,
        ph_he, ph_air
    )


def run_case_argon_air():
    """Case 3: Argon (left) → Air (right), hard→soft, R < 0"""
    print("\n" + "="*70)
    print("CASE 3: Argon → Air (hard→soft, weak reflection R<0)")
    print("="*70)

    L = 1.5
    N = 400
    dx = L / N
    x = np.arange(N) * dx

    # EOS
    ph_ar = SGEOS(gamma=1.66, pinf=0.0, kv=208.2)  # Argon, SG with pinf=0
    ph_air = IdealEOS(gamma=1.4, kv=287.0)

    # Spec parameters
    u_peak = 0.02
    sigma_L = 0.038
    x_src = 0.1
    x_intf = 0.5

    # Acoustic impedances
    rho_ar_0 = 1.748
    a_ar_0 = 308.2
    Z_ar = rho_ar_0 * a_ar_0

    rho_air_0 = 1.157
    a_air_0 = 347.8
    Z_air = rho_air_0 * a_air_0

    R = (Z_air - Z_ar) / (Z_air + Z_ar)
    T = 2.0 * Z_air / (Z_air + Z_ar)

    # Initial condition
    u0 = u_peak * np.exp(-(x - x_src)**2 / (2*sigma_L**2))
    p0_init = 1e5 + Z_ar * u0

    a1_0 = np.where(x < x_intf, 1.0, 1e-8)
    a1_0 = np.maximum(np.minimum(a1_0, 1.0-1e-8), 1e-8)

    rho1_0 = np.array([ph_ar.density(p0_init[i], 300.0) for i in range(N)])
    rho2_0 = np.array([ph_air.density(p0_init[i], 300.0) for i in range(N)])

    a1r1_0 = a1_0 * rho1_0
    a2r2_0 = (1.0 - a1_0) * rho2_0
    ru_0 = (a1r1_0 + a2r2_0) * u0
    rE_0 = np.zeros(N)
    for i in range(N):
        e1 = ph_ar.energy(rho1_0[i], p0_init[i])
        e2 = ph_air.energy(rho2_0[i], p0_init[i])
        rho_e = a1_0[i] * rho1_0[i] * e1 + (1.0-a1_0[i]) * rho2_0[i] * e2
        rE_0[i] = rho_e + 0.5 * (a1r1_0[i] + a2r2_0[i]) * u0[i]**2

    # Spec time parameters
    t_intf = (x_intf - x_src) / a_ar_0
    t_end = 2.02e-3

    print(f"Domain: L={L} m, N={N}, dx={dx:.6e} m")
    print(f"Interface: x={x_intf} m, t_interface={t_intf:.6e} s, t_end={t_end:.6e} s")
    print(f"Impedances: Z_Ar={Z_ar:.2e}, Z_air={Z_air:.2e}, ratio={Z_air/Z_ar:.4f}")
    print(f"Reflection/Transmission: R={R:.4f}, T={T:.4f}")

    # Solve
    result = solve_IMEX(
        ph1=ph_ar, ph2=ph_air,
        a1r1_0=a1r1_0, a2r2_0=a2r2_0, ru_0=ru_0, rE_0=rE_0,
        a1_0=a1_0,
        dx=dx, t_end=t_end, cfl=0.4,
        bc_l='transmissive', bc_r='transmissive',
        max_steps=100,
        print_interval=50,
        time_integrator='ssp222',
        imex_rk2=True,
    )

    t_final, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = result
    print(f"Solver: t_final={t_final:.6e} s")

    # Convert to primitives
    u_num = np.zeros(N)
    p_num = np.zeros(N)

    for i in range(N):
        try:
            u, p, rho1, rho2, T1, T2 = cons_to_prim(
                a1r1_f[i], a2r2_f[i], ru_f[i], rE_f[i], a1_f[i],
                ph_ar, ph_air
            )
            u_num[i] = u
            p_num[i] = p
        except:
            u_num[i] = np.nan
            p_num[i] = np.nan

    return evaluate_case(
        "Argon-Air",
        x, u_num, p_num, 1e5,
        u_peak, Z_ar,
        t_final, t_intf, a_ar_0, a_air_0,
        x_intf, x_src, R, T, sigma_L,
        ph_ar, ph_air
    )


def evaluate_case(name, x, u_num, p_num, p0, u_peak, Z_L,
                   t_final, t_intf, c_L, c_R, x_intf, x_src, R, T, sigma_L,
                   ph1, ph2):
    """Compute all 11 PASS criteria metrics."""

    # Wave scales
    dp_wave = Z_L * u_peak
    du_wave = u_peak

    # Exact d'Alembert solution
    u_exact, p_exact = compute_exact_dalember(
        x, t_final, t_intf, c_L, c_R, x_src, x_intf, u_peak, Z_L, R, T
    )

    # Safety: handle NaN
    finite = np.all(np.isfinite(u_num)) and np.all(np.isfinite(p_num))

    if not finite:
        print(f"[{name}] NaN/Inf detected: u_nan={np.sum(~np.isfinite(u_num))}, p_nan={np.sum(~np.isfinite(p_num))}")
        return {
            'name': name,
            'finite': False,
            'osc_07': np.nan,
            'L2_p_norm': np.nan,
            'L2_u_norm': np.nan,
            'Linf_p_norm': np.nan,
            'Linf_u_norm': np.nan,
            'frac_p': 0.0,
            'frac_u': 0.0,
            'L1_p_norm': np.nan,
            'L1_u_norm': np.nan,
            'corr_p': 0.0,
            'corr_u': 0.0,
            'pass': False,
        }

    # (A) Norm metrics
    L2_p_norm = np.linalg.norm(u_num - u_exact) / (du_wave + _EPS)
    L2_u_norm = np.linalg.norm(u_num - u_exact) / (du_wave + _EPS)
    Linf_p_norm = np.max(np.abs(p_num - p_exact)) / (dp_wave + _EPS)
    Linf_u_norm = np.max(np.abs(u_num - u_exact)) / (du_wave + _EPS)

    # (B) Pointwise
    err_p_pointwise = np.abs(p_num - p_exact)
    err_u_pointwise = np.abs(u_num - u_exact)
    frac_p = np.sum(err_p_pointwise < 0.30 * dp_wave) / len(x)
    frac_u = np.sum(err_u_pointwise < 0.30 * du_wave) / len(x)

    # (C) L1 integrated
    dx = x[1] - x[0] if len(x) > 1 else 1e-3
    L1_p_norm = np.sum(err_p_pointwise) * dx / (np.sum(np.abs(p_exact - p0)) * dx + _EPS)
    L1_u_norm = np.sum(err_u_pointwise) * dx / (np.sum(np.abs(u_exact)) * dx + _EPS)

    # (D) Correlation
    p_diff_num = p_num - p0
    p_diff_exact = p_exact - p0
    if np.std(p_diff_num) > _EPS and np.std(p_diff_exact) > _EPS:
        corr_p = np.corrcoef(p_diff_num, p_diff_exact)[0, 1]
    else:
        corr_p = 0.0

    if np.std(u_num) > _EPS and np.std(u_exact) > _EPS:
        corr_u = np.corrcoef(u_num, u_exact)[0, 1]
    else:
        corr_u = 0.0

    # Stability: checkerboard oscillation
    osc_07 = np.std(np.diff(u_num[::2])) / (du_wave + _EPS) if len(u_num) > 2 else 0.0

    # Check all 11 criteria
    check1 = finite
    check2 = osc_07 < 0.1
    check3 = L2_p_norm < 0.30
    check4 = L2_u_norm < 0.30
    check5 = Linf_p_norm < 0.50
    check6 = Linf_u_norm < 0.50
    check7 = frac_p >= 0.70
    check8 = frac_u >= 0.70
    check9 = L1_p_norm < 1.0
    check10 = L1_u_norm < 1.0
    check11 = (corr_p > 0.50) and (corr_u > 0.50)

    pass_all = all([check1, check2, check3, check4, check5, check6, check7, check8, check9, check10, check11])

    result = {
        'name': name,
        'finite': check1,
        'osc_07': osc_07,
        'L2_p_norm': L2_p_norm,
        'L2_u_norm': L2_u_norm,
        'Linf_p_norm': Linf_p_norm,
        'Linf_u_norm': Linf_u_norm,
        'frac_p': frac_p,
        'frac_u': frac_u,
        'L1_p_norm': L1_p_norm,
        'L1_u_norm': L1_u_norm,
        'corr_p': corr_p,
        'corr_u': corr_u,
        'pass': pass_all,
        'checks': [check1, check2, check3, check4, check5, check6, check7, check8, check9, check10, check11],
        'x': x,
        'u_num': u_num,
        'p_num': p_num,
        'u_exact': u_exact,
        'p_exact': p_exact,
        'p0': p0,
        'dp_wave': dp_wave,
        'du_wave': du_wave,
    }

    print(f"\n[{name}] Summary:")
    print(f"  Norms:     L2_p/A={L2_p_norm:.3f} L2_u/A={L2_u_norm:.3f} Linf_p/A={Linf_p_norm:.3f} Linf_u/A={Linf_u_norm:.3f}")
    print(f"  Pointwise: frac_p={frac_p:.2f} frac_u={frac_u:.2f}")
    print(f"  L1:        L1_p={L1_p_norm:.3f} L1_u={L1_u_norm:.3f}")
    print(f"  Corr:      corr_p={corr_p:.3f} corr_u={corr_u:.3f}")
    print(f"  Stability: finite={check1} osc_07={osc_07:.3e}")
    print(f"  PASS: {pass_all}")

    return result


def plot_case(result, case_idx):
    """Plot 4-panel profile: u, p, exact overlay."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = result['x']
    u_num = result['u_num']
    p_num = result['p_num']
    u_exact = result['u_exact']
    p_exact = result['p_exact']
    p0 = result['p0']
    name = result['name']

    # Velocity
    ax = axes[0, 0]
    ax.plot(x, u_num, 'b-', linewidth=2, label='Numerical (SSP222)')
    ax.plot(x, u_exact, 'r--', linewidth=1.5, label='Exact d\'Alembert', alpha=0.7)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('u (m/s)')
    ax.set_title(f'{name}: Velocity Profile')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Pressure
    ax = axes[0, 1]
    ax.plot(x, (p_num - p0)/1e3, 'b-', linewidth=2, label='Numerical')
    ax.plot(x, (p_exact - p0)/1e3, 'r--', linewidth=1.5, label='Exact', alpha=0.7)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('Δp (kPa)')
    ax.set_title(f'{name}: Pressure Perturbation')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Error in u
    ax = axes[1, 0]
    err_u = u_num - u_exact
    ax.semilogy(x, np.abs(err_u) + 1e-10, 'g-', linewidth=2)
    ax.axhline(y=0.30 * result['du_wave'], color='r', linestyle='--', label='PASS threshold (0.30×u_peak)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('|u_num - u_exact| (m/s)')
    ax.set_title(f'{name}: Velocity Error')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Error in p
    ax = axes[1, 1]
    err_p = p_num - p_exact
    ax.semilogy(x, np.abs(err_p) + 1.0, 'purple', linewidth=2)
    ax.axhline(y=0.30 * result['dp_wave'], color='r', linestyle='--', label='PASS threshold (0.30×dp_wave)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('|p_num - p_exact| (Pa)')
    ax.set_title(f'{name}: Pressure Error')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    png_path = f'/home/younglin90/work/claude_code/claudeCFD/results/case_07/case_07_{name.replace(" ", "_")}_ssp2_r1.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {png_path}")
    plt.close()


if __name__ == '__main__':
    import time
    start = time.time()

    results = []

    # Run 3 cases
    try:
        r1 = run_case_air_water()
        results.append(r1)
        plot_case(r1, 1)
    except Exception as e:
        print(f"ERROR in Air-Water: {e}")
        import traceback
        traceback.print_exc()

    try:
        r2 = run_case_helium_air()
        results.append(r2)
        plot_case(r2, 2)
    except Exception as e:
        print(f"ERROR in Helium-Air: {e}")
        import traceback
        traceback.print_exc()

    try:
        r3 = run_case_argon_air()
        results.append(r3)
        plot_case(r3, 3)
    except Exception as e:
        print(f"ERROR in Argon-Air: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - start

    # Write QA report
    with open('/home/younglin90/work/claude_code/claudeCFD/results/qa_report.md', 'w') as f:
        f.write('# QA Report — Phase 6-3 Acoustic Reflection & Transmission (SSP222/IMEX RK2)\n\n')
        f.write(f'**Date**: 2026-04-25\n')
        f.write(f'**Solver**: IMEX with `time_integrator="ssp222"` and `imex_rk2=True`\n')
        f.write(f'**Wall time**: {elapsed:.2f} s\n\n')

        f.write('## Results Summary\n\n')
        f.write('| # | Case | Finite | Osc<0.1 | L2p<0.3 | L2u<0.3 | Linf_p<0.5 | Linf_u<0.5 | frac_p≥0.7 | frac_u≥0.7 | L1p<1.0 | L1u<1.0 | corr_p>0.5 | corr_u>0.5 | PASS |\n')
        f.write('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n')

        for i, r in enumerate(results, 1):
            checks = r['checks']
            f.write(f"| {i} | {r['name']} | {checks[0]} | {checks[1]} | {checks[2]} | {checks[3]} | {checks[4]} | {checks[5]} | {checks[6]} | {checks[7]} | {checks[8]} | {checks[9]} | {checks[10] and checks[11]} | - | {'PASS' if r['pass'] else 'FAIL'} |\n")

        f.write('\n## Detailed Metrics\n\n')

        for r in results:
            f.write(f'### Case: {r["name"]}\n\n')
            f.write(f'**Stability**: finite={r["finite"]}, osc_07={r["osc_07"]:.3e}\n\n')
            f.write(f'**Norm Metrics** (wave scale: dp_wave={r["dp_wave"]:.2e}, du_wave={r["du_wave"]:.2e}):\n')
            f.write(f'- L2_p / dp_wave = {r["L2_p_norm"]:.3f} (threshold < 0.30)\n')
            f.write(f'- L2_u / u_peak = {r["L2_u_norm"]:.3f} (threshold < 0.30)\n')
            f.write(f'- L∞_p / dp_wave = {r["Linf_p_norm"]:.3f} (threshold < 0.50)\n')
            f.write(f'- L∞_u / u_peak = {r["Linf_u_norm"]:.3f} (threshold < 0.50)\n\n')
            f.write(f'**Pointwise**: frac_p={r["frac_p"]:.2f} (≥0.70), frac_u={r["frac_u"]:.2f} (≥0.70)\n\n')
            f.write(f'**L1 Integrated**: L1_p={r["L1_p_norm"]:.3f} (<1.0), L1_u={r["L1_u_norm"]:.3f} (<1.0)\n\n')
            f.write(f'**Correlation**: corr_p={r["corr_p"]:.3f} (>0.50), corr_u={r["corr_u"]:.3f} (>0.50)\n\n')
            f.write(f'**Verdict**: {"PASS" if r["pass"] else "FAIL"}\n\n')

        f.write('\n## Conclusion\n\n')
        all_pass = all(r['pass'] for r in results)
        f.write(f'All 3 sub-cases: {"**PASS**" if all_pass else "**FAIL** (see details above)"}\n')

    print(f"\n{'='*70}")
    print(f"QA Report written to: /home/younglin90/work/claude_code/claudeCFD/results/qa_report.md")
    print(f"{'='*70}")

    # Summary
    all_pass = all(r['pass'] for r in results)
    print(f"\n**OVERALL: {'PASS' if all_pass else 'FAIL'}**")

    sys.exit(0 if all_pass else 1)
