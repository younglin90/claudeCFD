#!/usr/bin/env python3
"""
Round 2 Validation: Case 07 (Acoustic Reflection/Transmission) with SSP2(2,2,2)
Testing 3 sub-cases: Air-Water, Helium-Air, Argon-Air
Spec: validation/1D/07_B_acoustic_reflection_transmission.md

Using pipeline API pattern (ph1/ph2 dicts).
"""

import sys
import os
sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import time

from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim

# ======================== EOS Defs (as dicts) ========================
# Spec parameters (all SG EOS form)

eos_air = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
eos_helium = {'gamma': 1.667, 'pinf': 0.0, 'kv': 2077.0, 'b': 0.0, 'eta': 0.0, 'q': 0.0}  # He: kv = R/(γ-1)
eos_argon = {'gamma': 1.66, 'pinf': 0.0, 'kv': 208.3, 'b': 0.0, 'eta': 0.0, 'q': 0.0}  # Argon: SG form
eos_water_sg = {'gamma': 4.1, 'pinf': 4.4e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}

# Spec reference densities at p0=1e5, T0=300K
rho_ref = {
    'Air': 1.157,
    'Helium': 0.164,
    'Argon': 1.748,
    'Water_SG': 998,
}

c_ref = {
    'Air': 347.8,
    'Helium': 1008.2,
    'Argon': 308.2,
    'Water_SG': 1344.6,
}

Z_ref = {
    'Air': 402.4,
    'Helium': 165.3,
    'Argon': 538.7,
    'Water_SG': 1.342e6,
}

# ======================== Cases ========================

cases_config = [
    {
        'name': 'Air-Water',
        'left': ('Air', eos_air),
        'right': ('Water_SG', eos_water_sg),
        'x_intf': 0.5,
        't_end': 1.63e-3,
        'sigma_L': 0.014,
        'x_src': 0.1,
        'u_peak': 0.02,
    },
    {
        'name': 'Helium-Air',
        'left': ('Helium', eos_helium),
        'right': ('Air', eos_air),
        'x_intf': 1.0,
        't_end': 1.513e-3,
        'sigma_L': 0.049,
        'x_src': 0.2,
        'u_peak': 0.02,
    },
    {
        'name': 'Argon-Air',
        'left': ('Argon', eos_argon),
        'right': ('Air', eos_air),
        'x_intf': 0.5,
        't_end': 2.02e-3,
        'sigma_L': 0.038,
        'x_src': 0.1,
        'u_peak': 0.02,
    },
]

# ======================== Exact Solution ========================

def exact_acoustic_solution(x, t, x_intf, x_src, sigma_L, u_peak, Z_L, Z_R, c_L, c_R):
    """Exact d'Alembert solution with reflection/transmission."""
    R = (Z_R - Z_L) / (Z_R + Z_L)
    T = 2.0 * Z_R / (Z_R + Z_L)

    u_exact = np.zeros_like(x, dtype=float)
    p_exact = np.zeros_like(x, dtype=float)
    p0 = 1e5

    # Time to reach interface
    t_interface = (x_intf - x_src) / c_L

    # Incident pulse (left-traveling, originating from x_src)
    if t >= 0:
        x_incid_peak = x_src + c_L * t
        u_incid = u_peak * np.exp(-0.5 * ((x - x_incid_peak) / sigma_L) ** 2)
        p_incid = p0 + Z_L * u_incid

        # Transmitted pulse (right-traveling from interface)
        if t >= t_interface:
            sigma_R = sigma_L * (c_R / c_L)
            x_trans_peak = x_intf + c_R * (t - t_interface)
            u_trans = T * u_peak * np.exp(-0.5 * ((x - x_trans_peak) / sigma_R) ** 2)
            mask_right = x >= x_intf
            u_exact[mask_right] += u_trans[mask_right]
            p_exact[mask_right] += p0 + Z_R * u_trans[mask_right]

        # Reflected pulse (left-traveling from interface)
        if t >= t_interface:
            x_refl_peak = x_intf - c_L * (t - t_interface)
            u_refl = R * u_peak * np.exp(-0.5 * ((x - x_refl_peak) / sigma_L) ** 2)
            mask_left = x < x_intf
            u_exact[mask_left] += u_refl[mask_left]
            p_exact[mask_left] += p0 + Z_L * u_refl[mask_left]

        # Add incident pulse to left side
        mask_left = x < x_intf
        u_exact[mask_left] += u_incid[mask_left]
        p_exact[mask_left] += p_incid[mask_left]

    return u_exact, p_exact

# ======================== Metrics ========================

def compute_metrics(u_num, p_num, u_exact, p_exact, u_peak, Z_L, x, dx):
    """Compute all 11 PASS criteria."""
    p0 = 1e5
    dp_wave = Z_L * u_peak

    # (A) Norm
    L2_p = np.sqrt(np.sum((p_num - p_exact)**2 * dx))
    L_inf_p = np.max(np.abs(p_num - p_exact))
    L2_u = np.sqrt(np.sum((u_num - u_exact)**2 * dx))
    L_inf_u = np.max(np.abs(u_num - u_exact))

    L2_p_norm = L2_p / dp_wave if dp_wave > 0 else 1e10
    L_inf_p_norm = L_inf_p / dp_wave if dp_wave > 0 else 1e10
    L2_u_norm = L2_u / u_peak if u_peak > 0 else 1e10
    L_inf_u_norm = L_inf_u / u_peak if u_peak > 0 else 1e10

    # (B) Pointwise
    frac_p = np.mean(np.abs(p_num - p_exact) < 0.30 * dp_wave) if dp_wave > 0 else 0.0
    frac_u = np.mean(np.abs(u_num - u_exact) < 0.30 * u_peak) if u_peak > 0 else 0.0

    # (C) L1 integrated
    L1_p = np.sum(np.abs(p_num - p_exact) * dx)
    L1_u = np.sum(np.abs(u_num - u_exact) * dx)
    p_exact_dev = np.abs(p_exact - p0)
    u_exact_dev = np.abs(u_exact)
    L1_p_norm_denom = np.sum(p_exact_dev * dx)
    L1_u_norm_denom = np.sum(u_exact_dev * dx)
    L1_p_norm = L1_p / L1_p_norm_denom if L1_p_norm_denom > 1e-12 else 1e10
    L1_u_norm = L1_u / L1_u_norm_denom if L1_u_norm_denom > 1e-12 else 1e10

    # (D) Correlation
    p_num_dev = p_num - p0
    p_exact_dev = p_exact - p0
    if np.std(p_num_dev) > 1e-12 and np.std(p_exact_dev) > 1e-12:
        corr_p, _ = pearsonr(p_num_dev, p_exact_dev)
    else:
        corr_p = -2.0  # Mark as invalid
    if np.std(u_num) > 1e-12 and np.std(u_exact) > 1e-12:
        corr_u, _ = pearsonr(u_num, u_exact)
    else:
        corr_u = -2.0

    # Stability
    finite = np.all(np.isfinite(u_num)) and np.all(np.isfinite(p_num))
    osc_07 = np.abs(np.mean((p_num[::2] - p_num[1::2]) / (np.abs(p_num[::2]) + 1e-10)))

    # PASS criteria (all 11)
    criteria = {
        '1_finite': finite,
        '2_osc_07': osc_07 < 0.1,
        '3_L2_p': L2_p_norm < 0.30,
        '4_L2_u': L2_u_norm < 0.30,
        '5_Linf_p': L_inf_p_norm < 0.50,
        '6_Linf_u': L_inf_u_norm < 0.50,
        '7_frac_p': frac_p >= 0.70,
        '8_frac_u': frac_u >= 0.70,
        '9_L1_p': L1_p_norm < 1.0,
        '10_L1_u': L1_u_norm < 1.0,
        '11_corr': (corr_p > 0.50) and (corr_u > 0.50),
    }

    pass_bool = all(criteria.values())

    return {
        'L2_p_norm': L2_p_norm,
        'L2_u_norm': L2_u_norm,
        'L_inf_p_norm': L_inf_p_norm,
        'L_inf_u_norm': L_inf_u_norm,
        'frac_p': frac_p,
        'frac_u': frac_u,
        'L1_p_norm': L1_p_norm,
        'L1_u_norm': L1_u_norm,
        'corr_p': corr_p,
        'corr_u': corr_u,
        'finite': finite,
        'osc_07': osc_07,
        'criteria': criteria,
        'PASS': pass_bool,
    }

# ======================== Main ========================

def run_case_07_sub(config, N=400):
    """Run single Case 07 sub-case."""
    print(f"\n{'='*70}")
    print(f"Case 07: {config['name']}")
    print(f"{'='*70}")

    left_name, eos_L = config['left']
    right_name, eos_R = config['right']

    # Reference properties
    rho_L = rho_ref[left_name]
    rho_R = rho_ref[right_name]
    c_L = c_ref[left_name]
    c_R = c_ref[right_name]
    Z_L = Z_ref[left_name]
    Z_R = Z_ref[right_name]

    print(f"  Left: {left_name} (ρ={rho_L:.1f}, c={c_L:.1f}, Z={Z_L:.1f})")
    print(f"  Right: {right_name} (ρ={rho_R:.1f}, c={c_R:.1f}, Z={Z_R:.2e})")

    # Grid
    L_domain = 1.5
    dx = L_domain / N
    x = np.linspace(dx/2, L_domain - dx/2, N)

    # Initial state
    p0 = 1e5
    T0 = 300.0
    x_intf = config['x_intf']
    x_src = config['x_src']
    sigma_L = config['sigma_L']
    u_peak = config['u_peak']

    # Volume fractions
    a_left = np.where(x < x_intf, 1.0, 1e-6)
    a_right = 1.0 - a_left

    # Densities (from EOS)
    rho_L_array = np.full_like(x, rho_L)
    rho_R_array = np.full_like(x, rho_R)

    # Conservative variables
    a_left_rho_L = a_left * rho_L_array
    a_right_rho_R = a_right * rho_R_array

    # Initial velocity pulse
    u_init = u_peak * np.exp(-0.5 * ((x - x_src) / sigma_L) ** 2)
    u_init[x >= x_intf] = 0.0

    # Initial density from pressure (left/right)
    rho_init = np.where(x < x_intf, rho_L_array, rho_R_array)
    ru_init = rho_init * u_init

    # Initial energy
    gm1_L = eos_L['gamma'] - 1.0
    gm1_R = eos_R['gamma'] - 1.0
    p_init = np.full_like(x, p0)
    rE_init = (a_left * (p_init + eos_L['gamma']*eos_L['pinf']) / gm1_L +
               a_right * (p_init + eos_R['gamma']*eos_R['pinf']) / gm1_R +
               0.5 * rho_init * u_init**2)

    print(f"  Grid: N={N}, Δx={dx:.2e} m, L={L_domain} m")
    print(f"  t_end={config['t_end']*1e3:.3f} ms")
    print(f"  Running solve_IMEX with time_integrator='ssp222'...")

    t0 = time.time()
    try:
        t, a_left_f, a_right_f, ru_f, rE_f, a_intf_f = solve_IMEX(
            eos_L, eos_R,
            a_left_rho_L, a_right_rho_R, ru_init, rE_init, a_left,
            dx, t_end=config['t_end'],
            cfl=0.4, bc_l='transmissive', bc_r='transmissive',
            max_steps=5000, print_interval=100,
            alpha_scheme='tvd',
            use_strang=False,  # Use time_integrator parameter instead
            use_defect_correction=False,
            use_material_cfl=False,
            time_integrator='ssp222',  # Round 1 fix: use ssp222
            acoustic_method='imex_5n',
        )

        wall_time = time.time() - t0

        # Reconstruct primitive variables
        p_f, u_f, T_f, Y1_f, Y2_f, a1_f, a2_f, rho_f = cons_to_prim(
            a_left_f, a_right_f, ru_f, rE_f, a_intf_f, eos_L, eos_R
        )

        print(f"  Completed in {wall_time:.2f}s, t_final={t:.2e}s")

        # Exact solution
        u_exact, p_exact = exact_acoustic_solution(
            x, config['t_end'], x_intf, x_src, sigma_L, u_peak, Z_L, Z_R, c_L, c_R
        )

        # Metrics
        metrics = compute_metrics(u_f, p_f, u_exact, p_exact, u_peak, Z_L, x, dx)

        print(f"\n  Metrics Summary:")
        for crit_name, crit_val in metrics['criteria'].items():
            status = "✓" if crit_val else "✗"
            print(f"    {crit_name}: {status}")

        print(f"\n  Values:")
        print(f"    L2_p/A = {metrics['L2_p_norm']:.4f} (< 0.30)")
        print(f"    L2_u/A = {metrics['L2_u_norm']:.4f} (< 0.30)")
        print(f"    L∞_p/A = {metrics['L_inf_p_norm']:.4f} (< 0.50)")
        print(f"    L∞_u/A = {metrics['L_inf_u_norm']:.4f} (< 0.50)")
        print(f"    frac_p = {metrics['frac_p']:.3f} (>= 0.70)")
        print(f"    frac_u = {metrics['frac_u']:.3f} (>= 0.70)")
        print(f"    corr_p = {metrics['corr_p']:.3f} (> 0.50)")
        print(f"    corr_u = {metrics['corr_u']:.3f} (> 0.50)")
        print(f"\n  RESULT: {'PASS ✓' if metrics['PASS'] else 'FAIL ✗'}")

        return metrics, u_f, p_f, rho_f, x, wall_time, u_exact, p_exact

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None, None, None

# ======================== Plotting ========================

def plot_result(case_name, u, p, rho, x, u_exact, p_exact, metrics):
    """4-panel plot."""
    os.makedirs('results/case_07', exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # u
    ax = axes[0, 0]
    ax.plot(x, u, 'b-', linewidth=1.5, label='Numerical (ssp222)')
    if u_exact is not None:
        ax.plot(x, u_exact, 'r--', linewidth=1.0, label='Exact (d\'Alembert)')
    ax.set_xlabel('x (m)'); ax.set_ylabel('u (m/s)')
    ax.set_title(f'{case_name}: Velocity')
    ax.grid(True, alpha=0.3); ax.legend()

    # p
    ax = axes[0, 1]
    ax.plot(x, p, 'b-', linewidth=1.5, label='Numerical (ssp222)')
    if p_exact is not None:
        ax.plot(x, p_exact, 'r--', linewidth=1.0, label='Exact (d\'Alembert)')
    ax.set_xlabel('x (m)'); ax.set_ylabel('p (Pa)')
    ax.set_title(f'{case_name}: Pressure')
    ax.grid(True, alpha=0.3); ax.legend()

    # rho
    ax = axes[1, 0]
    ax.plot(x, rho, 'b-', linewidth=1.5)
    ax.set_xlabel('x (m)'); ax.set_ylabel('ρ (kg/m³)')
    ax.set_title(f'{case_name}: Density')
    ax.grid(True, alpha=0.3)

    # Metrics
    ax = axes[1, 1]
    ax.axis('off')
    txt = f"SSP2(2,2,2) Metrics:\n"
    txt += f"L2_p/A={metrics['L2_p_norm']:.4f}\n"
    txt += f"L2_u/A={metrics['L2_u_norm']:.4f}\n"
    txt += f"L∞_p/A={metrics['L_inf_p_norm']:.4f}\n"
    txt += f"L∞_u/A={metrics['L_inf_u_norm']:.4f}\n"
    txt += f"frac_p={metrics['frac_p']:.3f}\n"
    txt += f"frac_u={metrics['frac_u']:.3f}\n"
    txt += f"corr_p={metrics['corr_p']:.3f}\n"
    txt += f"corr_u={metrics['corr_u']:.3f}\n"
    txt += f"\nRESULT: {'PASS ✓' if metrics['PASS'] else 'FAIL ✗'}"
    ax.text(0.1, 0.5, txt, fontfamily='monospace', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fname = f"results/case_07/case_07_{case_name.replace(' ', '_').lower()}_ssp2_r2.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    print(f"    Plot saved: {fname}")
    plt.close()

# ======================== Regression ========================

def regression_phase1():
    """Phase 1 quick check (02-A)."""
    print(f"\n{'='*70}\nREGRESSION: Phase 1 (case 02-A, default strang)\n{'='*70}")

    ph1 = {'gamma': 7.15, 'pinf': 3.1e8, 'kv': 0.0, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}

    N = 10; L = 1.0; dx = L/N; x = np.linspace(dx/2, L-dx/2, N)
    x_intf = 0.5
    u0 = np.ones(N) * 1.0
    p0 = np.ones(N) * 1e5

    a1 = np.where(x < x_intf, 1.0, 0.0)
    a2 = 1.0 - a1
    a1r1 = a1 * 1000.0
    a2r2 = a2 * 1.157
    ru = (a1*1000 + a2*1.157) * u0
    rE = (a1*(p0 + 7.15*3.1e8)/6.15 + a2*p0/0.4 + 0.5*(a1*1000+a2*1.157)*u0**2)

    t0 = time.time()
    try:
        t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
            ph1, ph2, a1r1, a2r2, ru, rE, a1,
            dx, t_end=0.1, max_steps=100,
            time_integrator='strang',  # Default (no ssp222)
            acoustic_method='imex_5n',
        )
        wall_time = time.time() - t0

        p_f, u_f, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
        p_dev = np.max(np.abs(p_f - p0))
        u_dev = np.max(np.abs(u_f - u0))

        print(f"  wall={wall_time:.1f}s, err_p={p_dev:.2e}, err_u={u_dev:.2e}")
        passed = (p_dev < 1e-8) and (u_dev < 1e-8)
        print(f"  {'PASS ✓' if passed else 'FAIL ✗'}")
        return passed
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def regression_phase22():
    """Phase 2-2 quick check."""
    print(f"\n{'='*70}\nREGRESSION: Phase 2-2 (HP Water/LP Air, default strang)\n{'='*70}")

    ph1 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}

    N = 100; L = 1.0; dx = L/N; x = np.linspace(dx/2, L-dx/2, N)
    x_intf = 0.7
    u0 = np.zeros(N)
    p0_arr = np.where(x < x_intf, 1e9, 1e5)

    a1 = np.where(x < x_intf, 1.0, 1e-6)
    a2 = 1.0 - a1
    a1r1 = a1 * 1000.0
    a2r2 = a2 * 50.0
    ru = np.zeros(N)
    rE = (a1*(p0_arr + 4.4*6e8)/3.4 + a2*p0_arr/0.4)

    t0 = time.time()
    try:
        t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
            ph1, ph2, a1r1, a2r2, ru, rE, a1,
            dx, t_end=2.29e-4, max_steps=5000,
            time_integrator='strang',
            acoustic_method='imex_5n',
        )
        wall_time = time.time() - t0

        p_f, u_f, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
        u_max = np.max(np.abs(u_f))

        print(f"  wall={wall_time:.1f}s, u_max={u_max:.1f} (ref ~486)")
        passed = (400 < u_max < 600)
        print(f"  {'PASS ✓' if passed else 'FAIL ✗'}")
        return passed
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

# ======================== Main ========================

if __name__ == '__main__':
    os.makedirs('results/case_07', exist_ok=True)

    results = []
    for cfg in cases_config:
        metrics, u, p, rho, x, wall, u_ex, p_ex = run_case_07_sub(cfg, N=400)
        if metrics is not None:
            results.append({'case': cfg['name'], 'metrics': metrics, 'wall': wall, 'PASS': metrics['PASS']})
            plot_result(cfg['name'], u, p, rho, x, u_ex, p_ex, metrics)
        else:
            results.append({'case': cfg['name'], 'metrics': None, 'wall': None, 'PASS': False})

    # Regressions
    reg1 = regression_phase1()
    reg22 = regression_phase22()

    # Summary
    print(f"\n{'='*70}\nROUND 2 SUMMARY\n{'='*70}")
    print("\nCase 07 Sub-cases:")
    for r in results:
        status = "PASS ✓" if r['PASS'] else "FAIL ✗"
        wall_str = f"{r['wall']:.1f}s" if r['wall'] is not None else "ERROR"
        print(f"  {r['case']:15s}: {status} ({wall_str})")

    print(f"\nRegressions:")
    print(f"  Phase 1: {'PASS ✓' if reg1 else 'FAIL ✗'}")
    print(f"  Phase 2-2: {'PASS ✓' if reg22 else 'FAIL ✗'}")

    all_pass = all(r['PASS'] for r in results) and reg1 and reg22
    print(f"\n{'='*70}")
    if all_pass:
        print("OVERALL RESULT: ALL PASS ✓")
        with open('results/all_pass.flag', 'w') as f:
            f.write("Round 2 validation ALL PASS\n")
    else:
        print("OVERALL RESULT: SOME FAIL ✗")
    print(f"{'='*70}\n")
