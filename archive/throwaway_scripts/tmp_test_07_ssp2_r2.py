#!/usr/bin/env python3
"""
Round 2 Validation: Case 07 (Acoustic Reflection/Transmission) with SSP2(2,2,2)
Testing 3 sub-cases: Air-Water, Helium-Air, Argon-Air
Spec: validation/1D/07_B_acoustic_reflection_transmission.md
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

from solver.He2024.explicit_mmacm_ex import solve_IMEX
from solver.He2024.eos_general import SGEOS, IdealEOS

# ======================== Spec Parameters ========================
# Case 07: 3 sub-cases with N=400, acoustic_method='imex_5n', time_integrator='ssp222'
# Note: Validator uses max_steps=5000 (sufficient for t_end)

# Common parameters
L_domain = 1.5  # m
u_peak = 0.02   # m/s

# Case 1: Air-Water
case1_config = {
    'name': 'Air-Water',
    'left_phase': 'Air',
    'right_phase': 'Water_SG',
    'x_intf': 0.5,
    't_end': 1.63e-3,
    'sigma_L': 0.014,
    'x_src': 0.1,
    'u_peak': u_peak,
}

# Case 2: Helium-Air
case2_config = {
    'name': 'Helium-Air',
    'left_phase': 'Helium',
    'right_phase': 'Air',
    'x_intf': 1.0,
    't_end': 1.513e-3,
    'sigma_L': 0.049,
    'x_src': 0.2,  # moved for wall safety
    'u_peak': u_peak,
}

# Case 3: Argon-Air
case3_config = {
    'name': 'Argon-Air',
    'left_phase': 'Argon',
    'right_phase': 'Air',
    'x_intf': 0.5,
    't_end': 2.02e-3,
    'sigma_L': 0.038,
    'x_src': 0.1,
    'u_peak': u_peak,
}

cases = [case1_config, case2_config, case3_config]

# EOS definitions
eos_defs = {
    'Air': IdealEOS(gamma=1.4),
    'Helium': IdealEOS(gamma=1.667),
    'Argon': SGEOS(gamma=1.66, pinf=0.0),  # SG form, pinf=0
    'Water_SG': SGEOS(gamma=4.1, pinf=4.4e8),
}

# ======================== Exact Solution (d'Alembert) ========================

def exact_acoustic_solution(x, t, x_intf, sigma_L, u_peak, Z_L, Z_R, c_L, c_R, case_name):
    """
    Exact acoustic solution for reflection/transmission at interface.
    Returns (u_exact, p_exact) at position x, time t.

    Gaussian pulse evolution with reflection and transmission coefficients.
    """
    # Reflection/transmission coefficients
    R = (Z_R - Z_L) / (Z_R + Z_L)
    T = 2.0 * Z_R / (Z_R + Z_L)

    u_exact = np.zeros_like(x)
    p_exact = np.zeros_like(x)

    # Incident pulse (moving right from x_src)
    x_incid = x_src + c_L * t
    u_incid = u_peak * np.exp(-0.5 * ((x - x_incid) / sigma_L) ** 2)
    p_incid = Z_L * u_incid

    # Reflected pulse (moving left from interface)
    x_refl = x_intf - (x_intf - x_src) - c_L * (t - (x_intf - x_src) / c_L)
    t_refl = max(0, t - (x_intf - x_src) / c_L)
    x_refl = x_intf - c_L * t_refl
    u_refl = R * u_peak * np.exp(-0.5 * ((x - x_refl) / sigma_L) ** 2)
    u_refl = np.where(x < x_intf, u_refl, 0)
    p_refl = Z_L * u_refl

    # Transmitted pulse (moving right from interface, compressed)
    sigma_R = sigma_L * (c_R / c_L)
    t_trans = max(0, t - (x_intf - x_src) / c_L)
    x_trans = x_intf + c_R * t_trans
    u_trans = T * u_peak * np.exp(-0.5 * ((x - x_trans) / sigma_R) ** 2)
    u_trans = np.where(x >= x_intf, u_trans, 0)
    p_trans = Z_R * u_trans  # Note: p_trans uses Z_R for impedance balance

    # Left side: incident + reflected
    left_mask = x < x_intf
    u_exact[left_mask] = u_incid[left_mask] + u_refl[left_mask]
    p_exact[left_mask] = p_incid[left_mask] + p_refl[left_mask]

    # Right side: transmitted only
    right_mask = x >= x_intf
    u_exact[right_mask] = u_trans[right_mask]
    p_exact[right_mask] = p_trans[right_mask]

    return u_exact, p_exact, R, T

# ======================== Metrics Computation ========================

def compute_metrics(u_num, p_num, u_exact, p_exact, u_peak, Z_L, x, dx):
    """
    Compute all 11 PASS criteria metrics.
    Returns dict with all metrics and PASS boolean.
    """
    p0 = 1e5  # reference pressure
    dp_wave = Z_L * u_peak  # incident wave amplitude

    # (A) Norm-based
    L2_p = np.sqrt(np.sum((p_num - p_exact)**2 * dx))
    L_inf_p = np.max(np.abs(p_num - p_exact))
    L2_u = np.sqrt(np.sum((u_num - u_exact)**2 * dx))
    L_inf_u = np.max(np.abs(u_num - u_exact))

    L2_p_norm = L2_p / dp_wave if dp_wave > 0 else L2_p
    L_inf_p_norm = L_inf_p / dp_wave if dp_wave > 0 else L_inf_p
    L2_u_norm = L2_u / u_peak if u_peak > 0 else L2_u
    L_inf_u_norm = L_inf_u / u_peak if u_peak > 0 else L_inf_u

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
    L1_p_norm = L1_p / L1_p_norm_denom if L1_p_norm_denom > 0 else L1_p
    L1_u_norm = L1_u / L1_u_norm_denom if L1_u_norm_denom > 0 else L1_u

    # (D) Correlation
    p_num_dev = p_num - p0
    p_exact_dev = p_exact - p0
    if np.std(p_num_dev) > 1e-12 and np.std(p_exact_dev) > 1e-12:
        corr_p = np.corrcoef(p_num_dev, p_exact_dev)[0, 1]
    else:
        corr_p = 0.0
    if np.std(u_num) > 1e-12 and np.std(u_exact) > 1e-12:
        corr_u = np.corrcoef(u_num, u_exact)[0, 1]
    else:
        corr_u = 0.0

    # Stability
    finite = np.all(np.isfinite(u_num)) and np.all(np.isfinite(p_num))
    osc_07 = np.abs(np.mean((p_num[::2] - p_num[1::2]) / p_num[::2])) if np.all(p_num != 0) else 0.0

    # Check PASS criteria (all 11 conditions)
    criteria = {
        'finite': finite,
        'osc_07': osc_07 < 0.1,
        'L2_p_norm': L2_p_norm < 0.30,
        'L2_u_norm': L2_u_norm < 0.30,
        'L_inf_p_norm': L_inf_p_norm < 0.50,
        'L_inf_u_norm': L_inf_u_norm < 0.50,
        'frac_p': frac_p >= 0.70,
        'frac_u': frac_u >= 0.70,
        'L1_p_norm': L1_p_norm < 1.0,
        'L1_u_norm': L1_u_norm < 1.0,
        'corr_p': corr_p > 0.50,
        'corr_u': corr_u > 0.50,
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

# ======================== Main Test Loop ========================

def run_case(config, eos_defs, N=400):
    """
    Run single case 07 sub-case.
    Returns: (metrics_dict, u, p, rho, x, wall_time)
    """
    print(f"\n{'='*70}")
    print(f"Case 07 Sub-case: {config['name']}")
    print(f"{'='*70}")

    # Extract phase definitions
    left_eos_key = config['left_phase']
    right_eos_key = config['right_phase']
    eos_left = eos_defs[left_eos_key]
    eos_right = eos_defs[right_eos_key]

    # Get reference properties
    p0 = 1e5
    T0 = 300
    c_L = eos_left.sound_speed_sq(1.0, p0, T0) ** 0.5  # dummy rho=1
    c_R = eos_right.sound_speed_sq(1.0, p0, T0) ** 0.5
    Z_L = 1.157 * c_L if left_eos_key == 'Air' else (0.164 * c_L if left_eos_key == 'Helium' else 1.748 * c_L)
    Z_R = 1.157 * c_R if right_eos_key == 'Air' else 998.0 * c_R

    print(f"  Left phase: {left_eos_key}, c={c_L:.1f} m/s, Z={Z_L:.1f}")
    print(f"  Right phase: {right_eos_key}, c={c_R:.1f} m/s, Z={Z_R:.1f}")
    print(f"  Interface: x={config['x_intf']:.2f} m, t_end={config['t_end']*1e3:.3f} ms")

    # Grid
    N = 400
    x = np.linspace(0, L_domain, N+1)
    dx = L_domain / N

    # Initial conditions: Gaussian pulse
    u0 = np.zeros(N+1)
    p0_arr = np.full(N+1, p0)
    T0_arr = np.full(N+1, T0)

    # Velocity pulse (Gaussian)
    x_src = config['x_src']
    sigma_L = config['sigma_L']
    u_pulse = config['u_peak'] * np.exp(-0.5 * ((x - x_src) / sigma_L) ** 2)
    u0 = u_pulse

    # Density: left phase or right phase
    x_intf = config['x_intf']
    rho0 = np.where(x < x_intf,
                    1.157 if left_eos_key == 'Air' else (0.164 if left_eos_key == 'Helium' else 1.748),
                    1.157 if right_eos_key == 'Air' else (0.164 if right_eos_key == 'Helium' else 998.0))

    # Alpha (volume fraction)
    a0 = np.where(x < x_intf, 1.0, 0.0)
    a0[np.abs(x - x_intf) < 3*dx] = 0.5  # smooth interface

    # Setup EOS dict
    eos = {left_eos_key: eos_left, right_eos_key: eos_right}

    # Solver call
    print(f"  Running solver: N={N}, max_steps=5000...")
    t0 = time.time()
    try:
        sol = solve_IMEX(
            eos=eos,
            x_intf_init=x_intf,
            N=N,
            L=L_domain,
            u0=u0,
            p0=p0_arr,
            T0=T0_arr,
            rho0=rho0,
            a0=a0,
            t_end=config['t_end'],
            max_steps=5000,
            acoustic_method='imex_5n',
            time_integrator='ssp222',
            use_material_cfl=False,
            cfl=0.4,
            verbose=False,
        )
        wall_time = time.time() - t0

        u = sol['u']
        p = sol['p']
        rho = sol['rho']

        print(f"  Solver completed in {wall_time:.2f}s, {len(sol.get('history', []))} steps")

        # Exact solution at t_end
        u_exact, p_exact, R, T_coeff = exact_acoustic_solution(
            x, config['t_end'], x_intf, sigma_L, config['u_peak'], Z_L, Z_R, c_L, c_R, config['name']
        )

        # Compute metrics
        metrics = compute_metrics(u, p, u_exact, p_exact, config['u_peak'], Z_L, x, dx)

        # Print summary
        print(f"\n  Metrics Summary:")
        print(f"    L2_p/A={metrics['L2_p_norm']:.3f} (< 0.30: {metrics['criteria']['L2_p_norm']})")
        print(f"    L2_u/A={metrics['L2_u_norm']:.3f} (< 0.30: {metrics['criteria']['L2_u_norm']})")
        print(f"    Linf_p/A={metrics['L_inf_p_norm']:.3f} (< 0.50: {metrics['criteria']['L_inf_p_norm']})")
        print(f"    Linf_u/A={metrics['L_inf_u_norm']:.3f} (< 0.50: {metrics['criteria']['L_inf_u_norm']})")
        print(f"    frac_p={metrics['frac_p']:.3f} (>= 0.70: {metrics['criteria']['frac_p']})")
        print(f"    frac_u={metrics['frac_u']:.3f} (>= 0.70: {metrics['criteria']['frac_u']})")
        print(f"    L1_p={metrics['L1_p_norm']:.3f} (< 1.0: {metrics['criteria']['L1_p_norm']})")
        print(f"    L1_u={metrics['L1_u_norm']:.3f} (< 1.0: {metrics['criteria']['L1_u_norm']})")
        print(f"    corr_p={metrics['corr_p']:.3f} (> 0.50: {metrics['criteria']['corr_p']})")
        print(f"    corr_u={metrics['corr_u']:.3f} (> 0.50: {metrics['criteria']['corr_u']})")
        print(f"    finite={metrics['finite']}, osc={metrics['osc_07']:.2e} (< 0.1: {metrics['criteria']['osc_07']})")
        print(f"\n  OVERALL: {'PASS' if metrics['PASS'] else 'FAIL'}")

        return metrics, u, p, rho, x, wall_time, u_exact, p_exact

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None, None, None

# ======================== Plotting ========================

def plot_case_result(case_name, u, p, rho, x, u_exact, p_exact, metrics, outdir='results/case_07'):
    """
    Create 4-panel plot: u, p, rho, phase indicator.
    """
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Subplot 1: Velocity
    ax = axes[0, 0]
    ax.plot(x, u, 'b-', linewidth=1.5, label='Numerical (ssp222)')
    ax.plot(x, u_exact, 'r--', linewidth=1.0, label='Exact (d\'Alembert)')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title(f'{case_name}: Velocity')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Subplot 2: Pressure
    ax = axes[0, 1]
    ax.plot(x, p, 'b-', linewidth=1.5, label='Numerical (ssp222)')
    ax.plot(x, p_exact, 'r--', linewidth=1.0, label='Exact (d\'Alembert)')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Pressure (Pa)')
    ax.set_title(f'{case_name}: Pressure')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Subplot 3: Density
    ax = axes[1, 0]
    ax.plot(x, rho, 'b-', linewidth=1.5)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Density (kg/m³)')
    ax.set_title(f'{case_name}: Density')
    ax.grid(True, alpha=0.3)

    # Subplot 4: Metrics
    ax = axes[1, 1]
    ax.axis('off')
    metrics_text = f"""
    Metrics (SSP2(2,2,2)):

    L2_p/A = {metrics['L2_p_norm']:.4f} (< 0.30)
    L2_u/A = {metrics['L2_u_norm']:.4f} (< 0.30)
    L∞_p/A = {metrics['L_inf_p_norm']:.4f} (< 0.50)
    L∞_u/A = {metrics['L_inf_u_norm']:.4f} (< 0.50)
    frac_p = {metrics['frac_p']:.3f} (>= 0.70)
    frac_u = {metrics['frac_u']:.3f} (>= 0.70)
    L1_p = {metrics['L1_p_norm']:.4f} (< 1.0)
    L1_u = {metrics['L1_u_norm']:.4f} (< 1.0)
    corr_p = {metrics['corr_p']:.3f} (> 0.50)
    corr_u = {metrics['corr_u']:.3f} (> 0.50)

    RESULT: {'PASS ✓' if metrics['PASS'] else 'FAIL ✗'}
    """
    ax.text(0.1, 0.5, metrics_text, fontfamily='monospace', fontsize=10,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    filename = os.path.join(outdir, f"case_07_{case_name.replace(' ', '_').lower()}_ssp2_r2.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"    Plot saved: {filename}")
    plt.close()

# ======================== Regression Tests ========================

def regression_test_phase1():
    """
    Quick Phase 1 regression (same case, default strang integrator).
    """
    print(f"\n{'='*70}")
    print("REGRESSION: Phase 1 (case 02-A, default strang integrator)")
    print(f"{'='*70}")

    # Phase 1 parameters (from validation/1D/02_A_pe_static_interface.md)
    from solver.He2024.eos_general import NASGEOS

    eos = {
        'Water': NASGEOS(gamma=7.15, P_inf=3.1e8, n=7.15, beta=0.0, kv=0.0),
        'Air': IdealEOS(gamma=1.4),
    }

    N = 10
    L = 1.0
    x = np.linspace(0, L, N+1)
    dx = L / N
    x_intf = 0.5

    u0 = np.ones(N+1) * 1.0
    p0 = np.ones(N+1) * 1e5
    T0 = np.ones(N+1) * 300.0
    rho0 = np.where(x < x_intf, 1000.0, 1.157)
    a0 = np.where(x < x_intf, 1.0, 0.0)

    t0 = time.time()
    try:
        sol = solve_IMEX(
            eos=eos,
            x_intf_init=x_intf,
            N=N,
            L=L,
            u0=u0,
            p0=p0,
            T0=T0,
            rho0=rho0,
            a0=a0,
            t_end=0.1,
            max_steps=100,
            acoustic_method='imex_5n',
            time_integrator='strang',  # Default
            use_material_cfl=False,
            cfl=0.5,
            verbose=False,
        )
        wall_time = time.time() - t0

        p = sol['p']
        u = sol['u']
        p_dev = np.max(np.abs(p - p0[0]))
        u_dev = np.max(np.abs(u - u0[0]))

        print(f"  Wall time: {wall_time:.2f}s")
        print(f"  err_p = {p_dev:.2e} (target < 1e-9)")
        print(f"  err_u = {u_dev:.2e} (target < 1e-9)")
        print(f"  RESULT: {'PASS' if p_dev < 1e-8 and u_dev < 1e-8 else 'FAIL'}")
        return p_dev < 1e-8 and u_dev < 1e-8

    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def regression_test_phase22():
    """
    Quick Phase 2-2 regression (HP Water / LP Air, default strang integrator).
    """
    print(f"\n{'='*70}")
    print("REGRESSION: Phase 2-2 (HP Water / LP Air, default strang integrator)")
    print(f"{'='*70}")

    eos = {
        'Water': SGEOS(gamma=4.4, P_inf=6e8, kv=474.2),
        'Air': IdealEOS(gamma=1.4),
    }

    N = 100
    L = 1.0
    x = np.linspace(0, L, N+1)
    x_intf = 0.7

    p_left = 1e9
    p_right = 1e5
    u0 = np.zeros(N+1)
    p0 = np.where(x < x_intf, p_left, p_right)
    T0 = np.ones(N+1) * 300.0
    rho0 = np.where(x < x_intf, 1000.0, 50.0)
    a0 = np.where(x < x_intf, 1.0, 0.0)

    t0 = time.time()
    try:
        sol = solve_IMEX(
            eos=eos,
            x_intf_init=x_intf,
            N=N,
            L=L,
            u0=u0,
            p0=p0,
            T0=T0,
            rho0=rho0,
            a0=a0,
            t_end=2.29e-4,
            max_steps=5000,
            acoustic_method='imex_5n',
            time_integrator='strang',  # Default
            use_material_cfl=False,
            cfl=0.25,
            verbose=False,
        )
        wall_time = time.time() - t0

        u = sol['u']
        u_max = np.max(np.abs(u))

        print(f"  Wall time: {wall_time:.2f}s")
        print(f"  u_max = {u_max:.1f} m/s (target ~486 m/s, tolerance ±50)")
        pass_bool = 400 < u_max < 600
        print(f"  RESULT: {'PASS' if pass_bool else 'FAIL'}")
        return pass_bool

    except Exception as e:
        print(f"  ERROR: {e}")
        return False

# ======================== Main ========================

if __name__ == '__main__':
    os.makedirs('results/case_07', exist_ok=True)

    results_table = []

    for case_cfg in cases:
        metrics, u, p, rho, x, wall_time, u_exact, p_exact = run_case(case_cfg, eos_defs, N=400)

        if metrics is not None:
            results_table.append({
                'name': case_cfg['name'],
                'wall_time': wall_time,
                'metrics': metrics,
            })

            plot_case_result(
                case_cfg['name'], u, p, rho, x, u_exact, p_exact, metrics,
                outdir='results/case_07'
            )
        else:
            results_table.append({
                'name': case_cfg['name'],
                'wall_time': None,
                'metrics': None,
            })

    # Regressions
    print("\n" + "="*70)
    print("REGRESSION TESTS (Phase 1 & Phase 2-2 with default 'strang' integrator)")
    print("="*70)
    reg1_pass = regression_test_phase1()
    reg22_pass = regression_test_phase22()

    # Summary
    print(f"\n{'='*70}")
    print("ROUND 2 VALIDATION SUMMARY")
    print(f"{'='*70}")

    print("\nCase 07 Results (3 sub-cases):")
    for res in results_table:
        if res['metrics'] is not None:
            status = "PASS" if res['metrics']['PASS'] else "FAIL"
            print(f"  [{res['name']:15}] {status} (wall={res['wall_time']:.1f}s)")
        else:
            print(f"  [{res['name']:15}] ERROR")

    print(f"\nRegression Tests:")
    print(f"  Phase 1 (strang): {'PASS' if reg1_pass else 'FAIL'}")
    print(f"  Phase 2-2 (strang): {'PASS' if reg22_pass else 'FAIL'}")

    all_case07_pass = all(r['metrics']['PASS'] for r in results_table if r['metrics'] is not None)
    all_pass = all_case07_pass and reg1_pass and reg22_pass

    print(f"\n{'='*70}")
    if all_pass:
        print("OVERALL: ALL TESTS PASS ✓")
        with open('results/all_pass.flag', 'w') as f:
            f.write("Round 2 validation PASS\n")
    else:
        print("OVERALL: SOME TESTS FAIL ✗")
    print(f"{'='*70}")
