#!/usr/bin/env python3
"""
Temporary Test A validation with 5 settings (material_cfl mode).

Test A: NASG Phase 1 Abgrall water-air advection
- Domain: [0, 1] m, periodic BC
- N: 10 cells
- u0=1.0 m/s, p0=1e5 Pa, T0=300 K
- water (NASG): x ∈ [0.4, 0.6], α_w=1
- air (Ideal): x ∉ [0.4, 0.6], α_w=0
- dt=0.01 s (fixed), max_iteration=100 (t_end=1.0 s)
- PASS: err_p < 1e-2, err_u < 1e-2

Settings:
1. cfl=0.1, use_material_cfl=True, alpha_scheme='cicsam', use_acid_face=True, acid_interface=True, iterative_im1=True
2. cfl=0.2, use_material_cfl=True, alpha_scheme='cicsam', use_acid_face=True, acid_interface=True, iterative_im1=True
3. cfl=0.1, use_material_cfl=True, alpha_scheme='cicsam', use_acid_face=True, acid_interface=True, iterative_im1=False
4. cfl=0.1, use_material_cfl=True, alpha_scheme='tvd', use_acid_face=True, acid_interface=True, iterative_im1=True
5. cfl=0.1, use_material_cfl=True, alpha_scheme='mstacs', use_acid_face=True, acid_interface=True, iterative_im1=True
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.He2024.explicit_mmacm_ex import solve_IMEX
from solver.eos.nasg import NASGEOS
from solver.eos.ideal import IdealGasEOS

# matplotlib backend for headless
plt.switch_backend('Agg')

# ============================================================================
# Test A: NASG Abgrall Water-Air Advection
# ============================================================================

def run_test_a(cfl, use_material_cfl, alpha_scheme, use_acid_face, acid_interface, iterative_im1):
    """
    Run Test A with given settings.
    Returns: (success, err_p, err_u, n_steps, wall_time, t_final, comments)
    """
    start_time = time.time()
    try:
        # Solver setup
        N = 10
        L = 1.0
        dx = L / N
        x = np.linspace(dx/2, L - dx/2, N)

        # Initial conditions
        u0 = 1.0  # m/s
        p0 = 1.0e5  # Pa
        T0 = 300.0  # K

        # NASG water: x in [0.4, 0.6]
        gamma_w = 1.187
        p_inf_w = 7.028e8
        b_w = 6.61e-4
        c_v_w = 3610.0  # c_v (internal energy coeff)
        q_w = -1.177788e6  # reference energy

        # Ideal air: elsewhere
        gamma_a = 1.4
        M_a = 28.97  # g/mol (air)

        # Water EOS object
        eos_w = NASGEOS(gamma=gamma_w, p_inf=p_inf_w, b=b_w, c_v=c_v_w, q=q_w)
        eos_a = IdealGasEOS(gamma=gamma_a, M=M_a)

        # Densities from (p, T)
        # For NASG: p = (gamma-1)*c_v*rho*T/(1-b*rho) - p_inf
        # => rho = (p + p_inf) / ((gamma-1)*c_v*T + (p+p_inf)*b)
        rho_w_ref = (p0 + p_inf_w) / ((gamma_w - 1.0) * c_v_w * T0 + (p0 + p_inf_w) * b_w)

        # For ideal gas: p = rho * R_s * T, so rho = p / (R_s * T)
        # R_s = 8.314 / M_a = 8.314 / 0.02897 ≈ 287.05 J/(kg·K)
        R_s_a = 8.314 / 0.02897  # J/(kg·K)
        rho_a_ref = p0 / (R_s_a * T0)

        # Initial fields
        a1 = np.where((x >= 0.4) & (x <= 0.6), 1.0, 0.0)
        rho1 = np.full(N, rho_w_ref)  # phase 1 (water)
        rho2 = np.full(N, rho_a_ref)  # phase 2 (air)
        u = np.full(N, u0)
        p = np.full(N, p0)

        # Solver config
        cfg = {
            'cfl': cfl,
            'use_material_cfl': use_material_cfl,
            'alpha_scheme': alpha_scheme,
            'use_acid_face': use_acid_face,
            'acid_interface': acid_interface,
            'iterative_im1': iterative_im1,
            'use_mmacm_ex': True,
            'use_compression': False,
            'use_apec': True,
            'use_slau2': True,
            'bc': 'periodic',
            'max_steps': 100,
            'max_newton': 5,
            'newton_tol': 1e-8,
            'verbose': 0,
        }

        # Run solver
        sol = solve_IMEX(
            rho1, rho2, u, p, a1,
            eos_w, eos_a,
            L, 0.01, **cfg  # dt=0.01 s (fixed)
        )

        # Extract results
        a1_f, rho1_f, rho2_f, u_f, p_f = sol

        # Check for NaN/Inf
        if np.any(np.isnan(a1_f)) or np.any(np.isinf(a1_f)):
            return (False, np.nan, np.nan, -1, time.time()-start_time, 0.0, "NaN in a1")
        if np.any(np.isnan(u_f)) or np.any(np.isinf(u_f)):
            return (False, np.nan, np.nan, -1, time.time()-start_time, 0.0, "NaN in u")
        if np.any(np.isnan(p_f)) or np.any(np.isinf(p_f)):
            return (False, np.nan, np.nan, -1, time.time()-start_time, 0.0, "NaN in p")

        # Errors
        err_p = np.max(np.abs(p_f - p0) / p0)
        err_u = np.max(np.abs(u_f - u0))

        # Check alpha bounds
        if np.any(a1_f < -1e-8) or np.any(a1_f > 1.0 + 1e-8):
            comments = f"alpha bounds violated: [{np.min(a1_f):.3e}, {np.max(a1_f):.3e}]"
        else:
            comments = "OK"

        return (
            err_p < 1e-2 and err_u < 1e-2 and np.all(a1_f >= 0) and np.all(a1_f <= 1.0),
            err_p,
            err_u,
            100,
            time.time() - start_time,
            1.0,
            comments
        )

    except Exception as e:
        wall_time = time.time() - start_time
        return (False, np.nan, np.nan, -1, wall_time, 0.0, str(e)[:100])

# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    settings = [
        {'cfl': 0.1, 'use_material_cfl': True, 'alpha_scheme': 'cicsam', 'use_acid_face': True, 'acid_interface': True, 'iterative_im1': True, 'label': '1'},
        {'cfl': 0.2, 'use_material_cfl': True, 'alpha_scheme': 'cicsam', 'use_acid_face': True, 'acid_interface': True, 'iterative_im1': True, 'label': '2'},
        {'cfl': 0.1, 'use_material_cfl': True, 'alpha_scheme': 'cicsam', 'use_acid_face': True, 'acid_interface': True, 'iterative_im1': False, 'label': '3'},
        {'cfl': 0.1, 'use_material_cfl': True, 'alpha_scheme': 'tvd', 'use_acid_face': True, 'acid_interface': True, 'iterative_im1': True, 'label': '4'},
        {'cfl': 0.1, 'use_material_cfl': True, 'alpha_scheme': 'mstacs', 'use_acid_face': True, 'acid_interface': True, 'iterative_im1': True, 'label': '5'},
    ]

    results = []

    for s in settings:
        print(f"\n{'='*70}")
        print(f"Test A — Setting {s['label']}: cfl={s['cfl']}, material_cfl={s['use_material_cfl']}, alpha={s['alpha_scheme']}")
        print(f"  use_acid_face={s['use_acid_face']}, acid_interface={s['acid_interface']}, iterative_im1={s['iterative_im1']}")
        print(f"{'='*70}")

        success, err_p, err_u, n_steps, wall_time, t_final, comments = run_test_a(
            cfl=s['cfl'],
            use_material_cfl=s['use_material_cfl'],
            alpha_scheme=s['alpha_scheme'],
            use_acid_face=s['use_acid_face'],
            acid_interface=s['acid_interface'],
            iterative_im1=s['iterative_im1']
        )

        status = "PASS" if success else "FAIL"
        print(f"Result: {status}")
        print(f"  err_p = {err_p:.6e}")
        print(f"  err_u = {err_u:.6e}")
        print(f"  steps = {n_steps}, t_final = {t_final:.4f} s, wall_time = {wall_time:.3f} s")
        print(f"  comment: {comments}")

        results.append({
            'label': s['label'],
            'cfl': s['cfl'],
            'use_material_cfl': s['use_material_cfl'],
            'alpha_scheme': s['alpha_scheme'],
            'use_acid_face': s['use_acid_face'],
            'acid_interface': s['acid_interface'],
            'iterative_im1': s['iterative_im1'],
            'success': success,
            'err_p': err_p,
            'err_u': err_u,
            'n_steps': n_steps,
            'wall_time': wall_time,
            't_final': t_final,
            'comments': comments,
        })

    # Summary table
    print(f"\n{'='*100}")
    print("SUMMARY TABLE")
    print(f"{'='*100}")
    print(f"{'#':<2} {'status':<6} {'err_p':<12} {'err_u':<12} {'wall_time':<10} | cfl | alpha | iterIM1 | comment")
    print(f"{'-'*100}")

    for r in results:
        status = "PASS " if r['success'] else "FAIL "
        print(f"{r['label']:<2} {status:<6} {r['err_p']:<12.4e} {r['err_u']:<12.4e} {r['wall_time']:<10.3f} | {r['cfl']:<3} | {r['alpha_scheme']:<5} | {str(r['iterative_im1']):<7} | {r['comments'][:30]}")

    # Write summary to file
    summary_path = Path(__file__).parent / "02A_mat_cfl_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Test A Material CFL Validation — 5 Settings\n")
        f.write("="*100 + "\n\n")
        for r in results:
            f.write(f"Setting {r['label']}:\n")
            f.write(f"  cfl={r['cfl']}, use_material_cfl={r['use_material_cfl']}\n")
            f.write(f"  alpha_scheme={r['alpha_scheme']}\n")
            f.write(f"  use_acid_face={r['use_acid_face']}, acid_interface={r['acid_interface']}, iterative_im1={r['iterative_im1']}\n")
            f.write(f"  Status: {'PASS' if r['success'] else 'FAIL'}\n")
            f.write(f"  err_p = {r['err_p']:.6e} (limit 1e-2)\n")
            f.write(f"  err_u = {r['err_u']:.6e} (limit 1e-2)\n")
            f.write(f"  steps = {r['n_steps']}, t_final = {r['t_final']:.4f} s\n")
            f.write(f"  wall_time = {r['wall_time']:.3f} s\n")
            f.write(f"  comment: {r['comments']}\n\n")

    print(f"\nSummary saved to: {summary_path}")
    print("Done.")
