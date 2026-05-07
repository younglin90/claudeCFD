#!/usr/bin/env python3
"""
Test A validation with 5 material CFL settings.
NASG Abgrall Water-Air Advection.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')

from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from solver.He2024.eos_general import IdealEOS, NASGEOS

# Test A: 5 settings
SETTINGS = [
    {'cfl': 0.1, 'use_material_cfl': True, 'alpha_scheme': 'cicsam', 'use_acid_face': True, 'acid_interface': True, 'iterative_im1': True, 'label': '1'},
    {'cfl': 0.2, 'use_material_cfl': True, 'alpha_scheme': 'cicsam', 'use_acid_face': True, 'acid_interface': True, 'iterative_im1': True, 'label': '2'},
    {'cfl': 0.1, 'use_material_cfl': True, 'alpha_scheme': 'cicsam', 'use_acid_face': True, 'acid_interface': True, 'iterative_im1': False, 'label': '3'},
    {'cfl': 0.1, 'use_material_cfl': True, 'alpha_scheme': 'tvd', 'use_acid_face': True, 'acid_interface': True, 'iterative_im1': True, 'label': '4'},
    {'cfl': 0.1, 'use_material_cfl': True, 'alpha_scheme': 'mstacs', 'use_acid_face': True, 'acid_interface': True, 'iterative_im1': True, 'label': '5'},
]

def run_test_a(setting):
    """Run Test A with given setting. Returns (success, err_p, err_u, steps, wall, t_final, comment)"""
    start = time.time()
    try:
        # EOS setup
        ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}  # Air
        ph2 = {'gamma': 1.187, 'pinf': 7.028e8, 'kv': 3610.0, 'b': 6.61e-4, 'eta': -1.177788e6}  # Water

        eos1 = IdealEOS(gamma=1.4, kv=717.5)
        eos2 = NASGEOS(gamma=1.187, pinf=7.028e8, kv=3610.0, b=6.61e-4, eta=-1.177788e6)

        N, L = 10, 1.0
        dx = L / N
        x = np.linspace(dx/2, L-dx/2, N)
        p0, u0, T0 = 1.0e5, 1.0, 300.0

        # α_air outside [0.4, 0.6], α_water inside
        a_water = ((x >= 0.4) & (x <= 0.6)).astype(float)
        a1 = (1 - a_water) * (1 - 1e-6) + a_water * 1e-6

        # Densities
        rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
        rho2 = eos2.density(np.full(N, p0), np.full(N, T0))
        u = np.full(N, u0)
        p = np.full(N, p0)

        # Conservative vars
        a1r1 = a1 * rho1
        a2r2 = (1 - a1) * rho2
        rho = a1r1 + a2r2
        ru = rho * u
        e1 = eos1.energy(rho1, p)
        e2 = eos2.energy(rho2, p)
        rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2

        # Solver call
        t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
            ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
            dx=dx, t_end=1.0, cfl=setting['cfl'],
            use_material_cfl=setting['use_material_cfl'],
            bc_l='periodic', bc_r='periodic',
            max_steps=100, print_interval=999999,
            alpha_scheme=setting['alpha_scheme'],
            use_mmacm_ex=False, use_apec=False,
            use_compression=False, dissipation='none',
            primitive_recon='tvd',
            use_acid_face=setting['use_acid_face'],
            acid_interface=setting['acid_interface'],
            iterative_im1=setting['iterative_im1'],
            iterative_im1_max=5, iterative_im1_tol=1e-6
        )

        # Cons to prim
        p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(
            a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)

        # Check NaN/Inf
        if np.any(~np.isfinite(p_f)) or np.any(~np.isfinite(u_f)):
            return (False, np.nan, np.nan, -1, time.time()-start, 0.0, "NaN/Inf in results")

        # Errors
        err_p = float(np.max(np.abs(p_f - p0)) / p0)
        err_u = float(np.max(np.abs(u_f - u0)))

        # Alpha bounds
        if np.any(a1_f < -1e-8) or np.any(a1_f > 1.0 + 1e-8):
            return (False, err_p, err_u, 100, time.time()-start, t if np.isfinite(t) else 0.0,
                    f"alpha out of bounds: [{np.min(a1_f):.2e}, {np.max(a1_f):.2e}]")

        # PASS criteria
        success = err_p < 1e-2 and err_u < 1e-2
        return (success, err_p, err_u, 100, time.time()-start, t if np.isfinite(t) else 0.0, "OK")

    except Exception as e:
        import traceback
        msg = str(e)[:80]
        return (False, np.nan, np.nan, -1, time.time()-start, 0.0, msg)

# Main
if __name__ == '__main__':
    print("="*90)
    print("Test A — 5 Material CFL Settings")
    print("="*90)

    results = []
    for s in SETTINGS:
        print(f"\nSetting {s['label']}: cfl={s['cfl']}, material_cfl={s['use_material_cfl']}, "
              f"alpha={s['alpha_scheme']}, iter_im1={s['iterative_im1']}")

        success, err_p, err_u, steps, wall, t_final, comment = run_test_a(s)
        status = "PASS" if success else "FAIL"

        print(f"  {status}: err_p={err_p:.4e}, err_u={err_u:.4e}, wall={wall:.3f}s, t={t_final:.4f}s")
        print(f"  {comment}")

        results.append({
            'label': s['label'],
            'cfl': s['cfl'],
            'use_material_cfl': s['use_material_cfl'],
            'alpha_scheme': s['alpha_scheme'],
            'iterative_im1': s['iterative_im1'],
            'success': success,
            'err_p': err_p,
            'err_u': err_u,
            'wall_time': wall,
            't_final': t_final,
            'comment': comment,
        })

    # Summary
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    print(f"{'#':<3} {'Status':<6} {'err_p':<12} {'err_u':<12} {'wall':<8} | Setting")
    print("-"*90)
    for r in results:
        status = "PASS" if r['success'] else "FAIL"
        print(f"{r['label']:<3} {status:<6} {r['err_p']:<12.4e} {r['err_u']:<12.4e} {r['wall_time']:<8.3f} | "
              f"cfl={r['cfl']}, mat_cfl={r['use_material_cfl']}, alpha={r['alpha_scheme']}, "
              f"iter_im1={r['iterative_im1']}")

    # Write report
    report_path = '/home/younglin90/work/claude_code/claudeCFD/results/qa_report_02A_5settings.md'
    with open(report_path, 'w') as f:
        f.write("# QA Report — Test A 5-Setting Validation\n\n")
        f.write("## Summary\n\n")
        f.write("| Setting | Status | err_p | err_u | wall_time | t_final | Comment |\n")
        f.write("|---------|--------|-------|-------|-----------|---------|----------|\n")
        for r in results:
            status = "PASS" if r['success'] else "FAIL"
            f.write(f"| {r['label']} | {status} | {r['err_p']:.4e} | {r['err_u']:.4e} | {r['wall_time']:.3f}s | {r['t_final']:.4f}s | {r['comment']} |\n")

        f.write("\n## Details\n\n")
        for r in results:
            f.write(f"### Setting {r['label']}\n")
            f.write(f"- CFL: {r['cfl']}\n")
            f.write(f"- use_material_cfl: {r['use_material_cfl']}\n")
            f.write(f"- alpha_scheme: {r['alpha_scheme']}\n")
            f.write(f"- iterative_im1: {r['iterative_im1']}\n")
            f.write(f"- Status: {'PASS' if r['success'] else 'FAIL'}\n")
            f.write(f"- err_p: {r['err_p']:.6e} (limit 1e-2)\n")
            f.write(f"- err_u: {r['err_u']:.6e} (limit 1e-2)\n")
            f.write(f"- wall_time: {r['wall_time']:.3f}s\n")
            f.write(f"- t_final: {r['t_final']:.6f}s\n")
            f.write(f"- comment: {r['comment']}\n\n")

    print(f"\nReport saved: {report_path}")
    print("Done.")
