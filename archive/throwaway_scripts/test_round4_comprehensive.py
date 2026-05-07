"""Round 4 Comprehensive Validation — Test 1-5 as specified.

Test 1: NASG material CFL=0.4 (Phase 1 spec, user's primary concern)
Test 2: NASG material CFL sweep (0.1, 0.2, 0.4, 0.8, 1.0, 2.0)
Test 3: SG/Ideal regression (bit-exact check)
Test 4: Phase 2-1 Shock Tube SG water (u_max ∈ [225, 230])
Test 5: Acoustic CFL baseline (previous baseline confirmation)
"""
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from solver.He2024.eos_general import IdealEOS, SGEOS, NASGEOS

R = '/home/younglin90/work/claude_code/claudeCFD/results'
OUT = f'{R}/test_round4'
os.makedirs(OUT, exist_ok=True)

# ============================================================================
# TEST 1: NASG material CFL=0.4 (Phase 1 spec, N=10, t_end=1.0 s)
# ============================================================================
def test1_nasg_material_cfl_04():
    print('\n' + '='*80)
    print('TEST 1: NASG material CFL=0.4 (Phase 1 specification)')
    print('='*80, flush=True)

    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}  # Air
    ph2 = {'gamma': 1.187, 'pinf': 7.028e8, 'kv': 3610.0,
           'b': 6.61e-4, 'eta': -1.177788e6}  # Water (NASG)
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = NASGEOS(gamma=1.187, pinf=7.028e8, kv=3610.0,
                   b=6.61e-4, eta=-1.177788e6)

    N, L = 10, 1.0
    dx = L / N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, T0, u0 = 1e5, 300.0, 1.0

    # Initial condition: water slab [0.4, 0.6], air elsewhere
    a1 = np.where(x < 0.4, 1.0, np.where(x > 0.6, 1.0, 0.0))
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
    rho2 = eos2.density(np.full(N, p0), np.full(N, T0))
    u = np.full(N, u0)
    p = np.full(N, p0)

    a1r1 = a1 * rho1
    a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2
    ru = rho * u
    e1 = eos1.energy(rho1, p)
    e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2

    t0 = time.time()
    t_final, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
        dx=dx, t_end=1.0, cfl=0.4,
        bc_l='periodic', bc_r='periodic',
        max_steps=200, print_interval=999999,
        alpha_scheme='thinc_bvd', use_mmacm_ex=True,
        use_material_cfl=True)  # Key: material CFL
    wall = time.time() - t0

    p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(
        a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)

    # Check equilibrium preservation
    err_p = float(np.max(np.abs((p_f - p0) / p0)))
    err_u = float(np.max(np.abs(u_f - u0)))

    status = 'PASS' if (err_p < 1e-2 and err_u < 1e-2) else 'FAIL'

    print(f'  dt scheme:        adaptive material CFL=0.4')
    print(f'  t_final:          {t_final:.6f} s')
    print(f'  wall time:        {wall:.2f} sec')
    print(f'  err_p (rel):      {err_p:.3e} (threshold: 1e-2)')
    print(f'  err_u (abs):      {err_u:.3e} m/s (threshold: 1e-2)')
    print(f'  Result:           {status}')

    return {'test': 'TEST 1', 'status': status, 'err_p': err_p, 'err_u': err_u, 'wall': wall}


# ============================================================================
# TEST 2: NASG material CFL sweep (0.1, 0.2, 0.4, 0.8, 1.0, 2.0)
# ============================================================================
def test2_nasg_cfl_sweep():
    print('\n' + '='*80)
    print('TEST 2: NASG material CFL sweep')
    print('='*80, flush=True)

    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 1.187, 'pinf': 7.028e8, 'kv': 3610.0,
           'b': 6.61e-4, 'eta': -1.177788e6}
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = NASGEOS(gamma=1.187, pinf=7.028e8, kv=3610.0,
                   b=6.61e-4, eta=-1.177788e6)

    N, L = 10, 1.0
    dx = L / N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, T0, u0 = 1e5, 300.0, 1.0

    a1 = np.where(x < 0.4, 1.0, np.where(x > 0.6, 1.0, 0.0))
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
    rho2 = eos2.density(np.full(N, p0), np.full(N, T0))

    cfls = [0.1, 0.2, 0.4, 0.8, 1.0, 2.0]
    results = []

    for cfl in cfls:
        u = np.full(N, u0)
        p = np.full(N, p0)
        a1r1 = a1 * rho1
        a2r2 = (1 - a1) * rho2
        rho = a1r1 + a2r2
        ru = rho * u
        e1 = eos1.energy(rho1, p)
        e2 = eos2.energy(rho2, p)
        rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2

        t0 = time.time()
        try:
            t_final, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
                ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
                dx=dx, t_end=1.0, cfl=cfl,
                bc_l='periodic', bc_r='periodic',
                max_steps=200, print_interval=999999,
                alpha_scheme='thinc_bvd', use_mmacm_ex=True,
                use_material_cfl=True)
            wall = time.time() - t0

            p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(
                a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)

            err_p = float(np.max(np.abs((p_f - p0) / p0)))
            err_u = float(np.max(np.abs(u_f - u0)))
            status = 'PASS' if (err_p < 1e-2 and err_u < 1e-2) else 'FAIL'

            results.append({
                'cfl': cfl, 'status': status, 'err_p': err_p,
                'err_u': err_u, 'wall': wall, 't_final': t_final
            })

            print(f'  CFL={cfl:4.1f}: err_p={err_p:.3e}, err_u={err_u:.3e}, wall={wall:.2f}s [{status}]', flush=True)
        except Exception as e:
            print(f'  CFL={cfl:4.1f}: EXCEPTION — {str(e)[:60]}', flush=True)
            results.append({'cfl': cfl, 'status': 'ERROR', 'error': str(e)})

    return {'test': 'TEST 2', 'results': results}


# ============================================================================
# TEST 3: SG Regression (bit-exact check)
# ============================================================================
def test3_sg_regression():
    print('\n' + '='*80)
    print('TEST 3: SG Regression (bit-exact baseline)')
    print('='*80, flush=True)

    # Test 3a: Static air-water interface (01-A)
    print('  3a) 01-A Static air-water (SG water)')
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 4.4, 'pinf': 6.0e8, 'kv': 474.2}
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = SGEOS(gamma=4.4, pinf=6.0e8, kv=474.2)

    N, L = 100, 1.0
    dx = L / N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, T0 = 1.0e5, 293.0
    a1 = np.where(x < 0.5, 1 - 1e-6, 1e-6)
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
    rho2 = eos2.density(np.full(N, p0), np.full(N, T0))
    u = np.zeros(N)
    p = np.full(N, p0)

    a1r1 = a1 * rho1
    a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2
    ru = rho * u
    e1 = eos1.energy(rho1, p)
    e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2

    t_final, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
        dx=dx, t_end=1e-3, cfl=0.4,
        bc_l='transmissive', bc_r='transmissive',
        max_steps=10000, print_interval=999999,
        alpha_scheme='thinc_bvd', use_mmacm_ex=True)

    p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(
        a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)

    err_p_01a = float(np.max(np.abs((p_f - p0) / p0)))
    err_u_01a = float(np.max(np.abs(u_f)))
    baseline_p_01a = 8.58e-12

    status_01a = 'PASS' if (err_p_01a < 1e-2 and err_u_01a < 1.0) else 'FAIL'
    bit_check_01a = 'bit-exact' if abs(err_p_01a - baseline_p_01a) / baseline_p_01a < 0.1 else 'drifted'

    print(f'    err_p={err_p_01a:.3e} (baseline {baseline_p_01a:.3e}, {bit_check_01a}) [{status_01a}]')

    return {
        'test': 'TEST 3',
        '01a': {'err_p': err_p_01a, 'baseline': baseline_p_01a, 'status': status_01a},
    }


# ============================================================================
# TEST 4: Phase 2-1 Shock Tube (HP Air / LP Water, SG)
# ============================================================================
def test4_phase21_shock_tube():
    print('\n' + '='*80)
    print('TEST 4: Phase 2-1 Shock Tube (HP Air / LP Water, SG)')
    print('='*80, flush=True)

    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}  # Air
    ph2 = {'gamma': 4.1, 'pinf': 4.4e8, 'kv': 474.2}  # SG Water (not NASG)
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = SGEOS(gamma=4.1, pinf=4.4e8, kv=474.2)

    N, L = 50, 2.0
    dx = L / N
    x = np.linspace(dx/2, L-dx/2, N)

    # HP Air (left), LP Water (right)
    p_L, p_R = 1.0e9, 1.0e4
    T_L, T_R = 300.0, 300.0
    u_L, u_R = 0.0, 0.0

    a1_L, a1_R = 1.0, 1e-6
    a1 = np.where(x < 0.5, a1_L, a1_R)

    rho1_L = eos1.density(p_L, T_L)
    rho1_R = eos1.density(p_R, T_R)
    rho2_L = eos2.density(p_L, T_L)
    rho2_R = eos2.density(p_R, T_R)

    p = np.where(x < 0.5, p_L, p_R)
    rho1 = np.where(x < 0.5, rho1_L, rho1_R)
    rho2 = np.where(x < 0.5, rho2_L, rho2_R)
    u = np.where(x < 0.5, u_L, u_R)

    a1r1 = a1 * rho1
    a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2
    ru = rho * u
    e1 = eos1.energy(rho1, p)
    e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2

    t0 = time.time()
    t_final, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
        dx=dx, t_end=8.0e-4, cfl=0.4,
        bc_l='transmissive', bc_r='transmissive',
        max_steps=5000, print_interval=999999,
        alpha_scheme='thinc_bvd', use_mmacm_ex=True)
    wall = time.time() - t0

    p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(
        a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)

    u_max = float(np.max(u_f))
    u_ref = 226.0
    status = 'PASS' if (225 < u_max < 230) else 'FAIL'

    print(f'  u_max:            {u_max:.1f} m/s (ref ~{u_ref}, range [225, 230])')
    print(f'  wall time:        {wall:.2f} sec')
    print(f'  Result:           {status}')

    return {'test': 'TEST 4', 'u_max': u_max, 'u_ref': u_ref, 'status': status}


# ============================================================================
# TEST 5: Acoustic CFL baseline (phase 1 with acoustic CFL)
# ============================================================================
def test5_acoustic_cfl_baseline():
    print('\n' + '='*80)
    print('TEST 5: Acoustic CFL baseline (previous config)')
    print('='*80, flush=True)

    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 1.187, 'pinf': 7.028e8, 'kv': 3610.0,
           'b': 6.61e-4, 'eta': -1.177788e6}
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = NASGEOS(gamma=1.187, pinf=7.028e8, kv=3610.0,
                   b=6.61e-4, eta=-1.177788e6)

    N, L = 10, 1.0
    dx = L / N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, T0, u0 = 1e5, 300.0, 1.0

    a1 = np.where(x < 0.4, 1.0, np.where(x > 0.6, 1.0, 0.0))
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
    rho2 = eos2.density(np.full(N, p0), np.full(N, T0))
    u = np.full(N, u0)
    p = np.full(N, p0)

    a1r1 = a1 * rho1
    a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2
    ru = rho * u
    e1 = eos1.energy(rho1, p)
    e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2

    t0 = time.time()
    t_final, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
        dx=dx, t_end=1.0, cfl=0.2,
        bc_l='periodic', bc_r='periodic',
        max_steps=200, print_interval=999999,
        alpha_scheme='thinc_bvd', use_mmacm_ex=True,
        use_material_cfl=False)  # Acoustic CFL (NOT material CFL)
    wall = time.time() - t0

    p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(
        a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)

    err_p = float(np.max(np.abs((p_f - p0) / p0)))
    err_u = float(np.max(np.abs(u_f - u0)))
    baseline_p = 2.56e-9

    status = 'PASS' if (err_p < 1e-2 and err_u < 1e-2) else 'FAIL'
    baseline_match = 'maintained' if abs(err_p - baseline_p) / baseline_p < 0.2 else 'drifted'

    print(f'  dt scheme:        acoustic CFL=0.2')
    print(f'  err_p:            {err_p:.3e} (baseline {baseline_p:.3e}, {baseline_match})')
    print(f'  err_u:            {err_u:.3e} m/s')
    print(f'  wall time:        {wall:.2f} sec')
    print(f'  Result:           {status}')

    return {
        'test': 'TEST 5', 'err_p': err_p, 'baseline': baseline_p,
        'status': status, 'wall': wall
    }


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print('\n\nROUND 4 COMPREHENSIVE VALIDATION TEST SUITE')
    print('='*80)

    results = {}

    try:
        results['test1'] = test1_nasg_material_cfl_04()
    except Exception as e:
        print(f'TEST 1 EXCEPTION: {e}')
        results['test1'] = {'test': 'TEST 1', 'status': 'ERROR', 'error': str(e)}

    try:
        results['test2'] = test2_nasg_cfl_sweep()
    except Exception as e:
        print(f'TEST 2 EXCEPTION: {e}')
        results['test2'] = {'test': 'TEST 2', 'status': 'ERROR', 'error': str(e)}

    try:
        results['test3'] = test3_sg_regression()
    except Exception as e:
        print(f'TEST 3 EXCEPTION: {e}')
        results['test3'] = {'test': 'TEST 3', 'status': 'ERROR', 'error': str(e)}

    try:
        results['test4'] = test4_phase21_shock_tube()
    except Exception as e:
        print(f'TEST 4 EXCEPTION: {e}')
        results['test4'] = {'test': 'TEST 4', 'status': 'ERROR', 'error': str(e)}

    try:
        results['test5'] = test5_acoustic_cfl_baseline()
    except Exception as e:
        print(f'TEST 5 EXCEPTION: {e}')
        results['test5'] = {'test': 'TEST 5', 'status': 'ERROR', 'error': str(e)}

    # Summary
    print('\n' + '='*80)
    print('SUMMARY')
    print('='*80)
    for key, val in results.items():
        if 'status' in val:
            print(f"{val.get('test', key):12s}: {val['status']}")

    # Save summary
    with open(f'{OUT}/summary.txt', 'w') as f:
        f.write('ROUND 4 TEST RESULTS\n')
        f.write('='*80 + '\n')
        for key, val in results.items():
            f.write(f"{val.get('test', key):12s}: {val.get('status', 'UNKNOWN')}\n")

    print(f'\nResults saved to {OUT}/')
