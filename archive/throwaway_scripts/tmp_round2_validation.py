"""Round 2 Validation: Test Matrix T1-T9 (proper EOS per spec).

Tests Fix 1-5 effect on NASG Phase 1 with material CFL modes."""
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
from solver.He2024.kapila_k import solve_kapila_K, cons_to_prim_K

R = '/home/younglin90/work/claude_code/claudeCFD/results'


def setup_02A():
    """02-A Test A: NASG water + Ideal air, u=1 m/s, uniform p0, T0."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 1.187, 'pinf': 7.028e8, 'kv': 3610.0,
           'b': 6.61e-4, 'eta': -1.177788e6}
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = NASGEOS(gamma=1.187, pinf=7.028e8, kv=3610.0,
                   b=6.61e-4, eta=-1.177788e6)
    N, L = 10, 1.0
    dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, u0, T0 = 1.0e5, 1.0, 300.0
    a_water = ((x >= 0.4) & (x <= 0.6)).astype(float)
    a1 = (1 - a_water) * (1 - 1e-6) + a_water * 1e-6
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
    rho2 = eos2.density(np.full(N, p0), np.full(N, T0))
    u = np.full(N, u0); p = np.full(N, p0)
    a1r1 = a1 * rho1
    a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2
    ru = rho * u
    e1 = eos1.energy(rho1, p); e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2
    return dict(ph1=ph1, ph2=ph2, a1r1=a1r1, a2r2=a2r2, ru=ru, rE=rE, a1=a1,
                dx=dx, x=x, N=N, L=L, p0=p0, u0=u0, T0=T0)


def setup_01A():
    """01-A Static SG air-water."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 4.4, 'pinf': 6.0e8, 'kv': 474.2}
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = SGEOS(gamma=4.4, pinf=6.0e8, kv=474.2)
    N, L = 100, 1.0
    dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, T0 = 1.0e5, 293.0
    a1 = np.where(x < 0.5, 1 - 1e-6, 1e-6)
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
    rho2 = eos2.density(np.full(N, p0), np.full(N, T0))
    u = np.zeros(N); p = np.full(N, p0)
    a1r1 = a1 * rho1
    a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2
    ru = rho * u
    e1 = eos1.energy(rho1, p); e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2
    return dict(ph1=ph1, ph2=ph2, a1r1=a1r1, a2r2=a2r2, ru=ru, rE=rE, a1=a1,
                dx=dx, x=x, N=N, L=L, p0=p0, u0=0.0, T0=T0)


def setup_02C():
    """02-C Moving contact u=100 m/s, p=1e9 Pa, SG Water + Ideal Air."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 4.4, 'pinf': 6.0e8, 'kv': 474.2}
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = SGEOS(gamma=4.4, pinf=6.0e8, kv=474.2)
    N, L = 100, 1.0
    dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, u0, T0 = 1.0e9, 100.0, 300.0
    a1 = np.where(x < 0.5, 1 - 1e-6, 1e-6)
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
    rho2 = eos2.density(np.full(N, p0), np.full(N, T0))
    u = np.full(N, u0); p = np.full(N, p0)
    a1r1 = a1 * rho1
    a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2
    ru = rho * u
    e1 = eos1.energy(rho1, p); e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2
    return dict(ph1=ph1, ph2=ph2, a1r1=a1r1, a2r2=a2r2, ru=ru, rE=rE, a1=a1,
                dx=dx, x=x, N=N, L=L, p0=p0, u0=u0, T0=T0)


def run_02A_case(name, **kwargs):
    """Run 02-A with kwargs. Returns (status, err_p, err_u, wall, step, t_final)."""
    S = setup_02A()
    t0 = time.time()
    try:
        t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
            S['ph1'], S['ph2'],
            S['a1r1'].copy(), S['a2r2'].copy(), S['ru'].copy(),
            S['rE'].copy(), S['a1'].copy(),
            dx=S['dx'], bc_l='periodic', bc_r='periodic',
            print_interval=999999, **kwargs)
        wall = time.time() - t0
        p_f, u_f, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f,
                                     S['ph1'], S['ph2'])
        if np.any(np.isnan(p_f)) or np.any(np.isnan(u_f)):
            return ('FAIL-NaN', float('nan'), float('nan'), wall, -1, t)
        err_p = float(np.max(np.abs(p_f - S['p0']) / S['p0']))
        err_u = float(np.max(np.abs(u_f - S['u0'])))
        status = 'PASS' if (err_p < 1e-2 and err_u < 1e-2 and t >= 0.99) else 'FAIL'
        return (status, err_p, err_u, wall, -1, t)
    except Exception as e:
        return (f'ERROR:{type(e).__name__}', float('nan'), float('nan'),
                time.time()-t0, -1, -1)


def run_01A():
    S = setup_01A()
    t0 = time.time()
    try:
        t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
            S['ph1'], S['ph2'],
            S['a1r1'].copy(), S['a2r2'].copy(), S['ru'].copy(),
            S['rE'].copy(), S['a1'].copy(),
            dx=S['dx'], t_end=1e-3, cfl=0.4,
            bc_l='transmissive', bc_r='transmissive',
            max_steps=10000, print_interval=999999,
            alpha_scheme='thinc_bvd', use_mmacm_ex=True)
        wall = time.time() - t0
        p_f, u_f, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f,
                                     S['ph1'], S['ph2'])
        err_p = float(np.max(np.abs(p_f - S['p0']) / S['p0']))
        err_u = float(np.max(np.abs(u_f)))
        status = 'PASS' if (err_p < 1e-3 and err_u < 1.0) else 'FAIL'
        return (status, err_p, err_u, wall, -1, t)
    except Exception as e:
        return (f'ERROR:{type(e).__name__}', float('nan'), float('nan'),
                time.time()-t0, -1, -1)


def run_02C():
    S = setup_02C()
    t0 = time.time()
    try:
        t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
            S['ph1'], S['ph2'],
            S['a1r1'].copy(), S['a2r2'].copy(), S['ru'].copy(),
            S['rE'].copy(), S['a1'].copy(),
            dx=S['dx'], t_end=0.01, cfl=0.4,
            bc_l='periodic', bc_r='periodic',
            max_steps=20000, print_interval=999999,
            alpha_scheme='thinc_bvd', use_mmacm_ex=True, use_apec=True)
        wall = time.time() - t0
        p_f, u_f, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f,
                                     S['ph1'], S['ph2'])
        err_p = float(np.max(np.abs(p_f - S['p0']) / S['p0']))
        err_u = float(np.max(np.abs(u_f - S['u0'])))
        status = 'PASS' if (err_p < 1e-10 and err_u < 1e-10) else 'FAIL'
        return (status, err_p, err_u, wall, -1, t)
    except Exception as e:
        return (f'ERROR:{type(e).__name__}', float('nan'), float('nan'),
                time.time()-t0, -1, -1)


if __name__ == '__main__':
    results = []
    COMMON_NASG = dict(use_acid_face=True, acid_interface=True,
                       alpha_scheme='tvd', use_mmacm_ex=False,
                       use_apec=False, use_compression=False,
                       dissipation='none', primitive_recon='tvd',
                       t_end=1.0, max_steps=100000)

    # T1: Baseline (acoustic CFL, known PASS)
    print('\n=== T1: Baseline acoustic CFL ===')
    r = run_02A_case('T1', cfl=0.2, use_material_cfl=False,
                     iterative_im1=True, iterative_im1_max=5,
                     iterative_im1_tol=1e-6, **COMMON_NASG)
    print(f'  {r}'); results.append(('T1-Baseline-ac0.2', r))

    # T2: Material CFL 0.1 + iter (MAIN TARGET)
    print('\n=== T2: Material CFL 0.1 iter ===')
    r = run_02A_case('T2', cfl=0.1, use_material_cfl=True,
                     iterative_im1=True, iterative_im1_max=5,
                     iterative_im1_tol=1e-6, **COMMON_NASG)
    print(f'  {r}'); results.append(('T2-MatCFL0.1-iter', r))

    # T3: Material CFL 0.2
    print('\n=== T3: Material CFL 0.2 iter ===')
    r = run_02A_case('T3', cfl=0.2, use_material_cfl=True,
                     iterative_im1=True, iterative_im1_max=5,
                     iterative_im1_tol=1e-6, **COMMON_NASG)
    print(f'  {r}'); results.append(('T3-MatCFL0.2-iter', r))

    # T4: Material CFL 0.4
    print('\n=== T4: Material CFL 0.4 iter ===')
    r = run_02A_case('T4', cfl=0.4, use_material_cfl=True,
                     iterative_im1=True, iterative_im1_max=5,
                     iterative_im1_tol=1e-6, **COMMON_NASG)
    print(f'  {r}'); results.append(('T4-MatCFL0.4-iter', r))

    # T5: Material CFL 0.1 + CICSAM (SIM test)
    print('\n=== T5: Material CFL 0.1 CICSAM SIM ===')
    common_cicsam = dict(COMMON_NASG); common_cicsam['alpha_scheme'] = 'cicsam'
    r = run_02A_case('T5', cfl=0.1, use_material_cfl=True,
                     iterative_im1=True, iterative_im1_max=5,
                     iterative_im1_tol=1e-6, **common_cicsam)
    print(f'  {r}'); results.append(('T5-MatCFL0.1-CICSAM', r))

    # T6: DC path (Dumbser-Casulli)
    print('\n=== T6: Material CFL 0.1 Dumbser-Casulli ===')
    r = run_02A_case('T6', cfl=0.1, use_material_cfl=True,
                     acoustic_method='dumbser_casulli',
                     iterative_im1=False, **COMMON_NASG)
    print(f'  {r}'); results.append(('T6-DC-path', r))

    # T9: Regression 01-A (SG static)
    print('\n=== T9: 01-A Regression ===')
    r = run_01A()
    print(f'  {r}'); results.append(('T9-01A-SG-static', r))

    # T8: Regression 02-C (SG moving contact)
    print('\n=== T8: 02-C SG Regression ===')
    r = run_02C()
    print(f'  {r}'); results.append(('T8-02C-SG-contact', r))

    # Save summary
    with open(f'{R}/round2_results.txt', 'w') as f:
        f.write('Round 2 Validation Results (Fix 1-5 applied)\n')
        f.write('=' * 70 + '\n\n')
        for name, r in results:
            status, ep, eu, wall, step, tf = r
            f.write(f'{name:30s} | {status:15s} | err_p={ep:.3e} | '
                    f'err_u={eu:.3e} | wall={wall:6.1f}s | t={tf:.4f}\n')
    print(f'\nSummary saved to {R}/round2_results.txt')
