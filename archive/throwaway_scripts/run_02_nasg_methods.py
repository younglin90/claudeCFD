"""Case 02 NASG advection — compare im1 / imex_5n_strang / schur_5n / imex_5n.
Tests NASG + general EOS compatibility across all acoustic methods.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from solver.He2024.eos_general import IdealEOS, NASGEOS


def run_case02(method):
    print(f'\n=== Method: {method} ===', flush=True)
    ph1 = {'gamma': 1.4, 'pinf': 0, 'kv': 717.5}
    ph2 = {'gamma': 1.187, 'pinf': 7.028e8, 'kv': 3610,
           'b': 6.61e-4, 'eta': -1.177788e6}
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = NASGEOS(gamma=1.187, pinf=7.028e8, kv=3610,
                   b=6.61e-4, eta=-1.177788e6)
    N, L = 10, 1.0
    dx = L / N
    x = np.linspace(dx/2, L - dx/2, N)
    p0, u0, T0 = 1e5, 1.0, 300.0
    a_w = ((x >= 0.4) & (x <= 0.6)).astype(float)
    a1 = (1 - a_w) * (1 - 1e-6) + a_w * 1e-6
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
    try:
        # dt=0.01 via material CFL (max_speed floor = 1.0 m/s, dx=0.1)
        # → dt = cfl·dx/1.0 = 0.01 when cfl=0.1
        t, ar1, ar2, ru_f, rE_f, a1_f = solve_IMEX(
            ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
            dx=dx, t_end=1.0, cfl=0.1, use_material_cfl=True,
            bc_l='periodic', bc_r='periodic',
            max_steps=10000, print_interval=999999,
            acoustic_method=method, imex_rk2=True,
            nl_picard_max=5 if method == 'schur_5n' else 0,
            nl_picard_tol=1e-6, nl_picard_relax=0.5)
        wall = time.time() - t0
        p_f, u_f, *_ = cons_to_prim(ar1, ar2, ru_f, rE_f, a1_f, ph1, ph2)
        ep = float(np.max(np.abs(p_f - p0)) / p0)
        eu = float(np.max(np.abs(u_f - u0)))
        rho_max = float(np.max(ar1 + ar2))
        rho_min = float(np.min(ar1 + ar2))
        status = 'PASS' if (ep < 1e-8 and eu < 1e-4) else 'FAIL'
        print(f'  t={t:.4e} wall={wall:.2f}s ep={ep:.3e} eu={eu:.3e} '
              f'ρ=[{rho_min:.2f},{rho_max:.2f}] → {status}', flush=True)
        return status, wall, ep, eu
    except Exception as e:
        wall = time.time() - t0
        print(f'  EXCEPTION wall={wall:.2f}s: {type(e).__name__}: {e}', flush=True)
        return 'ERROR', wall, float('nan'), float('nan')


if __name__ == '__main__':
    methods = ['imex_5n', 'imex_5n_strang', 'schur_5n']
    results = {}
    for m in methods:
        results[m] = run_case02(m)

    print('\n' + '=' * 70)
    print(f'{"Method":<20} {"Status":<8} {"Wall":>8} {"ep":>12} {"eu":>12}')
    print('-' * 70)
    for m, (st, w, ep, eu) in results.items():
        print(f'{m:<20} {st:<8} {w:>7.2f}s {ep:>12.3e} {eu:>12.3e}')
    print('=' * 70)
    passed = sum(1 for st, _, _, _ in results.values() if st == 'PASS')
    print(f'\n{passed}/{len(methods)} PASS')
