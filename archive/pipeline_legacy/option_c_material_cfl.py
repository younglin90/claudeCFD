"""Option C: Material CFL benefits — IMEX speedup demonstration.

Phase 1 (Abgrall NASG, Mach ~6e-4) is the canonical test:
  - Acoustic CFL: dt ~ dx/c ~ 1e-4 → thousands of steps to t=1s
  - Material CFL: dt ~ dx/u ~ 0.1 → tens of steps to t=1s
  - Both preserve p/u equilibrium
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')


def phase1_setup():
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 2.35, 'pinf': 1e9, 'kv': 943.8, 'b': 6.61e-4, 'eta': -1167e3, 'q': 0.0}
    N = 10; L = 1.0; dx = L / N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, u0, T0 = 1e5, 1.0, 300.0
    a1 = np.where((x >= 0.4) & (x <= 0.6), 1e-6, 1-1e-6)
    a2 = 1.0 - a1
    rho1 = p0 / (0.4 * 717.5 * T0) * np.ones(N)
    rho2 = (p0 + ph2['pinf']) / ((ph2['gamma']-1)*ph2['kv']*T0) * np.ones(N)
    a1r1 = a1 * rho1; a2r2 = a2 * rho2
    rho = a1r1 + a2r2
    ru = rho * u0
    rho_e = a1 * (p0 + ph1['gamma']*ph1['pinf'])/0.4 + a2*(p0 + ph2['gamma']*ph2['pinf'])/1.35
    rE = rho_e + 0.5 * rho * u0**2
    return ph1, ph2, dx, a1r1, a2r2, ru, rE, a1, p0, u0


def run(use_mat, cfl, t_end):
    ph1, ph2, dx, a1r1, a2r2, ru, rE, a1, p0, u0 = phase1_setup()
    t0 = time.time()
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1.copy(), a2r2.copy(), ru.copy(), rE.copy(), a1.copy(),
        dx, t_end=t_end, cfl=cfl, bc_l='periodic', bc_r='periodic',
        max_steps=200000, print_interval=200000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=False, use_material_cfl=use_mat)
    wall = time.time() - t0
    p_f, u_f, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    err_p = np.max(np.abs(p_f/p0 - 1))
    err_u = np.max(np.abs(u_f/u0 - 1))
    return t, wall, err_p, err_u


if __name__ == '__main__':
    t_end = 1e-3  # short: focus on per-step cost comparison without long-time drift
    print(f"\nPhase 1 Abgrall (t_end={t_end}s):")
    print("="*70)
    # Acoustic CFL
    t_a, w_a, ep_a, eu_a = run(use_mat=False, cfl=0.4, t_end=t_end)
    # Material CFL
    t_m, w_m, ep_m, eu_m = run(use_mat=True, cfl=0.4, t_end=t_end)

    print(f"\n{'Mode':<20} {'Wall time':>12} {'err_p':>12} {'err_u':>12}")
    print(f"{'Acoustic CFL=0.4':<20} {w_a:>10.2f}s {ep_a:>12.2e} {eu_a:>12.2e}")
    print(f"{'Material CFL=0.4':<20} {w_m:>10.2f}s {ep_m:>12.2e} {eu_m:>12.2e}")
    print(f"\n Speedup (Acoustic/Material): {w_a/w_m:.1f}×")
    print(f" Both preserve p/u equilibrium: "
          f"{'YES' if ep_a<1e-2 and ep_m<1e-2 and eu_a<1e-2 and eu_m<1e-2 else 'NO'}")
