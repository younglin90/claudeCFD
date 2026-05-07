"""Quick Phase 1 regression test with TVD and BVD."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim

ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
ph2 = {'gamma': 2.35, 'pinf': 1e9, 'kv': 943.8, 'b': 6.61e-4, 'eta': -1167e3, 'q': 0.0}
N = 10; L = 1.0; dx = L / N
x = np.linspace(dx/2, L - dx/2, N)
p0 = 1e5; u0 = 1.0; T0 = 300.0
a1 = np.where((x >= 0.4) & (x <= 0.6), 1e-6, 1.0 - 1e-6)
a2 = 1.0 - a1
rho1 = p0 / ((ph1['gamma'] - 1.0) * ph1['kv'] * T0)
rho2 = (p0 + ph2['pinf']) / ((ph2['gamma'] - 1.0) * ph2['kv'] * T0)
a1r1 = a1 * rho1 * np.ones(N)
a2r2 = a2 * rho2 * np.ones(N)
rho = a1r1 + a2r2
ru = rho * u0
gm1, gm2 = ph1['gamma'] - 1.0, ph2['gamma'] - 1.0
rho_e = a1 * (p0 + ph1['gamma'] * ph1['pinf']) / gm1 + a2 * (p0 + ph2['gamma'] * ph2['pinf']) / gm2
rE = rho_e + 0.5 * rho * u0**2

for scheme in ['tvd', 'thinc_bvd']:
    print(f"\n{'='*50}")
    print(f"Phase 1: {scheme}")
    print(f"{'='*50}")
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1.copy(), a2r2.copy(), ru.copy(), rE.copy(), a1.copy(),
        dx, t_end=1.0, cfl=0.4, bc_l='periodic', bc_r='periodic',
        max_steps=100, print_interval=20,
        alpha_scheme=scheme, use_strang=True,
        use_defect_correction=True, use_material_cfl=True)
    p_f, u_f, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    err_p = np.max(np.abs(p_f / p0 - 1.0))
    err_u = np.max(np.abs(u_f / u0 - 1.0))
    print(f"err_p={err_p:.2e}, err_u={err_u:.2e} -> {'PASS' if err_p < 1e-2 and err_u < 1e-2 else 'FAIL'}")
