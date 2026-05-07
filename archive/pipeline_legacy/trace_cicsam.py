"""Trace CICSAM step-by-step to find where divergence starts."""
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
gm1, gm2 = 0.4, 1.35
rho_e = a1 * (p0 + ph1['gamma'] * ph1['pinf']) / gm1 + a2 * (p0 + ph2['gamma'] * ph2['pinf']) / gm2
rE = rho_e + 0.5 * rho * u0**2

for max_steps, label in [(1, "step 1"), (2, "step 2"), (3, "step 3"), (5, "step 5")]:
    print(f"\n===== {label} =====")
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1.copy(), a2r2.copy(), ru.copy(), rE.copy(), a1.copy(),
        dx, t_end=1.0, cfl=0.4, bc_l='periodic', bc_r='periodic',
        max_steps=max_steps, print_interval=1,
        alpha_scheme='cicsam', use_strang=True,
        use_defect_correction=True, use_material_cfl=False)
    p, u, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    print(f"  p range: [{p.min():.3e}, {p.max():.3e}]")
    print(f"  u range: [{u.min():.3e}, {u.max():.3e}]")
    print(f"  a1: {a1_f}")
