"""Quick single-config test."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim

ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
N = 200; L = 1.0; dx = L/N
x = np.linspace(dx/2, L-dx/2, N)
x0 = 0.5
p0 = 1e5; p_L = p0*1.01; p_R = p0
p_init = np.where(x < x0, p_L, p_R)
rho1 = p_init / (0.4 * 717.5 * 293.0)
rho2 = (p_init + 6e8) / (3.4 * 474.2 * 293.0)
a_air = 1e-6 * np.ones(N)
a1r1 = a_air * rho1; a2r2 = (1-a_air) * rho2
rho_e0 = a_air * p_init / 0.4 + (1-a_air) * (p_init + 4.4*6e8) / 3.4

for diss, coef in [('none', 0.0), ('shapiro', 0.3), ('shapiro', 0.7), ('shapiro', 1.0), ('mwi', 0.5), ('mwi', 1.0)]:
    import time; t0=time.time()
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1.copy(), a2r2.copy(), np.zeros(N), rho_e0.copy(), a_air.copy(),
        dx, t_end=3e-4, cfl=0.4, bc_l='transmissive', bc_r='transmissive',
        max_steps=500, print_interval=1000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=False, use_material_cfl=False,
        dissipation=diss, diss_coef=coef)
    p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    elapsed = time.time() - t0
    p_osc = (p_n - p_n.mean()).std() / p0
    print(f"diss={diss} coef={coef}: t={t:.3e}, p_osc={p_osc:.3e}, elapsed={elapsed:.1f}s")
