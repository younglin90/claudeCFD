"""Diagnose EB1: which component causes the overshoot?"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim

ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
ph2 = {'gamma': 7.15, 'pinf': 3.31e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}

N = 400; L = 1.0; dx = L / N
x = np.linspace(dx/2, L-dx/2, N)
x0 = 0.5; t_end = 2.4e-4
p_L = 1e9; p_R = 1e5
rho_L_gas = 1.0; rho_R_water = 1000.0

a_air = np.where(x < x0, 1.0-1e-6, 1e-6)
p_init = np.where(x < x0, p_L, p_R)
rho1 = np.where(x < x0, rho_L_gas, rho_L_gas)
rho2 = np.where(x < x0, rho_R_water, rho_R_water)
a1r1 = a_air * rho1; a2r2 = (1-a_air) * rho2
rho = a1r1 + a2r2
rho_e0 = a_air * p_init / 0.4 + (1-a_air) * (p_init + 7.15*3.31e8) / (7.15-1.0)
rE = rho_e0

print("="*60)
print("EB1 diagnosis: Exact u_star = 390")
print("="*60)

for use_mmacm, label in [(True, "MMACM ON"), (False, "MMACM OFF")]:
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1.copy(), a2r2.copy(), np.zeros(N), rE.copy(), a_air.copy(),
        dx, t_end=t_end, cfl=0.25, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=100000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=False, use_material_cfl=False,
        use_mmacm_ex=use_mmacm)
    p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    u_max = np.max(np.abs(u_n))
    u_argmax = x[np.argmax(np.abs(u_n))]
    print(f"  {label}: u_max={u_max:.2f} at x={u_argmax:.3f}")
