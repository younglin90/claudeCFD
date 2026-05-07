"""Check Phase 2-5A: is overshoot at right boundary (BC artifact)?"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from pipeline.exact_riemann import exact_profile

ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}

N = 200; L = 1.0; dx = L / N
x = np.linspace(dx/2, L-dx/2, N)
x0 = 0.5; T0 = 308.2

# Case A: gas (high-p) left, liquid (low-p) right
p_L = 1e8; p_R = 1e5
alpha_L, alpha_R = 1.0-1e-6, 1e-6

# Exact
rho_L_gas = p_L / (0.4 * 717.5 * T0)
rho_R_water = (p_R + 6e8) / (3.4 * 474.2 * T0)

# Try t_end = 5e-4 (original) and shorter
for t_end in [5e-4, 2e-4, 1e-4]:
    rho_e, u_e, p_e, _ = exact_profile(
        x, t_end, x0,
        pL=p_L, rhoL=rho_L_gas, uL=0.0, gammaL=1.4, pinfL=0.0,
        pR=p_R, rhoR=rho_R_water, uR=0.0, gammaR=4.4, pinfR=6e8)

    a_air = np.where(x < x0, alpha_L, alpha_R)
    p_init = np.where(x < x0, p_L, p_R)
    rho1 = p_init / (0.4 * 717.5 * T0)
    rho2 = (p_init + 6e8) / (3.4 * 474.2 * T0)
    a1r1 = a_air * rho1; a2r2 = (1-a_air) * rho2
    rho_e0 = a_air * p_init / 0.4 + (1-a_air) * (p_init + 4.4*6e8) / 3.4

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, np.zeros(N), rho_e0, a_air,
        dx, t_end=t_end, cfl=0.25, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=100000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=False, use_material_cfl=False)
    p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)

    idx = int(np.argmax(np.abs(u_n)))
    print(f"t_end={t_end:.1e}: u_max_num={u_n[idx]:.2f} at x={x[idx]:.3f}, u_e at same={u_e[idx]:.2f}")
    print(f"  u_e max in domain={u_e.max():.2f}, u_n max in interior (x<0.9)={u_n[x<0.9].max():.2f}")
