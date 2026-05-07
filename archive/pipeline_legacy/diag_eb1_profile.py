"""Plot EB1 profile to understand overshoot source."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from pipeline.exact_riemann import exact_profile

ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
ph2 = {'gamma': 7.15, 'pinf': 3.31e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}

N = 400; L = 1.0; dx = L / N
x = np.linspace(dx/2, L-dx/2, N)
x0 = 0.5; t_end = 2.4e-4

p_L = 1e9; p_R = 1e5
rho_L_gas = 1.0; rho_R_water = 1000.0

rho_e, u_e, p_e, _ = exact_profile(
    x, t_end, x0,
    pL=p_L, rhoL=rho_L_gas, uL=0.0, gammaL=1.4, pinfL=0.0,
    pR=p_R, rhoR=rho_R_water, uR=0.0, gammaR=7.15, pinfR=3.31e8)

a_air = np.where(x < x0, 1.0-1e-6, 1e-6)
p_init = np.where(x < x0, p_L, p_R)
a1r1 = a_air * rho_L_gas; a2r2 = (1-a_air) * rho_R_water
rho_e0 = a_air * p_init / 0.4 + (1-a_air) * (p_init + 7.15*3.31e8) / 6.15

t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
    ph1, ph2, a1r1, a2r2, np.zeros(N), rho_e0, a_air,
    dx, t_end=t_end, cfl=0.25, bc_l='transmissive', bc_r='transmissive',
    max_steps=100000, print_interval=100000,
    alpha_scheme='tvd', use_strang=True,
    use_defect_correction=False, use_material_cfl=False)
p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
rho_n = a1r1_f + a2r2_f

idx_shock_exact = int(np.argmax(np.abs(np.diff(p_e))))
idx_umax_num = int(np.argmax(np.abs(u_n)))
print(f"u_max numerical = {u_n[idx_umax_num]:.2f} at x={x[idx_umax_num]:.3f}")
print(f"u at same index exact = {u_e[idx_umax_num]:.2f}")
print(f"Shock front exact at x={x[idx_shock_exact]:.3f}")
print(f"Near shock: u_e[shock-1]={u_e[idx_shock_exact-1]:.2f}, u_e[shock]={u_e[idx_shock_exact]:.2f}, u_e[shock+1]={u_e[idx_shock_exact+1]:.2f}")
print(f"Near shock: u_n[shock-1]={u_n[idx_shock_exact-1]:.2f}, u_n[shock]={u_n[idx_shock_exact]:.2f}, u_n[shock+1]={u_n[idx_shock_exact+1]:.2f}")
print(f"u_e max (star plateau)={u_e.max():.2f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0,0].plot(x, p_e, 'k-', lw=2, label='Exact')
axes[0,0].plot(x, p_n, 'b.-', markersize=3, label='Numerical')
axes[0,0].set_title('Pressure'); axes[0,0].legend(); axes[0,0].set_yscale('log')
axes[0,1].plot(x, u_e, 'k-', lw=2, label='Exact')
axes[0,1].plot(x, u_n, 'b.-', markersize=3, label='Numerical')
axes[0,1].axhline(390, ls='--', color='g', alpha=0.5, label='u*=390')
axes[0,1].set_title(f'Velocity (u_max={u_n.max():.1f} at x={x[idx_umax_num]:.3f})')
axes[0,1].legend()
axes[1,0].plot(x, rho_e, 'k-', lw=2, label='Exact')
axes[1,0].plot(x, rho_n, 'b.-', markersize=3, label='Numerical')
axes[1,0].set_title('Density'); axes[1,0].legend()
axes[1,1].plot(x, a1_f, 'b.-', markersize=3, label='alpha_air')
axes[1,1].set_title('Alpha'); axes[1,1].legend()
for ax in axes.flat: ax.set_xlabel('x')

plt.suptitle('EB1 Shyue Gas-Water: overshoot location')
plt.tight_layout()
plt.savefig('results/diag_eb1_profile.png', dpi=150)
print(f"Plot: results/diag_eb1_profile.png")
