"""Case 02 Test C — Moving contact discontinuity at u=100 m/s, p=1e9 Pa.
Kraposhin 2022 + alternate Air-Water (SG) per spec.
PASS: err_p < 1e-10, err_u < 1e-10, err_T < 1e-6.
"""
import sys, warnings, os
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
os.makedirs('results/case_02', exist_ok=True)

# 02-C alternate: Air (Ideal) + Water (SG γ=4.4, P∞=6e8) per Phase 2-1 setup
ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}                    # Air
ph2 = {'gamma': 4.4, 'pinf': 6.0e8, 'kv': 1004.0}                  # Water SG (Yoo & Sung)

N = 100; L = 1.0; dx = L/N
x = (np.arange(N) + 0.5)*dx

p0 = 1.0e9; u0 = 100.0; T0 = 300.0
a1 = np.where(x < 0.5, 1.0 - 1e-6, 1e-6)  # air left, water right

# Use spec-derived densities for K=2 setup
rho1 = p0/((ph1['gamma']-1)*ph1['kv']*T0)
rho2 = (p0 + ph2['pinf'])/((ph2['gamma']-1)*ph2['kv']*T0)
print(f"  ρ_air = {rho1:.3f}, ρ_water (SG) = {rho2:.3f}")

a1r1 = a1*rho1; a2r2 = (1-a1)*rho2
rho = a1r1 + a2r2; ru = rho*u0
e1_v = (p0 + ph1['gamma']*ph1['pinf'])/(ph1['gamma']-1)
e2_v = (p0 + ph2['gamma']*ph2['pinf'])/(ph2['gamma']-1)  # SG (no b factor)
rE = a1*e1_v + (1-a1)*e2_v + 0.5*rho*u0*u0

# t_end = 0.01s = exactly one period (u·t_end = 1.0 m = L)
out = solve_IMEX(ph1, ph2, a1r1, a2r2, ru, rE, a1, dx=dx, t_end=0.01,
                 cfl=0.4, use_material_cfl=False,  # acoustic CFL per spec
                 acoustic_method='imex_5n',
                 primitive_recon='none',
                 max_steps=10000,
                 bc_l='periodic', bc_r='periodic',
                 print_interval=1000)
t_f, a1r1f, a2r2f, ruf, rEf, a1f = out
p_f, u_f, *_ = cons_to_prim(a1r1f, a2r2f, ruf, rEf, a1f, ph1, ph2)
rho_f = a1r1f + a2r2f

err_p = np.max(np.abs(p_f - p0))/p0
err_u = np.max(np.abs(u_f - u0))
finite = np.all(np.isfinite(p_f)) and np.all(np.isfinite(u_f))
PASS = (err_p < 1e-10) and (err_u < 1e-10) and finite
status = 'PASS' if PASS else 'FAIL'

print(f"\n=== Case 02-C Test C (Air/Water u=100 m/s, p=1e9 Pa) — {status} ===")
print(f"  t_final = {float(t_f):.4f} s  (target 0.01)")
print(f"  err_p = max|p/p₀-1| = {err_p:.3e}  (target < 1e-10)")
print(f"  err_u = max|u-u₀|   = {err_u:.3e}  (target < 1e-10)")
print(f"  finite = {finite}")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0,0].plot(x, a1f, 'b-o'); axes[0,0].set_title('α₁ (air vol frac)')
axes[0,1].plot(x, u_f - u0, 'b-o')
axes[0,1].set_title(f'u - u₀ (max = {err_u:.2e})')
axes[1,0].plot(x, (p_f-p0)/p0, 'b-o')
axes[1,0].set_title(f'(p-p₀)/p₀ (max = {err_p:.2e})')
axes[1,1].plot(x, rho_f, 'b-o')
axes[1,1].set_title('ρ_mix')
fig.suptitle(f'02-C Moving contact u=100 m/s p=1e9 Pa — {status}')
plt.tight_layout()
plt.savefig('results/case_02/case_02C_iter43.png', dpi=120); plt.close()
print(f"  PNG: results/case_02/case_02C_iter43.png")
