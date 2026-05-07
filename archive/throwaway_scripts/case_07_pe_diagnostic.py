"""User review §18 PE diagnostic: sharp Air-Water interface advection,
verify max|p(x,t) - p₀| stays small without APEC.
This is the critical check: if PE fails, energy flux needs APEC or quasi-cons correction.
"""
import sys, warnings, os
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX
os.makedirs('results/case_07', exist_ok=True)

# Sharp Air-Water interface, p₀ = 1e5, u₀ = 1.0 m/s, advect interface
# With APEC OFF, expect max|p-p₀| < 1e-2 (PE preserving)
N = 200; L = 1.0; dx = L/N
x = (np.arange(N)+0.5)*dx
ph1 = {'gamma':1.400, 'pinf':0.0,    'kv':717.5, 'rho':1.157, 'c':347.8}     # air
ph2 = {'gamma':4.100, 'pinf':4.4e8,  'kv':474.2, 'rho':998.0, 'c':1344.6}    # water
x_intf = 0.4
a1 = np.where(x < x_intf, 1.0-1e-6, 1e-6)  # air left, water right
p0 = 1e5; u0 = 1.0
rho1 = ph1['rho']; rho2 = ph2['rho']
a1r1 = a1*rho1; a2r2 = (1.0-a1)*rho2
rho = a1r1 + a2r2; ru = rho*u0
e1 = (p0+ph1['gamma']*ph1['pinf'])/(ph1['gamma']-1)
e2 = (p0+ph2['gamma']*ph2['pinf'])/(ph2['gamma']-1)
rE = a1*e1 + (1-a1)*e2 + 0.5*rho*u0*u0

# Run with default (no APEC by user spec, im1 acoustic, SSP2 + Richardson)
t_end = 0.05  # advect ~50× dx
out = solve_IMEX(ph1, ph2, a1r1, a2r2, ru, rE, a1, dx=dx, t_end=t_end,
                 cfl=0.4, max_steps=5000,
                 bc_l='periodic', bc_r='periodic',
                 time_integrator='ssp222',
                 use_apec=False,  # explicit user spec
                 print_interval=99999)
t_f, a1r1f, a2r2f, ruf, rEf, a1f = out
rho_f = a1r1f + a2r2f; u_f = ruf/rho_f; ke = 0.5*ruf*u_f
Pi = a1f*ph1['gamma']*ph1['pinf']/(ph1['gamma']-1)+(1-a1f)*ph2['gamma']*ph2['pinf']/(ph2['gamma']-1)
Gi = a1f/(ph1['gamma']-1)+(1-a1f)/(ph2['gamma']-1)
p_f = (rEf - ke - Pi)/Gi

err_p = np.max(np.abs(p_f - p0))
err_u = np.max(np.abs(u_f - u0))
print(f"\n=== PE Diagnostic (sharp Air-Water interface advection) ===")
print(f"  t_end = {float(t_f):.3e} s ({float(t_f)/(dx/u0):.1f} cells advected)")
print(f"  max|p - p₀| = {err_p:.3e} Pa  (target < 1e-2 for PE preservation)")
print(f"  max|u - u₀| = {err_u:.3e} m/s  (target < 1e-6)")
print(f"  PE pass: {err_p < 1e-2}")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0,0].plot(x, a1f, 'b-'); axes[0,0].set_title('α₁ (air vol frac)')
axes[0,1].plot(x, u_f - u0, 'b-'); axes[0,1].set_title(f'u - u₀ (max={err_u:.2e})')
axes[1,0].plot(x, p_f - p0, 'b-'); axes[1,0].set_title(f'p - p₀ (max={err_p:.2e} Pa)')
axes[1,1].plot(x, rho_f, 'b-'); axes[1,1].set_title('ρ_mix')
plt.tight_layout()
plt.savefig('results/case_07/pe_diagnostic.png', dpi=120); plt.close()
print('  PNG: results/case_07/pe_diagnostic.png')
