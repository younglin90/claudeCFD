"""Round 101: 02-A REAL PASS with imex_5n + dt=0.01 fixed."""
import sys, os, time, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim

ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
ph2 = {'gamma': 1.187, 'pinf': 7.028e8, 'kv': 3610.0, 'b': 6.61e-4, 'eta': -1.177788e6}
N = 10; L = 1.0; dx = L/N
x = (np.arange(N)+0.5)*dx
p0 = 1e5; u0 = 1.0; T0 = 300.0
a1_0 = np.where((x>=0.4)&(x<0.6), 1e-6, 1.0-1e-6)
inv_rho2 = (ph2['gamma']-1)*ph2['kv']*T0/(p0+ph2['pinf'])+ph2['b']
rho2 = 1.0/inv_rho2
rho1 = p0/((ph1['gamma']-1)*ph1['kv']*T0)
a1r1 = a1_0*rho1; a2r2 = (1-a1_0)*rho2
rho_init = a1r1 + a2r2
ru = rho_init*u0
e1v = (p0+ph1['gamma']*ph1['pinf'])/(ph1['gamma']-1)
br2 = ph2['b']*rho2
e2v = (p0+ph2['gamma']*ph2['pinf'])*(1-br2)/(ph2['gamma']-1)+rho2*ph2['eta']
rE = a1_0*e1v + (1-a1_0)*e2v + 0.5*rho_init*u0*u0

t0 = time.time()
out = solve_IMEX(ph1, ph2, a1r1.copy(), a2r2.copy(), ru.copy(), rE.copy(), a1_0.copy(),
                 dx=dx, t_end=1.0, dt_fixed=0.01,
                 cfl=0.4, use_material_cfl=False,
                 time_integrator='strang', acoustic_method='imex_5n',
                 primitive_recon='none', alpha_scheme='tvd',
                 max_steps=200, bc_l='periodic', bc_r='periodic',
                 print_interval=99999)
wall = time.time() - t0
tf, a1r1f, a2r2f, ruf, rEf, a1f = out
pf, uf, *_ = cons_to_prim(a1r1f, a2r2f, ruf, rEf, a1f, ph1, ph2)
ep = float(np.max(np.abs((pf-p0)/p0)))
eu = float(np.max(np.abs(uf-u0)))
fin = bool(np.all(np.isfinite(pf)))
PASS = (ep<1e-2) and (eu<1e-2) and fin

# Exact (periodic, full cycle u₀×t_end / L = 1×1/1 = 1 → returns to start)
a1_ex = a1_0
p_ex = np.full_like(x, p0)
u_ex = np.full_like(x, u0)
rho_ex = a1_0*rho1 + (1-a1_0)*rho2

os.makedirs('results/1D/02_A', exist_ok=True)
rho_num = a1r1f + a2r2f
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for i, (lab, num, ex) in enumerate([('p', pf, p_ex), ('u', uf, u_ex), ('rho_mix', rho_num, rho_ex), ('a1', a1f, a1_ex)]):
    axes[0, i].plot(x, num, 'b-o', label='num', markersize=5)
    axes[0, i].plot(x, ex, 'r--', label='exact')
    axes[0, i].set_title(f'{lab} at t={float(tf):.4f}')
    axes[0, i].grid(); axes[0, i].legend()
    axes[1, i].plot(x, np.abs(num-ex), 'k-o', markersize=5)
    axes[1, i].set_title(f'|num-exact| {lab}')
    axes[1, i].grid()
plt.suptitle(f'Round 101 — 02-A NASG (imex_5n + dt=0.01 fixed) — t={float(tf):.3f}, ep={ep:.2e}, eu={eu:.2e}, {"PASS" if PASS else "FAIL"} (wall={wall:.2f}s)')
plt.tight_layout()
plt.savefig('results/1D/02_A/diff_vs_exact.png', dpi=120)
plt.close()
print(f'02-A imex_5n+dt=0.01: ep={ep:.3e} eu={eu:.3e} fin={fin} → {"PASS" if PASS else "FAIL"}')
print(f'  Wall: {wall:.2f}s,  PNG: results/1D/02_A/diff_vs_exact.png')
