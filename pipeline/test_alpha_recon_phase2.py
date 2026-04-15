"""
Phase 2-1 & Phase 2-2: Compare α reconstruction schemes in explicit MMACM-Ex.
Schemes: TVD, THINC-BVD, THINC, CICSAM, MSTACS, STACS, SAISH
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from solver.He2024.explicit_mmacm_ex import (
    run_phase2_1, cons_to_prim, _sg_density_from_pT, _sg_internal_energy, _EPS, solve
)

os.makedirs('results', exist_ok=True)

SCHEMES = [
    ('TVD',           'tvd',       'gray',   '--', 1.0),
    ('THINC-BVD',     'thinc_bvd', 'red',    '-',  1.5),
    ('THINC(no BVD)', 'thinc',     'blue',   '-',  1.2),
    ('CICSAM',        'cicsam',    'green',  '-',  1.5),
    ('MSTACS',        'mstacs',    'purple', '-',  1.5),
    ('STACS',         'superbee',  'orange', '-',  1.2),
    ('SAISH',         'saish',     'brown',  '-',  1.2),
]

# ======================================================================
# Phase 2-1: HP Air / LP Water  [0,2]m, N=200, CFL=0.4, t_end=8e-4
# ======================================================================
print("=" * 80)
print("Phase 2-1: HP Air / LP Water Shock Tube (N=200, MMACM-Ex)")
print("=" * 80)

results_21 = {}
for label, recon, color, ls, lw in SCHEMES:
    t0 = time.time()
    print(f"\n  [{recon}] Running...", end=' ', flush=True)
    try:
        x, t_f, ar1, ar2, ru, rE, a1, ph1, ph2 = run_phase2_1(
            N=200, cfl=0.4, t_end=8.0e-4, use_mmacm_ex=True,
            print_interval=9999, alpha_recon=recon)
        elapsed = time.time() - t0
        p, u, T, rho1, rho2, *_ = cons_to_prim(ar1, ar2, ru, rE, a1, ph1, ph2)
        rho = ar1 + ar2
        results_21[recon] = dict(x=x, p=p, u=u, rho=rho, a1=a1, t=t_f, ok=True)
        print(f"t={t_f:.2e}, u_max={np.abs(u).max():.0f}, "
              f"a1_min={a1.min():.2e}, {elapsed:.1f}s")
    except Exception as e:
        results_21[recon] = dict(ok=False)
        print(f"FAILED: {e}")

# Phase 2-1 summary
print(f"\n{'='*80}")
print(f"Phase 2-1 Results (N=200, t_end=8e-4)")
print(f"{'='*80}")
print(f"{'Scheme':<16s} {'u_max':>8s} {'a1_min':>10s} {'a1 width':>10s} {'PASS?':>6s}")
print(f"{'-'*80}")
for label, recon, *_ in SCHEMES:
    r = results_21[recon]
    if not r['ok']:
        print(f"{label:<16s} {'FAIL':>8s}")
        continue
    umax = np.abs(r['u']).max()
    amin = r['a1'].min()
    # Interface width: count cells where 0.01 < a1 < 0.99
    n_intf = np.sum((r['a1'] > 0.01) & (r['a1'] < 0.99))
    v = 'PASS' if 200 < umax < 300 else 'FAIL'
    print(f"{label:<16s} {umax:>8.0f} {amin:>10.3e} {n_intf:>10d} {v:>6s}")
print(f"{'='*80}")

# Phase 2-1 plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Phase 2-1: HP Air / LP Water (N=200, MMACM-Ex, t=8e-4 s)',
             fontsize=13, fontweight='bold')
panels = [(axes[0,0], 'p', 'Pressure (Pa)'), (axes[0,1], 'u', 'Velocity (m/s)'),
          (axes[1,0], 'rho', 'Density (kg/m³)'), (axes[1,1], 'a1', 'α₁ (Air)')]
for ax, key, ylabel in panels:
    for label, recon, color, ls, lw in SCHEMES:
        r = results_21[recon]
        if not r['ok']: continue
        ax.plot(r['x'], r[key], color=color, ls=ls, lw=lw, label=label)
    ax.set_xlabel('x (m)'); ax.set_ylabel(ylabel)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/alpha_recon_phase2_1.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved: results/alpha_recon_phase2_1.png")

# ======================================================================
# Phase 2-2: HP Water / LP Air  [0,1]m, N=200, CFL=0.25, t_end=2.29e-4
# ======================================================================
print(f"\n{'='*80}")
print("Phase 2-2: HP Water / LP Air Shock Tube (N=200, MMACM-Ex)")
print("=" * 80)

# Phase 2-2 setup (Yoo & Sung 2018)
ph1_22 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}       # Air
ph2_22 = {'gamma': 4.4, 'pinf': 6.0e8, 'kv': 474.2}     # Water SG
N22 = 200; L22 = 1.0; dx22 = L22 / N22
x22 = np.linspace(0.5*dx22, L22 - 0.5*dx22, N22)
x_intf = 0.7

# Left (Water): p=1e9, ρ₁=50, ρ₂=1000, α_air=1e-6
# Right (Air):  p=1e5, ρ₁=50, ρ₂=1000, α_air=1-1e-6
rho1_val = 50.0; rho2_val = 1000.0
p_L = 1.0e9; p_R = 1.0e5
eps_a = 1e-6

a1_22 = np.where(x22 < x_intf, eps_a, 1.0 - eps_a)
a2_22 = 1.0 - a1_22
rho1_22 = np.full(N22, rho1_val)
rho2_22 = np.full(N22, rho2_val)
a1r1_22 = a1_22 * rho1_22; a2r2_22 = a2_22 * rho2_22
rho_22 = a1r1_22 + a2r2_22
ru_22 = np.zeros(N22)

g1,pinf1 = ph1_22['gamma'], ph1_22['pinf']
g2,pinf2 = ph2_22['gamma'], ph2_22['pinf']
gm1, gm2 = g1-1, g2-1
p_arr = np.where(x22 < x_intf, p_L, p_R)
e1 = (p_arr + g1*pinf1) / (gm1 * rho1_val)
e2 = (p_arr + g2*pinf2) / (gm2 * rho2_val)
rE_22 = a1_22*rho1_val*e1 + a2_22*rho2_val*e2

results_22 = {}
for label, recon, color, ls, lw in SCHEMES:
    t0 = time.time()
    print(f"\n  [{recon}] Running...", end=' ', flush=True)
    try:
        t_f, ar1, ar2, ru_f, rE_f, a1_f = solve(
            ph1_22, ph2_22, a1r1_22.copy(), a2r2_22.copy(),
            ru_22.copy(), rE_22.copy(), a1_22.copy(),
            dx22, t_end=2.29e-4, cfl=0.25,
            bc_l='transmissive', bc_r='transmissive',
            use_mmacm_ex=True, print_interval=9999,
            alpha_recon=recon)
        elapsed = time.time() - t0
        p, u, T, rho1, rho2, *_ = cons_to_prim(ar1, ar2, ru_f, rE_f, a1_f,
                                                  ph1_22, ph2_22)
        rho = ar1 + ar2
        results_22[recon] = dict(x=x22, p=p, u=u, rho=rho, a1=a1_f, t=t_f, ok=True)
        print(f"t={t_f:.2e}, u_max={np.abs(u).max():.0f}, "
              f"a1_min={a1_f.min():.2e}, {elapsed:.1f}s")
    except Exception as e:
        results_22[recon] = dict(ok=False)
        print(f"FAILED: {e}")

# Phase 2-2 summary
print(f"\n{'='*80}")
print(f"Phase 2-2 Results (N={N22}, t_end=2.29e-4)")
print(f"{'='*80}")
print(f"{'Scheme':<16s} {'u_max':>8s} {'a1_min':>10s} {'ρ_peak':>10s} {'PASS?':>6s}")
print(f"{'-'*80}")
for label, recon, *_ in SCHEMES:
    r = results_22[recon]
    if not r['ok']:
        print(f"{label:<16s} {'FAIL':>8s}")
        continue
    umax = np.abs(r['u']).max()
    amin = r['a1'].min()
    rho_peak = r['rho'].max()
    v = 'PASS' if 400 < umax < 600 else 'FAIL'
    print(f"{label:<16s} {umax:>8.0f} {amin:>10.3e} {rho_peak:>10.1f} {v:>6s}")
print(f"{'='*80}")

# Phase 2-2 plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Phase 2-2: HP Water / LP Air (N={N22}, MMACM-Ex, t=2.29e-4 s)',
             fontsize=13, fontweight='bold')
panels = [(axes[0,0], 'p', 'Pressure (Pa)'), (axes[0,1], 'u', 'Velocity (m/s)'),
          (axes[1,0], 'rho', 'Density (kg/m³)'), (axes[1,1], 'a1', 'α₁ (Air)')]
for ax, key, ylabel in panels:
    for label, recon, color, ls, lw in SCHEMES:
        r = results_22[recon]
        if not r['ok']: continue
        ax.plot(r['x'], r[key], color=color, ls=ls, lw=lw, label=label)
    ax.set_xlabel('x (m)'); ax.set_ylabel(ylabel)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/alpha_recon_phase2_2.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved: results/alpha_recon_phase2_2.png")
