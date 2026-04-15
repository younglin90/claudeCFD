"""
Phase 1: 1D Water-Air Advection (Abgrall test)
N=10/20/40, t_end=1.0 s, u₀=1 m/s, p₀=1e5 Pa, T₀=300 K
Water at [0.4, 0.6], Air elsewhere, periodic BC
Full segregated solver (autograd): p/u + α sharpness
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from solver.He2024.explicit_mmacm_ex import (
    solve_segregated, cons_to_prim,
    _sg_density_from_pT, _sg_internal_energy, _EPS,
)

os.makedirs('results', exist_ok=True)

# === EOS ===
ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
ph2 = {'gamma': 4.4, 'pinf': 6.0e8, 'kv': 474.2}
g1, pinf1, kv1 = ph1['gamma'], ph1['pinf'], ph1['kv']
g2, pinf2, kv2 = ph2['gamma'], ph2['pinf'], ph2['kv']
T0, p0, u0 = 300.0, 1.0e5, 1.0
eps_pure = 1e-8

r1v = _sg_density_from_pT(p0, T0, g1, pinf1, kv1)
r2v = _sg_density_from_pT(p0, T0, g2, pinf2, kv2)
e1v = _sg_internal_energy(p0, r1v, g1, pinf1)
e2v = _sg_internal_energy(p0, r2v, g2, pinf2)


def make_ic(N):
    dx = 1.0 / N
    x = np.linspace(0.5*dx, 1-0.5*dx, N)
    a1 = np.where((x >= 0.4) & (x <= 0.6), eps_pure, 1-eps_pure)
    a2 = 1-a1
    rho1 = np.full(N, r1v); rho2 = np.full(N, r2v)
    a1r1 = a1*rho1; a2r2 = a2*rho2
    rho = a1r1+a2r2; ru = rho*u0
    rE = a1*rho1*e1v + a2*rho2*e2v + 0.5*rho*u0**2
    return x, dx, a1r1, a2r2, ru, rE, a1


def run_one(N, scheme, beta=2.0):
    x, dx, a1r1, a2r2, ru, rE, a1 = make_ic(N)
    dt = 0.01
    steps = 100
    _, ar1, ar2, ru_f, rE_f, a1_f = solve_segregated(
        ph1, ph2, a1r1.copy(), a2r2.copy(), ru.copy(), rE.copy(), a1.copy(),
        dx, t_end=1.0, dt=dt,
        bc_l='periodic', bc_r='periodic',
        max_steps=steps, max_newton=20, newton_tol=1e-10,
        print_interval=50,
        thinc_beta=beta, alpha_scheme=scheme,
        jacobian_method='autograd')
    p_f, u_f, *_ = cons_to_prim(ar1, ar2, ru_f, rE_f, a1_f, ph1, ph2)
    ep = np.sqrt(np.mean(((p_f - p0) / p0)**2))
    eu = np.sqrt(np.mean(((u_f - u0) / u0)**2))
    return x, ar1, ar2, ru_f, rE_f, a1_f, p_f, u_f, ep, eu


# === Run all cases ===
results = {}
for N in [10, 20]:
    for scheme in ['thinc_bvd', 'thinc', 'cicsam']:
        label = f"N={N}_{scheme}"
        print(f"\n{'='*60}")
        print(f"  {label}: dt=0.01, 100 steps, autograd")
        print(f"{'='*60}")
        x, ar1, ar2, ru_f, rE_f, a1_f, p_f, u_f, ep, eu = run_one(N, scheme)
        results[label] = dict(N=N, scheme=scheme, x=x,
                              a1=a1_f, p=p_f, u=u_f, ep=ep, eu=eu)
        v = 'PASS' if ep < 1e-2 and eu < 1e-2 else 'FAIL'
        print(f"  → err_p={ep:.3e}  err_u={eu:.3e}  α_min={a1_f.min():.3e}  {v}")

# === Summary table ===
print(f"\n{'='*84}")
print(f"Phase 1 Results: Segregated Solver (autograd, dt=0.01, 100 steps)")
print(f"PASS: err_p < 1e-2 AND err_u < 1e-2")
print(f"{'='*84}")
print(f"{'N':>4s} {'Scheme':<16s} {'err_p':>10s} {'err_u':>10s} {'α_min':>12s} {'PASS?':>6s}")
print(f"{'-'*84}")
for key, r in results.items():
    v = 'PASS' if r['ep'] < 1e-2 and r['eu'] < 1e-2 else 'FAIL'
    print(f"{r['N']:>4d} {r['scheme']:<16s} {r['ep']:>10.3e} {r['eu']:>10.3e} "
          f"{r['a1'].min():>12.4e} {v:>6s}")
print(f"{'='*84}")

# === Plot ===
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Phase 1: Segregated Solver — THINC vs THINC-BVD vs CICSAM\n'
             'dt=0.01, 100 steps, autograd, periodic BC',
             fontsize=13, fontweight='bold')

colors = {'thinc_bvd': ('r', 'o', 'THINC-BVD'),
           'thinc':     ('b', '^', 'THINC(no BVD)'),
           'cicsam':    ('g', 's', 'CICSAM')}

for col, N in enumerate([10, 20]):
    x_ic, _, _, _, _, _, a1_ic = make_ic(N)

    # Row 0: α profiles
    ax = axes[0, col]
    ax.plot(x_ic, a1_ic, 'k--', lw=1, alpha=0.4, label='Initial')
    for sch, (c, m, lbl) in colors.items():
        r = results[f'N={N}_{sch}']
        ax.plot(r['x'], r['a1'], f'{c}-{m}', lw=1.5, ms=3,
                label=f"{lbl} (α_min={r['a1'].min():.1e})")
    ax.set_xlabel('x'); ax.set_ylabel('α₁ (Air)')
    ax.set_title(f'N={N}: Volume Fraction α₁')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Row 1: Pressure
    ax = axes[1, col]
    ax.plot(x_ic, np.full(N, p0), 'k--', lw=1, alpha=0.4, label='Exact')
    for sch, (c, m, lbl) in colors.items():
        r = results[f'N={N}_{sch}']
        ax.plot(r['x'], r['p'], f'{c}-{m}', lw=1.5, ms=3,
                label=f"{lbl} (err={r['ep']:.1e})")
    ax.set_xlabel('x'); ax.set_ylabel('Pressure (Pa)')
    ax.set_title(f'N={N}: Pressure')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = 'results/phase1_cicsam_comparison.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved: {save_path}")
