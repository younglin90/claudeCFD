"""
Pure α advection comparison: all NVD schemes + THINC variants
N=20, dt=0.01, 250 SSP-RK3 steps, u₀=1, periodic BC
No implicit solver — pure explicit α transport only.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from solver.He2024.explicit_mmacm_ex import (
    _thinc_bvd_reconstruct, _nvd_face, _ghost, _EPS
)

os.makedirs('results', exist_ok=True)

N = 20; L = 1.0; dx = L / N; u0 = 1.0
x = np.linspace(0.5*dx, L-0.5*dx, N)
a1_init = np.where((x >= 0.4) & (x <= 0.6), 1e-8, 1-1e-8)

# CFL = 0.4 for explicit stability
dt = 0.4 * dx / abs(u0)
n_steps = int(np.ceil(1.0 / dt))
dt = 1.0 / n_steps
u_face = np.full(N+1, u0)

print(f"Pure α Advection Comparison (N={N}, dt={dt:.4f}, {n_steps} steps)")
print()

def run_scheme(a1_0, scheme, beta=2.0):
    a1 = a1_0.copy()
    for _ in range(n_steps):
        def _rhs(a):
            if scheme == 'thinc_bvd':
                aL, aR = _thinc_bvd_reconstruct(a, 'periodic', 'periodic',
                                                  beta=beta, use_bvd=True)
                af = np.where(u_face >= 0, aL, aR)
            elif scheme == 'thinc':
                aL, aR = _thinc_bvd_reconstruct(a, 'periodic', 'periodic',
                                                  beta=beta, use_bvd=False)
                af = np.where(u_face >= 0, aL, aR)
            elif scheme == 'upwind':
                ag = _ghost(a, 'periodic', 'periodic')
                af = np.where(u_face >= 0, ag[:-1], ag[1:])
            else:
                cds_map = {'cicsam': 'hyper_c', 'stacs': 'superbee',
                           'mstacs': 'mstacs', 'saish': 'saish'}
                af = _nvd_face(a, u_face, dt, dx, 'periodic', 'periodic',
                               cds=cds_map[scheme])
            return -(af * u_face)[1:] + (af * u_face)[:-1]
        # SSP-RK3
        k1 = _rhs(a1) / dx
        s1 = np.clip(a1 + dt*k1, _EPS, 1-_EPS)
        k2 = _rhs(s1) / dx
        s2 = np.clip(0.75*a1 + 0.25*(s1 + dt*k2), _EPS, 1-_EPS)
        k3 = _rhs(s2) / dx
        a1 = np.clip((1./3)*a1 + (2./3)*(s2 + dt*k3), _EPS, 1-_EPS)
    return a1

schemes = [
    ('1st Upwind',    'upwind',    'gray',  '--'),
    ('THINC-BVD β=2', 'thinc_bvd', 'red',   '-o'),
    ('THINC (no BVD)', 'thinc',    'blue',  '-^'),
    ('CICSAM (Hyper-C)', 'cicsam', 'green', '-s'),
    ('STACS (SUPERBEE)', 'stacs',  'orange', '-D'),
    ('MSTACS',        'mstacs',    'purple', '-v'),
    ('SAISH (Bd.Down)', 'saish',  'brown',  '-P'),
]

results = {}
for label, sch, _, _ in schemes:
    print(f"  Running {label}...")
    a1_final = run_scheme(a1_init, sch)
    err_L2 = np.sqrt(np.mean((a1_final - a1_init)**2))
    results[sch] = (a1_final, err_L2)
    print(f"    α_min={a1_final.min():.3e}, L2={err_L2:.3e}")

# Table
print(f"\n{'='*70}")
print(f"α Advection Results (N={N}, t=1.0 s, Co={u0*dt/dx:.2f})")
print(f"{'='*70}")
print(f"{'Scheme':<22s} {'α_min':>10s} {'L2(α)':>10s} {'Sharp?':>8s}")
print(f"{'-'*70}")
for label, sch, _, _ in schemes:
    a1f, eL2 = results[sch]
    sharp = 'SHARP' if a1f.min() < 1e-3 else f'{a1f.min():.2e}'
    print(f"{label:<22s} {a1f.min():>10.3e} {eL2:>10.3e} {sharp:>8s}")
print(f"{'='*70}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'α Advection: NVD & THINC Schemes (N={N}, Co={u0*dt/dx:.2f}, {n_steps} steps)',
             fontsize=13, fontweight='bold')

ax = axes[0]
ax.plot(x, a1_init, 'k--', lw=2, label='Exact', alpha=0.5)
for label, sch, color, marker in schemes:
    a1f, _ = results[sch]
    ax.plot(x, a1f, marker, color=color, lw=1.2, ms=4, label=label)
ax.set_xlabel('x'); ax.set_ylabel('α₁')
ax.set_title('Full Profile'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

ax = axes[1]
mask = (x > 0.3) & (x < 0.55)
ax.plot(x[mask], a1_init[mask], 'k--', lw=2, alpha=0.5)
for label, sch, color, marker in schemes:
    a1f, _ = results[sch]
    ax.plot(x[mask], a1f[mask], marker, color=color, lw=1.5, ms=5, label=label)
ax.set_xlabel('x'); ax.set_ylabel('α₁')
ax.set_title('Zoom: Left Interface'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = 'results/phase1_all_schemes_comparison.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved: {save_path}")
