"""
Test Phase 2-1 with paper-exact MMACM-Ex implementation.
Compares TVD-only vs MMACM-Ex for N=200 and N=800.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from solver.He2024.explicit_mmacm_ex import run_phase2_1, cons_to_prim, _EPS

os.makedirs('results', exist_ok=True)

# --- Run Phase 2-1 with both methods ---
configs = [
    {'N': 200, 'use_mmacm_ex': False, 'label': 'N=200 TVD-only'},
    {'N': 200, 'use_mmacm_ex': True,  'label': 'N=200 MMACM-Ex'},
    {'N': 800, 'use_mmacm_ex': False, 'label': 'N=800 TVD-only'},
    {'N': 800, 'use_mmacm_ex': True,  'label': 'N=800 MMACM-Ex'},
]

results = []
for cfg in configs:
    print(f"\n{'='*60}")
    print(f"Running: {cfg['label']}")
    print(f"{'='*60}")
    x, t_final, a1r1, a2r2, ru, rE, a1, ph1, ph2 = run_phase2_1(
        N=cfg['N'], cfl=0.4, t_end=8.0e-4,
        use_mmacm_ex=cfg['use_mmacm_ex'],
        print_interval=100)

    p, u_vel, T, rho1, rho2, c1, c2, c_wood = cons_to_prim(
        a1r1, a2r2, ru, rE, a1, ph1, ph2)
    rho = a1r1 + a2r2
    mach = np.abs(u_vel) / np.maximum(c_wood, _EPS)

    results.append({
        'cfg': cfg,
        'x': x, 't': t_final,
        'p': p, 'u': u_vel, 'T': T, 'rho': rho,
        'a1': a1, 'mach': mach, 'c_wood': c_wood,
        'rho1': rho1, 'rho2': rho2,
    })

    print(f"  t_final={t_final:.4e}, u_max={u_vel.max():.1f} m/s, "
          f"p_range=[{p.min():.3e}, {p.max():.3e}]")

# --- Plot comparison ---
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('Phase 2-1: HP Air / LP Water — Paper-Exact MMACM-Ex\n'
             '(Zhao 2025: proper HLLC velocity Eq.25 + F̂^α in J_k Eq.29)',
             fontsize=13, fontweight='bold')

colors = ['C0', 'C1', 'C2', 'C3']
styles = ['--', '-', ':', '-']
widths = [1.0, 1.5, 1.0, 2.0]

panels = [
    (axes[0, 0], 'rho', 'Mixture Density (kg/m³)'),
    (axes[0, 1], 'p',   'Pressure (Pa)'),
    (axes[1, 0], 'u',   'Velocity (m/s)'),
    (axes[1, 1], 'mach','Mach Number'),
    (axes[2, 0], 'T',   'Temperature (K)'),
    (axes[2, 1], 'a1',  'Volume Fraction α₁ (Air)'),
]

for ax, key, ylabel in panels:
    for i, res in enumerate(results):
        ax.plot(res['x'], res[key], styles[i], color=colors[i],
                linewidth=widths[i], label=res['cfg']['label'], alpha=0.85)
    ax.set_xlabel('x (m)')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = 'results/phase2_1_paper_exact_comparison.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved: {save_path}")

# --- Also save individual N=200 MMACM-Ex plot with impedance ---
res200 = results[1]  # N=200 MMACM-Ex
Z = res200['rho'] * res200['c_wood']

fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
fig2.suptitle(f'Phase 2-1: Paper-Exact MMACM-Ex  N=200  t={res200["t"]:.4e} s\n'
              f'(Zhao 2025 Eqs.25,26,27,29,30,32 — all paper-exact)', fontsize=12)

panels2 = [
    (axes2[0, 0], res200['rho'],  'Density (kg/m³)'),
    (axes2[0, 1], res200['p'],    'Pressure (Pa)'),
    (axes2[0, 2], res200['u'],    'Velocity (m/s)'),
    (axes2[1, 0], res200['mach'], 'Mach Number'),
    (axes2[1, 1], Z,              'Impedance (kg/m²/s)'),
    (axes2[1, 2], res200['a1'],   'α₁ (Air)'),
]

for ax, data, ylabel in panels2:
    ax.plot(res200['x'], data, 'b-', lw=1.2)
    ax.set_xlabel('x (m)')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path2 = 'results/phase2_1_paper_exact_N200.png'
plt.savefig(save_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot saved: {save_path2}")

# --- Summary ---
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for res in results:
    print(f"  {res['cfg']['label']:25s}: steps→t={res['t']:.4e}  "
          f"u_max={res['u'].max():.1f}  p_min={res['p'].min():.2e}  "
          f"a1=[{res['a1'].min():.6f},{res['a1'].max():.6f}]")
