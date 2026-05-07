"""
Test Phase 2-2: HP Water / LP Air shock tube.
Validation: phase2_high_p_water_low_p_air_shock_tube.md

Domain [0,1]m, Water(left, x<0.7, p=1GPa) / Air(right, x>=0.7, p=1e5)
Water EOS: SG gamma=4.4, Pinf=6e8
Air EOS: Ideal gamma=1.4
N=100, CFL=0.25, t_end=2.29e-4 s
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from solver.He2024.explicit_mmacm_ex import (
    solve, cons_to_prim, _sg_density_from_pT, _sg_internal_energy, _EPS
)

os.makedirs('results', exist_ok=True)


def run_phase2_2(N=100, cfl=0.25, t_end=2.29e-4, use_mmacm_ex=True,
                 print_interval=50):
    """Run Phase 2-2: HP Water / LP Air shock tube (Yoo & Sung 2018).

    Domain: [0, 1] m, interface at x=0.7
    Water(left, x<0.7): SG gamma=4.4, Pinf=6e8, p=1e9 Pa, rho2=1000
    Air(right, x>=0.7): Ideal gamma=1.4, p=1e5 Pa, rho1=50

    Density directly specified (paper IC), NOT from (p,T):
      rho1(air)=50 kg/m3, rho2(water)=1000 kg/m3 everywhere.
      T is implied: T_water_left=992K, T_air_right=6.97K
    """
    # Phase 1 = Air (Ideal Gas)
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    # Phase 2 = Water (Stiffened Gas)
    ph2 = {'gamma': 4.4, 'pinf': 6.0e8, 'kv': 474.2}

    L = 1.0
    dx = L / N
    x = np.linspace(0.5 * dx, L - 0.5 * dx, N)

    x_intf = 0.7
    p_L = 1.0e9   # Water (left) — 1 GPa
    p_R = 1.0e5   # Air (right) — 100 kPa
    u0 = 0.0

    g1, pinf1, kv1 = ph1['gamma'], ph1['pinf'], ph1['kv']
    g2, pinf2, kv2 = ph2['gamma'], ph2['pinf'], ph2['kv']

    # Volume fraction: α₁(air) = 10⁻⁶ on left (paper), 1-10⁻⁶ on right
    eps_pure = 1e-6
    a1 = np.where(x < x_intf, eps_pure, 1.0 - eps_pure)
    a2 = 1.0 - a1

    # Pressure field
    p_field = np.where(x < x_intf, p_L, p_R)

    # Phase densities: DIRECTLY SPECIFIED (paper Yoo & Sung 2018, Section 4.1)
    rho1 = np.full_like(x, 50.0)    # air: 50 kg/m³ everywhere
    rho2 = np.full_like(x, 1000.0)  # water: 1000 kg/m³ everywhere

    # Implied temperatures (for reference):
    #   T_air_left  = (1e9)/(0.4*717.5*50) = 69686 K
    #   T_air_right = (1e5)/(0.4*717.5*50) = 6.97 K
    #   T_water_left  = (1e9+6e8)/(3.4*474.2*1000) = 992 K
    #   T_water_right = (1e5+6e8)/(3.4*474.2*1000) = 372 K

    # Conservative variables
    a1r1 = a1 * rho1
    a2r2 = a2 * rho2
    rho = a1r1 + a2r2
    ru = rho * u0

    # Energy from EOS: e_k = (p + γ_k P∞_k) / ((γ_k-1) ρ_k)
    e1 = _sg_internal_energy(p_field, rho1, g1, pinf1)
    e2 = _sg_internal_energy(p_field, rho2, g2, pinf2)
    rho_e = a1 * rho1 * e1 + a2 * rho2 * e2
    rE = rho_e + 0.5 * rho * u0 ** 2

    print(f"Phase 2-2: HP Water / LP Air shock tube (Yoo & Sung 2018 IC)")
    print(f"  N={N}, dx={dx:.4f} m, CFL={cfl}, t_end={t_end:.2e} s")
    print(f"  Air: gamma={g1}, Pinf={pinf1}, kv={kv1}")
    print(f"  Water: gamma={g2}, Pinf={pinf2}, kv={kv2}")
    print(f"  p_L={p_L:.2e} Pa (Water), p_R={p_R:.2e} Pa (Air)")
    print(f"  rho1(air)={rho1[0]:.1f}, rho2(water)={rho2[0]:.1f} (directly specified)")
    print(f"  MMACM-Ex: {use_mmacm_ex}")

    t_final, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve(
        ph1, ph2, a1r1, a2r2, ru, rE, a1,
        dx, t_end, cfl=cfl,
        bc_l='transmissive', bc_r='transmissive',
        use_mmacm_ex=use_mmacm_ex,
        print_interval=print_interval)

    return x, t_final, a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2


# --- Run mesh convergence study: N=100, 200, 400 with MMACM-Ex ---
configs = [
    {'N': 100, 'use_mmacm_ex': True,  'label': 'N=100 MMACM-Ex'},
    {'N': 200, 'use_mmacm_ex': True,  'label': 'N=200 MMACM-Ex'},
    {'N': 400, 'use_mmacm_ex': True,  'label': 'N=400 MMACM-Ex'},
]

results = []
for cfg in configs:
    print(f"\n{'='*60}")
    print(f"Running: {cfg['label']}")
    print(f"{'='*60}")
    x, t_final, a1r1, a2r2, ru, rE, a1, ph1, ph2 = run_phase2_2(
        N=cfg['N'], cfl=0.25, t_end=2.29e-4,
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
        'a1': a1, 'mach': mach,
    })

    print(f"  t_final={t_final:.4e}, u_max={u_vel.max():.1f} m/s, "
          f"p_range=[{p.min():.3e}, {p.max():.3e}]")


# --- Plot comparison ---
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle(f'Phase 2-2: HP Water / LP Air — Mesh Convergence\n'
             f'CFL=0.25, t_end=2.29e-4 s, MMACM-Ex + THINC-BVD\n'
             f'Water: SG (gamma=4.4, Pinf=6e8), Air: Ideal (gamma=1.4)',
             fontsize=12, fontweight='bold')

colors = ['C0', 'C1', 'C2']
styles = ['--', '-.', '-']
widths = [1.0, 1.2, 1.5]

panels = [
    (axes[0, 0], 'rho', 'Mixture Density (kg/m3)'),
    (axes[0, 1], 'p',   'Pressure (Pa)'),
    (axes[1, 0], 'u',   'Velocity (m/s)'),
    (axes[1, 1], 'mach','Mach Number'),
    (axes[2, 0], 'T',   'Temperature (K)'),
    (axes[2, 1], 'a1',  'Volume Fraction alpha1 (Air)'),
]

for ax, key, ylabel in panels:
    for i, res in enumerate(results):
        ax.plot(res['x'], res[key], styles[i], color=colors[i],
                linewidth=widths[i], label=res['cfg']['label'], alpha=0.85)
    ax.set_xlabel('x (m)')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = 'results/phase2_2_hp_water_lp_air.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved: {save_path}")

# --- Summary ---
print(f"\n{'='*60}")
print("SUMMARY — Phase 2-2")
print(f"{'='*60}")
for res in results:
    print(f"  {res['cfg']['label']:15s}: t={res['t']:.4e}  "
          f"u_max={res['u'].max():.1f}  p_min={res['p'].min():.2e}  "
          f"a1=[{res['a1'].min():.6f},{res['a1'].max():.6f}]")
