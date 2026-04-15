"""
Compression term comparison: MMACM-Ex vs Compression (FCT) vs TVD only
Phase 2-1 & Phase 2-2
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from solver.He2024.explicit_mmacm_ex import (
    run_phase2_1, cons_to_prim, solve, _sg_density_from_pT, _sg_internal_energy, _EPS
)

os.makedirs('results', exist_ok=True)

CONFIGS = [
    ('TVD only',        dict(use_mmacm_ex=False, use_compression=False, alpha_recon='tvd'),
     'gray', '--', 1.0),
    ('THINC-BVD only',  dict(use_mmacm_ex=False, use_compression=False),
     'blue', '--', 1.0),
    ('MMACM-Ex',        dict(use_mmacm_ex=True),
     'red', '-', 1.5),
    ('Compress C=1',    dict(use_mmacm_ex=False, use_compression=True, C_alpha=1.0),
     'green', '-', 1.5),
    ('Compress C=2',    dict(use_mmacm_ex=False, use_compression=True, C_alpha=2.0),
     'purple', '-', 1.5),
    ('Compress C=4',    dict(use_mmacm_ex=False, use_compression=True, C_alpha=4.0),
     'orange', '-', 1.2),
]

def run_phase(phase, configs):
    results = {}
    for label, cfg, *_ in configs:
        t0 = time.time()
        print(f"  [{label}]...", end=' ', flush=True)
        try:
            if phase == '2-1':
                x, t_f, ar1, ar2, ru, rE, a1, ph1, ph2 = run_phase2_1(
                    N=200, cfl=0.4, t_end=8e-4, print_interval=9999, **cfg)
            else:
                # Phase 2-2 setup
                ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
                ph2 = {'gamma': 4.4, 'pinf': 6.0e8, 'kv': 474.2}
                N22 = 200; dx22 = 1.0/N22
                x = np.linspace(0.5*dx22, 1-0.5*dx22, N22)
                a1_ic = np.where(x < 0.7, 1e-6, 1-1e-6)
                a2_ic = 1-a1_ic
                rho1v=50.; rho2v=1000.
                a1r1_ic=a1_ic*rho1v; a2r2_ic=a2_ic*rho2v; ru_ic=np.zeros(N22)
                g1,pinf1=1.4,0.; g2,pinf2=4.4,6e8
                p_arr=np.where(x<0.7,1e9,1e5)
                e1=(p_arr+g1*pinf1)/((g1-1)*rho1v)
                e2=(p_arr+g2*pinf2)/((g2-1)*rho2v)
                rE_ic=a1_ic*rho1v*e1+a2_ic*rho2v*e2
                t_f, ar1, ar2, ru, rE, a1 = solve(
                    ph1, ph2, a1r1_ic.copy(), a2r2_ic.copy(), ru_ic.copy(),
                    rE_ic.copy(), a1_ic.copy(), dx22, t_end=2.29e-4, cfl=0.25,
                    bc_l='transmissive', bc_r='transmissive',
                    print_interval=9999, **cfg)
            elapsed = time.time() - t0
            p, u, T, rho1, rho2, *_ = cons_to_prim(ar1, ar2, ru, rE, a1, ph1, ph2)
            rho = ar1 + ar2
            n_intf = np.sum((a1 > 0.01) & (a1 < 0.99))
            results[label] = dict(x=x, p=p, u=u, rho=rho, a1=a1, ok=True)
            print(f"u_max={np.abs(u).max():.0f}, a1_min={a1.min():.2e}, "
                  f"intf={n_intf} cells, {elapsed:.1f}s")
        except Exception as e:
            results[label] = dict(ok=False)
            print(f"FAILED: {e}")
    return results

# === Phase 2-1 ===
print("=" * 70)
print("Phase 2-1: HP Air / LP Water (N=200)")
print("=" * 70)
r21 = run_phase('2-1', CONFIGS)

# === Phase 2-2 ===
print(f"\n{'='*70}")
print("Phase 2-2: HP Water / LP Air (N=200)")
print("=" * 70)
r22 = run_phase('2-2', CONFIGS)

# === Summary ===
for phase_label, results in [("Phase 2-1", r21), ("Phase 2-2", r22)]:
    print(f"\n{'='*70}")
    print(f"{phase_label} Summary")
    print(f"{'='*70}")
    print(f"{'Config':<18s} {'u_max':>8s} {'a1_min':>10s} {'ρ_max':>10s} {'PASS?':>6s}")
    print(f"{'-'*70}")
    for label, *_ in CONFIGS:
        r = results[label]
        if not r['ok']:
            print(f"{label:<18s} {'FAIL':>8s}"); continue
        umax = np.abs(r['u']).max()
        amin = r['a1'].min()
        rmax = r['rho'].max()
        v = 'PASS' if (200<umax<600) else 'FAIL'
        print(f"{label:<18s} {umax:>8.0f} {amin:>10.2e} {rmax:>10.1f} {v:>6s}")

# === Plots ===
for phase_label, results, fname in [
    ("Phase 2-1: HP Air / LP Water", r21, 'results/compression_phase2_1.png'),
    ("Phase 2-2: HP Water / LP Air", r22, 'results/compression_phase2_2.png')]:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{phase_label} (N=200, MMACM-Ex vs Compression)',
                 fontsize=13, fontweight='bold')
    panels = [(axes[0,0],'p','Pressure (Pa)'),(axes[0,1],'u','Velocity (m/s)'),
              (axes[1,0],'rho','Density (kg/m³)'),(axes[1,1],'a1','α₁ (Air)')]
    for ax, key, ylabel in panels:
        for label, _, color, ls, lw in CONFIGS:
            r = results[label]
            if not r['ok']: continue
            ax.plot(r['x'], r[key], color=color, ls=ls, lw=lw, label=label)
        ax.set_xlabel('x (m)'); ax.set_ylabel(ylabel)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {fname}")
