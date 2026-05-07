"""Phase 2-3: 1D Cavitation Problem (Yoo & Sung §4.1.3)

Setup:
- Domain [0, 1], N=100, CFL=0.25, t_end=1e-3 s
- u_L=-100, u_R=+100 (opposite velocities)
- p = 1e5 Pa, alpha_air = 0.01 (uniform)
- Water SG: gamma=4.4, Pinf=6e8
- Air: gamma=1.4

Expected: central rarefaction -> cavity forms (alpha_air increases at center)
PASS: t_end completion, cavity visible, no oscillation
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim

def run_cavitation(alpha_scheme='tvd', label='TVD'):
    # EOS: Air (ideal), Water (SG)
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}

    N = 100; L = 1.0; dx = L / N
    x = np.linspace(dx/2, L - dx/2, N)
    xd = 0.5

    # Uniform mixture: 1% air + 99% water everywhere
    a_air = 0.01 * np.ones(N)
    a_water = 1.0 - a_air
    p_init = 1e5 * np.ones(N)
    T_init = 300.0

    # Phase densities at p=1e5, T=300
    rho1 = p_init / ((ph1['gamma'] - 1.0) * ph1['kv'] * T_init)   # air ~0.814
    rho2 = (p_init + ph2['pinf']) / ((ph2['gamma'] - 1.0) * ph2['kv'] * T_init)  # water ~ 937

    # Velocity: left -100, right +100
    u_init = np.where(x < xd, -100.0, 100.0)

    a1r1 = a_air * rho1
    a2r2 = a_water * rho2
    rho = a1r1 + a2r2
    ru = rho * u_init
    gm1, gm2 = ph1['gamma'] - 1.0, ph2['gamma'] - 1.0
    rho_e = a_air * (p_init + ph1['gamma'] * ph1['pinf']) / gm1 + a_water * (p_init + ph2['gamma'] * ph2['pinf']) / gm2
    rE = rho_e + 0.5 * rho * u_init**2

    print(f"Phase 2-3 Cavitation ({label}):")
    print(f"  rho_water_init={rho2[0]:.2f}, rho_air_init={rho1[0]:.4f}")
    print(f"  rho_mix_init={rho[0]:.2f}")

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a_air,
        dx, t_end=1e-3, cfl=0.25, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=500,
        alpha_scheme=alpha_scheme, use_strang=True,
        use_defect_correction=False, use_material_cfl=False)

    p_f, u_f, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    rho_f = a1r1_f + a2r2_f

    # Check cavity formation
    center_idx = N // 2
    a_air_center = a1_f[center_idx]
    a_air_max = a1_f.max()
    p_min = p_f.min()
    rho_min = rho_f.min()
    completed = t >= 1e-3 * 0.99

    print(f"  t_final={t:.3e}")
    print(f"  alpha_air_center={a_air_center:.4f} (init 0.01)")
    print(f"  alpha_air_max={a_air_max:.4f}")
    print(f"  p_min={p_min:.3e} (init 1e5)")
    print(f"  rho_min={rho_min:.2f}")
    print(f"  u_range=[{u_f.min():.1f}, {u_f.max():.1f}]")

    # PASS criteria
    cavity_formed = a_air_max > 0.02  # increased from 0.01
    p_reduced = p_min < 0.5 * 1e5     # pressure drops in cavity
    no_divergence = not (np.any(np.isnan(a1_f)) or np.any(np.isnan(p_f)))
    passed = completed and cavity_formed and p_reduced and no_divergence
    status = "PASS" if passed else "FAIL"
    print(f"  completed={completed}, cavity_formed={cavity_formed}, p_reduced={p_reduced}")
    print(f"  >>> {status}\n")

    return x, p_f, u_f, a1_f, rho_f, passed

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    schemes = [('tvd', 'TVD'), ('thinc_bvd', 'THINC-BVD'),
               ('cicsam', 'CICSAM'), ('mstacs', 'MSTACS')]
    results = {}
    for s, l in schemes:
        results[s] = run_cavitation(s, l)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'tvd': 'blue', 'thinc_bvd': 'red', 'cicsam': 'green', 'mstacs': 'purple'}
    for s, l in schemes:
        x, p, u, a1, rho, _ = results[s]
        c = colors[s]
        axes[0,0].plot(x, p/1e5, color=c, label=l)
        axes[0,1].plot(x, u, color=c, label=l)
        axes[1,0].plot(x, a1, color=c, label=l)
        axes[1,1].plot(x, rho, color=c, label=l)

    axes[0,0].set_title('Pressure / 1e5'); axes[0,0].legend(); axes[0,0].set_xlabel('x')
    axes[0,0].axhline(0, ls='--', color='gray', alpha=0.5)
    axes[0,1].set_title('Velocity'); axes[0,1].legend(); axes[0,1].set_xlabel('x')
    axes[1,0].set_title('alpha_air (cavity)'); axes[1,0].legend(); axes[1,0].set_xlabel('x')
    axes[1,1].set_title('Density'); axes[1,1].legend(); axes[1,1].set_xlabel('x')

    plt.suptitle('Phase 2-3: 1D Cavitation Problem', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/phase2_3_cavitation.png', dpi=150)
    print("Plot saved: results/phase2_3_cavitation.png")

    # Summary
    print("="*50); print("SUMMARY")
    for s, l in schemes:
        _, _, _, _, _, p = results[s]
        print(f"  {l:12s}: {'PASS' if p else 'FAIL'}")
