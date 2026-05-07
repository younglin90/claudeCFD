"""Test: G_rE limiter sweep for THINC-BVD Phase 2-2."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim

def run_phase2_2(alpha_scheme='tvd', G_rE_limit=None, label='TVD'):
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    N = 200; L = 1.0; dx = L / N
    x = np.linspace(dx/2, L - dx/2, N)
    xd = 0.7

    a_air = np.where(x < xd, 1e-6, 1.0 - 1e-6)
    a_water = 1.0 - a_air
    p_init = np.where(x < xd, 1e9, 1e5)
    rho1 = 50.0 * np.ones(N)
    rho2 = 1000.0 * np.ones(N)

    a1r1 = a_air * rho1
    a2r2 = a_water * rho2
    ru = np.zeros(N)
    gm1, gm2 = ph1['gamma'] - 1.0, ph2['gamma'] - 1.0
    rho_e = a_air * (p_init + ph1['gamma'] * ph1['pinf']) / gm1 + a_water * (p_init + ph2['gamma'] * ph2['pinf']) / gm2
    rE = rho_e + 0.0

    print(f"\n{'='*60}")
    print(f"Phase 2-2: {label} (G_rE_limit={G_rE_limit})")
    print(f"{'='*60}")
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a_air,
        dx, t_end=2.29e-4, cfl=0.25, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=200,
        alpha_scheme=alpha_scheme, use_strang=True,
        use_defect_correction=False, use_material_cfl=False,
        G_rE_limit=G_rE_limit)

    p_f, u_f, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    u_max = np.max(np.abs(u_f))
    rho_f = a1r1_f + a2r2_f
    print(f"\n{label}: u_max={u_max:.1f} (ref ~486)")
    passed = t >= 2.29e-4 * 0.99 and 400 < u_max < 600
    print(f">>> {'PASS' if passed else 'FAIL'}")
    return x, p_f, u_f, a1_f, rho_f, u_max

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    # Baseline TVD (no limit)
    x_t, p_t, u_t, a1_t, rho_t, um_t = run_phase2_2('tvd', None, 'TVD-noLimit')

    # BVD with different G_rE_limit values
    results = {}
    for limit in [None, 0.5, 0.2, 0.1, 0.05]:
        lbl = f'BVD-lim{limit}' if limit else 'BVD-noLimit'
        x_b, p_b, u_b, a1_b, rho_b, um_b = run_phase2_2('thinc_bvd', limit, lbl)
        results[limit] = (x_b, p_b, u_b, a1_b, rho_b, um_b)

    # ===== Plot =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {None: 'r', 0.5: 'orange', 0.2: 'green', 0.1: 'cyan', 0.05: 'purple'}

    axes[0,0].plot(x_t, p_t, 'b-', lw=2, label=f'TVD u={um_t:.0f}')
    for lim, (x_b, p_b, u_b, a1_b, rho_b, um_b) in results.items():
        lbl = f'lim={lim}' if lim else 'noLim'
        axes[0,0].plot(x_b, p_b, color=colors[lim], ls='--', label=f'{lbl} u={um_b:.0f}')
    axes[0,0].set_title('Pressure'); axes[0,0].legend(fontsize=8)

    axes[0,1].plot(x_t, u_t, 'b-', lw=2, label='TVD')
    for lim, (x_b, p_b, u_b, a1_b, rho_b, um_b) in results.items():
        lbl = f'lim={lim}' if lim else 'noLim'
        axes[0,1].plot(x_b, u_b, color=colors[lim], ls='--', label=lbl)
    axes[0,1].set_title('Velocity'); axes[0,1].legend(fontsize=8)

    axes[1,0].plot(x_t, a1_t, 'b-', lw=2, label='TVD')
    for lim, (x_b, p_b, u_b, a1_b, rho_b, um_b) in results.items():
        lbl = f'lim={lim}' if lim else 'noLim'
        axes[1,0].plot(x_b, a1_b, color=colors[lim], ls='--', label=lbl)
    axes[1,0].set_title('α_air'); axes[1,0].legend(fontsize=8)

    axes[1,1].plot(x_t, rho_t, 'b-', lw=2, label='TVD')
    for lim, (x_b, p_b, u_b, a1_b, rho_b, um_b) in results.items():
        lbl = f'lim={lim}' if lim else 'noLim'
        axes[1,1].plot(x_b, rho_b, color=colors[lim], ls='--', label=lbl)
    axes[1,1].set_title('Density'); axes[1,1].legend(fontsize=8)

    plt.suptitle('Phase 2-2: G_rE limiter sweep (THINC-BVD)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/phase2_2_GrE_limit.png', dpi=150)
    print(f"\nPlot saved: results/phase2_2_GrE_limit.png")

    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY:")
    print(f"{'='*50}")
    print(f"TVD (no limit):     u_max={um_t:.1f}")
    for lim, (_, _, _, _, _, um_b) in results.items():
        lbl = f'lim={lim}' if lim else 'noLimit'
        print(f"BVD {lbl:>12s}:  u_max={um_b:.1f}")
