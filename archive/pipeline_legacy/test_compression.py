"""Test: Compression sharpening (no THINC-BVD) for Phase 2-2.
TVD α + Compression + MMACM-Ex G corrections.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim

def run_phase2_2(alpha_scheme='tvd', use_compression=False, C_alpha=1.0, label='TVD'):
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
    print(f"Phase 2-2: {label}")
    print(f"{'='*60}")
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a_air,
        dx, t_end=2.29e-4, cfl=0.25, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=200,
        alpha_scheme=alpha_scheme, use_strang=True,
        use_defect_correction=False, use_material_cfl=False,
        use_compression=use_compression, C_alpha=C_alpha)

    p_f, u_f, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    u_max = np.max(np.abs(u_f))
    rho_f = a1r1_f + a2r2_f
    print(f"\n{label}: u_max={u_max:.1f} (ref ~486)")
    passed = t >= 2.29e-4 * 0.99 and 400 < u_max < 600
    print(f">>> {'PASS' if passed else 'FAIL'}")
    return x, p_f, u_f, a1_f, rho_f, u_max

def run_phase2_1(alpha_scheme='tvd', use_compression=False, C_alpha=1.0, label='TVD'):
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.1, 'pinf': 4.4e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    N = 200; L = 2.0; dx = L / N
    x = np.linspace(dx/2, L - dx/2, N)
    xd = 0.5
    p_L, p_R = 1e9, 1e4
    rho1_L = p_L / ((ph1['gamma'] - 1.0) * ph1['kv'] * 300.0)
    rho1_R = p_R / ((ph1['gamma'] - 1.0) * ph1['kv'] * 300.0)
    rho2_L = (p_L + ph2['pinf']) / ((ph2['gamma'] - 1.0) * ph2['kv'] * 300.0)
    rho2_R = (p_R + ph2['pinf']) / ((ph2['gamma'] - 1.0) * ph2['kv'] * 300.0)

    a_air = np.where(x < xd, 1.0 - 1e-6, 1e-6)
    a_water = 1.0 - a_air
    p_init = np.where(x < xd, p_L, p_R)
    rho1 = np.where(x < xd, rho1_L, rho1_R)
    rho2 = np.where(x < xd, rho2_L, rho2_R)

    a1r1 = a_air * rho1
    a2r2 = a_water * rho2
    ru = np.zeros(N)
    gm1, gm2 = ph1['gamma'] - 1.0, ph2['gamma'] - 1.0
    rho_e = a_air * (p_init + ph1['gamma'] * ph1['pinf']) / gm1 + a_water * (p_init + ph2['gamma'] * ph2['pinf']) / gm2
    rE = rho_e

    print(f"\n{'='*60}")
    print(f"Phase 2-1: {label}")
    print(f"{'='*60}")
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a_air,
        dx, t_end=8e-4, cfl=0.4, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=200,
        alpha_scheme=alpha_scheme, use_strang=True,
        use_defect_correction=False, use_material_cfl=False,
        use_compression=use_compression, C_alpha=C_alpha)

    p_f, u_f, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    u_max = np.max(np.abs(u_f))
    print(f"\n{label}: u_max={u_max:.1f} (ref ~228)")
    passed = t >= 8e-4 * 0.99 and u_max < 400
    print(f">>> {'PASS' if passed else 'FAIL'}")
    return x, p_f, u_f, a1_f, u_max

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    # Phase 2-1 tests
    x21_t, p21_t, u21_t, a121_t, um21_t = run_phase2_1('tvd', False, 1.0, 'TVD')
    x21_c, p21_c, u21_c, a121_c, um21_c = run_phase2_1('tvd', True, 1.0, 'TVD+Comp1.0')

    # Phase 2-2 tests
    x_t, p_t, u_t, a1_t, rho_t, um_t = run_phase2_2('tvd', False, 1.0, 'TVD only')
    x_b, p_b, u_b, a1_b, rho_b, um_b = run_phase2_2('thinc_bvd', False, 1.0, 'BVD only')

    results_c = {}
    for ca in [0.5, 1.0, 2.0]:
        x_c, p_c, u_c, a1_c, rho_c, um_c = run_phase2_2('tvd', True, ca, f'TVD+Comp{ca}')
        results_c[ca] = (x_c, p_c, u_c, a1_c, rho_c, um_c)

    # ===== Plot =====
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Phase 2-1
    axes[0,0].plot(x21_t, p21_t, 'b-', lw=2, label=f'TVD u={um21_t:.0f}')
    axes[0,0].plot(x21_c, p21_c, 'r--', label=f'TVD+Comp u={um21_c:.0f}')
    axes[0,0].set_title('P2-1 Pressure'); axes[0,0].legend(fontsize=8)
    axes[0,1].plot(x21_t, u21_t, 'b-', lw=2)
    axes[0,1].plot(x21_c, u21_c, 'r--')
    axes[0,1].set_title('P2-1 Velocity')
    axes[0,2].plot(x21_t, a121_t, 'b-', lw=2, label='TVD')
    axes[0,2].plot(x21_c, a121_c, 'r--', label='TVD+Comp')
    axes[0,2].set_title('P2-1 α_air'); axes[0,2].legend(fontsize=8)

    # Summary text
    axes[0,3].axis('off')
    txt = f'Phase 2-1:\n  TVD:       u={um21_t:.0f}\n  TVD+Comp:  u={um21_c:.0f}\n\n'
    txt += f'Phase 2-2:\n  TVD:       u={um_t:.0f}\n  BVD:       u={um_b:.0f}\n'
    for ca, (_, _, _, _, _, um_c) in results_c.items():
        txt += f'  TVD+C{ca}: u={um_c:.0f}\n'
    axes[0,3].text(0.1, 0.5, txt, fontsize=12, va='center', family='monospace')
    axes[0,3].set_title('Summary')

    # Phase 2-2
    colors_c = {0.5: 'green', 1.0: 'orange', 2.0: 'purple'}
    axes[1,0].plot(x_t, p_t, 'b-', lw=2, label=f'TVD u={um_t:.0f}')
    axes[1,0].plot(x_b, p_b, 'r--', label=f'BVD u={um_b:.0f}')
    for ca, (x_c, p_c, u_c, a1_c, rho_c, um_c) in results_c.items():
        axes[1,0].plot(x_c, p_c, color=colors_c[ca], ls=':', label=f'C{ca} u={um_c:.0f}')
    axes[1,0].set_title('P2-2 Pressure'); axes[1,0].legend(fontsize=7)

    axes[1,1].plot(x_t, u_t, 'b-', lw=2)
    axes[1,1].plot(x_b, u_b, 'r--')
    for ca, (x_c, p_c, u_c, a1_c, rho_c, um_c) in results_c.items():
        axes[1,1].plot(x_c, u_c, color=colors_c[ca], ls=':')
    axes[1,1].set_title('P2-2 Velocity')

    axes[1,2].plot(x_t, a1_t, 'b-', lw=2, label='TVD')
    axes[1,2].plot(x_b, a1_b, 'r--', label='BVD')
    for ca, (x_c, p_c, u_c, a1_c, rho_c, um_c) in results_c.items():
        axes[1,2].plot(x_c, a1_c, color=colors_c[ca], ls=':', label=f'C{ca}')
    axes[1,2].set_title('P2-2 α_air'); axes[1,2].legend(fontsize=8)

    axes[1,3].plot(x_t, rho_t, 'b-', lw=2, label='TVD')
    axes[1,3].plot(x_b, rho_b, 'r--', label='BVD')
    for ca, (x_c, p_c, u_c, a1_c, rho_c, um_c) in results_c.items():
        axes[1,3].plot(x_c, rho_c, color=colors_c[ca], ls=':', label=f'C{ca}')
    axes[1,3].set_title('P2-2 Density'); axes[1,3].legend(fontsize=8)

    plt.suptitle('Compression vs THINC-BVD for interface sharpening (IMEX)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/phase2_2_compression.png', dpi=150)
    print(f"\nPlot saved: results/phase2_2_compression.png")
