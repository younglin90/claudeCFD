"""Diagnose Phase 2-2: Compare α profiles from TVD, BVD, CICSAM, MSTACS."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim

def run(scheme, label):
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    N = 200; L = 1.0; dx = L / N
    x = np.linspace(dx/2, L - dx/2, N)
    xd = 0.7

    a_air = np.where(x < xd, 1e-6, 1.0 - 1e-6)
    p_init = np.where(x < xd, 1e9, 1e5)
    rho1 = 50.0 * np.ones(N); rho2 = 1000.0 * np.ones(N)
    a1r1 = a_air * rho1; a2r2 = (1 - a_air) * rho2
    ru = np.zeros(N)
    gm1, gm2 = 0.4, 3.4
    rho_e = a_air * p_init / gm1 + (1 - a_air) * (p_init + ph2['gamma'] * ph2['pinf']) / gm2
    rE = rho_e

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a_air,
        dx, t_end=2.29e-4, cfl=0.25, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=10000,
        alpha_scheme=scheme, use_strang=True,
        use_defect_correction=False, use_material_cfl=False)

    p, u, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    rho = a1r1_f + a2r2_f
    u_max = np.max(np.abs(u))
    print(f"  {label:12s}: u_max={u_max:.2f}")
    return x, p, u, a1_f, rho, u_max

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    schemes = [('tvd', 'TVD'), ('thinc_bvd', 'THINC-BVD'), ('cicsam', 'CICSAM'), ('mstacs', 'MSTACS')]
    results = {}
    for s, l in schemes:
        results[s] = run(s, l)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = {'tvd': 'blue', 'thinc_bvd': 'red', 'cicsam': 'green', 'mstacs': 'purple'}
    zoom_x = (0.5, 0.9)

    for s, l in schemes:
        x, p, u, a1, rho, um = results[s]
        c = colors[s]
        axes[0,0].plot(x, p, color=c, label=l)
        axes[0,1].plot(x, u, color=c, label=f'{l} u={um:.0f}')
        axes[0,2].plot(x, a1, color=c, label=l)
        axes[1,0].plot(x, rho, color=c, label=l)
        mask = (x >= zoom_x[0]) & (x <= zoom_x[1])
        axes[1,1].plot(x[mask], a1[mask], color=c, marker='o', ms=3, label=l)
        axes[1,2].plot(x[mask], u[mask], color=c, marker='o', ms=3, label=l)

    axes[0,0].set_title('Pressure'); axes[0,0].legend()
    axes[0,1].set_title('Velocity'); axes[0,1].legend()
    axes[0,2].set_title('alpha_air'); axes[0,2].legend()
    axes[1,0].set_title('Density'); axes[1,0].legend()
    axes[1,1].set_title('alpha zoom'); axes[1,1].legend()
    axes[1,2].set_title('u zoom'); axes[1,2].legend()

    plt.suptitle('Phase 2-2: Alpha scheme diagnosis (IMEX)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/diagnose_alpha_schemes.png', dpi=150)
    print("Plot: results/diagnose_alpha_schemes.png")

    print("\nInterface sharpness (cells between alpha=0.1 and alpha=0.9):")
    for s, l in schemes:
        _, _, _, a1, _, _ = results[s]
        n_transition = int(np.sum((a1 > 0.1) & (a1 < 0.9)))
        print(f"  {l:12s}: {n_transition} cells in transition")
