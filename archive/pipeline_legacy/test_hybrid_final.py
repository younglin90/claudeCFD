"""Hybrid dissipation: comprehensive validation with exact Riemann."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from pipeline.exact_riemann import exact_profile, exact_riemann_star


def phase2_1(diss):
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.1, 'pinf': 4.4e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    N = 200; L = 2.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    x0 = 0.5; t_end = 8e-4; T0 = 300.0
    p_L = 1e9; p_R = 1e4
    a_air = np.where(x < x0, 1.0-1e-6, 1e-6)
    p_init = np.where(x < x0, p_L, p_R)
    rho1 = p_init / (0.4 * 717.5 * T0)
    rho2 = (p_init + 4.4e8) / (3.1 * 474.2 * T0)
    a1r1 = a_air * rho1; a2r2 = (1-a_air) * rho2
    rho_e0 = a_air * p_init / 0.4 + (1-a_air) * (p_init + 4.1*4.4e8) / 3.1

    rho1_L = p_L / (0.4 * 717.5 * T0); rho2_R = (p_R + 4.4e8) / (3.1 * 474.2 * T0)
    rho_e, u_e, p_e, _ = exact_profile(x, t_end, x0,
        pL=p_L, rhoL=rho1_L, uL=0.0, gammaL=1.4, pinfL=0.0,
        pR=p_R, rhoR=rho2_R, uR=0.0, gammaR=4.1, pinfR=4.4e8)

    _, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, np.zeros(N), rho_e0, a_air,
        dx, t_end=t_end, cfl=0.4, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=10000,
        alpha_scheme='tvd', use_strang=True, use_defect_correction=False,
        use_material_cfl=False, dissipation=diss)
    p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    rho_n = a1r1_f + a2r2_f
    return x, p_n, u_n, rho_n, p_e, u_e, rho_e


def phase2_2(diss):
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    N = 200; L = 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    x0 = 0.7; t_end = 2.29e-4
    a_air = np.where(x < x0, 1e-6, 1.0-1e-6)
    p_init = np.where(x < x0, 1e9, 1e5)
    rho1 = 50.0*np.ones(N); rho2 = 1000.0*np.ones(N)
    a1r1 = a_air * rho1; a2r2 = (1-a_air) * rho2
    rho_e0 = a_air * p_init / 0.4 + (1-a_air) * (p_init + 4.4*6e8) / 3.4

    rho_e, u_e, p_e, _ = exact_profile(x, t_end, x0,
        pL=1e9, rhoL=1000.0, uL=0.0, gammaL=4.4, pinfL=6e8,
        pR=1e5, rhoR=50.0, uR=0.0, gammaR=1.4, pinfR=0.0)

    _, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, np.zeros(N), rho_e0, a_air,
        dx, t_end=t_end, cfl=0.25, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=10000,
        alpha_scheme='tvd', use_strang=True, use_defect_correction=False,
        use_material_cfl=False, dissipation=diss)
    p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    rho_n = a1r1_f + a2r2_f
    return x, p_n, u_n, rho_n, p_e, u_e, rho_e


def eb4(diss):
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    N = 200; L = 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    x0 = 0.5
    p0 = 1e5; p_L = p0*1.01; p_R = p0
    p_init = np.where(x < x0, p_L, p_R)
    rho1 = p_init / (0.4 * 717.5 * 293.0)
    rho2 = (p_init + 6e8) / (3.4 * 474.2 * 293.0)
    a_air = 1e-6 * np.ones(N)
    a1r1 = a_air * rho1; a2r2 = (1-a_air) * rho2
    rho_e0 = a_air * p_init / 0.4 + (1-a_air) * (p_init + 4.4*6e8) / 3.4

    rho_L_w = (p_L + 6e8) / (3.4 * 474.2 * 293.0)
    rho_R_w = (p_R + 6e8) / (3.4 * 474.2 * 293.0)
    rho_e, u_e, p_e, _ = exact_profile(x, 3e-4, x0,
        pL=p_L, rhoL=rho_L_w, uL=0.0, gammaL=4.4, pinfL=6e8,
        pR=p_R, rhoR=rho_R_w, uR=0.0, gammaR=4.4, pinfR=6e8)

    _, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, np.zeros(N), rho_e0, a_air,
        dx, t_end=3e-4, cfl=0.4, bc_l='transmissive', bc_r='transmissive',
        max_steps=500, print_interval=10000,
        alpha_scheme='tvd', use_strang=True, use_defect_correction=False,
        use_material_cfl=False, dissipation=diss)
    p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    return x, p_n, u_n, p_e, u_e


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    configs = [('none', 'Baseline', 'red'),
               ('hybrid', 'Hybrid (NEW)', 'blue')]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # Phase 2-1
    print("Phase 2-1:")
    for diss, label, color in configs:
        x, p_n, u_n, rho_n, p_e, u_e, rho_e = phase2_1(diss)
        u_max = np.abs(u_n).max()
        err_u = abs(u_max - np.abs(u_e).max()) / max(np.abs(u_e).max(), 1e-30) * 100
        print(f"  {label}: u_max={u_max:.1f}, err={err_u:.2f}%")
        axes[0,0].plot(x, p_n, color=color, label=f'{label} u_err={err_u:.1f}%')
        axes[0,1].plot(x, u_n, color=color, label=label)
        axes[0,2].plot(x, rho_n, color=color, label=label)
    x, _, _, _, p_e, u_e, rho_e = phase2_1('none')
    axes[0,0].plot(x, p_e, 'k--', lw=2, label='Exact')
    axes[0,1].plot(x, u_e, 'k--', lw=2, label='Exact')
    axes[0,2].plot(x, rho_e, 'k--', lw=2, label='Exact')
    axes[0,0].set_title('Phase 2-1: p'); axes[0,0].legend(fontsize=8); axes[0,0].set_yscale('log')
    axes[0,1].set_title('Phase 2-1: u'); axes[0,1].legend(fontsize=8)
    axes[0,2].set_title('Phase 2-1: rho'); axes[0,2].legend(fontsize=8)

    # Phase 2-2
    print("Phase 2-2:")
    for diss, label, color in configs:
        x, p_n, u_n, rho_n, p_e, u_e, rho_e = phase2_2(diss)
        u_max = np.abs(u_n).max()
        err_u = abs(u_max - np.abs(u_e).max()) / max(np.abs(u_e).max(), 1e-30) * 100
        print(f"  {label}: u_max={u_max:.1f}, err={err_u:.2f}%")
        axes[1,0].plot(x, p_n, color=color, label=f'{label} u_err={err_u:.1f}%')
        axes[1,1].plot(x, u_n, color=color, label=label)
        axes[1,2].plot(x, rho_n, color=color, label=label)
    x, _, _, _, p_e, u_e, rho_e = phase2_2('none')
    axes[1,0].plot(x, p_e, 'k--', lw=2, label='Exact')
    axes[1,1].plot(x, u_e, 'k--', lw=2, label='Exact')
    axes[1,2].plot(x, rho_e, 'k--', lw=2, label='Exact')
    axes[1,0].set_title('Phase 2-2: p'); axes[1,0].legend(fontsize=8)
    axes[1,1].set_title('Phase 2-2: u'); axes[1,1].legend(fontsize=8)
    axes[1,2].set_title('Phase 2-2: rho'); axes[1,2].legend(fontsize=8)

    # EB4
    print("EB4:")
    p0 = 1e5
    for diss, label, color in configs:
        x, p_n, u_n, p_e, u_e = eb4(diss)
        d2 = p_n[2:] - 2*p_n[1:-1] + p_n[:-2]
        d2_rms = np.sqrt(np.mean(d2**2)) / p0
        print(f"  {label}: d2_rms={d2_rms:.2e}")
        axes[2,0].plot(x, (p_n - p0)/p0 * 100, color=color, label=f'{label} d2={d2_rms:.1e}')
        mask = (x > 0.35) & (x < 0.65)
        axes[2,1].plot(x[mask], (p_n[mask] - p0)/p0 * 100, 'o-', color=color, markersize=3, label=label)
        axes[2,2].plot(x, u_n*1000, color=color, label=label)
    x, _, _, p_e, u_e = eb4('none')
    axes[2,0].plot(x, (p_e - p0)/p0 * 100, 'k--', lw=2, label='Exact')
    axes[2,1].plot(x[(x > 0.35) & (x < 0.65)], (p_e[(x > 0.35) & (x < 0.65)] - p0)/p0 * 100, 'k--', lw=2, label='Exact')
    axes[2,2].plot(x, u_e*1000, 'k--', lw=2, label='Exact')
    axes[2,0].set_title('EB4 Low-Mach: p (%)'); axes[2,0].legend(fontsize=8)
    axes[2,1].set_title('EB4 Center zoom: p (%)'); axes[2,1].legend(fontsize=8)
    axes[2,2].set_title('EB4: u (mm/s)'); axes[2,2].legend(fontsize=8)

    for ax in axes.flat: ax.set_xlabel('x')
    plt.suptitle('Hybrid Dissipation: projection (smooth) + face-flux (shock) — exact comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/hybrid_comparison.png', dpi=150)
    print(f"\nPlot: results/hybrid_comparison.png")
