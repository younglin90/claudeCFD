"""세 가지 dissipation 방법 비교 테스트.
1. MWI (Denner Rhie-Chow style 4th-order pressure filter)
2. HLLC-like explicit p-Laplacian
3. Shapiro filter (post-step L-stable smoothing)

EB4 Low-Mach 테스트에서 2Δx oscillation 제거 효과 비교.
또한 Phase 1 regression 체크.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from pipeline.exact_riemann import exact_profile


def run_eb4(dissipation='none', diss_coef=0.5):
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    N = 200; L = 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    x0 = 0.5; t_end = 3e-4
    p0 = 1e5; p_L = p0*1.01; p_R = p0
    p_init = np.where(x < x0, p_L, p_R)
    rho1 = p_init / (0.4 * 717.5 * 293.0)
    rho2 = (p_init + 6e8) / (3.4 * 474.2 * 293.0)
    a_air = 1e-6 * np.ones(N)
    a1r1 = a_air * rho1; a2r2 = (1-a_air) * rho2
    rho_e0 = a_air * p_init / 0.4 + (1-a_air) * (p_init + 4.4*6e8) / 3.4

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, np.zeros(N), rho_e0, a_air,
        dx, t_end=t_end, cfl=0.4, bc_l='transmissive', bc_r='transmissive',
        max_steps=200000, print_interval=100000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=False, use_material_cfl=False,
        dissipation=dissipation, diss_coef=diss_coef)
    p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)

    rho_L_w = (p_L + 6e8) / (3.4 * 474.2 * 293.0)
    rho_R_w = (p_R + 6e8) / (3.4 * 474.2 * 293.0)
    rho_e, u_e, p_e, _ = exact_profile(
        x, t_end, x0,
        pL=p_L, rhoL=rho_L_w, uL=0.0, gammaL=4.4, pinfL=6e8,
        pR=p_R, rhoR=rho_R_w, uR=0.0, gammaR=4.4, pinfR=6e8)
    return x, p_n, u_n, p_e, u_e


def run_phase1(dissipation='none', diss_coef=0.5):
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 2.35, 'pinf': 1e9, 'kv': 943.8, 'b': 6.61e-4, 'eta': -1167e3, 'q': 0.0}
    N = 10; L = 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0 = 1e5; u0 = 1.0; T0 = 300.0
    a1 = np.where((x >= 0.4) & (x <= 0.6), 1e-6, 1.0 - 1e-6)
    rho1 = p0 / (0.4 * 717.5 * T0)
    rho2 = (p0 + 1e9) / (1.35 * 943.8 * T0)
    a1r1 = a1 * rho1 * np.ones(N)
    a2r2 = (1-a1) * rho2 * np.ones(N)
    rho = a1r1 + a2r2
    ru = rho * u0
    rho_e0 = a1 * p0 / 0.4 + (1-a1) * (p0 + 2.35*1e9) / 1.35
    rE = rho_e0 + 0.5 * rho * u0**2

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1,
        dx, t_end=1.0, cfl=0.4, bc_l='periodic', bc_r='periodic',
        max_steps=100, print_interval=1000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=True, use_material_cfl=False,
        dissipation=dissipation, diss_coef=diss_coef)
    p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    return np.max(np.abs(p_n/p0 - 1)), np.max(np.abs(u_n/u0 - 1))


def run_phase2_2(dissipation='none', diss_coef=0.5):
    """Phase 2-2 regression."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    N = 100; L = 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    x0 = 0.7; t_end = 2.29e-4
    a_air = np.where(x < x0, 1e-6, 1.0-1e-6)
    p_init = np.where(x < x0, 1e9, 1e5)
    rho1 = 50.0*np.ones(N); rho2 = 1000.0*np.ones(N)
    a1r1 = a_air * rho1; a2r2 = (1-a_air) * rho2
    rho_e0 = a_air * p_init / 0.4 + (1-a_air) * (p_init + 4.4*6e8) / 3.4

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, np.zeros(N), rho_e0, a_air,
        dx, t_end=t_end, cfl=0.25, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=100000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=False, use_material_cfl=False,
        dissipation=dissipation, diss_coef=diss_coef)
    p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    return np.max(np.abs(u_n))


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    p0 = 1e5

    configs = [
        ('none', 0.0, 'Baseline (no diss)'),
        ('mwi', 0.5, 'MWI β=0.5'),
        ('mwi', 1.0, 'MWI β=1.0'),
        ('shapiro', 0.1, 'Shapiro β=0.1'),
        ('shapiro', 0.3, 'Shapiro β=0.3'),
        ('shapiro', 0.5, 'Shapiro β=0.5'),
    ]

    print("="*90)
    print("Dissipation method comparison — EB4 Low-Mach + Phase 1/2-2 regression")
    print("="*90)
    print(f"{'Config':<25s} {'EB4 p_osc':<12s} {'EB4 u_err':<12s} {'P1 err_p':<12s} {'P2-2 u_max':<12s}")

    results = {}
    for diss, coef, label in configs:
        # EB4
        x, p_n, u_n, p_e, u_e = run_eb4(diss, coef)
        p_osc = (p_n - p_e).std() / p0
        u_max = np.max(np.abs(u_n))
        u_err = abs(u_max - np.abs(u_e).max()) / max(abs(u_e).max(), 1e-30) * 100

        # Phase 1
        err_p, err_u = run_phase1(diss, coef)

        # Phase 2-2
        u22 = run_phase2_2(diss, coef)

        print(f"{label:<25s} {p_osc:<12.3e} {u_err:<12.2f} {err_p:<12.2e} {u22:<12.1f}")
        results[label] = (x, p_n, u_n, p_e, u_e, p_osc, err_p, u22)

    # Plot best configs vs baseline
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for label, data in results.items():
        x, p_n, u_n, p_e, u_e, p_osc, _, _ = data
        if label in ['Baseline (no diss)', 'MWI β=1.0', 'Shapiro β=0.5', 'Shapiro β=0.3']:
            axes[0,0].plot(x, (p_n - 1e5)/1e5 * 100, label=f'{label} osc={p_osc:.1e}')
            mask = (x > 0.35) & (x < 0.65)
            axes[0,1].plot(x[mask], (p_n[mask] - 1e5)/1e5 * 100, 'o-', markersize=3, label=label)
            axes[1,0].plot(x, u_n*1000, label=label)
            axes[1,1].plot(x[mask], u_n[mask]*1000, 'o-', markersize=3, label=label)
    # Exact
    x = results['Baseline (no diss)'][0]
    p_e = results['Baseline (no diss)'][3]
    u_e = results['Baseline (no diss)'][4]
    axes[0,0].plot(x, (p_e - 1e5)/1e5 * 100, 'k--', lw=2, label='Exact')
    mask = (x > 0.35) & (x < 0.65)
    axes[0,1].plot(x[mask], (p_e[mask] - 1e5)/1e5 * 100, 'k--', lw=2, label='Exact')
    axes[1,0].plot(x, u_e*1000, 'k--', lw=2, label='Exact')
    axes[1,1].plot(x[mask], u_e[mask]*1000, 'k--', lw=2, label='Exact')

    axes[0,0].set_title('EB4 full: p perturbation (%)'); axes[0,0].legend(fontsize=8); axes[0,0].set_xlabel('x')
    axes[0,1].set_title('EB4 center zoom: p (%)'); axes[0,1].legend(fontsize=8); axes[0,1].set_xlabel('x')
    axes[1,0].set_title('EB4 full: u (mm/s)'); axes[1,0].legend(fontsize=8); axes[1,0].set_xlabel('x')
    axes[1,1].set_title('EB4 center zoom: u (mm/s)'); axes[1,1].legend(fontsize=8); axes[1,1].set_xlabel('x')

    plt.suptitle('EB4 2Δx Dissipation Comparison: MWI vs HLLC vs Shapiro', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/dissipation_comparison.png', dpi=150)
    print(f"\nPlot saved: results/dissipation_comparison.png")
