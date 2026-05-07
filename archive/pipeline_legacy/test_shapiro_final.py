"""Shapiro filter 최종 검증: 모든 기존 테스트 regression + EB4 개선."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from pipeline.exact_riemann import exact_profile, exact_riemann_star


def run_phase1(diss='none', coef=0.0):
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 2.35, 'pinf': 1e9, 'kv': 943.8, 'b': 6.61e-4, 'eta': -1167e3, 'q': 0.0}
    N = 10; L = 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0 = 1e5; u0 = 1.0; T0 = 300.0
    a1 = np.where((x >= 0.4) & (x <= 0.6), 1e-6, 1.0 - 1e-6)
    rho1 = p0 / (0.4 * 717.5 * T0)
    rho2 = (p0 + 1e9) / (1.35 * 943.8 * T0)
    a1r1 = a1 * rho1 * np.ones(N); a2r2 = (1-a1) * rho2 * np.ones(N)
    rho = a1r1 + a2r2; ru = rho * u0
    rho_e0 = a1 * p0 / 0.4 + (1-a1) * (p0 + 2.35*1e9) / 1.35
    rE = rho_e0 + 0.5 * rho * u0**2
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1, dx, t_end=1.0, cfl=0.4,
        bc_l='periodic', bc_r='periodic',
        max_steps=100, print_interval=1000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=True, use_material_cfl=False,
        dissipation=diss, diss_coef=coef)
    p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    return np.max(np.abs(p_n/p0 - 1)), np.max(np.abs(u_n/u0 - 1))


def run_phase2_1(diss='none', coef=0.0):
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.1, 'pinf': 4.4e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    N = 50; L = 2.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    x0 = 0.5; t_end = 8e-4; T0 = 300.0
    a_air = np.where(x < x0, 1.0-1e-6, 1e-6)
    p_init = np.where(x < x0, 1e9, 1e4)
    rho1 = p_init / (0.4 * 717.5 * T0)
    rho2 = (p_init + 4.4e8) / (3.1 * 474.2 * T0)
    a1r1 = a_air * rho1; a2r2 = (1-a_air) * rho2
    rho_e0 = a_air * p_init / 0.4 + (1-a_air) * (p_init + 4.1*4.4e8) / 3.1
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, np.zeros(N), rho_e0, a_air,
        dx, t_end=t_end, cfl=0.4, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=10000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=False, use_material_cfl=False,
        dissipation=diss, diss_coef=coef)
    p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    return np.max(np.abs(u_n))


def run_phase2_2(diss='none', coef=0.0):
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
        max_steps=100000, print_interval=10000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=False, use_material_cfl=False,
        dissipation=diss, diss_coef=coef)
    p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    return np.max(np.abs(u_n))


def run_eb4_with_profile(diss='none', coef=0.0):
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
        max_steps=500, print_interval=10000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=False, use_material_cfl=False,
        dissipation=diss, diss_coef=coef)
    p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    # Nyquist amp
    sign_alt = (-1.0) ** np.arange(N)
    nyq_amp = np.abs(np.mean(p_n * sign_alt)) / p0
    d2 = p_n[2:] - 2*p_n[1:-1] + p_n[:-2]
    d2_rms = np.sqrt(np.mean(d2**2)) / p0
    return x, p_n, u_n, nyq_amp, d2_rms


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    print("="*80)
    print("Shapiro Filter Final Validation")
    print("="*80)
    print(f"{'Config':<20s} {'P1 err_p':<14s} {'P2-1 u_max':<13s} {'P2-2 u_max':<13s} {'EB4 d2_rms':<14s}")

    configs = [
        ('none', 0.0, 'Baseline'),
        ('shapiro', 0.05, 'Shapiro 0.05'),
        ('shapiro', 0.1, 'Shapiro 0.1'),
        ('shapiro', 0.2, 'Shapiro 0.2'),
        ('shapiro', 0.5, 'Shapiro 0.5'),
    ]

    results = {}
    for diss, coef, label in configs:
        err_p, err_u = run_phase1(diss, coef)
        u21 = run_phase2_1(diss, coef)
        u22 = run_phase2_2(diss, coef)
        x, p_n, u_n, nyq, d2 = run_eb4_with_profile(diss, coef)
        print(f"{label:<20s} {err_p:<14.3e} {u21:<13.1f} {u22:<13.1f} {d2:<14.3e}")
        results[label] = (x, p_n, u_n, nyq, d2, err_p, u21, u22)

    # Plot EB4 comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(configs)))
    p0 = 1e5
    for i, (label, data) in enumerate(results.items()):
        x, p_n, u_n, nyq, d2, err_p, u21, u22 = data
        axes[0,0].plot(x, (p_n - p0)/p0 * 100, color=colors[i], label=f'{label} d2={d2:.1e}')
        mask = (x > 0.35) & (x < 0.65)
        axes[0,1].plot(x[mask], (p_n[mask] - p0)/p0 * 100, 'o-', color=colors[i], markersize=3, label=label)
        axes[1,0].plot(x, u_n*1000, color=colors[i], label=label)
        axes[1,1].plot(x[mask], u_n[mask]*1000, 'o-', color=colors[i], markersize=3, label=label)

    axes[0,0].set_title('EB4 Full: p perturbation (%)')
    axes[0,0].legend(fontsize=8); axes[0,0].set_xlabel('x')
    axes[0,1].set_title('EB4 Center zoom: p (%)')
    axes[0,1].legend(fontsize=8); axes[0,1].set_xlabel('x')
    axes[1,0].set_title('EB4 Full: u (mm/s)')
    axes[1,0].legend(fontsize=8); axes[1,0].set_xlabel('x')
    axes[1,1].set_title('EB4 Center zoom: u (mm/s)')
    axes[1,1].legend(fontsize=8); axes[1,1].set_xlabel('x')

    plt.suptitle('Shapiro Filter: EB4 2Δx suppression (Post-IM1 filter on ru, rE)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/shapiro_final.png', dpi=150)
    print(f"\nPlot: results/shapiro_final.png")
