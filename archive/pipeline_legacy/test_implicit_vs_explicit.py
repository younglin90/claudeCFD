"""
Phase 2-1 & Phase 2-2 비교: Implicit BE vs Explicit MMACM-Ex

Implicit: solve_implicit_be (1st order, autograd Jacobian)
  - CFL=0.5, N=50 (autograd feasible)
Explicit: solve (SSP-RK3, TVD+THINC-BVD, MMACM-Ex, APEC, compression)
  - CFL=0.4, N=200

결과 그래프 → results/implicit_vs_explicit_phase2_*.png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from solver.He2024.explicit_mmacm_ex import (
    _sg_density_from_pT, _sg_internal_energy, cons_to_prim,
    prim_to_cons, solve, solve_implicit_be, _compute_dt,
)


# ============================================================
# Phase 2-1 setup: HP Air (left) / LP Water (right)
# ============================================================

def setup_phase2_1(N):
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}      # Air (ideal)
    ph2 = {'gamma': 4.1, 'pinf': 4.4e8, 'kv': 474.2}     # Water (SG)
    L, x_intf = 2.0, 0.5
    dx = L / N
    x = np.linspace(0.5*dx, L - 0.5*dx, N)
    T0, u0 = 300.0, 0.0
    p_L, p_R = 1.0e9, 1.0e4
    eps_pure = 1e-8
    a1 = np.where(x < x_intf, 1.0 - eps_pure, eps_pure)
    p_field = np.where(x < x_intf, p_L, p_R)
    g1, pinf1, kv1 = ph1['gamma'], ph1['pinf'], ph1['kv']
    g2, pinf2, kv2 = ph2['gamma'], ph2['pinf'], ph2['kv']
    rho1 = _sg_density_from_pT(p_field, T0, g1, pinf1, kv1)
    rho2 = _sg_density_from_pT(p_field, T0, g2, pinf2, kv2)
    a2 = 1.0 - a1
    a1r1 = a1 * rho1
    a2r2 = a2 * rho2
    rho = a1r1 + a2r2
    ru = rho * u0
    e1 = _sg_internal_energy(p_field, rho1, g1, pinf1)
    e2 = _sg_internal_energy(p_field, rho2, g2, pinf2)
    rho_e = a1 * rho1 * e1 + a2 * rho2 * e2
    rE = rho_e + 0.5 * rho * u0**2
    return x, dx, a1r1, a2r2, ru, rE, a1, ph1, ph2


# ============================================================
# Phase 2-2 setup: HP Water (left) / LP Air (right)
# ============================================================

def setup_phase2_2(N):
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}       # Air (ideal)
    ph2 = {'gamma': 4.4, 'pinf': 6.0e8, 'kv': 474.2}     # Water (SG, Yoo&Sung 2018)
    L, x_intf = 1.0, 0.7
    dx = L / N
    x = np.linspace(0.5*dx, L - 0.5*dx, N)
    p_L, p_R = 1.0e9, 1.0e5
    rho1_val, rho2_val = 50.0, 1000.0  # direct from paper
    eps_pure = 1e-6
    # left = Water dominant, right = Air dominant
    a1_air = np.where(x < x_intf, eps_pure, 1.0 - eps_pure)
    a1 = a1_air
    a2 = 1.0 - a1
    g1, pinf1, kv1 = ph1['gamma'], ph1['pinf'], ph1['kv']
    g2, pinf2, kv2 = ph2['gamma'], ph2['pinf'], ph2['kv']
    p_field = np.where(x < x_intf, p_L, p_R)
    rho1 = np.full(N, rho1_val)
    rho2 = np.full(N, rho2_val)
    a1r1 = a1 * rho1
    a2r2 = a2 * rho2
    rho = a1r1 + a2r2
    ru = rho * 0.0
    e1 = _sg_internal_energy(p_field, rho1, g1, pinf1)
    e2 = _sg_internal_energy(p_field, rho2, g2, pinf2)
    rho_e = a1 * rho1 * e1 + a2 * rho2 * e2
    rE = rho_e
    return x, dx, a1r1, a2r2, ru, rE, a1, ph1, ph2


# ============================================================
# Run & Compare
# ============================================================

def run_and_plot_phase2_1():
    """Phase 2-1: implicit (N=50, CFL=0.5) vs explicit (N=200, CFL=0.4)"""
    t_end = 8.0e-4

    # --- Explicit N=200 (reference) ---
    print("=" * 60)
    print("Phase 2-1 Explicit: N=200, CFL=0.4, MMACM-Ex+APEC+Compress")
    print("=" * 60)
    N_exp = 200
    x_exp, dx_exp, a1r1_e, a2r2_e, ru_e, rE_e, a1_e, ph1, ph2 = setup_phase2_1(N_exp)
    t_f_exp, a1r1_ef, a2r2_ef, ru_ef, rE_ef, a1_ef = solve(
        ph1, ph2, a1r1_e, a2r2_e, ru_e, rE_e, a1_e,
        dx_exp, t_end, cfl=0.4,
        bc_l='transmissive', bc_r='transmissive',
        use_mmacm_ex=True, print_interval=100,
        alpha_recon='thinc_bvd',
        use_compression=True, C_alpha=1.0,
        compress_corrections=True, use_apec=True)
    p_exp, u_exp, T_exp, rho1_exp, rho2_exp, _, _, _ = cons_to_prim(
        a1r1_ef, a2r2_ef, ru_ef, rE_ef, a1_ef, ph1, ph2)
    rho_exp = a1r1_ef + a2r2_ef
    print(f"  Explicit done: u_max={u_exp.max():.1f}, p_max={p_exp.max():.3e}")

    # --- Implicit N=50 (autograd, fast) ---
    print("\n" + "=" * 60)
    print("Phase 2-1 Implicit: N=50, CFL=0.5, autograd")
    print("=" * 60)
    N_imp = 50
    x_imp, dx_imp, a1r1_i, a2r2_i, ru_i, rE_i, a1_i, _, _ = setup_phase2_1(N_imp)
    t_f_imp, a1r1_if, a2r2_if, ru_if, rE_if, a1_if = solve_implicit_be(
        ph1, ph2, a1r1_i, a2r2_i, ru_i, rE_i, a1_i,
        dx_imp, t_end, cfl=0.5,
        bc_l='transmissive', bc_r='transmissive',
        max_newton=20, newton_tol=1e-8,
        print_interval=10,
        jacobian_method='autograd')
    p_imp, u_imp, T_imp, rho1_imp, rho2_imp, _, _, _ = cons_to_prim(
        a1r1_if, a2r2_if, ru_if, rE_if, a1_if, ph1, ph2)
    rho_imp = a1r1_if + a2r2_if
    print(f"  Implicit done: u_max={u_imp.max():.1f}, p_max={p_imp.max():.3e}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Phase 2-1: HP Air / LP Water  (t={t_end:.2e} s)\n'
                 f'Implicit BE (N=50, CFL=0.5, 1st order) vs '
                 f'Explicit SSP-RK3 (N=200, CFL=0.4, TVD+THINC-BVD+MMACM-Ex+APEC)',
                 fontsize=13, fontweight='bold')

    titles = ['Pressure [Pa]', 'Velocity [m/s]', 'Density [kg/m³]',
              'α₁ (Air)', 'Temperature [K]', 'Phase Densities [kg/m³]']
    exp_data = [p_exp, u_exp, rho_exp, a1_ef, T_exp, None]
    imp_data = [p_imp, u_imp, rho_imp, a1_if, T_imp, None]

    for idx in range(6):
        ax = axes[idx // 3][idx % 3]
        ax.set_title(titles[idx], fontsize=12)
        if idx == 5:  # phase densities
            ax.plot(x_exp, rho1_exp, 'b-', alpha=0.7, label=f'ρ₁ Explicit N={N_exp}')
            ax.plot(x_exp, rho2_exp, 'r-', alpha=0.7, label=f'ρ₂ Explicit N={N_exp}')
            ax.plot(x_imp, rho1_imp, 'b--', linewidth=2, label=f'ρ₁ Implicit N={N_imp}')
            ax.plot(x_imp, rho2_imp, 'r--', linewidth=2, label=f'ρ₂ Implicit N={N_imp}')
        else:
            ax.plot(x_exp, exp_data[idx], 'b-', alpha=0.7, label=f'Explicit N={N_exp}')
            ax.plot(x_imp, imp_data[idx], 'r--', linewidth=2, label=f'Implicit N={N_imp}')
        ax.legend(fontsize=9)
        ax.set_xlabel('x [m]')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = 'results/implicit_vs_explicit_phase2_1.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nPlot saved: {out_path}")


def run_and_plot_phase2_2():
    """Phase 2-2: implicit (N=50, CFL=0.25) vs explicit (N=200, CFL=0.25)"""
    t_end = 2.29e-4

    # --- Explicit N=200 (reference) ---
    print("\n" + "=" * 60)
    print("Phase 2-2 Explicit: N=200, CFL=0.25, MMACM-Ex+APEC+Compress")
    print("=" * 60)
    N_exp = 200
    x_exp, dx_exp, a1r1_e, a2r2_e, ru_e, rE_e, a1_e, ph1, ph2 = setup_phase2_2(N_exp)
    t_f_exp, a1r1_ef, a2r2_ef, ru_ef, rE_ef, a1_ef = solve(
        ph1, ph2, a1r1_e, a2r2_e, ru_e, rE_e, a1_e,
        dx_exp, t_end, cfl=0.25,
        bc_l='transmissive', bc_r='transmissive',
        use_mmacm_ex=True, print_interval=100,
        alpha_recon='thinc_bvd',
        use_compression=True, C_alpha=1.0,
        compress_corrections=True, use_apec=True)
    p_exp, u_exp, T_exp, rho1_exp, rho2_exp, _, _, _ = cons_to_prim(
        a1r1_ef, a2r2_ef, ru_ef, rE_ef, a1_ef, ph1, ph2)
    rho_exp = a1r1_ef + a2r2_ef
    print(f"  Explicit done: u_max={u_exp.max():.1f}, p_max={p_exp.max():.3e}")

    # --- Implicit N=50 ---
    print("\n" + "=" * 60)
    print("Phase 2-2 Implicit: N=50, CFL=0.5, autograd")
    print("=" * 60)
    N_imp = 50
    x_imp, dx_imp, a1r1_i, a2r2_i, ru_i, rE_i, a1_i, _, _ = setup_phase2_2(N_imp)
    t_f_imp, a1r1_if, a2r2_if, ru_if, rE_if, a1_if = solve_implicit_be(
        ph1, ph2, a1r1_i, a2r2_i, ru_i, rE_i, a1_i,
        dx_imp, t_end, cfl=0.5,
        bc_l='transmissive', bc_r='transmissive',
        max_newton=20, newton_tol=1e-8,
        print_interval=10,
        jacobian_method='autograd')
    p_imp, u_imp, T_imp, rho1_imp, rho2_imp, _, _, _ = cons_to_prim(
        a1r1_if, a2r2_if, ru_if, rE_if, a1_if, ph1, ph2)
    rho_imp = a1r1_if + a2r2_if
    print(f"  Implicit done: u_max={u_imp.max():.1f}, p_max={p_imp.max():.3e}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Phase 2-2: HP Water / LP Air  (t={t_end:.2e} s)\n'
                 f'Implicit BE (N=50, CFL=0.5, 1st order) vs '
                 f'Explicit SSP-RK3 (N=200, CFL=0.25, TVD+THINC-BVD+MMACM-Ex+APEC)',
                 fontsize=13, fontweight='bold')

    titles = ['Pressure [Pa]', 'Velocity [m/s]', 'Density [kg/m³]',
              'α₁ (Air)', 'Temperature [K]', 'Phase Densities [kg/m³]']
    exp_data = [p_exp, u_exp, rho_exp, a1_ef, T_exp, None]
    imp_data = [p_imp, u_imp, rho_imp, a1_if, T_imp, None]

    for idx in range(6):
        ax = axes[idx // 3][idx % 3]
        ax.set_title(titles[idx], fontsize=12)
        if idx == 5:
            ax.plot(x_exp, rho1_exp, 'b-', alpha=0.7, label=f'ρ₁ Explicit N={N_exp}')
            ax.plot(x_exp, rho2_exp, 'r-', alpha=0.7, label=f'ρ₂ Explicit N={N_exp}')
            ax.plot(x_imp, rho1_imp, 'b--', linewidth=2, label=f'ρ₁ Implicit N={N_imp}')
            ax.plot(x_imp, rho2_imp, 'r--', linewidth=2, label=f'ρ₂ Implicit N={N_imp}')
        else:
            ax.plot(x_exp, exp_data[idx], 'b-', alpha=0.7, label=f'Explicit N={N_exp}')
            ax.plot(x_imp, imp_data[idx], 'r--', linewidth=2, label=f'Implicit N={N_imp}')
        ax.legend(fontsize=9)
        ax.set_xlabel('x [m]')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = 'results/implicit_vs_explicit_phase2_2.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nPlot saved: {out_path}")


if __name__ == '__main__':
    run_and_plot_phase2_1()
    run_and_plot_phase2_2()
    print("\nAll tests done!")
