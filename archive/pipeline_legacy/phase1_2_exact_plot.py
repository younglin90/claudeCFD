"""Dedicated exact vs numerical comparison for Phase 1, 2-1, 2-2.
Phase 1: periodic advection -> exact solution is the initial condition shifted
Phase 2-1, 2-2: Riemann problem -> use exact Riemann solver
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from pipeline.exact_riemann import exact_profile, exact_riemann_star


def l1_err(y_n, y_e):
    s = np.sum(np.abs(y_e))
    if s < 1e-30: return 0.0
    return np.sum(np.abs(y_n - y_e)) / s


# ========================================================================
# PHASE 1: Abgrall advection (periodic) — exact = uniform (p,u) preservation
# ========================================================================
def phase1_exact():
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 2.35, 'pinf': 1e9, 'kv': 943.8, 'b': 6.61e-4, 'eta': -1167e3, 'q': 0.0}
    N = 10; L = 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0 = 1e5; u0 = 1.0; T0 = 300.0

    a1 = np.where((x >= 0.4) & (x <= 0.6), 1e-6, 1.0 - 1e-6)
    a2 = 1.0 - a1
    rho1 = p0 / (0.4 * 717.5 * T0)
    rho2 = (p0 + 1e9) / (1.35 * 943.8 * T0)
    a1r1 = a1 * rho1 * np.ones(N)
    a2r2 = a2 * rho2 * np.ones(N)
    rho = a1r1 + a2r2
    ru = rho * u0
    rho_e0 = a1 * p0 / 0.4 + a2 * (p0 + 2.35*1e9) / 1.35
    rE = rho_e0 + 0.5 * rho * u0**2

    t_end = 1.0  # multiple periods
    # Run 100 steps (matches spec for Phase 1)
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1,
        dx, t_end=t_end, cfl=0.4, bc_l='periodic', bc_r='periodic',
        max_steps=100, print_interval=1000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=True, use_material_cfl=False)
    p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)

    # Exact: p stays at p0, u stays at u0 for PE-preserving scheme
    p_e = p0 * np.ones(N)
    u_e = u0 * np.ones(N)

    err_p = np.max(np.abs(p_n/p0 - 1))
    err_u = np.max(np.abs(u_n/u0 - 1))
    print(f"Phase 1 Abgrall: err_p={err_p:.2e}, err_u={err_u:.2e}")
    return x, p_n, u_n, a1_f, p_e, u_e, err_p, err_u


# ========================================================================
# PHASE 2-1: HP Air / LP Water Shock Tube
# ========================================================================
def phase2_1_exact():
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.1, 'pinf': 4.4e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    N = 200; L = 2.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    x0 = 0.5; t_end = 8e-4; T0 = 300.0
    p_L = 1e9; p_R = 1e4
    rho1_L = p_L / (0.4 * 717.5 * T0)
    rho2_R = (p_R + 4.4e8) / (3.1 * 474.2 * T0)

    rho_e, u_e, p_e, _ = exact_profile(
        x, t_end, x0,
        pL=p_L, rhoL=rho1_L, uL=0.0, gammaL=1.4, pinfL=0.0,
        pR=p_R, rhoR=rho2_R, uR=0.0, gammaR=4.1, pinfR=4.4e8)
    p_star, u_star = exact_riemann_star(p_L, rho1_L, 0.0, 1.4, 0.0,
                                         p_R, rho2_R, 0.0, 4.1, 4.4e8)

    a_air = np.where(x < x0, 1.0-1e-6, 1e-6)
    p_init = np.where(x < x0, p_L, p_R)
    rho1 = p_init / (0.4 * 717.5 * T0)
    rho2 = (p_init + 4.4e8) / (3.1 * 474.2 * T0)
    a1r1 = a_air * rho1; a2r2 = (1-a_air) * rho2
    rho_e0 = a_air * p_init / 0.4 + (1-a_air) * (p_init + 4.1*4.4e8) / 3.1

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, np.zeros(N), rho_e0, a_air,
        dx, t_end=t_end, cfl=0.4, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=10000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=False, use_material_cfl=False)
    p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    rho_n = a1r1_f + a2r2_f

    L1_p = l1_err(p_n, p_e); L1_u = l1_err(u_n, u_e); L1_rho = l1_err(rho_n, rho_e)
    u_max = np.max(np.abs(u_n))
    err_u = abs(u_max - u_star) / max(abs(u_star), 1e-30) * 100
    print(f"Phase 2-1: u*_exact={u_star:.2f}, u_max={u_max:.2f}, err={err_u:.2f}%")
    print(f"  L1: p={L1_p*100:.2f}%, u={L1_u*100:.2f}%, rho={L1_rho*100:.2f}%")
    return x, p_n, u_n, rho_n, p_e, u_e, rho_e, p_star, u_star, L1_p, L1_u, L1_rho


# ========================================================================
# PHASE 2-2: HP Water / LP Air Shock Tube
# ========================================================================
def phase2_2_exact():
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    N = 200; L = 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    x0 = 0.7; t_end = 2.29e-4
    p_L = 1e9; p_R = 1e5
    rho1_val = 50.0; rho2_val = 1000.0

    rho_e, u_e, p_e, _ = exact_profile(
        x, t_end, x0,
        pL=p_L, rhoL=rho2_val, uL=0.0, gammaL=4.4, pinfL=6e8,
        pR=p_R, rhoR=rho1_val, uR=0.0, gammaR=1.4, pinfR=0.0)
    p_star, u_star = exact_riemann_star(p_L, rho2_val, 0.0, 4.4, 6e8,
                                         p_R, rho1_val, 0.0, 1.4, 0.0)

    a_air = np.where(x < x0, 1e-6, 1.0-1e-6)
    p_init = np.where(x < x0, p_L, p_R)
    rho1 = rho1_val * np.ones(N); rho2 = rho2_val * np.ones(N)
    a1r1 = a_air * rho1; a2r2 = (1-a_air) * rho2
    rho_e0 = a_air * p_init / 0.4 + (1-a_air) * (p_init + 4.4*6e8) / 3.4

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, np.zeros(N), rho_e0, a_air,
        dx, t_end=t_end, cfl=0.25, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=10000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=False, use_material_cfl=False)
    p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    rho_n = a1r1_f + a2r2_f

    L1_p = l1_err(p_n, p_e); L1_u = l1_err(u_n, u_e); L1_rho = l1_err(rho_n, rho_e)
    u_max = np.max(np.abs(u_n))
    err_u = abs(u_max - u_star) / max(abs(u_star), 1e-30) * 100
    print(f"Phase 2-2: u*_exact={u_star:.2f}, u_max={u_max:.2f}, err={err_u:.2f}%")
    print(f"  L1: p={L1_p*100:.2f}%, u={L1_u*100:.2f}%, rho={L1_rho*100:.2f}%")
    return x, p_n, u_n, rho_n, p_e, u_e, rho_e, p_star, u_star, L1_p, L1_u, L1_rho


# ==================================================================
if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    print("="*70)
    print("Phase 1, 2-1, 2-2 Exact Solution Comparison (Post Round 2)")
    print("="*70)

    r1 = phase1_exact()
    r21 = phase2_1_exact()
    r22 = phase2_2_exact()

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))

    # Phase 1
    x1, p1_n, u1_n, a1_n, p1_e, u1_e, err_p, err_u = r1
    axes[0,0].plot(x1, p1_e/1e5, 'k-', lw=2, label='Exact (p0)')
    axes[0,0].plot(x1, p1_n/1e5, 'b.-', markersize=8, label='IMEX')
    axes[0,0].set_title(f'Phase 1: p (err={err_p:.1e})')
    axes[0,0].set_ylabel('p / p0'); axes[0,0].legend(); axes[0,0].set_xlabel('x')
    axes[0,1].plot(x1, u1_e, 'k-', lw=2, label='Exact (u0=1)')
    axes[0,1].plot(x1, u1_n, 'b.-', markersize=8, label='IMEX')
    axes[0,1].set_title(f'Phase 1: u (err={err_u:.1e})')
    axes[0,1].legend(); axes[0,1].set_xlabel('x')
    axes[0,2].plot(x1, a1_n, 'b.-', markersize=8, label='alpha_air (100 steps)')
    axes[0,2].set_title(f'Phase 1: alpha_air (advected)')
    axes[0,2].legend(); axes[0,2].set_xlabel('x')
    axes[0,3].axis('off')
    axes[0,3].text(0.05, 0.5,
                   f'Phase 1 Abgrall Advection:\n\n'
                   f'  err_p = {err_p:.2e}\n'
                   f'  err_u = {err_u:.2e}\n\n'
                   f'  PE preservation:\n'
                   f'  -> machine precision',
                   transform=axes[0,3].transAxes, fontsize=12, family='monospace')

    # Phase 2-1
    for i, (r, lbl, ptype) in enumerate([(r21, 'Phase 2-1 HP Air/LP Water', 1),
                                           (r22, 'Phase 2-2 HP Water/LP Air', 2)]):
        row = i + 1
        x, p_n, u_n, rho_n, p_e, u_e, rho_e, p_star, u_star, L1_p, L1_u, L1_rho = r
        axes[row,0].plot(x, p_e, 'k-', lw=2, label='Exact')
        axes[row,0].plot(x, p_n, 'b.-', markersize=3, label='IMEX')
        axes[row,0].set_title(f'{lbl}: p (L1={L1_p*100:.1f}%)')
        axes[row,0].legend(); axes[row,0].set_xlabel('x')
        if ptype == 1:
            axes[row,0].set_yscale('log')

        axes[row,1].plot(x, u_e, 'k-', lw=2, label='Exact')
        axes[row,1].plot(x, u_n, 'b.-', markersize=3, label='IMEX')
        axes[row,1].axhline(u_star, ls='--', color='g', alpha=0.5, label=f'u*={u_star:.1f}')
        u_err = abs(np.max(np.abs(u_n)) - u_star) / abs(u_star) * 100
        axes[row,1].set_title(f'u (u* err={u_err:.2f}%, L1={L1_u*100:.1f}%)')
        axes[row,1].legend(); axes[row,1].set_xlabel('x')

        axes[row,2].plot(x, rho_e, 'k-', lw=2, label='Exact')
        axes[row,2].plot(x, rho_n, 'b.-', markersize=3, label='IMEX')
        axes[row,2].set_title(f'rho (L1={L1_rho*100:.1f}%)')
        axes[row,2].legend(); axes[row,2].set_xlabel('x')

        axes[row,3].axis('off')
        axes[row,3].text(0.05, 0.5,
                         f'{lbl}:\n\n'
                         f'  Exact:\n'
                         f'    p* = {p_star:.2e}\n'
                         f'    u* = {u_star:.2f}\n\n'
                         f'  Numerical:\n'
                         f'    u_max = {np.max(np.abs(u_n)):.2f}\n'
                         f'    err u* = {u_err:.2f}%\n\n'
                         f'  L1 errors:\n'
                         f'    p:   {L1_p*100:.2f}%\n'
                         f'    u:   {L1_u*100:.2f}%\n'
                         f'    rho: {L1_rho*100:.2f}%',
                         transform=axes[row,3].transAxes, fontsize=10, family='monospace')

    plt.suptitle('IMEX Solver — Exact Solution Comparison (Phase 1, 2-1, 2-2)',
                 fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig('results/phase1_2_exact_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: results/phase1_2_exact_comparison.png")
