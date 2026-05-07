"""Compare numerical IMEX solutions with exact Riemann solutions.

Cases:
- Phase 2-1: HP Air / LP Water
- Phase 2-2: HP Water / LP Air
- Phase 2-4: Pressure wave in liquid
- Phase 3-1 (psi=0): Pure air Ms=10
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from pipeline.exact_riemann import exact_profile, exact_riemann_star

def l1_error(x, y_num, y_exact):
    dx = x[1] - x[0]
    return np.sum(np.abs(y_num - y_exact)) * dx / (np.max(np.abs(y_exact)) + 1e-30)

def linf_error(y_num, y_exact):
    return np.max(np.abs(y_num - y_exact)) / (np.max(np.abs(y_exact)) + 1e-30)

# ============ Phase 2-1 ============
def validate_phase2_1():
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.1, 'pinf': 4.4e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    N = 200; L = 2.0; dx = L / N
    x = np.linspace(dx/2, L - dx/2, N)
    x0 = 0.5; t_end = 8e-4
    T0 = 300.0
    p_L, p_R = 1e9, 1e4
    rho1_L = p_L / ((ph1['gamma']-1.0) * ph1['kv'] * T0)
    rho2_R = (p_R + ph2['pinf']) / ((ph2['gamma']-1.0) * ph2['kv'] * T0)

    # Exact
    rho_e, u_e, p_e, phase_e = exact_profile(
        x, t_end, x0,
        pL=p_L, rhoL=rho1_L, uL=0.0, gammaL=ph1['gamma'], pinfL=ph1['pinf'],
        pR=p_R, rhoR=rho2_R, uR=0.0, gammaR=ph2['gamma'], pinfR=ph2['pinf'])

    # Numerical
    a_air = np.where(x < x0, 1.0-1e-6, 1e-6)
    a_water = 1.0 - a_air
    p_init = np.where(x < x0, p_L, p_R)
    rho1 = p_init / ((ph1['gamma']-1.0) * ph1['kv'] * T0)
    rho2 = (p_init + ph2['pinf']) / ((ph2['gamma']-1.0) * ph2['kv'] * T0)
    a1r1 = a_air * rho1; a2r2 = a_water * rho2
    rho = a1r1 + a2r2; ru = np.zeros(N)
    gm1, gm2 = 0.4, 3.1
    rho_e0 = a_air * p_init / gm1 + a_water * (p_init + ph2['gamma'] * ph2['pinf']) / gm2
    rE = rho_e0

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a_air,
        dx, t_end=t_end, cfl=0.4, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=10000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=False, use_material_cfl=False)
    p_num, u_num, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    rho_num = a1r1_f + a2r2_f

    err = {'p': linf_error(p_num, p_e), 'u': linf_error(u_num, u_e),
           'rho': linf_error(rho_num, rho_e),
           'L1_p': l1_error(x, p_num, p_e), 'L1_u': l1_error(x, u_num, u_e)}
    print(f"Phase 2-1: Linf errors: p={err['p']:.3f}, u={err['u']:.3f}, rho={err['rho']:.3f}")
    print(f"           L1 errors:   p={err['L1_p']:.4f}, u={err['L1_u']:.4f}")
    return x, p_num, u_num, rho_num, p_e, u_e, rho_e, err

# ============ Phase 2-2 ============
def validate_phase2_2():
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    N = 200; L = 1.0; dx = L / N
    x = np.linspace(dx/2, L - dx/2, N)
    x0 = 0.7; t_end = 2.29e-4

    # Initial conditions: phase2-2 uses DIRECT density specification
    rho1_val = 50.0; rho2_val = 1000.0
    p_L, p_R = 1e9, 1e5
    rho_L = rho2_val  # water on left (high-p)
    rho_R = rho1_val  # air on right

    # Exact with water (SG) as left phase, air (ideal) as right
    rho_e, u_e, p_e, phase_e = exact_profile(
        x, t_end, x0,
        pL=p_L, rhoL=rho_L, uL=0.0, gammaL=ph2['gamma'], pinfL=ph2['pinf'],
        pR=p_R, rhoR=rho_R, uR=0.0, gammaR=ph1['gamma'], pinfR=ph1['pinf'])

    # Numerical
    a_air = np.where(x < x0, 1e-6, 1.0-1e-6)
    a_water = 1.0 - a_air
    p_init = np.where(x < x0, p_L, p_R)
    rho1 = rho1_val * np.ones(N); rho2 = rho2_val * np.ones(N)
    a1r1 = a_air * rho1; a2r2 = a_water * rho2
    rho = a1r1 + a2r2; ru = np.zeros(N)
    gm1, gm2 = 0.4, 3.4
    rho_e0 = a_air * p_init / gm1 + a_water * (p_init + ph2['gamma'] * ph2['pinf']) / gm2
    rE = rho_e0

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a_air,
        dx, t_end=t_end, cfl=0.25, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=10000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=False, use_material_cfl=False)
    p_num, u_num, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    rho_num = a1r1_f + a2r2_f

    err = {'p': linf_error(p_num, p_e), 'u': linf_error(u_num, u_e),
           'rho': linf_error(rho_num, rho_e),
           'L1_p': l1_error(x, p_num, p_e), 'L1_u': l1_error(x, u_num, u_e)}
    print(f"Phase 2-2: Linf errors: p={err['p']:.3f}, u={err['u']:.3f}, rho={err['rho']:.3f}")
    print(f"           L1 errors:   p={err['L1_p']:.4f}, u={err['L1_u']:.4f}")
    return x, p_num, u_num, rho_num, p_e, u_e, rho_e, err

# ============ Phase 2-4 ============
def validate_phase2_4():
    # Single water, Riemann problem p=1e9 | p=1e8
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    N = 100; L = 1.0; dx = L / N
    x = np.linspace(dx/2, L - dx/2, N)
    x0 = 0.5; t_end = 1.5e-4; T0 = 293.0
    p_L = 1e9; p_R = 1e8
    rho_L = (p_L + ph2['pinf']) / ((ph2['gamma']-1.0) * ph2['kv'] * T0)
    rho_R = (p_R + ph2['pinf']) / ((ph2['gamma']-1.0) * ph2['kv'] * T0)

    # Single-phase exact: both sides are SG water
    rho_e, u_e, p_e, phase_e = exact_profile(
        x, t_end, x0,
        pL=p_L, rhoL=rho_L, uL=0.0, gammaL=ph2['gamma'], pinfL=ph2['pinf'],
        pR=p_R, rhoR=rho_R, uR=0.0, gammaR=ph2['gamma'], pinfR=ph2['pinf'])
    p_star_exact, u_star_exact = exact_riemann_star(
        p_L, rho_L, 0.0, ph2['gamma'], ph2['pinf'],
        p_R, rho_R, 0.0, ph2['gamma'], ph2['pinf'])
    print(f"Phase 2-4 Exact: p*={p_star_exact:.4e}, u*={u_star_exact:.3f}")

    # Numerical
    a_air = 1e-6 * np.ones(N)
    a_water = 1.0 - a_air
    p_init = np.where(x < x0, p_L, p_R)
    rho1 = p_init / ((ph1['gamma']-1.0) * ph1['kv'] * T0)
    rho2 = (p_init + ph2['pinf']) / ((ph2['gamma']-1.0) * ph2['kv'] * T0)
    a1r1 = a_air * rho1; a2r2 = a_water * rho2
    rho = a1r1 + a2r2; ru = np.zeros(N)
    gm1, gm2 = 0.4, 3.4
    rho_e0 = a_air * p_init / gm1 + a_water * (p_init + ph2['gamma'] * ph2['pinf']) / gm2
    rE = rho_e0

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a_air,
        dx, t_end=t_end, cfl=0.25, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=10000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=False, use_material_cfl=False)
    p_num, u_num, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    rho_num = a1r1_f + a2r2_f

    err = {'p': linf_error(p_num, p_e), 'u': linf_error(u_num, u_e),
           'rho': linf_error(rho_num, rho_e),
           'L1_p': l1_error(x, p_num, p_e), 'L1_u': l1_error(x, u_num, u_e),
           'p_star_exact': p_star_exact, 'u_star_exact': u_star_exact}
    print(f"Phase 2-4: Linf errors: p={err['p']:.3f}, u={err['u']:.3f}, rho={err['rho']:.3f}")
    print(f"           L1 errors:   p={err['L1_p']:.4f}, u={err['L1_u']:.4f}")
    return x, p_num, u_num, rho_num, p_e, u_e, rho_e, err

# ============ Phase 3-1 pure air Ms=10 ============
def validate_phase3_1_air():
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    N = 200; L = 1.0; dx = L / N
    x = np.linspace(dx/2, L - dx/2, N)
    x0 = 0.1

    Ms = 10.0
    p_pre = 1e5; rho_pre = 1.1574
    p_post = p_pre * (2*1.4*Ms**2 - 0.4) / 2.4
    rho_post = rho_pre * 2.4 * Ms**2 / (0.4 * Ms**2 + 2)
    c_pre = np.sqrt(1.4 * p_pre / rho_pre)
    u_post = 2 * (Ms**2 - 1) / (2.4 * Ms) * c_pre
    V_s = Ms * c_pre
    t_end = 0.7 / V_s

    print(f"Phase 3-1 air Ms=10: V_s={V_s:.1f}, t_end={t_end:.4e}")
    print(f"  Post-shock: p={p_post:.3e}, rho={rho_post:.3f}, u={u_post:.2f}")

    # Exact: single-phase air Riemann
    rho_e, u_e, p_e, phase_e = exact_profile(
        x, t_end, x0,
        pL=p_post, rhoL=rho_post, uL=u_post, gammaL=1.4, pinfL=0.0,
        pR=p_pre, rhoR=rho_pre, uR=0.0, gammaR=1.4, pinfR=0.0)

    # Numerical: pure air (a1=1-eps)
    a_air = (1.0 - 1e-6) * np.ones(N)
    a_water = 1.0 - a_air
    p_init = np.where(x < x0, p_post, p_pre)
    u_init = np.where(x < x0, u_post, 0.0)
    rho1 = np.where(x < x0, rho_post, rho_pre)
    rho2_water = (p_pre + ph2['pinf']) / ((ph2['gamma']-1.0) * ph2['kv'] * 293.0)
    rho2 = rho2_water * np.ones(N)

    a1r1 = a_air * rho1; a2r2 = a_water * rho2
    rho = a1r1 + a2r2; ru = rho * u_init
    gm1, gm2 = 0.4, 3.4
    rho_e0 = a_air * p_init / gm1 + a_water * (p_init + ph2['gamma'] * ph2['pinf']) / gm2
    rE = rho_e0 + 0.5 * rho * u_init**2

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a_air,
        dx, t_end=t_end, cfl=0.5, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=10000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=False, use_material_cfl=False)
    p_num, u_num, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    rho_num = a1r1_f + a2r2_f

    err = {'p': linf_error(p_num, p_e), 'u': linf_error(u_num, u_e),
           'rho': linf_error(rho_num, rho_e),
           'L1_p': l1_error(x, p_num, p_e), 'L1_u': l1_error(x, u_num, u_e)}
    print(f"Phase 3-1 air: Linf errors: p={err['p']:.3f}, u={err['u']:.3f}, rho={err['rho']:.3f}")
    return x, p_num, u_num, rho_num, p_e, u_e, rho_e, err

# ============ Main ============
if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    print("="*70)
    print("Exact Riemann vs IMEX Numerical Comparison")
    print("="*70)

    cases = []
    print("\n--- Phase 2-1 ---")
    cases.append(('Phase 2-1: HP Air / LP Water', validate_phase2_1()))
    print("\n--- Phase 2-2 ---")
    cases.append(('Phase 2-2: HP Water / LP Air', validate_phase2_2()))
    print("\n--- Phase 2-4 ---")
    cases.append(('Phase 2-4: Pressure Wave in Liquid', validate_phase2_4()))
    print("\n--- Phase 3-1 pure air ---")
    cases.append(('Phase 3-1: Air Ms=10', validate_phase3_1_air()))

    # Plot
    fig, axes = plt.subplots(4, 3, figsize=(18, 18))
    for i, (title, (x, pn, un, rn, pe, ue, re, err)) in enumerate(cases):
        axes[i,0].plot(x, pe, 'k-', lw=2, label='Exact', alpha=0.7)
        axes[i,0].plot(x, pn, 'b.-', lw=1, label='Numerical', markersize=3)
        axes[i,0].set_title(f'{title}: p (Linf={err["p"]*100:.1f}%)')
        axes[i,0].legend(fontsize=8); axes[i,0].set_xlabel('x')

        axes[i,1].plot(x, ue, 'k-', lw=2, label='Exact', alpha=0.7)
        axes[i,1].plot(x, un, 'b.-', lw=1, label='Numerical', markersize=3)
        axes[i,1].set_title(f'u (Linf={err["u"]*100:.1f}%)')
        axes[i,1].legend(fontsize=8); axes[i,1].set_xlabel('x')

        axes[i,2].plot(x, re, 'k-', lw=2, label='Exact', alpha=0.7)
        axes[i,2].plot(x, rn, 'b.-', lw=1, label='Numerical', markersize=3)
        axes[i,2].set_title(f'rho (Linf={err["rho"]*100:.1f}%)')
        axes[i,2].legend(fontsize=8); axes[i,2].set_xlabel('x')

    plt.suptitle('IMEX Numerical vs Exact Riemann Solutions', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/exact_comparison.png', dpi=150)
    print("\nPlot saved: results/exact_comparison.png")

    print("\n" + "="*70)
    print("SUMMARY (Linf errors relative to exact)")
    print("="*70)
    for title, (x, pn, un, rn, pe, ue, re, err) in cases:
        print(f"{title}:")
        print(f"  p: {err['p']*100:.2f}%, u: {err['u']*100:.2f}%, rho: {err['rho']*100:.2f}%")
