"""Final summary: compare key physical quantities (p*, u*, shock pos) vs exact."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from pipeline.exact_riemann import exact_profile, exact_riemann_star


def find_shock_pos(x, p, threshold_ratio=0.5):
    """Find position where p crosses midpoint."""
    p_mid = 0.5 * (p.min() + p.max())
    if p[0] > p[-1]:  # descending (shock on right)
        idx = np.where(p < p_mid)[0]
        if len(idx) == 0: return np.nan
        return x[idx[0]]
    else:
        idx = np.where(p > p_mid)[0]
        if len(idx) == 0: return np.nan
        return x[idx[0]]


def run_case(name, x, x0, t_end, cfl, BC, init_fn, exact_kwargs, alpha_scheme='tvd',
             ph1=None, ph2=None):
    """Generic runner."""
    if ph1 is None:
        ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    if ph2 is None:
        ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}

    # Exact star
    p_star_e, u_star_e = exact_riemann_star(
        exact_kwargs['pL'], exact_kwargs['rhoL'], exact_kwargs['uL'],
        exact_kwargs['gammaL'], exact_kwargs['pinfL'],
        exact_kwargs['pR'], exact_kwargs['rhoR'], exact_kwargs['uR'],
        exact_kwargs['gammaR'], exact_kwargs['pinfR'])

    # Exact full profile
    rho_e, u_e, p_e, phase_e = exact_profile(x, t_end, x0, **exact_kwargs)

    # Numerical
    a1r1, a2r2, ru, rE, a_air = init_fn()
    dx = x[1] - x[0]
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a_air,
        dx, t_end=t_end, cfl=cfl, bc_l=BC, bc_r=BC,
        max_steps=100000, print_interval=10000,
        alpha_scheme=alpha_scheme, use_strang=True,
        use_defect_correction=False, use_material_cfl=False)
    p_num, u_num, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    rho_num = a1r1_f + a2r2_f

    # Key quantities
    u_max_num = u_num[np.argmax(np.abs(u_num))]
    p_star_num = p_num[np.argmin(np.abs(x - x0))]  # at initial contact
    # Also try: median p in central region
    center = (x > x0 - 0.1) & (x < x0 + 0.1)
    p_center_mean = p_num[center].mean()

    # Shock position: on the right side (after contact)
    right = x > x0
    if np.any(right):
        try:
            shock_pos_num = find_shock_pos(x[right], p_num[right])
            shock_pos_e = find_shock_pos(x[right], p_e[right])
        except:
            shock_pos_num = shock_pos_e = np.nan

    # L1 error
    dx_uniform = dx
    L1_p = np.sum(np.abs(p_num - p_e)) * dx_uniform / (np.abs(p_e).sum() * dx_uniform + 1e-30)
    L1_u = np.sum(np.abs(u_num - u_e)) * dx_uniform / (np.abs(u_e).sum() * dx_uniform + 1e-30)
    L1_rho = np.sum(np.abs(rho_num - rho_e)) * dx_uniform / (np.abs(rho_e).sum() * dx_uniform + 1e-30)

    print(f"\n{name}:")
    print(f"  Exact star:  p*={p_star_e:.4e}, u*={u_star_e:.2f}")
    print(f"  Numerical:   p_cnt={p_center_mean:.4e}, u_max={u_max_num:.2f}")
    print(f"  Error in u*: {abs(u_max_num - u_star_e)/max(abs(u_star_e), 1e-30)*100:.2f}%")
    print(f"  L1 errors: p={L1_p*100:.2f}%, u={L1_u*100:.2f}%, rho={L1_rho*100:.2f}%")
    print(f"  Shock pos: exact={shock_pos_e:.4f}, num={shock_pos_num:.4f}")

    return {
        'name': name, 'x': x, 'p_num': p_num, 'u_num': u_num, 'rho_num': rho_num,
        'p_e': p_e, 'u_e': u_e, 'rho_e': rho_e,
        'p_star_e': p_star_e, 'u_star_e': u_star_e,
        'u_max_num': u_max_num, 'p_center_num': p_center_mean,
        'L1_p': L1_p, 'L1_u': L1_u, 'L1_rho': L1_rho,
        'shock_num': shock_pos_num, 'shock_exact': shock_pos_e,
    }


if __name__ == '__main__':
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2_21 = {'gamma': 4.1, 'pinf': 4.4e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2_22 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}

    # ==== Phase 2-1 ====
    N = 200; L = 2.0; dx = L / N
    x21 = np.linspace(dx/2, L-dx/2, N); x0_21 = 0.5
    p_L, p_R = 1e9, 1e4; T0 = 300.0
    rho1_L = p_L / (0.4 * 717.5 * T0)
    rho2_R = (p_R + 4.4e8) / (3.1 * 474.2 * T0)
    def init_21():
        a_air = np.where(x21 < x0_21, 1.0-1e-6, 1e-6)
        a_water = 1.0 - a_air
        p_init = np.where(x21 < x0_21, p_L, p_R)
        rho1 = p_init / (0.4 * 717.5 * T0)
        rho2 = (p_init + 4.4e8) / (3.1 * 474.2 * T0)
        a1r1 = a_air * rho1; a2r2 = a_water * rho2
        rho = a1r1 + a2r2
        rho_e0 = a_air * p_init / 0.4 + a_water * (p_init + 4.1 * 4.4e8) / 3.1
        return a1r1, a2r2, np.zeros(N), rho_e0, a_air
    r21 = run_case('Phase 2-1 HP Air/LP Water', x21, x0_21, 8e-4, 0.4, 'transmissive',
                   init_21, dict(pL=p_L, rhoL=rho1_L, uL=0.0, gammaL=1.4, pinfL=0.0,
                                 pR=p_R, rhoR=rho2_R, uR=0.0, gammaR=4.1, pinfR=4.4e8),
                   ph1=ph1, ph2=ph2_21)

    # ==== Phase 2-2 ====
    N = 200; L = 1.0; dx = L / N
    x22 = np.linspace(dx/2, L-dx/2, N); x0_22 = 0.7
    p_L, p_R = 1e9, 1e5
    rho1_val = 50.0; rho2_val = 1000.0
    def init_22():
        a_air = np.where(x22 < x0_22, 1e-6, 1.0-1e-6)
        a_water = 1.0 - a_air
        p_init = np.where(x22 < x0_22, p_L, p_R)
        rho1 = rho1_val * np.ones(len(x22)); rho2 = rho2_val * np.ones(len(x22))
        a1r1 = a_air * rho1; a2r2 = a_water * rho2
        rho_e0 = a_air * p_init / 0.4 + a_water * (p_init + 4.4 * 6e8) / 3.4
        return a1r1, a2r2, np.zeros(len(x22)), rho_e0, a_air
    r22 = run_case('Phase 2-2 HP Water/LP Air', x22, x0_22, 2.29e-4, 0.25, 'transmissive',
                   init_22, dict(pL=p_L, rhoL=rho2_val, uL=0.0, gammaL=4.4, pinfL=6e8,
                                 pR=p_R, rhoR=rho1_val, uR=0.0, gammaR=1.4, pinfR=0.0),
                   ph1=ph1, ph2=ph2_22)

    # ==== Phase 3-1 pure air Ms=10 ====
    N = 200; L = 1.0; dx = L / N
    x31 = np.linspace(dx/2, L-dx/2, N); x0_31 = 0.1
    Ms = 10.0; p_pre = 1e5; rho_pre = 1.1574
    p_post = p_pre * (2*1.4*Ms**2 - 0.4) / 2.4
    rho_post = rho_pre * 2.4 * Ms**2 / (0.4 * Ms**2 + 2)
    c_pre = np.sqrt(1.4 * p_pre / rho_pre)
    u_post = 2 * (Ms**2 - 1) / (2.4 * Ms) * c_pre
    t_end_31 = 0.7 / (Ms * c_pre)
    def init_31():
        a_air = (1.0-1e-6) * np.ones(len(x31))
        a_water = 1.0 - a_air
        p_init = np.where(x31 < x0_31, p_post, p_pre)
        u_init = np.where(x31 < x0_31, u_post, 0.0)
        rho1 = np.where(x31 < x0_31, rho_post, rho_pre)
        rho2 = 1240.0 * np.ones(len(x31))
        a1r1 = a_air * rho1; a2r2 = a_water * rho2
        rho = a1r1 + a2r2
        rho_e0 = a_air * p_init / 0.4 + a_water * (p_init + 4.4 * 6e8) / 3.4
        return a1r1, a2r2, rho * u_init, rho_e0 + 0.5 * rho * u_init**2, a_air
    r31 = run_case('Phase 3-1 Air Ms=10', x31, x0_31, t_end_31, 0.5, 'transmissive',
                   init_31, dict(pL=p_post, rhoL=rho_post, uL=u_post, gammaL=1.4, pinfL=0.0,
                                 pR=p_pre, rhoR=rho_pre, uR=0.0, gammaR=1.4, pinfR=0.0),
                   ph1=ph1, ph2=ph2_22)

    # ============ Summary Table ============
    print("\n" + "="*75)
    print("EXACT vs NUMERICAL COMPARISON - SUMMARY TABLE")
    print("="*75)
    print(f"\n{'Case':<30s} | {'Exact u*':>10s} | {'Num u_max':>10s} | {'Err':>6s} | {'L1(p)':>7s} | {'L1(rho)':>8s}")
    print("-"*75)
    for r in [r21, r22, r31]:
        err_u = abs(r['u_max_num'] - r['u_star_e']) / max(abs(r['u_star_e']), 1e-30) * 100
        print(f"{r['name']:<30s} | {r['u_star_e']:10.2f} | {r['u_max_num']:10.2f} | {err_u:5.2f}% | {r['L1_p']*100:6.2f}% | {r['L1_rho']*100:7.2f}%")

    print("\nShock position comparison:")
    for r in [r21, r22, r31]:
        if not np.isnan(r['shock_num']):
            err = abs(r['shock_num'] - r['shock_exact']) * 100  # in cm (domain=1m)
            print(f"  {r['name']}: exact={r['shock_exact']:.4f}, num={r['shock_num']:.4f}, dx-units={err:.2f}")

    # Plot summary
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for i, r in enumerate([r21, r22, r31]):
        axes[i,0].plot(r['x'], r['p_e'], 'k-', lw=2, label='Exact')
        axes[i,0].plot(r['x'], r['p_num'], 'b.-', markersize=3, label='IMEX')
        axes[i,0].set_title(f'{r["name"]}: p (L1={r["L1_p"]*100:.1f}%)')
        axes[i,0].legend(fontsize=8); axes[i,0].set_xlabel('x')

        axes[i,1].plot(r['x'], r['u_e'], 'k-', lw=2, label='Exact')
        axes[i,1].plot(r['x'], r['u_num'], 'b.-', markersize=3, label='IMEX')
        err_u = abs(r['u_max_num'] - r['u_star_e']) / max(abs(r['u_star_e']), 1e-30) * 100
        axes[i,1].set_title(f'u (u*_err={err_u:.1f}%, L1={r["L1_u"]*100:.1f}%)')
        axes[i,1].legend(fontsize=8); axes[i,1].set_xlabel('x')

        axes[i,2].plot(r['x'], r['rho_e'], 'k-', lw=2, label='Exact')
        axes[i,2].plot(r['x'], r['rho_num'], 'b.-', markersize=3, label='IMEX')
        axes[i,2].set_title(f'rho (L1={r["L1_rho"]*100:.1f}%)')
        axes[i,2].legend(fontsize=8); axes[i,2].set_xlabel('x')

    plt.suptitle('IMEX Numerical vs Exact Riemann Solution (L1 errors)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/exact_summary.png', dpi=150)
    print("\nPlot saved: results/exact_summary.png")
