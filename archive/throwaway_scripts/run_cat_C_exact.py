"""Category C (Subsonic Shock, M<1) 검증 with exact solution overlay + 고 CFL.

solve_IMEX semi-implicit → acoustic CFL 제약 없음. 각 case별 CFL=0.5~0.8 시도.
Toro SG exact Riemann overlay.
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from solver.He2024.eos_general import SGEOS, IdealEOS
from pipeline.exact_riemann import exact_profile

R = '/home/younglin90/work/claude_code/claudeCFD/results'
os.makedirs(f'{R}/cat_C_exact', exist_ok=True)


def _plot(x, a1, rho1, rho2, u, p, title, out, exact=None):
    fig, ax = plt.subplots(1, 5, figsize=(22, 4))
    for a, y, lbl in zip(ax, [a1, rho1, rho2, u, p], ['alpha_1', 'rho_1', 'rho_2', 'u', 'p']):
        a.plot(x, y, 'b-', lw=1.3, label='numerical')
        a.set_xlabel('x'); a.set_ylabel(lbl); a.grid(alpha=0.3)
    if exact:
        if 'rho_exact' in exact:
            ax[1].plot(x, exact['rho_exact'], 'r--', lw=1.0, label='exact')
            ax[1].legend(fontsize=7)
        if 'u_exact' in exact:
            ax[3].plot(x, exact['u_exact'], 'r--', lw=1.0, label='exact')
            ax[3].legend(fontsize=7)
        if 'p_exact' in exact:
            ax[4].plot(x, exact['p_exact'], 'r--', lw=1.0, label='exact')
            ax[4].legend(fontsize=7)
    fig.suptitle(f'{title}  blue=numerical, red-dashed=exact')
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f'  saved: {out}', flush=True)


def run_case(name, N, L, x0,
             a1_L, a1_R, rho1_L, rho1_R, rho2_L, rho2_R,
             u_L, u_R, p_L, p_R,
             eos1, eos2, ph1, ph2, t_end, cfl,
             exact_params=None, plot_name=None,
             alpha_scheme='thinc_bvd', use_mmacm_ex=True,
             bc='transmissive'):
    t0 = time.time()
    print(f'[{name}] N={N} L={L} CFL={cfl} t_end={t_end:.3e} scheme={alpha_scheme} mmacm={use_mmacm_ex}', flush=True)
    dx = L / N
    x = np.linspace(dx/2, L-dx/2, N)
    a1 = np.where(x < x0, a1_L, a1_R).astype(float)
    rho1 = np.where(x < x0, rho1_L, rho1_R).astype(float)
    rho2 = np.where(x < x0, rho2_L, rho2_R).astype(float)
    u = np.where(x < x0, u_L, u_R).astype(float)
    p = np.where(x < x0, p_L, p_R).astype(float)
    a1r1 = a1 * rho1; a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2; ru = rho * u
    e1 = eos1.energy(rho1, p); e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2
    try:
        t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
            ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
            dx=dx, t_end=t_end, cfl=cfl,
            bc_l=bc, bc_r=bc, max_steps=50000, print_interval=5000,
            alpha_scheme=alpha_scheme, use_mmacm_ex=use_mmacm_ex)
        p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    except Exception as e:
        print(f'  FAILED: {e}', flush=True); return
    wall = time.time() - t0
    # Exact
    exact = None
    if exact_params:
        rho_e_arr, u_e_arr, p_e_arr, _ = exact_profile(x, t, x0, **exact_params)
        exact = {'rho_exact': rho_e_arr, 'u_exact': u_e_arr, 'p_exact': p_e_arr}
        # Compute L1 errors
        err_p = np.mean(np.abs(p_f - p_e_arr)) / np.max(np.abs(p_e_arr))
        err_u = np.mean(np.abs(u_f - u_e_arr)) / max(np.max(np.abs(u_e_arr)), 1e-10)
        print(f'  L1 err_p={err_p:.3e} err_u={err_u:.3e} wall={wall:.1f}s', flush=True)
    else:
        print(f'  u_max={np.max(u_f):.3e} p_range=[{p_f.min():.3e},{p_f.max():.3e}] wall={wall:.1f}s', flush=True)
    _plot(x, a1_f, rho1_f, rho2_f, u_f, p_f, name, f'{R}/cat_C_exact/{plot_name}.png', exact=exact)


# =====================================================================
# 11_C: He-Air subsonic shock tube (Denner 2018 §7.5.1)
# =====================================================================
def run_11():
    eos1 = IdealEOS(gamma=1.66, kv=3116)   # He-like (L)
    eos2 = IdealEOS(gamma=1.4, kv=717.5)    # Air (R)
    ph1 = {'gamma': 1.66, 'pinf': 0, 'kv': 3116}
    ph2 = {'gamma': 1.4, 'pinf': 0, 'kv': 717.5}
    exact = {'pL': 2e5, 'rhoL': 3.57, 'uL': 0.0, 'gammaL': 1.66, 'pinfL': 0.0,
             'pR': 1e5, 'rhoR': 1.20, 'uR': 0.0, 'gammaR': 1.4, 'pinfR': 0.0}
    run_case('11_C He-Air subsonic shock (CFL=0.8)', N=400, L=1.0, x0=0.5,
             a1_L=1-1e-8, a1_R=1e-8, rho1_L=3.57, rho1_R=3.57,
             rho2_L=1.20, rho2_R=1.20, u_L=0.0, u_R=0.0,
             p_L=2e5, p_R=1e5, eos1=eos1, eos2=eos2, ph1=ph1, ph2=ph2,
             t_end=8e-4, cfl=0.8, exact_params=exact, plot_name='11_he_air')


# =====================================================================
# 12_C: Shock impedance matching (Denner 2018 §7.4.5)
# =====================================================================
def run_12():
    # Pre-existing shock Ms=1.22 in air, matched gas on right
    eos1 = IdealEOS(gamma=1.4, kv=720)
    eos2 = IdealEOS(gamma=1.648, kv=512.41)
    ph1 = {'gamma': 1.4, 'pinf': 0, 'kv': 720}
    ph2 = {'gamma': 1.648, 'pinf': 0, 'kv': 512.41}
    N, L = 500, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    # Pre-shock state in both phases
    rho_air_post = eos1.density(1.59060e5, 402.67)
    rho_air_pre = eos1.density(1.01325e5, 351.82)
    rho_gas_R = eos2.density(1.01325e5, 351.82)
    # Left of shock (x<0.05): post-shock state
    # 0.05 < x < 0.15: pre-shock air
    # x > 0.15: matched gas
    a1 = np.where(x < 0.15, 1-1e-8, 1e-8)
    p_init = np.where(x < 0.05, 1.59060e5, 1.01325e5)
    u_init = np.where(x < 0.05, 125.65, 0.0)
    rho1_arr = np.where(x < 0.05, rho_air_post, rho_air_pre)
    rho2_arr = np.full(N, rho_gas_R)
    a1r1 = a1 * rho1_arr; a2r2 = (1-a1) * rho2_arr
    rho = a1r1 + a2r2; ru = rho * u_init
    e1 = eos1.energy(rho1_arr, p_init); e2 = eos2.energy(rho2_arr, p_init)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u_init**2

    t0 = time.time()
    try:
        t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
            ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
            dx=dx, t_end=2e-4, cfl=0.5,
            bc_l='transmissive', bc_r='transmissive',
            max_steps=20000, print_interval=5000,
            alpha_scheme='thinc_bvd', use_mmacm_ex=True)
        p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    except Exception as e:
        print(f'12 FAILED: {e}', flush=True); return
    wall = time.time() - t0
    # Exact = piecewise plateaus
    p_exact = np.where(x < 0.05 + 125.65 * t, 1.59060e5, 1.01325e5)
    u_exact = np.where(x < 0.05 + 125.65 * t, 125.65, 0.0)
    rho_exact = np.where(x < 0.05 + 125.65 * t, rho_air_post, rho_air_pre)
    exact = {'rho_exact': rho_exact, 'u_exact': u_exact, 'p_exact': p_exact}
    # Measure reflected wave amplitude: left of shock (x<0.03) should be flat post-shock
    p_post_ref = 1.59060e5
    refl_amp = np.max(np.abs(p_f[x < 0.03] - p_post_ref)) / p_post_ref
    print(f'[12 Impedance shock] reflected amp={refl_amp*100:.2f}% wall={wall:.1f}s', flush=True)
    _plot(x, a1_f, rho1_f, rho2_f, u_f, p_f,
          f'12_C Shock impedance match (CFL=0.5) refl={refl_amp*100:.2f}%',
          f'{R}/cat_C_exact/12_shock_impedance.png', exact=exact)


# =====================================================================
# 13_C: Pressure discharge — Case A (Gas→Liquid) and Case B (Liquid→Gas)
# =====================================================================
def run_13_A():
    """Case A: High-p gas pushing low-p liquid."""
    # Air (SG with pinf=0 = ideal) on left, water SG on right
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = SGEOS(gamma=4.1, pinf=4.4e8, kv=474.2)
    ph1 = {'gamma': 1.4, 'pinf': 0, 'kv': 717.5}
    ph2 = {'gamma': 4.1, 'pinf': 4.4e8, 'kv': 474.2}
    # Gas HP, Liquid LP (assumed 10 bar / 1 bar per typical Kraposhin setup)
    p_L, p_R = 1e7, 1e5
    T0 = 308.2
    rho_L = eos1.density(p_L, T0)
    rho_R = eos2.density(p_R, T0)
    exact = {'pL': p_L, 'rhoL': rho_L, 'uL': 0.0, 'gammaL': 1.4, 'pinfL': 0.0,
             'pR': p_R, 'rhoR': rho_R, 'uR': 0.0, 'gammaR': 4.1, 'pinfR': 4.4e8}
    run_case('13_C Case A Gas→Liquid (CFL=0.6)', N=500, L=1.0, x0=0.5,
             a1_L=1-1e-8, a1_R=1e-8, rho1_L=rho_L, rho1_R=rho_L,
             rho2_L=rho_R, rho2_R=rho_R, u_L=0.0, u_R=0.0,
             p_L=p_L, p_R=p_R, eos1=eos1, eos2=eos2, ph1=ph1, ph2=ph2,
             t_end=4e-4, cfl=0.6, exact_params=exact, plot_name='13_A_gas_liquid',
             alpha_scheme='thinc_bvd', use_mmacm_ex=True)


def run_13_B():
    """Case B: HP liquid pushing LP gas. Low-Mach in liquid (~0.002)."""
    eos1 = SGEOS(gamma=4.1, pinf=4.4e8, kv=474.2)  # water (L)
    eos2 = IdealEOS(gamma=1.4, kv=717.5)  # gas (R)
    ph1 = {'gamma': 4.1, 'pinf': 4.4e8, 'kv': 474.2}
    ph2 = {'gamma': 1.4, 'pinf': 0, 'kv': 717.5}
    p_L, p_R = 1e9, 5e8
    T0 = 308.2
    rho_L = eos1.density(p_L, T0)
    rho_R = eos2.density(p_R, T0)
    exact = {'pL': p_L, 'rhoL': rho_L, 'uL': 0.0, 'gammaL': 4.1, 'pinfL': 4.4e8,
             'pR': p_R, 'rhoR': rho_R, 'uR': 0.0, 'gammaR': 1.4, 'pinfR': 0.0}
    run_case('13_C Case B Liquid→Gas (CFL=0.5)', N=500, L=1.0, x0=0.5,
             a1_L=1-1e-8, a1_R=1e-8, rho1_L=rho_L, rho1_R=rho_L,
             rho2_L=rho_R, rho2_R=rho_R, u_L=0.0, u_R=0.0,
             p_L=p_L, p_R=p_R, eos1=eos1, eos2=eos2, ph1=ph1, ph2=ph2,
             t_end=2e-4, cfl=0.5, exact_params=exact, plot_name='13_B_liquid_gas',
             alpha_scheme='thinc_bvd', use_mmacm_ex=True)


# =====================================================================
# 26_G: Pure water pressure discontinuity (merged 14+26) — 2 tests
# =====================================================================
def run_26_A():
    """Test A: Pressure wave in water 1GPa/0.1GPa (mild ratio 10)."""
    eos = SGEOS(gamma=4.4, pinf=6e8, kv=474.2)
    ph = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2}
    p_L, p_R = 1e9, 1e8
    T0 = 293.0
    rho = eos.density(p_L, T0)  # both sides same rho roughly
    exact = {'pL': p_L, 'rhoL': eos.density(p_L, T0), 'uL': 0.0, 'gammaL': 4.4, 'pinfL': 6e8,
             'pR': p_R, 'rhoR': eos.density(p_R, T0), 'uR': 0.0, 'gammaR': 4.4, 'pinfR': 6e8}
    run_case('26_G water A 1GPa/0.1GPa (CFL=0.5)', N=400, L=1.0, x0=0.5,
             a1_L=1-1e-8, a1_R=1-1e-8,
             rho1_L=eos.density(p_L, T0), rho1_R=eos.density(p_R, T0),
             rho2_L=1.2, rho2_R=1.2, u_L=0.0, u_R=0.0,
             p_L=p_L, p_R=p_R, eos1=eos, eos2=IdealEOS(gamma=1.4, kv=717.5), ph1=ph, ph2={'gamma': 1.4, 'pinf': 0, 'kv': 717.5},
             t_end=2e-4, cfl=0.5, exact_params=exact, plot_name='26_A_water_mild',
             alpha_scheme='thinc_bvd', use_mmacm_ex=True)


def run_26_B():
    """Test B: Water hammer stiff SG 1e8/1e5."""
    eos = SGEOS(gamma=4.4, pinf=6e8, kv=474.2)
    ph = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2}
    p_L, p_R = 1e8, 1e5
    T0 = 293.0
    exact = {'pL': p_L, 'rhoL': eos.density(p_L, T0), 'uL': 0.0, 'gammaL': 4.4, 'pinfL': 6e8,
             'pR': p_R, 'rhoR': eos.density(p_R, T0), 'uR': 0.0, 'gammaR': 4.4, 'pinfR': 6e8}
    run_case('26_G water B hammer 1e8/1e5 (CFL=0.3)', N=200, L=1.0, x0=0.5,
             a1_L=1-1e-8, a1_R=1-1e-8,
             rho1_L=eos.density(p_L, T0), rho1_R=eos.density(p_R, T0),
             rho2_L=1.2, rho2_R=1.2, u_L=0.0, u_R=0.0,
             p_L=p_L, p_R=p_R, eos1=eos, eos2=IdealEOS(gamma=1.4, kv=717.5), ph1=ph, ph2={'gamma': 1.4, 'pinf': 0, 'kv': 717.5},
             t_end=2e-4, cfl=0.3, exact_params=exact, plot_name='26_B_water_hammer',
             alpha_scheme='thinc_bvd', use_mmacm_ex=True)


CASES = {'11': run_11, '12': run_12, '13A': run_13_A, '13B': run_13_B,
         '26A': run_26_A, '26B': run_26_B}

if __name__ == '__main__':
    key = sys.argv[1]
    CASES[key]()
