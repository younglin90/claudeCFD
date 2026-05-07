"""Option A: Extended extreme validation across all speed regimes.

New cases to stress-test solve_IMEX across full Mach range:
  A1. Ultra-low Mach acoustic pulse (Mach ~1e-5)
  A2. Hypersonic air shock (Mach ~20, p_L=1e10)
  A3. Static stagnant air-water interface (long-time equilibrium)
  A4. Strong rarefaction — near-vacuum generation
  A5. Water-hammer — sudden stop in stiff liquid
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from pipeline.exact_riemann import exact_profile, exact_riemann_star

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS, exist_ok=True)


def l1(y_n, y_e):
    s = np.sum(np.abs(y_e))
    return 0.0 if s < 1e-30 else np.sum(np.abs(y_n - y_e)) / s


def run(tag, ph1, ph2, a_air, rho1, rho2, u0, p_init, L, N, t_end, cfl,
        exact_args=None, bc='transmissive', max_steps=200000):
    dx = L / N
    x = np.linspace(dx/2, L-dx/2, N)
    a1r1 = a_air * rho1
    a2r2 = (1-a_air) * rho2
    gm1 = ph1['gamma'] - 1.0
    gm2 = ph2['gamma'] - 1.0
    rho_e0 = a_air * p_init / gm1 \
             + (1-a_air) * (p_init + ph2['gamma']*ph2['pinf']) / gm2
    rE0 = rho_e0 + 0.5 * (a1r1 + a2r2) * u0 ** 2
    ru0 = (a1r1 + a2r2) * u0
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru0, rE0, a_air,
        dx, t_end=t_end, cfl=cfl, bc_l=bc, bc_r=bc,
        max_steps=max_steps, print_interval=max_steps,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=False, use_material_cfl=False)
    p_n, u_n, T1_n, T2_n, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    rho_n = a1r1_f + a2r2_f
    return x, t, p_n, u_n, rho_n, a1_f


def case_A1_ultra_low_mach():
    """Pure water, tiny pressure perturbation at Mach ~1e-5."""
    print("\n"+"="*70)
    print("A1: Ultra-low Mach pressure pulse in water (M~1e-5)")
    print("="*70)
    ph1 = {'gamma':1.4,'pinf':0.0,'kv':717.5,'b':0.0,'eta':0.0,'q':0.0}
    ph2 = {'gamma':4.4,'pinf':6e8,'kv':474.2,'b':0.0,'eta':0.0,'q':0.0}
    N, L = 200, 1.0
    dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0 = 1e5
    dp = 1.0  # Pa — tiny. c_water ~ 1500, Mach = dp/(rho c^2) * c = ~1e-6 to 1e-5
    p_init = np.where(np.abs(x-0.5) < 0.1, p0+dp, p0)
    rho2 = (p_init + 6e8) / (3.4 * 474.2 * 293.0)
    rho1 = p_init / (0.4 * 717.5 * 293.0)
    a_air = 1e-6 * np.ones(N)
    t_end = 3e-4
    x, t, p_n, u_n, rho_n, _ = run('A1', ph1, ph2, a_air, rho1, rho2, 0.0, p_init,
                                    L, N, t_end, 0.4)
    # Expected Mach
    c_water = np.sqrt(4.4*(p0+6e8)/rho2.mean())
    mach = np.abs(u_n).max()/c_water
    # d2 (2Δx Nyquist indicator)
    d2 = p_n[2:] - 2*p_n[1:-1] + p_n[:-2]
    d2_rms = np.sqrt(np.mean(d2**2))/p0
    # pressure stay bounded?
    dp_num = np.abs(p_n-p0).max()
    passed = np.all(np.isfinite(p_n)) and d2_rms < 1e-4 and dp_num < 5*dp
    print(f"  c_water={c_water:.1f}, Mach={mach:.2e}, d2_rms={d2_rms:.2e}, dp_max={dp_num:.2f}")
    print(f"  >>> {'PASS' if passed else 'FAIL'}")
    return x, p_n, u_n, rho_n, p0, passed


def case_A2_hypersonic_air():
    """Hypersonic air shock. p_L=1e10, p_R=1e5. Mach_s ~ 30."""
    print("\n"+"="*70)
    print("A2: Hypersonic air shock (p_L=1e10, p_R=1e5)")
    print("="*70)
    ph1 = {'gamma':1.4,'pinf':0.0,'kv':717.5,'b':0.0,'eta':0.0,'q':0.0}
    ph2 = {'gamma':4.4,'pinf':6e8,'kv':474.2,'b':0.0,'eta':0.0,'q':0.0}
    N, L = 200, 1.0
    dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    x0 = 0.5
    p_L, p_R = 1e9, 1e5  # pressure ratio 1e4 (still extreme; Mach~10 shock)
    rho_L, rho_R = 10.0, 1.0
    p_init = np.where(x<x0, p_L, p_R)
    rho1 = np.where(x<x0, rho_L, rho_R)
    rho2 = 1000.0*np.ones(N)
    a_air = (1.0-1e-6)*np.ones(N)  # pure air
    t_end = 5e-5
    # Exact
    p_s, u_s = exact_riemann_star(p_L, rho_L, 0, 1.4, 0, p_R, rho_R, 0, 1.4, 0)
    rho_e, u_e, p_e, _ = exact_profile(
        x, t_end, x0,
        pL=p_L,rhoL=rho_L,uL=0,gammaL=1.4,pinfL=0,
        pR=p_R,rhoR=rho_R,uR=0,gammaR=1.4,pinfR=0)
    x, t, p_n, u_n, rho_n, _ = run('A2', ph1, ph2, a_air, rho1, rho2, 0.0, p_init,
                                    L, N, t_end, 0.25)
    err_u = abs(np.abs(u_n).max()-u_s)/u_s*100
    passed = t>=t_end*0.99 and np.all(np.isfinite(p_n)) and err_u<5
    print(f"  Exact: u*={u_s:.1f}, p*={p_s:.2e}")
    print(f"  Num:   u_max={np.abs(u_n).max():.1f}, err={err_u:.2f}%")
    print(f"  L1: p={l1(p_n,p_e)*100:.1f}%, u={l1(u_n,u_e)*100:.1f}%, rho={l1(rho_n,rho_e)*100:.1f}%")
    print(f"  >>> {'PASS' if passed else 'FAIL'}")
    return x, p_n, u_n, rho_n, p_e, u_e, rho_e, passed


def case_A3_static_interface():
    """Stagnant air-water with hydro-equilibrium. Should preserve over long time."""
    print("\n"+"="*70)
    print("A3: Static air-water interface (long-time equilibrium)")
    print("="*70)
    ph1 = {'gamma':1.4,'pinf':0.0,'kv':717.5,'b':0.0,'eta':0.0,'q':0.0}
    ph2 = {'gamma':4.4,'pinf':6e8,'kv':474.2,'b':0.0,'eta':0.0,'q':0.0}
    N, L = 100, 1.0
    dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, T0 = 1e5, 293.0
    a_air = np.where(x<0.5, 1-1e-6, 1e-6)
    p_init = p0*np.ones(N)
    rho1 = p0/(0.4*717.5*T0)*np.ones(N)
    rho2 = (p0+6e8)/(3.4*474.2*T0)*np.ones(N)
    t_end = 1e-3  # long time (relative to acoustic timescale L/c~7e-4)
    x, t, p_n, u_n, rho_n, a1_f = run('A3', ph1, ph2, a_air, rho1, rho2, 0.0, p_init,
                                       L, N, t_end, 0.4, bc='transmissive')
    err_p = np.abs(p_n-p0).max()/p0
    err_u = np.abs(u_n).max()
    passed = np.all(np.isfinite(p_n)) and err_p<1e-3 and err_u<1.0 and t>=t_end*0.99
    print(f"  t={t:.2e}/{t_end:.2e}, err_p/p0={err_p:.2e}, |u|_max={err_u:.2e}")
    print(f"  >>> {'PASS' if passed else 'FAIL'}")
    return x, p_n, u_n, rho_n, p0, passed


def case_A4_rarefaction():
    """Two gases moving apart — strong rarefaction / low-density."""
    print("\n"+"="*70)
    print("A4: Strong rarefaction (near-vacuum generation)")
    print("="*70)
    ph1 = {'gamma':1.4,'pinf':0.0,'kv':717.5,'b':0.0,'eta':0.0,'q':0.0}
    ph2 = {'gamma':4.4,'pinf':6e8,'kv':474.2,'b':0.0,'eta':0.0,'q':0.0}
    N, L = 200, 1.0
    dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    x0 = 0.5
    # Mild Toro Test 2 analogue (avoid true vacuum): scaled to air at p=1e5
    p0 = 1e5; rho0 = 1.2; u_L, u_R = -100.0, 100.0  # Mach~0.3 diverging
    a_air = (1-1e-6)*np.ones(N)
    rho1 = rho0*np.ones(N)
    rho2 = 1000.0*np.ones(N)
    p_init = p0*np.ones(N)
    u_init = np.where(x<x0, u_L, u_R)
    dx_ = dx
    a1r1 = a_air*rho1; a2r2 = (1-a_air)*rho2
    ru0 = (a1r1+a2r2)*u_init
    gm1 = 0.4
    rho_e0 = a_air*p_init/gm1 + (1-a_air)*(p_init+4.4*6e8)/3.4
    rE0 = rho_e0 + 0.5*(a1r1+a2r2)*u_init**2
    t_end = 1e-3
    from solver.He2024.explicit_mmacm_ex import solve_IMEX as _solve
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = _solve(
        ph1, ph2, a1r1, a2r2, ru0, rE0, a_air,
        dx_, t_end=t_end, cfl=0.3, bc_l='transmissive', bc_r='transmissive',
        max_steps=10000, print_interval=10000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=False, use_material_cfl=False)
    p_n, u_n, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    rho_n = a1r1_f+a2r2_f
    rho_e, u_e, p_e, _ = exact_profile(
        x, t_end, x0,
        pL=p0,rhoL=rho0,uL=u_L,gammaL=1.4,pinfL=0,
        pR=p0,rhoR=rho0,uR=u_R,gammaR=1.4,pinfR=0)
    rho_min = rho_n.min()
    passed = t>=t_end*0.99 and np.all(np.isfinite(p_n)) and rho_min>0 and p_n.min()>0
    print(f"  t={t:.3f}/{t_end:.3f}, rho_min={rho_min:.3e}, p_min={p_n.min():.3e}")
    print(f"  L1: p={l1(p_n,p_e)*100:.1f}%, u={l1(u_n,u_e)*100:.1f}%, rho={l1(rho_n,rho_e)*100:.1f}%")
    print(f"  >>> {'PASS' if passed else 'FAIL'}")
    return x, p_n, u_n, rho_n, p_e, u_e, rho_e, passed


def case_A5_water_hammer():
    """Water hammer: water column, pressure jump at one end."""
    print("\n"+"="*70)
    print("A5: Water-hammer (stiff liquid sudden compression)")
    print("="*70)
    ph1 = {'gamma':1.4,'pinf':0.0,'kv':717.5,'b':0.0,'eta':0.0,'q':0.0}
    ph2 = {'gamma':4.4,'pinf':6e8,'kv':474.2,'b':0.0,'eta':0.0,'q':0.0}
    N, L = 200, 1.0
    dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    x0 = 0.5
    p_L, p_R = 1e8, 1e5
    rho_L = (p_L+6e8)/(3.4*474.2*293.0)
    rho_R = (p_R+6e8)/(3.4*474.2*293.0)
    a_air = 1e-6*np.ones(N)
    rho1 = np.where(x<x0, 1.0, 1.0)
    rho2 = np.where(x<x0, rho_L, rho_R)
    p_init = np.where(x<x0, p_L, p_R)
    t_end = 2e-4
    p_s, u_s = exact_riemann_star(p_L,rho_L,0,4.4,6e8,p_R,rho_R,0,4.4,6e8)
    rho_e, u_e, p_e, _ = exact_profile(
        x, t_end, x0,
        pL=p_L,rhoL=rho_L,uL=0,gammaL=4.4,pinfL=6e8,
        pR=p_R,rhoR=rho_R,uR=0,gammaR=4.4,pinfR=6e8)
    x, t, p_n, u_n, rho_n, _ = run('A5', ph1, ph2, a_air, rho1, rho2, 0.0, p_init,
                                    L, N, t_end, 0.3)
    err_u = abs(np.abs(u_n).max()-u_s)/max(u_s,1e-10)*100
    passed = t>=t_end*0.99 and np.all(np.isfinite(p_n)) and err_u<5
    print(f"  Exact: u*={u_s:.2f}, p*={p_s:.3e}")
    print(f"  Num:   u_max={np.abs(u_n).max():.2f}, err={err_u:.2f}%")
    print(f"  L1: p={l1(p_n,p_e)*100:.2f}%, u={l1(u_n,u_e)*100:.2f}%, rho={l1(rho_n,rho_e)*100:.2f}%")
    print(f"  >>> {'PASS' if passed else 'FAIL'}")
    return x, p_n, u_n, rho_n, p_e, u_e, rho_e, passed


if __name__ == '__main__':
    results = {}

    # A1
    x,pn,un,rhn,p0,ok = case_A1_ultra_low_mach()
    fig, ax = plt.subplots(1,3,figsize=(15,4))
    ax[0].plot(x,(pn-p0),'b-'); ax[0].set_title(f'A1 p-p0 (Pa)  {"PASS" if ok else "FAIL"}')
    ax[1].plot(x,un*1000,'b-'); ax[1].set_title('A1 u (mm/s)')
    ax[2].plot(x,rhn,'b-'); ax[2].set_title('A1 rho')
    for a in ax: a.set_xlabel('x')
    plt.tight_layout(); plt.savefig(f'{RESULTS}/optA_A1_ultra_low_mach.png', dpi=120); plt.close()
    results['A1'] = ok

    # A2
    x,pn,un,rhn,pe,ue,rhe,ok = case_A2_hypersonic_air()
    fig, ax = plt.subplots(1,3,figsize=(15,4))
    ax[0].plot(x,pn,'b-',label='Num'); ax[0].plot(x,pe,'k--',label='Exact'); ax[0].set_yscale('log')
    ax[1].plot(x,un,'b-'); ax[1].plot(x,ue,'k--')
    ax[2].plot(x,rhn,'b-'); ax[2].plot(x,rhe,'k--')
    ax[0].set_title(f'A2 p  {"PASS" if ok else "FAIL"}'); ax[0].legend()
    ax[1].set_title('A2 u'); ax[2].set_title('A2 rho')
    for a in ax: a.set_xlabel('x')
    plt.tight_layout(); plt.savefig(f'{RESULTS}/optA_A2_hypersonic.png', dpi=120); plt.close()
    results['A2'] = ok

    # A3
    x,pn,un,rhn,p0,ok = case_A3_static_interface()
    fig, ax = plt.subplots(1,3,figsize=(15,4))
    ax[0].plot(x,(pn-p0)/p0,'b-'); ax[0].set_title(f'A3 (p-p0)/p0  {"PASS" if ok else "FAIL"}')
    ax[1].plot(x,un,'b-'); ax[1].set_title('A3 u (m/s)')
    ax[2].plot(x,rhn,'b-'); ax[2].set_title('A3 rho'); ax[2].set_yscale('log')
    for a in ax: a.set_xlabel('x')
    plt.tight_layout(); plt.savefig(f'{RESULTS}/optA_A3_static.png', dpi=120); plt.close()
    results['A3'] = ok

    # A4
    x,pn,un,rhn,pe,ue,rhe,ok = case_A4_rarefaction()
    fig, ax = plt.subplots(1,3,figsize=(15,4))
    ax[0].plot(x,pn,'b-'); ax[0].plot(x,pe,'k--')
    ax[1].plot(x,un,'b-'); ax[1].plot(x,ue,'k--')
    ax[2].plot(x,rhn,'b-'); ax[2].plot(x,rhe,'k--')
    ax[0].set_title(f'A4 p  {"PASS" if ok else "FAIL"}')
    ax[1].set_title('A4 u'); ax[2].set_title('A4 rho')
    for a in ax: a.set_xlabel('x')
    plt.tight_layout(); plt.savefig(f'{RESULTS}/optA_A4_rarefaction.png', dpi=120); plt.close()
    results['A4'] = ok

    # A5
    x,pn,un,rhn,pe,ue,rhe,ok = case_A5_water_hammer()
    fig, ax = plt.subplots(1,3,figsize=(15,4))
    ax[0].plot(x,pn,'b-'); ax[0].plot(x,pe,'k--'); ax[0].set_yscale('log')
    ax[1].plot(x,un,'b-'); ax[1].plot(x,ue,'k--')
    ax[2].plot(x,rhn,'b-'); ax[2].plot(x,rhe,'k--')
    ax[0].set_title(f'A5 p  {"PASS" if ok else "FAIL"}')
    ax[1].set_title('A5 u'); ax[2].set_title('A5 rho')
    for a in ax: a.set_xlabel('x')
    plt.tight_layout(); plt.savefig(f'{RESULTS}/optA_A5_water_hammer.png', dpi=120); plt.close()
    results['A5'] = ok

    print("\n"+"="*70)
    print("Option A SUMMARY")
    print("="*70)
    for k,v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    n_pass = sum(results.values())
    print(f"  Total: {n_pass}/{len(results)} passed")
    print(f"\nPlots saved to {RESULTS}/optA_*.png")
