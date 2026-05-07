"""Case 07 with same solver as 02 (imex_5n 5N coupled NK direct).
Test all 3 sub-cases. PASS criteria from spec §PASS Round 17.
"""
import sys, warnings, os, time
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
os.makedirs('results/case_07', exist_ok=True)

EOS = {
    'air':    {'gamma':1.400, 'pinf':0.0,    'kv':717.5,  'rho':1.157, 'c':347.8},
    'helium': {'gamma':1.667, 'pinf':0.0,    'kv':2077.0, 'rho':0.164, 'c':1008.2},
    'argon':  {'gamma':1.660, 'pinf':0.0,    'kv':312.2,  'rho':1.748, 'c':308.2},
    'water':  {'gamma':4.100, 'pinf':4.4e8,  'kv':474.2,  'rho':998.0, 'c':1344.6},
}
CASES = {
    'argon-air':   {'left':'argon',  'right':'air',   'x_intf':0.5, 'sigma_L':0.038, 'x_src':0.1, 't_end':2.02e-3},
    'helium-air':  {'left':'helium', 'right':'air',   'x_intf':1.0, 'sigma_L':0.049, 'x_src':0.2, 't_end':1.513e-3},
    'air-water':   {'left':'air',    'right':'water', 'x_intf':0.5, 'sigma_L':0.014, 'x_src':0.1, 't_end':1.63e-3},
}

L = 1.5; N = 400; u_peak = 0.02; p0 = 1e5

def gauss(x, x0, sig): return np.exp(-(x-x0)**2/(2*sig**2))

def exact_dalembert(x, t, case):
    cL=EOS[case['left']]['c']; rL=EOS[case['left']]['rho']; ZL=rL*cL
    cR=EOS[case['right']]['c']; rR=EOS[case['right']]['rho']; ZR=rR*cR
    R=(ZR-ZL)/(ZR+ZL); Tu=2*ZL/(ZL+ZR); Tp=2*ZR/(ZL+ZR)
    sigL=case['sigma_L']; sigR=sigL*(cR/cL); xs=case['x_src']; xi=case['x_intf']
    t_intf=(xi-xs)/cL
    left=x<xi
    pos_inc=xs+cL*t; pos_ref=2*xi-xs-cL*t
    u_in=u_peak*gauss(x,pos_inc,sigL)*left
    u_ref=u_peak*gauss(x,pos_ref,sigL)*left
    if t > t_intf:
        pos_tr=xi+cR*(t-t_intf)
        u_tr=u_peak*gauss(x,pos_tr,sigR)*(~left)
    else:
        u_tr=np.zeros_like(x)
    u_ex = u_in - R*u_ref + Tu*u_tr
    dp = ZL*u_in + R*ZL*u_ref + Tp*ZL*u_tr
    return u_ex, p0+dp

def run_case(name):
    case = CASES[name]
    ph1 = EOS[case['left']]; ph2 = EOS[case['right']]
    dx = L/N
    x = (np.arange(N)+0.5)*dx
    a1 = np.where(x < case['x_intf'], 1.0-1e-6, 1e-6)
    u0 = u_peak*gauss(x, case['x_src'], case['sigma_L'])*(x < case['x_intf'])
    ZL = ph1['rho']*ph1['c']
    p_init = p0 + ZL*u0
    rho1 = ph1['rho']*(1.0 + (p_init-p0)/(ph1['rho']*ph1['c']**2))
    rho2 = ph2['rho']*(1.0 + (p_init-p0)/(ph2['rho']*ph2['c']**2))
    a1r1 = a1*rho1; a2r2 = (1-a1)*rho2; rho = a1r1+a2r2; ru = rho*u0
    e1 = (p_init+ph1['gamma']*ph1['pinf'])/(ph1['gamma']-1)
    e2 = (p_init+ph2['gamma']*ph2['pinf'])/(ph2['gamma']-1)
    rE = a1*e1+(1-a1)*e2 + 0.5*rho*u0*u0

    print(f"\n>>> Running {name} with imex_5n (5N coupled NK)...")
    t0 = time.time()
    out = solve_IMEX(ph1, ph2, a1r1, a2r2, ru, rE, a1, dx=dx, t_end=case['t_end'],
                     cfl=0.4, max_steps=5000,
                     acoustic_method='imex_5n_stage',  # 통합 솔버 (SSP2 stage + 5N NK fast)
                     time_integrator='ssp222',
                     primitive_recon='tvd',
                     bc_l='reflective', bc_r='transmissive',
                     print_interval=200)
    wall = time.time()-t0
    t_f, a1r1f, a2r2f, ruf, rEf, a1f = out
    p_num, u_num, *_ = cons_to_prim(a1r1f, a2r2f, ruf, rEf, a1f, ph1, ph2)
    u_ex, p_ex = exact_dalembert(x, float(t_f), case)

    dp_wave = ZL*u_peak
    err_p = p_num - p_ex; err_u = u_num - u_ex
    L2p = float(np.sqrt(np.mean(err_p**2))/dp_wave)
    L2u = float(np.sqrt(np.mean(err_u**2))/u_peak)
    Lip = float(np.max(np.abs(err_p))/dp_wave)
    Liu = float(np.max(np.abs(err_u))/u_peak)
    frac_p = float(np.mean(np.abs(err_p) < 0.30*dp_wave))
    frac_u = float(np.mean(np.abs(err_u) < 0.30*u_peak))
    L1pn = float(np.sum(np.abs(err_p))/max(np.sum(np.abs(p_ex-p0)), 1e-30))
    L1un = float(np.sum(np.abs(err_u))/max(np.sum(np.abs(u_ex)), 1e-30))
    pn=p_num-p0; pe=p_ex-p0
    corr_p = float(np.corrcoef(pn,pe)[0,1]) if np.std(pn)>1e-30 and np.std(pe)>1e-30 else 0.0
    corr_u = float(np.corrcoef(u_num,u_ex)[0,1]) if np.std(u_num)>1e-30 and np.std(u_ex)>1e-30 else 0.0
    finite = bool(np.all(np.isfinite(p_num)) and np.all(np.isfinite(u_num)))
    osc = float(np.max(np.abs(np.diff(np.diff(p_num))))/max(dp_wave, 1.0))

    pc = [finite, osc<0.1, L2p<0.30, L2u<0.30, Lip<0.50, Liu<0.50,
          frac_p>=0.70, frac_u>=0.70, L1pn<1.0, L1un<1.0, corr_p>0.50 and corr_u>0.50]
    status = 'PASS' if all(pc) else 'FAIL'

    fig,ax=plt.subplots(2,2,figsize=(12,8))
    ax[0,0].plot(x,a1f,'b-'); ax[0,0].set_title(f'{name} α₁'); ax[0,0].axvline(case['x_intf'],color='g',ls=':')
    ax[0,1].plot(x,u_num,'b-',label='num'); ax[0,1].plot(x,u_ex,'r--',label='exact')
    ax[0,1].set_title(f'u (m/s) t={t_f*1000:.3f}ms'); ax[0,1].legend(); ax[0,1].axvline(case['x_intf'],color='g',ls=':')
    ax[1,0].plot(x,p_num-p0,'b-',label='num'); ax[1,0].plot(x,p_ex-p0,'r--',label='exact')
    ax[1,0].set_title('δp (Pa)'); ax[1,0].legend(); ax[1,0].axvline(case['x_intf'],color='g',ls=':')
    ax[1,1].plot(x,a1r1f+a2r2f,'b-'); ax[1,1].set_title('ρ_mix'); ax[1,1].axvline(case['x_intf'],color='g',ls=':')
    fig.suptitle(f'07 {name} — imex_5n (same as 02 solver) — {status}')
    plt.tight_layout()
    fname=f'results/case_07/case_07_{name}_imex5n_unified.png'
    plt.savefig(fname,dpi=120); plt.close()

    print(f"=== {name} ({status}, wall={wall:.1f}s) ===")
    print(f"  L2p/A={L2p:.3f} L2u/A={L2u:.3f} Lip/A={Lip:.3f} Liu/A={Liu:.3f}")
    print(f"  frac_p={frac_p:.3f} frac_u={frac_u:.3f} L1p={L1pn:.3f} L1u={L1un:.3f}")
    print(f"  corr_p={corr_p:.3f} corr_u={corr_u:.3f} osc={osc:.2e} finite={finite}")
    print(f"  PNG: {fname}")
    return status

if __name__ == '__main__':
    # Start with simplest (argon-air) — fast feedback. If too slow, abort.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='argon-air', help='argon-air|helium-air|air-water|all')
    args = parser.parse_args()
    if args.case == 'all':
        for name in ['argon-air', 'helium-air', 'air-water']:
            run_case(name)
    else:
        run_case(args.case)
