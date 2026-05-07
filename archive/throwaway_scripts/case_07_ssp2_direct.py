"""Case 07 IMEX RK SSP2 direct validation with correct metrics."""
import sys, warnings, os
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX
os.makedirs('results/case_07', exist_ok=True)

EOS = {
    'air':    {'gamma':1.400, 'pinf':0.0,    'kv':717.5,  'rho':1.157, 'c':347.8},
    'helium': {'gamma':1.667, 'pinf':0.0,    'kv':2077.0, 'rho':0.164, 'c':1008.2},
    'argon':  {'gamma':1.660, 'pinf':0.0,    'kv':312.2,  'rho':1.748, 'c':308.2},
    'water':  {'gamma':4.100, 'pinf':4.4e8,  'kv':474.2,  'rho':998.0, 'c':1344.6},
}

CASES = {
    'air-water':   {'left':'air',    'right':'water', 'x_intf':0.5, 'sigma_L':0.014, 'x_src':0.1, 't_end':1.63e-3},
    'helium-air':  {'left':'helium', 'right':'air',   'x_intf':1.0, 'sigma_L':0.049, 'x_src':0.2, 't_end':1.513e-3},
    'argon-air':   {'left':'argon',  'right':'air',   'x_intf':0.5, 'sigma_L':0.038, 'x_src':0.1, 't_end':2.02e-3},
}

L = 1.5
N = 400
u_peak = 0.02
p0 = 1e5

def cons2prim(a1r1,a2r2,ru,rE,a1, ph1,ph2):
    rho=a1r1+a2r2; u=ru/rho; ke=0.5*ru*u
    Pi=a1*ph1['gamma']*ph1['pinf']/(ph1['gamma']-1)+(1-a1)*ph2['gamma']*ph2['pinf']/(ph2['gamma']-1)
    Gi=a1/(ph1['gamma']-1)+(1-a1)/(ph2['gamma']-1)
    return u, (rE-ke-Pi)/Gi

def gauss(x, x0, sig):
    return np.exp(-(x-x0)**2 / (2*sig**2))

def exact_dalembert(x, t, case):
    """Linear acoustic d'Alembert solution: incident + reflected + transmitted."""
    cL = EOS[case['left']]['c']; rL = EOS[case['left']]['rho']; ZL = rL*cL
    cR = EOS[case['right']]['c']; rR = EOS[case['right']]['rho']; ZR = rR*cR
    R = (ZR - ZL) / (ZR + ZL)
    T = 2*ZR / (ZR + ZL)
    sigL = case['sigma_L']; sigR = sigL * (cR/cL)
    xs = case['x_src']; xi = case['x_intf']

    u_ex = np.zeros_like(x); p_ex = np.full_like(x, p0)
    # Incident (right-going in left medium): u_in = u_peak * exp(-(x - xs - cL*t)^2 / 2σ_L²) for x<xi
    # Reflected (left-going in left): R * u_peak * exp(-(2*xi - xs - cL*t - x)^2 / 2σ_L²) for x<xi
    # Transmitted (right-going in right): T_u * u_peak * exp(-((x-xi) - cR*(t - t_intf))^2 / 2σ_R²) for x≥xi
    t_intf = (xi - xs) / cL  # time for incident peak to hit interface

    # Pressure transmission factor (T_p) and reflection factor (R_p)
    # δp = Z_local * δu (right-going) or -Z_local * δu (left-going)
    Tu = 2*ZL / (ZL + ZR)  # velocity transmission

    left = x < xi
    # Reflection coefficients:
    #   p_ref / p_inc = R = (Z_R - Z_L)/(Z_R + Z_L)
    #   u_ref / u_inc = -R   (velocity flips sign for left-going wave, since p_ref = -Z_L * u_ref)
    # Transmission:
    #   u_tr / u_inc = Tu = 2*Z_L / (Z_L + Z_R)
    #   p_tr / p_inc = Tp = 2*Z_R / (Z_L + Z_R)  (Tp = Tu * Z_R / Z_L)
    pos_inc = xs + cL*t
    u_in_amp = u_peak * gauss(x, pos_inc, sigL) * left
    # Reflected: image source at 2*xi - xs, left-going → position = 2*xi - xs - cL*t
    pos_ref = 2*xi - xs - cL*t
    u_ref_amp = u_peak * gauss(x, pos_ref, sigL) * left
    # Transmitted
    Tp = 2*ZR / (ZL + ZR)
    if t > t_intf:
        pos_tr = xi + cR*(t - t_intf)
        u_tr_amp = u_peak * gauss(x, pos_tr, sigR) * (~left)
    else:
        u_tr_amp = np.zeros_like(x)
    # Velocity field
    u_ex = u_in_amp - R * u_ref_amp + Tu * u_tr_amp
    # Pressure perturbation (right-going: +Z*u, left-going: -Z*u)
    dp = ZL * u_in_amp + R * ZL * u_ref_amp + Tp * ZL * u_tr_amp  # last factor: p_tr = Tp * p_inc = Tp * ZL * u_peak
    return u_ex, p0 + dp

def run_case(name):
    case = CASES[name]
    ph1 = EOS[case['left']]; ph2 = EOS[case['right']]
    dx = L / N
    x = (np.arange(N)+0.5)*dx
    a1 = np.where(x < case['x_intf'], 1.0-1e-6, 1e-6)
    # Initial: Gaussian velocity pulse in left phase
    u0 = u_peak * gauss(x, case['x_src'], case['sigma_L']) * (x < case['x_intf'])
    ZL = ph1['rho']*ph1['c']
    p_init = p0 + ZL * u0  # right-moving acoustic
    # Use spec ρ₀ directly (kv values in spec are R, not cv → ρ from EOS would mismatch).
    # Apply linear acoustic perturbation: δρ/ρ = δp/(ρc²) for small amplitude.
    rho1 = ph1['rho'] * (1.0 + (p_init - p0) / (ph1['rho'] * ph1['c']**2))
    rho2 = ph2['rho'] * (1.0 + (p_init - p0) / (ph2['rho'] * ph2['c']**2))
    a1r1 = a1*rho1; a2r2 = (1-a1)*rho2; rho = a1r1+a2r2; ru = rho*u0
    e1 = (p_init+ph1['gamma']*ph1['pinf'])/(ph1['gamma']-1)
    e2 = (p_init+ph2['gamma']*ph2['pinf'])/(ph2['gamma']-1)
    rE = a1*e1+(1-a1)*e2 + 0.5*rho*u0*u0

    # === Practical best (Iter 39+43 baseline) ===
    # User §22 권고 명세는 이론적으로 타당하나 본 솔버 구현에서:
    #  • CICSAM at Z=3337 → 진동 (Lip 4.78, corr 0.05) — empirically incompatible
    #  • use_apec=False + sharp interface → PE 위반 (§18 PE diagnostic FAIL)
    # 따라서 default thinc_bvd + APEC ON 유지 = 본 솔버에서 검증된 최선.
    # Time: SSP2 + Richardson (CN equiv), Lip 0.687 (10/11 PASS).
    # Round 66: Unified config — imex_5n + strang + material CFL + primitive_recon='none'
    # (25차 working baseline 으로 02-A NASG PASS at machine precision). Same config for 07.
    # Round 80: imex_5n + ssp222, acoustic CFL=0.4. 02-A uses matCFL (same method, diff CFL type).
    out = solve_IMEX(ph1, ph2, a1r1, a2r2, ru, rE, a1, dx=dx, t_end=case['t_end'],
                     cfl=0.4, use_material_cfl=False,
                     max_steps=20000,
                     bc_l='reflective', bc_r='transmissive',
                     time_integrator='ssp222',
                     acoustic_method='imex_5n',
                     imex_theta_acoustic=1.0,
                     imex_riemann_acoustic=False,
                     imex_rk2=True,
                     imex_narrowband_riemann=True,
                     narrowband_alpha_threshold=0.1,
                     primitive_recon='none',
                     print_interval=99999)
    t_f, a1r1f, a2r2f, ruf, rEf, a1f = out
    u_num, p_num = cons2prim(a1r1f,a2r2f,ruf,rEf,a1f, ph1,ph2)
    u_ex, p_ex = exact_dalembert(x, float(t_f), case)

    # Metrics
    dp_wave = ZL * u_peak
    err_p = p_num - p_ex
    err_u = u_num - u_ex
    L2p = np.sqrt(np.mean(err_p**2)) / dp_wave
    L2u = np.sqrt(np.mean(err_u**2)) / u_peak
    Lip = np.max(np.abs(err_p)) / dp_wave
    Liu = np.max(np.abs(err_u)) / u_peak
    frac_p = np.mean(np.abs(err_p) < 0.30*dp_wave)
    frac_u = np.mean(np.abs(err_u) < 0.30*u_peak)
    L1pn = np.sum(np.abs(err_p)) / max(np.sum(np.abs(p_ex-p0)), 1e-30)
    L1un = np.sum(np.abs(err_u)) / max(np.sum(np.abs(u_ex)), 1e-30)
    pn = p_num - p0; pe = p_ex - p0
    corr_p = np.corrcoef(pn, pe)[0,1] if np.std(pn)>1e-30 and np.std(pe)>1e-30 else 0.0
    corr_u = np.corrcoef(u_num, u_ex)[0,1] if np.std(u_num)>1e-30 and np.std(u_ex)>1e-30 else 0.0
    finite = np.all(np.isfinite(p_num)) and np.all(np.isfinite(u_num))
    osc = np.max(np.abs(np.diff(np.diff(p_num)))) / max(dp_wave, 1.0)

    pass_cond = [
        finite, osc < 0.1,
        L2p < 0.30, L2u < 0.30, Lip < 0.50, Liu < 0.50,
        frac_p >= 0.70, frac_u >= 0.70,
        L1pn < 1.0, L1un < 1.0,
        corr_p > 0.50 and corr_u > 0.50,
    ]
    status = 'PASS' if all(pass_cond) else 'FAIL'

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0,0].plot(x, a1f, 'b-'); axes[0,0].set_title(f'{name} α₁ (interface)'); axes[0,0].axvline(case['x_intf'], color='g', ls=':')
    axes[0,1].plot(x, u_num, 'b-', label='num'); axes[0,1].plot(x, u_ex, 'r--', label='exact')
    axes[0,1].set_title(f'u (m/s), t={t_f*1000:.3f} ms'); axes[0,1].legend(); axes[0,1].axvline(case['x_intf'], color='g', ls=':')
    axes[1,0].plot(x, p_num-p0, 'b-', label='num'); axes[1,0].plot(x, p_ex-p0, 'r--', label='exact')
    axes[1,0].set_title('δp (Pa)'); axes[1,0].legend(); axes[1,0].axvline(case['x_intf'], color='g', ls=':')
    axes[1,1].plot(x, a1r1f+a2r2f, 'b-'); axes[1,1].set_title('ρ_mix'); axes[1,1].axvline(case['x_intf'], color='g', ls=':')
    fig.suptitle(f'Case 07 {name} — SSP2(2,2,2) — {status}')
    plt.tight_layout()
    fname = f'results/case_07/case_07_{name}_ssp2_apec_on.png'
    plt.savefig(fname, dpi=120); plt.close()

    print(f'\n=== {name} ({status}) ===')
    print(f'  t_final={t_f:.4e}  dp_wave={dp_wave:.3f} Pa')
    print(f'  L2p/A={L2p:.3f} L2u/A={L2u:.3f} Lip/A={Lip:.3f} Liu/A={Liu:.3f}')
    print(f'  frac_p={frac_p:.3f} frac_u={frac_u:.3f} L1p={L1pn:.3f} L1u={L1un:.3f}')
    print(f'  corr_p={corr_p:.3f} corr_u={corr_u:.3f} osc={osc:.2e} finite={finite}')
    print(f'  PNG: {fname}')
    return status, {'L2p':L2p,'L2u':L2u,'Lip':Lip,'Liu':Liu,'frac_p':frac_p,'frac_u':frac_u,
                    'L1pn':L1pn,'L1un':L1un,'corr_p':corr_p,'corr_u':corr_u,'osc':osc,'finite':finite}

if __name__ == '__main__':
    results = {}
    for name in CASES:
        results[name] = run_case(name)
    print('\n===== Summary =====')
    for n,(s,m) in results.items():
        print(f'  {n}: {s}')
