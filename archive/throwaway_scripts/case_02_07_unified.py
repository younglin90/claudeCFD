"""Round 89 unified driver — single config for both 02-A NASG and 07-B acoustic.

Config (rule A 통일):
  acoustic_method='im1', dissipation='project', acid_interface=True,
  time_integrator='strang', primitive_recon='thinc_bvd', alpha_scheme='thinc_bvd',
  acoustic CFL=0.4 (use_material_cfl=False)

This combination is NEW (not in attempts_catalog as Iter 78-87 used 'ssp222').
'project' uses NASG-aware general EOS for energy reconstruction.
"""
import sys, os, time, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
os.makedirs('results/case_02', exist_ok=True)
os.makedirs('results/case_07', exist_ok=True)

UNIFIED_OPTS = dict(
    cfl=0.4, use_material_cfl=False,
    time_integrator='strang',
    acoustic_method='im1',
    dissipation='project',
    acid_interface=True,
    primitive_recon='thinc_bvd',
    alpha_scheme='thinc_bvd',
    print_interval=99999,
)

# ===================== 02-A: NASG water + Ideal air PE advection =====================
def case_02A(max_steps=200, t_end=1.0):
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 1.187, 'pinf': 7.028e8, 'kv': 3610.0,
           'b': 6.61e-4, 'eta': -1.177788e6}
    N = 10; L = 1.0; dx = L/N
    x = (np.arange(N) + 0.5)*dx
    p0 = 1e5; u0 = 1.0; T0 = 300.0
    a1 = np.where((x >= 0.4) & (x < 0.6), 1e-6, 1.0 - 1e-6)

    def rho_nasg(p, T, eos):
        inv_rho = (eos['gamma']-1.0)*eos['kv']*T/(p + eos['pinf']) + eos['b']
        return 1.0/inv_rho
    def rho_ideal(p, T, eos):
        return p/((eos['gamma']-1.0)*eos['kv']*T)

    rho1 = rho_ideal(p0, T0, ph1)
    rho2 = rho_nasg(p0, T0, ph2)
    a1r1 = a1*rho1; a2r2 = (1.0 - a1)*rho2
    rho = a1r1 + a2r2; ru = rho*u0
    e1_v = (p0 + ph1['gamma']*ph1['pinf'])/(ph1['gamma']-1)
    br2 = ph2['b']*rho2
    e2_v = (p0 + ph2['gamma']*ph2['pinf'])*(1.0 - br2)/(ph2['gamma']-1) + rho2*ph2['eta']
    rE = a1*e1_v + (1-a1)*e2_v + 0.5*rho*u0*u0

    t0 = time.time()
    out = solve_IMEX(ph1, ph2, a1r1, a2r2, ru, rE, a1, dx=dx, t_end=t_end,
                     bc_l='periodic', bc_r='periodic',
                     max_steps=max_steps, **UNIFIED_OPTS)
    wall = time.time() - t0
    t_f, a1r1f, a2r2f, ruf, rEf, a1f = out
    p_f, u_f, *_ = cons_to_prim(a1r1f, a2r2f, ruf, rEf, a1f, ph1, ph2)
    err_p = float(np.max(np.abs((p_f - p0)/p0)))
    err_u = float(np.max(np.abs(u_f - u0)))
    finite = bool(np.all(np.isfinite(p_f)) and np.all(np.isfinite(u_f)))
    return dict(t_final=float(t_f), err_p=err_p, err_u=err_u, finite=finite, wall=wall, x=x, p=p_f, u=u_f)


# ===================== 07-B: Air-Water acoustic reflection (Z=3337) =====================
def case_07B_air_water(max_steps=20000):
    air   = {'gamma':1.400, 'pinf':0.0,    'kv':717.5,  'rho':1.157, 'c':347.8}
    water = {'gamma':4.100, 'pinf':4.4e8,  'kv':474.2,  'rho':998.0, 'c':1344.6}
    L = 1.5; N = 400; dx = L/N
    x = (np.arange(N) + 0.5)*dx
    u_peak = 0.02; p0 = 1e5
    x_intf = 0.5; sigma_L = 0.014; x_src = 0.1; t_end = 1.63e-3

    a1 = np.where(x < x_intf, 1.0-1e-6, 1e-6)  # phase1=air left, phase2=water right
    rho = a1*air['rho'] + (1-a1)*water['rho']
    u_init = u_peak * np.exp(-(x-x_src)**2 / (2*sigma_L**2))
    p_init = np.full_like(x, p0)
    a1r1 = a1*air['rho']; a2r2 = (1-a1)*water['rho']
    ru = rho*u_init
    e1 = (p_init + air['gamma']*air['pinf'])/(air['gamma']-1)
    e2 = (p_init + water['gamma']*water['pinf'])/(water['gamma']-1)
    rE = a1*e1 + (1-a1)*e2 + 0.5*rho*u_init**2

    t0 = time.time()
    out = solve_IMEX(air, water, a1r1, a2r2, ru, rE, a1, dx=dx, t_end=t_end,
                     bc_l='reflective', bc_r='transmissive',
                     max_steps=max_steps, **UNIFIED_OPTS)
    wall = time.time() - t0
    t_f, a1r1f, a2r2f, ruf, rEf, a1f = out
    p_f, u_f, *_ = cons_to_prim(a1r1f, a2r2f, ruf, rEf, a1f, air, water)
    finite = bool(np.all(np.isfinite(p_f)) and np.all(np.isfinite(u_f)))

    # Linear acoustic theory comparison
    cL=air['c']; rL=air['rho']; ZL=rL*cL
    cR=water['c']; rR=water['rho']; ZR=rR*cR
    R = (ZR-ZL)/(ZR+ZL); Tu = 2*ZL/(ZL+ZR); Tp = 2*ZR/(ZL+ZR)
    sigR = sigma_L*(cR/cL); t_intf = (x_intf-x_src)/cL
    def gauss(xx, x0, s): return np.exp(-(xx-x0)**2/(2*s**2))
    pos_inc = x_src + cL*float(t_f); pos_ref = 2*x_intf - x_src - cL*float(t_f)
    left = x < x_intf
    u_in = u_peak*gauss(x, pos_inc, sigma_L)*left
    u_ref = u_peak*gauss(x, pos_ref, sigma_L)*left
    if float(t_f) > t_intf:
        pos_tr = x_intf + cR*(float(t_f)-t_intf); u_tr = u_peak*gauss(x, pos_tr, sigR)*(~left)
    else:
        u_tr = np.zeros_like(x)
    u_ex = u_in - R*u_ref + Tu*u_tr
    dp = ZL*u_in + R*ZL*u_ref + Tp*ZL*u_tr
    p_ex = p0 + dp
    dp_wave = ZL*u_peak
    err_p = p_f - p_ex; err_u = u_f - u_ex
    L2p = float(np.sqrt(np.mean(err_p**2))/dp_wave)
    Lip = float(np.max(np.abs(err_p))/dp_wave)
    return dict(t_final=float(t_f), L2p=L2p, Lip=Lip, finite=finite, wall=wall,
                x=x, p=p_f, u=u_f, p_ex=p_ex, u_ex=u_ex)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'trial'
    if mode == 'trial':
        print("=== Round 89 trial: 02-A 100 steps + 07-B 100 steps ===")
        r02 = case_02A(max_steps=100, t_end=1.0)
        print(f"02-A trial(100step): wall={r02['wall']:.2f}s t={r02['t_final']:.2e} "
              f"err_p={r02['err_p']:.2e} err_u={r02['err_u']:.2e} finite={r02['finite']}")
        r07 = case_07B_air_water(max_steps=100)
        print(f"07-B trial(100step): wall={r07['wall']:.2f}s t={r07['t_final']:.2e} "
              f"L2p={r07['L2p']:.2e} Lip={r07['Lip']:.2e} finite={r07['finite']}")
    elif mode == '02':
        r = case_02A(max_steps=60000, t_end=1.0)
        print(f"02-A: wall={r['wall']:.1f}s t={r['t_final']:.4f} err_p={r['err_p']:.3e} "
              f"err_u={r['err_u']:.3e} finite={r['finite']}")
        PASS = r['err_p']<1e-2 and r['err_u']<1e-2 and r['finite']
        print(f"02-A: {'PASS' if PASS else 'FAIL'}")
    elif mode == '07':
        r = case_07B_air_water(max_steps=20000)
        print(f"07-B air-water: wall={r['wall']:.1f}s t={r['t_final']:.4e} "
              f"L2p={r['L2p']:.3f} Lip={r['Lip']:.3f} finite={r['finite']}")
        PASS = r['Lip']<0.5 and r['finite']
        print(f"07-B: {'PASS' if PASS else 'FAIL'} (Lip<0.5 target)")
