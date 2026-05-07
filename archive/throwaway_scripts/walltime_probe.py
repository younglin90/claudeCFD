"""사전 wall time 측정: SSP2 + imex_5n_stage 통합 솔버.
20 step trial로 dt_wall 측정, 추정 총 시간 계산.
"""
import sys, time, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np
from solver.He2024.explicit_mmacm_ex import solve_IMEX

def probe_02A():
    """02-A NASG: dx=0.1, t_end=1.0, acoustic CFL high (advection-dominated)."""
    ph1 = {'gamma':1.4, 'pinf':0.0, 'kv':717.5}
    ph2 = {'gamma':1.187, 'pinf':7.028e8, 'kv':3610.0, 'b':6.61e-4, 'eta':-1.177788e6}
    N=10; dx=0.1
    x=(np.arange(N)+0.5)*dx
    a1=np.where((x>=0.4)&(x<0.6), 1e-6, 1.0-1e-6)
    p0=1e5; u0=1.0; T0=300.0
    rho1 = p0/((ph1['gamma']-1)*ph1['kv']*T0)
    inv_rho2 = (ph2['gamma']-1)*ph2['kv']*T0/(p0+ph2['pinf']) + ph2['b']
    rho2 = 1.0/inv_rho2
    a1r1=a1*rho1; a2r2=(1-a1)*rho2; rho=a1r1+a2r2; ru=rho*u0
    e1v=(p0+ph1['gamma']*ph1['pinf'])/(ph1['gamma']-1)
    br2=ph2['b']*rho2
    e2v=(p0+ph2['gamma']*ph2['pinf'])*(1-br2)/(ph2['gamma']-1) + rho2*ph2['eta']
    rE=a1*e1v+(1-a1)*e2v + 0.5*rho*u0*u0

    t0=time.time()
    out = solve_IMEX(ph1, ph2, a1r1, a2r2, ru, rE, a1, dx=dx, t_end=1e9,
                     cfl=200.0, use_material_cfl=False,
                     acoustic_method='imex_5n_stage',  # SSP2 stage + 5N NK
                     time_integrator='ssp222',
                     primitive_recon='tvd',
                     max_steps=20,
                     bc_l='periodic', bc_r='periodic',
                     print_interval=99999)
    wall = time.time()-t0
    t_f = float(out[0])
    dt_step = t_f/20.0
    print(f"02-A: 20 steps in {wall:.2f}s (dt_wall={wall/20:.3f}s), reached t={t_f:.3e} (dt_per_step={dt_step:.3e})")
    n_steps_full = int(np.ceil(1.0/max(dt_step,1e-12)))
    est = wall/20 * n_steps_full
    print(f"  est total: {n_steps_full} steps × {wall/20:.3f}s = {est:.1f}s ({'SKIP' if est>600 else 'RUN'})")
    return est

def probe_07():
    """07 argon-air: dx=3.75e-3, t_end=2.02e-3, acoustic CFL=0.4."""
    ph1 = {'gamma':1.66, 'pinf':0.0, 'kv':312.2}
    ph2 = {'gamma':1.4, 'pinf':0.0, 'kv':717.5}
    N=400; L=1.5; dx=L/N
    x=(np.arange(N)+0.5)*dx
    x_intf=0.5; x_src=0.1; sigma=0.038; u_peak=0.02
    a1=np.where(x<x_intf, 1.0-1e-6, 1e-6)
    u0 = u_peak*np.exp(-(x-x_src)**2/(2*sigma**2))*(x<x_intf)
    rho1=1.748; rho2=1.157; ZL=rho1*308.2
    p0=1e5; p_init=p0+ZL*u0
    rho1f = rho1*(1+(p_init-p0)/(rho1*308.2**2))
    rho2f = rho2*(1+(p_init-p0)/(rho2*347.8**2))
    a1r1=a1*rho1f; a2r2=(1-a1)*rho2f; rho=a1r1+a2r2; ru=rho*u0
    e1v=(p_init)/(ph1['gamma']-1)
    e2v=(p_init)/(ph2['gamma']-1)
    rE=a1*e1v+(1-a1)*e2v+0.5*rho*u0*u0

    t0=time.time()
    out = solve_IMEX(ph1, ph2, a1r1, a2r2, ru, rE, a1, dx=dx, t_end=1e9,
                     cfl=0.4, use_material_cfl=False,
                     acoustic_method='imex_5n_stage',
                     time_integrator='ssp222',
                     primitive_recon='tvd',
                     max_steps=20,
                     bc_l='reflective', bc_r='transmissive',
                     print_interval=99999)
    wall = time.time()-t0
    t_f = float(out[0])
    dt_step = t_f/20.0
    print(f"07 argon-air: 20 steps in {wall:.2f}s (dt_wall={wall/20:.3f}s), reached t={t_f:.3e}")
    n_steps_full = int(np.ceil(2.02e-3/max(dt_step,1e-12)))
    est = wall/20 * n_steps_full
    print(f"  est total: {n_steps_full} steps × {wall/20:.3f}s = {est:.1f}s ({'SKIP' if est>600 else 'RUN'})")
    return est

if __name__=='__main__':
    print("=== Wall time probe: SSP2 + imex_5n_stage ===\n")
    print("--- 02-A NASG (dx=0.1, t_end=1.0, CFL=200) ---")
    e02 = probe_02A()
    print()
    print("--- 07 argon-air (dx=3.75e-3, t_end=2.02e-3, CFL=0.4) ---")
    e07 = probe_07()
    print(f"\nSummary: 02-A est={e02:.1f}s, 07-Argon est={e07:.1f}s")
    print(f"  10min limit: 02-A {'PASS' if e02<600 else 'SKIP'}, 07-Argon {'PASS' if e07<600 else 'SKIP'}")
