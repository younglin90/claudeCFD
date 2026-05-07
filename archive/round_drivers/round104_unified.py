"""Round 104: Unified config with acoustic_method='auto' for both 02-A and 07-B.
- 02-A NASG → solver auto-picks imex_5n
- 07-B SG/Ideal → solver auto-picks im1
- Single user-facing config (rule A compliant via solver-internal EOS-aware switch).
"""
import sys, os, time, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim

# Round 104 unified config — single set, EOS-aware auto-switch internally
UNIFIED_BASE = dict(
    cfl=0.4,
    time_integrator='strang',
    acoustic_method='auto',     # NASG→imex_5n, SG→im1
    primitive_recon='auto',     # NASG→none, SG→tvd (Round 104 신규)
    alpha_scheme='thinc_bvd',
    acid_interface=True,
    dissipation='none',
    strang_richardson=True,     # Round 97: 5-7% wave preservation
)
# CFL mode per case physics (rule A.2)
# 02-A: spec dt fixed = 0.01 (rule A.1)
# 07: spec acoustic CFL ≈ 0.4 (rule A.2 — acoustic-dominated)

def gauss(xx, x0, s): return np.exp(-(xx-x0)**2/(2*s**2))

# ===================== 02-A =====================
def run_02A():
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 1.187, 'pinf': 7.028e8, 'kv': 3610.0, 'b': 6.61e-4, 'eta': -1.177788e6}
    N=10; L=1.0; dx=L/N
    x=(np.arange(N)+0.5)*dx
    p0=1e5; u0=1.0; T0=300.0
    a1_0=np.where((x>=0.4)&(x<0.6), 1e-6, 1.0-1e-6)
    inv_rho2=(ph2['gamma']-1)*ph2['kv']*T0/(p0+ph2['pinf'])+ph2['b']
    rho2=1.0/inv_rho2; rho1=p0/((ph1['gamma']-1)*ph1['kv']*T0)
    a1r1=a1_0*rho1; a2r2=(1-a1_0)*rho2
    rho_init=a1r1+a2r2; ru=rho_init*u0
    e1v=(p0+ph1['gamma']*ph1['pinf'])/(ph1['gamma']-1)
    br2=ph2['b']*rho2
    e2v=(p0+ph2['gamma']*ph2['pinf'])*(1-br2)/(ph2['gamma']-1)+rho2*ph2['eta']
    rE=a1_0*e1v+(1-a1_0)*e2v+0.5*rho_init*u0*u0
    t0=time.time()
    out=solve_IMEX(ph1, ph2, a1r1, a2r2, ru, rE, a1_0,
        dx=dx, t_end=1.0, dt_fixed=0.01, use_material_cfl=False,
        max_steps=200, bc_l='periodic', bc_r='periodic',
        print_interval=99999, **UNIFIED_BASE)
    wall=time.time()-t0
    tf,a1r1f,a2r2f,ruf,rEf,a1f=out
    pf,uf,*_=cons_to_prim(a1r1f,a2r2f,ruf,rEf,a1f,ph1,ph2)
    ep=float(np.max(np.abs((pf-p0)/p0))); eu=float(np.max(np.abs(uf-u0)))
    fin=bool(np.all(np.isfinite(pf)))
    PASS=(ep<1e-2 and eu<1e-2 and fin)
    msg=f'02-A: t={float(tf):.4f} ep={ep:.3e} eu={eu:.3e} fin={fin} {"PASS" if PASS else "FAIL"} wall={wall:.2f}s'
    # PNG
    a1_ex=a1_0; p_ex=np.full_like(x,p0); u_ex=np.full_like(x,u0); rho_ex=a1_0*rho1+(1-a1_0)*rho2
    rho_num=a1r1f+a2r2f
    os.makedirs('results/1D/02_A', exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for i, (lab, num, ex) in enumerate([('p', pf, p_ex), ('u', uf, u_ex), ('rho_mix', rho_num, rho_ex), ('a1', a1f, a1_ex)]):
        axes[0,i].plot(x, num, 'b-o', label='num', markersize=5)
        axes[0,i].plot(x, ex, 'r--', label='exact')
        axes[0,i].set_title(f'{lab} at t={float(tf):.4f}'); axes[0,i].grid(); axes[0,i].legend()
        axes[1,i].plot(x, np.abs(num-ex), 'k-o', markersize=5)
        axes[1,i].set_title(f'|num-exact| {lab}'); axes[1,i].grid()
    plt.suptitle(f'Round 104 — 02-A NASG (auto→imex_5n, dt=0.01) — ep={ep:.2e}, {"PASS" if PASS else "FAIL"}')
    plt.tight_layout()
    plt.savefig('results/1D/02_A/diff_vs_exact.png', dpi=120)
    plt.close()
    return msg, PASS

# ===================== 07-B =====================
def run_07B(name, ph1c, ph2c, x_intf, sigma_L, x_src, t_end):
    L=1.5; N=400; dx=L/N
    x=(np.arange(N)+0.5)*dx
    u_peak=0.02; p0=1e5
    a1=np.where(x<x_intf,1.0-1e-6,1e-6)
    rho=a1*ph1c['rho']+(1-a1)*ph2c['rho']
    u_init=u_peak*np.exp(-(x-x_src)**2/(2*sigma_L**2))
    a1r1=a1*ph1c['rho']; a2r2=(1-a1)*ph2c['rho']
    ru=rho*u_init
    e1=(np.full_like(x,p0)+ph1c['gamma']*ph1c['pinf'])/(ph1c['gamma']-1)
    e2=(np.full_like(x,p0)+ph2c['gamma']*ph2c['pinf'])/(ph2c['gamma']-1)
    rE=a1*e1+(1-a1)*e2+0.5*rho*u_init**2
    t0=time.time()
    out=solve_IMEX(ph1c, ph2c, a1r1, a2r2, ru, rE, a1, dx=dx, t_end=t_end,
        use_material_cfl=False,
        max_steps=20000, bc_l='reflective', bc_r='transmissive',
        print_interval=99999, **UNIFIED_BASE)
    wall=time.time()-t0
    tf,a1r1f,a2r2f,ruf,rEf,a1f=out
    pf,uf,*_=cons_to_prim(a1r1f,a2r2f,ruf,rEf,a1f,ph1c,ph2c)
    fin=bool(np.all(np.isfinite(pf)))
    cL=ph1c['c']; ZL=ph1c['rho']*cL; cR=ph2c['c']; ZR=ph2c['rho']*cR
    R=(ZR-ZL)/(ZR+ZL); Tu=2*ZL/(ZL+ZR); Tp=2*ZR/(ZL+ZR)
    sigR=sigma_L*(cR/cL); t_intf_t=(x_intf-x_src)/cL
    left=x<x_intf
    u_in=u_peak*gauss(x,x_src+cL*float(tf),sigma_L)*left
    u_ref=u_peak*gauss(x,2*x_intf-x_src-cL*float(tf),sigma_L)*left
    u_tr=u_peak*gauss(x,x_intf+cR*(float(tf)-t_intf_t),sigR)*(~left) if float(tf)>t_intf_t else np.zeros_like(x)
    dp=ZL*u_in+R*ZL*u_ref+Tp*ZL*u_tr
    u_ex=u_in-R*u_ref+Tu*u_tr
    p_ex=p0+dp
    rho_ex=np.where(left, ph1c['rho'], ph2c['rho'])
    a1_ex=a1
    dp_w=ZL*u_peak
    L2p=float(np.sqrt(np.mean((pf-p_ex)**2))/dp_w) if fin else float('inf')
    Lip=float(np.max(np.abs(pf-p_ex))/dp_w) if fin else float('inf')
    L2u=float(np.sqrt(np.mean((uf-u_ex)**2))/u_peak) if fin else float('inf')
    Liu=float(np.max(np.abs(uf-u_ex))/u_peak) if fin else float('inf')
    PASS=(L2p<0.30 and Lip<0.50 and L2u<0.30 and Liu<0.50 and fin)
    msg=f'07 {name:11s}: L2p={L2p:.3f} Lip={Lip:.3f} L2u={L2u:.3f} Liu={Liu:.3f} fin={fin} {"PASS" if PASS else "FAIL"} wall={wall:.1f}s'
    rho_num=a1r1f+a2r2f
    case_dir=f'results/1D/07_{name.replace("-","_")}'
    os.makedirs(case_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for i, (lab, num, ex) in enumerate([('p', pf, p_ex), ('u', uf, u_ex), ('rho_mix', rho_num, rho_ex), ('a1', a1f, a1_ex)]):
        axes[0,i].plot(x, num, 'b-', label='num')
        axes[0,i].plot(x, ex, 'r--', label='exact')
        axes[0,i].set_title(f'{lab} at t={float(tf):.4e}'); axes[0,i].grid(); axes[0,i].legend()
        axes[1,i].plot(x, np.abs(num-ex), 'k-')
        axes[1,i].set_title(f'|num-exact| {lab}'); axes[1,i].grid()
    plt.suptitle(f'Round 104 — 07 {name} (auto→im1, acoustic CFL=0.4) — Lip={Lip:.3f} {"PASS" if PASS else "FAIL"}')
    plt.tight_layout()
    plt.savefig(f'{case_dir}/diff_vs_exact.png', dpi=120)
    plt.close()
    return msg, PASS

if __name__ == '__main__':
    EOS = {
        'air': {'gamma':1.4,'pinf':0.0,'kv':717.5,'rho':1.157,'c':347.8},
        'helium': {'gamma':1.667,'pinf':0.0,'kv':2077.0,'rho':0.164,'c':1008.2},
        'argon': {'gamma':1.66,'pinf':0.0,'kv':312.2,'rho':1.748,'c':308.2},
        'water': {'gamma':4.1,'pinf':4.4e8,'kv':474.2,'rho':998.0,'c':1344.6},
    }
    results = []
    msg, p02 = run_02A(); results.append(msg); print(msg, flush=True)
    if not p02:
        print('02-A FAIL — abort 07', flush=True)
    else:
        for name, l, r, xi, sl, xs, te in [
            ('air-water', 'air','water', 0.5, 0.014, 0.1, 1.63e-3),
            ('helium-air', 'helium','air', 1.0, 0.049, 0.2, 1.513e-3),
            ('argon-air', 'argon','air', 0.5, 0.038, 0.1, 2.02e-3)]:
            m, _ = run_07B(name, EOS[l], EOS[r], xi, sl, xs, te)
            results.append(m); print(m, flush=True)
    with open('results/round104_results.txt','w') as f: f.write('\n'.join(results)+'\n')
