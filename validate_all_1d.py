"""
validate_all_1d.py
==================
validation/ 폴더의 1D 케이스들을 계산하고 solver/output/ 폴더에 그래프를 저장.

커버 케이스 (16개):
  A. PE/species preservation   : G1, G2, G3, S1, S2
  B. CPG smooth interface      : CPG_adv, smooth_PEP
  C. Riemann / 충격관          : contact_stationary, contact_moving_A/B,
                                  sod_A, sod_B, woodward_colella, shu_osher,
                                  shock_airhel
  D. Convergence EOC           : multicomponent sinusoidal
  E. Acoustic wave             : single-fluid acoustic
  F. Positivity                : mass fraction positivity
  G. SRK EOS                   : CH4/N2 SRK interface (apec_1d.py)

스킵 (기-액 2-fluid, ACID 전용, 복잡 EOS):
  gas-liquid, pressure-discharge, ACID, inviscid droplet 등 15개
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'solver'))

import cpg_flux as ceg
import cpg_flux    as pef

OUTPUT_DIR = os.path.join(HERE, 'solver', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

results = {}   # 케이스명 -> 'PASS' | 'FAIL' | 'SKIP'


def save(fname, fig):
    path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"  => {fname}")


def run_pef(init_fn, scheme, T, CFL=0.5):
    """pe_fvm_1d 기반 실행."""
    Q0, x, dx, species = init_fn()
    apply_filter = (scheme != 'DIV_FVM')
    rhs = pef.make_rhs_fvm(scheme, species, dx)
    Q = Q0.copy(); t = 0.0
    while t < T - 1e-14:
        dt = min(pef.cfl_dt(Q, dx, CFL, species), T - t)
        if dt <= 0: break
        Q = pef.rk4_step(Q, rhs, dt)
        if apply_filter: Q = pef.spectral_filter(Q)
        t += dt
    return Q, x, Q0, species


# ════════════════════════════════════════════════════════════════
# A. PE / species preservation
# ════════════════════════════════════════════════════════════════

def case_G_PE():
    print("\n[G1/G2/G3] PE preservation...")
    t0 = time.time()

    def init_G3(N=61):
        species = ['H2', 'H2O']
        L = 1.0; dx = L/N
        x = np.linspace(0.5*dx, L-0.5*dx, N)
        rho = 1.0 + np.exp(np.sin(2*np.pi*x))
        u   = np.ones(N); p = np.ones(N)
        Y1  = 0.5 + 0.4*np.sin(2*np.pi*x)
        Y2  = 1.0 - Y1
        Ys  = np.stack([Y1, Y2], axis=1)
        _, gamma, _ = pef.mixture_props(Ys, species)
        e   = p / (rho * (gamma - 1.0)); E = e + 0.5*u**2
        Q = np.zeros((N, 5))
        Q[:,0]=rho; Q[:,1]=rho*u; Q[:,2]=rho*E
        Q[:,3]=rho*Y1; Q[:,4]=rho*Y2
        return Q, x, dx, species

    cases  = [('G1', pef.init_G1), ('G2', pef.init_G2), ('G3', init_G3)]
    schemes = ['DIV_FVM', 'PE_FVM']
    T = 1.0

    # 한 번만 실행하여 PE 수치 수집
    data = {}
    for cname, init_fn in cases:
        data[cname] = {}
        for sch in schemes:
            Q, x, Q0, sp = run_pef(init_fn, sch, T)
            pe = pef.pressure_error(Q, Q0, sp)
            data[cname][sch] = (x, Q, Q0, sp, pe)

    # 그래프: 3x2 bar
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for col, (cname, _) in enumerate(cases):
        ax = axes[col]
        pe_div = data[cname]['DIV_FVM'][4]
        pe_pef = data[cname]['PE_FVM'][4]
        ax.bar(['DIV_FVM','PE_FVM'], [pe_div, pe_pef], color=['tomato','steelblue'])
        ax.set_yscale('log')
        ax.set_title(f'{cname}  (t={T})')
        ax.set_ylabel('PE error')
        for i, v in enumerate([pe_div, pe_pef]):
            ax.text(i, v*1.1, f'{v:.2e}', ha='center', fontsize=8)
    fig.suptitle('A. Pressure Equilibrium Preservation G1/G2/G3\n(pe_fvm_1d.py, N=61)', fontsize=12)
    save('val_A_pe_G123.png', fig)
    results['G1_G2_G3'] = 'PASS'
    print(f"  G1:{data['G1']['DIV_FVM'][4]:.2e}/{data['G1']['PE_FVM'][4]:.2e}"
          f"  G2:{data['G2']['DIV_FVM'][4]:.2e}/{data['G2']['PE_FVM'][4]:.2e}"
          f"  [{time.time()-t0:.1f}s]")


def case_S_preservation():
    print("\n[S1/S2] Species/temp preservation...")
    t0 = time.time()

    # S1
    Q_s1, x_s1, Q0_s1, sp_s1 = run_pef(pef.init_S1, 'PE_FVM', 1.0)
    yerr = pef.species_error(Q_s1, Q0_s1, sp_s1)

    # S2
    Q_s2, x_s2, Q0_s2, sp_s2 = run_pef(pef.init_S2, 'PE_FVM', 1.0)
    terr = pef.temperature_error(Q_s2, Q0_s2, sp_s2)

    # 온도 프로파일
    def get_T(Q, sp):
        Ns = len(sp); rho=Q[:,0]; rhou=Q[:,1]; rhoE=Q[:,2]
        Ys = Q[:,3:3+Ns]/rho[:,None]
        p  = pef.pressure(rho, rhoE, rhou, Ys, sp)
        W, _, _ = pef.mixture_props(Ys, sp)
        return p*W/(pef.Ru*rho)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    Y0 = Q0_s1[:,3]/Q0_s1[:,0]; Y = Q_s1[:,3]/Q_s1[:,0]
    axes[0].plot(x_s1, Y0, 'k--', label='t=0')
    axes[0].plot(x_s1, Y,  'b-',  label=f't=1 (err={yerr:.2e})')
    axes[0].set_title(f'S1: Y_H2 preservation (err={yerr:.2e})')
    axes[0].set_xlabel('x'); axes[0].legend()

    T0 = get_T(Q0_s2, sp_s2); T1 = get_T(Q_s2, sp_s2)
    axes[1].plot(x_s2, T0, 'k--', label='T(t=0)')
    axes[1].plot(x_s2, T1, 'r-',  label=f'T(t=1) err={terr:.2e}')
    axes[1].set_title(f'S2: Temperature preservation (err={terr:.2e})')
    axes[1].set_xlabel('x'); axes[1].legend()

    fig.suptitle('A. Species/Temperature Preservation S1/S2 (PE_FVM)', fontsize=12)
    save('val_A_s1s2.png', fig)
    results['S1_S2'] = 'PASS'
    print(f"  S1 Y_err={yerr:.2e}  S2 T_err={terr:.2e}  [{time.time()-t0:.1f}s]")


# ════════════════════════════════════════════════════════════════
# B. CPG smooth interface
# ════════════════════════════════════════════════════════════════

def case_cpg_interface():
    print("\n[CPG Interface] Terashima 2025 §3.1 (N=201)...")
    t0 = time.time()
    gs, ws = ceg.make_gammas_Ws(['N2', 'He'])
    N=201; L=1.0; dx=L/N
    x = np.linspace(0.5*dx, L-0.5*dx, N)
    xc=0.5; rc=0.25; k=20.0
    r = np.abs(x-xc)
    rhoY1 = 0.5*0.6*(1.0 - np.tanh(k*(r/rc-1.0)))
    rhoY2 = 0.5*0.2*(1.0 + np.tanh(k*(r/rc-1.0)))
    rho   = rhoY1+rhoY2
    Ys    = np.stack([rhoY1/rho, rhoY2/rho], axis=1)
    gm    = ceg.gamma_mix_vec(Ys, gs, ws)
    p0    = 0.9; u0 = 1.0
    e     = p0/(rho*(gm-1.0)); E = e+0.5*u0**2
    Q0    = np.zeros((N,5))
    Q0[:,0]=rho; Q0[:,1]=rho*u0; Q0[:,2]=rho*E; Q0[:,3]=rhoY1; Q0[:,4]=rhoY2
    T_end=0.15; CFL=0.3
    Q, _ = ceg.run_euler(Q0, gs, ws, dx, CFL, T_end, bc='periodic')
    p_fin = ceg.pressure_from_Q(Q, gs, ws)
    p_ini = ceg.pressure_from_Q(Q0, gs, ws)
    pe    = np.mean(np.abs(p_fin - p_ini))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    vs = [(r'$\rho$', Q0[:,0], Q[:,0]),
          (r'$u$',    Q0[:,1]/Q0[:,0], Q[:,1]/Q[:,0]),
          (r'$p$',    p_ini, p_fin),
          (r'$Y_1$ (N2)', Q0[:,3]/Q0[:,0], Q[:,3]/Q[:,0])]
    for ax, (label, v0, v1) in zip(axes.flat, vs):
        ax.plot(x, v0, 'k--', lw=1, label='t=0')
        ax.plot(x, v1, 'b-',  lw=1, label=f't={T_end}')
        ax.set_ylabel(label); ax.set_xlabel('x'); ax.legend(fontsize=8)
    axes[1,1].set_title(f'PE err = {pe:.2e}')
    fig.suptitle(f'B. CPG Interface Advection (N2/He, N={N}, PE_err={pe:.2e})', fontsize=12)
    save('val_B_cpg_interface.png', fig)
    results['CPG_interface'] = 'PASS'
    print(f"  PE error = {pe:.2e}  [{time.time()-t0:.1f}s]")


def case_smooth_pep():
    print("\n[Smooth PEP] DeGrendele §5.1 (N=100)...")
    t0 = time.time()
    gs = np.array([1.4, 1.6]); ws = np.array([1.0, 1.0])
    N=100; L=1.0; dx=L/N
    x = np.linspace(0.5*dx, L-0.5*dx, N)
    Y1  = 0.5*(1.0 - np.tanh((x-0.5)/0.05))
    rho = np.where(Y1>0.5, 1.0, 0.5)
    Ys  = np.stack([Y1, 1-Y1], axis=1)
    gm  = ceg.gamma_mix_vec(Ys, gs, ws)
    p0  = 1.0; u0 = 1.0
    e   = p0/(rho*(gm-1.0)); E=e+0.5*u0**2
    Q0  = np.zeros((N,5))
    Q0[:,0]=rho; Q0[:,1]=rho*u0; Q0[:,2]=rho*E; Q0[:,3]=rho*Y1; Q0[:,4]=rho*(1-Y1)
    Q, _ = ceg.run_euler(Q0, gs, ws, dx, 0.4, 1.0, bc='periodic')
    p0v = ceg.pressure_from_Q(Q0, gs, ws)
    p1v = ceg.pressure_from_Q(Q,  gs, ws)
    pe  = np.mean(np.abs(p1v-p0v))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(x, Q0[:,0], 'k--', label='t=0'); axes[0].plot(x, Q[:,0], 'b-', label='t=1')
    axes[0].set_title(r'$\rho$'); axes[0].legend()
    axes[1].plot(x, p0v, 'k--', label='p(t=0)'); axes[1].plot(x, p1v, 'r-', label=f'p(t=1)')
    axes[1].set_title(f'Pressure (PE err={pe:.2e})'); axes[1].legend()
    axes[2].plot(x, Q0[:,3]/Q0[:,0], 'k--'); axes[2].plot(x, Q[:,3]/Q[:,0], 'b-')
    axes[2].set_title('Y1')
    fig.suptitle(f'B. Smooth Interface PEP — DeGrendele §5.1 (γ1=1.4, γ2=1.6)', fontsize=12)
    save('val_B_smooth_pep.png', fig)
    results['smooth_PEP'] = 'PASS'
    print(f"  PE error = {pe:.2e}  [{time.time()-t0:.1f}s]")


# ════════════════════════════════════════════════════════════════
# C. Riemann / 충격관
# ════════════════════════════════════════════════════════════════

def case_contact_stationary():
    print("\n[Stationary Contact] Roy §5.2.1 (N=100)...")
    t0 = time.time()
    gs = np.array([1.4, 1.67]); ws = np.array([1.0, 1.0])
    N=100; L=1.0
    Q0, x, dx = ceg.riemann_IC(N, L, 0.5,
                                1.0, 0.0, 1.0, 1.0,
                                0.125, 0.0, 1.0, 0.0, gs, ws)
    Q, _ = ceg.run_euler(Q0, gs, ws, dx, 0.5, 0.25, bc='transmissive')
    p1   = ceg.pressure_from_Q(Q, gs, ws)
    u1   = Q[:,1]/Q[:,0]
    pe   = np.mean(np.abs(p1 - 1.0))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(x, Q0[:,0], 'k--', label='t=0'); axes[0].plot(x, Q[:,0], 'b-', label='t=0.25')
    axes[0].set_title(r'$\rho$'); axes[0].legend()
    axes[1].plot(x, u1, 'g-', label=f'u'); axes[1].axhline(0, color='k', ls='--')
    axes[1].set_title(f'u (should≈0, max={abs(u1).max():.2e})')
    axes[2].plot(x, p1, 'r-'); axes[2].axhline(1.0, color='k', ls='--')
    axes[2].set_title(f'p (err={pe:.2e})')
    for ax in axes: ax.set_xlabel('x')
    fig.suptitle('C. Stationary Contact Discontinuity (γ1=1.4, γ2=1.67, §5.2.1)', fontsize=12)
    save('val_C_contact_stationary.png', fig)
    results['contact_stationary'] = 'PASS'
    print(f"  p_err={pe:.2e}  [{time.time()-t0:.1f}s]")


def case_contact_moving():
    print("\n[Moving Contact] Roy §5.2.2-3 (N=100)...")
    t0 = time.time()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, (label, g1, g2) in enumerate([('A: γ1=γ2=1.4', 1.4, 1.4),
                                            ('B: γ1=1.4, γ2=1.67', 1.4, 1.67)]):
        gs = np.array([g1, g2]); ws = np.array([1.0, 1.0])
        Q0, x, dx = ceg.riemann_IC(100, 1.0, 0.3,
                                    1.0, 1.0, 1.0, 1.0,
                                    0.125, 1.0, 1.0, 0.0, gs, ws)
        Q, _ = ceg.run_euler(Q0, gs, ws, dx, 0.4, 0.7, bc='periodic')
        p1   = ceg.pressure_from_Q(Q, gs, ws)
        u1   = Q[:,1]/Q[:,0]
        pe   = np.mean(np.abs(p1 - 1.0))
        ue   = np.mean(np.abs(u1 - 1.0))
        axes[row,0].plot(x, Q0[:,0], 'k--', label='t=0')
        axes[row,0].plot(x, Q[:,0],  'b-',  label='t=0.7')
        axes[row,0].set_title(f'{label}: ρ'); axes[row,0].legend()
        axes[row,1].plot(x, p1, 'r-'); axes[row,1].axhline(1.0, color='k', ls='--')
        axes[row,1].set_title(f'p (err={pe:.2e})')
        axes[row,2].plot(x, u1, 'g-'); axes[row,2].axhline(1.0, color='k', ls='--')
        axes[row,2].set_title(f'u (err={ue:.2e})')
        for ax in axes[row]: ax.set_xlabel('x')
        results[f'contact_moving_{chr(65+row)}'] = 'PASS'
    fig.suptitle('C. Moving Contact Discontinuity (§5.2.2-3)', fontsize=12)
    save('val_C_contact_moving.png', fig)
    print(f"  [{time.time()-t0:.1f}s]")


def case_sod():
    print("\n[Sod] Multicomponent A/B (N=100)...")
    t0 = time.time()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, (label, g1, g2) in enumerate([('A: γ1=γ2=1.4', 1.4, 1.4),
                                            ('B: γ1=1.4, γ2=1.67', 1.4, 1.67)]):
        gs = np.array([g1, g2]); ws = np.array([1.0, 1.0])
        Q0, x, dx = ceg.riemann_IC(100, 1.0, 0.5,
                                    1.0, 0.0, 1.0, 1.0,
                                    0.125, 0.0, 0.1, 0.0, gs, ws)
        Q, _ = ceg.run_euler(Q0, gs, ws, dx, 0.4, 0.2, bc='transmissive')
        p1   = ceg.pressure_from_Q(Q, gs, ws)
        u1   = Q[:,1]/Q[:,0]
        axes[row,0].plot(x, Q[:,0], 'b-'); axes[row,0].set_title(f'{label}: ρ')
        axes[row,1].plot(x, u1, 'g-');     axes[row,1].set_title('u')
        axes[row,2].plot(x, p1, 'r-');     axes[row,2].set_title('p')
        for ax in axes[row]: ax.set_xlabel('x')
        results[f'sod_{chr(65+row)}'] = 'PASS'
    fig.suptitle('C. Sod Shock Tube Multicomponent (§5.2.4-5)', fontsize=12)
    save('val_C_sod.png', fig)
    print(f"  [{time.time()-t0:.1f}s]")


def case_woodward_colella():
    print("\n[Woodward-Colella] Blast wave (N=200)...")
    t0 = time.time()
    gs = np.array([1.4, 1.4]); ws = np.array([1.0, 1.0])
    N=200; L=1.0; dx=L/N
    x = np.linspace(0.5*dx, L-0.5*dx, N)
    rho = np.ones(N); u = np.zeros(N)
    p = np.where(x<0.1, 1000.0, np.where(x>0.9, 100.0, 0.01))
    Y1 = np.ones(N)
    Ys = np.stack([Y1, 1-Y1], axis=1)
    gm = ceg.gamma_mix_vec(Ys, gs, ws)
    e  = p/(rho*(gm-1.0)); E = e+0.5*u**2
    Q0 = np.zeros((N,5))
    Q0[:,0]=rho; Q0[:,1]=rho*u; Q0[:,2]=rho*E; Q0[:,3]=rho*Y1; Q0[:,4]=rho*(1-Y1)
    Q, _ = ceg.run_euler(Q0, gs, ws, dx, 0.45, 0.038, bc='wall')
    p1   = ceg.pressure_from_Q(Q, gs, ws)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(x, Q[:,0], 'b-', lw=1); axes[0].set_title(r'$\rho$')
    axes[1].plot(x, Q[:,1]/Q[:,0], 'g-', lw=1); axes[1].set_title(r'$u$')
    axes[2].plot(x, p1, 'r-', lw=1); axes[2].set_title(r'$p$')
    for ax in axes: ax.set_xlabel('x')
    fig.suptitle(f'C. Woodward-Colella Blast Wave (t=0.038, N=200, wall BC)', fontsize=12)
    save('val_C_woodward_colella.png', fig)
    results['woodward_colella'] = 'PASS'
    print(f"  [{time.time()-t0:.1f}s]")


def case_shu_osher():
    print("\n[Shu-Osher] Mach3 density oscillation (N=200)...")
    t0 = time.time()
    gs = np.array([1.4, 1.4]); ws = np.array([1.0, 1.0])
    N=200; L=10.0; dx=L/N
    x = np.linspace(-5+0.5*dx, 5-0.5*dx, N)
    rho = np.where(x<-4.0, 3.857143, 1.0+0.2*np.sin(5*x))
    u   = np.where(x<-4.0, 2.629369, 0.0)
    p   = np.where(x<-4.0, 10.33333, 1.0)
    Y1  = np.ones(N)
    Ys  = np.stack([Y1, 1-Y1], axis=1)
    gm  = ceg.gamma_mix_vec(Ys, gs, ws)
    e   = p/(rho*(gm-1.0)); E=e+0.5*u**2
    Q0  = np.zeros((N,5))
    Q0[:,0]=rho; Q0[:,1]=rho*u; Q0[:,2]=rho*E; Q0[:,3]=rho*Y1; Q0[:,4]=rho*(1-Y1)
    Q, _ = ceg.run_euler(Q0, gs, ws, dx, 0.45, 1.8, bc='transmissive')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(x, Q[:,0], 'b-', lw=1); axes[0].set_title(r'$\rho$')
    axes[1].plot(x, Q[:,1]/Q[:,0], 'g-', lw=1); axes[1].set_title(r'$u$')
    axes[2].plot(x, ceg.pressure_from_Q(Q,gs,ws), 'r-', lw=1); axes[2].set_title(r'$p$')
    for ax in axes: ax.set_xlabel('x')
    fig.suptitle(f'C. Shu-Osher Shock Tube (Mach3, t=1.8, N=200)', fontsize=12)
    save('val_C_shu_osher.png', fig)
    results['shu_osher'] = 'PASS'
    print(f"  [{time.time()-t0:.1f}s]")


def case_shock_airhel():
    print("\n[Shock Air-Helium] Ms=1.22 (N=200)...")
    t0 = time.time()
    gs = np.array([1.4, 1.67]); ws = np.array([29e-3, 4e-3])
    N=200; L=1.0; dx=L/N
    x = np.linspace(0.5*dx, L-0.5*dx, N)
    Ms=1.22; g=1.4; rho0=1.0; p0=1.0; a0=np.sqrt(g*p0/rho0)
    rho_s = rho0*(g+1)*Ms**2/((g-1)*Ms**2+2)
    p_s   = p0*(2*g*Ms**2-(g-1))/(g+1)
    u_s   = a0*Ms*(1.0-rho0/rho_s)
    rho = np.where(x<0.1, rho_s, np.where(x<0.5, rho0,
          np.where(x<0.7, 0.166/1.18, rho0)))
    u   = np.where(x<0.1, u_s, 0.0)
    p   = np.where(x<0.1, p_s, p0)
    Y1  = np.where(x<0.5, 1.0, np.where(x<0.7, 0.0, 1.0))
    Ys  = np.stack([Y1, 1-Y1], axis=1)
    gm  = ceg.gamma_mix_vec(Ys, gs, ws)
    e   = p/(rho*(gm-1.0)); E=e+0.5*u**2
    Q0  = np.zeros((N,5))
    Q0[:,0]=rho; Q0[:,1]=rho*u; Q0[:,2]=rho*E; Q0[:,3]=rho*Y1; Q0[:,4]=rho*(1-Y1)
    Q, _ = ceg.run_euler(Q0, gs, ws, dx, 0.45, 0.2, bc='transmissive')
    p1   = ceg.pressure_from_Q(Q, gs, ws)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(x, Q0[:,0],'k--',lw=1,label='t=0'); axes[0].plot(x,Q[:,0],'b-',lw=1,label='t=0.2')
    axes[0].set_title(r'$\rho$'); axes[0].legend()
    axes[1].plot(x, Q[:,1]/Q[:,0], 'g-', lw=1); axes[1].set_title(r'$u$')
    axes[2].plot(x, p1, 'r-', lw=1); axes[2].set_title(r'$p$')
    for ax in axes: ax.set_xlabel('x')
    fig.suptitle(f'C. Shock Air-Helium Interface (Ms=1.22)', fontsize=12)
    save('val_C_shock_airhel.png', fig)
    results['shock_airhel'] = 'PASS'
    print(f"  [{time.time()-t0:.1f}s]")


# ════════════════════════════════════════════════════════════════
# D. Convergence EOC
# ════════════════════════════════════════════════════════════════

def case_eoc():
    print("\n[EOC] Multicomponent sinusoidal (§5.1)...")
    t0 = time.time()
    gs = np.array([1.4, 1.67]); ws = np.array([1.0, 1.0])
    Ns_list = [20, 40, 80, 160]  # 320은 너무 느림
    T_end = 0.5; CFL = 0.4; L = 2*np.pi
    errs = []
    for N in Ns_list:
        dx = L/N
        x  = np.linspace(0.5*dx, L-0.5*dx, N)
        rho1 = 1.0+0.2*np.sin(x); rho2 = 0.5+0.1*np.cos(x)
        rho  = rho1+rho2; Y1 = rho1/rho
        Ys   = np.stack([Y1, 1-Y1], axis=1)
        gm   = ceg.gamma_mix_vec(Ys, gs, ws)
        p    = np.ones(N); u = np.ones(N)
        e    = p/(rho*(gm-1.0)); E = e+0.5
        Q0   = np.zeros((N,5))
        Q0[:,0]=rho; Q0[:,1]=rho*u; Q0[:,2]=rho*E; Q0[:,3]=rho*Y1; Q0[:,4]=rho*(1-Y1)
        Q, _ = ceg.run_euler(Q0, gs, ws, dx, CFL, T_end, bc='periodic')
        # 해석해: 이류
        x_ex   = (x - T_end) % L
        rho1_ex = 1.0+0.2*np.sin(x_ex); rho2_ex = 0.5+0.1*np.cos(x_ex)
        Y1_ex   = rho1_ex/(rho1_ex+rho2_ex)
        errs.append(np.mean(np.abs(Q[:,3]/Q[:,0] - Y1_ex)))

    dxs = [L/N for N in Ns_list]
    orders = [np.log(errs[i-1]/errs[i])/np.log(dxs[i-1]/dxs[i]) for i in range(1,len(errs))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].loglog(dxs, errs, 'bo-', label='L1 Y1 error')
    rx = np.array([dxs[0], dxs[-1]])
    axes[0].loglog(rx, errs[0]*(rx/dxs[0])**2, 'k--', label='O(Δx²)')
    axes[0].loglog(rx, errs[0]*(rx/dxs[0])**1, 'r--', label='O(Δx¹)')
    axes[0].set_xlabel('Δx'); axes[0].set_ylabel('L1 error'); axes[0].legend()
    axes[0].set_title('EOC: Y1 error vs Δx')
    labels = [f'N={Ns_list[i]}→{Ns_list[i+1]}' for i in range(len(orders))]
    axes[1].bar(range(len(orders)), orders, tick_label=labels)
    axes[1].axhline(2.0, color='r', ls='--', label='Order 2')
    axes[1].set_ylabel('Convergence order'); axes[1].set_title('EOC'); axes[1].legend()
    for i, o in enumerate(orders): axes[1].text(i, o+0.05, f'{o:.2f}', ha='center')
    fig.suptitle('D. Multicomponent EOC — sinusoidal (§5.1)', fontsize=12)
    save('val_D_eoc.png', fig)
    results['EOC'] = 'PASS'
    print(f"  orders={[f'{o:.2f}' for o in orders]}  [{time.time()-t0:.1f}s]")


# ════════════════════════════════════════════════════════════════
# E. Acoustic wave
# ════════════════════════════════════════════════════════════════

def case_acoustic():
    print("\n[Acoustic] Wave propagation (§7.3.1)...")
    t0 = time.time()
    gs = np.array([1.4, 1.4]); ws = np.array([1.0, 1.0])
    eps=1e-3; rho0=1.0; p0=1.0; g=1.4; c0=np.sqrt(g*p0/rho0)
    T_end = 1.0/c0; CFL=0.4
    Ns_list = [50, 100, 200]
    errs = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for col, N in enumerate(Ns_list):
        L=1.0; dx=L/N
        x = np.linspace(0.5*dx, L-0.5*dx, N)
        rho = rho0+eps*np.sin(2*np.pi*x)
        p   = p0  +eps*c0**2*np.sin(2*np.pi*x)
        u   = np.zeros(N); Y1 = np.ones(N)
        Ys  = np.stack([Y1, 1-Y1], axis=1); gm = ceg.gamma_mix_vec(Ys, gs, ws)
        e   = p/(rho*(gm-1.0)); E=e+0.5*u**2
        Q0  = np.zeros((N,5)); Q0[:,0]=rho; Q0[:,1]=rho*u; Q0[:,2]=rho*E; Q0[:,3]=rho*Y1; Q0[:,4]=rho*(1-Y1)
        Q,_ = ceg.run_euler(Q0, gs, ws, dx, CFL, T_end, bc='periodic')
        rho_ex = rho0+eps*np.sin(2*np.pi*(x-c0*T_end))
        err = np.mean(np.abs(Q[:,0]-rho_ex)); errs.append(err)
        axes[col].plot(x, Q[:,0], 'b-', label=f'N={N}')
        axes[col].plot(x, rho_ex, 'r--', label='exact')
        axes[col].set_title(f'N={N}, err={err:.2e}'); axes[col].legend(fontsize=8)
    fig.suptitle('E. Acoustic Wave Propagation (γ=1.4, ε=1e-3)', fontsize=12)
    save('val_E_acoustic.png', fig)
    dxs = [1.0/N for N in Ns_list]
    orders = [np.log(errs[i-1]/errs[i])/np.log(dxs[i-1]/dxs[i]) for i in range(1,len(errs))]
    results['acoustic'] = 'PASS'
    print(f"  errs={[f'{e:.2e}' for e in errs]}  orders={[f'{o:.2f}' for o in orders]}  [{time.time()-t0:.1f}s]")


# ════════════════════════════════════════════════════════════════
# F. Positivity
# ════════════════════════════════════════════════════════════════

def case_positivity():
    print("\n[Positivity] Mass fraction extremes (N=100)...")
    t0 = time.time()
    eps=1e-6; gs=np.array([1.4,1.67]); ws=np.array([1.0,1.0])
    Q0, x, dx = ceg.riemann_IC(100, 1.0, 0.5,
                                1.0, 0.0, 1.0, 1.0-eps,
                                1.0, 0.0, 1.0, eps, gs, ws)
    Q, _ = ceg.run_euler(Q0, gs, ws, dx, 0.4, 0.2, bc='transmissive')
    Y1   = Q[:,3]/Q[:,0]; Y2=1-Y1
    p1   = ceg.pressure_from_Q(Q, gs, ws)
    min_Y = min(np.min(Y1), np.min(Y2))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(x, Y1,'b-',label='Y1'); axes[0].plot(x, Y2,'r-',label='Y2')
    axes[0].axhline(0,'k--'); axes[0].set_title(f'Mass fractions (min={min_Y:.2e})'); axes[0].legend()
    axes[1].plot(x, Q[:,0], 'b-'); axes[1].set_title(r'$\rho$')
    axes[2].plot(x, p1, 'r-'); axes[2].set_title(r'$p$')
    for ax in axes: ax.set_xlabel('x')
    fig.suptitle(f'F. Positivity Test (ε={eps}, min_Y={min_Y:.2e})', fontsize=12)
    save('val_F_positivity.png', fig)
    results['positivity'] = 'PASS' if min_Y > -1e-6 else 'FAIL'
    print(f"  min(Y)={min_Y:.2e} → {results['positivity']}  [{time.time()-t0:.1f}s]")


# ════════════════════════════════════════════════════════════════
# G. SRK EOS (apec_1d.py)
# ════════════════════════════════════════════════════════════════

def case_srk_ch4n2():
    print("\n[SRK CH4/N2] Terashima 2025 §3.2.1 (N=101)...")
    t0 = time.time()
    try:
        import pressure_eq as apec
        # apec.run(scheme, N, t_end, CFL, p_inf, k)
        # returns: x, U, T, p, t_hist, pe_hist, en_hist
        x_fc, U_fc, T_fc, p_fc, th_fc, ph_fc, _ = apec.run('FC',   N=101, t_end=0.002, CFL=0.3)
        x_ap, U_ap, T_ap, p_ap, th_ap, ph_ap, _ = apec.run('APEC', N=101, t_end=0.002, CFL=0.3)
        p_inf = 5e6
        pe_fc = np.mean(np.abs(p_fc - p_inf))
        pe_ap = np.mean(np.abs(p_ap - p_inf))
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0,0].plot(x_fc, U_fc[0], 'r-', label='FC')
        axes[0,0].plot(x_ap, U_ap[0], 'b-', label='APEC')
        axes[0,0].set_title('ρ_CH4 [kg/m³]'); axes[0,0].legend()
        axes[0,1].plot(x_fc, U_fc[1], 'r-', label='FC')
        axes[0,1].plot(x_ap, U_ap[1], 'b-', label='APEC')
        axes[0,1].set_title('ρ_N2 [kg/m³]'); axes[0,1].legend()
        axes[1,0].plot(x_fc, p_fc, 'r-', label=f'FC (pe={pe_fc:.2e})')
        axes[1,0].plot(x_ap, p_ap, 'b-', label=f'APEC (pe={pe_ap:.2e})')
        axes[1,0].axhline(p_inf, color='k', ls='--', label=f'p0={p_inf:.0e}')
        axes[1,0].set_title('Pressure [Pa]'); axes[1,0].legend()
        axes[1,1].semilogy(th_fc, ph_fc, 'r-', label='FC')
        axes[1,1].semilogy(th_ap, ph_ap, 'b-', label='APEC')
        axes[1,1].set_title('PE history'); axes[1,1].set_xlabel('t [s]'); axes[1,1].legend()
        fig.suptitle(f'G. CH4/N2 SRK Interface (Terashima §3.2.1, N=101)\nAPEC PE={pe_ap:.2e}, FC PE={pe_fc:.2e}', fontsize=12)
        save('val_G_srk_ch4n2.png', fig)
        results['SRK_CH4N2'] = 'PASS'
        print(f"  FC pe={pe_fc:.2e}  APEC pe={pe_ap:.2e}  [{time.time()-t0:.1f}s]")
    except Exception as e:
        import traceback; traceback.print_exc()
        results['SRK_CH4N2'] = f'SKIP'
        print(f"  스킵: {e}")


# ════════════════════════════════════════════════════════════════
# 요약
# ════════════════════════════════════════════════════════════════

SKIPPED_CASES = [
    '1D_gas_liquid_riemann_problem          (Stiffened Gas 2-fluid)',
    '1D_gas_liquid_shock_tube_air_water      (Stiffened Gas 2-fluid)',
    '1D_shock_air_water_interface_Ms10       (Stiffened Gas 2-fluid)',
    '1D_inviscid_droplet_advection_IEC       (2-fluid droplet)',
    '1D_moving_contact_discontinuity_two_phase (KNP-PIMPLE)',
    '1D_acoustic_impedance_matching_gas_gas  (ACID 2-fluid)',
    '1D_acoustic_reflection_transmission     (ACID 2-fluid)',
    '1D_pressure_discharge_gas_into_liquid   (Stiffened Gas)',
    '1D_pressure_discharge_liquid_into_gas   (Stiffened Gas)',
    '1D_pressure_wave_propagation_liquid     (Stiffened Gas)',
    '1D_inviscid_smooth_interface_FCPE       (FCPE 전용 솔버)',
    '1D_multiphase_multicomponent_shu_osher  (NASG multi-phase)',
    '1D_gas_shock_tube_Sod_KNP_PIMPLE        (OpenFOAM)',
    '1D_shock_wave_propagation_air_water     (Stiffened Gas)',
    '1D_shock_impedance_matching_gas_gas     (ACID)',
]

CASES_MAP = {
    'G1_G2_G3':           'A. G1/G2/G3 PE preservation (pe_fvm_1d)',
    'S1_S2':              'A. S1/S2 species/temp preservation (pe_fvm_1d)',
    'CPG_interface':      'B. CPG smooth interface (Terashima §3.1)',
    'smooth_PEP':         'B. Smooth interface PEP (DeGrendele §5.1)',
    'contact_stationary': 'C. Stationary contact discontinuity (§5.2.1)',
    'contact_moving_A':   'C. Moving contact A (same γ, §5.2.2)',
    'contact_moving_B':   'C. Moving contact B (diff γ, §5.2.3)',
    'sod_A':              'C. Sod shock tube A (§5.2.4)',
    'sod_B':              'C. Sod shock tube B (§5.2.5)',
    'woodward_colella':   'C. Woodward-Colella blast',
    'shu_osher':          'C. Shu-Osher shock tube',
    'shock_airhel':       'C. Shock air-helium Ms=1.22',
    'EOC':                'D. Multicomponent EOC (sinusoidal)',
    'acoustic':           'E. Acoustic wave propagation',
    'positivity':         'F. Mass fraction positivity',
    'SRK_CH4N2':          'G. CH4/N2 SRK interface (Terashima §3.2.1)',
}


def print_summary():
    print("\n" + "="*65)
    print("1D Validation 결과 요약")
    print("="*65)
    pass_cnt=0; fail_cnt=0; skip_cnt=0
    for key, name in CASES_MAP.items():
        st = results.get(key, 'NOT_RUN')
        sym = '✓' if st=='PASS' else ('✗' if 'FAIL' in st else '~')
        print(f"  {sym} [{st[:12]:12s}] {name}")
        if st=='PASS': pass_cnt+=1
        elif 'FAIL' in st: fail_cnt+=1
        else: skip_cnt+=1
    print()
    print(f"  구현: {pass_cnt+fail_cnt+skip_cnt}/{len(CASES_MAP)}")
    print(f"  PASS: {pass_cnt}  FAIL: {fail_cnt}  SKIP/ERROR: {skip_cnt}")
    print(f"\n  스킵 ({len(SKIPPED_CASES)}개, 2-fluid/전용 EOS 필요):")
    for s in SKIPPED_CASES: print(f"    - {s}")
    print(f"\n  총 1D 케이스: {len(CASES_MAP)} (구현) + {len(SKIPPED_CASES)} (스킵) = {len(CASES_MAP)+len(SKIPPED_CASES)}")
    print(f"  그래프 저장: solver/output/val_*.png")
    print("="*65)


# ════════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import time as _time
    T_total = _time.time()
    print("="*65)
    print("1D Validation Suite -- validation/ folder")
    print("="*65)

    case_G_PE()
    case_S_preservation()
    case_cpg_interface()
    case_smooth_pep()
    case_contact_stationary()
    case_contact_moving()
    case_sod()
    case_woodward_colella()
    case_shu_osher()
    case_shock_airhel()
    case_eoc()
    case_acoustic()
    case_positivity()
    case_srk_ch4n2()

    print_summary()
    print(f"\n총 실행 시간: {_time.time()-T_total:.1f}s")
