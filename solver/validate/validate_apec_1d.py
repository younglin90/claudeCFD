"""
validate_apec_1d.py
===================
solver/apec.py 기반 1D validation — 2개 케이스 (APEC 적용 가능한 전체)

[V-A1] 1D_CH4_N2_interface_advection_SRK_EOS.md
        SRK 실기체 CH4/N2 계면 이류 (Terashima §3.2.1)
        FC-NPE vs APEC vs PEqC
        출력: PE@1step × N, PE vs time, 공간 프로파일 (ρ,p,u,Y_CH4)

[V-A2] 1D_calorically_perfect_gas_interface_advection.md
        CPG 이상기체 계면 이류 (Terashima §3.1)
        FC vs APEC vs PEqC
        출력: PE vs time, 공간 프로파일 (ρ1,ρ2,u,p), 격자 수렴

결과: output/1D/apec_*.png
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from solver.apec import (
    # SRK
    initial_condition, srk_c2, pe_err, energy_err,
    run as srk_run,
    # CPG
    run_cpg, pe_err_cpg,
)

OUTPUT = os.path.join(ROOT, 'output', '1D')
os.makedirs(OUTPUT, exist_ok=True)

COLORS  = {'FC': 'tab:blue', 'APEC': 'tab:orange', 'PEqC': 'tab:green'}
SCHEMES = ['FC', 'APEC', 'PEqC']

PASS_ALL = True
summary  = []

def record(name, ok, msg):
    global PASS_ALL
    if not ok: PASS_ALL = False
    tag = 'PASS' if ok else 'FAIL'
    summary.append(f'  [{tag}]  {name}: {msg}')

# ═══════════════════════════════════════════════════════════
# [V-A1]  SRK CH4/N2 계면 이류
# ═══════════════════════════════════════════════════════════

def va1_resolution(k=15.0, CFL=0.3, t_end=1e-4):
    """1-step PE vs N=51,101,201,501  (Table 7.1)."""
    print('\n[V-A1-res] SRK PE@1step  resolution study')
    from solver.apec import rkstep, prim

    Ns  = [51, 101, 201, 501]
    hdr = f"  {'N':>5}  {'FC':>12}  {'APEC':>12}  {'PEqC':>12}"
    print(hdr); print('  ' + '-'*50)

    rows = []
    for N in Ns:
        dx = 1.0/N
        x  = np.linspace(dx/2, 1-dx/2, N)
        r1, r2, u0, rhoE0, T0, p0 = initial_condition(x, k=k)
        U0 = [r1.copy(), r2.copy(), (r1+r2)*u0, rhoE0.copy()]
        row = [N]
        for sc in SCHEMES:
            lam = float(np.max(np.abs(u0) + np.sqrt(srk_c2(r1, r2, T0))))
            dt  = CFL * dx / (lam + 1e-12)
            U1, T1, p1 = rkstep(U0, sc, dx, dt, T0)
            row.append(pe_err(p1))
        rows.append(row)
        print(f"  {N:>5}  {row[1]:>12.3e}  {row[2]:>12.3e}  {row[3]:>12.3e}")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    dxs = [1.0/r[0] for r in rows]
    for i, sc in enumerate(SCHEMES):
        pes = [r[i+1] for r in rows]
        ax.loglog(dxs, pes, 'o-', lw=2, label=sc, color=COLORS[sc])
    # 2차 기울기 참조선
    ref = [rows[0][1] * (dxs[j]/dxs[0])**2 for j in range(len(dxs))]
    ax.loglog(dxs, ref, 'k--', lw=1, label='O(Δx²)')
    ax.set_xlabel('Δx'); ax.set_ylabel('PE (1-step)')
    ax.set_title('[V-A1] SRK CH4/N2  —  1-step PE vs Δx  (k=15)')
    ax.legend(); ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    fname = os.path.join(OUTPUT, 'apec_va1_resolution.png')
    plt.savefig(fname, dpi=120); plt.close()
    print(f'  Saved: {fname}')

    # 검증: APEC < FC at N=501
    fc_501 = rows[-1][1]; ap_501 = rows[-1][2]; pq_501 = rows[-1][3]
    ok = ap_501 < fc_501
    ratio = fc_501 / max(ap_501, 1e-30)
    record('V-A1-res', ok,
           f'N=501 FC={fc_501:.2e} APEC={ap_501:.2e} ratio={ratio:.1f}x')
    return rows


def va1_pe_time(N=101, CFL=0.3, k=15.0, t_end=0.06):
    """PE vs time — FC / APEC / PEqC."""
    print(f'\n[V-A1-time] SRK PE vs time  N={N}  k={k}  t={t_end*1e3:.0f}ms')
    res = {}
    for sc in SCHEMES:
        x, U, T, p, th, ph, eh, div = srk_run(
            sc, N=N, t_end=t_end, CFL=CFL, k=k)
        res[sc] = dict(t=th, pe=ph, en=eh, div=div, x=x, U=U, p=p)
        print(f'  {sc:4s}  PE_init={ph[1]:.3e}  PE_final={ph[-1]:.3e}  '
              f'diverged={div}  steps={len(th)-1}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for sc, d in res.items():
        mask = np.isfinite(d['pe'])
        ax1.semilogy(d['t'][mask]*1e3, d['pe'][mask], lw=2,
                     label=sc, color=COLORS[sc])
        ax2.semilogy(d['t'][mask]*1e3, np.abs(d['en'][mask])+1e-16, lw=2,
                     label=sc, color=COLORS[sc])
    ax1.set_xlabel('t [ms]'); ax1.set_ylabel('max |Δp/p₀|')
    ax1.set_title(f'[V-A1] SRK PE error  N={N}  k={k}')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('t [ms]'); ax2.set_ylabel('|ΔE_total/E₀|')
    ax2.set_title('Energy conservation')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(OUTPUT, f'apec_va1_pe_time_N{N}.png')
    plt.savefig(fname, dpi=120); plt.close()
    print(f'  Saved: {fname}')

    fc_pe = res['FC']['pe'][-1]; ap_pe = res['APEC']['pe'][-1]
    ok = (not res['APEC']['div']) and ap_pe < fc_pe
    record('V-A1-time', ok,
           f'FC={fc_pe:.2e} APEC={ap_pe:.2e}')
    return res


def va1_profiles(N=101, CFL=0.3, k=15.0, t_snap=5e-3):
    """공간 프로파일 비교: ρ, p, u, Y_CH4."""
    print(f'\n[V-A1-prof] SRK profiles at t={t_snap*1e3:.1f}ms  N={N}')
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for sc in SCHEMES:
        x, U, T, p, *_ = srk_run(sc, N=N, t_end=t_snap, CFL=CFL, k=k)
        r1, r2, rhoU, rhoE = U
        rho = r1 + r2
        u_  = rhoU / np.maximum(rho, 1e-30)
        Y1  = r1 / np.maximum(rho, 1e-30)
        axes[0].plot(x, rho,    lw=1.5, label=sc, color=COLORS[sc])
        axes[1].plot(x, p*1e-6, lw=1.5, label=sc, color=COLORS[sc])
        axes[2].plot(x, u_,     lw=1.5, label=sc, color=COLORS[sc])
        axes[3].plot(x, Y1,     lw=1.5, label=sc, color=COLORS[sc])

    for ax, ttl in zip(axes, ['ρ [kg/m³]', 'p [MPa]', 'u [m/s]', 'Y_CH4']):
        ax.set_xlabel('x'); ax.set_title(ttl)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(
        f'[V-A1] SRK CH4/N2  t={t_snap*1e3:.1f}ms  N={N}  k={k}', fontsize=11)
    plt.tight_layout()
    fname = os.path.join(OUTPUT, f'apec_va1_profiles_N{N}.png')
    plt.savefig(fname, dpi=120); plt.close()
    print(f'  Saved: {fname}')
    record('V-A1-prof', True, f'saved {os.path.basename(fname)}')


# ═══════════════════════════════════════════════════════════
# [V-A2]  CPG 계면 이류
# ═══════════════════════════════════════════════════════════

def va2_pe_time(N=501, CFL=0.6, k=20.0, t_end=8.0):
    """CPG PE vs time — FC / APEC / PEqC."""
    print(f'\n[V-A2-time] CPG PE vs time  N={N}  k={k}  t={t_end}')
    res = {}
    for sc in SCHEMES:
        x, U, _, p, th, ph, eh, div = srk_run(
            sc, N=N, t_end=t_end, CFL=CFL)
        # srk_run은 SRK 전용이므로 run_cpg 사용
        pass

    # run_cpg 사용 (flux='KEEP' 기본)
    res = {}
    for sc in SCHEMES:
        print(f'  Running {sc} ...', end=' ', flush=True)
        x, U, _, p, th, ph, eh, div = srk_run.__wrapped__(sc, N=N, t_end=t_end, CFL=CFL) \
            if hasattr(srk_run, '__wrapped__') else _run_cpg_full(sc, N, CFL, t_end, k)
        res[sc] = dict(t=th, pe=ph, en=eh, div=div, p=p, x=x, U=U)
        print(f'PE_final={ph[-1]:.3e}  diverged={div}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for sc, d in res.items():
        mask = np.isfinite(d['pe'])
        ax1.semilogy(d['t'][mask], d['pe'][mask], lw=2,
                     label=sc, color=COLORS[sc])
        ax2.semilogy(d['t'][mask], np.abs(d['en'][mask])+1e-16, lw=2,
                     label=sc, color=COLORS[sc])
    ax1.set_xlabel('t'); ax1.set_ylabel('PE_rms')
    ax1.set_title(f'[V-A2] CPG PE error  N={N}  k={k}')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('t'); ax2.set_ylabel('|ΔE/E₀|')
    ax2.set_title('Energy conservation')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(OUTPUT, f'apec_va2_pe_time_N{N}.png')
    plt.savefig(fname, dpi=120); plt.close()
    print(f'  Saved: {fname}')

    fc_pe = res['FC']['pe'][-1]; ap_pe = res['APEC']['pe'][-1]
    ok = ap_pe < fc_pe or res['FC']['div']
    record('V-A2-time', ok, f'FC={fc_pe:.2e} APEC={ap_pe:.2e}')
    return res


def _run_cpg_full(sc, N, CFL, t_end, k):
    """run_cpg 래퍼 — srk_run과 동일 인터페이스로 반환."""
    from solver.apec import run_cpg, pe_err_cpg, cpg_prim
    x, U, _, p_final, th, ph, eh, div = run_cpg(sc, N=N, t_end=t_end,
                                                  CFL=CFL, k=k)
    return x, U, None, p_final, th, ph, eh, div


def va2_pe_time_v2(N=501, CFL=0.6, k=20.0, t_end=8.0):
    """CPG PE vs time (run_cpg 직접 사용)."""
    print(f'\n[V-A2-time] CPG PE vs time  N={N}  k={k}  t={t_end}')
    res = {}
    for sc in SCHEMES:
        print(f'  Running {sc} ...', end=' ', flush=True)
        x, U, _, p, th, ph, eh, div = run_cpg(
            sc, N=N, t_end=t_end, CFL=CFL, k=k)
        res[sc] = dict(t=th, pe=ph, en=eh, div=div, p=p, x=x, U=U)
        fin = ph[np.isfinite(ph)][-1] if np.any(np.isfinite(ph)) else np.inf
        print(f'PE_final={fin:.3e}  diverged={div}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for sc, d in res.items():
        mask = np.isfinite(d['pe'])
        ax1.semilogy(d['t'][mask], d['pe'][mask], lw=2,
                     label=sc, color=COLORS[sc])
        ax2.semilogy(d['t'][mask], np.abs(d['en'][mask])+1e-16, lw=2,
                     label=sc, color=COLORS[sc])
    ax1.set_xlabel('t [s]'); ax1.set_ylabel('PE_rms')
    ax1.set_title(f'[V-A2] CPG  N={N}  k={k}')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('t [s]'); ax2.set_ylabel('|ΔE/E₀|')
    ax2.set_title('Energy conservation')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(OUTPUT, f'apec_va2_pe_time_N{N}.png')
    plt.savefig(fname, dpi=120); plt.close()
    print(f'  Saved: {fname}')

    fc_pe = res['FC']['pe'][np.isfinite(res['FC']['pe'])][-1]
    ap_pe = res['APEC']['pe'][np.isfinite(res['APEC']['pe'])][-1]
    ok = ap_pe < fc_pe or res['FC']['div']
    record('V-A2-time', ok, f'FC={fc_pe:.2e} APEC={ap_pe:.2e}')
    return res


def va2_profiles(N=501, CFL=0.6, k=20.0, t_end=8.0):
    """CPG 공간 프로파일: ρ1, ρ2, u, p."""
    print(f'\n[V-A2-prof] CPG profiles  N={N}  t={t_end}')
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for sc in SCHEMES:
        x, U, _, p, *_ = run_cpg(sc, N=N, t_end=t_end, CFL=CFL, k=k)
        r1, r2, rhoU, rhoE = U
        rho = r1 + r2
        u_  = rhoU / np.maximum(rho, 1e-30)
        axes[0].plot(x, r1,  lw=1.5, label=sc, color=COLORS[sc])
        axes[1].plot(x, r2,  lw=1.5, label=sc, color=COLORS[sc])
        axes[2].plot(x, u_,  lw=1.5, label=sc, color=COLORS[sc])
        axes[3].plot(x, p,   lw=1.5, label=sc, color=COLORS[sc])

    for ax, ttl in zip(axes, ['ρ₁ [kg/m³]', 'ρ₂ [kg/m³]', 'u [-]', 'p [-]']):
        ax.set_xlabel('x'); ax.set_title(ttl)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f'[V-A2] CPG 계면 이류  t={t_end}  N={N}', fontsize=11)
    plt.tight_layout()
    fname = os.path.join(OUTPUT, f'apec_va2_profiles_N{N}.png')
    plt.savefig(fname, dpi=120); plt.close()
    print(f'  Saved: {fname}')
    record('V-A2-prof', True, f'saved {os.path.basename(fname)}')


def va2_convergence(CFL=0.6, k=20.0):
    """격자 수렴 (t=20, N=251 vs N=501)."""
    print(f'\n[V-A2-conv] CPG convergence  k={k}')
    Ns = [251, 501]
    res = {}
    for N in Ns:
        x, U, _, p, th, ph, eh, div = run_cpg(
            'APEC', N=N, t_end=20.0, CFL=CFL, k=k)
        rms = float(np.sqrt(np.mean(ph[np.isfinite(ph)]**2)))
        res[N] = rms
        print(f'  N={N:4d}  PE_rms={rms:.4e}  diverged={div}')

    ok = res[501] < res[251]
    record('V-A2-conv', ok,
           f'N=251 PE_rms={res[251]:.3e}  N=501 PE_rms={res[501]:.3e}')
    return res


# ═══════════════════════════════════════════════════════════
# 요약 출력
# ═══════════════════════════════════════════════════════════

def print_summary():
    print('\n' + '='*65)
    print('  APEC 1D Validation Results  (solver/apec.py)')
    print('  Applicable cases: 2 / 31 validation/*.md')
    print('='*65)
    for r in summary:
        print(r)
    nP = sum(1 for r in summary if '[PASS]' in r)
    nF = sum(1 for r in summary if '[FAIL]' in r)
    print('  ' + '-'*60)
    print(f'  PASS={nP}  FAIL={nF}  Total={nP+nF}')
    print('='*65)
    print('\nN/A 케이스 (apec.py 미지원 EOS/solver):')
    na_cases = [
        '1D_acoustic_*.md              → single-phase ACID',
        '1D_gas_liquid_*.md            → NASG EOS (acid.py)',
        '1D_gas_shock_tube_Sod_*.md    → 단일상 shock tube',
        '1D_inviscid_droplet_*.md      → four_eq IEC',
        '1D_moving_contact_*.md        → Kinetic scheme',
        '1D_multiphase_*.md            → four_eq IEC',
        '1D_pressure_discharge_*.md    → NASG EOS',
        '1D_pressure_equilibrium_*.md  → cpg_flux KEEPPE',
        '1D_pressure_wave_*.md         → NASG EOS',
        '1D_shock_air_*.md             → NASG EOS',
        '1D_shock_wave_*.md            → NASG EOS',
        '1D_shu_osher_*.md             → 단일상 shock tube',
        '1D_sod_shock_tube_*.md        → Kinetic scheme',
        '1D_smooth_interface_PEP.md    → Exact PEP (다른 프레임워크)',
        '1D_species_temperature_*.md   → cpg_flux KEEPPE',
        '1D_steady_contact_*.md        → Kinetic scheme',
        '1D_interface_adv_ACID.md      → ACID scheme',
        '1D_multicomponent_EOC_*.md    → Kinetic scheme',
        '1D_positivity_*.md            → Kinetic scheme',
        '1D_inviscid_smooth_FCPE.md    → FC-PE (다른 프레임워크)',
    ]
    for c in na_cases:
        print(f'  [N/A]  {c}')


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('='*65)
    print('  APEC 1D Validation  (solver/apec.py only)')
    print(f'  Output: {OUTPUT}')
    print('='*65)

    # ── [V-A1] SRK CH4/N2 ─────────────────────────────────
    print('\n' + '─'*65)
    print('[V-A1]  1D_CH4_N2_interface_advection_SRK_EOS.md')
    print('─'*65)
    va1_resolution(k=15.0, CFL=0.3)
    va1_pe_time(N=101, CFL=0.3, k=15.0, t_end=0.06)
    va1_profiles(N=101, CFL=0.3, k=15.0, t_snap=5e-3)

    # ── [V-A2] CPG ideal gas ───────────────────────────────
    print('\n' + '─'*65)
    print('[V-A2]  1D_calorically_perfect_gas_interface_advection.md')
    print('─'*65)
    va2_pe_time_v2(N=501, CFL=0.6, k=20.0, t_end=8.0)
    va2_profiles(N=501, CFL=0.6, k=20.0, t_end=8.0)
    va2_convergence(CFL=0.6, k=20.0)

    print_summary()
    sys.exit(0 if PASS_ALL else 1)
