"""§4.2.4 Mach-100 Water Jet — resolution study + IEC vs STD comparison.

고정 물리적 계면 폭 eps_abs=0.04 사용:
  N=100 → 4 cells, N=200 → 8 cells, N=400 → 16 cells (모두 안정적)

eps_factor*dx 방식은 N=400에서 eps=0.01 (4 cells)로 너무 날카로워
물/공기 density ratio 6001에 의해 WENO 진동이 Amagat 비선형성으로 증폭 → 발산.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from four_eq_1d import ic_mach100_jet, nasg_prim, rhs_iec, rkstep

sp0, sp1 = 'water_nd', 'air_nd'
EPS_ABS   = 0.04   # fixed physical interface width for all N
CFL       = 0.1
T_END     = 1.0e-3

# ── 1. 단일 비교 (N=200, IEC vs STD) ──────────────────────────────────────────
N_MAIN = 200
dx_m = 1.0 / N_MAIN
x_m  = np.linspace(-0.5 + dx_m/2, 0.5 - dx_m/2, N_MAIN)

r0_0, r1_0, m_ic_0, rhoE_0 = ic_mach100_jet(x_m, sp0=sp0, sp1=sp1, eps_abs=EPS_ABS)
rho0 = r0_0 + r1_0
Y0_0 = r0_0 / np.maximum(rho0, 1e-30)
u0, e0, P0, T0, c20 = nasg_prim(Y0_0, rho0, m_ic_0, rhoE_0, sp0, sp1)

print("=" * 55)
print(f"§4.2.4 Mach-100 Water Jet  eps_abs={EPS_ABS}  N={N_MAIN}")
print("=" * 55)
print(f"  Pmin={P0.min():.4e}  Pmax={P0.max():.4e}")
print(f"  u_jet={u0.max():.3f}   rhomax={rho0.max():.1f}")

main_results = {}
for scheme in ['IEC', 'STD']:
    r0_, r1_, m_, rhoE_ = ic_mach100_jet(x_m, sp0=sp0, sp1=sp1, eps_abs=EPS_ABS)
    U = [r0_.copy(), r1_.copy(), m_.copy(), rhoE_.copy()]

    use_iec = (scheme == 'IEC')
    def rhs_fn(U, dx, _iec=use_iec):
        return rhs_iec(U, dx, sp0, sp1, iec=_iec, use_char=False, weno_order=5,
                       bc='periodic')

    t, step = 0.0, 0
    times, pe_hist = [], []
    print(f"\n[{scheme}] Running to t={T_END:.2e} ...")
    while t < T_END - 1e-14:
        r0c, r1c, mc, Ec = U
        rhoc = r0c + r1c
        Y0c  = r0c / np.maximum(rhoc, 1e-30)
        uc, ec, Pc, Tc, c2c = nasg_prim(Y0c, rhoc, mc, Ec, sp0, sp1)
        lam  = float(np.max(np.abs(uc) + np.sqrt(c2c)))
        dt   = min(CFL * dx_m / (lam + 1e-10), T_END - t)
        U, _ = rkstep(U, rhs_fn, dx_m, dt)
        t += dt; step += 1
        if not np.all(np.isfinite(U[3])):
            print(f"  NaN at step {step} t={t:.3e}"); break
        pe = float(np.max(np.abs(Pc - 1.0)))
        times.append(t); pe_hist.append(pe)
        if step % 50 == 0:
            print(f"  step={step} t={t:.3e} PE={pe:.3e}")

    r0f, r1f, mf, Ef = U
    rhof = r0f + r1f
    Y0f  = r0f / np.maximum(rhof, 1e-30)
    uf, ef, Pf, Tf, c2f = nasg_prim(Y0f, rhof, mf, Ef, sp0, sp1)
    pe_final = float(np.max(np.abs(Pf - 1.0)))
    print(f"  [{scheme}] Done t={t:.3e}  PE={pe_final:.3e}  Pmin={Pf.min():.4f}  Pmax={Pf.max():.4f}")
    main_results[scheme] = dict(x=x_m, rho=rhof, u=uf, P=Pf, Y0=Y0f,
                                 times=np.array(times), pe=np.array(pe_hist))

# ── 2. 해상도 연구 ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"Resolution Study  eps_abs={EPS_ABS}  t_end={T_END:.2e}")
print("=" * 55)

N_list = [50, 100, 200, 400]
res_study = {}
for N in N_list:
    dx = 1.0 / N
    x  = np.linspace(-0.5 + dx/2, 0.5 - dx/2, N)
    row = {}
    for scheme in ['IEC', 'STD']:
        r0_, r1_, m_, rhoE_ = ic_mach100_jet(x, sp0=sp0, sp1=sp1, eps_abs=EPS_ABS)
        U = [r0_.copy(), r1_.copy(), m_.copy(), rhoE_.copy()]
        use_iec = (scheme == 'IEC')
        def rhs_fn2(U, dx, _iec=use_iec):
            return rhs_iec(U, dx, sp0, sp1, iec=_iec, use_char=False, weno_order=5,
                           bc='periodic')
        t, ok = 0.0, True
        while t < T_END - 1e-14:
            r0c, r1c, mc, Ec = U
            rhoc = r0c + r1c
            Y0c  = r0c / np.maximum(rhoc, 1e-30)
            uc, ec, Pc, Tc, c2c = nasg_prim(Y0c, rhoc, mc, Ec, sp0, sp1)
            lam  = float(np.max(np.abs(uc) + np.sqrt(c2c)))
            dt   = min(CFL * dx / (lam + 1e-10), T_END - t)
            U, _ = rkstep(U, rhs_fn2, dx, dt)
            t += dt
            if not np.all(np.isfinite(U[3])):
                ok = False; break
        r0f, r1f, mf, Ef = U
        rhof = r0f + r1f
        Y0f  = r0f / np.maximum(rhof, 1e-30)
        uf, ef, Pf, Tf, c2f = nasg_prim(Y0f, rhof, mf, Ef, sp0, sp1)
        pe = float(np.max(np.abs(Pf - 1.0))) if ok else float('inf')
        row[scheme] = pe
        status = f"{pe:.3e}" if ok else "BLOWN UP"
        print(f"  N={N:4d}  {scheme}  PE={status}")
    res_study[N] = row

print("\n  Summary (max|P-1|):")
print(f"  {'N':>6}  {'IEC':>12}  {'STD':>12}  {'ratio':>8}")
for N in N_list:
    iec = res_study[N]['IEC']
    std = res_study[N]['STD']
    ratio = std / iec if iec > 0 and np.isfinite(iec) else float('nan')
    ie_s = f"{iec:.3e}" if np.isfinite(iec) else "BLOWN UP"
    st_s = f"{std:.3e}" if np.isfinite(std) else "BLOWN UP"
    ra_s = f"{ratio:.1f}x" if np.isfinite(ratio) else "—"
    print(f"  {N:>6}  {ie_s:>12}  {st_s:>12}  {ra_s:>8}")

# ── 3. 플롯 ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
colors = {'IEC': 'b', 'STD': 'r'}

# 최종 프로파일 (N=200)
for scheme, res in main_results.items():
    c = colors[scheme]
    axes[0,0].plot(res['x'], res['rho'], c+'-', lw=1.5, label=scheme)
    axes[0,1].plot(res['x'], res['u'],   c+'-', lw=1.5, label=scheme)
    axes[1,0].plot(res['x'], res['P'],   c+'-', lw=1.5, label=scheme)
    axes[1,1].plot(res['x'], res['Y0'],  c+'-', lw=1.5, label=scheme)

titles = ['Density (non-dim)', 'Velocity (non-dim)', 'Pressure (non-dim)', 'Y₀ (water)']
for ax, title in zip(axes.flat, titles):
    ax.set_title(title); ax.set_xlabel('x'); ax.legend(); ax.grid(alpha=0.3)
axes[1,0].axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.5, label='P=1')

plt.suptitle(f'§4.2.4 Mach-100 Water Jet (IEC vs STD)  N={N_MAIN}  t={T_END:.1e}\n'
             f'eps_abs={EPS_ABS}  (fixed physical interface width)', fontsize=10)
plt.tight_layout()
os.makedirs('output', exist_ok=True)
fname = 'output/mach100_comparison.png'
plt.savefig(fname, dpi=120)
print(f"\nSaved: {fname}")

# PE 이력 + 해상도 연구 플롯
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))

ax = axes2[0]
for scheme, res in main_results.items():
    if len(res['times']) > 0:
        ax.semilogy(res['times'], res['pe'], colors[scheme]+'-', lw=1.5, label=scheme)
ax.set_xlabel('t'); ax.set_ylabel('max|P - P₀|')
ax.set_title(f'PE Error History  N={N_MAIN}  eps_abs={EPS_ABS}')
ax.legend(); ax.grid(True, which='both', alpha=0.3)

ax = axes2[1]
Ns = [N for N in N_list if np.isfinite(res_study[N]['IEC'])]
iec_pe = [res_study[N]['IEC'] for N in Ns]
std_pe = [res_study[N]['STD'] for N in N_list]
ax.loglog(Ns, iec_pe, 'b-o', lw=1.5, ms=6, label='IEC')
ax.loglog(N_list, std_pe, 'r-s', lw=1.5, ms=6, label='STD')
ax.set_xlabel('N'); ax.set_ylabel('PE error (max|P-1|) at t=1e-3')
ax.set_title(f'Resolution Study  eps_abs={EPS_ABS}')
ax.legend(); ax.grid(True, which='both', alpha=0.3)

plt.suptitle('§4.2.4 Mach-100 Water Jet  —  fixed eps_abs=0.04', fontsize=10)
plt.tight_layout()
fname2 = 'output/mach100_resolution.png'
plt.savefig(fname2, dpi=120)
print(f"Saved: {fname2}")
