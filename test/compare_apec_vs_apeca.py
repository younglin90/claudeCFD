"""
APEC (Sec. 2.8, KEEP) vs APEC_A (App. A, HLLC) 비교
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cpg_apec_1d import run, OUTPUT_DIR

N     = 501
T_END = 8.0
CFL   = 0.6

print("Running APEC (KEEP, Sec.2.8)...")
res_apec  = run('APEC',   N=N, t_end=T_END, CFL=CFL, verbose=True)

print("\nRunning FC (KEEP baseline)...")
res_fc    = run('FC',     N=N, t_end=T_END, CFL=CFL, verbose=True)

print("\nRunning APEC_A (HLLC, App.A)...")
res_apeca = run('APEC_A', N=N, t_end=T_END, CFL=CFL, verbose=True)

print("\nRunning FC_HLLC (HLLC baseline)...")
res_hllc  = run('FC_HLLC',N=N, t_end=T_END, CFL=CFL, verbose=True)

# ─── unpack ──────────────────────────────────────────────────
def unpack(res):
    x, r1, r2, u, p = res[:5]
    t_h, pe_h, en_h = res[5], res[6], res[7]
    return x, r1, r2, u, p, t_h, pe_h, en_h

xa,  r1a,  r2a,  ua,  pa,  ta,  pea,  ena  = unpack(res_apec)
xf,  r1f,  r2f,  uf,  pf,  tf,  pef,  enf  = unpack(res_fc)
xaa, r1aa, r2aa, uaa, paa, taa, peaa, enaa = unpack(res_apeca)
xh,  r1h,  r2h,  uh,  ph,  th,  peh,  enh  = unpack(res_hllc)

# ─── Figure ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

C = {'KEEP':'#1f77b4', 'HLLC':'#d62728',
     'FC_K':'#aec7e8', 'FC_H':'#f7b6b2'}

# ── Row 1: PE error, Energy error, Pressure profile ──────────
ax = axes[0, 0]
ax.semilogy(tf,  np.maximum(pef,  1e-16), color=C['FC_K'], ls='--', label='FC-NPE (KEEP baseline)')
ax.semilogy(ta,  np.maximum(pea,  1e-16), color=C['KEEP'], ls='-',  lw=2, label='APEC (KEEP, Sec.2.8)')
ax.semilogy(th,  np.maximum(peh,  1e-16), color=C['FC_H'], ls='--', label='FC-NPE (HLLC baseline)')
ax.semilogy(taa, np.maximum(peaa, 1e-16), color=C['HLLC'], ls='-',  lw=2, label='APEC-A (HLLC, App.A)')
ax.set_xlabel('t');  ax.set_ylabel(r'$\|p - p_0\|_{L_2}\,/\,p_0$')
ax.set_title('(a) PE error history')
ax.legend(fontsize=8)

ax = axes[0, 1]
ax.semilogy(tf,  np.maximum(enf,  1e-16), color=C['FC_K'], ls='--', label='FC-NPE (KEEP)')
ax.semilogy(ta,  np.maximum(ena,  1e-16), color=C['KEEP'], ls='-',  lw=2, label='APEC (KEEP)')
ax.semilogy(th,  np.maximum(enh,  1e-16), color=C['FC_H'], ls='--', label='FC-NPE (HLLC)')
ax.semilogy(taa, np.maximum(enaa, 1e-16), color=C['HLLC'], ls='-',  lw=2, label='APEC-A (HLLC)')
ax.set_xlabel('t');  ax.set_ylabel('Energy conservation error')
ax.set_title('(b) Total energy conservation')
ax.legend(fontsize=8)

ax = axes[0, 2]
ax.plot(xf,  pf,  color=C['FC_K'], ls='--', label='FC-NPE (KEEP)')
ax.plot(xa,  pa,  color=C['KEEP'], ls='-',  lw=2, label='APEC (KEEP)')
ax.plot(xh,  ph,  color=C['FC_H'], ls='--', label='FC-NPE (HLLC)')
ax.plot(xaa, paa, color=C['HLLC'], ls='-',  lw=2, label='APEC-A (HLLC)')
ax.axhline(0.9, color='k', ls=':', lw=0.8, label='p₀ = 0.9')
ax.set_xlabel('x');  ax.set_ylabel('p')
ax.set_title(f'(c) Pressure profile  t = {T_END:.1f}')
ax.legend(fontsize=8)

# ── Row 2: ρ₁, ρ₂ profiles, PE improvement ratio ─────────────
ax = axes[1, 0]
ax.plot(xf,  r1f,  color=C['FC_K'], ls='--', label='FC-NPE (KEEP)')
ax.plot(xa,  r1a,  color=C['KEEP'], ls='-',  lw=2, label='APEC (KEEP)')
ax.plot(xh,  r1h,  color=C['FC_H'], ls='--', label='FC-NPE (HLLC)')
ax.plot(xaa, r1aa, color=C['HLLC'], ls='-',  lw=2, label='APEC-A (HLLC)')
ax.set_xlabel('x');  ax.set_ylabel(r'$\rho_1$')
ax.set_title(r'(d) $\rho_1$ profile  t = 8.0')
ax.legend(fontsize=8)

ax = axes[1, 1]
ax.plot(xf,  r2f,  color=C['FC_K'], ls='--', label='FC-NPE (KEEP)')
ax.plot(xa,  r2a,  color=C['KEEP'], ls='-',  lw=2, label='APEC (KEEP)')
ax.plot(xh,  r2h,  color=C['FC_H'], ls='--', label='FC-NPE (HLLC)')
ax.plot(xaa, r2aa, color=C['HLLC'], ls='-',  lw=2, label='APEC-A (HLLC)')
ax.set_xlabel('x');  ax.set_ylabel(r'$\rho_2$')
ax.set_title(r'(e) $\rho_2$ profile  t = 8.0')
ax.legend(fontsize=8)

# ── (f) PE 개선 비율: APEC/FC vs APEC_A/FC_HLLC ──────────────
ax = axes[1, 2]
# APEC improvement over FC-KEEP
min_len_k = min(len(ta), len(tf))
ratio_keep = np.maximum(pef[:min_len_k], 1e-16) / np.maximum(pea[:min_len_k], 1e-16)

# APEC_A improvement over FC-HLLC
min_len_h = min(len(taa), len(th))
ratio_hllc = np.maximum(peh[:min_len_h], 1e-16) / np.maximum(peaa[:min_len_h], 1e-16)

ax.plot(ta[:min_len_k],  ratio_keep, color=C['KEEP'], ls='-', lw=2,
        label='FC / APEC  (KEEP)')
ax.plot(taa[:min_len_h], ratio_hllc, color=C['HLLC'], ls='-', lw=2,
        label='FC_HLLC / APEC_A  (HLLC)')
ax.axhline(1.0, color='k', ls=':', lw=0.8)
ax.set_xlabel('t');  ax.set_ylabel('PE improvement ratio  (baseline / APEC)')
ax.set_title('(f) PE improvement: APEC vs APEC-A')
ax.set_yscale('log')
ax.legend(fontsize=9)

fig.suptitle(
    f'APEC (Sec.2.8, KEEP)  vs  APEC-A (App.A, HLLC)   N={N}, CFL={CFL}',
    fontsize=13)
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, 'compare_apec_vs_apeca.png')
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved → {path}")
