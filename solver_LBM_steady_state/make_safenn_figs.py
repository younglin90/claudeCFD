"""Generate Safe-NN paper figures.

Fig 1: Convergence histories (Kol + Channel + Cavity) — Baseline / Lean / NN / Safe-NN
Fig 2: Per-case speedup bar chart
Fig 3: Composite score comparison
Fig 4: Cavity Re=400 stress — NN NaN vs Safe-NN converged
Fig 5: Ablation bar chart
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lbm_periodic import KolmogorovCase
from lbm_channel import ChannelCase
from lbm_core import LBMCavity
from solver_scmk import solve_baseline_periodic
from solver_baseline import solve_baseline
from solver_lean import solve_lean
from solver_nesterov_newton import solve_nn
from solver_safe_nn import solve_safe_nn

os.makedirs('figs', exist_ok=True)

# === Fig 1: Convergence histories ===
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

def hist_arrays(h):
    lbe = [e[2] for e in h]
    res = [e[1] for e in h]
    return np.array(lbe), np.array(res)

# Kolmogorov
c = KolmogorovCase(N=32, nu=0.05, F0=2e-4, kf=1)
_, hb = solve_baseline_periodic(c, max_steps=10000, tol=1e-7, check_every=50, verbose=False)
_, hl = solve_lean(KolmogorovCase(N=32, nu=0.05, F0=2e-4, kf=1), max_outer=200, tol=1e-7, krylov_max=10, krylov_tol=1e-3, kinetic_substeps=15, verbose=False)
_, hn = solve_nn(KolmogorovCase(N=32, nu=0.05, F0=2e-4, kf=1), max_outer=200, tol=1e-7, krylov_max=10, krylov_tol=1e-3, kinetic_substeps=15, beta_max=0.7, verbose=False)
_, hs = solve_safe_nn(KolmogorovCase(N=32, nu=0.05, F0=2e-4, kf=1), max_outer=200, tol=1e-7, krylov_max=10, krylov_tol=1e-3, kinetic_substeps=15, beta_max=0.7, eps_accept=0.10, verbose=False)
for h, l, c_ in [(hb, 'Baseline LBM', 'gray'), (hl, 'Lean SCMK', 'tab:blue'), (hn, 'NN', 'tab:orange'), (hs, 'Safe-NN', 'tab:red')]:
    lbe, res = hist_arrays(h)
    axes[0].semilogy(lbe, res, '-', label=l, color=c_, lw=1.5)
axes[0].set_xlabel('LBE call count'); axes[0].set_ylabel('Residual norm')
axes[0].set_title('Kolmogorov N=32'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

# Channel
c = ChannelCase(N=32, nu=0.05, F0=1e-5)
_, hb = solve_baseline_periodic(c, max_steps=12000, tol=1e-7, check_every=50, verbose=False)
_, hl = solve_lean(ChannelCase(N=32, nu=0.05, F0=1e-5), max_outer=200, tol=1e-7, krylov_max=10, krylov_tol=1e-3, kinetic_substeps=15, verbose=False)
_, hn = solve_nn(ChannelCase(N=32, nu=0.05, F0=1e-5), max_outer=200, tol=1e-7, krylov_max=10, krylov_tol=1e-3, kinetic_substeps=15, beta_max=0.7, verbose=False)
_, hs = solve_safe_nn(ChannelCase(N=32, nu=0.05, F0=1e-5), max_outer=200, tol=1e-7, krylov_max=10, krylov_tol=1e-3, kinetic_substeps=15, beta_max=0.7, eps_accept=0.10, verbose=False)
for h, l, c_ in [(hb, 'Baseline LBM', 'gray'), (hl, 'Lean SCMK', 'tab:blue'), (hn, 'NN', 'tab:orange'), (hs, 'Safe-NN', 'tab:red')]:
    lbe, res = hist_arrays(h)
    axes[1].semilogy(lbe, res, '-', label=l, color=c_, lw=1.5)
axes[1].set_xlabel('LBE call count'); axes[1].set_ylabel('Residual norm')
axes[1].set_title('Channel N=32'); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

# Cavity Re=400 (stress)
c = LBMCavity(N=49, Re=400, U_wall=0.1)
_, hb = solve_baseline(c, max_steps=10000, tol=5e-7, check_every=50, verbose=False)
_, hl = solve_lean(LBMCavity(N=49, Re=400, U_wall=0.1), max_outer=200, tol=5e-7, krylov_max=10, krylov_tol=1e-3, kinetic_substeps=15, verbose=False)
try:
    _, hn = solve_nn(LBMCavity(N=49, Re=400, U_wall=0.1), max_outer=200, tol=5e-7, krylov_max=10, krylov_tol=1e-3, kinetic_substeps=15, beta_max=0.7, verbose=False)
except Exception:
    hn = []
_, hs = solve_safe_nn(LBMCavity(N=49, Re=400, U_wall=0.1), max_outer=200, tol=5e-7, krylov_max=10, krylov_tol=1e-3, kinetic_substeps=15, beta_max=0.7, eps_accept=0.10, verbose=False)
plot_pairs = [(hb, 'Baseline LBM', 'gray'), (hl, 'Lean SCMK', 'tab:blue')]
if hn: plot_pairs.append((hn, 'NN (diverges)', 'tab:orange'))
plot_pairs.append((hs, 'Safe-NN', 'tab:red'))
for h, l, c_ in plot_pairs:
    lbe, res = hist_arrays(h)
    res_safe = np.where(np.isfinite(res), res, np.nan)
    axes[2].semilogy(lbe, res_safe, '-', label=l, color=c_, lw=1.5)
axes[2].set_xlabel('LBE call count'); axes[2].set_ylabel('Residual norm')
axes[2].set_title('Cavity Re=400 N=49 (stress)'); axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figs/fig1_convergence.png', dpi=120, bbox_inches='tight')
plt.close()
print('Fig1 saved')

# === Fig 2: Per-case speedup ===
cases = ['Kol N=32', 'Chan N=32', 'Couette', 'Cav Re=100', 'Multi-cyl', 'Cav Re=400']
lean = [21.7, 46.6, 188.0, 8.67, 3.0, 5.30]
san = [56.7, 19.6, 188.0, 5.75, 2.22, 4.22]
nn  = [33.0, 97.3, 188.0, 6.03, 4.15, 0]      # NaN → 0
sfn = [22.5, 31.9, 194.3, 6.82, 6.16, 5.66]

x = np.arange(len(cases)); w = 0.20
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - 1.5*w, lean, w, label='Lean SCMK', color='tab:blue')
ax.bar(x - 0.5*w, san,  w, label='SAN',       color='tab:green')
ax.bar(x + 0.5*w, nn,   w, label='NN',        color='tab:orange')
ax.bar(x + 1.5*w, sfn,  w, label='Safe-NN',   color='tab:red')
# Mark NN NaN
ax.annotate('NaN', xy=(x[-1] + 0.5*w, 2), ha='center', fontsize=9, color='tab:orange', weight='bold')
ax.set_xticks(x); ax.set_xticklabels(cases, fontsize=9)
ax.set_ylabel('LBE-call speedup vs baseline'); ax.set_yscale('log')
ax.set_title('Per-case Speedup')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('figs/fig2_speedup_per_case.png', dpi=120, bbox_inches='tight')
plt.close()
print('Fig2 saved')

# === Fig 3: Composite score ===
solvers = ['Baseline', 'Lean SCMK', 'SAN', 'NN', 'Safe-NN v4']
comp = [1.0, 41.39, 42.31, 44.74, 45.41]
colors = ['gray', 'tab:blue', 'tab:green', 'tab:orange', 'tab:red']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(solvers, comp, color=colors)
for bar, val in zip(bars, comp):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.7, f'{val:.2f}',
            ha='center', fontsize=10)
ax.set_ylabel('Composite score')
ax.set_title('5-case Composite Score (higher = better)')
ax.set_ylim(0, 50)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('figs/fig3_composite.png', dpi=120, bbox_inches='tight')
plt.close()
print('Fig3 saved')

# === Fig 4: Ablation ===
variants = ['Safe-NN v4\n(full)', 'No K-anneal', 'No safeguard\n(=NN)', 'No Nesterov\n(=Lean)', 'No AP corr']
scores = [45.41, 40.69, 44.74, 41.39, 30.0]
fail = [False, False, True, False, False]  # NN: Cavity NaN
colors_a = ['tab:red' if not f else 'tab:orange' for f in fail]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(variants, scores, color=colors_a)
for bar, val, f in zip(bars, scores, fail):
    label = f'{val:.2f}' + (' (Cav NaN)' if f else '')
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.7, label,
            ha='center', fontsize=9)
ax.set_ylabel('Composite score')
ax.set_title('Ablation: removing each component degrades performance')
ax.set_ylim(0, 50)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('figs/fig4_ablation.png', dpi=120, bbox_inches='tight')
plt.close()
print('Fig4 saved')

print('All figures saved to figs/')
