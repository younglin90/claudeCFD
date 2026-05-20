"""V2 figures: baseline LBM vs Safe-NN only."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lbm_periodic import KolmogorovCase
from lbm_channel import ChannelCase
from lbm_couette import CouetteCase
from lbm_core import LBMCavity
from lbm_voxel import VoxelCase, build_cylinder_mask
from solver_scmk import solve_baseline_periodic
from solver_baseline import solve_baseline
from solver_safe_nn import solve_safe_nn

os.makedirs('figs', exist_ok=True)


def hist_arrays(h):
    return np.array([e[2] for e in h]), np.array([e[1] for e in h])


# ============================================================
# Fig 1: Convergence — Baseline vs Safe-NN (4 panels)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
cases = [
    ('Kolmogorov N=32',     lambda: KolmogorovCase(N=32, nu=0.05, F0=2e-4, kf=1), solve_baseline_periodic, 1e-7, 10000),
    ('Channel N=32',         lambda: ChannelCase(N=32, nu=0.05, F0=1e-5),          solve_baseline_periodic, 1e-7, 12000),
    ('Cavity Re=100 N=33',   lambda: LBMCavity(N=33, Re=100, U_wall=0.1),          solve_baseline,          5e-7, 50000),
    ('Cavity Re=400 N=49',   lambda: LBMCavity(N=49, Re=400, U_wall=0.1),          solve_baseline,          5e-7, 100000),
]
for ax, (label, mk, bsl, tol, mx) in zip(axes.ravel(), cases):
    c = mk()
    _, hb = bsl(c, max_steps=mx, tol=tol, check_every=50, verbose=False)
    c2 = mk()
    _, hs = solve_safe_nn(c2, max_outer=200, tol=tol, krylov_max=10, krylov_tol=1e-3,
                            kinetic_substeps=15, beta_max=0.7, eps_accept=0.10, verbose=False)
    lbe_b, res_b = hist_arrays(hb)
    lbe_s, res_s = hist_arrays(hs)
    ax.semilogy(lbe_b, res_b, '-', color='gray', label='Baseline LBM', lw=2)
    ax.semilogy(lbe_s, res_s, '-', color='tab:red', label='Safe-NN', lw=2)
    su = hb[-1][2] / hs[-1][2] if hs[-1][2] > 0 and np.isfinite(hs[-1][1]) else 0
    ax.set_title(f'{label}  (speedup = {su:.1f}×)')
    ax.set_xlabel('LBE call count'); ax.set_ylabel('Residual norm')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figs/v2_fig1_convergence.png', dpi=120, bbox_inches='tight')
plt.close()
print('Fig 1 saved')


# ============================================================
# Fig 2: Per-case LBE bar chart — Baseline vs Safe-NN
# ============================================================
cases2 = ['Kol N=32', 'Chan N=32', 'Couette\nN=32', 'Cav Re=100\nN=33', 'Multi-cyl\nN=32', 'Cav Re=400\nN=49']
base_lbe = [3015, 5427, 5829, 3216, 2211, 8040]
sfn_lbe  = [134, 170, 30, 472, 359, 1421]
speedup  = [b / s for b, s in zip(base_lbe, sfn_lbe)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(cases2)); w = 0.35
ax1.bar(x - w/2, base_lbe, w, label='Baseline LBM', color='gray')
ax1.bar(x + w/2, sfn_lbe,  w, label='Safe-NN',     color='tab:red')
ax1.set_xticks(x); ax1.set_xticklabels(cases2, fontsize=9)
ax1.set_ylabel('LBE call count'); ax1.set_yscale('log')
ax1.set_title('LBE Call Count (log scale)')
ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3, axis='y')

ax2.bar(x, speedup, color='tab:red')
for i, s in enumerate(speedup):
    ax2.text(i, s + 5, f'{s:.1f}×', ha='center', fontsize=10, weight='bold')
ax2.set_xticks(x); ax2.set_xticklabels(cases2, fontsize=9)
ax2.set_ylabel('Speedup (Baseline / Safe-NN)')
ax2.set_title('Safe-NN Speedup over Baseline')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figs/v2_fig2_lbe_speedup.png', dpi=120, bbox_inches='tight')
plt.close()
print('Fig 2 saved')


# ============================================================
# Fig 3: Field profile validation — Channel u(y) + Cavity centerline
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Channel u(y) profile
c_ch = ChannelCase(N=32, nu=0.05, F0=1e-5)
fb, _ = solve_baseline_periodic(c_ch, max_steps=12000, tol=1e-7, check_every=50, verbose=False)
c_ch2 = ChannelCase(N=32, nu=0.05, F0=1e-5)
fs, _ = solve_safe_nn(c_ch2, max_outer=200, tol=1e-7, krylov_max=10, krylov_tol=1e-3,
                        kinetic_substeps=15, beta_max=0.7, eps_accept=0.10, verbose=False)
Ub = c_ch.project(fb)[1].mean(axis=1)
Us = c_ch.project(fs)[1].mean(axis=1)
y = np.arange(len(Ub))
ax1.plot(y, Ub, 'o-', color='gray', label='Baseline LBM', lw=2, ms=6)
ax1.plot(y, Us, 's--', color='tab:red', label='Safe-NN', lw=2, ms=5, alpha=0.8)
ax1.set_xlabel('y (lattice unit)'); ax1.set_ylabel('u(y) (lattice unit)')
ax1.set_title('Channel u-velocity Profile')
ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)

# Cavity Re=100 centerline u-velocity vs y
c_cv = LBMCavity(N=65, Re=100, U_wall=0.1)
fb_cv, _ = solve_baseline(c_cv, max_steps=80000, tol=5e-7, check_every=200, verbose=False)
c_cv2 = LBMCavity(N=65, Re=100, U_wall=0.1)
fs_cv, _ = solve_safe_nn(c_cv2, max_outer=300, tol=5e-7, krylov_max=10, krylov_tol=1e-3,
                            kinetic_substeps=15, beta_max=0.7, eps_accept=0.10, verbose=False)
ub_cv = c_cv.project(fb_cv)[1][:, c_cv.N // 2]
us_cv = c_cv.project(fs_cv)[1][:, c_cv.N // 2]
y2 = np.linspace(0, 1, c_cv.N)
ax2.plot(ub_cv / 0.1, y2, 'o-', color='gray', label='Baseline LBM', lw=2, ms=6)
ax2.plot(us_cv / 0.1, y2, 's--', color='tab:red', label='Safe-NN', lw=2, ms=5, alpha=0.8)
ax2.set_xlabel('u / U_wall'); ax2.set_ylabel('y / L')
ax2.set_title('Lid-driven Cavity Re=100, Centerline u')
ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figs/v2_fig3_field_profiles.png', dpi=120, bbox_inches='tight')
plt.close()
print('Fig 3 saved')


# ============================================================
# Fig 4: Cavity Re=100 contour — Baseline vs Safe-NN
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
U_b = c_cv.project(fb_cv)
U_s = c_cv.project(fs_cv)
u_mag_b = np.sqrt(U_b[1]**2 + U_b[2]**2)
u_mag_s = np.sqrt(U_s[1]**2 + U_s[2]**2)
diff = np.abs(u_mag_b - u_mag_s)

im0 = axes[0].imshow(u_mag_b, origin='lower', cmap='viridis')
axes[0].set_title('Baseline LBM  |u|')
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(u_mag_s, origin='lower', cmap='viridis',
                       vmin=u_mag_b.min(), vmax=u_mag_b.max())
axes[1].set_title('Safe-NN  |u|')
plt.colorbar(im1, ax=axes[1], fraction=0.046)

im2 = axes[2].imshow(diff, origin='lower', cmap='Reds')
axes[2].set_title('Absolute Difference  ||u|_base - |u|_safenn|')
plt.colorbar(im2, ax=axes[2], fraction=0.046)

for ax in axes: ax.set_xticks([]); ax.set_yticks([])
plt.suptitle(f'Cavity Re=100 N={c_cv.N}: max diff = {diff.max():.2e}', fontsize=11)
plt.tight_layout()
plt.savefig('figs/v2_fig4_cavity_contour.png', dpi=120, bbox_inches='tight')
plt.close()
print('Fig 4 saved')

print('All v2 figures saved')
