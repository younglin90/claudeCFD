"""V3 figures: accuracy verification for ALL 6 cases."""
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


def run_pair(case_make, baseline_runner, tol, max_steps, **safenn_kw):
    c = case_make()
    fb, _ = baseline_runner(c, max_steps=max_steps, tol=tol, check_every=100, verbose=False)
    c2 = case_make()
    fs, _ = solve_safe_nn(c2, max_outer=300, tol=tol, krylov_max=10, krylov_tol=1e-3,
                            kinetic_substeps=15, beta_max=0.7, eps_accept=0.10, verbose=False)
    return c, fb, fs


# ============================================================
# Case 1: Kolmogorov u(y) = U_amp sin(k_f * 2π y / N), analytical exists
# ============================================================
c, fb, fs = run_pair(lambda: KolmogorovCase(N=32, nu=0.05, F0=2e-4, kf=1),
                      solve_baseline_periodic, 1e-7, 15000)
Ub_y = c.project(fb)[1].mean(axis=1)
Us_y = c.project(fs)[1].mean(axis=1)
y = np.arange(c.N)
y_fine = np.linspace(0, c.N, 200)
u_exact = c.U_amp * np.sin(c.k_lat * y_fine)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(y_fine, u_exact, '-', color='black', label='Analytical', lw=2)
ax.plot(y, Ub_y, 'o', color='gray', label=f'Baseline LBM (RMS={np.sqrt(np.mean((Ub_y - c.U_amp*np.sin(c.k_lat*y))**2)):.2e})', ms=7)
ax.plot(y, Us_y, 's', color='tab:red', label=f'Safe-NN (RMS={np.sqrt(np.mean((Us_y - c.U_amp*np.sin(c.k_lat*y))**2)):.2e})', ms=6, alpha=0.8)
ax.set_xlabel('y'); ax.set_ylabel('u(y)')
ax.set_title('Case 1. Kolmogorov flow N=32 — u-velocity profile vs analytical')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('figs/v3_acc1_kolmogorov.png', dpi=120, bbox_inches='tight')
plt.close(); print('Acc1 Kolmogorov saved')


# ============================================================
# Case 2: Channel parabolic u(y), analytical u_max = F0*L²/(8ν)
# ============================================================
c, fb, fs = run_pair(lambda: ChannelCase(N=32, nu=0.05, F0=1e-5),
                      solve_baseline_periodic, 1e-7, 15000)
Ub_y = c.project(fb)[1].mean(axis=1)
Us_y = c.project(fs)[1].mean(axis=1)
y = np.arange(c.N)
L = c.N - 1
u_max = c.F0 * L * L / (8 * c.nu)
u_exact = 4 * u_max * (y / L) * (1 - y / L)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(y, u_exact, '-', color='black', label='Analytical (parabolic)', lw=2)
ax.plot(y, Ub_y, 'o', color='gray', label=f'Baseline LBM (RMS={np.sqrt(np.mean((Ub_y - u_exact)**2)):.2e})', ms=7)
ax.plot(y, Us_y, 's', color='tab:red', label=f'Safe-NN (RMS={np.sqrt(np.mean((Us_y - u_exact)**2)):.2e})', ms=6, alpha=0.8)
ax.set_xlabel('y'); ax.set_ylabel('u(y)')
ax.set_title('Case 2. Channel (Poiseuille) N=32 — u-velocity profile vs analytical')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('figs/v3_acc2_channel.png', dpi=120, bbox_inches='tight')
plt.close(); print('Acc2 Channel saved')


# ============================================================
# Case 3: Couette linear u(y) = U_wall * y / L
# ============================================================
c, fb, fs = run_pair(lambda: CouetteCase(N=32, nu=0.05, U_wall=0.05),
                      solve_baseline_periodic, 1e-7, 15000)
Ub_y = c.project(fb)[1].mean(axis=1)
Us_y = c.project(fs)[1].mean(axis=1)
y = np.arange(c.N)
L = c.N - 1
u_exact = c.U_wall * y / L
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(y, u_exact, '-', color='black', label='Analytical (linear)', lw=2)
ax.plot(y, Ub_y, 'o', color='gray', label=f'Baseline LBM (RMS={np.sqrt(np.mean((Ub_y - u_exact)**2)):.2e})', ms=7)
ax.plot(y, Us_y, 's', color='tab:red', label=f'Safe-NN (RMS={np.sqrt(np.mean((Us_y - u_exact)**2)):.2e})', ms=6, alpha=0.8)
ax.set_xlabel('y'); ax.set_ylabel('u(y)')
ax.set_title('Case 3. Couette flow N=32 — u-velocity profile vs analytical')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('figs/v3_acc3_couette.png', dpi=120, bbox_inches='tight')
plt.close(); print('Acc3 Couette saved')


# ============================================================
# Case 4: Lid-driven cavity Re=100 N=65 (Ghia comparison)
# ============================================================
c, fb, fs = run_pair(lambda: LBMCavity(N=65, Re=100, U_wall=0.1),
                      solve_baseline, 5e-7, 100000)
U_b = c.project(fb); U_s = c.project(fs)
ub_cl = U_b[1][:, c.N // 2] / c.U_wall
us_cl = U_s[1][:, c.N // 2] / c.U_wall
y_norm = np.linspace(0, 1, c.N)
# Ghia 1982 Re=100 reference (excerpt)
ghia_y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
                    0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])
ghia_u = np.array([0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090,
                    -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.00000])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(ub_cl, y_norm, 'o-', color='gray', label='Baseline LBM', lw=2, ms=5)
axes[0].plot(us_cl, y_norm, 's--', color='tab:red', label='Safe-NN', lw=2, ms=5, alpha=0.8)
axes[0].plot(ghia_u, ghia_y, 'D', color='black', label='Ghia 1982 ref', ms=8, markerfacecolor='none')
axes[0].set_xlabel('u / U_wall'); axes[0].set_ylabel('y / L')
axes[0].set_title('Case 4. Cavity Re=100 N=65 — Centerline u-velocity (Ghia 1982)')
axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.3)

u_mag_b = np.sqrt(U_b[1]**2 + U_b[2]**2)
u_mag_s = np.sqrt(U_s[1]**2 + U_s[2]**2)
diff = np.abs(u_mag_b - u_mag_s)
im = axes[1].imshow(diff, origin='lower', cmap='Reds')
axes[1].set_title(f'Case 4. Cavity Re=100 — |u_baseline - u_safe-nn|, max={diff.max():.2e}')
plt.colorbar(im, ax=axes[1], fraction=0.046)
axes[1].set_xticks([]); axes[1].set_yticks([])

plt.tight_layout(); plt.savefig('figs/v3_acc4_cavity_re100.png', dpi=120, bbox_inches='tight')
plt.close(); print('Acc4 Cavity Re=100 saved')


# ============================================================
# Case 5: Multi-cylinder voxel
# ============================================================
N = 32
def make_voxel():
    chi = np.ones((N, N))
    rng = np.random.RandomState(7)
    radius = max(2, N // 12)
    for _ in range(6):
        cx = rng.randint(radius, N - radius); cy = rng.randint(radius, N - radius)
        chi = chi * build_cylinder_mask(N, cx, cy, radius)
    return VoxelCase(chi, nu=0.05, F0=2e-4, kf=1)
c, fb, fs = run_pair(make_voxel, solve_baseline_periodic, 1e-7, 8000)
U_b = c.project(fb); U_s = c.project(fs)
u_mag_b = np.sqrt(U_b[1]**2 + U_b[2]**2)
u_mag_s = np.sqrt(U_s[1]**2 + U_s[2]**2)
diff = np.abs(u_mag_b - u_mag_s)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
im0 = axes[0].imshow(u_mag_b, origin='lower', cmap='viridis')
axes[0].set_title('Baseline LBM  |u|')
plt.colorbar(im0, ax=axes[0], fraction=0.046)
im1 = axes[1].imshow(u_mag_s, origin='lower', cmap='viridis',
                       vmin=u_mag_b.min(), vmax=u_mag_b.max())
axes[1].set_title('Safe-NN  |u|')
plt.colorbar(im1, ax=axes[1], fraction=0.046)
im2 = axes[2].imshow(diff, origin='lower', cmap='Reds')
axes[2].set_title(f'|Baseline - Safe-NN|  max={diff.max():.2e}')
plt.colorbar(im2, ax=axes[2], fraction=0.046)
for ax in axes: ax.set_xticks([]); ax.set_yticks([])
plt.suptitle('Case 5. Multi-cylinder voxel flow N=32', fontsize=12)
plt.tight_layout(); plt.savefig('figs/v3_acc5_multicyl.png', dpi=120, bbox_inches='tight')
plt.close(); print('Acc5 Multi-cylinder saved')


# ============================================================
# Case 6: Lid-driven cavity Re=400 N=65 (Ghia comparison)
# ============================================================
c, fb, fs = run_pair(lambda: LBMCavity(N=65, Re=400, U_wall=0.1),
                      solve_baseline, 5e-7, 150000)
U_b = c.project(fb); U_s = c.project(fs)
ub_cl = U_b[1][:, c.N // 2] / c.U_wall
us_cl = U_s[1][:, c.N // 2] / c.U_wall
y_norm = np.linspace(0, 1, c.N)
# Ghia 1982 Re=400 reference (excerpt)
ghia_y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
                    0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])
ghia_u_re400 = np.array([0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, -0.17119,
                          -0.11477, 0.02135, 0.16256, 0.29093, 0.55892, 0.61756, 0.68439, 0.75837, 1.00000])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(ub_cl, y_norm, 'o-', color='gray', label='Baseline LBM', lw=2, ms=5)
axes[0].plot(us_cl, y_norm, 's--', color='tab:red', label='Safe-NN', lw=2, ms=5, alpha=0.8)
axes[0].plot(ghia_u_re400, ghia_y, 'D', color='black', label='Ghia 1982 ref', ms=8, markerfacecolor='none')
axes[0].set_xlabel('u / U_wall'); axes[0].set_ylabel('y / L')
axes[0].set_title('Case 6. Cavity Re=400 N=65 — Centerline u-velocity (Ghia 1982)')
axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.3)

u_mag_b = np.sqrt(U_b[1]**2 + U_b[2]**2)
u_mag_s = np.sqrt(U_s[1]**2 + U_s[2]**2)
diff = np.abs(u_mag_b - u_mag_s)
im = axes[1].imshow(diff, origin='lower', cmap='Reds')
axes[1].set_title(f'Case 6. Cavity Re=400 — |u_baseline - u_safe-nn|, max={diff.max():.2e}')
plt.colorbar(im, ax=axes[1], fraction=0.046)
axes[1].set_xticks([]); axes[1].set_yticks([])

plt.tight_layout(); plt.savefig('figs/v3_acc6_cavity_re400.png', dpi=120, bbox_inches='tight')
plt.close(); print('Acc6 Cavity Re=400 saved')

print('All 6 accuracy figures saved')
