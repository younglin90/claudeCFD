"""Test 02A: Abgrall PE advection — NASG Water + Ideal Air
using acoustic_method='dumbser_casulli' + material CFL.

Ref: plan_report.md §6 test spec.
     Dumbser & Casulli 2016 AMC 272:479
     Casulli & Zanolli 2012 JCAM 239:185

PASS criterion: err_p < 1e-2, err_u < 1e-2 at t_end=1.0 s
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from solver.He2024.eos_general import IdealEOS, NASGEOS

R = '/home/younglin90/work/claude_code/claudeCFD/results'
OUT = f'{R}/cat_A_exact'
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# EOS definitions — SPEC values (must not be changed)
# ---------------------------------------------------------------------------
# Air (phase 1 = gas, solver convention)
ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
eos1 = IdealEOS(gamma=1.4, kv=717.5)

# Water (phase 2 = liquid, NASG spec)
ph2 = {'gamma': 1.187, 'pinf': 7.028e8, 'kv': 3610.0,
       'b': 6.61e-4, 'eta': -1.177788e6}
eos2 = NASGEOS(gamma=1.187, pinf=7.028e8, kv=3610.0,
               b=6.61e-4, eta=-1.177788e6)

# ---------------------------------------------------------------------------
# Domain and initial conditions
# ---------------------------------------------------------------------------
N, L = 50, 1.0
dx = L / N
x = np.linspace(dx / 2, L - dx / 2, N)

p0, u0, T0 = 1.0e5, 1.0, 300.0

# α_air = 1 - α_water; water region [0.4, 0.6]
a_water = ((x >= 0.4) & (x <= 0.6)).astype(float)
a1 = (1 - a_water) * (1 - 1e-6) + a_water * 1e-6   # α_air

rho1_0 = eos1.density(np.full(N, p0), np.full(N, T0))  # air density
rho2_0 = eos2.density(np.full(N, p0), np.full(N, T0))  # water density
u = np.full(N, u0)
p = np.full(N, p0)
a1r1 = a1 * rho1_0
a2r2 = (1 - a1) * rho2_0
rho = a1r1 + a2r2
ru = rho * u
e1 = eos1.energy(rho1_0, p)
e2 = eos2.energy(rho2_0, p)
rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2

print(f'Initial state: p0={p0:.3e}, u0={u0}, T0={T0}', flush=True)
print(f'Grid: N={N}, dx={dx:.4f}, L={L}', flush=True)
print(f'rho1_water={float(rho2_0[0]):.2f}, rho1_air={float(rho1_0[0]):.4f}', flush=True)

# ---------------------------------------------------------------------------
# Run 1: Dumbser-Casulli + material CFL (target: NASG PASS)
# ---------------------------------------------------------------------------
print('\n--- Run 1: acoustic_method=dumbser_casulli, material CFL=0.4 ---', flush=True)
t1, a1r1_f1, a2r2_f1, ru_f1, rE_f1, a1_f1 = solve_IMEX(
    ph1, ph2, a1r1.copy(), a2r2.copy(), ru.copy(), rE.copy(), a1.copy(),
    dx=dx, t_end=1.0, cfl=0.4,
    bc_l='periodic', bc_r='periodic',
    use_material_cfl=True,
    acoustic_method='dumbser_casulli',
    dc_outer_max=3, dc_outer_tol=1e-8,
    use_rusanov_diss=False,
    alpha_scheme='tvd', use_mmacm_ex=False, use_apec=False,
    use_compression=False, dissipation='none',
    primitive_recon='tvd',
    use_acid_face=False, acid_interface=False,
    max_steps=200, print_interval=50)

p_f1, u_f1, T_f1, rho1_f1, rho2_f1, *_ = cons_to_prim(
    a1r1_f1, a2r2_f1, ru_f1, rE_f1, a1_f1, ph1, ph2)

err_p1 = float(np.max(np.abs(p_f1 - p0)) / p0)
err_u1 = float(np.max(np.abs(u_f1 - u0)))
pass1 = (err_p1 < 1e-2) and (err_u1 < 1e-2)
print(f'  t={t1:.4f}, err_p={err_p1:.3e}, err_u={err_u1:.3e}  [{" PASS" if pass1 else "FAIL"}]',
      flush=True)

# ---------------------------------------------------------------------------
# Run 2: Dumbser-Casulli + acoustic CFL (sanity: should also converge)
# ---------------------------------------------------------------------------
print('\n--- Run 2: acoustic_method=dumbser_casulli, acoustic CFL=0.2 ---', flush=True)
t2, a1r1_f2, a2r2_f2, ru_f2, rE_f2, a1_f2 = solve_IMEX(
    ph1, ph2, a1r1.copy(), a2r2.copy(), ru.copy(), rE.copy(), a1.copy(),
    dx=dx, t_end=1.0, cfl=0.2,
    bc_l='periodic', bc_r='periodic',
    use_material_cfl=False,
    acoustic_method='dumbser_casulli',
    dc_outer_max=3, dc_outer_tol=1e-8,
    use_rusanov_diss=False,
    alpha_scheme='tvd', use_mmacm_ex=False, use_apec=False,
    use_compression=False, dissipation='none',
    primitive_recon='tvd',
    use_acid_face=False, acid_interface=False,
    max_steps=100000, print_interval=99999)

p_f2, u_f2, T_f2, rho1_f2, rho2_f2, *_ = cons_to_prim(
    a1r1_f2, a2r2_f2, ru_f2, rE_f2, a1_f2, ph1, ph2)

err_p2 = float(np.max(np.abs(p_f2 - p0)) / p0)
err_u2 = float(np.max(np.abs(u_f2 - u0)))
pass2 = (err_p2 < 1e-2) and (err_u2 < 1e-2)
print(f'  t={t2:.4f}, err_p={err_p2:.3e}, err_u={err_u2:.3e}  [{"PASS" if pass2 else "FAIL"}]',
      flush=True)

# ---------------------------------------------------------------------------
# Run 3: Baseline (Peluchon IM1 + iterative, acoustic CFL=0.2) for comparison
# ---------------------------------------------------------------------------
print('\n--- Run 3: acoustic_method=im1 (baseline), acoustic CFL=0.2 ---', flush=True)
t3, a1r1_f3, a2r2_f3, ru_f3, rE_f3, a1_f3 = solve_IMEX(
    ph1, ph2, a1r1.copy(), a2r2.copy(), ru.copy(), rE.copy(), a1.copy(),
    dx=dx, t_end=1.0, cfl=0.2, use_material_cfl=False,
    bc_l='periodic', bc_r='periodic',
    max_steps=100000, print_interval=99999,
    alpha_scheme='tvd', use_mmacm_ex=False, use_apec=False,
    use_compression=False, dissipation='none',
    primitive_recon='tvd', use_acid_face=True, acid_interface=True,
    iterative_im1=True, iterative_im1_max=5, iterative_im1_tol=1e-6)

p_f3, u_f3, T_f3, rho1_f3, rho2_f3, *_ = cons_to_prim(
    a1r1_f3, a2r2_f3, ru_f3, rE_f3, a1_f3, ph1, ph2)

err_p3 = float(np.max(np.abs(p_f3 - p0)) / p0)
err_u3 = float(np.max(np.abs(u_f3 - u0)))
pass3 = (err_p3 < 1e-2) and (err_u3 < 1e-2)
print(f'  t={t3:.4f}, err_p={err_p3:.3e}, err_u={err_u3:.3e}  [{"PASS" if pass3 else "FAIL"}]',
      flush=True)

# ---------------------------------------------------------------------------
# Plot comparison
# ---------------------------------------------------------------------------
# Exact: α shifted by u0*t_end (periodic), p and u should be uniform
shift1 = int(np.round(u0 * float(t1) / dx)) if np.isfinite(t1) else 0
a1_ex1 = np.roll(a1, shift1)

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
fig.suptitle(
    f'02A NASG Abgrall — Dumbser-Casulli vs IM1 comparison\n'
    f'Run1 DC+material(err_p={err_p1:.2e}, {" PASS" if pass1 else "FAIL"})  '
    f'Run2 DC+acoustic(err_p={err_p2:.2e}, {"PASS" if pass2 else "FAIL"})  '
    f'Run3 IM1(err_p={err_p3:.2e}, {"PASS" if pass3 else "FAIL"})',
    fontsize=9)

row_data = [
    (a1_f1, p_f1, u_f1, rho2_f1, 'DC+materialCFL'),
    (a1_f2, p_f2, u_f2, rho2_f2, 'DC+acousticCFL'),
]
for row_i, (a1_plot, p_plot, u_plot, rho2_plot, label) in enumerate(row_data):
    axes[row_i][0].plot(x, a1_plot, 'b-', label=label)
    axes[row_i][0].plot(x, a1_ex1, 'r--', label='exact(shift)', lw=0.8)
    axes[row_i][0].set_title(f'{label} α_air')
    axes[row_i][0].legend(fontsize=7)

    axes[row_i][1].plot(x, p_plot / p0, 'b-', label=label)
    axes[row_i][1].axhline(1.0, color='r', ls='--', lw=0.8, label='exact')
    axes[row_i][1].set_title(f'{label} p/p0  (err={err_p1 if row_i==0 else err_p2:.2e})')
    axes[row_i][1].legend(fontsize=7)

    axes[row_i][2].plot(x, u_plot, 'b-', label=label)
    axes[row_i][2].axhline(u0, color='r', ls='--', lw=0.8, label='exact')
    axes[row_i][2].set_title(f'{label} u  (err={err_u1 if row_i==0 else err_u2:.2e})')
    axes[row_i][2].legend(fontsize=7)

    axes[row_i][3].plot(x, rho2_plot, 'b-', label=label)
    axes[row_i][3].set_title(f'{label} ρ_water')
    axes[row_i][3].legend(fontsize=7)

for ax in axes.flat:
    ax.grid(alpha=0.3)
    ax.set_xlabel('x')
fig.tight_layout()
out_path = f'{OUT}/02A_abgrall_nasg_dc.png'
fig.savefig(out_path, dpi=130)
plt.close(fig)
print(f'\nPlot saved: {out_path}', flush=True)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print('\n' + '='*60, flush=True)
print('SUMMARY — 02A NASG Dumbser-Casulli validation', flush=True)
print('='*60, flush=True)
print(f'  Run1 DC + material CFL=0.4: err_p={err_p1:.3e}, err_u={err_u1:.3e}  '
      f'[{"PASS" if pass1 else "FAIL"}]  t={t1:.4f}', flush=True)
print(f'  Run2 DC + acoustic CFL=0.2: err_p={err_p2:.3e}, err_u={err_u2:.3e}  '
      f'[{"PASS" if pass2 else "FAIL"}]  t={t2:.4f}', flush=True)
print(f'  Run3 IM1 + iterative:        err_p={err_p3:.3e}, err_u={err_u3:.3e}  '
      f'[{"PASS" if pass3 else "FAIL"}]  t={t3:.4f}', flush=True)
print('='*60, flush=True)

# Final PASS/FAIL for this script
overall = pass1
print(f'\nFINAL: {"PASS" if overall else "FAIL"}  '
      f'(primary target: Run1 DC+material CFL err_p<1e-2, err_u<1e-2)', flush=True)
