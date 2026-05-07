"""Validation 01-A (Static) + 02-A (Unified PE Advection Test A/B/C).

Output: results/cat_A_exact/01_static.png, 02A_abgrall_nasg.png,
                         02B_3species.png, 02C_moving_contact.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from solver.He2024.eos_general import IdealEOS, SGEOS, NASGEOS
from solver.He2024.kapila_k import solve_kapila_K, cons_to_prim_K

R = '/home/younglin90/work/claude_code/claudeCFD/results'
OUT = f'{R}/cat_A_exact'
os.makedirs(OUT, exist_ok=True)


def _plot_exact(x, a1, rho1, rho2, u, p, T, title, out, exact=None):
    fig, ax = plt.subplots(1, 5, figsize=(22, 4))
    data = [(a1, r'$\alpha_1$'), (rho1, r'$\rho_1$'), (rho2, r'$\rho_2$'),
            (u, 'u'), (p, 'p')]
    keys = ['a1_exact', 'rho1_exact', 'rho2_exact', 'u_exact', 'p_exact']
    for a, (y, lbl), k in zip(ax, data, keys):
        a.plot(x, y, 'b-', lw=1.4, label='numerical')
        if exact is not None and k in exact:
            a.plot(x, exact[k], 'r--', lw=1.2, label='exact')
            a.legend(fontsize=8)
        a.set_xlabel('x'); a.set_ylabel(lbl); a.grid(alpha=0.3)
    fig.suptitle(f'{title}  (blue=num, red-dashed=exact)')
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f'  saved: {out}', flush=True)


# ---------------------------------------------------------------------------
# 01-A — Static air-water interface (was old 02)
# ---------------------------------------------------------------------------
def run_01():
    print('\n[01-A: Static air-water interface]', flush=True)
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 4.4, 'pinf': 6.0e8, 'kv': 474.2}
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = SGEOS(gamma=4.4, pinf=6.0e8, kv=474.2)
    N, L = 100, 1.0
    dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, T0 = 1.0e5, 293.0
    a1 = np.where(x < 0.5, 1 - 1e-6, 1e-6)
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
    rho2 = eos2.density(np.full(N, p0), np.full(N, T0))
    u = np.zeros(N); p = np.full(N, p0)
    a1r1 = a1 * rho1
    a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2
    ru = rho * u
    e1 = eos1.energy(rho1, p); e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
        dx=dx, t_end=1e-3, cfl=0.4,
        bc_l='transmissive', bc_r='transmissive',
        max_steps=10000, print_interval=999999,
        alpha_scheme='thinc_bvd', use_mmacm_ex=True)
    p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(
        a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)

    exact = {'a1_exact': a1, 'rho1_exact': rho1, 'rho2_exact': rho2,
             'u_exact': np.zeros(N), 'p_exact': np.full(N, p0)}

    err_p = float(np.max(np.abs(p_f - p0)) / p0)
    err_u = float(np.max(np.abs(u_f)))
    print(f'  err_p={err_p:.3e}, err_u={err_u:.3e}, t={t:.3e}')

    status = 'PASS' if (err_p < 1e-3 and err_u < 1.0) else 'FAIL'
    title = f'01-A Static Interface  [{status}]  err_p={err_p:.2e}  err_u={err_u:.2e}'
    _plot_exact(x, a1_f, rho1_f, rho2_f, u_f, p_f, T_f, title,
                f'{OUT}/01_static.png', exact=exact)
    return status


# ---------------------------------------------------------------------------
# 02-A — Test A: Abgrall 2-phase water(NASG) + air(Ideal), u=1 m/s
# ---------------------------------------------------------------------------
def run_02A():
    print('\n[02-A Test A: Abgrall 2-phase Air + NASG-water (phase1=gas convention)]', flush=True)
    # Solver convention (Phase 2-1/2-2): phase 1 = gas, phase 2 = liquid
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}  # Air (Ideal)
    ph2 = {'gamma': 1.187, 'pinf': 7.028e8, 'kv': 3610.0,
           'b': 6.61e-4, 'eta': -1.177788e6}  # Water (NASG)
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = NASGEOS(gamma=1.187, pinf=7.028e8, kv=3610.0,
                   b=6.61e-4, eta=-1.177788e6)
    # N=10: spec 기본 (도메인 1m, dx=0.1m).
    # N=50 에서 acoustic CFL=0.2 는 dt=2.6e-6 → t=1.0 위해 385k steps → max_steps 초과.
    N, L = 10, 1.0
    dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, u0, T0 = 1.0e5, 1.0, 300.0
    a_water = ((x >= 0.4) & (x <= 0.6)).astype(float)
    # a1 = α_air (phase 1 = air). Air outside [0.4, 0.6], water inside.
    a1 = (1 - a_water) * (1 - 1e-6) + a_water * 1e-6
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))  # air
    rho2 = eos2.density(np.full(N, p0), np.full(N, T0))  # water
    u = np.full(N, u0); p = np.full(N, p0)
    a1r1 = a1 * rho1
    a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2
    ru = rho * u
    e1 = eos1.energy(rho1, p); e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2

    # NASG + IMEX 안정 구성 (25차 Round 7 구조적 혁신 → material CFL PASS):
    #   - 5N coupled IMEX Newton-Krylov (α·ρ_k, α, ρu, ρE 동시 implicit)
    #   - Acoustic (∇p, p·u, α·∇·u) 만 implicit, advection (APEC+ACID) explicit
    #   - FD sparse Jacobian + GMRES + ILU preconditioner + Armijo line search
    #   - Material CFL ≥ 0.1 안정 (user-facing 그대로), 100× 가속 (mat CFL=0.4 에서)
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
        dx=dx, t_end=1.0, cfl=0.2, use_material_cfl=True,
        bc_l='periodic', bc_r='periodic',
        max_steps=10000, print_interval=99999,
        acoustic_method='imex_5n')
    p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(
        a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)

    # Exact at t: α shifted by u₀·t (periodic advection)
    t_safe = t if np.isfinite(t) else 0.0
    shift_cells = int(np.round(u0 * t_safe / dx)) if np.isfinite(t) else 0
    a1_ex = np.roll(a1, shift_cells)
    rho1_ex = np.roll(rho1, shift_cells)
    rho2_ex = np.roll(rho2, shift_cells)
    exact = {'a1_exact': a1_ex, 'rho1_exact': rho1_ex, 'rho2_exact': rho2_ex,
             'u_exact': np.full(N, u0), 'p_exact': np.full(N, p0)}

    err_p = float(np.max(np.abs(p_f - p0)) / p0)
    err_u = float(np.max(np.abs(u_f - u0)))
    print(f'  err_p={err_p:.3e}, err_u={err_u:.3e}, t={t:.3e}')

    status = 'PASS' if (err_p < 1e-2 and err_u < 1e-2) else 'FAIL'
    title = f'02-A Test A Abgrall NASG water-air  [{status}]  err_p={err_p:.2e}  err_u={err_u:.2e}'
    _plot_exact(x, a1_f, rho1_f, rho2_f, u_f, p_f, T_f, title,
                f'{OUT}/02A_abgrall_nasg.png', exact=exact)
    return status


# ---------------------------------------------------------------------------
# 02-A — Test B: 3-species air/He/SF6 u=100
# ---------------------------------------------------------------------------
def run_02B():
    print('\n[02-A Test B: 3-species air/He/SF6, u=100 m/s]', flush=True)
    eos_list = [IdealEOS(gamma=1.4, kv=717.5),
                IdealEOS(gamma=1.667, kv=3116.0),
                IdealEOS(gamma=1.094, kv=665.0)]
    ph_list = [{'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5},
               {'gamma': 1.667, 'pinf': 0.0, 'kv': 3116.0},
               {'gamma': 1.094, 'pinf': 0.0, 'kv': 665.0}]
    N, L = 100, 1.0
    dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, u0, T0 = 1.0e5, 100.0, 300.0

    a = np.zeros((3, N))
    a[0] = ((x < 0.25) | (x >= 0.75)).astype(float)  # air
    a[1] = ((x >= 0.25) & (x < 0.50)).astype(float)  # helium
    a[2] = ((x >= 0.50) & (x < 0.75)).astype(float)  # SF6
    eps = 1e-6
    a = np.clip(a, eps, 1.0 - eps)
    # Re-normalize so Σα=1
    a = a / np.sum(a, axis=0, keepdims=True)

    rho = np.array([e.density(np.full(N, p0), np.full(N, T0)) for e in eos_list])
    ar = a * rho  # shape (3, N)
    u = np.full(N, u0); p = np.full(N, p0)
    rho_mix = np.sum(ar, axis=0)
    ru = rho_mix * u
    e = np.array([eos_list[k].energy(rho[k], p) for k in range(3)])
    rE = np.sum(ar * e, axis=0) + 0.5 * rho_mix * u**2

    res = solve_kapila_K(
        eos_list, list(ar), ru, rE, list(a),
        dx=dx, t_end=1e-2, cfl=0.4,
        bc_l='periodic', bc_r='periodic',
        max_steps=20000, print_interval=999999)
    t, ar_f_list, ru_f, rE_f, a_f_list = res
    ar_f = np.array(ar_f_list); a_f = np.array(a_f_list)

    prim = cons_to_prim_K(list(ar_f), ru_f, rE_f, list(a_f), eos_list)
    # prim returns (p, u, T, *rhos)
    p_f, u_f, T_f = prim[0], prim[1], prim[2]

    err_p = float(np.max(np.abs(p_f - p0)) / p0)
    err_u = float(np.max(np.abs(u_f - u0)))
    print(f'  err_p={err_p:.3e}, err_u={err_u:.3e}, t={t:.3e}')

    # a_f may have K-1 rows (solver drops last). Reconstruct full K by 1-sum.
    if a_f.shape[0] == 2:
        a_last = np.maximum(1.0 - a_f[0] - a_f[1], 1e-12)
        a_f_full = np.vstack([a_f, a_last[None, :]])
    else:
        a_f_full = a_f

    # Plot mass fractions (α) + p + u, overlay exact (initial, because periodic full rev)
    fig, ax = plt.subplots(1, 5, figsize=(22, 4))
    labels = ['air', 'He', 'SF6']
    colors = ['b', 'g', 'm']
    for k in range(3):
        ax[0].plot(x, a_f_full[k], colors[k]+'-', lw=1.3, label=f'α_{labels[k]}')
        ax[0].plot(x, a[k], colors[k]+'--', lw=0.8, alpha=0.6)
    ax[0].set_xlabel('x'); ax[0].set_ylabel('α'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    for k, lbl in enumerate(labels):
        ax[1].plot(x, ar_f[k], colors[k]+'-', lw=1.2, label=f'(αρ)_{lbl}')
        ax[1].plot(x, ar[k], colors[k]+'--', lw=0.8, alpha=0.5)
    ax[1].set_xlabel('x'); ax[1].set_ylabel(r'$\alpha_k \rho_k$'); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)

    ax[2].plot(x, u_f, 'b-', label='u num')
    ax[2].axhline(u0, color='r', ls='--', label='u exact')
    ax[2].set_xlabel('x'); ax[2].set_ylabel('u'); ax[2].legend(); ax[2].grid(alpha=0.3)

    ax[3].plot(x, p_f, 'b-', label='p num')
    ax[3].axhline(p0, color='r', ls='--', label='p exact')
    ax[3].set_xlabel('x'); ax[3].set_ylabel('p'); ax[3].legend(); ax[3].grid(alpha=0.3)

    ax[4].plot(x, T_f, 'b-', label='T num')
    ax[4].axhline(T0, color='r', ls='--', label='T exact')
    ax[4].set_xlabel('x'); ax[4].set_ylabel('T'); ax[4].legend(); ax[4].grid(alpha=0.3)

    status = 'PASS' if (err_p < 1e-6 and err_u < 1e-2) else 'FAIL'
    fig.suptitle(f'02-A Test B 3-species (air/He/SF6) u=100  [{status}]  err_p={err_p:.2e}  err_u={err_u:.2e}')
    fig.tight_layout()
    fig.savefig(f'{OUT}/02B_3species.png', dpi=130)
    plt.close(fig)
    print(f'  saved: {OUT}/02B_3species.png')
    return status


# ---------------------------------------------------------------------------
# 02-A — Test C: Moving contact u=100, p=1e9 (Kraposhin 2022)
# ---------------------------------------------------------------------------
def run_02C():
    print('\n[02-A Test C: Moving contact u=100 m/s, p=1e9 Pa]', flush=True)
    # Two ideal gases, same γ, just different "species" marker via α
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = IdealEOS(gamma=1.4, kv=717.5)
    N, L = 200, 1.0
    dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, u0, T0 = 1.0e9, 100.0, 300.0
    a1 = np.where(x < 0.5, 1 - 1e-6, 1e-6)
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
    rho2 = eos2.density(np.full(N, p0), np.full(N, T0))
    u = np.full(N, u0); p = np.full(N, p0)
    a1r1 = a1 * rho1
    a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2
    ru = rho * u
    e1 = eos1.energy(rho1, p); e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
        dx=dx, t_end=0.01, cfl=0.4,
        bc_l='periodic', bc_r='periodic',
        max_steps=20000, print_interval=999999,
        alpha_scheme='tvd', use_mmacm_ex=False)  # TVD + MMACM off (optimal for smooth)
    p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(
        a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)

    # Exact: u0*t = 1 m = full rev → α back to initial
    exact = {'a1_exact': a1, 'rho1_exact': rho1, 'rho2_exact': rho2,
             'u_exact': np.full(N, u0), 'p_exact': np.full(N, p0)}

    err_p = float(np.max(np.abs(p_f - p0)) / p0)
    err_u = float(np.max(np.abs(u_f - u0)))
    err_T = float(np.max(np.abs(T_f - T0)) / T0)
    print(f'  err_p={err_p:.3e}, err_u={err_u:.3e}, err_T={err_T:.3e}, t={t:.3e}')

    status = 'PASS' if (err_p < 1e-10 and err_u < 1e-8) else 'FAIL'
    title = (f'02-A Test C Moving Contact u=100, p=1e9  [{status}]  '
             f'err_p={err_p:.2e}  err_u={err_u:.2e}')
    _plot_exact(x, a1_f, rho1_f, rho2_f, u_f, p_f, T_f, title,
                f'{OUT}/02C_moving_contact.png', exact=exact)
    return status


if __name__ == '__main__':
    results = {}
    results['01'] = run_01()
    results['02A'] = run_02A()
    results['02B'] = run_02B()
    results['02C'] = run_02C()
    print('\n============================================================')
    print('Summary:')
    for k, v in results.items():
        print(f'  {k}: {v}')
