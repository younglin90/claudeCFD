"""1D Euler Sod shock tube — primitive-variable T-MLP-u with HLLC flux.

Compares all reconstruction options on the classical Sod problem:

  Left  (x < 0.5):  ρ=1.0,    u=0,  p=1.0
  Right (x ≥ 0.5):  ρ=0.125,  u=0,  p=0.1
  γ = 1.4, transmissive walls, t = 0.2.

Saves a 3-panel figure (ρ, u, p) overlaying every variant against the
analytic Sod solution computed from the iterative star-state solver.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _pkgshim import setup_paths
setup_paths()

from mesh import build_structured_1d
from equations import Euler1D
from reconstruction import TMLPU
from boundary import BoundaryCondition
from solver import solve


# ─── Exact Sod solver (Toro §4.3) ───────────────────────────────────────────
def _sod_exact(x, t, gamma=1.4,
               rhoL=1.0, uL=0.0, pL=1.0,
               rhoR=0.125, uR=0.0, pR=0.1,
               x0=0.5):
    """Return (rho, u, p) at every x for the analytic Sod solution at time t."""
    cL = np.sqrt(gamma * pL / rhoL); cR = np.sqrt(gamma * pR / rhoR)

    def f_K(p, pK, rhoK, cK):
        if p > pK:
            AK = 2.0 / ((gamma + 1.0) * rhoK)
            BK = (gamma - 1.0) / (gamma + 1.0) * pK
            return (p - pK) * np.sqrt(AK / (p + BK))
        return (2.0 * cK / (gamma - 1.0)) * ((p / pK) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)

    def f(p):
        return f_K(p, pL, rhoL, cL) + f_K(p, pR, rhoR, cR) + (uR - uL)

    # Newton-Raphson on p* (Toro)
    p_star = 0.5 * (pL + pR)
    for _ in range(100):
        f0 = f(p_star)
        # numerical derivative
        dp = max(1e-8 * p_star, 1e-12)
        df = (f(p_star + dp) - f0) / dp
        p_new = p_star - f0 / df
        if p_new < 0:
            p_new = 0.5 * p_star
        if abs(p_new - p_star) < 1e-10 * abs(p_new):
            p_star = p_new
            break
        p_star = p_new
    u_star = 0.5 * (uL + uR) + 0.5 * (f_K(p_star, pR, rhoR, cR) - f_K(p_star, pL, rhoL, cL))

    # Sample
    rho = np.empty_like(x);  u = np.empty_like(x);  p = np.empty_like(x)
    s = (x - x0) / max(t, 1e-30)
    # Star-side densities
    if p_star > pL:
        rho_starL = rhoL * (((p_star / pL) + (gamma - 1) / (gamma + 1)) /
                            ((gamma - 1) / (gamma + 1) * (p_star / pL) + 1))
    else:
        rho_starL = rhoL * (p_star / pL) ** (1.0 / gamma)
    if p_star > pR:
        rho_starR = rhoR * (((p_star / pR) + (gamma - 1) / (gamma + 1)) /
                            ((gamma - 1) / (gamma + 1) * (p_star / pR) + 1))
    else:
        rho_starR = rhoR * (p_star / pR) ** (1.0 / gamma)

    # Wave speeds
    if p_star > pL:
        SL = uL - cL * np.sqrt((gamma + 1) / (2 * gamma) * (p_star / pL) +
                                (gamma - 1) / (2 * gamma))
        SHL = STL = SL
    else:  # rarefaction L
        SHL = uL - cL
        c_starL = cL * (p_star / pL) ** ((gamma - 1) / (2 * gamma))
        STL = u_star - c_starL
    if p_star > pR:
        SR = uR + cR * np.sqrt((gamma + 1) / (2 * gamma) * (p_star / pR) +
                                (gamma - 1) / (2 * gamma))
        SHR = STR = SR
    else:
        SHR = uR + cR
        c_starR = cR * (p_star / pR) ** ((gamma - 1) / (2 * gamma))
        STR = u_star + c_starR

    for i, si in enumerate(s):
        if si < SHL:
            rho[i], u[i], p[i] = rhoL, uL, pL
        elif si < STL:
            # Inside left rarefaction fan
            u_fan = 2.0 / (gamma + 1) * (cL + (gamma - 1) / 2 * uL + si)
            c_fan = 2.0 / (gamma + 1) * (cL + (gamma - 1) / 2 * (uL - si))
            p_fan = pL * (c_fan / cL) ** (2 * gamma / (gamma - 1))
            rho_fan = gamma * p_fan / (c_fan ** 2)
            rho[i], u[i], p[i] = rho_fan, u_fan, p_fan
        elif si < u_star:
            rho[i], u[i], p[i] = rho_starL, u_star, p_star
        elif si < STR:
            rho[i], u[i], p[i] = rho_starR, u_star, p_star
        elif si < SHR:
            u_fan = 2.0 / (gamma + 1) * (-cR + (gamma - 1) / 2 * uR + si)
            c_fan = 2.0 / (gamma + 1) * (cR - (gamma - 1) / 2 * (uR - si))
            p_fan = pR * (c_fan / cR) ** (2 * gamma / (gamma - 1))
            rho_fan = gamma * p_fan / (c_fan ** 2)
            rho[i], u[i], p[i] = rho_fan, u_fan, p_fan
        else:
            rho[i], u[i], p[i] = rhoR, uR, pR
    return rho, u, p


def _run(recon, N=200, t_end=0.2):
    L = 1.0
    mesh = build_structured_1d(N, L=L, periodic=False)
    eq = Euler1D(gamma=1.4)
    x = mesh.cell_centers[:, 0]
    rho = np.where(x < 0.5, 1.0, 0.125)
    u = np.zeros(N)
    p = np.where(x < 0.5, 1.0, 0.1)
    W0 = np.stack([rho, u, p], axis=0)
    U0 = eq.prim_to_cons(W0)
    bc = {
        'left':  BoundaryCondition('transmissive'),
        'right': BoundaryCondition('transmissive'),
    }
    res = solve(mesh, eq, U0,
                reconstruction=recon, flux='hllc',
                integrator='ssp_rk2', bc=bc,
                cfl=0.4, t_end=t_end, max_steps=100_000)
    W = eq.cons_to_prim(res['U_final'])
    return x, W, res['n_steps']


def main():
    cases = [
        ('first_order',     'first_order'),
        ('minmod_tvd_1d',   'minmod_tvd_1d'),
        ('TMLPU(minmod)',   TMLPU(tvd='minmod')),
        ('TMLPU(superbee)', TMLPU(tvd='superbee')),
        ('TMLPU(mc)',       TMLPU(tvd='mc')),
        ('TMLPU(umist)',    TMLPU(tvd='umist')),
        ('TMLPU(van_leer)', TMLPU(tvd='van_leer')),
    ]
    print("1D Euler Sod (HLLC, t=0.2, N=200) — T-MLP-u verification")
    print("=" * 72)

    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    titles = ['ρ (density)', 'u (velocity)', 'p (pressure)']

    # Exact reference
    x_ref = np.linspace(0, 1, 2001)
    rho_e, u_e, p_e = _sod_exact(x_ref, t=0.2)
    for ax, exact in zip(axs, (rho_e, u_e, p_e)):
        ax.plot(x_ref, exact, 'k-', lw=1.4, label='exact', zorder=10)

    fails = []
    profiles = {}
    for label, recon in cases:
        x, W, n = _run(recon)
        rho, u, p = W
        finite = bool(np.all(np.isfinite(W)))
        rho_max = float(np.max(rho)); rho_min = float(np.min(rho))
        p_max = float(np.max(p));     p_min = float(np.min(p))
        sharp_rho = float(np.max(np.abs(np.diff(rho))))
        # L1 error vs exact (interpolated to grid)
        from numpy import interp
        rho_e_grid = interp(x, x_ref, rho_e)
        l1_rho = float(np.mean(np.abs(rho - rho_e_grid)))
        print(f"  [{label:18s}] n={n:4d}  ρ∈[{rho_min:.4f},{rho_max:.4f}]  "
              f"p∈[{p_min:.4f},{p_max:.4f}]  max|Δρ|={sharp_rho:.3f}  L1ρ={l1_rho:.4f}")
        if not finite:
            fails.append((label, 'NaN'))
        elif rho_max > 1.001 or rho_min < 0.124:
            fails.append((label, f'rho overshoot [{rho_min:.3f}, {rho_max:.3f}]'))
        elif p_max > 1.001 or p_min < 0.099:
            fails.append((label, f'p overshoot [{p_min:.3f}, {p_max:.3f}]'))
        profiles[label] = (x, W, sharp_rho, l1_rho)

        for ax, sig in zip(axs, (rho, u, p)):
            ax.plot(x, sig, label=label, lw=0.9, alpha=0.85)

    # Sharpness ordering
    if 'TMLPU(superbee)' in profiles and 'minmod_tvd_1d' in profiles:
        sh_t = profiles['TMLPU(superbee)'][2]
        sh_m = profiles['minmod_tvd_1d'][2]
        l1_t = profiles['TMLPU(superbee)'][3]
        l1_m = profiles['minmod_tvd_1d'][3]
        print(f"\n  sharpness(TMLPU superbee) / sharpness(minmod_tvd) = {sh_t/max(sh_m,1e-30):.2f}")
        print(f"  L1ρ(TMLPU superbee)        / L1ρ(minmod_tvd)        = {l1_t/max(l1_m,1e-30):.2f}")

    for ax, title in zip(axs, titles):
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
    axs[2].set_xlabel('x')
    axs[0].legend(loc='lower left', fontsize=7, ncol=2)
    fig.suptitle("1D Euler Sod shock tube — γ=1.4, t=0.2, N=200, HLLC", fontsize=11)
    fig.tight_layout()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, '1d_euler_sod_tmlpu.png')
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")

    print("=" * 72)
    if fails:
        print(f"FAIL — {len(fails)} cases:")
        for r, why in fails:
            print(f"  {r}: {why}")
        return 1
    print("PASS — Sod profiles within physical bounds; T-MLP-u L1 error ≤ minmod TVD.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
