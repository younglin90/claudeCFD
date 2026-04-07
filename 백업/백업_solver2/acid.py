"""
acid.py
=======
ACID: Acoustically-Conservative Interface Discretisation (Denner 2018)
for two-component SRK (real-gas) mixtures.

All EOS functions are imported from apec.py.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

from solver.apec import (
    Ru, M, Tc, pc, om, a_sp, b_sp, fom, Cv0,
    _mix, srk_p, srk_rhoe, srk_c2, epsilon_v,
    T_from_rhoe, _dpdT,
    muscl_lr, prim,
    initial_condition, pe_err, energy_err,
    T_from_p,
)

# ═══════════════════════════════════════════════════════════════════
# ACID SECTION
# Acoustically-Conservative Interface Discretisation (Denner 2018)
# Functions prefixed acid_* to avoid name conflicts with APEC section.
# ═══════════════════════════════════════════════════════════════════
def _fc_apec_fluxes(r1, r2, u, rhoe, p, T, lam_cell, scheme, eps_pair):
    """FC-NPE or APEC fluxes (symmetric, conservative).

    FC  : standard MUSCL-LLF energy flux
    APEC: Terashima 2025 Appendix A (Eq. A.4) — SRK version
          Uses cell-centered fluxes + PE-consistent correction terms.
          Matches the validated formula in apec_1d.py.
    """
    rho  = r1 + r2
    rhoE = rhoe + 0.5*rho*u**2

    r1L, r1R = muscl_lr(r1)
    r2L, r2R = muscl_lr(r2)
    uL,  uR  = muscl_lr(u)
    pL,  pR  = muscl_lr(p)

    rhoL  = r1L + r2L
    rhoR  = r1R + r2R
    lam   = np.maximum(lam_cell, np.roll(lam_cell, -1))

    # Conservative mass & momentum
    F1 = 0.5*(r1L*uL + r1R*uR) - 0.5*lam*(r1R - r1L)
    F2 = 0.5*(r2L*uL + r2R*uR) - 0.5*lam*(r2R - r2L)
    FU = 0.5*(rhoL*uL**2 + pL + rhoR*uR**2 + pR) - 0.5*lam*(rhoR*uR - rhoL*uL)

    if scheme == 'APEC':
        # ── SRK Appendix A (Eq. A.4) — identical to apec_1d.py ──────
        # Cell-centred fluxes (no upwinding)
        F1_cell  = r1 * u
        F2_cell  = r2 * u
        FU_cell  = rho * u**2 + p
        FE_cell  = (rhoE + p) * u

        eps0, eps1 = eps_pair
        eps0_m1  = np.roll(eps0, -1)
        eps1_m1  = np.roll(eps1, -1)
        u_m1     = np.roll(u,    -1)
        FE_m1    = np.roll(FE_cell, -1)

        # Correction from left cell (m) and right cell (m+1) at face m+1/2
        c0m  = eps0    - 0.5*u**2
        c1m  = eps1    - 0.5*u**2
        c0m1 = eps0_m1 - 0.5*u_m1**2
        c1m1 = eps1_m1 - 0.5*u_m1**2

        tm  = (c0m *(F1 - F1_cell)             + c1m *(F2 - F2_cell)
               + u   *(FU - FU_cell))
        tm1 = (c0m1*(np.roll(F1_cell,-1) - F1) + c1m1*(np.roll(F2_cell,-1) - F2)
               + u_m1*(np.roll(FU_cell,-1) - FU))

        FE = 0.5*(FE_cell + FE_m1) + 0.5*tm - 0.5*tm1
    else:
        # FC-NPE: standard MUSCL-LLF energy flux
        # Each face state uses the temperature of its parent cell:
        #   Left state at face m+1/2  → T[m]      (current cell)
        #   Right state at face m+1/2 → T[m+1]    (adjacent cell, np.roll(T,-1))
        # This is the correct baseline that produces the expected PE error.
        rhoEL = srk_rhoe(r1L, r2L, T)            + 0.5*rhoL*uL**2
        rhoER = srk_rhoe(r1R, r2R, np.roll(T,-1)) + 0.5*rhoR*uR**2
        FE = 0.5*((rhoEL+pL)*uL + (rhoER+pR)*uR) - 0.5*lam*(rhoER - rhoEL)

    return F1, F2, FU, FE


# ─────────────────────────────────────────────────────────────
# 5.  ACID energy flux  (Denner 2018 §5, adapted to density-based)
# ─────────────────────────────────────────────────────────────
def _acid_fluxes(r1, r2, u, rhoe, p, T, lam_cell, eps_pair):
    """ACID flux set — NON-conservative energy using cell-m ε for both faces.

    Adaptation of Denner 2018 §5 to a density-based MUSCL-LLF solver.

    Key idea
    ────────
    APEC (Terashima App. A) at face m+1/2:
        FE = 0.5*(FE_m + FE_{m+1}) + 0.5*tm - 0.5*tm1
      where tm uses ε[m] and tm1 uses ε[m+1].

    ACID: cell m computes its energy RHS using ε[m] for BOTH adjacent faces:
      Face m+1/2 (cell m is LEFT):
        FE_left[m] = 0.5*(FE_m+FE_{m+1}) + 0.5*tm - 0.5*tm1_acid
        where tm1_acid uses ε[m] instead of ε[m+1] (ACID modification)
      Face m-1/2 (cell m is RIGHT):
        FE_right[m] = 0.5*(FE_{m-1}+FE_m) + 0.5*tm_mm_acid - 0.5*tm1_mm
        where tm_mm_acid uses ε[m] instead of ε[m-1]

    Cell m energy RHS (NON-conservative):
        dE[m] = -(FE_left[m] - FE_right[m]) / dx

    Away from interface (ε[m] ≈ ε[m±1]): ACID ≈ APEC.
    At interface: cell m sees consistent thermodynamics at BOTH adjacent faces.

    Returns
    ───────
    F1, F2, FU : conservative fluxes at face m+1/2
    FE_left    : ACID energy flux at face m+1/2 (cell m's view)
    FE_right   : ACID energy flux at face m-1/2 (cell m's view)
    """
    rho  = r1 + r2
    rhoE = rhoe + 0.5*rho*u**2

    r1L, r1R = muscl_lr(r1)
    r2L, r2R = muscl_lr(r2)
    uL,  uR  = muscl_lr(u)
    pL,  pR  = muscl_lr(p)
    rhoL = r1L + r2L
    rhoR = r1R + r2R
    lam  = np.maximum(lam_cell, np.roll(lam_cell, -1))

    # ── Conservative mass & momentum ─────────────────────────────
    F1 = 0.5*(r1L*uL + r1R*uR) - 0.5*lam*(r1R - r1L)
    F2 = 0.5*(r2L*uL + r2R*uR) - 0.5*lam*(r2R - r2L)
    FU = 0.5*(rhoL*uL**2+pL + rhoR*uR**2+pR) - 0.5*lam*(rhoR*uR - rhoL*uL)

    # ── Cell-centred fluxes ──────────────────────────────────────
    F1_cell = r1 * u
    F2_cell = r2 * u
    FU_cell = rho * u**2 + p
    FE_cell = (rhoE + p) * u
    FE_m1   = np.roll(FE_cell, -1)
    FE_mm1  = np.roll(FE_cell,  1)

    eps0, eps1 = eps_pair
    u_m1  = np.roll(u,  -1)   # u[m+1]
    u_mm1 = np.roll(u,   1)   # u[m-1]

    # ── ACID face m+1/2: use ε[m] for both tm and tm1 ────────────
    # tm (left-cell correction, cell m is left): same as APEC
    c0m = eps0 - 0.5*u**2
    c1m = eps1 - 0.5*u**2
    tm = c0m*(F1 - F1_cell) + c1m*(F2 - F2_cell) + u*(FU - FU_cell)

    # tm1_acid (right-cell correction, use ε[m] but velocity u[m+1])
    c0m_right = eps0 - 0.5*u_m1**2   # ε[m] with u[m+1] for kinetic consistency
    c1m_right = eps1 - 0.5*u_m1**2
    tm1_acid = (c0m_right*(np.roll(F1_cell,-1) - F1)
                + c1m_right*(np.roll(F2_cell,-1) - F2)
                + u_m1*(np.roll(FU_cell,-1) - FU))

    FE_left = 0.5*(FE_cell + FE_m1) + 0.5*tm - 0.5*tm1_acid

    # ── ACID face m-1/2: use ε[m] for both tm and tm1 ────────────
    F1_mm = np.roll(F1, 1)     # face flux at m-1/2
    F2_mm = np.roll(F2, 1)
    FU_mm = np.roll(FU, 1)
    F1_cell_mm1 = np.roll(F1_cell, 1)   # F1_cell[m-1]
    F2_cell_mm1 = np.roll(F2_cell, 1)
    FU_cell_mm1 = np.roll(FU_cell, 1)

    # tm_mm_acid: left-cell (m-1) correction at face m-1/2 using ε[m]
    c0m_left_mm = eps0 - 0.5*u_mm1**2   # ε[m] with u[m-1] for kinetic consistency
    c1m_left_mm = eps1 - 0.5*u_mm1**2
    tm_mm_acid = (c0m_left_mm*(F1_mm - F1_cell_mm1)
                  + c1m_left_mm*(F2_mm - F2_cell_mm1)
                  + u_mm1*(FU_mm - FU_cell_mm1))

    # tm1_mm (right-cell correction, cell m is right): use ε[m] (same as ACID)
    tm1_mm = c0m*(F1_cell - F1_mm) + c1m*(F2_cell - F2_mm) + u*(FU_cell - FU_mm)

    FE_right = 0.5*(FE_mm1 + FE_cell) + 0.5*tm_mm_acid - 0.5*tm1_mm

    return F1, F2, FU, FE_left, FE_right


# ─────────────────────────────────────────────────────────────
# 6.  RHS
# ─────────────────────────────────────────────────────────────
def acid_rhs(U, scheme, dx, T_prev):
    r1, r2, rhoU, rhoE = U
    u, rhoe, T, p = prim(r1, r2, rhoU, rhoE, T_prev)
    lam_c = np.abs(u) + np.sqrt(srk_c2(r1, r2, T))

    eps_pair = (epsilon_v(r1, r2, T, 0), epsilon_v(r1, r2, T, 1)) \
               if scheme in ('APEC', 'ACID') else None

    if scheme == 'ACID':
        # Non-conservative energy: cell m uses ε[m] at BOTH adjacent faces
        F1, F2, FU, FE_left, FE_right = _acid_fluxes(
            r1, r2, u, rhoe, p, T, lam_c, eps_pair)
        d1 = -(F1      - np.roll(F1, 1)) / dx  # conservative
        d2 = -(F2      - np.roll(F2, 1)) / dx  # conservative
        dU = -(FU      - np.roll(FU, 1)) / dx  # conservative
        dE = -(FE_left - FE_right       ) / dx  # non-conservative
    else:
        F1, F2, FU, FE = _fc_apec_fluxes(r1, r2, u, rhoe, p, T,
                                          lam_c, scheme, eps_pair)
        d1 = -(F1 - np.roll(F1, 1)) / dx
        d2 = -(F2 - np.roll(F2, 1)) / dx
        dU = -(FU - np.roll(FU, 1)) / dx
        dE = -(FE - np.roll(FE, 1)) / dx

    return [d1, d2, dU, dE], T, p


# ─────────────────────────────────────────────────────────────
# 7.  SSP-RK3 time integration
# ─────────────────────────────────────────────────────────────
def _clip(U):
    return [np.maximum(U[0], 0.0), np.maximum(U[1], 0.0), U[2], U[3]]


def acid_rkstep(U, scheme, dx, dt, T_prev):
    k1, T1, p1 = acid_rhs(U, scheme, dx, T_prev)
    U1 = _clip([U[q] + dt*k1[q] for q in range(4)])
    k2, T2, p2 = acid_rhs(U1, scheme, dx, T1)
    U2 = _clip([0.75*U[q] + 0.25*(U1[q] + dt*k2[q]) for q in range(4)])
    k3, T3, p3 = acid_rhs(U2, scheme, dx, T2)
    return _clip([(1/3)*U[q] + (2/3)*(U2[q] + dt*k3[q]) for q in range(4)]), T3, p3


# ─────────────────────────────────────────────────────────────
# 8.  Diagnostics
# ─────────────────────────────────────────────────────────────
def pe_err(p, p0=5e6):
    return float(np.max(np.abs(p - p0)) / p0)


def energy_err(rhoE, rhoE0):
    return float(abs(np.sum(rhoE) - np.sum(rhoE0)) / (abs(np.sum(rhoE0)) + 1e-30))


# ─────────────────────────────────────────────────────────────
# 9.  Main runner
# ─────────────────────────────────────────────────────────────
def acid_run(scheme, N=101, t_end=0.07, CFL=0.4, p_inf=5e6, k=15.0,
        verbose=True):
    dx = 1.0 / N
    x  = np.linspace(dx/2, 1 - dx/2, N)
    if verbose:
        print(f"\n{'='*55}")
        print(f"Scheme: {scheme}  N={N}  t_end={t_end:.4f}  CFL={CFL}  k={k}")

    r1, r2, u, rhoE, T, p = initial_condition(x, p_inf, k=k)
    U    = [r1.copy(), r2.copy(), (r1+r2)*u, rhoE.copy()]
    rhoE0 = rhoE.copy()

    t_hist, pe_hist, en_hist = [0.0], [pe_err(p, p_inf)], [0.0]
    t, step, diverged = 0.0, 0, False

    while t < t_end - 1e-14:
        lam = float(np.max(
            np.abs(U[2] / np.maximum(U[0]+U[1], 1e-30))
            + np.sqrt(srk_c2(U[0], U[1], T))
        ))
        dt = min(CFL*dx/(lam + 1e-10), t_end - t)

        try:
            U, T, p = acid_rkstep(U, scheme, dx, dt, T)
        except Exception as e:
            if verbose:
                print(f"  Exception at t={t:.5f}: {e}")
            diverged = True
            break

        t += dt; step += 1
        pe_ = pe_err(p, p_inf)
        en_ = energy_err(U[3], rhoE0)
        t_hist.append(t); pe_hist.append(pe_); en_hist.append(en_)

        if not np.isfinite(pe_) or pe_ > 5.0:
            if verbose:
                print(f"  Diverged (PE={pe_:.2e}) at t={t:.5f}")
            diverged = True
            break

    if verbose:
        status = 'Completed' if not diverged else 'Diverged'
        print(f"  --> {status} at t={t:.5f}  ({step} steps)")

    return (x, U, T, p,
            np.array(t_hist), np.array(pe_hist), np.array(en_hist),
            diverged)


# ─────────────────────────────────────────────────────────────
# 10.  Comparison plots & validation
# ─────────────────────────────────────────────────────────────
def compare_pe_time(N=101, t_end=0.06, CFL=0.3, k=15.0):
    """PE vs time for FC / APEC / ACID."""
    print("\n[Test 1] PE vs time  N=%d  CFL=%.2f  k=%.0f" % (N, CFL, k))
    results = {}
    for sc in ('FC', 'APEC', 'ACID'):
        x, U, T, p, th, ph, eh, div = acid_run(sc, N=N, t_end=t_end,
                                           CFL=CFL, k=k)
        results[sc] = (th, ph, div)
        print(f"  {sc:4s}  PE(t=1step)={ph[1]:.3e}  "
              f"PE(final)={ph[-1]:.3e}  diverged={div}")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'FC': 'tab:blue', 'APEC': 'tab:orange', 'ACID': 'tab:green'}
    for sc, (th, ph, div) in results.items():
        ax.semilogy(th*1e3, ph, label=sc, color=colors[sc])
    ax.set_xlabel('t  [ms]')
    ax.set_ylabel('max |Δp/p₀|')
    ax.set_title(f'Pressure-equilibrium error  N={N}  k={k}  CFL={CFL}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f'acid_pe_time_N{N}.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")
    return results


def compare_resolution(t_end=1e-4, CFL=0.3, k=15.0):
    """PE at t≈1 step for different N (FC / APEC / ACID)."""
    print("\n[Test 2] Resolution study  k=%.0f" % k)
    Ns = [51, 101, 201, 501]
    header = f"{'N':>6}  {'FC PE':>12}  {'APEC PE':>12}  {'ACID PE':>12}"
    print("  " + header)
    print("  " + "-"*len(header))
    rows = []
    for N in Ns:
        dx  = 1.0 / N
        x   = np.linspace(dx/2, 1 - dx/2, N)
        r1, r2, u, rhoE, T, p0 = initial_condition(x, k=k)
        U = [r1.copy(), r2.copy(), (r1+r2)*u, rhoE.copy()]
        row = [N]
        for sc in ('FC', 'APEC', 'ACID'):
            lam = float(np.max(np.abs(u) + np.sqrt(srk_c2(r1, r2, T))))
            dt  = CFL * dx / (lam + 1e-10)
            U1, T1, p1 = acid_rkstep(U, sc, dx, dt, T)
            row.append(pe_err(p1))
        rows.append(row)
        fc_, ap_, ac_ = row[1], row[2], row[3]
        print(f"  {N:>6}  {fc_:>12.3e}  {ap_:>12.3e}  {ac_:>12.3e}")
    return rows


def compare_divergence(N=101, t_end=0.06, CFL=0.3, k=15.0):
    """Divergence time comparison."""
    print("\n[Test 3] Divergence time  N=%d  CFL=%.2f  k=%.0f" % (N, CFL, k))
    for sc in ('FC', 'APEC', 'ACID'):
        x, U, T, p, th, ph, _, div = acid_run(sc, N=N, t_end=t_end,
                                          CFL=CFL, k=k)
        t_div = th[-1]
        print(f"  {sc:4s}  t_diverge={t_div*1e3:.2f} ms  diverged={div}")


def plot_profiles(N=101, CFL=0.3, k=15.0, t_snap=5e-3):
    """Snapshot profiles: density, pressure, velocity."""
    print(f"\n[Test 4] Profiles at t={t_snap*1e3:.1f} ms  N={N}")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = {'FC': 'tab:blue', 'APEC': 'tab:orange', 'ACID': 'tab:green'}

    for sc in ('FC', 'APEC', 'ACID'):
        x, U, T, p, *_ = acid_run(sc, N=N, t_end=t_snap, CFL=CFL, k=k)
        r1, r2, rhoU, rhoE = U
        rho = r1 + r2
        u   = rhoU / np.maximum(rho, 1e-30)
        axes[0].plot(x, rho, label=sc, color=colors[sc])
        axes[1].plot(x, p*1e-6, label=sc, color=colors[sc])
        axes[2].plot(x, u, label=sc, color=colors[sc])

    titles = ['Density  [kg/m³]', 'Pressure  [MPa]', 'Velocity  [m/s]']
    for ax, ttl in zip(axes, titles):
        ax.set_xlabel('x')
        ax.set_title(ttl)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f't = {t_snap*1e3:.1f} ms,  N={N},  k={k}', fontsize=11)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f'acid_profiles_N{N}.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────
# 11.  Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # ── Quick single-step PE comparison across resolutions ────
    compare_resolution(k=15.0)

    # ── PE vs time  (N=101, up to divergence) ────────────────
    compare_pe_time(N=101, t_end=0.055, CFL=0.3, k=15.0)

    # ── Divergence time ───────────────────────────────────────
    compare_divergence(N=101, t_end=0.055, CFL=0.3, k=15.0)

    # ── Snapshot profiles ─────────────────────────────────────
    plot_profiles(N=101, CFL=0.3, k=15.0, t_snap=3e-3)

    print("\nDone.")
