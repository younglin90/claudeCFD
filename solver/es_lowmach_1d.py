"""
es_lowmach_1d.py
================
Entropy-Stable (ES) schemes with flux preconditioning for low-Mach 1D flows.

Schemes:
  EC          — Chandrasekhar entropy-conserving flux (no dissipation)
  ES          — EC + standard Roe dissipation  (Eq.64)
  ES_Turkel   — EC + Turkel-preconditioned dissipation (Eq.44, 65)
  ES_Miczek   — EC + Miczek-preconditioned dissipation (Eq.45, 65)

Test:
  1D smooth density wave, single-species CPG (γ=1.4), periodic domain.
  Mach sweep M = 0.1 → 0.001: pressure perturbation should scale O(M2)
  for preconditioned schemes (correct low-Mach behaviour).

Entropy variables (γ-law EOS, Eq.33):
  v1 = (γ − s)/(γ−1) − β u2
  v2 = 2β u
  v3 = −2β
  where  s = ln p − γ ln ρ,   β = ρ/(2p)

Chandrasekhar EC flux (Eq.58):
  f*1 = ρ_ln · ū
  f*2 = ρ_ln · ū2 + p̂        p̂ = ρ_ln / (2β̄)
  f*3 = ρ_ln · ū · Ĥ         Ĥ = γ/[2(γ−1)β_ln] + ū2/2

Preconditioned ES dissipation (Eq.65):
  d = P⁻¹|PA| α_wave
  Turkel: acoustic eigenvalues λ± = ½(1+θ)u ± √[¼(1−θ)2u2 + θc2]
  Miczek: θ = (u/c)2 local, capped at 1
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GAMMA = 1.4
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Numerically stable logarithmic mean
# ─────────────────────────────────────────────────────────────

def log_mean(aL, aR):
    """
    Logarithmic mean: (aR - aL) / (ln aR - ln aL).
    Uses Taylor expansion when aL ≈ aR to avoid 0/0.
    """
    xi  = np.maximum(aR, 1e-300) / np.maximum(aL, 1e-300)
    f   = (xi - 1.0) / (xi + 1.0)
    u2  = f * f
    small = u2 < 1e-6
    # 6th-order Taylor: ln(ξ)/(ξ-1) ≈ 2/(1 + u2/3 + u⁴/5 + ...)
    F_small = 1.0 + u2 / 3.0 + u2**2 / 5.0 + u2**3 / 7.0
    F_large = np.log(np.maximum(xi, 1e-300)) / (
                  2.0 * np.where(np.abs(f) > 1e-15, f, 1e-15))
    F = np.where(small, F_small, F_large)
    return (aL + aR) / (2.0 * F)


# ─────────────────────────────────────────────────────────────
# Primitive / entropy helpers
# ─────────────────────────────────────────────────────────────

def prim(rho, rhou, rhoE, gam=GAMMA):
    u    = rhou / np.maximum(rho, 1e-300)
    p    = (gam - 1.0) * (rhoE - 0.5 * rho * u**2)
    p    = np.maximum(p, 1e-12)
    return u, p


def entropy_vars(rho, u, p, gam=GAMMA):
    """Entropy variables v1,v2,v3 (Eq.33)."""
    s    = np.log(np.maximum(p, 1e-300)) - gam * np.log(np.maximum(rho, 1e-300))
    beta = rho / (2.0 * np.maximum(p, 1e-300))
    v1   = (gam - s) / (gam - 1.0) - beta * u**2
    v2   = 2.0 * beta * u
    v3   = -2.0 * beta
    return v1, v2, v3


# ─────────────────────────────────────────────────────────────
# Chandrasekhar EC flux  (Eq.58)
# ─────────────────────────────────────────────────────────────

def chandrasekhar_ec(rhoL, uL, pL, rhoR, uR, pR, gam=GAMMA):
    """Two-point entropy-conserving flux (Chandrasekhar 1961)."""
    betaL   = rhoL / (2.0 * np.maximum(pL, 1e-300))
    betaR   = rhoR / (2.0 * np.maximum(pR, 1e-300))

    rho_ln  = log_mean(rhoL, rhoR)
    beta_ln = log_mean(betaL, betaR)
    beta_av = 0.5 * (betaL + betaR)
    u_av    = 0.5 * (uL + uR)

    p_hat   = rho_ln / (2.0 * beta_av)
    H_hat   = gam / (2.0 * (gam - 1.0) * beta_ln) + 0.5 * u_av**2

    f1 = rho_ln * u_av
    f2 = rho_ln * u_av**2 + p_hat
    f3 = rho_ln * u_av * H_hat
    return f1, f2, f3


# ─────────────────────────────────────────────────────────────
# Roe dissipation  (standard, Eq.64 D = R|Λ|Rᵀ)
# ─────────────────────────────────────────────────────────────

def _roe_state(rhoL, uL, pL, rhoR, uR, pR, gam=GAMMA):
    """Roe-averaged state."""
    rhoEL = pL / (gam - 1.0) + 0.5 * rhoL * uL**2
    rhoER = pR / (gam - 1.0) + 0.5 * rhoR * uR**2
    HL    = (rhoEL + pL) / np.maximum(rhoL, 1e-300)
    HR    = (rhoER + pR) / np.maximum(rhoR, 1e-300)

    sqL   = np.sqrt(np.maximum(rhoL, 0.0))
    sqR   = np.sqrt(np.maximum(rhoR, 0.0))
    denom = np.maximum(sqL + sqR, 1e-15)

    u_r   = (sqL * uL + sqR * uR) / denom
    H_r   = (sqL * HL + sqR * HR) / denom
    c2_r  = np.maximum((gam - 1.0) * (H_r - 0.5 * u_r**2), 1e-10)
    c_r   = np.sqrt(c2_r)
    rho_r = sqL * sqR   # = sqrt(ρL * ρR)
    return u_r, c_r, c2_r, H_r, rho_r


def _harten_fix(lam, c_r, eps_fac=0.1):
    delta = eps_fac * c_r
    return np.where(np.abs(lam) >= delta,
                    np.abs(lam),
                    0.5 * (lam**2 / delta + delta))


def _wave_strengths(rhoL, uL, pL, rhoR, uR, pR, u_r, c2_r, rho_r):
    dp   = pR - pL
    du   = uR - uL
    drho = rhoR - rhoL
    c_r  = np.sqrt(c2_r)
    alpha1 = (dp - rho_r * c_r * du) / (2.0 * c2_r)
    alpha2 = drho - dp / c2_r
    alpha3 = (dp + rho_r * c_r * du) / (2.0 * c2_r)
    return alpha1, alpha2, alpha3


def roe_dissipation(rhoL, uL, pL, rhoR, uR, pR, gam=GAMMA):
    """Standard Roe dissipation vector [dρ, dρu, dρE]."""
    u_r, c_r, c2_r, H_r, rho_r = _roe_state(rhoL, uL, pL, rhoR, uR, pR, gam)

    lam1 = _harten_fix(u_r - c_r, c_r)
    lam2 = _harten_fix(u_r,       c_r)
    lam3 = _harten_fix(u_r + c_r, c_r)

    alpha1, alpha2, alpha3 = _wave_strengths(
        rhoL, uL, pL, rhoR, uR, pR, u_r, c2_r, rho_r)

    d1 = (lam1 * alpha1
        + lam2 * alpha2
        + lam3 * alpha3)
    d2 = (lam1 * alpha1 * (u_r - c_r)
        + lam2 * alpha2 * u_r
        + lam3 * alpha3 * (u_r + c_r))
    d3 = (lam1 * alpha1 * (H_r - u_r * c_r)
        + lam2 * alpha2 * 0.5 * u_r**2
        + lam3 * alpha3 * (H_r + u_r * c_r))
    return d1, d2, d3


# ─────────────────────────────────────────────────────────────
# Turkel-preconditioned dissipation  (Eq.44, 65)
# ─────────────────────────────────────────────────────────────

def turkel_dissipation(rhoL, uL, pL, rhoR, uR, pR, Mr, gam=GAMMA):
    """
    Turkel preconditioner: acoustic eigenvalues replaced by
      λ± = ½(1+θ)u_r ± √[¼(1−θ)2u_r2 + θ·c_r2]
    θ = Mr2  (reference Mach2)
    Contact wave λ2 = u_r  unchanged.
    """
    u_r, c_r, c2_r, H_r, rho_r = _roe_state(rhoL, uL, pL, rhoR, uR, pR, gam)
    theta = float(np.clip(Mr**2, 1e-8, 1.0))

    half_sum = 0.5 * (1.0 + theta) * u_r
    discrim  = np.maximum((0.5 * (1.0 - theta) * u_r)**2 + theta * c2_r, 0.0)
    lam_ac   = np.sqrt(discrim)

    lam1 = _harten_fix(half_sum - lam_ac, c_r)
    lam2 = _harten_fix(u_r,               c_r)
    lam3 = _harten_fix(half_sum + lam_ac, c_r)

    alpha1, alpha2, alpha3 = _wave_strengths(
        rhoL, uL, pL, rhoR, uR, pR, u_r, c2_r, rho_r)

    # Eigenvectors: same Roe columns (u_r ± c_r for acoustic, u_r for contact)
    d1 = lam1 * alpha1 + lam2 * alpha2 + lam3 * alpha3
    d2 = (lam1 * alpha1 * (u_r - c_r)
        + lam2 * alpha2 * u_r
        + lam3 * alpha3 * (u_r + c_r))
    d3 = (lam1 * alpha1 * (H_r - u_r * c_r)
        + lam2 * alpha2 * 0.5 * u_r**2
        + lam3 * alpha3 * (H_r + u_r * c_r))
    return d1, d2, d3


# ─────────────────────────────────────────────────────────────
# Miczek-preconditioned dissipation  (Eq.45, 65)
# ─────────────────────────────────────────────────────────────

def miczek_dissipation(rhoL, uL, pL, rhoR, uR, pR, gam=GAMMA):
    """
    Miczek preconditioning: local θ = min((u_r/c_r)2, 1).
    Same structure as Turkel but θ is cell-interface-local.
    This ensures acoustic dissipation is NOT reduced (θ→1 at high Mach)
    while momentum dissipation is preconditioned at low Mach.
    """
    u_r, c_r, c2_r, H_r, rho_r = _roe_state(rhoL, uL, pL, rhoR, uR, pR, gam)
    M_loc = np.abs(u_r) / np.maximum(c_r, 1e-10)
    theta = np.minimum(M_loc**2, 1.0)
    theta = np.maximum(theta, 1e-8)

    half_sum = 0.5 * (1.0 + theta) * u_r
    discrim  = np.maximum((0.5 * (1.0 - theta) * u_r)**2 + theta * c2_r, 0.0)
    lam_ac   = np.sqrt(discrim)

    lam1 = _harten_fix(half_sum - lam_ac, c_r)
    lam2 = _harten_fix(u_r,               c_r)
    lam3 = _harten_fix(half_sum + lam_ac, c_r)

    alpha1, alpha2, alpha3 = _wave_strengths(
        rhoL, uL, pL, rhoR, uR, pR, u_r, c2_r, rho_r)

    d1 = lam1 * alpha1 + lam2 * alpha2 + lam3 * alpha3
    d2 = (lam1 * alpha1 * (u_r - c_r)
        + lam2 * alpha2 * u_r
        + lam3 * alpha3 * (u_r + c_r))
    d3 = (lam1 * alpha1 * (H_r - u_r * c_r)
        + lam2 * alpha2 * 0.5 * u_r**2
        + lam3 * alpha3 * (H_r + u_r * c_r))
    return d1, d2, d3


# ─────────────────────────────────────────────────────────────
# Interface flux assembly
# ─────────────────────────────────────────────────────────────

def interface_flux(rho, u, p, scheme, Mr=0.0, gam=GAMMA):
    """
    Compute m+1/2 fluxes for all cells simultaneously (periodic BCs).
    rho, u, p : cell-centered arrays (length N)
    Returns (f1, f2, f3) arrays of length N  [flux at m+1/2]
    """
    rhoR = np.roll(rho, -1)
    uR   = np.roll(u,   -1)
    pR   = np.roll(p,   -1)

    f1, f2, f3 = chandrasekhar_ec(rho, u, p, rhoR, uR, pR, gam)

    if scheme == 'EC':
        return f1, f2, f3

    if scheme == 'ES':
        d1, d2, d3 = roe_dissipation(rho, u, p, rhoR, uR, pR, gam)
    elif scheme == 'ES_Turkel':
        d1, d2, d3 = turkel_dissipation(rho, u, p, rhoR, uR, pR, Mr, gam)
    elif scheme == 'ES_Miczek':
        d1, d2, d3 = miczek_dissipation(rho, u, p, rhoR, uR, pR, gam)
    else:
        raise ValueError(f'Unknown scheme: {scheme}')

    return f1 - 0.5 * d1, f2 - 0.5 * d2, f3 - 0.5 * d3


# ─────────────────────────────────────────────────────────────
# RHS  d/dt [ρ, ρu, ρE] = −∂f/∂x
# ─────────────────────────────────────────────────────────────

def rhs(U, scheme, dx, Mr=0.0):
    rho, rhou, rhoE = U
    u, p = prim(rho, rhou, rhoE)

    f1, f2, f3 = interface_flux(rho, u, p, scheme, Mr)

    def div(f):
        return (f - np.roll(f, 1)) / dx

    return [-div(f1), -div(f2), -div(f3)], p


# ─────────────────────────────────────────────────────────────
# SSP-RK3
# ─────────────────────────────────────────────────────────────

def rkstep(U, scheme, dx, dt, Mr=0.0):
    def safe(V):
        V[0] = np.maximum(V[0], 1e-14)   # rho > 0
        return V

    k1, p1 = rhs(U, scheme, dx, Mr)
    U1 = safe([U[q] + dt * k1[q] for q in range(3)])

    k2, p2 = rhs(U1, scheme, dx, Mr)
    U2 = safe([0.75 * U[q] + 0.25 * (U1[q] + dt * k2[q]) for q in range(3)])

    k3, p3 = rhs(U2, scheme, dx, Mr)
    Un = safe([(1.0/3.0) * U[q] + (2.0/3.0) * (U2[q] + dt * k3[q])
               for q in range(3)])
    return Un, p3


# ─────────────────────────────────────────────────────────────
# Initial condition: smooth density wave
# ─────────────────────────────────────────────────────────────

def ic_density_wave(x, M0=0.1, delta=0.1, gam=GAMMA):
    """
    ρ = 1 + δ·sin(2πx),  p = 1/γ (uniform),  u = M0.
    c = 1 → Mach = M0.  Exact solution: advect density at u=M0.
    """
    rho  = 1.0 + delta * np.sin(2.0 * np.pi * x)
    p    = np.full_like(x, 1.0 / gam)
    u    = np.full_like(x, M0)
    rhoE = p / (gam - 1.0) + 0.5 * rho * u**2
    return rho, u, p, rho * u, rhoE


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def pressure_err(p, p0):
    return float(np.sqrt(np.mean((p - p0)**2)))


def entropy_err(rho, rhoE, rho0, rhoE0):
    return float(abs(np.sum(rhoE) - np.sum(rhoE0)) / (abs(np.sum(rhoE0)) + 1e-60))


# ─────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────

def run(scheme, M0=0.1, N=200, CFL=0.4, n_periods=2, verbose=True, gam=GAMMA):
    """
    Run one simulation.

    t_end = n_periods / M0  (flow-through periods)
    """
    dx   = 1.0 / N
    x    = np.linspace(dx / 2, 1.0 - dx / 2, N)
    t_end = float(n_periods) / M0

    rho0, u0, p0_arr, rhou0, rhoE0 = ic_density_wave(x, M0, gam=gam)
    p0 = 1.0 / gam

    U    = [rho0.copy(), rhou0.copy(), rhoE0.copy()]
    Mr   = M0   # reference Mach for Turkel

    t_hist  = [0.0]
    pe_hist = [pressure_err(p0_arr, p0)]
    en_hist = [0.0]

    t = 0.0; step = 0; diverged = False
    while t < t_end - 1e-12:
        rho_, rhou_, rhoE_ = U
        u_cur, p_cur = prim(rho_, rhou_, rhoE_, gam)
        c_cur = np.sqrt(np.maximum(gam * p_cur / np.maximum(rho_, 1e-300), 0.0))
        lam   = float(np.max(np.abs(u_cur) + c_cur))
        dt    = min(CFL * dx / max(lam, 1e-10), t_end - t)

        try:
            U, p_cur = rkstep(U, scheme, dx, dt, Mr)
        except Exception as e:
            print(f'  Exception at t={t:.4f}: {e}')
            diverged = True; break

        t += dt; step += 1

        pe_ = pressure_err(p_cur, p0)
        en_ = entropy_err(U[0], U[2], rho0, rhoE0)

        t_hist.append(t)
        pe_hist.append(pe_)
        en_hist.append(en_)

        if not np.isfinite(pe_) or pe_ > 1e3:
            print(f'  Diverged (PE={pe_:.2e}) at t={t:.4f}')
            diverged = True; break

        if verbose and (step % 500 == 0 or t >= t_end - 1e-12):
            print(f'  t={t:.4f}  step={step}  PE={pe_:.3e}  En={en_:.3e}')

    status = 'Completed' if not diverged else 'DIVERGED'
    if verbose:
        print(f'  --> {status} at t={t:.4f} ({step} steps)')

    rho_f, rhou_f, rhoE_f = U
    u_f, p_f = prim(rho_f, rhou_f, rhoE_f, gam)

    return (x, rho_f, u_f, p_f,
            np.array(t_hist), np.array(pe_hist), np.array(en_hist),
            diverged)


# ─────────────────────────────────────────────────────────────
# Mach scaling study
# ─────────────────────────────────────────────────────────────

def mach_scaling_study(Machs=(0.3, 0.1, 0.03, 0.01), N=200, n_periods=1):
    """
    For each M0, run EC / ES / ES_Turkel / ES_Miczek and record
    max pressure error at end.
    Expected:  ||Δp||_max ~ M^k
      ES (standard):          k ≈ 0   (bad low-Mach)
      ES_Turkel / ES_Miczek:  k ≈ 2   (correct scaling)
    """
    schemes = ['ES', 'ES_Turkel', 'ES_Miczek']
    colors  = {'ES': '#d62728', 'ES_Turkel': '#1f77b4', 'ES_Miczek': '#2ca02c'}
    markers = {'ES': 'o',       'ES_Turkel': 's',       'ES_Miczek': '^'}
    labels  = {'ES': 'ES (standard Roe)',
               'ES_Turkel': 'ES Turkel precond.',
               'ES_Miczek': 'ES Miczek precond.'}

    pe_max = {s: [] for s in schemes}

    for M0 in Machs:
        print(f'\n--- M0 = {M0} ---')
        for sch in schemes:
            print(f'  scheme={sch}', end='  ')
            res = run(sch, M0=M0, N=N, n_periods=n_periods, verbose=False)
            x, rho_f, u_f, p_f, t_h, pe_h, en_h, div = res
            pe_end = pe_h[-1] if not div else np.nan
            pe_max[sch].append(pe_end)
            print(f'||Δp||={pe_end:.3e}')

    return Machs, pe_max, schemes, colors, markers, labels


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────

def savefig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def plot_profiles(M0=0.1, N=200, n_periods=2):
    """Fig A: density + pressure profiles at t=t_end for all schemes."""
    schemes = ['EC', 'ES', 'ES_Turkel', 'ES_Miczek']
    colors  = ['#7f7f7f', '#d62728', '#1f77b4', '#2ca02c']
    lstyles = [':',       '--',       '-',        '-.']
    labels  = ['EC (no dissipation)', 'ES (Roe)', 'ES Turkel', 'ES Miczek']

    t_end = n_periods / M0
    p0 = 1.0 / GAMMA

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax_rho, ax_p, ax_pe = axes

    # exact (advected) density
    dx = 1.0 / N
    x_ref = np.linspace(dx/2, 1.0 - dx/2, N)
    rho_ex = 1.0 + 0.1 * np.sin(2.0 * np.pi * (x_ref - M0 * t_end))
    ax_rho.plot(x_ref, rho_ex, 'k-', lw=2, label='Exact', zorder=5)

    results = {}
    for sch, c, ls, lbl in zip(schemes, colors, lstyles, labels):
        print(f'scheme={sch}  M0={M0}  N={N}  n_per={n_periods}', end='  ')
        res = run(sch, M0=M0, N=N, n_periods=n_periods, verbose=False)
        results[sch] = res
        x, rho_f, u_f, p_f, t_h, pe_h, en_h, div = res
        print(f'PE_end={pe_h[-1]:.3e}  En_end={en_h[-1]:.3e}')

        ax_rho.plot(x, rho_f, color=c, ls=ls, label=lbl)
        ax_p.plot(x,   p_f,   color=c, ls=ls, label=lbl)
        ax_pe.semilogy(t_h, np.maximum(pe_h, 1e-18), color=c, ls=ls, label=lbl)

    ax_rho.set_title(f'Density  (M0={M0}, t={t_end:.1f})')
    ax_rho.set_xlabel('x'); ax_rho.legend(fontsize=8)

    ax_p.axhline(p0, color='k', lw=2, label=f'Exact p={p0:.3f}', zorder=5)
    ax_p.set_title('Pressure (exact: uniform)')
    ax_p.set_xlabel('x'); ax_p.legend(fontsize=8)

    ax_pe.set_title('||Δp||2 history')
    ax_pe.set_xlabel('t'); ax_pe.set_ylabel('||p - p0||2')
    ax_pe.legend(fontsize=8)

    fig.suptitle(f'ES schemes: density wave advection  M0={M0}  N={N}',
                 fontsize=12)
    plt.tight_layout()
    savefig(fig, f'es_profiles_M{str(M0).replace(".", "")}.png')
    return results


def plot_mach_scaling(Machs, pe_max, schemes, colors, markers, labels):
    """Fig B: Mach scaling of pressure error (log-log)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    M_arr = np.array(Machs, dtype=float)

    for sch in schemes:
        vals = np.array(pe_max[sch], dtype=float)
        ax.loglog(M_arr, vals, color=colors[sch],
                  marker=markers[sch], ls='-', label=labels[sch])

    # Reference lines
    ref1 = pe_max['ES'][0] * (M_arr / M_arr[0])**1   # O(M)
    ref2 = pe_max['ES'][0] * (M_arr / M_arr[0])**2   # O(M2)
    ax.loglog(M_arr, ref1, 'k--', lw=0.8, label=r'$O(M)$')
    ax.loglog(M_arr, ref2, 'k:',  lw=0.8, label=r'$O(M^2)$')

    ax.set_xlabel('Reference Mach  M0')
    ax.set_ylabel(r'$\|p - p_0\|_2$  at  $t = 1/M_0$')
    ax.set_title('Low-Mach pressure scaling: ES schemes')
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig(fig, 'es_mach_scaling.png')


def plot_pe_history_mach(M0=0.1, N=200, n_periods=2):
    """Fig C: PE error time history + energy conservation."""
    schemes = ['ES', 'ES_Turkel', 'ES_Miczek']
    colors  = {'ES': '#d62728', 'ES_Turkel': '#1f77b4', 'ES_Miczek': '#2ca02c'}
    lstyles = {'ES': '--',      'ES_Turkel': '-',        'ES_Miczek': '-.'}
    labels  = {'ES': 'ES (Roe)',
               'ES_Turkel': 'ES Turkel',
               'ES_Miczek': 'ES Miczek'}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))

    for sch in schemes:
        res = run(sch, M0=M0, N=N, n_periods=n_periods, verbose=False)
        x, rho_f, u_f, p_f, t_h, pe_h, en_h, div = res
        c = colors[sch]; ls = lstyles[sch]; lbl = labels[sch]
        a1.semilogy(t_h, np.maximum(pe_h, 1e-18), color=c, ls=ls, label=lbl)
        a2.semilogy(t_h, np.maximum(en_h, 1e-18), color=c, ls=ls, label=lbl)

    a1.set_xlabel('t'); a1.set_ylabel(r'$\|p - p_0\|_2$')
    a1.set_title(f'Pressure error  M0={M0}  N={N}')
    a1.legend()

    a2.set_xlabel('t'); a2.set_ylabel('Energy conservation error')
    a2.set_title('Total energy error')
    a2.legend()

    plt.tight_layout()
    savefig(fig, f'es_pe_history_M{str(M0).replace(".", "")}.png')


# ─────────────────────────────────────────────────────────────
# Entropy-conservation check (EC flux should give flat entropy)
# ─────────────────────────────────────────────────────────────

def entropy_conservation_check(M0=0.3, N=200, n_periods=1):
    """Verify EC flux conserves total entropy dΣ(−ρs/(γ-1))/dt ≈ 0."""
    dx   = 1.0 / N
    x    = np.linspace(dx / 2, 1.0 - dx / 2, N)
    rho0, u0, p0_arr, rhou0, rhoE0 = ic_density_wave(x, M0)
    p0   = 1.0 / GAMMA

    gam = GAMMA
    def total_entropy(rho, rhou, rhoE):
        u, p = prim(rho, rhou, rhoE)
        s    = np.log(np.maximum(p, 1e-300)) - gam * np.log(np.maximum(rho, 1e-300))
        return float(np.sum(-rho * s / (gam - 1.0)) * dx)

    t_end = 1.0 / M0
    U     = [rho0.copy(), rhou0.copy(), rhoE0.copy()]
    eta0  = total_entropy(*U)

    results_ec = {'t': [0.0], 'deta': [0.0]}
    results_es = {'t': [0.0], 'deta': [0.0]}

    t = 0.0
    while t < t_end - 1e-12:
        u_cur, p_cur = prim(U[0], U[1], U[2])
        c_cur  = np.sqrt(np.maximum(gam * p_cur / np.maximum(U[0], 1e-300), 0.0))
        lam    = float(np.max(np.abs(u_cur) + c_cur))
        dt     = min(0.4 * dx / max(lam, 1e-10), t_end - t)

        U_ec, _ = rkstep(U,    'EC', dx, dt, M0)
        U_es, _ = rkstep(U,    'ES', dx, dt, M0)
        U,    _ = rkstep(U,    'EC', dx, dt, M0)   # advance on EC branch
        t += dt

        results_ec['t'].append(t)
        results_ec['deta'].append((total_entropy(*U_ec) - eta0) / abs(eta0 + 1e-60))
        results_es['t'].append(t)
        results_es['deta'].append((total_entropy(*U_es) - eta0) / abs(eta0 + 1e-60))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(results_ec['t'],
                np.maximum(np.abs(results_ec['deta']), 1e-18),
                'b-', label='EC (Chandrasekhar)')
    ax.semilogy(results_es['t'],
                np.maximum(np.abs(results_es['deta']), 1e-18),
                'r--', label='ES (Roe)')
    ax.set_xlabel('t')
    ax.set_ylabel(r'$|\Delta\sum\eta| / |\sum\eta_0|$')
    ax.set_title(f'Entropy conservation check  M0={M0}  N={N}')
    ax.legend()
    plt.tight_layout()
    savefig(fig, 'es_entropy_conservation.png')
    print(f'  EC entropy drift: {results_ec["deta"][-1]:.3e}')
    print(f'  ES entropy drift: {results_es["deta"][-1]:.3e}')


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 60)
    print('  ES / EC scheme validation  (single-species CPG)')
    print('=' * 60)

    # 1. Entropy conservation check
    print('\n[1] Entropy conservation check (M0=0.3, N=200, 1 period)')
    entropy_conservation_check(M0=0.3, N=200, n_periods=1)

    # 2. Profile plots at M0=0.1
    print('\n[2] Profile comparison  M0=0.1  N=200  2 periods')
    plot_profiles(M0=0.1, N=200, n_periods=2)

    # 3. Low Mach profile M0=0.01
    print('\n[3] Profile comparison  M0=0.01  N=200  2 periods')
    plot_profiles(M0=0.01, N=200, n_periods=2)

    # 4. PE history at M0=0.1
    print('\n[4] PE & energy error history  M0=0.1')
    plot_pe_history_mach(M0=0.1, N=200, n_periods=2)

    # 5. Mach scaling study
    print('\n[5] Mach scaling study  N=200  1 period')
    Machs, pe_max, schemes, colors, markers, labels = mach_scaling_study(
        Machs=(0.3, 0.1, 0.03, 0.01), N=200, n_periods=1)
    plot_mach_scaling(Machs, pe_max, schemes, colors, markers, labels)

    # Print summary table
    print('\n  Mach scaling summary:')
    print('  {:>6}  '.format('M0') + '  '.join('{:>14}'.format(s) for s in schemes))
    for i, M0 in enumerate(Machs):
        vals = '  '.join('{:>14.3e}'.format(pe_max[s][i]) for s in schemes)
        print('  {:>6.3f}  {}'.format(M0, vals))

    print('\nDone. Figures saved to ./output/')
