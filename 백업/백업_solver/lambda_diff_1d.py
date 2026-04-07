"""
lambda_diff_1d.py
=================
1D Multi-Component Euler Equations — Kinetic Scheme with Positivity Preservation

Implements the numerical scheme from:
  Roy & Raghurama Rao (2025), "A Kinetic Scheme Based On Positivity Preservation
  For Multi-component Euler Equations", arXiv:2411.00285v2

State vector  U = [rho*W, rho, rho*u, rho*E]^T         (Eq. 13)
Flux vector   G = [rho*W*u, rho*u, rho*u^2+p, (rho*E+p)*u]^T

Interface flux (1st order, Eq. 27):
  G_{j+1/2} = 0.5*(G_j + G_{j+1}) - lambda_{j+1/2}/2 * (U_{j+1} - U_j)

lambda (Eq. 42): see compute_lambda()
SSPRK3 (Eq. 51) + Chakravarthy-Osher minmod flux limiter (Eq. 50, b=4)

Usage:
  python lambda_diff_1d.py --test steady_contact   # Fig. 5
  python lambda_diff_1d.py --test moving_same      # Fig. 6
  python lambda_diff_1d.py --test sod_same         # Fig. 9
  python lambda_diff_1d.py --test sod_diff         # Fig. 10
  python lambda_diff_1d.py --test positivity       # Fig. 11
  python lambda_diff_1d.py --test eoc              # Table 1-4
  python lambda_diff_1d.py --test all
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe; change to "TkAgg" for interactive
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
EPS0 = 1e-10      # division-by-zero guard (paper notation: epsilon_0)
B_COMPRESS = 4    # Chakravarthy-Osher compression parameter (paper uses b=4)
CFL = 0.8         # default CFL number sigma


# ─────────────────────────────────────────────
# Primitive variable extraction
# ─────────────────────────────────────────────

def compute_primitives(U, gam1, gam2):
    """
    Given conserved variables U = [rho*W, rho, rho*u, rho*E] and
    the two pure-component adiabatic constants gam1, gam2, return
    primitive variables at every cell.

    Parameters
    ----------
    U     : (4, N) array, conserved state
    gam1  : float, gamma for component 1
    gam2  : float, gamma for component 2

    Returns
    -------
    rho, u, p, gam, a : each (N,) array
        gam = mixture gamma = cp_mix / cv_mix    (Eq. 10, 14)
        a   = sqrt(gamma * p / rho)              (speed of sound)
    """
    rhoW = U[0]
    rho  = U[1]
    rhou = U[2]
    rhoE = U[3]

    W    = np.clip(rhoW / (rho + EPS0), 0.0, 1.0)    # mass fraction component 1
    u    = rhou / (rho + EPS0)

    # Mixture thermodynamics (Eq. 14)
    # cv_c = 1 / (gamma_c - 1) for unit R_c (or unit cv_c = 1 assumed in paper tests)
    # The paper sets (cv)1 = (cv)2 = 1 in most tests.
    # General: cv_mix = W*cv1 + (1-W)*cv2,  cp_mix = W*gam1*cv1 + (1-W)*gam2*cv2
    cv1 = 1.0 / (gam1 - 1.0)
    cv2 = 1.0 / (gam2 - 1.0)
    cv  = W * cv1 + (1.0 - W) * cv2
    cp  = W * gam1 * cv1 + (1.0 - W) * gam2 * cv2
    gam = cp / (cv + EPS0)

    # Pressure from EOS: rhoE = p/(gamma-1) + rho*u^2/2   (Eq. 11)
    p = (gam - 1.0) * (rhoE - 0.5 * rho * u * u)
    p = np.maximum(p, EPS0)          # positivity guard

    a = np.sqrt(np.maximum(gam * p / (rho + EPS0), EPS0))

    return rho, u, p, gam, a


# ─────────────────────────────────────────────
# Flux vector
# ─────────────────────────────────────────────

def compute_flux(U, gam1, gam2):
    """
    Physical flux G = [rho*W*u, rho*u, rho*u^2+p, (rho*E+p)*u]  (Eq. 13)

    Returns G : (4, N) array
    """
    rhoW = U[0]
    rho  = U[1]
    _, u, p, _, _ = compute_primitives(U, gam1, gam2)
    rhoE = U[3]
    G = np.array([
        rhoW * u,
        rho  * u,
        rho  * u * u + p,
        (rhoE + p) * u,
    ])
    return G


# ─────────────────────────────────────────────
# Lambda (numerical diffusion coefficient)
# ─────────────────────────────────────────────

def abs_sign(x):
    """
    abs_sign(x) = 1 if |x| > EPS0, else 0    (Eq. 42d)
    Note: this is NOT the sign function — it returns 0 at the origin.
    """
    return np.where(np.abs(x) > EPS0, 1.0, 0.0)


def lambda_RH(U, G):
    """
    Rankine-Hugoniot numerical wave speed (Eq. 41, lambda_RH,b).

    Uses differences between adjacent cells j and j+1 to give
    lambda at interface j+1/2.

    Parameters
    ----------
    U : (4, N) array  — states at cells 0..N-1
    G : (4, N) array  — fluxes at cells 0..N-1

    Returns
    -------
    lam_rh : (N-1,) array — lambda_RH at interfaces 1/2, 3/2, ..., (N-3/2)
    """
    # Delta = ()_{j+1} - ()_j  for j = 0..N-2
    dU = U[:, 1:] - U[:, :-1]       # (4, N-1)
    dG = G[:, 1:] - G[:, :-1]       # (4, N-1)

    # Eq. 41: min over i=2,3,4 of |dG_i| / (|dU_i| + eps0)
    # Index mapping (0-based): i=2 → rho*u, i=3 → rho*u^2+p, i=4 → rho*E*u+p*u
    terms = np.array([
        np.abs(dG[1]) / (np.abs(dU[1]) + EPS0),   # |Δ(ρu)| / |Δρ|
        np.abs(dG[2]) / (np.abs(dU[2]) + EPS0),   # |Δ(ρu²+p)| / |Δ(ρu)|
        np.abs(dG[3]) / (np.abs(dU[3]) + EPS0),   # |Δ((ρE+p)u)| / |Δ(ρE)|
    ])                                              # (3, N-1)
    return np.min(terms, axis=0)                    # (N-1,)


def compute_lambda(U, G, gam1, gam2):
    """
    Compute lambda at every interface j+1/2 for j = 0..N-2.

    Applies Eq. 42 — the modified definition that:
      (a) satisfies positivity preservation (Eq. 32)
      (b) exactly captures a steady contact discontinuity

    Parameters
    ----------
    U    : (4, N) array — interior + ghost cells
    G    : (4, N) array — physical flux
    gam1, gam2 : float

    Returns
    -------
    lam : (N-1,) array — lambda at N-1 interfaces
    """
    rho, u, p, gam, a = compute_primitives(U, gam1, gam2)

    # Left / Right cell at each interface (j = left, j+1 = right)
    rho_L, rho_R = rho[:-1], rho[1:]
    u_L,   u_R   = u[:-1],   u[1:]
    p_L,   p_R   = p[:-1],   p[1:]
    gam_L, gam_R = gam[:-1], gam[1:]
    a_L,   a_R   = a[:-1],   a[1:]

    # Positivity terms (Eq. 32)
    alpha_L = -u_L + np.sqrt(np.maximum((gam_L - 1.0) / (2.0 * gam_L), 0.0)) * a_L
    alpha_R =  u_R + np.sqrt(np.maximum((gam_R - 1.0) / (2.0 * gam_R), 0.0)) * a_R

    lam_rh = lambda_RH(U, G)   # (N-1,)

    # Default lambda (Eq. 42b)
    lam_base = np.maximum.reduce([lam_rh, alpha_L, alpha_R])  # element-wise max of 3

    # Contact discontinuity detector (Eq. 42 condition)
    rho_avg = 0.5 * (rho_L + rho_R)
    p_avg   = 0.5 * (p_L   + p_R)
    rel_rho_jump = np.abs(rho_R - rho_L) / (rho_avg + EPS0)
    rel_p_jump   = np.abs(p_R   - p_L)   / (p_avg   + EPS0)

    is_contact = (rel_rho_jump > 0.1) & (rel_p_jump < 0.1)

    # At contact discontinuity: lambda = abs_sign(u_L + u_R) * lam_base  (Eq. 42a)
    lam_contact = abs_sign(u_L + u_R) * lam_base

    lam = np.where(is_contact, lam_contact, lam_base)
    return np.maximum(lam, 0.0)   # lambda >= 0


# ─────────────────────────────────────────────
# Time step
# ─────────────────────────────────────────────

def compute_dt(U, lam, dx, gam1, gam2, sigma=CFL, order=1):
    """
    Global time step satisfying both positivity (Eq. 33) and
    stability (Eq. 37) criteria.

    Δt = σ · min(Δt_p, Δt_s)   for 1st order  (Eq. 38)
    Δt = σ · min(Δt_p/2, Δt_s) for higher order (Eq. 52)
    """
    # Positivity time step (Eq. 33)
    # lambda at j+1/2 and j-1/2 for interior cells j=1..N-2
    lam_plus  = lam[1:]    # lambda_{j+1/2}
    lam_minus = lam[:-1]   # lambda_{j-1/2}
    dt_p = np.min(2.0 * dx / (lam_plus + lam_minus + EPS0))

    # Stability time step (Eq. 37)
    _, u, p, gam, a = compute_primitives(U[:, 1:-1], gam1, gam2)   # interior
    lambda_max = np.maximum.reduce([np.abs(u - a), np.abs(u), np.abs(u + a)])
    dt_s = np.min(dx / (lambda_max + EPS0))

    if order > 1:
        dt_p = dt_p / 2.0    # Eq. 52

    return sigma * min(dt_p, dt_s)


# ─────────────────────────────────────────────
# Interface fluxes
# ─────────────────────────────────────────────

def interface_flux_1st(U, G, lam):
    """
    First-order interface flux (Eq. 27).
    G_{j+1/2} = 0.5*(G_j + G_{j+1}) - lam_{j+1/2}/2 * (U_{j+1} - U_j)

    Parameters
    ----------
    U   : (4, N)   — all cells (including ghosts)
    G   : (4, N)   — physical flux
    lam : (N-1,)   — lambda at N-1 interfaces

    Returns
    -------
    Gf : (4, N-1) — interface fluxes at all N-1 interfaces
    """
    avg_G = 0.5 * (G[:, :-1] + G[:, 1:])    # (4, N-1)
    dU    = U[:, 1:] - U[:, :-1]             # (4, N-1)
    Gf    = avg_G - 0.5 * lam[np.newaxis, :] * dU
    return Gf


def minmod(x, y):
    """Minmod limiter (Eq. 48b)."""
    return np.where(x * y > 0,
                    np.where(np.abs(x) < np.abs(y), x, y),
                    0.0)


def interface_flux_3rd(U, G, lam, b=B_COMPRESS, bc="transmissive"):
    """
    Chakravarthy-Osher flux-limited interface flux (Eq. 43/50).

    b=4  → 3rd order limited (Eq. 50, default)
    b=1  → 2nd order limited (Eq. 45, symmetric simplification)
    b=0  → 3rd order unlimited (Eq. 44, φ=1)

    G_{j+1/2}^{3O} = G_{j+1/2}^{1O}
      + 1/6 · φ(b·ΔG+_{j+1/2}, ΔG+_{j-1/2})
      - 1/6 · φ(b·ΔG-_{j+1/2}, ΔG-_{j+3/2})
      + 1/3 · φ(b·ΔG+_{j-1/2}, ΔG+_{j+1/2})
      - 1/3 · φ(b·ΔG-_{j+3/2}, ΔG-_{j+1/2})

    where ΔG±_{j+1/2} from Eq. 64c/64d.

    For periodic BC, boundary interfaces (0 and M-1) are also corrected
    using the wrap-around neighbor stencil.
    """
    # --- First-order fluxes and upwind differences at all interfaces ---
    dU  = U[:, 1:] - U[:, :-1]         # (4, M) where M = N-1
    dG  = G[:, 1:] - G[:, :-1]         # (4, M)

    # ΔG+_{j+1/2} = 0.5*(G_{j+1}-G_j) + lam/2*(U_{j+1}-U_j)  (Eq. 64c)
    dGp = 0.5 * dG + 0.5 * lam[np.newaxis, :] * dU   # (4, M)
    # ΔG-_{j+1/2} = 0.5*(G_{j+1}-G_j) - lam/2*(U_{j+1}-U_j)  (Eq. 64d)
    dGm = 0.5 * dG - 0.5 * lam[np.newaxis, :] * dU   # (4, M)

    Gf1 = interface_flux_1st(U, G, lam)                # (4, M)
    Gf3 = Gf1.copy()

    i = slice(1, -1)   # inner interface indices: 1 .. M-2

    if b == 0:
        # Unlimited 3rd order (Eq. 44): set all limiters φ = 1.
        # φ(b*x, y) with φ=1 returns the second argument y.
        Gf3[:, i] += (
              (1.0/6.0) * dGp[:, 0:-2]   # ΔG+_{j-1/2}
            - (1.0/6.0) * dGm[:, 2:  ]   # ΔG-_{j+3/2}
            + (1.0/3.0) * dGp[:, i    ]  # ΔG+_{j+1/2}
            - (1.0/3.0) * dGm[:, i    ]  # ΔG-_{j+1/2}
        )
        if bc == "periodic":
            # Interface 0 (= interface M-1 physically):
            #   left neighbor ΔG+_{-1/2} = dGp[:, -2] (= dGp[:, Nx-1])
            #   right+1 neighbor ΔG-_{3/2} = dGm[:, 1]
            corr = (
                  (1.0/6.0) * dGp[:, -2]
                - (1.0/6.0) * dGm[:,  1]
                + (1.0/3.0) * dGp[:,  0]
                - (1.0/3.0) * dGm[:,  0]
            )
            Gf3[:,  0] += corr
            Gf3[:, -1] += corr   # interface 0 and M-1 are the same physical interface
    else:
        # Limited (Eq. 50): b=1 → 2nd order, b=4 → 3rd order
        Gf3[:, i] += (
              (1.0/6.0) * minmod(b * dGp[:, i],   dGp[:, 0:-2])
            - (1.0/6.0) * minmod(b * dGm[:, i],   dGm[:, 2:  ])
            + (1.0/3.0) * minmod(b * dGp[:, 0:-2], dGp[:, i   ])
            - (1.0/3.0) * minmod(b * dGm[:, 2:  ], dGm[:, i   ])
        )
        if bc == "periodic":
            corr = (
                  (1.0/6.0) * minmod(b * dGp[:,  0], dGp[:, -2])
                - (1.0/6.0) * minmod(b * dGm[:,  0], dGm[:,  1])
                + (1.0/3.0) * minmod(b * dGp[:, -2], dGp[:,  0])
                - (1.0/3.0) * minmod(b * dGm[:,  1], dGm[:,  0])
            )
            Gf3[:,  0] += corr
            Gf3[:, -1] += corr

    return Gf3


# ─────────────────────────────────────────────
# Residual and time integration
# ─────────────────────────────────────────────

def residual(U_ext, dx, gam1, gam2, order=1, bc="transmissive"):
    """
    Spatial residual R(U) = (1/dx) * [G_{j+1/2} - G_{j-1/2}]
    for interior cells j = 1..N-2 (ghost cells at 0 and N-1).

    order=1 → 1st-order upwind (Eq. 27)
    order=2 → flux-limited, b=1 (Eq. 45)
    order=3 → flux-limited, b=4 (Eq. 50)
    order=4 → unlimited 3rd order (Eq. 44)

    Returns R : (4, N-2) — one entry per interior cell
    """
    G   = compute_flux(U_ext, gam1, gam2)
    lam = compute_lambda(U_ext, G, gam1, gam2)   # (N-1,)

    if order == 1:
        Gf = interface_flux_1st(U_ext, G, lam)                       # (4, N-1)
    elif order == 2:
        Gf = interface_flux_3rd(U_ext, G, lam, b=1, bc=bc)           # 2nd order, b=1
    elif order == 3:
        Gf = interface_flux_3rd(U_ext, G, lam, b=B_COMPRESS, bc=bc)  # 3rd order, b=4
    else:
        Gf = interface_flux_3rd(U_ext, G, lam, b=0, bc=bc)           # unlimited, b=0

    # Interior cells: j = 1..N-2
    R = (Gf[:, 1:] - Gf[:, :-1]) / dx            # (4, N-2)
    return R


def apply_bc_transmissive(U):
    """
    Transmissive (zero-gradient / extrapolation) boundary conditions.
    Ghost cells: U[0] = U[1],  U[-1] = U[-2].
    U shape: (4, N) where N = Nx + 2 (two ghost cells).
    """
    U[:, 0]  = U[:, 1]
    U[:, -1] = U[:, -2]


def apply_bc_periodic(U):
    """Periodic boundary conditions (used for EOC test)."""
    U[:, 0]  = U[:, -2]
    U[:, -1] = U[:, 1]


def step_euler(U_int, dx, dt, gam1, gam2, order=1, bc="transmissive"):
    """
    One forward-Euler time step (1st-order in time).
    U_int : (4, Nx) interior cells
    Returns U_int_new : (4, Nx)
    """
    N = U_int.shape[1]
    U_ext = np.zeros((4, N + 2))
    U_ext[:, 1:-1] = U_int
    if bc == "periodic":
        apply_bc_periodic(U_ext)
    else:
        apply_bc_transmissive(U_ext)

    R = residual(U_ext, dx, gam1, gam2, order=order, bc=bc)
    return U_int - dt * R


def step_ssprk3(U_int, dx, dt, gam1, gam2, order=3, bc="transmissive"):
    """
    SSPRK3 (Strong Stability Preserving Runge-Kutta, 3rd order) (Eq. 51).
    U_int : (4, Nx) interior cells
    Returns U_int_new : (4, Nx)
    """
    def R(U):
        N = U.shape[1]
        U_ext = np.zeros((4, N + 2))
        U_ext[:, 1:-1] = U
        if bc == "periodic":
            apply_bc_periodic(U_ext)
        else:
            apply_bc_transmissive(U_ext)
        return residual(U_ext, dx, gam1, gam2, order=order, bc=bc)

    U0 = U_int
    U1 =            U0 - dt * R(U0)                            # Eq. 51a
    U2 = 0.75*U0 + 0.25*U1 - 0.25*dt * R(U1)                  # Eq. 51b
    U3 = (1.0/3.0)*U0 + (2.0/3.0)*U2 - (2.0/3.0)*dt * R(U2)  # Eq. 51c
    return U3


# ─────────────────────────────────────────────
# Main simulation runner
# ─────────────────────────────────────────────

def run_simulation(init_U, Nx, dx, t_end, gam1, gam2,
                   order=1, sigma=CFL, bc="transmissive",
                   max_steps=500_000):
    """
    Run simulation from t=0 to t=t_end.

    Parameters
    ----------
    init_U   : (4, Nx) initial interior state
    Nx       : int, number of interior cells
    dx       : float, cell width
    t_end    : float, final time
    gam1, gam2 : float, gamma values for components 1, 2
    order    : int, 1 or 3
    sigma    : float, CFL number
    bc       : "transmissive" or "periodic"
    max_steps: int, safety cap on iterations

    Returns
    -------
    U : (4, Nx) final interior state
    t : float, actual final time
    """
    U = init_U.copy()
    t = 0.0

    for _ in range(max_steps):
        if t >= t_end:
            break

        # Build extended state for dt computation
        N = U.shape[1]
        U_ext = np.zeros((4, N + 2))
        U_ext[:, 1:-1] = U
        if bc == "periodic":
            apply_bc_periodic(U_ext)
        else:
            apply_bc_transmissive(U_ext)

        G   = compute_flux(U_ext, gam1, gam2)
        lam = compute_lambda(U_ext, G, gam1, gam2)

        dt = compute_dt(U_ext, lam, dx, gam1, gam2, sigma=sigma, order=order)
        dt = min(dt, t_end - t)

        if order == 1:
            U = step_euler(U, dx, dt, gam1, gam2, order=1, bc=bc)
        else:
            U = step_ssprk3(U, dx, dt, gam1, gam2, order=order, bc=bc)

        t += dt

    return U, t


# ─────────────────────────────────────────────
# Conserved ↔ Primitive helpers
# ─────────────────────────────────────────────

def prim_to_cons(W_arr, rho_arr, u_arr, p_arr, gam1, gam2):
    """
    Convert primitive variables to conserved variables U.

    Parameters (all arrays of length Nx):
        W_arr  : mass fraction of component 1
        rho_arr: total density
        u_arr  : velocity
        p_arr  : pressure
    Returns U : (4, Nx)
    """
    cv1 = 1.0 / (gam1 - 1.0)
    cv2 = 1.0 / (gam2 - 1.0)
    cv  = W_arr * cv1 + (1.0 - W_arr) * cv2
    cp  = W_arr * gam1 * cv1 + (1.0 - W_arr) * gam2 * cv2
    gam = cp / (cv + EPS0)

    rhoE = p_arr / (gam - 1.0) + 0.5 * rho_arr * u_arr ** 2
    return np.array([
        rho_arr * W_arr,
        rho_arr,
        rho_arr * u_arr,
        rhoE,
    ])


def cons_to_prim(U, gam1, gam2):
    """Extract (W, rho, u, p) from conserved state."""
    rho, u, p, _, _ = compute_primitives(U, gam1, gam2)
    W = np.clip(U[0] / (rho + EPS0), 0.0, 1.0)
    return W, rho, u, p


# ─────────────────────────────────────────────
# Plotting helper
# ─────────────────────────────────────────────

def _plot_result(x, results, labels, title, filename,
                 exact=None, exact_label="Exact"):
    """
    Plot W, rho, u, p for one or more solutions.
    results : list of (W, rho, u, p) tuples
    exact   : (W, rho, u, p) or None
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    var_names = ["Mass fraction W", "Density ρ", "Velocity u", "Pressure p"]

    for ax, name, idx in zip(axes, var_names, range(4)):
        for (W, rho, u, p), lbl in zip(results, labels):
            vals = [W, rho, u, p][idx]
            ax.plot(x, vals, marker='.', ms=3, lw=1.2, label=lbl)
        if exact is not None:
            vals = exact[idx]
            ax.plot(x, vals, 'k-', lw=1.5, label=exact_label)
        ax.set_xlabel("x")
        ax.set_title(name)
        ax.legend(fontsize=7)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"  Saved: {filename}")


# ─────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────

def _shock_tube_ic(Nx, WL, rhoL, uL, pL, WR, rhoR, uR, pR,
                   gam1, gam2, x_disc=0.5):
    """Build initial condition for a Riemann problem on [0,1]."""
    dx = 1.0 / Nx
    x  = (np.arange(Nx) + 0.5) * dx
    W   = np.where(x <= x_disc, WL,   WR)
    rho = np.where(x <= x_disc, rhoL, rhoR)
    u   = np.where(x <= x_disc, uL,   uR)
    p   = np.where(x <= x_disc, pL,   pR)
    return x, prim_to_cons(W, rho, u, p, gam1, gam2)


def run_test_steady_contact(Nx=200, sigma=CFL):
    """
    Test 5.2.1 — Steady contact discontinuity (Fig. 5, Eq. 90).
    Left:  rho=1,   W=1, u=0, p=1, gam=1.6
    Right: rho=0.1, W=0, u=0, p=1, gam=1.4
    """
    print("\n[Test] Steady contact discontinuity (Sec. 5.2.1)")
    gam1, gam2 = 1.6, 1.4
    t_end = 0.1

    x, U0 = _shock_tube_ic(Nx, 1.0, 1.0,   0.0, 1.0,
                                0.0, 0.1,   0.0, 1.0,
                                gam1, gam2)

    results = []
    labels  = []
    for order, lbl in [(1, "1O"), (2, "2O"), (3, "3O")]:
        U, _ = run_simulation(U0.copy(), Nx, 1.0/Nx, t_end, gam1, gam2,
                              order=order, sigma=sigma)
        results.append(cons_to_prim(U, gam1, gam2))
        labels.append(lbl)

    # Exact: profile unchanged (steady)
    W_ex  = np.where(x <= 0.5, 1.0, 0.0)
    rho_ex = np.where(x <= 0.5, 1.0, 0.1)
    u_ex  = np.zeros_like(x)
    p_ex  = np.ones_like(x)
    exact = (W_ex, rho_ex, u_ex, p_ex)

    _plot_result(x, results, labels,
                 f"Steady contact discontinuity (t={t_end}, Nx={Nx})",
                 "test_steady_contact.png", exact=exact)
    print("  Contact should be captured exactly at all orders.")


def run_test_moving_same(Nx=200, sigma=CFL):
    """
    Test 5.2.2 — Moving contact discontinuity, same gamma (Fig. 6, Eq. 91).
    Left:  rho=1,   W=1, u=1, p=1, gam=1.4
    Right: rho=0.1, W=0, u=1, p=1, gam=1.4
    """
    print("\n[Test] Moving contact discontinuity, same gamma (Sec. 5.2.2)")
    gam1, gam2 = 1.4, 1.4
    t_end = 0.3   # propagated to the right

    x, U0 = _shock_tube_ic(Nx, 1.0, 1.0, 1.0, 1.0,
                                0.0, 0.1, 1.0, 1.0,
                                gam1, gam2)

    results, labels = [], []
    for order, lbl in [(1, "1O"), (2, "2O"), (3, "3O")]:
        U, _ = run_simulation(U0.copy(), Nx, 1.0/Nx, t_end, gam1, gam2,
                              order=order, sigma=sigma)
        results.append(cons_to_prim(U, gam1, gam2))
        labels.append(lbl)

    # Exact: discontinuity moves at u=1, p and u remain uniform
    x_disc_t = 0.5 + 1.0 * t_end
    W_ex  = np.where(x <= min(x_disc_t, 1.0), 1.0, 0.0)
    rho_ex = np.where(x <= min(x_disc_t, 1.0), 1.0, 0.1)
    u_ex  = np.ones_like(x)
    p_ex  = np.ones_like(x)
    exact = (W_ex, rho_ex, u_ex, p_ex)

    _plot_result(x, results, labels,
                 f"Moving contact discontinuity, same γ (t={t_end}, Nx={Nx})",
                 "test_moving_same.png", exact=exact)


def run_test_sod_same(Nx=200, sigma=CFL):
    """
    Test 5.2.4 — Sod's shock tube, same gamma (Fig. 9, Eq. 93).
    Left:  rho=2, W=1, u=0, p=10, gam=1.4
    Right: rho=1, W=0, u=0, p=1,  gam=1.4
    """
    print("\n[Test] Sod's shock tube, same gamma (Sec. 5.2.4)")
    gam1, gam2 = 1.4, 1.4
    t_end = 0.1

    x, U0 = _shock_tube_ic(Nx, 1.0, 2.0, 0.0, 10.0,
                                0.0, 1.0, 0.0,  1.0,
                                gam1, gam2)

    results, labels = [], []
    for order, lbl in [(1, "1O"), (2, "2O"), (3, "3O")]:
        U, _ = run_simulation(U0.copy(), Nx, 1.0/Nx, t_end, gam1, gam2,
                              order=order, sigma=sigma)
        results.append(cons_to_prim(U, gam1, gam2))
        labels.append(lbl)

    exact = _exact_sod(x, t_end, rhoL=2.0, uL=0.0, pL=10.0,
                        rhoR=1.0, uR=0.0, pR=1.0, gam=1.4)
    # For two-component, W follows characteristic exactly
    W_ex = np.where(x <= 0.5 + exact[1]*t_end, 1.0, 0.0)   # approx; use density
    W_ex_exact = np.where(x <= 0.5, 1.0, 0.0)               # initial layout (contact moves)

    _plot_result(x, results, labels,
                 f"Sod shock tube, same γ (t={t_end}, Nx={Nx})",
                 "test_sod_same.png",
                 exact=(W_ex_exact, exact[0], exact[1], exact[2]))


def run_test_sod_diff(Nx=200, sigma=CFL):
    """
    Test 5.2.5 — Sod's shock tube, different gamma (Fig. 10, Eq. 94).
    Left:  rho=1,     W=1, u=0, p=1,   gam=1.4
    Right: rho=0.125, W=0, u=0, p=0.1, gam=1.2
    cv1 = cv2 = 1.
    """
    print("\n[Test] Sod's shock tube, different gamma (Sec. 5.2.5)")
    gam1, gam2 = 1.4, 1.2
    t_end = 0.2

    x, U0 = _shock_tube_ic(Nx, 1.0, 1.0,   0.0, 1.0,
                                0.0, 0.125, 0.0, 0.1,
                                gam1, gam2)

    results, labels = [], []
    for order, lbl in [(1, "1O"), (2, "2O"), (3, "3O")]:
        U, _ = run_simulation(U0.copy(), Nx, 1.0/Nx, t_end, gam1, gam2,
                              order=order, sigma=sigma)
        results.append(cons_to_prim(U, gam1, gam2))
        labels.append(lbl)

    _plot_result(x, results, labels,
                 f"Sod shock tube, different γ (t={t_end}, Nx={Nx})",
                 "test_sod_diff.png")
    print("  3O solution may show mass fraction outside [0,1] -- expected (Sec 5.2.5).")


def run_test_positivity(Nx=200, sigma=CFL):
    """
    Test 5.2.6 — Positivity of mass fraction problem (Fig. 11, Eq. 95).
    Left:  rho=1, W=1, u=-1, H=1,  gam=1.4  → p=(H - u^2/2)*rho*(gam-1)/gam
    Right: rho=1, W=0, u=1,  H=5,  gam=1.4
    H = E + p/rho = total enthalpy per unit mass
    """
    print("\n[Test] Positivity of mass fraction (Sec. 5.2.6)")
    gam1, gam2 = 1.4, 1.4
    t_end = 0.15

    # Recover p from H: H = e + p/rho + u^2/2, e = p/((gam-1)*rho)
    # H = p/(rho*(gam-1)) + p/rho + u^2/2 = p*(gam/(rho*(gam-1))) + u^2/2
    # p = (H - u^2/2) * rho*(gam-1)/gam
    def H_to_p(H, u, rho, gam):
        return (H - 0.5 * u**2) * rho * (gam - 1.0) / gam

    pL = H_to_p(1.0, -1.0, 1.0, gam1)
    pR = H_to_p(5.0,  1.0, 1.0, gam2)

    x, U0 = _shock_tube_ic(Nx, 1.0, 1.0, -1.0, pL,
                                0.0, 1.0,  1.0, pR,
                                gam1, gam2)

    results, labels = [], []
    for order, lbl in [(1, "1O"), (2, "2O"), (3, "3O")]:
        U, _ = run_simulation(U0.copy(), Nx, 1.0/Nx, t_end, gam1, gam2,
                              order=order, sigma=sigma)
        results.append(cons_to_prim(U, gam1, gam2))
        labels.append(lbl)

    _plot_result(x, results, labels,
                 f"Positivity of mass fraction (t={t_end}, Nx={Nx})",
                 "test_positivity.png")


# ─────────────────────────────────────────────
# EOC (Experimental Order of Convergence, Sec. 5.1)
# ─────────────────────────────────────────────

def run_test_eoc():
    """
    EOC test (Sec. 5.1, Tables 1-4).
    IC (Eq. 84): rho_j = 0.5 + 0.1*sin(pi*x), u=0.1, p=0.5, gam=1.4
    Exact (Eq. 85): rho(x,t) = 0.5 + 0.1*sin(pi*(x - 0.1*t))
    Domain [0,2], periodic BC.
    """
    print("\n[Test] Experimental Order of Convergence (Sec. 5.1)")
    gam1 = gam2 = 1.4
    t_end  = 0.5
    domain = 2.0
    Nx_list = [40, 80, 160, 320, 640, 1280]

    def exact_rho_avg(x, t, dx):
        """Cell-averaged exact density (Eq. 86, paper note after Eq. 89).
        Uses analytic integral: avg = 1 + 0.2*sin(π*(x-0.1t)) * sinc(dx/2)
        where sinc(h) = sin(π*h)/(π*h).
        """
        h = dx / 2.0
        sinc_h = np.sin(np.pi * h) / (np.pi * h)
        return 1.0 + 0.2 * np.sin(np.pi * (x - 0.1 * t)) * sinc_h

    label_map = {1: "1O", 2: "2O (limited)", 3: "3O (limited)", 4: "3O (unlimited)"}

    def run_order(order):
        print(f"\n  {label_map[order]}")
        print(f"  {'Nx':>6}  {'dx':>10}  {'L1 error':>14}  {'EOC_L1':>8}  "
              f"{'L2 error':>14}  {'EOC_L2':>8}")
        prev_e1, prev_e2 = None, None
        for Nx in Nx_list:
            dx = domain / Nx
            x  = (np.arange(Nx) + 0.5) * dx

            # Cell-averaged initial condition (Eq. 84 + paper note)
            rho0 = exact_rho_avg(x, 0.0, dx)
            W0   = 0.5 * np.ones(Nx)
            u0   = 0.1 * np.ones(Nx)
            p0   = 0.5 * np.ones(Nx)
            U0   = prim_to_cons(W0, rho0, u0, p0, gam1, gam2)

            U, _ = run_simulation(U0, Nx, dx, t_end, gam1, gam2,
                                  order=order, sigma=CFL, bc="periodic",
                                  max_steps=2_000_000)

            rho_num = U[1]
            rho_ex  = exact_rho_avg(x, t_end, dx)   # compare against cell-averaged exact

            e1 = dx * np.sum(np.abs(rho_num - rho_ex))
            e2 = np.sqrt(dx * np.sum((rho_num - rho_ex)**2))

            if prev_e1 is not None:
                eoc1 = np.log2(prev_e1 / e1)
                eoc2 = np.log2(prev_e2 / e2)
                print(f"  {Nx:>6}  {dx:>10.7f}  {e1:>14.10f}  {eoc1:>8.6f}  "
                      f"{e2:>14.10f}  {eoc2:>8.6f}")
            else:
                print(f"  {Nx:>6}  {dx:>10.7f}  {e1:>14.10f}  {'':>8}  "
                      f"{e2:>14.10f}  {'':>8}")
            prev_e1, prev_e2 = e1, e2

    for order in [1, 2, 3, 4]:
        run_order(order)


# ─────────────────────────────────────────────
# Simple exact Sod solver (single-component)
# ─────────────────────────────────────────────

def _exact_sod(x, t, rhoL, uL, pL, rhoR, uR, pR, gam=1.4):
    """
    Exact solution for the single-component Sod shock tube.
    Returns (rho, u, p) arrays at positions x and time t.
    Only valid for uL=uR=0 initial conditions.
    Uses the standard iterative Riemann solver.
    """
    from scipy.optimize import brentq

    aL = np.sqrt(gam * pL / rhoL)
    aR = np.sqrt(gam * pR / rhoR)
    gm1 = gam - 1.0
    gp1 = gam + 1.0

    def f_star(p_star):
        # Left rarefaction
        if p_star <= pL:
            fL = (2.0*aL/gm1) * ((p_star/pL)**((gam-1.0)/(2.0*gam)) - 1.0)
        else:
            AL = 2.0 / (gp1 * rhoL)
            BL = gm1 / gp1 * pL
            fL = (p_star - pL) * np.sqrt(AL / (p_star + BL))
        # Right shock / rarefaction
        if p_star <= pR:
            fR = (2.0*aR/gm1) * ((p_star/pR)**((gam-1.0)/(2.0*gam)) - 1.0)
        else:
            AR = 2.0 / (gp1 * rhoR)
            BR = gm1 / gp1 * pR
            fR = (p_star - pR) * np.sqrt(AR / (p_star + BR))
        return fL + fR + (uR - uL)

    p_lo, p_hi = 1e-6 * min(pL, pR), 20.0 * max(pL, pR)
    p_star = brentq(f_star, p_lo, p_hi, xtol=1e-12)

    # Star velocities
    AL = 2.0 / (gp1 * rhoL); BL = gm1/gp1 * pL
    if p_star <= pL:
        u_star = uL + (2.0*aL/gm1) * ((p_star/pL)**((gam-1.0)/(2.0*gam)) - 1.0)
    else:
        u_star = uL - (p_star - pL) * np.sqrt(AL / (p_star + BL))

    # Wave speeds
    if p_star <= pL:
        SHL = uL - aL
        STL = u_star - aL * (p_star/pL)**((gam-1.0)/(2.0*gam))
    else:
        SL  = uL - aL * np.sqrt(gp1/(2.0*gam) * p_star/pL + gm1/(2.0*gam))

    AR = 2.0 / (gp1 * rhoR); BR = gm1/gp1 * pR
    SR  = uR + aR * np.sqrt(gp1/(2.0*gam) * p_star/pR + gm1/(2.0*gam))
    S_contact = u_star

    # Star densities
    rho_starL = rhoL * (p_star/pL)**(1.0/gam) if p_star <= pL else \
                rhoL * (p_star/pL + gm1/gp1) / (gm1/gp1 * p_star/pL + 1.0)
    rho_starR = rhoR * (p_star/pR)**(1.0/gam) if p_star <= pR else \
                rhoR * (p_star/pR + gm1/gp1) / (gm1/gp1 * p_star/pR + 1.0)

    xi = (x - 0.5) / (t + 1e-30)  # similarity variable

    rho = np.empty_like(x)
    u_  = np.empty_like(x)
    p_  = np.empty_like(x)

    for k, xi_k in enumerate(xi):
        if p_star <= pL:    # left rarefaction
            if xi_k < SHL:
                rho[k], u_[k], p_[k] = rhoL, uL, pL
            elif xi_k < STL:
                a_fan = (2.0*aL + gm1*(uL - xi_k)) / gp1
                rho[k] = rhoL * (a_fan/aL)**(2.0/gm1)
                u_[k]  = (2.0*(aL + gm1/2.0*uL) + 2.0*xi_k) / gp1
                p_[k]  = pL * (a_fan/aL)**(2.0*gam/gm1)
            elif xi_k < S_contact:
                rho[k], u_[k], p_[k] = rho_starL, u_star, p_star
            elif xi_k < SR:
                rho[k], u_[k], p_[k] = rho_starR, u_star, p_star
            else:
                rho[k], u_[k], p_[k] = rhoR, uR, pR
        else:               # left shock
            if xi_k < SL:
                rho[k], u_[k], p_[k] = rhoL, uL, pL
            elif xi_k < S_contact:
                rho[k], u_[k], p_[k] = rho_starL, u_star, p_star
            elif xi_k < SR:
                rho[k], u_[k], p_[k] = rho_starR, u_star, p_star
            else:
                rho[k], u_[k], p_[k] = rhoR, uR, pR

    return rho, u_, p_


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

TESTS = {
    "steady_contact": run_test_steady_contact,
    "moving_same":    run_test_moving_same,
    "sod_same":       run_test_sod_same,
    "sod_diff":       run_test_sod_diff,
    "positivity":     run_test_positivity,
    "eoc":            run_test_eoc,
}


def main():
    parser = argparse.ArgumentParser(
        description="1D Multi-Component Euler — Kinetic Scheme (Roy & Raghurama Rao 2025)"
    )
    parser.add_argument(
        "--test",
        choices=list(TESTS.keys()) + ["all"],
        default="sod_same",
        help="Which test case to run (default: sod_same)",
    )
    parser.add_argument("--Nx",    type=int,   default=200,  help="Number of cells")
    parser.add_argument("--sigma", type=float, default=CFL,  help="CFL number")
    args = parser.parse_args()

    if args.test == "all":
        for name, fn in TESTS.items():
            if name == "eoc":
                fn()
            else:
                fn(Nx=args.Nx, sigma=args.sigma)
    elif args.test == "eoc":
        TESTS["eoc"]()
    else:
        TESTS[args.test](Nx=args.Nx, sigma=args.sigma)


if __name__ == "__main__":
    main()
