"""Forward-Euler explicit step for the v2 R6 solver — HLLC Riemann flux.

One time step:
  1) ghost-extended primitive W_ext (ng = 1)
  2) HLLC face flux F  via `flux_hllc.compute_face_flux_hllc`
  3) U^{n+1}_i = U^n_i − Δt/Δx · (F_{i+½} − F_{i−½})
  4) Allaire α-source: α₁ += Δt · α₁_cell · (S*_{i+½} − S*_{i−½}) / Δx
     where S* (HLLC contact wave speed) plays the role of the face velocity
     for the non-conservative correction.  When S* is unavailable in the
     current implementation we fall back to a central face velocity
     u_f = ½(u_L + u_R) — equivalent on smooth states.
  5) U → W via cons_to_prim_W.

No free parameters: dt is supplied by the caller.
"""
from __future__ import annotations
import numpy as np

from .boundary import extend_W, extend_U
from .flux_hllc import compute_face_flux_hllc
from .state import prim_to_cons, cons_to_prim


__all__ = ['euler_step', 'ssp_rk2_step']


def euler_step(W, dt, dx, eos1, eos2, bc_l, bc_r):
    """Advance W by one forward-Euler step (R6 HLLC face flux)."""
    alpha, T1, T2, u, p = (np.asarray(c, dtype=float) for c in W)
    N = alpha.shape[0]

    # 1) ghost-extended primitive
    W_ext = extend_W((alpha, T1, T2, u, p), bc_l, bc_r, ng=1)

    # 2) HLLC face flux  (5, Nf=N+1)
    F = compute_face_flux_hllc(W_ext, eos1, eos2)

    # face velocity for the α-source — central avg suffices on smooth data
    u_face = 0.5 * (W_ext[3][:-1] + W_ext[3][1:])

    # 3) conservative update for the four "true" conservative components
    U_n, _aux = prim_to_cons((alpha, T1, T2, u, p), eos1, eos2)
    inv_dx = 1.0 / dx
    U_next = list(U_n)
    for k in range(4):
        U_next[k] = U_n[k] - dt * inv_dx * (F[k, 1:] - F[k, :-1])

    # 4) Allaire non-conservative α-source
    div_u = (u_face[1:] - u_face[:-1]) * inv_dx
    alpha_advection = (F[4, 1:] - F[4, :-1]) * inv_dx
    U_next[4] = U_n[4] - dt * (alpha_advection - alpha * div_u)

    # 5) U → W
    W_next = cons_to_prim(tuple(U_next), eos1, eos2,
                          T1_init=T1, T2_init=T2)

    rho_next = U_next[0] + U_next[1]
    info = dict(
        rho_min=float(np.min(rho_next)),
        p_min=float(np.min(W_next[4])),
        alpha_min=float(np.min(W_next[0])),
        alpha_max=float(np.max(W_next[0])),
    )
    return W_next, info


def ssp_rk2_step(W, dt, dx, eos1, eos2, bc_l, bc_r):
    """Strong-Stability-Preserving RK2 (Gottlieb-Shu Heun method) on
    conservative U.

    SSP-RK2 (no free parameters):
        U^(1) = U^n + Δt · L(U^n)
        U^{n+1} = ½ U^n + ½ (U^(1) + Δt · L(U^(1)))
                 = ½ U^n + ½ U^(2)

    Implemented as two forward-Euler steps with the final result
    averaged on conservative variables (so the standard SSP property
    transfers to the conservative scalars α₁ρ₁, α₂ρ₂, ρu, ρE, α₁).
    """
    # Stage 1: U^(1) ≈ W_1
    W_1, _info1 = euler_step(W, dt, dx, eos1, eos2, bc_l, bc_r)
    # Stage 2: U^(2) ≈ W_2 (forward Euler from W^(1))
    W_2, info2 = euler_step(W_1, dt, dx, eos1, eos2, bc_l, bc_r)

    U_n, _aux = prim_to_cons(W, eos1, eos2)
    U_2, _aux = prim_to_cons(W_2, eos1, eos2)
    U_next = tuple(0.5 * U_n[k] + 0.5 * U_2[k] for k in range(5))

    W_next = cons_to_prim(tuple(U_next), eos1, eos2,
                          T1_init=W[1], T2_init=W[2])

    rho_next = U_next[0] + U_next[1]
    info = dict(
        rho_min=float(np.min(rho_next)),
        p_min=float(np.min(W_next[4])),
        alpha_min=float(np.min(W_next[0])),
        alpha_max=float(np.max(W_next[0])),
    )
    return W_next, info
