"""Primitive/conservative transforms for 2-D and 3-D five-equation states.

The 1-D clean-room solver uses W = (alpha1, T1, T2, u, p).  Multidimensional
validation needs vector velocity, so this module keeps the extension explicit:

    2-D W = (alpha1, T1, T2, ux, uy, p)
    3-D W = (alpha1, T1, T2, ux, uy, uz, p)

Conservative variables are:

    U = (alpha1*rho1, alpha2*rho2, rho*u_0, ..., rho*E, alpha1)
"""
from __future__ import annotations

import numpy as np

_EPS = 1.0e-12


def split_W_nd(W, dim: int):
    """Return alpha, T1, T2, velocity tuple, p from an ND primitive tuple."""
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim}")
    expected = dim + 4
    if len(W) != expected:
        raise ValueError(f"Expected W length {expected} for {dim}D, got {len(W)}")
    alpha = np.asarray(W[0], dtype=float)
    T1 = np.asarray(W[1], dtype=float)
    T2 = np.asarray(W[2], dtype=float)
    vel = tuple(np.asarray(W[3 + i], dtype=float) for i in range(dim))
    p = np.asarray(W[3 + dim], dtype=float)
    return alpha, T1, T2, vel, p


def mixture_density_from_W(W, eos1, eos2, dim: int, alpha_eps: float = _EPS):
    """Return rho, rho1, rho2 for an ND primitive tuple."""
    alpha, T1, T2, _, p = split_W_nd(W, dim)
    a = np.clip(alpha, alpha_eps, 1.0 - alpha_eps)
    rho1 = eos1.density(p, T1)
    rho2 = eos2.density(p, T2)
    rho = a * rho1 + (1.0 - a) * rho2
    return rho, rho1, rho2


def prim_to_cons_nd(W, eos1, eos2, *, dim: int, alpha_eps: float = _EPS):
    """Convert ND primitive tuple to conservative tuple."""
    alpha, T1, T2, vel, p = split_W_nd(W, dim)
    a = np.clip(alpha, alpha_eps, 1.0 - alpha_eps)
    a2 = 1.0 - a
    rho1 = eos1.density(p, T1)
    rho2 = eos2.density(p, T2)
    e1 = eos1.energy(rho1, p)
    e2 = eos2.energy(rho2, p)
    rho = a * rho1 + a2 * rho2
    kinetic = 0.5 * sum(v * v for v in vel)
    rhoE = a * rho1 * e1 + a2 * rho2 * e2 + rho * kinetic
    U = [a * rho1, a2 * rho2]
    U.extend(rho * v for v in vel)
    U.extend([rhoE, alpha])
    return tuple(np.asarray(q, dtype=float) for q in U)


def _pressure_from_internal_energy_fixed_mass(
    q1,
    q2,
    alpha,
    rhoe,
    eos1,
    eos2,
    p_seed,
    *,
    alpha_eps: float,
    p_floor: float,
    p_max_iter: int,
):
    """Recover common pressure from phase masses and mixture internal energy.

    For fixed alpha*rho_k and alpha, rho_k is fixed.  We solve:

        alpha*rho1*e1(rho1,p) + alpha2*rho2*e2(rho2,p) = rhoe.

    This scalar solve is the multidimensional analogue of the 1-D pressure
    recovery, kept local so the ND extension does not mutate legacy modules.
    """
    a = np.clip(alpha, alpha_eps, 1.0 - alpha_eps)
    a2 = 1.0 - a
    rho1 = np.maximum(q1 / a, _EPS)
    rho2 = np.maximum(q2 / a2, _EPS)
    p = np.maximum(np.asarray(p_seed, dtype=float), p_floor)
    target = np.asarray(rhoe, dtype=float)

    for _ in range(p_max_iter):
        e1 = eos1.energy(rho1, p)
        e2 = eos2.energy(rho2, p)
        f = a * rho1 * e1 + a2 * rho2 * e2 - target
        de1_dp = 1.0 / np.maximum(eos1.dpde_rho(rho1, e1), _EPS)
        de2_dp = 1.0 / np.maximum(eos2.dpde_rho(rho2, e2), _EPS)
        df = a * rho1 * de1_dp + a2 * rho2 * de2_dp
        dp = f / np.maximum(df, _EPS)
        p_new = np.maximum(p - dp, p_floor)
        if np.max(np.abs(dp) / np.maximum(np.abs(p_new), 1.0)) < 1.0e-11:
            p = p_new
            break
        p = p_new
    return p, rho1, rho2


def cons_to_prim_nd(
    U,
    eos1,
    eos2,
    *,
    dim: int,
    W_seed=None,
    alpha_eps: float = _EPS,
    p_floor: float = 1.0e-8,
    p_max_iter: int = 24,
):
    """Convert conservative ND tuple to primitive tuple.

    `W_seed` is used only as a Newton pressure seed.  If absent, a conservative
    ideal-gas-scale positive seed is used.
    """
    expected = dim + 4
    if len(U) != expected:
        raise ValueError(f"Expected U length {expected} for {dim}D, got {len(U)}")
    q1 = np.maximum(np.asarray(U[0], dtype=float), _EPS)
    q2 = np.maximum(np.asarray(U[1], dtype=float), _EPS)
    mom = tuple(np.asarray(U[2 + i], dtype=float) for i in range(dim))
    rhoE = np.asarray(U[2 + dim], dtype=float)
    alpha = np.clip(np.asarray(U[3 + dim], dtype=float), alpha_eps, 1.0 - alpha_eps)
    rho = np.maximum(q1 + q2, _EPS)
    vel = tuple(m / rho for m in mom)
    kinetic_rho = 0.5 * rho * sum(v * v for v in vel)
    rhoe = np.maximum(rhoE - kinetic_rho, _EPS)

    if W_seed is not None:
        p_seed = np.asarray(W_seed[3 + dim], dtype=float)
    else:
        p_seed = np.maximum(0.4 * rhoe, 1.0)

    p, rho1, rho2 = _pressure_from_internal_energy_fixed_mass(
        q1,
        q2,
        alpha,
        rhoe,
        eos1,
        eos2,
        p_seed,
        alpha_eps=alpha_eps,
        p_floor=p_floor,
        p_max_iter=p_max_iter,
    )
    e1 = eos1.energy(rho1, p)
    e2 = eos2.energy(rho2, p)
    T1 = eos1.temperature(rho1, e1)
    T2 = eos2.temperature(rho2, e2)
    return tuple([alpha, T1, T2] + list(vel) + [p])


def clip_cons_alpha(U, *, dim: int, alpha_eps: float = _EPS):
    """Clip volume fraction in conservative tuple while preserving phase masses."""
    Uc = [np.asarray(q, dtype=float).copy() for q in U]
    Uc[3 + dim] = np.clip(Uc[3 + dim], alpha_eps, 1.0 - alpha_eps)
    return tuple(Uc)
