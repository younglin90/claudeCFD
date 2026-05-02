"""Autodiff-Jacobian IMEX step for the explicit five-equation baseline.

The split is deliberately narrow:

1. Explicit material transport for q1, q2, momentum advection, and alpha, using
   the same CICSAM alpha face path as ``explicit.py``.
2. Implicit linear-acoustic solve for cell-centered (u, p).  The residual is
   differentiated with ``autograd.jacobian``; no hand-derived implicit Jacobian
   is assembled here.

The implicit acoustic coefficients (rho*c^2 and impedance) are frozen at the
stage anchor, which keeps the block robust for Ideal/SG/NASG and isolates the
experiment to the time splitting/Jacobian mechanism.
"""
from __future__ import annotations

import autograd.numpy as anp
from autograd import jacobian
import os
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import torch
from torch.func import jacrev, vmap

from .boundary import extend_W
from .energy_flux import total_energy_flux
from .explicit import _alpha_face, _phase_acoustic
from .primitive import prim_to_cons_W
from .reconstruction import (
    normalise_primitive_scheme,
    reconstruct_lr_faces,
    reconstruct_upwind_faces,
    reconstruct_primitive_upwind_faces,
)
from .source_d1 import D_K_kapila

_EPS = 1.0e-30


def _same_eos(eos1, eos2):
    keys = ("name", "gamma", "pinf", "kv", "b", "eta", "q")
    return all(
        abs(float(getattr(eos1, k, 0.0) if k != "name" else 0.0)
            - float(getattr(eos2, k, 0.0) if k != "name" else 0.0)) <= 1.0e-14
        for k in keys if k != "name"
    ) and getattr(eos1, "name", None) == getattr(eos2, "name", None)


def _recover_pressure_from_total_energy(q1, q2, rhoE, alpha, u, p_seed,
                                        eos1, eos2):
    """Recover pressure from conservative total energy at fixed phase masses.

    This is the conservative shock-capturing closure.  The old acoustic
    pressure equation is excellent for small-amplitude waves but does not
    enforce Rankine-Hugoniot shock speeds.  Here the common pressure is the
    scalar root of the mixture internal-energy constraint.
    """
    alpha_c = np.clip(np.asarray(alpha, dtype=float), 1.0e-12, 1.0 - 1.0e-12)
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    rhoE = np.asarray(rhoE, dtype=float)
    u = np.asarray(u, dtype=float)
    rho = np.maximum(q1 + q2, _EPS)
    rho1 = np.maximum(q1 / alpha_c, _EPS)
    rho2 = np.maximum(q2 / np.maximum(1.0 - alpha_c, 1.0e-12), _EPS)
    target_rhoe = rhoE - 0.5 * rho * u * u
    p = np.maximum(np.asarray(p_seed, dtype=float), 1.0e-12)
    for _ in range(16):
        e1 = eos1.energy(rho1, p)
        e2 = eos2.energy(rho2, p)
        f = alpha_c * rho1 * e1 + (1.0 - alpha_c) * rho2 * e2 - target_rhoe
        dpde1 = np.maximum(eos1.dpde_rho(rho1, e1), _EPS)
        dpde2 = np.maximum(eos2.dpde_rho(rho2, e2), _EPS)
        df = alpha_c * rho1 / dpde1 + (1.0 - alpha_c) * rho2 / dpde2
        step = f / np.maximum(df, _EPS)
        p_next = np.maximum(p - step, 1.0e-12)
        if float(np.max(np.abs(p_next - p) / np.maximum(p_next, 1.0))) < 1.0e-10:
            p = p_next
            break
        p = p_next
    return p


def _compressive_pressure_mask(W):
    """Return cells adjacent to non-small compressive pressure waves.

    Conservative total-energy pressure recovery is needed at shocks to enforce
    Rankine-Hugoniot coupling, but applying it to expansions/cavitation fans
    over-damps the pressure drop and leaves high-frequency ringing.  The
    recovery therefore follows the compression part of the pressure field:
    a resolved pressure jump must coincide with local velocity convergence.
    """
    p = np.asarray(W[4], dtype=float)
    u = np.asarray(W[3], dtype=float)
    if p.size < 2 or not np.all(np.isfinite(p)) or not np.all(np.isfinite(u)):
        return np.zeros_like(p, dtype=bool)
    denom = np.maximum(np.maximum(np.abs(p[:-1]), np.abs(p[1:])), 1.0)
    rel_jump = np.abs(p[1:] - p[:-1]) / denom
    compression = u[:-1] > u[1:]
    face_mask = (rel_jump > np.finfo(float).eps ** 0.25) & compression
    cell_mask = np.zeros_like(p, dtype=bool)
    cell_mask[:-1] |= face_mask
    cell_mask[1:] |= face_mask
    return cell_mask


def _has_resolved_pressure_wave(W):
    """Compatibility wrapper for older diagnostics."""
    return bool(np.any(_compressive_pressure_mask(W)))


def _pure_material_cell_mask(alpha, pure_tol):
    """Cells adjacent to a material jump with at least one pure-side state."""
    alpha = np.asarray(alpha, dtype=float)
    if alpha.size < 2:
        return np.zeros_like(alpha, dtype=bool)
    jump_tol = np.finfo(float).eps ** 0.25
    a_L = alpha[:-1]
    a_R = alpha[1:]
    alpha_jump = np.abs(a_R - a_L)
    pure_face = (
        (alpha_jump > jump_tol)
        & (
            (a_L <= pure_tol) | (a_L >= 1.0 - pure_tol)
            | (a_R <= pure_tol) | (a_R >= 1.0 - pure_tol)
        )
    )
    cells = np.zeros_like(alpha, dtype=bool)
    cells[:-1] |= pure_face
    cells[1:] |= pure_face
    return cells


def _regularize_near_vacuum_velocity(W_n, q1_new, q2_new, u_new, p_new,
                                      eos1, eos2, *,
                                      mixture_kind, alpha_pure_tol,
                                      bc_l, bc_r, passes=6):
    """Smooth primitive velocity only where cavitation makes it ill-defined.

    In a double-rarefaction/vacuum state, density and pressure can collapse by
    orders of magnitude while the remaining conservative momentum is tiny.  The
    primitive ratio ``u = m/rho`` then creates a visually sharp sign jump that
    is not a meaningful physical velocity inside the near-vacuum pocket.  Limit
    the correction to expanding cells with both density and pressure below 1%
    of the stage anchor, so shocks and ordinary material interfaces are left
    untouched.
    """
    u = np.asarray(u_new, dtype=float)
    p = np.asarray(p_new, dtype=float)
    rho = np.maximum(np.asarray(q1_new, dtype=float) + np.asarray(q2_new, dtype=float), _EPS)
    rho_anchor, _, _ = _phase_acoustic(
        W_n, eos1, eos2, mixture_kind=mixture_kind,
        alpha_pure_tol=alpha_pure_tol)
    rho_anchor = np.maximum(np.asarray(rho_anchor, dtype=float), _EPS)
    p_anchor = np.maximum(np.abs(np.asarray(W_n[4], dtype=float)), 1.0)
    rho_domain = max(float(np.max(rho_anchor)), 1.0)
    p_domain = max(float(np.max(p_anchor)), 1.0)

    u_ext = _extend_np(u, bc_l, bc_r, odd=True)
    expanding_face = np.diff(u_ext) > 0.0
    expanding_cell = expanding_face[:-1] | expanding_face[1:]
    density_collapse = (rho < 1.0e-2 * rho_anchor) | (rho < 1.0e-3 * rho_domain)
    pressure_collapse = (p < 1.0e-2 * p_anchor) | (p < 1.0e-3 * p_domain)
    low_pressure_vacuum = p < min(5.0e-2 * p_domain, 5.0e3)
    mask = expanding_cell & (
        (density_collapse & pressure_collapse)
        | low_pressure_vacuum
    )
    if not np.any(mask):
        return u, mask

    u_reg = u.copy()
    for _ in range(int(max(passes, 1))):
        u_ext = _extend_np(u_reg, bc_l, bc_r, odd=True)
        smooth = 0.25 * u_ext[:-2] + 0.5 * u_ext[1:-1] + 0.25 * u_ext[2:]
        u_next = u_reg.copy()
        u_next[mask] = smooth[mask]
        u_reg = u_next
    return u_reg, mask


def _normalise_pressure_closure(pressure_closure):
    key = str(pressure_closure or "regime_auto").strip().lower()
    aliases = {
        "current": "compressive_recovery",
        "baseline": "compressive_recovery",
        "none": "no_recovery",
        "off": "no_recovery",
        "no": "no_recovery",
        "pressure_work": "pressure_work_consistent",
        "path": "path_kapila",
        "path_conservative": "path_kapila",
        "entropy": "dual_entropy",
        "dual_energy": "dual_entropy",
        "apec": "apec_pe",
        "pe": "apec_pe",
        "auto": "regime_auto",
        "auto_regime": "regime_auto",
    }
    key = aliases.get(key, key)
    allowed = {
        "compressive_recovery",
        "no_recovery",
        "implicit_energy",
        "pressure_work_consistent",
        "path_kapila",
        "dual_entropy",
        "apec_pe",
        "regime_auto",
    }
    if key not in allowed:
        raise ValueError(f"Unknown pressure_closure='{pressure_closure}'.")
    return key


def _acoustic_faces_np(u, p, Z, bc_l, bc_r, *, u_inlet=None, p_inlet=None):
    """Return acoustic Riemann face pressure and velocity for cell arrays."""
    u = np.asarray(u, dtype=float)
    p = np.asarray(p, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if bc_l == 'periodic' and bc_r == 'periodic':
        u_ext = np.concatenate(([u[-1]], u, [u[0]]))
        p_ext = np.concatenate(([p[-1]], p, [p[0]]))
        Z_ext = np.concatenate(([Z[-1]], Z, [Z[0]]))
    else:
        if bc_l == 'reflective':
            u_left = -u[0]
            p_left = p[0]
        elif bc_l in ('inlet', 'inlet_acoustic', 'dirichlet'):
            u_left = float(u_inlet) if u_inlet is not None else u[0]
            p_left = float(p_inlet) if p_inlet is not None else p[0]
        else:
            u_left = u[0]
            p_left = p[0]
        if bc_r == 'reflective':
            u_right = -u[-1]
            p_right = p[-1]
        else:
            u_right = u[-1]
            p_right = p[-1]
        u_ext = np.concatenate(([u_left], u, [u_right]))
        p_ext = np.concatenate(([p_left], p, [p_right]))
        Z_ext = np.concatenate(([Z[0]], Z, [Z[-1]]))

    Z_L = Z_ext[:-1]
    Z_R = Z_ext[1:]
    p_L = p_ext[:-1]
    p_R = p_ext[1:]
    u_L = u_ext[:-1]
    u_R = u_ext[1:]
    den = np.maximum(Z_L + Z_R, _EPS)
    p_star = (Z_R * p_L + Z_L * p_R + Z_L * Z_R * (u_L - u_R)) / den
    u_star = (p_L - p_R + Z_L * u_L + Z_R * u_R) / den
    if bc_l == 'reflective':
        p_star[0] = p[0]
        u_star[0] = 0.0
    if bc_r == 'reflective':
        p_star[-1] = p[-1]
        u_star[-1] = 0.0
    return p_star, u_star


def _pure_bulk_muscl_face_mask(alpha, bc_l, bc_r, alpha_pure_tol):
    n = len(alpha)
    high = np.zeros(n + 1, dtype=bool)
    if alpha_pure_tol <= 0.0:
        return high
    alpha_ext = _extend_np(np.asarray(alpha, dtype=float), bc_l, bc_r, odd=False)
    pure_tol = max(float(alpha_pure_tol), np.finfo(float).eps ** 0.25)
    a_L = alpha_ext[:-1]
    a_R = alpha_ext[1:]
    pure_L = (a_L >= 1.0 - pure_tol) | (a_L <= pure_tol)
    pure_R = (a_R >= 1.0 - pure_tol) | (a_R <= pure_tol)
    high[:] = pure_L & pure_R
    high[0] = False
    high[-1] = False
    if n + 1 > 3 and not (bc_l == 'periodic' and bc_r == 'periodic'):
        high[1] = False
        high[-2] = False
    return high


def _same_pure_material_face_mask(alpha, bc_l, bc_r, alpha_pure_tol):
    """Faces whose adjacent states are the same resolved pure material."""
    n = len(alpha)
    high = np.zeros(n + 1, dtype=bool)
    if alpha_pure_tol <= 0.0:
        return high
    alpha_ext = _extend_np(np.asarray(alpha, dtype=float), bc_l, bc_r, odd=False)
    pure_tol = max(float(alpha_pure_tol), np.finfo(float).eps ** 0.25)
    a_L = alpha_ext[:-1]
    a_R = alpha_ext[1:]
    phase1 = (a_L >= 1.0 - pure_tol) & (a_R >= 1.0 - pure_tol)
    phase2 = (a_L <= pure_tol) & (a_R <= pure_tol)
    high[:] = phase1 | phase2
    high[0] = False
    high[-1] = False
    if n + 1 > 3 and not (bc_l == 'periodic' and bc_r == 'periodic'):
        high[1] = False
        high[-2] = False
    return high


def _acoustic_faces_muscl_np(u, p, Z, alpha, bc_l, bc_r, alpha_pure_tol, *,
                             u_inlet=None, p_inlet=None,
                             primitive_scheme='upwind'):
    p_star, u_star = _acoustic_faces_np(
        u, p, Z, bc_l, bc_r, u_inlet=u_inlet, p_inlet=p_inlet)
    primitive_scheme = normalise_primitive_scheme(primitive_scheme)
    if (primitive_scheme == 'upwind'
            or os.environ.get("FIVE_EQ_IMEX_ACOUSTIC_MUSCL", "0") == "0"):
        return p_star, u_star, np.zeros(len(u) + 1, dtype=bool)
    high_face = _pure_bulk_muscl_face_mask(alpha, bc_l, bc_r, alpha_pure_tol)
    if not np.any(high_face):
        return p_star, u_star, high_face
    tvd_kind = os.environ.get("FIVE_EQ_IMEX_TMLPU_TVD", "minmod")
    u = np.asarray(u, dtype=float)
    p = np.asarray(p, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if bc_l == 'periodic' and bc_r == 'periodic':
        u_ext = np.concatenate(([u[-1]], u, [u[0]]))
        p_ext = np.concatenate(([p[-1]], p, [p[0]]))
        Z_ext = np.concatenate(([Z[-1]], Z, [Z[0]]))
    else:
        if bc_l == 'reflective':
            u_left = -u[0]
            p_left = p[0]
        elif bc_l in ('inlet', 'inlet_acoustic', 'dirichlet'):
            u_left = float(u_inlet) if u_inlet is not None else u[0]
            p_left = float(p_inlet) if p_inlet is not None else p[0]
        else:
            u_left = u[0]
            p_left = p[0]
        if bc_r == 'reflective':
            u_right = -u[-1]
            p_right = p[-1]
        else:
            u_right = u[-1]
            p_right = p[-1]
        u_ext = np.concatenate(([u_left], u, [u_right]))
        p_ext = np.concatenate(([p_left], p, [p_right]))
        Z_ext = np.concatenate(([Z[0]], Z, [Z[-1]]))

    for f in np.flatnonzero(high_face):
        if f <= 0 or f + 2 >= len(p_ext):
            continue
        sp_L = _tvd_pair(p_ext[f] - p_ext[f - 1], p_ext[f + 1] - p_ext[f], tvd_kind)
        sp_R = _tvd_pair(p_ext[f + 1] - p_ext[f], p_ext[f + 2] - p_ext[f + 1], tvd_kind)
        su_L = _tvd_pair(u_ext[f] - u_ext[f - 1], u_ext[f + 1] - u_ext[f], tvd_kind)
        su_R = _tvd_pair(u_ext[f + 1] - u_ext[f], u_ext[f + 2] - u_ext[f + 1], tvd_kind)
        p_L = p_ext[f] + 0.5 * float(sp_L)
        p_R = p_ext[f + 1] - 0.5 * float(sp_R)
        u_L = u_ext[f] + 0.5 * float(su_L)
        u_R = u_ext[f + 1] - 0.5 * float(su_R)
        Z_L = Z_ext[f]
        Z_R = Z_ext[f + 1]
        den = max(float(Z_L + Z_R), _EPS)
        p_star[f] = (Z_R * p_L + Z_L * p_R + Z_L * Z_R * (u_L - u_R)) / den
        u_star[f] = (p_L - p_R + Z_L * u_L + Z_R * u_R) / den
    return p_star, u_star, high_face


def _face_energy_dict(W_ext, p_star, u_star, upwind_left, alpha_f,
                      rho1_f, rho2_f, eos1, eos2):
    """Build the minimal face dictionary consumed by energy_flux.py."""
    alpha_ext, T1_ext, T2_ext, u_ext, _ = W_ext
    T1_f = np.where(upwind_left, T1_ext[:-1], T1_ext[1:])
    T2_f = np.where(upwind_left, T2_ext[:-1], T2_ext[1:])
    e1_f = eos1.energy(rho1_f, p_star)
    e2_f = eos2.energy(rho2_f, p_star)
    rho_f = alpha_f * rho1_f + (1.0 - alpha_f) * rho2_f
    return {
        "alpha": alpha_f,
        "p": p_star,
        "a_L": alpha_ext[:-1],
        "a_R": alpha_ext[1:],
        "rho1": rho1_f,
        "rho2": rho2_f,
        "rho1_L": eos1.density(p_star, T1_ext[:-1]),
        "rho1_R": eos1.density(p_star, T1_ext[1:]),
        "rho2_L": eos2.density(p_star, T2_ext[:-1]),
        "rho2_R": eos2.density(p_star, T2_ext[1:]),
        "T1": T1_f,
        "T2": T2_f,
        "e1": e1_f,
        "e2": e2_f,
        "rho": rho_f,
        "u": u_star,
    }


def _single_phase_hllc_flux(U_L, F_L, rho_L, u_L, p_L,
                            U_R, F_R, rho_R, u_R, p_R, s_L, s_R):
    """HLLC flux for the conservative single-phase Euler shortcut.

    The formula only uses conservative states, pressure, and estimated wave
    speeds, so it applies to the pure-phase EOS handled by the facade.  Faces
    that become numerically inadmissible are replaced by the more dissipative
    Rusanov flux at the caller.
    """
    denom = rho_L * (s_L - u_L) - rho_R * (s_R - u_R)
    denom = np.where(np.abs(denom) < 1.0e-14,
                     np.sign(denom + 1.0e-300) * 1.0e-14,
                     denom)
    s_M = (
        p_R - p_L
        + rho_L * u_L * (s_L - u_L)
        - rho_R * u_R * (s_R - u_R)
    ) / denom

    d_L = np.where(np.abs(s_L - s_M) < 1.0e-14,
                   np.sign(s_L - s_M + 1.0e-300) * 1.0e-14,
                   s_L - s_M)
    d_R = np.where(np.abs(s_R - s_M) < 1.0e-14,
                   np.sign(s_R - s_M + 1.0e-300) * 1.0e-14,
                   s_R - s_M)
    rho_star_L = rho_L * (s_L - u_L) / d_L
    rho_star_R = rho_R * (s_R - u_R) / d_R

    pd_L = rho_L * (s_L - u_L)
    pd_R = rho_R * (s_R - u_R)
    pd_L = np.where(np.abs(pd_L) < 1.0e-14,
                    np.sign(pd_L + 1.0e-300) * 1.0e-14,
                    pd_L)
    pd_R = np.where(np.abs(pd_R) < 1.0e-14,
                    np.sign(pd_R + 1.0e-300) * 1.0e-14,
                    pd_R)
    E_star_L = rho_star_L * (
        U_L[2] / rho_L
        + (s_M - u_L) * (s_M + p_L / pd_L)
    )
    E_star_R = rho_star_R * (
        U_R[2] / rho_R
        + (s_M - u_R) * (s_M + p_R / pd_R)
    )
    U_star_L = np.array([rho_star_L, rho_star_L * s_M, E_star_L])
    U_star_R = np.array([rho_star_R, rho_star_R * s_M, E_star_R])

    flux = np.empty_like(F_L)
    mask_L = 0.0 <= s_L
    mask_star_L = (s_L <= 0.0) & (0.0 <= s_M)
    mask_star_R = (s_M <= 0.0) & (0.0 <= s_R)
    mask_R = s_R <= 0.0
    flux[:, mask_L] = F_L[:, mask_L]
    flux[:, mask_star_L] = F_L[:, mask_star_L] + s_L[mask_star_L] * (
        U_star_L[:, mask_star_L] - U_L[:, mask_star_L])
    flux[:, mask_star_R] = F_R[:, mask_star_R] + s_R[mask_star_R] * (
        U_star_R[:, mask_star_R] - U_R[:, mask_star_R])
    flux[:, mask_R] = F_R[:, mask_R]
    unresolved = ~(mask_L | mask_star_L | mask_star_R | mask_R)
    if np.any(unresolved):
        s_max = np.maximum(np.abs(s_L[unresolved]), np.abs(s_R[unresolved]))
        flux[:, unresolved] = 0.5 * (F_L[:, unresolved] + F_R[:, unresolved]) - 0.5 * s_max * (
            U_R[:, unresolved] - U_L[:, unresolved])
    return flux


def _minmod_pair(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    same = (a * b) > 0.0
    return np.where(same, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def _mc_pair(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    same = (a * b) > 0.0
    centered = 0.5 * (a + b)
    limited = np.minimum(np.minimum(2.0 * np.abs(a), 2.0 * np.abs(b)), np.abs(centered))
    return np.where(same, np.sign(centered) * limited, 0.0)


def _vanleer_pair(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    same = (a * b) > 0.0
    den = a + b
    den_safe = np.where(np.abs(den) > _EPS, den, 1.0)
    slope = 2.0 * a * b / den_safe
    return np.where(same, slope, 0.0)


def _superbee_pair(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    same = (a * b) > 0.0
    s = np.sign(a)
    cand1 = np.minimum(2.0 * np.abs(a), np.abs(b))
    cand2 = np.minimum(np.abs(a), 2.0 * np.abs(b))
    return np.where(same, s * np.maximum(cand1, cand2), 0.0)


def _tvd_pair(a, b, kind):
    key = str(kind or "minmod").strip().lower().replace("-", "_")
    if key in ("superbee", "sb"):
        return _superbee_pair(a, b)
    if key in ("mc", "monotonized_central"):
        return _mc_pair(a, b)
    if key in ("vanleer", "van_leer"):
        return _vanleer_pair(a, b)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    same = (a * b) > 0.0
    return np.where(same, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def _single_phase_euler_rusanov_step(W_n, dt, eos, dx, bc_l, bc_r, *,
                                     u_inlet=None, p_inlet=None,
                                     primitive_scheme='upwind'):
    """Conservative Euler limit for identical-EOS, constant-alpha states."""
    primitive_scheme = normalise_primitive_scheme(primitive_scheme)
    alpha, T1, T2, u, p = W_n
    rho1 = eos.density(p, T1)
    rho2 = eos.density(p, T2)
    rho = np.maximum(alpha * rho1 + (1.0 - alpha) * rho2, _EPS)
    e1 = eos.energy(rho1, p)
    e2 = eos.energy(rho2, p)
    rhoe = alpha * rho1 * e1 + (1.0 - alpha) * rho2 * e2
    U = np.array([rho, rho * u, rhoe + 0.5 * rho * u * u])

    def ext(phi, odd=False, dirichlet_l=None):
        if bc_l == 'periodic' and bc_r == 'periodic':
            return np.concatenate(([phi[-1]], phi, [phi[0]]))
        if bc_l in ('inlet', 'inlet_acoustic', 'dirichlet') and dirichlet_l is not None:
            left = float(dirichlet_l)
        else:
            left = -phi[0] if (odd and bc_l == 'reflective') else phi[0]
        right = -phi[-1] if (odd and bc_r == 'reflective') else phi[-1]
        return np.concatenate(([left], phi, [right]))

    T_e = ext(T1)
    u_e = ext(u, odd=True, dirichlet_l=u_inlet)
    p_e = ext(p, dirichlet_l=p_inlet)
    if primitive_scheme in ('upwind', 'tmlpu', 'weno3'):
        # Pure-phase acoustics use the conservative Euler shortcut.  In this
        # reduced system, the stable high-order primitive reconstruction is the
        # MUSCL-Hancock TVD predictor below; using the multidimensional T-MLP-u
        # face interpolation without the half-step predictor over-amplifies
        # monochromatic acoustic inlet waves.
        rho_e = np.maximum(eos.density(p_e, T_e), _EPS)
        drho = np.zeros_like(rho_e)
        du = np.zeros_like(u_e)
        dp = np.zeros_like(p_e)
        drho[1:-1] = _minmod_pair(rho_e[1:-1] - rho_e[:-2], rho_e[2:] - rho_e[1:-1])
        du[1:-1] = _minmod_pair(u_e[1:-1] - u_e[:-2], u_e[2:] - u_e[1:-1])
        dp[1:-1] = _minmod_pair(p_e[1:-1] - p_e[:-2], p_e[2:] - p_e[1:-1])
        rho_x = drho / dx
        u_x = du / dx
        p_x = dp / dx
        e_e = eos.energy(rho_e, p_e)
        c_sq_e = np.maximum(eos.sound_speed_sq(rho_e, e_e, p_e), _EPS)

        # MUSCL-Hancock primitive predictor for smooth pure-phase acoustics.
        # The 1/2 factor is fixed by the second-order Taylor half-step.
        rho_t = -u_e * rho_x - rho_e * u_x
        u_t = -u_e * u_x - p_x / rho_e
        p_t = -u_e * p_x - rho_e * c_sq_e * u_x
        rho_L = rho_e[:-1] + 0.5 * drho[:-1] + 0.5 * dt * rho_t[:-1]
        rho_R = rho_e[1:] - 0.5 * drho[1:] + 0.5 * dt * rho_t[1:]
        u_L_h = u_e[:-1] + 0.5 * du[:-1] + 0.5 * dt * u_t[:-1]
        u_R_h = u_e[1:] - 0.5 * du[1:] + 0.5 * dt * u_t[1:]
        p_L_h = p_e[:-1] + 0.5 * dp[:-1] + 0.5 * dt * p_t[:-1]
        p_R_h = p_e[1:] - 0.5 * dp[1:] + 0.5 * dt * p_t[1:]

        # Boundary states are imposed directly; reconstruct only from owned cells.
        if not (bc_l == 'periodic' and bc_r == 'periodic'):
            rho_L[0] = rho_e[0]
            u_L_h[0] = u_e[0]
            p_L_h[0] = p_e[0]
            rho_R[-1] = rho_e[-1]
            u_R_h[-1] = u_e[-1]
            p_R_h[-1] = p_e[-1]

        invalid = (
            (~np.isfinite(rho_L)) | (~np.isfinite(rho_R))
            | (~np.isfinite(u_L_h)) | (~np.isfinite(u_R_h))
            | (~np.isfinite(p_L_h)) | (~np.isfinite(p_R_h))
            | (rho_L <= _EPS) | (rho_R <= _EPS)
            | (p_L_h <= _EPS) | (p_R_h <= _EPS)
        )
        rho_L_h = np.where(invalid, rho_e[:-1], rho_L)
        rho_R_h = np.where(invalid, rho_e[1:], rho_R)
        u_L_h = np.where(invalid, u_e[:-1], u_L_h)
        u_R_h = np.where(invalid, u_e[1:], u_R_h)
        p_L_h = np.where(invalid, p_e[:-1], p_L_h)
        p_R_h = np.where(invalid, p_e[1:], p_R_h)
        e_L_h = eos.energy(rho_L_h, p_L_h)
        e_R_h = eos.energy(rho_R_h, p_R_h)
        rhoE_L_h = rho_L_h * e_L_h + 0.5 * rho_L_h * u_L_h * u_L_h
        rhoE_R_h = rho_R_h * e_R_h + 0.5 * rho_R_h * u_R_h * u_R_h
        U_L = np.array([rho_L_h, rho_L_h * u_L_h, rhoE_L_h])
        U_R = np.array([rho_R_h, rho_R_h * u_R_h, rhoE_R_h])
        F_L = np.array([
            rho_L_h * u_L_h,
            rho_L_h * u_L_h * u_L_h + p_L_h,
            (rhoE_L_h + p_L_h) * u_L_h,
        ])
        F_R = np.array([
            rho_R_h * u_R_h,
            rho_R_h * u_R_h * u_R_h + p_R_h,
            (rhoE_R_h + p_R_h) * u_R_h,
        ])
        c_L = np.sqrt(np.maximum(eos.sound_speed_sq(rho_L_h, e_L_h, p_L_h), _EPS))
        c_R = np.sqrt(np.maximum(eos.sound_speed_sq(rho_R_h, e_R_h, p_R_h), _EPS))
    else:
        T_L_h, T_R_h = reconstruct_lr_faces(T_e, scheme=primitive_scheme, floor=1.0)
        u_L_h, u_R_h = reconstruct_lr_faces(u_e, scheme=primitive_scheme)
        p_L_h, p_R_h = reconstruct_lr_faces(p_e, scheme=primitive_scheme, floor=1.0e-12)
        rho_L_h = np.maximum(eos.density(p_L_h, T_L_h), _EPS)
        rho_R_h = np.maximum(eos.density(p_R_h, T_R_h), _EPS)
        e_L_h = eos.energy(rho_L_h, p_L_h)
        e_R_h = eos.energy(rho_R_h, p_R_h)
        rhoE_L_h = rho_L_h * e_L_h + 0.5 * rho_L_h * u_L_h * u_L_h
        rhoE_R_h = rho_R_h * e_R_h + 0.5 * rho_R_h * u_R_h * u_R_h
        U_L = np.array([rho_L_h, rho_L_h * u_L_h, rhoE_L_h])
        U_R = np.array([rho_R_h, rho_R_h * u_R_h, rhoE_R_h])
        F_L = np.array([
            rho_L_h * u_L_h,
            rho_L_h * u_L_h * u_L_h + p_L_h,
            (rhoE_L_h + p_L_h) * u_L_h,
        ])
        F_R = np.array([
            rho_R_h * u_R_h,
            rho_R_h * u_R_h * u_R_h + p_R_h,
            (rhoE_R_h + p_R_h) * u_R_h,
        ])
        c_L = np.sqrt(np.maximum(eos.sound_speed_sq(rho_L_h, e_L_h, p_L_h), _EPS))
        c_R = np.sqrt(np.maximum(eos.sound_speed_sq(rho_R_h, e_R_h, p_R_h), _EPS))
    s_rusanov = np.maximum(np.abs(u_L_h) + c_L, np.abs(u_R_h) + c_R)
    F_rusanov = 0.5 * (F_L + F_R) - 0.5 * s_rusanov * (U_R - U_L)
    s_L = np.minimum(u_L_h - c_L, u_R_h - c_R)
    s_R = np.maximum(u_L_h + c_L, u_R_h + c_R)
    F_face = _single_phase_hllc_flux(
        U_L, F_L, rho_L_h, u_L_h, p_L_h,
        U_R, F_R, rho_R_h, u_R_h, p_R_h,
        s_L, s_R)
    bad_face = ~np.all(np.isfinite(F_face), axis=0)
    if np.any(bad_face):
        F_face[:, bad_face] = F_rusanov[:, bad_face]
    if bc_l == 'reflective':
        F_face[:, 0] = np.array([0.0, p[0], 0.0])
    if bc_r == 'reflective':
        F_face[:, -1] = np.array([0.0, p[-1], 0.0])
    U_new = U - dt * (F_face[:, 1:] - F_face[:, :-1]) / dx
    rho_new = np.maximum(U_new[0], _EPS)
    u_new = U_new[1] / rho_new
    e_new = U_new[2] / rho_new - 0.5 * u_new * u_new
    p_new = eos.pressure(rho_new, e_new)
    T_new = eos.temperature(rho_new, e_new)
    if (not np.all(np.isfinite(p_new))) or np.min(p_new) <= 0.0 or (
            not np.all(np.isfinite(T_new))) or np.min(T_new) <= 0.0:
        U_new = U - dt * (F_rusanov[:, 1:] - F_rusanov[:, :-1]) / dx
        rho_new = np.maximum(U_new[0], _EPS)
        u_new = U_new[1] / rho_new
        e_new = U_new[2] / rho_new - 0.5 * u_new * u_new
        p_new = eos.pressure(rho_new, e_new)
        T_new = eos.temperature(rho_new, e_new)
    W_new = (alpha.copy(), T_new, T_new.copy(), u_new, p_new)
    return W_new, {
        'scheme': 'single_phase_conservative_euler_hllc',
        'primitive_scheme': primitive_scheme,
        'mass': float(np.sum(rho_new)),
    }


def _extend_np(phi, bc_l, bc_r, *, odd=False):
    if bc_l == 'periodic' and bc_r == 'periodic':
        return np.concatenate(([phi[-1]], phi, [phi[0]]))
    left = -phi[0] if (odd and bc_l == 'reflective') else phi[0]
    right = -phi[-1] if (odd and bc_r == 'reflective') else phi[-1]
    return np.concatenate(([left], phi, [right]))


def _extend_ag(phi, bc_l, bc_r, *, odd=False):
    if bc_l == 'periodic' and bc_r == 'periodic':
        return anp.concatenate((phi[-1:], phi, phi[:1]))
    left = -phi[:1] if (odd and bc_l == 'reflective') else phi[:1]
    right = -phi[-1:] if (odd and bc_r == 'reflective') else phi[-1:]
    return anp.concatenate((left, phi, right))


def _acoustic_faces_ag(u, p, Z, bc_l, bc_r):
    u_ext = _extend_ag(u, bc_l, bc_r, odd=True)
    p_ext = _extend_ag(p, bc_l, bc_r, odd=False)
    Z_ext = anp.asarray(_extend_np(np.asarray(Z, dtype=float), bc_l, bc_r, odd=False))
    p_L = p_ext[:-1]
    p_R = p_ext[1:]
    u_L = u_ext[:-1]
    u_R = u_ext[1:]
    Z_L = Z_ext[:-1]
    Z_R = Z_ext[1:]
    den = anp.maximum(Z_L + Z_R, _EPS)
    p_star = (Z_R * p_L + Z_L * p_R + Z_L * Z_R * (u_L - u_R)) / den
    u_star = (p_L - p_R + Z_L * u_L + Z_R * u_R) / den
    if bc_l == 'reflective':
        p_star = anp.concatenate((p[:1], p_star[1:]))
        u_star = anp.concatenate((anp.zeros(1), u_star[1:]))
    if bc_r == 'reflective':
        p_star = anp.concatenate((p_star[:-1], p[-1:]))
        u_star = anp.concatenate((u_star[:-1], anp.zeros(1)))
    return p_star, u_star


def _material_update(W_n, dt, eos1, eos2, dx, bc_l, bc_r, *,
                     u_inlet=None, p_inlet=None,
                     mixture_kind, kapila_closure, alpha_pure_tol,
                     alpha_scheme, primitive_scheme='upwind',
                     kapila_source_mode='hybrid',
                     material_energy_form='allaire', return_aux=False):
    primitive_scheme = normalise_primitive_scheme(primitive_scheme)
    U_n, _ = prim_to_cons_W(W_n, eos1, eos2)
    W_ext = extend_W(W_n, bc_l, bc_r, ng=1,
                     u_inlet_l=u_inlet, p_inlet_l=p_inlet,
                     eos1=eos1, eos2=eos2)
    U_ext, _ = prim_to_cons_W(W_ext, eos1, eos2)
    _, c_mix_sq_ext, Z_ext = _phase_acoustic(
        W_ext, eos1, eos2, mixture_kind=mixture_kind,
        alpha_pure_tol=alpha_pure_tol)

    _, _, _, u_ext, p_ext = W_ext
    p_L = p_ext[:-1]
    p_R = p_ext[1:]
    u_L = u_ext[:-1]
    u_R = u_ext[1:]
    Z_L = Z_ext[:-1]
    Z_R = Z_ext[1:]
    den = np.maximum(Z_L + Z_R, _EPS)
    p_star = (Z_R * p_L + Z_L * p_R + Z_L * Z_R * (u_L - u_R)) / den
    u_star = (p_L - p_R + Z_L * u_L + Z_R * u_R) / den

    if bc_l == 'reflective':
        u_star[0] = 0.0
    if bc_r == 'reflective':
        u_star[-1] = 0.0

    upwind_left = u_star >= 0.0
    prim_f = reconstruct_primitive_upwind_faces(
        W_ext, u_star, scheme=primitive_scheme)
    T1_f = prim_f["T1"]
    T2_f = prim_f["T2"]
    u_adv_f = prim_f["u"]
    p_adv_f = prim_f["p"]
    rho1_f = np.maximum(eos1.density(p_adv_f, T1_f), _EPS)
    rho2_f = np.maximum(eos2.density(p_adv_f, T2_f), _EPS)
    if os.environ.get("FIVE_EQ_IMEX_DENSITY_RECON", "1") == "1":
        # Reconstruct EOS-consistent phase densities directly for the material
        # flux.  Reconstructing p and T independently can perturb rho=p/T on
        # pressure-equilibrium plateaus; rho is the conservative thermodynamic
        # variable for phase-mass advection, while p still defines the pressure
        # work and phase energies.
        rho1_ext = np.maximum(eos1.density(W_ext[4], W_ext[1]), _EPS)
        rho2_ext = np.maximum(eos2.density(W_ext[4], W_ext[2]), _EPS)
        rho1_f = reconstruct_upwind_faces(
            rho1_ext, u_star, scheme=primitive_scheme, floor=_EPS)
        rho2_f = reconstruct_upwind_faces(
            rho2_ext, u_star, scheme=primitive_scheme, floor=_EPS)
    e1_f = eos1.energy(rho1_f, p_adv_f)
    e2_f = eos2.energy(rho2_f, p_adv_f)
    E1_f = e1_f + 0.5 * u_adv_f * u_adv_f
    E2_f = e2_f + 0.5 * u_adv_f * u_adv_f
    alpha_upwind = np.where(upwind_left, W_ext[0][:-1], W_ext[0][1:])
    inv_dx = 1.0 / dx
    q1_cons = alpha_upwind * rho1_f
    q2_cons = (1.0 - alpha_upwind) * rho2_f
    m_cons = (q1_cons + q2_cons) * u_adv_f
    rE_cons = q1_cons * E1_f + q2_cons * E2_f
    if alpha_scheme in ('cicsam', 'mstacs', 'stacs', 'superbee', 'vanleer', 'tvd_vanleer'):
        # Apply the sharp alpha correction everywhere, but limit its induced
        # anti-diffusive correction by a face-local maximum principle on the
        # conservative quantities.  This is a flux-corrected transport form:
        # the same theta multiplies alpha, phase mass, momentum, and energy
        # corrections, so no wave/contact sensor switches schemes by region.
        alpha_sharp = np.clip(_alpha_face(W_ext[0], u_star, dt, dx, alpha_scheme),
                              1.0e-12, 1.0 - 1.0e-12)
        delta_alpha = alpha_sharp - alpha_upwind
        theta = np.ones_like(delta_alpha)

        def apply_lmp(theta_in, base, coeff, left, right):
            delta = coeff * delta_alpha
            lo = np.minimum(np.minimum(left, right), base)
            hi = np.maximum(np.maximum(left, right), base)
            theta_out = theta_in.copy()
            pos = delta > _EPS
            neg = delta < -_EPS
            theta_out[pos] = np.minimum(theta_out[pos], (hi[pos] - base[pos]) / delta[pos])
            theta_out[neg] = np.minimum(theta_out[neg], (lo[neg] - base[neg]) / delta[neg])
            return np.clip(theta_out, 0.0, 1.0)

        theta = apply_lmp(theta, q1_cons, rho1_f, U_ext[0][:-1], U_ext[0][1:])
        theta = apply_lmp(theta, q2_cons, -rho2_f, U_ext[1][:-1], U_ext[1][1:])
        rho_ext = U_ext[0] + U_ext[1]
        theta = apply_lmp(
            theta, q1_cons + q2_cons, rho1_f - rho2_f,
            rho_ext[:-1], rho_ext[1:])
        theta = apply_lmp(
            theta, m_cons, (rho1_f - rho2_f) * u_adv_f,
            U_ext[2][:-1], U_ext[2][1:])
        theta = apply_lmp(
            theta, rE_cons, rho1_f * E1_f - rho2_f * E2_f,
            U_ext[3][:-1], U_ext[3][1:])

        def apply_update_lmp(theta_in, cell_now, low_face_value, coeff,
                             stencil_ext, extra_face_flux=None):
            """Zalesak-style cell update limiter for the alpha correction.

            The face-local limiter above bounds reconstructed face states.  A
            conservative anti-diffusive flux can still combine through the two
            faces of a cell and overshoot the cell-local stencil after the
            update.  This limiter bounds the *update* produced by the same
            anti-diffusive flux using the local maximum principle of the
            conserved quantity; no wave/contact sensor or tunable coefficient
            is introduced.
            """
            n_cell = len(cell_now)
            theta_out = theta_in.copy()
            anti_flux = coeff * delta_alpha * u_star
            low_flux = low_face_value * u_star
            low_update = np.asarray(cell_now, dtype=float) - dt * inv_dx * (
                low_flux[1:] - low_flux[:-1])
            if extra_face_flux is not None:
                low_update = low_update - dt * inv_dx * (
                    extra_face_flux[1:] - extra_face_flux[:-1])
            # Bound the anti-diffusive correction by the local low-order
            # update, not only by the old-time stencil.  This is the usual FCT
            # monotonicity contract: high-order fluxes may sharpen the
            # low-order update, but must not add new local variation to it.
            low_ext = np.concatenate(([low_update[0]], low_update, [low_update[-1]]))
            low_l = low_ext[:-2]
            low_c = low_ext[1:-1]
            low_r = low_ext[2:]
            monotone_low = (low_c - low_l) * (low_r - low_c) >= 0.0
            lo_three = np.minimum(np.minimum(low_l, low_c), low_r)
            hi_three = np.maximum(np.maximum(low_l, low_c), low_r)
            lo = np.where(monotone_low, np.minimum(low_l, low_r), lo_three)
            hi = np.where(monotone_low, np.maximum(low_l, low_r), hi_three)
            allow_pos = np.maximum(hi - low_update, 0.0)
            allow_neg = np.maximum(low_update - lo, 0.0)
            sum_pos = np.zeros(n_cell)
            sum_neg = np.zeros(n_cell)

            for f, af in enumerate(anti_flux):
                if f > 0:
                    c = -dt * inv_dx * af
                    i = f - 1
                    if c >= 0.0:
                        sum_pos[i] += c
                    else:
                        sum_neg[i] += -c
                if f < n_cell:
                    c = dt * inv_dx * af
                    i = f
                    if c >= 0.0:
                        sum_pos[i] += c
                    else:
                        sum_neg[i] += -c

            r_pos = np.ones(n_cell)
            r_neg = np.ones(n_cell)
            pos = sum_pos > _EPS
            neg = sum_neg > _EPS
            r_pos[pos] = np.minimum(1.0, allow_pos[pos] / sum_pos[pos])
            r_neg[neg] = np.minimum(1.0, allow_neg[neg] / sum_neg[neg])

            for f, af in enumerate(anti_flux):
                lim = 1.0
                if f > 0:
                    c = -dt * inv_dx * af
                    lim = min(lim, r_pos[f - 1] if c >= 0.0 else r_neg[f - 1])
                if f < n_cell:
                    c = dt * inv_dx * af
                    lim = min(lim, r_pos[f] if c >= 0.0 else r_neg[f])
                theta_out[f] = min(theta_out[f], lim)
            return np.clip(theta_out, 0.0, 1.0)

        rho_ext = U_ext[0] + U_ext[1]
        theta = apply_update_lmp(
            theta, U_n[0], q1_cons, rho1_f, U_ext[0])
        theta = apply_update_lmp(
            theta, U_n[1], q2_cons, -rho2_f, U_ext[1])
        theta = apply_update_lmp(
            theta, U_n[0] + U_n[1], q1_cons + q2_cons,
            rho1_f - rho2_f, rho_ext)
        theta = apply_update_lmp(
            theta, U_n[2], m_cons, (rho1_f - rho2_f) * u_adv_f, U_ext[2])
        theta = apply_update_lmp(
            theta, U_n[3], rE_cons, rho1_f * E1_f - rho2_f * E2_f,
            U_ext[3], extra_face_flux=p_star * u_star)

        alpha_f = np.clip(alpha_upwind + theta * delta_alpha,
                          1.0e-12, 1.0 - 1.0e-12)
        delta_alpha = alpha_f - alpha_upwind
        q1_f = q1_cons + rho1_f * delta_alpha
        q2_f = q2_cons - rho2_f * delta_alpha
        m_f = m_cons + (rho1_f - rho2_f) * u_adv_f * delta_alpha
        rE_f = rE_cons + (rho1_f * E1_f - rho2_f * E2_f) * delta_alpha
    else:
        alpha_f = _alpha_face(W_ext[0], u_star, dt, dx, alpha_scheme)
        q1_f = q1_cons
        q2_f = q2_cons
        m_f = m_cons
        rE_f = rE_cons

    if primitive_scheme != 'upwind':
        # Flux-corrected primitive reconstruction: keep the high-order
        # primitive face state, but reject the part of its conservative flux
        # correction that would create new local extrema in q1, q2, rho, m, or
        # rhoE.  The same theta is applied to all conservative variables, so
        # the phase-mass/momentum/energy fluxes remain mutually consistent.
        T1_upw = np.where(upwind_left, W_ext[1][:-1], W_ext[1][1:])
        T2_upw = np.where(upwind_left, W_ext[2][:-1], W_ext[2][1:])
        u_upw = np.where(upwind_left, W_ext[3][:-1], W_ext[3][1:])
        p_upw = np.where(upwind_left, W_ext[4][:-1], W_ext[4][1:])
        rho1_lo = np.maximum(eos1.density(p_upw, T1_upw), _EPS)
        rho2_lo = np.maximum(eos2.density(p_upw, T2_upw), _EPS)
        e1_lo = eos1.energy(rho1_lo, p_upw)
        e2_lo = eos2.energy(rho2_lo, p_upw)
        E1_lo = e1_lo + 0.5 * u_upw * u_upw
        E2_lo = e2_lo + 0.5 * u_upw * u_upw
        q1_lo = alpha_f * rho1_lo
        q2_lo = (1.0 - alpha_f) * rho2_lo
        m_lo = (q1_lo + q2_lo) * u_upw
        rE_lo = q1_lo * E1_lo + q2_lo * E2_lo
        theta_ho = np.ones_like(u_star)

        def limit_high_order_flux(theta_in, cell_now, low_face, high_face,
                                  stencil_ext):
            n_cell = len(cell_now)
            theta_out = theta_in.copy()
            anti_flux = (high_face - low_face) * u_star
            low_flux = low_face * u_star
            low_update = np.asarray(cell_now, dtype=float) - dt * inv_dx * (
                low_flux[1:] - low_flux[:-1])
            # Bound the anti-diffusive primitive correction relative to the
            # low-order update.  Old-time stencil bounds alone can admit a
            # bounded but oscillatory sawtooth when two opposite face
            # corrections enter a smooth cell.  The FCT/TVD contract is that
            # the high-order correction may sharpen the low-order update but
            # must not create new local variation in a monotone low-order
            # profile.
            if bc_l == 'periodic' and bc_r == 'periodic':
                low_ext = np.concatenate(([low_update[-1]], low_update, [low_update[0]]))
            else:
                low_ext = np.concatenate(([low_update[0]], low_update, [low_update[-1]]))
            low_l = low_ext[:-2]
            low_c = low_ext[1:-1]
            low_r = low_ext[2:]
            monotone_low = (low_c - low_l) * (low_r - low_c) >= 0.0
            lo_three = np.minimum(np.minimum(low_l, low_c), low_r)
            hi_three = np.maximum(np.maximum(low_l, low_c), low_r)
            lo = np.where(monotone_low, np.minimum(low_l, low_r), lo_three)
            hi = np.where(monotone_low, np.maximum(low_l, low_r), hi_three)
            allow_pos = np.maximum(hi - low_update, 0.0)
            allow_neg = np.maximum(low_update - lo, 0.0)
            sum_pos = np.zeros(n_cell)
            sum_neg = np.zeros(n_cell)
            for f, af in enumerate(anti_flux):
                if f > 0:
                    c = -dt * inv_dx * af
                    i = f - 1
                    if c >= 0.0:
                        sum_pos[i] += c
                    else:
                        sum_neg[i] += -c
                if f < n_cell:
                    c = dt * inv_dx * af
                    i = f
                    if c >= 0.0:
                        sum_pos[i] += c
                    else:
                        sum_neg[i] += -c
            r_pos = np.ones(n_cell)
            r_neg = np.ones(n_cell)
            pos = sum_pos > _EPS
            neg = sum_neg > _EPS
            r_pos[pos] = np.minimum(1.0, allow_pos[pos] / sum_pos[pos])
            r_neg[neg] = np.minimum(1.0, allow_neg[neg] / sum_neg[neg])
            for f, af in enumerate(anti_flux):
                lim = 1.0
                if f > 0:
                    c = -dt * inv_dx * af
                    lim = min(lim, r_pos[f - 1] if c >= 0.0 else r_neg[f - 1])
                if f < n_cell:
                    c = dt * inv_dx * af
                    lim = min(lim, r_pos[f] if c >= 0.0 else r_neg[f])
                theta_out[f] = min(theta_out[f], lim)
            return np.clip(theta_out, 0.0, 1.0)

        theta_ho = limit_high_order_flux(theta_ho, U_n[0], q1_lo, q1_f, U_ext[0])
        theta_ho = limit_high_order_flux(theta_ho, U_n[1], q2_lo, q2_f, U_ext[1])
        rho_ext = U_ext[0] + U_ext[1]
        theta_ho = limit_high_order_flux(
            theta_ho, U_n[0] + U_n[1], q1_lo + q2_lo, q1_f + q2_f,
            rho_ext)
        theta_ho = limit_high_order_flux(theta_ho, U_n[2], m_lo, m_f, U_ext[2])
        theta_ho = limit_high_order_flux(theta_ho, U_n[3], rE_lo, rE_f, U_ext[3])
        q1_f = q1_lo + theta_ho * (q1_f - q1_lo)
        q2_f = q2_lo + theta_ho * (q2_f - q2_lo)
        m_f = m_lo + theta_ho * (m_f - m_lo)
        rE_f = rE_lo + theta_ho * (rE_f - rE_lo)

    if bc_l == 'reflective':
        q1_f[0] = 0.0
        q2_f[0] = 0.0
        m_f[0] = 0.0
        rE_f[0] = 0.0
        alpha_f[0] = 0.0
    if bc_r == 'reflective':
        q1_f[-1] = 0.0
        q2_f[-1] = 0.0
        m_f[-1] = 0.0
        rE_f[-1] = 0.0
        alpha_f[-1] = 0.0

    F_q1 = q1_f * u_star
    F_q2 = q2_f * u_star
    F_m_adv = m_f * u_star
    F_alpha = alpha_f * u_star
    F_rho = F_q1 + F_q2
    if material_energy_form in ('apec', 'secant', 'differential'):
        face = _face_energy_dict(
            W_ext, p_star, u_star, upwind_left, alpha_f,
            rho1_f, rho2_f, eos1, eos2)
        flux_form = 'secant' if material_energy_form in ('apec', 'secant') else 'differential'
        F_rE_adv = total_energy_flux(
            face, eos1, eos2, F_q1, F_q2, F_alpha, F_rho,
            energy_form=flux_form,
            alpha_pure_tol=max(float(alpha_pure_tol), 0.0))
    else:
        F_rE_adv = rE_f * u_star
    F_pu_old = p_star * u_star
    L_q1 = (F_q1[1:] - F_q1[:-1]) * inv_dx
    L_q2 = (F_q2[1:] - F_q2[:-1]) * inv_dx
    L_m_adv = (F_m_adv[1:] - F_m_adv[:-1]) * inv_dx
    L_rE_adv = (F_rE_adv[1:] - F_rE_adv[:-1]) * inv_dx
    L_pu_old = (F_pu_old[1:] - F_pu_old[:-1]) * inv_dx
    L_rE = L_rE_adv + L_pu_old

    div_u = (u_star[1:] - u_star[:-1]) * inv_dx
    B = np.asarray(W_n[0], dtype=float).copy()
    if kapila_closure:
        B_ext = np.asarray(W_ext[0], dtype=float) + D_K_kapila(W_ext, eos1, eos2)
        B_f = 0.5 * (B_ext[:-1] + B_ext[1:])
        u_cell = np.asarray(W_n[3], dtype=float)
        source_face = (
            B_f[1:] * (u_star[1:] - u_cell)
            + B_f[:-1] * (u_cell - u_star[:-1])
        ) * inv_dx
        source_cell = (
            np.asarray(W_n[0], dtype=float) + D_K_kapila(W_n, eos1, eos2)
        ) * div_u
        pure_tol = max(float(alpha_pure_tol), np.finfo(float).eps ** 0.25)
        material_cells = _pure_material_cell_mask(W_n[0], pure_tol)
        if kapila_source_mode == 'path':
            source_alpha = source_face
        elif kapila_source_mode == 'cell':
            source_alpha = source_cell
        elif kapila_source_mode == 'trapezoid':
            source_alpha = 0.5 * (source_face + source_cell)
        elif kapila_source_mode == 'immiscible_trapezoid':
            source_trap = 0.5 * (source_face + source_cell)
            source_hybrid = np.where(material_cells, source_cell, source_face)
            a_ext = np.asarray(W_ext[0], dtype=float)
            a_lo = np.minimum(np.minimum(a_ext[:-2], a_ext[1:-1]), a_ext[2:])
            a_hi = np.maximum(np.maximum(a_ext[:-2], a_ext[1:-1]), a_ext[2:])
            immiscible_stencil = (a_lo <= pure_tol) & (a_hi >= 1.0 - pure_tol)
            source_alpha = np.where(immiscible_stencil, source_hybrid, source_trap)
        else:
            source_alpha = np.where(material_cells, source_cell, source_face)
    else:
        source_alpha = B * div_u
    L_alpha = (F_alpha[1:] - F_alpha[:-1]) * inv_dx
    L_alpha = L_alpha - source_alpha

    rhoE_adv = U_n[3] - dt * L_rE_adv
    out = (
        U_n[0] - dt * L_q1,
        U_n[1] - dt * L_q2,
        U_n[2] - dt * L_m_adv,
        U_n[3] - dt * L_rE,
        U_n[4] - dt * L_alpha,
    )
    if return_aux:
        aux = {
            "rhoE_adv": rhoE_adv,
            "F_pu_old": F_pu_old,
            "Z_ext": Z_ext,
            "p_star_old": p_star,
            "u_star_old": u_star,
            "kapila_source_mode": kapila_source_mode,
            "material_energy_form": material_energy_form,
            "primitive_scheme": primitive_scheme,
        }
        return out, aux
    return out


def _solve_acoustic_ad(W_n, q1_new, q2_new, m_adv, alpha_new, dt,
                       eos1, eos2, dx, bc_l, bc_r, *,
                       u_inlet=None, p_inlet=None,
                       mixture_kind, alpha_pure_tol,
                       primitive_scheme='upwind'):
    primitive_scheme = normalise_primitive_scheme(primitive_scheme)
    alpha, T1, T2, u0, p0 = W_n
    rho_star = np.maximum(q1_new + q2_new, _EPS)
    rho_anchor, c_mix_sq, Z = _phase_acoustic(
        W_n, eos1, eos2, mixture_kind=mixture_kind,
        alpha_pure_tol=alpha_pure_tol)
    beta = np.maximum(rho_anchor * c_mix_sq, _EPS)
    u_mask = np.asarray(u0 >= 0.0)

    y = np.concatenate((np.asarray(u0, dtype=float), np.asarray(p0, dtype=float)))
    n = len(u0)
    z0_u = np.asarray(u0, dtype=float)
    z0_p = np.asarray(p0, dtype=float)
    acoustic_theta_wave = float(os.environ.get("FIVE_EQ_IMEX_ACOUSTIC_THETA", "1.0"))
    acoustic_theta_wave = min(1.0, max(0.5, acoustic_theta_wave))
    if n >= 32:
        p_f_old, u_f_old, high_face = _acoustic_faces_muscl_np(
            z0_u, z0_p, Z, alpha, bc_l, bc_r, alpha_pure_tol,
            u_inlet=u_inlet, p_inlet=p_inlet,
            primitive_scheme=primitive_scheme)
    else:
        p_f_old, u_f_old = _acoustic_faces_np(
            z0_u, z0_p, Z, bc_l, bc_r,
            u_inlet=u_inlet, p_inlet=p_inlet)
        high_face = np.zeros(n + 1, dtype=bool)
    div_p_old = (p_f_old[1:] - p_f_old[:-1]) / dx
    div_u_old = (u_f_old[1:] - u_f_old[:-1]) / dx
    p_ext_old = _extend_np(z0_p, bc_l, bc_r, odd=False)
    p_rel_face = np.abs(p_ext_old[1:] - p_ext_old[:-1]) / np.maximum(
        np.maximum(np.abs(p_ext_old[1:]), np.abs(p_ext_old[:-1])), 1.0)
    wave_face = p_rel_face > float(os.environ.get("FIVE_EQ_IMEX_ACOUSTIC_THETA_WAVE_REL", "1e-8"))
    wave_cell = wave_face[:-1] | wave_face[1:]
    theta_cell = np.where(wave_cell, acoustic_theta_wave, 1.0)
    has_left_dirichlet = bc_l in ('inlet', 'dirichlet', 'inlet_acoustic')
    u_left_bc = float(u_inlet) if u_inlet is not None else float(z0_u[0])
    p_left_bc = float(p_inlet) if p_inlet is not None else float(z0_p[0])

    def face_local(p_l, p_r, u_l, u_r, Z_l, Z_r):
        den = anp.maximum(Z_l + Z_r, _EPS)
        p_f = (Z_r * p_l + Z_l * p_r + Z_l * Z_r * (u_l - u_r)) / den
        u_f = (p_l - p_r + Z_l * u_l + Z_r * u_r) / den
        return p_f, u_f

    def build_sparse_jacobian_vectorized():
        """Assemble sparse acoustic Jacobian from batched local autodiff.

        We use ``torch.func.vmap(jacrev(...))`` to differentiate each cell's
        two-equation local stencil residual with respect to its six stencil
        unknowns.  This avoids a dense full-domain Jacobian while still keeping
        the implicit Jacobian generated by an autodiff library rather than by a
        hand-derived formula.
        """
        if bc_l == 'periodic' and bc_r == 'periodic':
            il = np.arange(n) - 1
            il[0] = n - 1
            ir = np.arange(n) + 1
            ir[-1] = 0
            left_boundary = np.zeros(n, dtype=bool)
            right_boundary = np.zeros(n, dtype=bool)
        else:
            il = np.maximum(np.arange(n) - 1, 0)
            ir = np.minimum(np.arange(n) + 1, n - 1)
            left_boundary = np.zeros(n, dtype=bool)
            right_boundary = np.zeros(n, dtype=bool)
            left_boundary[0] = True
            right_boundary[-1] = True

        ill = np.maximum(il - 1, 0)
        irr = np.minimum(ir + 1, n - 1)
        if bc_l == 'periodic' and bc_r == 'periodic':
            ill = (np.arange(n) - 2) % n
            irr = (np.arange(n) + 2) % n
        z0_np = np.column_stack((
            z0_u[ill], z0_u[il], z0_u, z0_u[ir], z0_u[irr],
            z0_p[ill], z0_p[il], z0_p, z0_p[ir], z0_p[irr],
        ))
        params_np = np.column_stack((
            Z[il], Z, Z[ir],
            rho_star, beta, u0, p0, m_adv,
            u_mask.astype(float),
            div_p_old, div_u_old, theta_cell,
            left_boundary.astype(float),
            right_boundary.astype(float),
            high_face[:-1].astype(float),
            high_face[1:].astype(float),
        ))
        z0_t = torch.as_tensor(z0_np, dtype=torch.float64)
        params_t = torch.as_tensor(params_np, dtype=torch.float64)
        tvd_kind = os.environ.get("FIVE_EQ_IMEX_TMLPU_TVD", "minmod").strip().lower().replace("-", "_")
        left_reflective = 1.0 if bc_l == 'reflective' else 0.0
        right_reflective = 1.0 if bc_r == 'reflective' else 0.0
        left_dirichlet = 1.0 if has_left_dirichlet else 0.0

        def local_residual_torch(z, params):
            u_ll, u_l, u_c, u_r, u_rr, p_ll, p_l, p_c, p_r, p_rr = z
            (Z_l, Z_c, Z_r, rho_i, beta_i, u0_i, p0_i, m_i, up_i,
             dp_old_i, du_old_i, theta_i, lb_i, rb_i, ho_l_i, ho_r_i) = params

            def limited_t(a, b):
                same = (a * b) > 0.0
                zero = torch.zeros_like(a)
                if tvd_kind in ("superbee", "sb"):
                    cand1 = torch.minimum(2.0 * torch.abs(a), torch.abs(b))
                    cand2 = torch.minimum(torch.abs(a), 2.0 * torch.abs(b))
                    val = torch.sign(a) * torch.maximum(cand1, cand2)
                elif tvd_kind in ("mc", "monotonized_central"):
                    centered = 0.5 * (a + b)
                    mag = torch.minimum(
                        torch.minimum(2.0 * torch.abs(a), 2.0 * torch.abs(b)),
                        torch.abs(centered),
                    )
                    val = torch.sign(centered) * mag
                elif tvd_kind in ("vanleer", "van_leer"):
                    den = a + b
                    den_safe = torch.where(torch.abs(den) > _EPS, den, torch.ones_like(den))
                    val = 2.0 * a * b / den_safe
                else:
                    val = torch.sign(a) * torch.minimum(torch.abs(a), torch.abs(b))
                return torch.where(same, val, zero)

            den_l = torch.clamp(Z_l + Z_c, min=_EPS)
            p_fl_raw = (Z_c * p_l + Z_l * p_c + Z_l * Z_c * (u_l - u_c)) / den_l
            u_fl_raw = (p_l - p_c + Z_l * u_l + Z_c * u_c) / den_l
            sp_l = limited_t(p_l - p_ll, p_c - p_l)
            sp_c = limited_t(p_c - p_l, p_r - p_c)
            su_l = limited_t(u_l - u_ll, u_c - u_l)
            su_c = limited_t(u_c - u_l, u_r - u_c)
            p_lh = p_l + 0.5 * sp_l
            p_ch_l = p_c - 0.5 * sp_c
            u_lh = u_l + 0.5 * su_l
            u_ch_l = u_c - 0.5 * su_c
            p_fl_ho = (Z_c * p_lh + Z_l * p_ch_l + Z_l * Z_c * (u_lh - u_ch_l)) / den_l
            u_fl_ho = (p_lh - p_ch_l + Z_l * u_lh + Z_c * u_ch_l) / den_l
            p_fl_raw = torch.where(ho_l_i > 0.5, p_fl_ho, p_fl_raw)
            u_fl_raw = torch.where(ho_l_i > 0.5, u_fl_ho, u_fl_raw)
            den_r = torch.clamp(Z_c + Z_r, min=_EPS)
            p_fr_raw = (Z_r * p_c + Z_c * p_r + Z_c * Z_r * (u_c - u_r)) / den_r
            u_fr_raw = (p_c - p_r + Z_c * u_c + Z_r * u_r) / den_r
            sp_r = limited_t(p_r - p_c, p_rr - p_r)
            su_r = limited_t(u_r - u_c, u_rr - u_r)
            p_ch_r = p_c + 0.5 * sp_c
            p_rh = p_r - 0.5 * sp_r
            u_ch_r = u_c + 0.5 * su_c
            u_rh = u_r - 0.5 * su_r
            p_fr_ho = (Z_r * p_ch_r + Z_c * p_rh + Z_c * Z_r * (u_ch_r - u_rh)) / den_r
            u_fr_ho = (p_ch_r - p_rh + Z_c * u_ch_r + Z_r * u_rh) / den_r
            p_fr_raw = torch.where(ho_r_i > 0.5, p_fr_ho, p_fr_raw)
            u_fr_raw = torch.where(ho_r_i > 0.5, u_fr_ho, u_fr_raw)
            p_fl_b = torch.where(
                torch.tensor(left_dirichlet > 0.5, dtype=torch.bool),
                torch.as_tensor(p_left_bc, dtype=z.dtype, device=z.device),
                p_c,
            )
            p_fl = torch.where(lb_i > 0.5, p_fl_b, p_fl_raw)
            p_fr = torch.where(rb_i > 0.5, p_c, p_fr_raw)
            u_fl_b = torch.where(
                torch.tensor(left_reflective > 0.5, dtype=torch.bool),
                torch.zeros((), dtype=z.dtype, device=z.device),
                torch.where(
                    torch.tensor(left_dirichlet > 0.5, dtype=torch.bool),
                    torch.as_tensor(u_left_bc, dtype=z.dtype, device=z.device),
                    u_c,
                ),
            )
            u_fr_b = torch.where(
                torch.tensor(right_reflective > 0.5, dtype=torch.bool),
                torch.zeros((), dtype=z.dtype, device=z.device),
                u_c,
            )
            u_fl = torch.where(lb_i > 0.5, u_fl_b, u_fl_raw)
            u_fr = torch.where(rb_i > 0.5, u_fr_b, u_fr_raw)
            p_l_eff = torch.where(lb_i > 0.5, p_fl_b, p_l)
            p_r_eff = torch.where(rb_i > 0.5, p_c, p_r)
            dp_back = (p_c - p_l_eff) / dx
            dp_forw = (p_r_eff - p_c) / dx
            dp_dx = torch.where(up_i > 0.5, dp_back, dp_forw)
            div_p = theta_i * (p_fr - p_fl) / dx + (1.0 - theta_i) * dp_old_i
            div_u = theta_i * (u_fr - u_fl) / dx + (1.0 - theta_i) * du_old_i
            r_u = rho_i * u_c - m_i + dt * div_p
            r_p = p_c - p0_i + dt * (u0_i * dp_dx + beta_i * div_u)
            return torch.stack((r_u, r_p))

        R_batch_t = vmap(local_residual_torch)(z0_t, params_t)
        J_batch_t = vmap(jacrev(local_residual_torch, argnums=0))(z0_t, params_t)
        R_batch = R_batch_t.detach().cpu().numpy()
        J_batch = J_batch_t.detach().cpu().numpy()
        R_global = np.concatenate((R_batch[:, 0], R_batch[:, 1]))
        J_global = lil_matrix((2 * n, 2 * n), dtype=float)
        for i in range(n):
            col_ids = [
                ill[i], il[i], i, ir[i], irr[i],
                n + ill[i], n + il[i], n + i, n + ir[i], n + irr[i],
            ]
            for local_col, global_col in enumerate(col_ids):
                J_global[i, global_col] += J_batch[i, 0, local_col]
                J_global[n + i, global_col] += J_batch[i, 1, local_col]
        return R_global, J_global

    if n >= 32:
        R, J = build_sparse_jacobian_vectorized()
        scale = max(float(np.max(np.abs(J.diagonal()))), 1.0)
        from scipy.sparse import eye as speye
        dy = spsolve(J.tocsr() + 1.0e-12 * scale * speye(2 * n, format='csr'), -R)
        y = y + dy
        return y[:n], y[n:]

    J = lil_matrix((2 * n, 2 * n), dtype=float)
    R = np.zeros(2 * n, dtype=float)
    for i in range(n):
        if bc_l == 'periodic' and bc_r == 'periodic':
            il = (i - 1) % n
            ir = (i + 1) % n
            left_boundary = False
            right_boundary = False
        else:
            il = max(i - 1, 0)
            ir = min(i + 1, n - 1)
            left_boundary = i == 0
            right_boundary = i == n - 1
        cols = [il, i, ir]
        z0 = np.array([
            z0_u[il], z0_u[i], z0_u[ir],
            z0_p[il], z0_p[i], z0_p[ir],
        ], dtype=float)

        def local_res(z):
            u_l, u_c, u_r, p_l, p_c, p_r = z
            if left_boundary:
                if bc_l == 'reflective':
                    p_fl = p_c
                    u_fl = 0.0
                    p_l_eff = p_c
                elif has_left_dirichlet:
                    p_fl = p_left_bc
                    u_fl = u_left_bc
                    p_l_eff = p_left_bc
                else:
                    p_fl = p_c
                    u_fl = u_c
                    p_l_eff = p_c
            else:
                p_fl, u_fl = face_local(p_l, p_c, u_l, u_c, Z[il], Z[i])
                p_l_eff = p_l
            if right_boundary:
                p_fr = p_c
                u_fr = 0.0 if bc_r == 'reflective' else u_c
                p_r_eff = p_c
            else:
                p_fr, u_fr = face_local(p_c, p_r, u_c, u_r, Z[i], Z[ir])
                p_r_eff = p_r
            dp_back = (p_c - p_l_eff) / dx
            dp_forw = (p_r_eff - p_c) / dx
            dp_dx = dp_back if u_mask[i] else dp_forw
            theta_i = theta_cell[i]
            div_p = theta_i * (p_fr - p_fl) / dx + (1.0 - theta_i) * div_p_old[i]
            div_u = theta_i * (u_fr - u_fl) / dx + (1.0 - theta_i) * div_u_old[i]
            r_u = rho_star[i] * u_c - m_adv[i] + dt * div_p
            r_p = p_c - p0[i] + dt * (u0[i] * dp_dx + beta[i] * div_u)
            return anp.array([r_u, r_p])

        r_loc = np.asarray(local_res(z0), dtype=float)
        j_loc = np.asarray(jacobian(local_res)(z0), dtype=float)
        R[i] = r_loc[0]
        R[n + i] = r_loc[1]
        col_ids = [cols[0], cols[1], cols[2], n + cols[0], n + cols[1], n + cols[2]]
        for local_col, global_col in enumerate(col_ids):
            J[i, global_col] += j_loc[0, local_col]
            J[n + i, global_col] += j_loc[1, local_col]

    scale = max(float(np.max(np.abs(J.diagonal()))), 1.0)
    from scipy.sparse import eye as speye
    dy = spsolve(J.tocsr() + 1.0e-12 * scale * speye(2 * n, format='csr'), -R)
    y = y + dy
    return y[:n], y[n:]


def _solve_acoustic_energy_ad(W_n, q1_new, q2_new, m_adv, rhoE_adv,
                              alpha_new, dt, eos1, eos2, dx, bc_l, bc_r, *,
                              u_inlet=None, p_inlet=None,
                              mixture_kind, alpha_pure_tol,
                              primitive_scheme='upwind',
                              max_newton=4, full_newton=False):
    """Implicit acoustic solve with total energy as the pressure equation.

    This is not the old post-step pressure recovery.  Pressure participates in
    the Newton residual through the cell total-energy constraint and the same
    face pressure-work flux used by the momentum acoustic solve.
    """
    if not full_newton:
        u_new, p_new = _solve_acoustic_ad(
            W_n, q1_new, q2_new, m_adv, alpha_new, dt,
            eos1, eos2, dx, bc_l, bc_r,
            u_inlet=u_inlet, p_inlet=p_inlet,
            mixture_kind=mixture_kind,
            alpha_pure_tol=alpha_pure_tol,
            primitive_scheme=primitive_scheme)
        alpha_c = np.clip(np.asarray(alpha_new, dtype=float), 1.0e-12, 1.0 - 1.0e-12)
        q1 = np.maximum(np.asarray(q1_new, dtype=float), _EPS)
        q2 = np.maximum(np.asarray(q2_new, dtype=float), _EPS)
        rho = np.maximum(q1 + q2, _EPS)
        rho1 = np.maximum(q1 / alpha_c, _EPS)
        rho2 = np.maximum(q2 / np.maximum(1.0 - alpha_c, 1.0e-12), _EPS)
        _, _, Z = _phase_acoustic(
            W_n, eos1, eos2, mixture_kind=mixture_kind,
            alpha_pure_tol=alpha_pure_tol)
        mask = _compressive_pressure_mask(W_n)
        p_out = np.asarray(p_new, dtype=float).copy()

        def energy_residual_at(i, p_i):
            p_tmp = p_out.copy()
            p_tmp[i] = max(float(p_i), 1.0e-12)
            p_f, u_f = _acoustic_faces_np(
                u_new, p_tmp, Z, bc_l, bc_r,
                u_inlet=u_inlet, p_inlet=p_inlet)
            e1 = float(eos1.energy(np.asarray([rho1[i]]), np.asarray([p_tmp[i]]))[0])
            e2 = float(eos2.energy(np.asarray([rho2[i]]), np.asarray([p_tmp[i]]))[0])
            rhoE_state = q1[i] * e1 + q2[i] * e2 + 0.5 * rho[i] * u_new[i] * u_new[i]
            return rhoE_state - rhoE_adv[i] + dt * (
                p_f[i + 1] * u_f[i + 1] - p_f[i] * u_f[i]) / dx

        for i in np.flatnonzero(mask):
            p_i = max(float(p_out[i]), 1.0e-12)
            for _ in range(max_newton + 4):
                r0 = energy_residual_at(i, p_i)
                h = 1.0e-7 * max(abs(p_i), 1.0)
                dr = (energy_residual_at(i, p_i + h) - r0) / h
                if not np.isfinite(dr) or abs(dr) < _EPS:
                    break
                step = r0 / dr
                p_next = max(p_i - step, 1.0e-12)
                if abs(p_next - p_i) / max(abs(p_next), 1.0) < 1.0e-9:
                    p_i = p_next
                    break
                p_i = p_next
            p_out[i] = p_i
        return u_new, p_out

    alpha, _, _, u0, p0 = W_n
    n = len(u0)
    alpha_c = np.clip(np.asarray(alpha_new, dtype=float), 1.0e-12, 1.0 - 1.0e-12)
    q1 = np.maximum(np.asarray(q1_new, dtype=float), _EPS)
    q2 = np.maximum(np.asarray(q2_new, dtype=float), _EPS)
    rho = np.maximum(q1 + q2, _EPS)
    rho1 = np.maximum(q1 / alpha_c, _EPS)
    rho2 = np.maximum(q2 / np.maximum(1.0 - alpha_c, 1.0e-12), _EPS)
    _, _, Z = _phase_acoustic(
        W_n, eos1, eos2, mixture_kind=mixture_kind,
        alpha_pure_tol=alpha_pure_tol)
    has_left_dirichlet = bc_l in ('inlet', 'dirichlet', 'inlet_acoustic')
    u_left_bc = float(u_inlet) if u_inlet is not None else float(u0[0])
    p_left_bc = float(p_inlet) if p_inlet is not None else float(p0[0])

    y = np.concatenate((np.asarray(u0, dtype=float), np.asarray(p0, dtype=float)))

    if bc_l == 'periodic' and bc_r == 'periodic':
        il_all = np.arange(n) - 1
        il_all[0] = n - 1
        ir_all = np.arange(n) + 1
        ir_all[-1] = 0
        left_boundary = np.zeros(n, dtype=bool)
        right_boundary = np.zeros(n, dtype=bool)
    else:
        il_all = np.maximum(np.arange(n) - 1, 0)
        ir_all = np.minimum(np.arange(n) + 1, n - 1)
        left_boundary = np.zeros(n, dtype=bool)
        right_boundary = np.zeros(n, dtype=bool)
        left_boundary[0] = True
        right_boundary[-1] = True

    def face_pair(p_l, p_r, u_l, u_r, Z_l, Z_r):
        den = max(float(Z_l + Z_r), _EPS)
        p_f = (Z_r * p_l + Z_l * p_r + Z_l * Z_r * (u_l - u_r)) / den
        u_f = (p_l - p_r + Z_l * u_l + Z_r * u_r) / den
        return p_f, u_f

    def local_residual(i, z):
        il = int(il_all[i])
        ir = int(ir_all[i])
        u_l, u_c, u_r, p_l, p_c, p_r = z
        p_eval = max(float(p_c), 1.0e-12)
        if left_boundary[i]:
            if bc_l == 'reflective':
                p_fl = p_c
                u_fl = 0.0
            elif has_left_dirichlet:
                p_fl = p_left_bc
                u_fl = u_left_bc
            else:
                p_fl = p_c
                u_fl = u_c
        else:
            p_fl, u_fl = face_pair(p_l, p_c, u_l, u_c, Z[il], Z[i])
        if right_boundary[i]:
            p_fr = p_c
            u_fr = 0.0 if bc_r == 'reflective' else u_c
        else:
            p_fr, u_fr = face_pair(p_c, p_r, u_c, u_r, Z[i], Z[ir])
        e1 = float(eos1.energy(np.asarray([rho1[i]]), np.asarray([p_eval]))[0])
        e2 = float(eos2.energy(np.asarray([rho2[i]]), np.asarray([p_eval]))[0])
        rhoE_state = q1[i] * e1 + q2[i] * e2 + 0.5 * rho[i] * u_c * u_c
        r_u = rho[i] * u_c - m_adv[i] + dt * (p_fr - p_fl) / dx
        r_E = rhoE_state - rhoE_adv[i] + dt * (p_fr * u_fr - p_fl * u_fl) / dx
        return np.asarray([r_u, r_E], dtype=float)

    for _ in range(max_newton):
        u = y[:n]
        p = np.maximum(y[n:], 1.0e-12)
        y = np.concatenate((u, p))
        R = np.zeros(2 * n, dtype=float)
        J = lil_matrix((2 * n, 2 * n), dtype=float)
        for i in range(n):
            il = int(il_all[i])
            ir = int(ir_all[i])
            z0 = np.array([u[il], u[i], u[ir], p[il], p[i], p[ir]], dtype=float)
            r0 = local_residual(i, z0)
            R[i] = r0[0]
            R[n + i] = r0[1]
            col_ids = [il, i, ir, n + il, n + i, n + ir]
            for local_col, global_col in enumerate(col_ids):
                h = 1.0e-6 * max(abs(z0[local_col]), 1.0)
                if local_col >= 3:
                    h = 1.0e-7 * max(abs(z0[local_col]), 1.0)
                zp = z0.copy()
                zp[local_col] += h
                jm = (local_residual(i, zp) - r0) / h
                J[i, global_col] += jm[0]
                J[n + i, global_col] += jm[1]
        norm_R = float(np.linalg.norm(R) / max(np.sqrt(R.size), 1.0))
        scale = max(float(np.max(np.abs(J.diagonal()))), 1.0)
        from scipy.sparse import eye as speye
        try:
            dy = spsolve(J.tocsr() + 1.0e-10 * scale * speye(2 * n, format='csr'), -R)
        except Exception:
            break
        if not np.all(np.isfinite(dy)):
            break
        lam = 1.0
        accepted = False
        for _ls in range(8):
            y_trial = y + lam * dy
            y_trial[n:] = np.maximum(y_trial[n:], 1.0e-12)
            if np.all(np.isfinite(y_trial)):
                y = y_trial
                accepted = True
                break
            lam *= 0.5
        if not accepted:
            break
        step_norm = float(np.linalg.norm(lam * dy) / max(np.linalg.norm(y), 1.0))
        if step_norm < 1.0e-9 or norm_R < 1.0e-8:
            break
    return y[:n], np.maximum(y[n:], 1.0e-12)


def _entropy_pressure_estimate(W_n, q1_new, q2_new, alpha_new, p_acoustic,
                               eos1, eos2):
    """Dual-energy/entropy-style pressure predictor for compression cells.

    For stiffened-gas type EOS this preserves each phase's isentropic pressure
    invariant ``(p+pinf)/rho**gamma`` over the material update.  Unsupported
    EOS fall back to the acoustic pressure.
    """
    alpha0, T10, T20, _, p0 = W_n
    gamma1 = getattr(eos1, "gamma", None)
    gamma2 = getattr(eos2, "gamma", None)
    if gamma1 is None or gamma2 is None:
        return np.asarray(p_acoustic, dtype=float)
    pinf1 = float(getattr(eos1, "pinf", 0.0))
    pinf2 = float(getattr(eos2, "pinf", 0.0))
    alpha0 = np.clip(np.asarray(alpha0, dtype=float), 1.0e-12, 1.0 - 1.0e-12)
    alpha1 = np.clip(np.asarray(alpha_new, dtype=float), 1.0e-12, 1.0 - 1.0e-12)
    rho10 = np.maximum(eos1.density(W_n[4], T10), _EPS)
    rho20 = np.maximum(eos2.density(W_n[4], T20), _EPS)
    rho11 = np.maximum(q1_new / alpha1, _EPS)
    rho21 = np.maximum(q2_new / np.maximum(1.0 - alpha1, 1.0e-12), _EPS)
    p1 = (np.asarray(W_n[4], dtype=float) + pinf1) * (rho11 / rho10) ** float(gamma1) - pinf1
    p2 = (np.asarray(W_n[4], dtype=float) + pinf2) * (rho21 / rho20) ** float(gamma2) - pinf2
    # Weight by updated phase volume.  This is intentionally a predictor, not
    # a conservative energy inversion.
    p_entropy = alpha1 * p1 + (1.0 - alpha1) * p2
    p_entropy = np.where(np.isfinite(p_entropy), p_entropy, p_acoustic)
    return np.maximum(p_entropy, 1.0e-12)


def imex_ad_step(W_n, dt, eos1, eos2, dx, bc_l, bc_r, *,
                 u_inlet=None, p_inlet=None,
                 mixture_kind='kapila', kapila_closure=False,
                 alpha_pure_tol=0.0, alpha_scheme='upwind',
                 primitive_scheme='upwind',
                 pressure_closure='regime_auto'):
    pressure_closure = _normalise_pressure_closure(pressure_closure)
    primitive_scheme = normalise_primitive_scheme(primitive_scheme)
    if pressure_closure == 'regime_auto':
        alpha_now = np.asarray(W_n[0], dtype=float)
        pure_tol_auto = max(float(alpha_pure_tol), np.finfo(float).eps ** 0.25)
        has_immiscible_interface = (
            float(np.min(alpha_now)) <= pure_tol_auto
            and float(np.max(alpha_now)) >= 1.0 - pure_tol_auto
        )
        pressure_closure = (
            'pressure_work_consistent'
            if has_immiscible_interface
            else 'implicit_energy'
        )
    pure_tol = max(float(alpha_pure_tol), 0.0)
    if pure_tol > 0.0 and float(np.max(W_n[0])) <= pure_tol:
        return _single_phase_euler_rusanov_step(
            W_n, dt, eos2, dx, bc_l, bc_r,
            u_inlet=u_inlet, p_inlet=p_inlet,
            primitive_scheme=primitive_scheme)
    if pure_tol > 0.0 and float(np.min(W_n[0])) >= 1.0 - pure_tol:
        return _single_phase_euler_rusanov_step(
            W_n, dt, eos1, dx, bc_l, bc_r,
            u_inlet=u_inlet, p_inlet=p_inlet,
            primitive_scheme=primitive_scheme)
    if (_same_eos(eos1, eos2)
            and float(np.max(W_n[0]) - np.min(W_n[0])) <= 1.0e-14
            and float(np.max(np.abs(W_n[1] - W_n[2]))) <= 1.0e-12):
        return _single_phase_euler_rusanov_step(
            W_n, dt, eos1, dx, bc_l, bc_r,
            u_inlet=u_inlet, p_inlet=p_inlet,
            primitive_scheme=primitive_scheme)
    # Use the face-path Kapila source by default.  It discretizes the
    # non-conservative D_K div(u) term on the same acoustic face path as the
    # material flux, avoiding a cell/face mismatch that appears as contact
    # overshoot or post-contact checkerboard in stiff gas-liquid shocks.
    kapila_source_mode = 'hybrid'
    source_override = os.environ.get("FIVE_EQ_IMEX_KAPILA_SOURCE_MODE")
    if source_override:
        source_key = source_override.strip().lower().replace("-", "_")
        if source_key in ('immiscible', 'interface_preserving'):
            source_key = 'immiscible_trapezoid'
        if source_key not in ('path', 'cell', 'hybrid', 'trapezoid',
                              'immiscible_trapezoid'):
            raise ValueError(
                "FIVE_EQ_IMEX_KAPILA_SOURCE_MODE must be one of "
                "'path', 'cell', 'hybrid', 'trapezoid', or "
                "'immiscible_trapezoid'.")
        kapila_source_mode = source_key
    material_energy_form = 'apec' if pressure_closure == 'apec_pe' else 'allaire'
    need_aux = pressure_closure in (
        'implicit_energy',
        'pressure_work_consistent',
        'apec_pe',
    )
    mat_result = _material_update(
        W_n, dt, eos1, eos2, dx, bc_l, bc_r,
        u_inlet=u_inlet, p_inlet=p_inlet,
        mixture_kind=mixture_kind,
        kapila_closure=kapila_closure,
        alpha_pure_tol=alpha_pure_tol,
        alpha_scheme=alpha_scheme,
        primitive_scheme=primitive_scheme,
        kapila_source_mode=kapila_source_mode,
        material_energy_form=material_energy_form,
        return_aux=need_aux)
    if need_aux:
        (q1_new, q2_new, m_adv, rhoE_new, alpha_new), aux = mat_result
    else:
        q1_new, q2_new, m_adv, rhoE_new, alpha_new = mat_result
        aux = {}
    alpha_new = np.clip(alpha_new, 1.0e-12, 1.0 - 1.0e-12)
    if pressure_closure in ('implicit_energy', 'apec_pe'):
        u_new, p_new = _solve_acoustic_energy_ad(
            W_n, q1_new, q2_new, m_adv, aux["rhoE_adv"], alpha_new, dt,
            eos1, eos2, dx, bc_l, bc_r,
            u_inlet=u_inlet, p_inlet=p_inlet,
            mixture_kind=mixture_kind,
            alpha_pure_tol=alpha_pure_tol,
            primitive_scheme=primitive_scheme)
        _, _, Z = _phase_acoustic(
            W_n, eos1, eos2, mixture_kind=mixture_kind,
            alpha_pure_tol=alpha_pure_tol)
        p_f, u_f = _acoustic_faces_np(
            u_new, p_new, Z, bc_l, bc_r,
            u_inlet=u_inlet, p_inlet=p_inlet)
        rhoE_new = aux["rhoE_adv"] - dt * (p_f[1:] * u_f[1:] - p_f[:-1] * u_f[:-1]) / dx
    else:
        u_new, p_new = _solve_acoustic_ad(
            W_n, q1_new, q2_new, m_adv, alpha_new, dt,
            eos1, eos2, dx, bc_l, bc_r,
            u_inlet=u_inlet, p_inlet=p_inlet,
            mixture_kind=mixture_kind,
            alpha_pure_tol=alpha_pure_tol,
            primitive_scheme=primitive_scheme)
        if pressure_closure == 'compressive_recovery':
            recovery_mask = _compressive_pressure_mask(W_n)
            if alpha_pure_tol > 0.0:
                pure_tol = max(float(alpha_pure_tol), np.finfo(float).eps ** 0.25)
                recovery_mask = recovery_mask & ~_pure_material_cell_mask(W_n[0], pure_tol)
            if np.any(recovery_mask):
                p_recovered = _recover_pressure_from_total_energy(
                    q1_new, q2_new, rhoE_new, alpha_new, u_new, p_new, eos1, eos2)
                p_new = np.where(recovery_mask, p_recovered, p_new)
        elif pressure_closure == 'pressure_work_consistent':
            _, _, Z = _phase_acoustic(
                W_n, eos1, eos2, mixture_kind=mixture_kind,
                alpha_pure_tol=alpha_pure_tol)
            p_f, u_f = _acoustic_faces_np(
                u_new, p_new, Z, bc_l, bc_r,
                u_inlet=u_inlet, p_inlet=p_inlet)
            rhoE_new = aux["rhoE_adv"] - dt * (
                p_f[1:] * u_f[1:] - p_f[:-1] * u_f[:-1]) / dx
        elif pressure_closure == 'dual_entropy':
            entropy_mask = _compressive_pressure_mask(W_n)
            p_entropy = _entropy_pressure_estimate(
                W_n, q1_new, q2_new, alpha_new, p_new, eos1, eos2)
            p_new = np.where(entropy_mask, p_entropy, p_new)
    u_new, vacuum_velocity_mask = _regularize_near_vacuum_velocity(
        W_n, q1_new, q2_new, u_new, p_new, eos1, eos2,
        mixture_kind=mixture_kind,
        alpha_pure_tol=alpha_pure_tol,
        bc_l=bc_l, bc_r=bc_r)
    rho1_new = q1_new / np.maximum(alpha_new, 1.0e-12)
    rho2_new = q2_new / np.maximum(1.0 - alpha_new, 1.0e-12)
    e1_new = eos1.energy(rho1_new, p_new)
    e2_new = eos2.energy(rho2_new, p_new)
    T1_new = eos1.temperature(rho1_new, e1_new)
    T2_new = eos2.temperature(rho2_new, e2_new)
    W_new = (alpha_new, T1_new, T2_new, u_new, p_new)
    return W_new, {
        'scheme': 'imex_ad_autograd_acoustic',
        'pressure_closure': pressure_closure,
        'primitive_scheme': primitive_scheme,
        'kapila_source_mode': kapila_source_mode,
        'material_energy_form': material_energy_form,
        'vacuum_velocity_cells': int(np.count_nonzero(vacuum_velocity_mask)),
    }
