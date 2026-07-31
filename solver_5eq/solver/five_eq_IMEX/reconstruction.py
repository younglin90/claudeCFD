"""Primitive-variable reconstruction helpers for the 1D five-equation solver.

The active solver is one-dimensional and cell-centred.  The T-MLP-u formulas
provided by the user are therefore reduced to their uniform-grid 1D form:

    phi_f = phi_L + 0.5 * psi * (phi_R - phi_L)

where ``psi`` is a TVD limiter additionally clipped by a local three-cell
maximum-principle bound.  This keeps the useful T-MLP-u range ``0 <= psi <= 2``
without introducing new primitive-variable extrema.
"""
from __future__ import annotations

import os
import numpy as np

_EPS = 1.0e-30

_TVD_PRIMITIVE_SCHEMES = {
    "minmod",
    "superbee",
    "mc",
    "vanleer",
    "vanalbada",
    "umist",
}

def normalise_primitive_scheme(scheme):
    key = str(scheme or "upwind").strip().lower().replace("-", "_")
    aliases = {
        "first_order": "upwind",
        "lo": "upwind",
        "t_mlp_u": "tmlpu",
        "t_mlpu": "tmlpu",
        "t_mlp": "tmlpu",
        "t-mlp-u": "tmlpu",
        "t-mlp": "tmlpu",
        "sb": "superbee",
        "van_leer": "vanleer",
        "monotonized_central": "mc",
        "monotonised_central": "mc",
        "van_albada": "vanalbada",
        "albada": "vanalbada",
    }
    key = aliases.get(key, key)
    if key not in {"upwind", "central", "tmlpu", "weno3", *_TVD_PRIMITIVE_SCHEMES}:
        raise ValueError(f"Unknown primitive_scheme='{scheme}'.")
    return key


def is_tvd_primitive_scheme(scheme):
    return normalise_primitive_scheme(scheme) in _TVD_PRIMITIVE_SCHEMES


def primitive_tvd_kind(scheme):
    """Limiter kind used by primitive reconstruction.

    ``tmlpu`` is the bounded T-MLP-u wrapper and gets its TVD limiter from
    FIVE_EQ_IMEX_TMLPU_TVD.  Direct schemes such as ``superbee`` or
    ``vanleer`` use that limiter by name without enabling T-MLP-u.
    """
    scheme = normalise_primitive_scheme(scheme)
    if scheme == "tmlpu":
        return os.environ.get("FIVE_EQ_IMEX_TMLPU_TVD", "vanleer")
    if scheme in _TVD_PRIMITIVE_SCHEMES:
        return scheme
    return os.environ.get("FIVE_EQ_IMEX_TMLPU_TVD", "vanleer")


def _tvd_limiter(r, kind):
    r = float(r)
    if not np.isfinite(r) or r <= 0.0:
        return 0.0
    kind = str(kind or "minmod").strip().lower().replace("-", "_")
    if kind in ("minmod", "minmod2"):
        psi = min(1.0, r)
    elif kind in ("superbee", "sb"):
        psi = max(min(2.0 * r, 1.0), min(r, 2.0), 0.0)
    elif kind in ("mc", "monotonized_central"):
        psi = max(0.0, min(2.0 * r, 0.5 * (1.0 + r), 2.0))
    elif kind in ("vanalbada", "van_albada", "albada"):
        psi = (r * r + r) / (r * r + 1.0)
    elif kind in ("umist",):
        psi = max(0.0, min(2.0 * r, 0.25 + 0.75 * r, 0.75 + 0.25 * r, 2.0))
    else:
        # van Leer limiter.  It naturally reaches the downwind side of the TVD
        # range for large r while remaining smooth near r=1.
        psi = 2.0 * r / (1.0 + r)
    return max(0.0, min(2.0, float(psi)))


def _limited_value(phi_LL, phi_L, phi_R, *, tvd_kind, courant=None):
    """Return a TVD-limited value from cell L to the L/R face."""
    phi_LL = float(phi_LL)
    phi_L = float(phi_L)
    phi_R = float(phi_R)
    if not (np.isfinite(phi_LL) and np.isfinite(phi_L) and np.isfinite(phi_R)):
        return phi_L
    num = phi_R - phi_L
    den = phi_L - phi_LL
    if abs(num) <= 1.0e-300:
        return phi_L
    if abs(den) <= 1.0e-300:
        return phi_L
    r = num / den
    psi = _tvd_limiter(r, tvd_kind)
    if psi <= 0.0:
        return phi_L

    delta = 0.5 * num
    if courant is not None:
        # MUSCL-Hancock time centering for scalar advection.  The slope
        # contribution is multiplied by (1-C); at C=1 a full cell average
        # crosses the face and the exact finite-volume flux is first upwind
        # in value while the method remains second-order for C<1.
        c = min(1.0, max(0.0, abs(float(courant))))
        delta *= (1.0 - c)
    lo = min(phi_LL, phi_L, phi_R)
    hi = max(phi_LL, phi_L, phi_R)
    if delta > 0.0:
        psi_bound = (hi - phi_L) / max(delta, _EPS)
    elif delta < 0.0:
        psi_bound = (lo - phi_L) / min(delta, -_EPS)
    else:
        psi_bound = 0.0
    if not np.isfinite(psi_bound):
        psi_bound = 0.0
    psi = max(0.0, min(2.0, psi, psi_bound))
    return min(hi, max(lo, phi_L + psi * delta))


def _weno3_value(phi_LL, phi_L, phi_R):
    """Bounded third-order WENO value from cell L to the L/R face."""
    phi_LL = float(phi_LL)
    phi_L = float(phi_L)
    phi_R = float(phi_R)
    if not (np.isfinite(phi_LL) and np.isfinite(phi_L) and np.isfinite(phi_R)):
        return phi_L
    q0 = 1.5 * phi_L - 0.5 * phi_LL
    q1 = 0.5 * (phi_L + phi_R)
    beta0 = (phi_L - phi_LL) ** 2
    beta1 = (phi_R - phi_L) ** 2
    eps = 1.0e-12 * max(1.0, abs(phi_LL), abs(phi_L), abs(phi_R)) ** 2
    a0 = (1.0 / 3.0) / (eps + beta0) ** 2
    a1 = (2.0 / 3.0) / (eps + beta1) ** 2
    val = (a0 * q0 + a1 * q1) / max(a0 + a1, _EPS)
    lo = min(phi_LL, phi_L, phi_R)
    hi = max(phi_LL, phi_L, phi_R)
    return min(hi, max(lo, float(val)))


def _pressure_discontinuity_fallback(p_ext):
    p_ext = np.asarray(p_ext, dtype=float)
    rel = np.abs(p_ext[1:] - p_ext[:-1]) / np.maximum(
        np.maximum(np.abs(p_ext[1:]), np.abs(p_ext[:-1])), 1.0)
    thresh = float(os.environ.get("FIVE_EQ_IMEX_TMLPU_PRESSURE_REL", "1e-3"))
    return rel > max(thresh, 0.0)


def reconstruct_upwind_faces(phi_ext, u_face, *, scheme="upwind", floor=None,
                             fallback_mask=None, tvd_kind=None, dt=None,
                             dx=None):
    """Reconstruct a scalar primitive to faces using the upwind cell as L."""
    scheme = normalise_primitive_scheme(scheme)
    phi_ext = np.asarray(phi_ext, dtype=float)
    u_face = np.asarray(u_face, dtype=float)
    face = np.where(u_face >= 0.0, phi_ext[:-1], phi_ext[1:]).astype(float)
    if scheme == "central":
        face = 0.5 * (phi_ext[:-1] + phi_ext[1:])
    elif scheme in ("tmlpu", "weno3") or is_tvd_primitive_scheme(scheme):
        tvd_kind = primitive_tvd_kind(scheme) if tvd_kind is None else tvd_kind
        fallback_mask = np.zeros_like(u_face, dtype=bool) if fallback_mask is None else np.asarray(fallback_mask, dtype=bool)
        out = face.copy()
        n_face = len(u_face)
        n_ext = len(phi_ext)
        for f in range(n_face):
            if f < fallback_mask.size and fallback_mask[f]:
                continue
            if u_face[f] >= 0.0:
                i_ll, i_l, i_r = f - 1, f, f + 1
            else:
                i_ll, i_l, i_r = f + 2, f + 1, f
            if i_ll < 0 or i_ll >= n_ext or i_l < 0 or i_l >= n_ext or i_r < 0 or i_r >= n_ext:
                continue
            if scheme == "weno3":
                out[f] = _weno3_value(phi_ext[i_ll], phi_ext[i_l], phi_ext[i_r])
            else:
                courant = None
                if dt is not None and dx is not None:
                    courant = abs(float(u_face[f])) * float(dt) / max(float(dx), _EPS)
                out[f] = _limited_value(
                    phi_ext[i_ll], phi_ext[i_l], phi_ext[i_r],
                    tvd_kind=tvd_kind, courant=courant)
        face = out
    if floor is not None:
        face = np.maximum(face, float(floor))
    return face


def reconstruct_lr_faces(phi_ext, *, scheme="upwind", floor=None, tvd_kind=None):
    """Return left/right primitive states at each face for conservative solvers."""
    scheme = normalise_primitive_scheme(scheme)
    phi_ext = np.asarray(phi_ext, dtype=float)
    left = phi_ext[:-1].astype(float).copy()
    right = phi_ext[1:].astype(float).copy()
    if scheme == "central":
        mid = 0.5 * (left + right)
        left = mid.copy()
        right = mid.copy()
    elif scheme in ("tmlpu", "weno3") or is_tvd_primitive_scheme(scheme):
        tvd_kind = primitive_tvd_kind(scheme) if tvd_kind is None else tvd_kind
        n_face = len(left)
        n_ext = len(phi_ext)
        for f in range(n_face):
            # State reconstructed from ext[f] to its right face.
            if f - 1 >= 0:
                if scheme == "weno3":
                    left[f] = _weno3_value(phi_ext[f - 1], phi_ext[f], phi_ext[f + 1])
                else:
                    left[f] = _limited_value(phi_ext[f - 1], phi_ext[f], phi_ext[f + 1],
                                             tvd_kind=tvd_kind)
            # State reconstructed from ext[f+1] to its left face.
            if f + 2 < n_ext:
                if scheme == "weno3":
                    right[f] = _weno3_value(phi_ext[f + 2], phi_ext[f + 1], phi_ext[f])
                else:
                    right[f] = _limited_value(phi_ext[f + 2], phi_ext[f + 1], phi_ext[f],
                                              tvd_kind=tvd_kind)
    if floor is not None:
        left = np.maximum(left, float(floor))
        right = np.maximum(right, float(floor))
    return left, right


def reconstruct_primitive_upwind_faces(W_ext, u_face, *, scheme="upwind",
                                       dt=None, dx=None):
    """Return face primitives (T1, T2, u, p) using upwind-biased reconstruction."""
    _, T1_ext, T2_ext, u_ext, p_ext = W_ext
    # T-MLP-u's limiter is the shock/discontinuity control; do not add a
    # pressure-sensor fallback that would use different schemes in selected
    # regions of the same validation problem.
    fallback = None
    thermo_tvd = os.environ.get("FIVE_EQ_IMEX_THERMO_TVD")
    return {
        "T1": reconstruct_upwind_faces(T1_ext, u_face, scheme=scheme, floor=1.0,
                                       fallback_mask=fallback,
                                       tvd_kind=thermo_tvd, dt=dt, dx=dx),
        "T2": reconstruct_upwind_faces(T2_ext, u_face, scheme=scheme, floor=1.0,
                                       fallback_mask=fallback,
                                       tvd_kind=thermo_tvd, dt=dt, dx=dx),
        "u": reconstruct_upwind_faces(u_ext, u_face, scheme=scheme, floor=None,
                                      fallback_mask=fallback, dt=dt, dx=dx),
        "p": reconstruct_upwind_faces(p_ext, u_face, scheme=scheme, floor=1.0e-12,
                                      fallback_mask=fallback, dt=dt, dx=dx),
    }
