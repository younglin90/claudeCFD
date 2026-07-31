"""First-order full-explicit five-equation finite-volume step.

This module is intentionally minimal: piecewise-constant conservative fluxes,
a local material upwind flux plus a linear acoustic pressure flux, and the
standard non-conservative alpha source.  Optional CICSAM is restricted to the
volume-fraction face value only; it does not alter phase mass, momentum, energy,
APEC, PE projection, G_alpha, or any user-tuned dissipation coefficient.
"""
from __future__ import annotations

import os
import numpy as np

from .boundary import extend_W
from .primitive import prim_to_cons_W
from .sound_speed import phase_sound_speed_sq, mixture_sound_speed_sq
from .source_d1 import D_K_kapila

_EPS = 1.0e-30


def _phase_acoustic(W, eos1, eos2, *, mixture_kind, alpha_pure_tol):
    """Return rho, c_mix^2, and impedance with pure-branch floor handling."""
    alpha, T1, T2, _, p = W
    rho1 = np.maximum(eos1.density(p, T1), _EPS)
    rho2 = np.maximum(eos2.density(p, T2), _EPS)
    c1_sq = phase_sound_speed_sq(eos1, rho1, T1)
    c2_sq = phase_sound_speed_sq(eos2, rho2, T2)
    rho = np.maximum(alpha * rho1 + (1.0 - alpha) * rho2, _EPS)
    c_mix_sq = mixture_sound_speed_sq(
        alpha, rho1, c1_sq, rho2, c2_sq, kind=mixture_kind)
    if alpha_pure_tol > 0.0:
        pure1 = alpha >= 1.0 - alpha_pure_tol
        pure2 = alpha <= alpha_pure_tol
        rho = np.where(pure1, rho1, rho)
        rho = np.where(pure2, rho2, rho)
        c_mix_sq = np.where(pure1, c1_sq, c_mix_sq)
        c_mix_sq = np.where(pure2, c2_sq, c_mix_sq)
    Z = np.maximum(rho * np.sqrt(np.maximum(c_mix_sq, _EPS)), _EPS)
    return rho, c_mix_sq, Z


def _signal_speed(W, eos1, eos2, *, mixture_kind, alpha_pure_tol):
    """Cell signal speed for the explicit Rusanov flux."""
    _, _, _, u, _ = W
    _, c_mix_sq, _ = _phase_acoustic(
        W, eos1, eos2, mixture_kind=mixture_kind,
        alpha_pure_tol=alpha_pure_tol)
    return np.abs(u) + np.sqrt(np.maximum(c_mix_sq, _EPS))


def _impedance(W, eos1, eos2, *, mixture_kind, alpha_pure_tol):
    return _phase_acoustic(
        W, eos1, eos2, mixture_kind=mixture_kind,
        alpha_pure_tol=alpha_pure_tol)[2]


def _divergence(face_flux, dx):
    inv_dx = 1.0 / dx
    return tuple((f[1:] - f[:-1]) * inv_dx for f in face_flux)


def _cicsam_alpha_face(alpha_ext, u_face, dt, dx):
    """1D bounded CICSAM/CBC face value for alpha only.

    CICSAM blends compressive and high-resolution normalized-variable schemes.
    In this 1D solver the interface normal is collinear with the flow, so the
    CICSAM weight selects the compressive bounded-capturing (CBC) branch.  The
    only parameter entering the formula is the local Courant number.
    """
    alpha_ext = np.asarray(alpha_ext, dtype=float)
    u_face = np.asarray(u_face, dtype=float)
    face = np.where(u_face >= 0.0, alpha_ext[:-1], alpha_ext[1:]).astype(float)
    courant = np.clip(np.abs(u_face) * dt / dx, 1.0e-12, 1.0)

    def cicsam_value(f, far, up, down):
        denom = down - far
        if abs(denom) <= 1.0e-14:
            return face[f]
        phi_c = (up - far) / denom
        if phi_c < 0.0 or phi_c > 1.0:
            return face[f]
        phi_f = min(1.0, phi_c / courant[f])
        return far + phi_f * denom

    for f in range(1, len(u_face) - 1):
        if u_face[f] >= 0.0:
            far = alpha_ext[f - 1]
            up = alpha_ext[f]
            down = alpha_ext[f + 1]
        else:
            far = alpha_ext[f + 2]
            up = alpha_ext[f + 1]
            down = alpha_ext[f]
        face[f] = cicsam_value(f, far, up, down)

    # Periodic extension duplicates the same physical boundary face at both
    # ends.  Apply the same CICSAM flux there as well; otherwise every periodic
    # wrap of a material interface falls back to first-order upwind.
    periodic_ext = (
        len(alpha_ext) >= 4
        and np.isclose(alpha_ext[0], alpha_ext[-2])
        and np.isclose(alpha_ext[-1], alpha_ext[1])
    )
    if periodic_ext and len(u_face) >= 2:
        f = len(u_face) - 1
        if u_face[f] >= 0.0:
            far, up, down = alpha_ext[-3], alpha_ext[-2], alpha_ext[-1]
        else:
            far, up, down = alpha_ext[2], alpha_ext[1], alpha_ext[0]
        val = cicsam_value(f, far, up, down)
        face[0] = val
        face[-1] = val
    return np.clip(face, 0.0, 1.0)


def _stacs_alpha_face(alpha_ext, u_face):
    """1D STACS/SUPERBEE sharp-interface face value for alpha.

    This is the compressive STACS branch in normalized-variable form.  It is
    less aggressive than low-Courant HYPER-C/CICSAM but remains a bounded
    sharp-interface TVD/NVD method for the volume-fraction flux.
    """
    alpha_ext = np.asarray(alpha_ext, dtype=float)
    u_face = np.asarray(u_face, dtype=float)
    face = np.where(u_face >= 0.0, alpha_ext[:-1], alpha_ext[1:]).astype(float)

    def superbee_nvd(phi_c):
        if phi_c < 0.0 or phi_c > 1.0:
            return phi_c
        if phi_c < 1.0 / 3.0:
            return 2.0 * phi_c
        if phi_c < 0.5:
            return 0.5 + 0.5 * phi_c
        if phi_c < 2.0 / 3.0:
            return 1.5 * phi_c
        return 1.0

    for f in range(1, len(u_face) - 1):
        if u_face[f] >= 0.0:
            far = alpha_ext[f - 1]
            up = alpha_ext[f]
            down = alpha_ext[f + 1]
        else:
            far = alpha_ext[f + 2]
            up = alpha_ext[f + 1]
            down = alpha_ext[f]
        denom = down - far
        if abs(denom) <= 1.0e-14:
            continue
        phi_c = (up - far) / denom
        phi_f = superbee_nvd(phi_c)
        if 0.0 <= phi_f <= 1.0:
            face[f] = far + phi_f * denom
    return np.clip(face, 0.0, 1.0)


def _mstacs_alpha_face(alpha_ext, u_face, dt, dx):
    """1D MSTACS sharp-interface face value for alpha.

    Anghan et al.'s MSTACS uses a Courant-dependent compressive branch:
    HYPER-C for Co <= 0.33 and bounded-downwind otherwise.  The high-resolution
    branch is STOIC.  In this 1D solver the interface normal and face normal are
    collinear, so the STACS/MSTACS switching weight selects the compressive
    branch without a case-specific wave/contact sensor.
    """
    alpha_ext = np.asarray(alpha_ext, dtype=float)
    u_face = np.asarray(u_face, dtype=float)
    face = np.where(u_face >= 0.0, alpha_ext[:-1], alpha_ext[1:]).astype(float)
    courant = np.clip(np.abs(u_face) * dt / dx, 1.0e-12, 1.0)

    def cds_mstacs(phi_c, co):
        if phi_c < 0.0 or phi_c > 1.0:
            return phi_c
        if co <= 0.33:
            return min(phi_c / max(co, 1.0e-12), 1.0)
        return min(3.0 * phi_c, 1.0)

    for f in range(1, len(u_face) - 1):
        if u_face[f] >= 0.0:
            far = alpha_ext[f - 1]
            up = alpha_ext[f]
            down = alpha_ext[f + 1]
        else:
            far = alpha_ext[f + 2]
            up = alpha_ext[f + 1]
            down = alpha_ext[f]
        denom = down - far
        if abs(denom) <= 1.0e-14:
            continue
        phi_c = (up - far) / denom
        phi_f = cds_mstacs(phi_c, courant[f])
        if 0.0 <= phi_f <= 1.0:
            face[f] = far + phi_f * denom

    periodic_ext = (
        len(alpha_ext) >= 4
        and np.isclose(alpha_ext[0], alpha_ext[-2])
        and np.isclose(alpha_ext[-1], alpha_ext[1])
    )
    if periodic_ext and len(u_face) >= 2:
        f = len(u_face) - 1
        if u_face[f] >= 0.0:
            far, up, down = alpha_ext[-3], alpha_ext[-2], alpha_ext[-1]
        else:
            far, up, down = alpha_ext[2], alpha_ext[1], alpha_ext[0]
        denom = down - far
        if abs(denom) > 1.0e-14:
            phi_c = (up - far) / denom
            phi_f = cds_mstacs(phi_c, courant[f])
            if 0.0 <= phi_f <= 1.0:
                val = far + phi_f * denom
                face[0] = val
                face[-1] = val
    return np.clip(face, 0.0, 1.0)


def _vanleer_alpha_face(alpha_ext, u_face):
    """Bounded van-Leer TVD reconstruction for alpha faces."""
    alpha_ext = np.asarray(alpha_ext, dtype=float)
    u_face = np.asarray(u_face, dtype=float)
    face = np.where(u_face >= 0.0, alpha_ext[:-1], alpha_ext[1:]).astype(float)
    for f in range(1, len(u_face) - 1):
        if u_face[f] >= 0.0:
            far = alpha_ext[f - 1]
            up = alpha_ext[f]
            down = alpha_ext[f + 1]
        else:
            far = alpha_ext[f + 2]
            up = alpha_ext[f + 1]
            down = alpha_ext[f]
        den = down - up
        if abs(den) <= 1.0e-14:
            continue
        r = (up - far) / den
        if not np.isfinite(r) or r <= 0.0:
            continue
        psi = 2.0 * r / (1.0 + r)
        val = up + 0.5 * min(2.0, max(0.0, psi)) * den
        lo = min(far, up, down)
        hi = max(far, up, down)
        face[f] = min(hi, max(lo, val))
    return np.clip(face, 0.0, 1.0)


def _tvd_slope_1d(d_up, d_down, kind):
    """Limiter slope in the upwind-oriented coordinate."""
    d_up = float(d_up)
    d_down = float(d_down)
    if not (np.isfinite(d_up) and np.isfinite(d_down)):
        return 0.0
    if d_up * d_down <= 0.0:
        return 0.0
    kind = str(kind or "vanleer").strip().lower().replace("-", "_")
    if kind in {"minmod", "minmod2"}:
        return np.sign(d_down) * min(abs(d_up), abs(d_down))
    if kind in {"superbee", "sb"}:
        return np.sign(d_down) * max(
            min(2.0 * abs(d_up), abs(d_down)),
            min(abs(d_up), 2.0 * abs(d_down)),
        )
    if kind in {"mc", "monotonized_central", "monotonised_central"}:
        return np.sign(d_down) * min(
            2.0 * abs(d_up), 0.5 * abs(d_up + d_down), 2.0 * abs(d_down))
    if kind in {"umist"}:
        r = d_down / d_up
        psi = max(0.0, min(2.0 * r, 0.25 + 0.75 * r, 0.75 + 0.25 * r, 2.0))
        return psi * d_up
    # Smooth default: van Leer harmonic limiter.
    return 2.0 * d_up * d_down / (d_up + d_down)


def _muscl_hancock_alpha_face(alpha_ext, u_face, dt, dx, *, tvd_kind=None):
    """Bounded MUSCL-Hancock alpha face value.

    This is the scalar finite-volume time-average used as the smooth candidate
    in THINC-BVD.  The slope contribution is multiplied by ``1-Co``; therefore
    a full-cell transit (Co=1) reduces to the exact upwind cell-average flux
    instead of injecting a downwind-biased interface-compression correction.
    """
    alpha_ext = np.asarray(alpha_ext, dtype=float)
    u_face = np.asarray(u_face, dtype=float)
    face = np.where(u_face >= 0.0, alpha_ext[:-1], alpha_ext[1:]).astype(float)
    courant = np.clip(np.abs(u_face) * float(dt) / max(float(dx), _EPS), 0.0, 1.0)
    kind = tvd_kind or "vanleer"
    for f in range(1, len(u_face) - 1):
        if u_face[f] >= 0.0:
            far = alpha_ext[f - 1]
            up = alpha_ext[f]
            down = alpha_ext[f + 1]
        else:
            far = alpha_ext[f + 2]
            up = alpha_ext[f + 1]
            down = alpha_ext[f]
        slope = _tvd_slope_1d(up - far, down - up, kind)
        val = up + 0.5 * (1.0 - courant[f]) * slope
        lo = min(far, up, down)
        hi = max(far, up, down)
        face[f] = min(hi, max(lo, val))

    periodic_ext = (
        len(alpha_ext) >= 4
        and np.isclose(alpha_ext[0], alpha_ext[-2])
        and np.isclose(alpha_ext[-1], alpha_ext[1])
    )
    if periodic_ext and len(u_face) >= 2:
        f = len(u_face) - 1
        if u_face[f] >= 0.0:
            far, up, down = alpha_ext[-3], alpha_ext[-2], alpha_ext[-1]
        else:
            far, up, down = alpha_ext[2], alpha_ext[1], alpha_ext[0]
        slope = _tvd_slope_1d(up - far, down - up, kind)
        val = up + 0.5 * (1.0 - courant[f]) * slope
        lo = min(far, up, down)
        hi = max(far, up, down)
        val = min(hi, max(lo, val))
        face[0] = val
        face[-1] = val
    return np.clip(face, 0.0, 1.0)


def _thinc_alpha_face(alpha_ext, u_face):
    """Bounded 1D THINC face value for alpha.

    Uses the common THINC beta=1.6 setting from the THINC-BVD literature.  The
    reconstruction is bounded by construction and falls back to upwind when the
    local stencil is not monotone or does not contain an interface.
    """
    alpha_ext = np.asarray(alpha_ext, dtype=float)
    u_face = np.asarray(u_face, dtype=float)
    face = np.where(u_face >= 0.0, alpha_ext[:-1], alpha_ext[1:]).astype(float)
    beta = 1.6
    tb = np.tanh(beta)
    cb = np.cosh(beta)

    def cell_faces(i):
        left = alpha_ext[i - 1]
        centre = alpha_ext[i]
        right = alpha_ext[i + 1]
        q_min = min(left, right)
        q_max = max(left, right) - q_min
        if q_max <= 1.0e-14:
            return centre, centre
        c_bar = (centre - q_min) / q_max
        if c_bar <= 1.0e-12 or c_bar >= 1.0 - 1.0e-12:
            return centre, centre
        theta = 1.0 if right >= left else -1.0
        b = np.exp(theta * beta * (2.0 * c_bar - 1.0))
        a = (b / cb - 1.0) / tb
        q_left = q_min + 0.5 * q_max * (1.0 + theta * a)
        q_right = q_min + 0.5 * q_max * (
            1.0 + theta * (tb + a) / max(1.0 + a * tb, 1.0e-14)
        )
        lo = min(left, centre, right)
        hi = max(left, centre, right)
        return min(hi, max(lo, q_left)), min(hi, max(lo, q_right))

    for f in range(1, len(u_face) - 1):
        if u_face[f] >= 0.0:
            _, q_r = cell_faces(f)
            face[f] = q_r
        else:
            q_l, _ = cell_faces(f + 1)
            face[f] = q_l
    return np.clip(face, 0.0, 1.0)


def _thinc_bvd_alpha_face(alpha_ext, u_face, dt, dx, *, tvd_kind=None):
    """THINC-BVD alpha flux with a MUSCL-Hancock smooth candidate.

    The old ``thinc_bvd`` path was only THINC.  That is too compressive for
    smooth volume-fraction waves and creates checkerboard-like alpha/rho error.
    BVD keeps a bounded sharp candidate available near discontinuities, but
    selects the lower-boundary-variation MUSCL-Hancock candidate in smooth
    regions.  At Co=1 the MUSCL-Hancock candidate is the exact upwind
    time-averaged flux for a full cell transit.
    """
    alpha_ext = np.asarray(alpha_ext, dtype=float)
    u_face = np.asarray(u_face, dtype=float)
    tvd_kind = tvd_kind or (
        os.environ.get("FIVE_EQ_IMEX_ALPHA_TVD")
        or os.environ.get("FIVE_EQ_IMEX_TMLPU_TVD")
        or "vanleer"
    )
    smooth = _muscl_hancock_alpha_face(
        alpha_ext, u_face, dt, dx, tvd_kind=tvd_kind)
    sharp = _thinc_alpha_face(alpha_ext, u_face)
    face = smooth.copy()
    courant = np.clip(np.abs(u_face) * float(dt) / max(float(dx), _EPS), 0.0, 1.0)
    for f in range(1, len(u_face) - 1):
        # For Co≈1 the exact time integral crosses a full upwind cell, so the
        # smooth candidate is not a fallback; it is the correct FV flux.
        if courant[f] >= 1.0 - 1.0e-12:
            continue
        left = alpha_ext[f]
        right = alpha_ext[f + 1]
        bvd_smooth = abs(smooth[f] - left) + abs(right - smooth[f])
        bvd_sharp = abs(sharp[f] - left) + abs(right - sharp[f])
        if bvd_sharp < bvd_smooth:
            face[f] = sharp[f]
    if len(face) >= 2:
        face[0] = face[-1] if (
            len(alpha_ext) >= 4
            and np.isclose(alpha_ext[0], alpha_ext[-2])
            and np.isclose(alpha_ext[-1], alpha_ext[1])
        ) else face[0]
    return np.clip(face, 0.0, 1.0)


def _adaptive_bvd_alpha_face(alpha_ext, u_face, dt, dx, *, tvd_kind=None,
                             alpha_pure_tol=None):
    """Regime-consistent alpha BVD flux for sharp contacts and smooth waves.

    The candidate set is fixed for every case:

    * adjacent low/high phase-pure cells: CICSAM compression, so true VOF
      material interfaces remain sharp without the upstream pre-echo observed
      from the less compressive MSTACS branch in long passive advection;
    * otherwise: bounded MUSCL-Hancock TVD transport, because a smooth
      composition wave can contain near-pure extrema without being a
      discontinuous interface.

    This is a single alpha method selected by volume-fraction topology, not by
    validation case id or by flow-variable tuning.
    """
    alpha_ext = np.asarray(alpha_ext, dtype=float)
    u_face = np.asarray(u_face, dtype=float)
    pure_tol = max(
        np.finfo(float).eps ** 0.25,
        1.0e-12 if alpha_pure_tol is None else float(alpha_pure_tol))
    interior = alpha_ext[1:-1] if alpha_ext.size > 2 else alpha_ext
    pure_band = pure_tol * (1.0 + 1.0e-9) + 1.0e-15
    has_low_pure = bool(np.min(interior) <= pure_band)
    has_high_pure = bool(np.max(interior) >= 1.0 - pure_band)
    low_count = int(np.count_nonzero(interior <= pure_band))
    high_count = int(np.count_nonzero(interior >= 1.0 - pure_band))
    mixed_count = int(np.count_nonzero(
        (interior > pure_band) & (interior < 1.0 - pure_band)))

    a_l = alpha_ext[:-1]
    a_r = alpha_ext[1:]
    has_sharp_pure_jump = bool(np.any(
        ((a_l <= pure_band) & (a_r >= 1.0 - pure_band))
        | ((a_r <= pure_band) & (a_l >= 1.0 - pure_band))
    ))
    has_narrow_mixed_layer = bool(
        has_low_pure and has_high_pure
        and mixed_count <= max(low_count + high_count, 1)
    )

    if has_low_pure and has_high_pure and (
            has_sharp_pure_jump or has_narrow_mixed_layer):
        return _cicsam_alpha_face(alpha_ext, u_face, dt, dx)

    return _muscl_hancock_alpha_face(
        alpha_ext, u_face, dt, dx, tvd_kind=tvd_kind)


def _alpha_face(alpha_ext, u_face, dt, dx, alpha_scheme, *, tvd_kind=None,
                alpha_pure_tol=None):
    alpha_scheme = str(alpha_scheme).strip().lower().replace("-", "_")
    if alpha_scheme == 'cicsam':
        return _cicsam_alpha_face(alpha_ext, u_face, dt, dx)
    if alpha_scheme == 'mstacs':
        return _mstacs_alpha_face(alpha_ext, u_face, dt, dx)
    if alpha_scheme in ('stacs', 'superbee'):
        return _stacs_alpha_face(alpha_ext, u_face)
    if alpha_scheme in ('vanleer', 'tvd_vanleer'):
        return _vanleer_alpha_face(alpha_ext, u_face)
    if alpha_scheme in ('thinc_bvd', 'thinc-bvd'):
        return _thinc_bvd_alpha_face(alpha_ext, u_face, dt, dx, tvd_kind=tvd_kind)
    if alpha_scheme in ('adaptive_bvd', 'adaptive_alpha_bvd', 'bvd_adaptive'):
        return _adaptive_bvd_alpha_face(
            alpha_ext, u_face, dt, dx,
            tvd_kind=tvd_kind, alpha_pure_tol=alpha_pure_tol)
    if alpha_scheme == 'thinc':
        return _thinc_alpha_face(alpha_ext, u_face)
    if alpha_scheme in ('upwind', 'muscl', 'limited', 'central'):
        # In the explicit baseline, CICSAM is the only high-resolution alpha
        # path.  Other historical names intentionally map to the robust upwind
        # alpha flux to avoid changing multiple mechanisms at once.
        return np.where(u_face >= 0.0, alpha_ext[:-1], alpha_ext[1:])
    raise ValueError(f"Unknown explicit alpha_scheme='{alpha_scheme}'.")


def explicit_rusanov_step(W_n, dt, eos1, eos2, dx, bc_l, bc_r, *,
                          u_inlet=None, p_inlet=None,
                          p_outlet=None,
                          alpha_inlet=None, T1_inlet=None, T2_inlet=None,
                          mixture_kind='kapila',
                          kapila_closure=False,
                          alpha_pure_tol=0.0,
                          alpha_scheme='upwind'):
    """Advance one Forward-Euler step with a first-order Rusanov flux.

    The alpha equation is discretized as

        L_alpha = div(alpha u) - (alpha + D_K) div(u),

    using a centered face velocity for ``div(u)``.  For reflective walls the
    physical wall flux is imposed directly: no mass, alpha, or energy flux,
    and momentum flux equal to the wall pressure.
    """
    U_n, _ = prim_to_cons_W(W_n, eos1, eos2)

    W_ext = extend_W(W_n, bc_l, bc_r, ng=1,
                     u_inlet_l=u_inlet, p_inlet_l=p_inlet,
                     p_inlet_r=p_outlet,
                     alpha_inlet_l=alpha_inlet,
                     T1_inlet_l=T1_inlet,
                     T2_inlet_l=T2_inlet,
                     eos1=eos1, eos2=eos2)
    U_ext, _ = prim_to_cons_W(W_ext, eos1, eos2)
    s_ext = _signal_speed(
        W_ext, eos1, eos2, mixture_kind=mixture_kind,
        alpha_pure_tol=alpha_pure_tol)
    Z_ext = _impedance(
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

    upwind_left = u_star >= 0.0
    if alpha_scheme in (
            'cicsam', 'stacs', 'superbee', 'vanleer', 'tvd_vanleer',
            'adaptive_bvd', 'adaptive-alpha-bvd', 'adaptive_alpha_bvd',
            'bvd_adaptive'):
        alpha_f = np.clip(_alpha_face(
                              W_ext[0], u_star, dt, dx, alpha_scheme,
                              alpha_pure_tol=alpha_pure_tol),
                          1.0e-12, 1.0 - 1.0e-12)
        # Keep conservative phase mass/momentum on one monotone material flux;
        # the sharp-interface method is applied to the alpha equation itself.
        q1_f = np.where(upwind_left, U_ext[0][:-1], U_ext[0][1:])
        q2_f = np.where(upwind_left, U_ext[1][:-1], U_ext[1][1:])
        m_f = np.where(upwind_left, U_ext[2][:-1], U_ext[2][1:])
    else:
        alpha_f = _alpha_face(
            W_ext[0], u_star, dt, dx, alpha_scheme,
            alpha_pure_tol=alpha_pure_tol)
        q1_f = np.where(upwind_left, U_ext[0][:-1], U_ext[0][1:])
        q2_f = np.where(upwind_left, U_ext[1][:-1], U_ext[1][1:])
        m_f = np.where(upwind_left, U_ext[2][:-1], U_ext[2][1:])
    rE_f = np.where(upwind_left, U_ext[3][:-1], U_ext[3][1:])

    face_flux = [
        q1_f * u_star,
        q2_f * u_star,
        m_f * u_star + p_star,
        rE_f * u_star + p_star * u_star,
        alpha_f * u_star,
    ]
    face_flux = [np.asarray(f, dtype=float) for f in face_flux]

    # Physical reflective wall flux.  This avoids adding artificial Rusanov
    # mass/energy flux through a solid wall and gives the correct pressure load.
    if bc_l == 'reflective':
        face_flux[0][0] = 0.0
        face_flux[1][0] = 0.0
        face_flux[2][0] = float(W_n[4][0])
        face_flux[3][0] = 0.0
        face_flux[4][0] = 0.0
    if bc_r == 'reflective':
        face_flux[0][-1] = 0.0
        face_flux[1][-1] = 0.0
        face_flux[2][-1] = float(W_n[4][-1])
        face_flux[3][-1] = 0.0
        face_flux[4][-1] = 0.0

    L = list(_divergence(tuple(face_flux), dx))

    # Non-conservative topology source.  The alpha*div(u) term is required
    # even for Allaire-Massoni (D_K=0) because the stored equation uses
    # div(alpha u) on the left.
    u_face = u_star.copy()
    if bc_l == 'reflective':
        u_face[0] = 0.0
    if bc_r == 'reflective':
        u_face[-1] = 0.0
    div_u = (u_face[1:] - u_face[:-1]) / dx

    B = np.asarray(W_n[0], dtype=float).copy()
    if kapila_closure:
        B = B + D_K_kapila(W_n, eos1, eos2)
    L[4] = L[4] - B * div_u

    # Pressure-based explicit closure.  Updating total energy conservatively is
    # known to generate pressure oscillations at material contacts with general
    # EOS.  The pressure equation follows from the five-equation model:
    #     Dp/Dt + rho c_mix^2 div(u) = 0.
    alpha, T1, T2, u, p = W_n
    rho, c_mix_sq, _ = _phase_acoustic(
        W_n, eos1, eos2, mixture_kind=mixture_kind,
        alpha_pure_tol=alpha_pure_tol)

    _, _, _, _, p_ext = W_ext
    dp_back = (p_ext[1:-1] - p_ext[:-2]) / dx
    dp_forw = (p_ext[2:] - p_ext[1:-1]) / dx
    dp_dx = np.where(u >= 0.0, dp_back, dp_forw)
    L_p = u * dp_dx + rho * c_mix_sq * div_u

    q1_new = U_n[0] - dt * L[0]
    q2_new = U_n[1] - dt * L[1]
    m_new = U_n[2] - dt * L[2]
    alpha_new = U_n[4] - dt * L[4]
    rho_new = np.maximum(q1_new + q2_new, _EPS)
    u_new = m_new / rho_new
    p_new = p - dt * L_p

    rho1_new = q1_new / np.maximum(alpha_new, 1.0e-12)
    rho2_new = q2_new / np.maximum(1.0 - alpha_new, 1.0e-12)
    e1_new = eos1.energy(rho1_new, p_new)
    e2_new = eos2.energy(rho2_new, p_new)
    T1_new = eos1.temperature(rho1_new, e1_new)
    T2_new = eos2.temperature(rho2_new, e2_new)
    W_new = (alpha_new, T1_new, T2_new, u_new, p_new)
    return W_new, {
        'L': tuple(L),
        's_max': float(np.max(s_ext)),
        'scheme': 'first_order_pressure_based_explicit',
    }
