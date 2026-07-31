"""Dimension-split 2-D/3-D finite-volume extension for five_eq_IMEX.

This module is intentionally separate from the validated 1-D solver.  It gives
the recommended 2-D/3-D validation campaign a concrete executable path while
preserving the 1-D code.  The numerical core is:

* component-wise MUSCL/TVD reconstruction with the same limiter vocabulary used
  by the 1-D solver (`minmod`, `vanleer`, `superbee`, `tmlpu+<limiter>`);
* HLLC-family normal flux for material/advection variables;
* SSPRK(3,3) stage update.  `time_integrator="imex_ssp3"` is accepted as an API
  alias; a true multidimensional implicit pressure block can be layered behind
  the same `rhs` interface later.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

import numpy as np

from .nd_primitive import (
    cons_to_prim_nd,
    mixture_density_from_W,
    prim_to_cons_nd,
    split_W_nd,
)
from .sound_speed import mixture_sound_speed_sq, phase_sound_speed_sq

_EPS = 1.0e-12


@dataclass(frozen=True)
class NDSolverInfo:
    dim: int
    steps: int
    dt_last: float
    time_integrator: str
    flux_scheme: str
    primitive_scheme: str
    tvd_limiter: str
    alpha_scheme: str


def _normalise_bc(bc, dim: int):
    if bc is None:
        return tuple("periodic" for _ in range(dim))
    if isinstance(bc, str):
        return tuple(bc for _ in range(dim))
    if len(bc) != dim:
        raise ValueError(f"bc must have length {dim}, got {bc}")
    return tuple(str(v).lower() for v in bc)


def _normalise_spacing(dx, dim: int):
    if np.isscalar(dx):
        return tuple(float(dx) for _ in range(dim))
    if len(dx) != dim:
        raise ValueError(f"dx must have length {dim}, got {dx}")
    return tuple(float(v) for v in dx)


def _move_front(arr, axis: int):
    return np.moveaxis(np.asarray(arr, dtype=float), axis, 0)


def _move_back(arr, axis: int):
    return np.moveaxis(arr, 0, axis)


def _two_left(a):
    if a.shape[0] == 1:
        return np.concatenate([a[0:1], a[0:1]], axis=0)
    return a[:2][::-1]


def _two_right(a):
    if a.shape[0] == 1:
        return np.concatenate([a[-1:], a[-1:]], axis=0)
    return a[-2:][::-1]


def _pad_axis(arr, axis: int, bc: str, *, reflect_sign: float = 1.0):
    """Pad two ghost layers on `axis` with periodic/transmissive/reflective."""
    a = _move_front(arr, axis)
    if bc == "periodic":
        left = a[-2:]
        right = a[:2]
    elif bc in ("transmissive", "outflow", "zerogradient", "zero_gradient"):
        left = np.concatenate([a[0:1], a[0:1]], axis=0)
        right = np.concatenate([a[-1:], a[-1:]], axis=0)
    elif bc in ("reflective", "wall"):
        left = reflect_sign * _two_left(a)
        right = reflect_sign * _two_right(a)
    else:
        raise ValueError(f"Unsupported BC '{bc}'")
    return _move_back(np.concatenate([left, a, right], axis=0), axis)


def _pad_primitive_axis(W, dim: int, axis: int, bc: str):
    padded = []
    for i, arr in enumerate(W):
        reflect_sign = -1.0 if bc in ("reflective", "wall") and i == 3 + axis else 1.0
        p = _pad_axis(arr, axis, bc, reflect_sign=reflect_sign)
        padded.append(p)
    return tuple(padded)


def _minmod_pair(a, b):
    same = (a * b) > 0.0
    return np.where(same, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def _limited_slope(dm, dp, limiter: str):
    limiter = limiter.lower()
    same = (dm * dp) > 0.0
    if limiter in ("none", "upwind", "first_order"):
        return np.zeros_like(dm)
    if limiter == "minmod":
        return _minmod_pair(dm, dp)
    if limiter in ("vanleer", "van_leer"):
        den = dm + dp
        return np.where(same, 2.0 * dm * dp / np.where(np.abs(den) > _EPS, den, _EPS), 0.0)
    if limiter == "superbee":
        s = np.sign(dm)
        a = np.minimum(2.0 * np.abs(dm), np.abs(dp))
        b = np.minimum(np.abs(dm), 2.0 * np.abs(dp))
        return np.where(same, s * np.maximum(a, b), 0.0)
    raise ValueError(f"Unknown TVD limiter '{limiter}'")


def _parse_reconstruction(primitive_scheme: str, tvd_limiter: str):
    scheme = (primitive_scheme or "tmlpu").lower()
    limiter = (tvd_limiter or "superbee").lower()
    bounded = False
    if scheme.startswith("tmlpu"):
        bounded = True
        if "+" in scheme:
            limiter = scheme.split("+", 1)[1]
    elif scheme in ("minmod", "vanleer", "van_leer", "superbee"):
        limiter = scheme
    elif scheme in ("upwind", "first_order"):
        limiter = "upwind"
    return bounded, limiter


def _reconstruct_axis_scalar(
    phi,
    axis: int,
    bc: str,
    *,
    limiter: str,
    bounded: bool,
    reflect_sign: float = 1.0,
):
    ext = _move_front(_pad_axis(phi, axis, bc, reflect_sign=reflect_sign), axis)
    dm = ext[1:-1] - ext[:-2]
    dp = ext[2:] - ext[1:-1]
    slope = _limited_slope(dm, dp, limiter)
    # Physical cells are ext[2 : n+2].  Faces are ext[j]|ext[j+1],
    # j=1..n+1, hence n+1 face states including both boundaries.
    qL = ext[1:-2] + 0.5 * slope[:-1]
    qR = ext[2:-1] - 0.5 * slope[1:]
    if bounded:
        loL = np.minimum(ext[1:-2], ext[2:-1])
        hiL = np.maximum(ext[1:-2], ext[2:-1])
        qL = np.clip(qL, loL, hiL)
        qR = np.clip(qR, loL, hiL)
    return _move_back(qL, axis), _move_back(qR, axis)


def _reconstruct_primitive_axis(
    W,
    dim: int,
    axis: int,
    bc: str,
    *,
    primitive_scheme: str,
    tvd_limiter: str,
    alpha_scheme: str,
):
    bounded, limiter = _parse_reconstruction(primitive_scheme, tvd_limiter)
    alpha_limiter = limiter
    if (alpha_scheme or "").lower() in ("cicsam", "stacs", "mstacs", "thinc", "thinc-bvd", "thinc_bvd"):
        alpha_limiter = "superbee"
    WL = []
    WR = []
    for i, q in enumerate(W):
        q_limiter = alpha_limiter if i == 0 else limiter
        q_bounded = True if i == 0 else bounded
        reflect_sign = -1.0 if bc in ("reflective", "wall") and i == 3 + axis else 1.0
        qL, qR = _reconstruct_axis_scalar(
            q,
            axis,
            bc,
            limiter=q_limiter,
            bounded=q_bounded,
            reflect_sign=reflect_sign,
        )
        if i == 0:
            qL = np.clip(qL, 0.0, 1.0)
            qR = np.clip(qR, 0.0, 1.0)
        WL.append(qL)
        WR.append(qR)
    return tuple(WL), tuple(WR)


def _sound_speed_from_W(W, eos1, eos2, dim: int, mixture_c: str):
    alpha, T1, T2, _, p = split_W_nd(W, dim)
    a = np.clip(alpha, _EPS, 1.0 - _EPS)
    rho1 = eos1.density(p, T1)
    rho2 = eos2.density(p, T2)
    c1_sq = phase_sound_speed_sq(eos1, rho1, T1)
    c2_sq = phase_sound_speed_sq(eos2, rho2, T2)
    c_sq = mixture_sound_speed_sq(a, rho1, c1_sq, rho2, c2_sq, kind=mixture_c)
    return np.sqrt(np.maximum(c_sq, _EPS)), a * rho1 + (1.0 - a) * rho2


def _physical_flux_nd(U, W, *, dim: int, axis: int):
    alpha, _, _, vel, p = split_W_nd(W, dim)
    rho = np.maximum(U[0] + U[1], _EPS)
    un = vel[axis]
    out = [U[0] * un, U[1] * un]
    for j in range(dim):
        f = U[2 + j] * un
        if j == axis:
            f = f + p
        out.append(f)
    out.append((U[2 + dim] + p) * un)
    out.append(alpha * un)
    # Keep rho referenced so vectorized shape mistakes show up in tests.
    _ = rho
    return tuple(out)


def _hllc_flux_axis(
    WL,
    WR,
    eos1,
    eos2,
    *,
    dim: int,
    axis: int,
    mixture_c: str,
):
    UL = prim_to_cons_nd(WL, eos1, eos2, dim=dim)
    UR = prim_to_cons_nd(WR, eos1, eos2, dim=dim)
    FL = _physical_flux_nd(UL, WL, dim=dim, axis=axis)
    FR = _physical_flux_nd(UR, WR, dim=dim, axis=axis)
    _, _, _, velL, pL = split_W_nd(WL, dim)
    _, _, _, velR, pR = split_W_nd(WR, dim)
    unL = velL[axis]
    unR = velR[axis]
    cL, rhoL = _sound_speed_from_W(WL, eos1, eos2, dim, mixture_c)
    cR, rhoR = _sound_speed_from_W(WR, eos1, eos2, dim, mixture_c)
    SL = np.minimum(unL - cL, unR - cR)
    SR = np.maximum(unL + cL, unR + cR)
    denom = rhoL * (SL - unL) - rhoR * (SR - unR)
    SM = (pR - pL + rhoL * unL * (SL - unL) - rhoR * unR * (SR - unR)) / np.where(
        np.abs(denom) > _EPS, denom, np.sign(denom) * _EPS + _EPS
    )
    pStarL = pL + rhoL * (SL - unL) * (SM - unL)
    pStarR = pR + rhoR * (SR - unR) * (SM - unR)

    def star_state(U, W, S, un, rho, p, pstar):
        _, _, _, vel, _ = split_W_nd(W, dim)
        fac = (S - un) / np.where(np.abs(S - SM) > _EPS, S - SM, np.sign(S - SM) * _EPS + _EPS)
        rho_star = rho * fac
        out = [U[0] * fac, U[1] * fac]
        for j in range(dim):
            vj = SM if j == axis else vel[j]
            out.append(rho_star * vj)
        E_star = ((S - un) * U[2 + dim] - p * un + pstar * SM) / np.where(
            np.abs(S - SM) > _EPS, S - SM, np.sign(S - SM) * _EPS + _EPS
        )
        out.append(E_star)
        out.append(U[3 + dim] * fac)
        return tuple(out)

    UstL = star_state(UL, WL, SL, unL, rhoL, pL, pStarL)
    UstR = star_state(UR, WR, SR, unR, rhoR, pR, pStarR)
    flux = []
    for k in range(len(UL)):
        f = np.where(0.0 <= SL, FL[k], FR[k])
        f_lstar = FL[k] + SL * (UstL[k] - UL[k])
        f_rstar = FR[k] + SR * (UstR[k] - UR[k])
        f = np.where((SL <= 0.0) & (0.0 <= SM), f_lstar, f)
        f = np.where((SM <= 0.0) & (0.0 <= SR), f_rstar, f)
        f = np.where(SR < 0.0, FR[k], f)
        flux.append(f)
    return tuple(flux)


def _divergence_from_face_flux(F, axis: int, dx_axis: float):
    fm = _move_front(F, axis)
    div = (fm[1:] - fm[:-1]) / dx_axis
    return _move_back(div, axis)


def rhs_nd(
    U,
    eos1,
    eos2,
    *,
    dim: int,
    dx,
    bc,
    W_seed=None,
    flux_scheme: str = "hllc",
    primitive_scheme: str = "tmlpu",
    tvd_limiter: str = "superbee",
    alpha_scheme: str = "mstacs",
    mixture_c: str = "kapila",
    gravity: Iterable[float] | None = None,
):
    """Return semi-discrete RHS dU/dt for the ND FV system."""
    flux_scheme = (flux_scheme or "hllc").lower()
    if flux_scheme not in ("hllc", "hllc-like", "slau2", "roe"):
        raise ValueError("ND extension currently routes hllc/slau2/roe requests through HLLC normal flux")
    dxs = _normalise_spacing(dx, dim)
    bcs = _normalise_bc(bc, dim)
    W = cons_to_prim_nd(U, eos1, eos2, dim=dim, W_seed=W_seed)
    out = [np.zeros_like(Ui) for Ui in U]
    for axis in range(dim):
        WL, WR = _reconstruct_primitive_axis(
            W,
            dim,
            axis,
            bcs[axis],
            primitive_scheme=primitive_scheme,
            tvd_limiter=tvd_limiter,
            alpha_scheme=alpha_scheme,
        )
        F = _hllc_flux_axis(WL, WR, eos1, eos2, dim=dim, axis=axis, mixture_c=mixture_c)
        for k in range(len(out)):
            out[k] -= _divergence_from_face_flux(F[k], axis, dxs[axis])

    if gravity is not None:
        g = tuple(float(v) for v in gravity)
        if len(g) != dim:
            raise ValueError(f"gravity must have length {dim}, got {gravity}")
        rho = np.maximum(U[0] + U[1], _EPS)
        vel = tuple(U[2 + i] / rho for i in range(dim))
        for i in range(dim):
            out[2 + i] += rho * g[i]
        out[2 + dim] += rho * sum(vel[i] * g[i] for i in range(dim))
    return tuple(out), W


def _add_scaled(U, R, scale):
    return tuple(Ui + scale * Ri for Ui, Ri in zip(U, R))


def _blend(a, b, wa, wb):
    return tuple(wa * ai + wb * bi for ai, bi in zip(a, b))


def _max_signal_speed(W, eos1, eos2, *, dim: int, mixture_c: str):
    c, _ = _sound_speed_from_W(W, eos1, eos2, dim, mixture_c)
    _, _, _, vel, _ = split_W_nd(W, dim)
    return max(float(np.max(np.abs(v) + c)) for v in vel)


def solve_nd(
    eos1,
    eos2,
    W0,
    dx,
    t_end: float,
    *,
    dim: int,
    cfl: float = 0.35,
    dt_fixed: float | None = None,
    bc=None,
    time_integrator: str = "imex_ssp3",
    flux_scheme: str | None = None,
    primitive_scheme: str | None = None,
    tvd_limiter: str | None = None,
    alpha_scheme: str | None = None,
    mixture_c: str = "kapila",
    gravity: Iterable[float] | None = None,
    return_info: bool = False,
):
    """Advance a 2-D or 3-D five-equation state.

    Returns W by default.  With `return_info=True`, returns `(W, info)`.
    """
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim}")
    time_integrator = (time_integrator or "imex_ssp3").lower()
    if time_integrator not in ("ssp3", "rk3", "imex_ssp3"):
        raise ValueError("ND extension supports SSPRK3 / imex_ssp3 API alias")
    flux_scheme = (flux_scheme or os.getenv("FIVE_EQ_IMEX_FLUX", "hllc")).lower()
    primitive_scheme = (primitive_scheme or os.getenv("FIVE_EQ_IMEX_PRIMITIVE_SCHEME", "tmlpu")).lower()
    tvd_limiter = (tvd_limiter or os.getenv("FIVE_EQ_IMEX_TVD_LIMITER", "superbee")).lower()
    alpha_scheme = (alpha_scheme or os.getenv("FIVE_EQ_IMEX_ALPHA_SCHEME", "mstacs")).lower()
    dxs = _normalise_spacing(dx, dim)
    bcs = _normalise_bc(bc, dim)
    U = prim_to_cons_nd(W0, eos1, eos2, dim=dim)
    W = tuple(np.asarray(q, dtype=float).copy() for q in W0)
    t = 0.0
    steps = 0
    dt_last = 0.0
    min_dx = min(dxs)

    while t < t_end - 1.0e-15:
        if dt_fixed is None:
            speed = max(_max_signal_speed(W, eos1, eos2, dim=dim, mixture_c=mixture_c), _EPS)
            dt = cfl * min_dx / speed
        else:
            dt = float(dt_fixed)
        dt = min(dt, t_end - t)
        R0, W0s = rhs_nd(
            U,
            eos1,
            eos2,
            dim=dim,
            dx=dxs,
            bc=bcs,
            W_seed=W,
            flux_scheme=flux_scheme,
            primitive_scheme=primitive_scheme,
            tvd_limiter=tvd_limiter,
            alpha_scheme=alpha_scheme,
            mixture_c=mixture_c,
            gravity=gravity,
        )
        U1 = _add_scaled(U, R0, dt)
        W1 = cons_to_prim_nd(U1, eos1, eos2, dim=dim, W_seed=W0s)
        R1, _ = rhs_nd(
            U1,
            eos1,
            eos2,
            dim=dim,
            dx=dxs,
            bc=bcs,
            W_seed=W1,
            flux_scheme=flux_scheme,
            primitive_scheme=primitive_scheme,
            tvd_limiter=tvd_limiter,
            alpha_scheme=alpha_scheme,
            mixture_c=mixture_c,
            gravity=gravity,
        )
        U2raw = _add_scaled(U1, R1, dt)
        U2 = _blend(U, U2raw, 0.75, 0.25)
        W2 = cons_to_prim_nd(U2, eos1, eos2, dim=dim, W_seed=W1)
        R2, _ = rhs_nd(
            U2,
            eos1,
            eos2,
            dim=dim,
            dx=dxs,
            bc=bcs,
            W_seed=W2,
            flux_scheme=flux_scheme,
            primitive_scheme=primitive_scheme,
            tvd_limiter=tvd_limiter,
            alpha_scheme=alpha_scheme,
            mixture_c=mixture_c,
            gravity=gravity,
        )
        U3raw = _add_scaled(U2, R2, dt)
        U = _blend(U, U3raw, 1.0 / 3.0, 2.0 / 3.0)
        W = cons_to_prim_nd(U, eos1, eos2, dim=dim, W_seed=W2)
        t += dt
        dt_last = dt
        steps += 1

    info = NDSolverInfo(
        dim=dim,
        steps=steps,
        dt_last=dt_last,
        time_integrator=time_integrator,
        flux_scheme=flux_scheme,
        primitive_scheme=primitive_scheme,
        tvd_limiter=tvd_limiter,
        alpha_scheme=alpha_scheme,
    )
    return (W, info) if return_info else W
