"""Prescribed-velocity 2-D alpha transport for interface validation cases."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nd_solver import _divergence_from_face_flux, _normalise_spacing, _reconstruct_axis_scalar


@dataclass(frozen=True)
class AlphaTransportInfo:
    steps: int
    dt_last: float
    limiter: str
    bounded: bool


def _rhs_alpha(alpha, ux_face, uy_face, dx, dy, limiter, bounded):
    aLx, aRx = _reconstruct_axis_scalar(
        alpha, 0, "periodic", limiter=limiter, bounded=bounded, reflect_sign=1.0
    )
    aLy, aRy = _reconstruct_axis_scalar(
        alpha, 1, "periodic", limiter=limiter, bounded=bounded, reflect_sign=1.0
    )
    fx = ux_face * np.where(ux_face >= 0.0, aLx, aRx)
    fy = uy_face * np.where(uy_face >= 0.0, aLy, aRy)
    return -_divergence_from_face_flux(fx, 0, dx) - _divergence_from_face_flux(fy, 1, dy)


def _clip_preserve_sum(alpha, lo, hi, target_sum):
    """Bound alpha while preserving the global sum for closed periodic tests."""
    a = np.clip(alpha, lo, hi)
    for _ in range(8):
        diff = float(target_sum - np.sum(a))
        if abs(diff) <= 1.0e-13 * max(1.0, abs(float(target_sum))):
            break
        if diff > 0.0:
            capacity = hi - a
            mask = capacity > 1.0e-14
            total = float(np.sum(capacity[mask]))
            if total <= 0.0:
                break
            a[mask] += diff * capacity[mask] / total
        else:
            capacity = a - lo
            mask = capacity > 1.0e-14
            total = float(np.sum(capacity[mask]))
            if total <= 0.0:
                break
            a[mask] += diff * capacity[mask] / total
        a = np.clip(a, lo, hi)
    return a


def solve_alpha_transport_2d(
    alpha0,
    dx,
    t_end,
    velocity_faces,
    *,
    cfl: float = 0.45,
    dt_fixed: float | None = None,
    limiter: str = "superbee",
    bounded: bool = True,
    alpha_floor: float = 0.0,
    alpha_ceil: float = 1.0,
    return_info: bool = False,
):
    """Solve alpha_t + div(u alpha)=0 for prescribed face velocities.

    `velocity_faces(t)` must return `(ux_face, uy_face)` with shapes
    `(nx+1, ny)` and `(nx, ny+1)`.  Periodic boundaries are assumed.  The
    time integrator is SSPRK(3,3), matching the multidimensional FV extension.
    """
    dx, dy = _normalise_spacing(dx, 2)
    alpha = np.asarray(alpha0, dtype=float).copy()
    t = 0.0
    steps = 0
    dt_last = 0.0
    mass0 = float(np.sum(alpha))

    while t < t_end - 1.0e-15:
        ux0, uy0 = velocity_faces(t)
        vmax = max(float(np.max(np.abs(ux0))), float(np.max(np.abs(uy0))), 1.0e-14)
        dt = dt_fixed if dt_fixed is not None else cfl * min(dx, dy) / vmax
        dt = min(float(dt), t_end - t)

        k0 = _rhs_alpha(alpha, ux0, uy0, dx, dy, limiter, bounded)
        a1 = _clip_preserve_sum(alpha + dt * k0, alpha_floor, alpha_ceil, mass0)

        ux1, uy1 = velocity_faces(t + dt)
        k1 = _rhs_alpha(a1, ux1, uy1, dx, dy, limiter, bounded)
        a2raw = _clip_preserve_sum(a1 + dt * k1, alpha_floor, alpha_ceil, mass0)
        a2 = 0.75 * alpha + 0.25 * a2raw
        a2 = _clip_preserve_sum(a2, alpha_floor, alpha_ceil, mass0)

        ux2, uy2 = velocity_faces(t + 0.5 * dt)
        k2 = _rhs_alpha(a2, ux2, uy2, dx, dy, limiter, bounded)
        a3raw = _clip_preserve_sum(a2 + dt * k2, alpha_floor, alpha_ceil, mass0)
        alpha = _clip_preserve_sum(
            (1.0 / 3.0) * alpha + (2.0 / 3.0) * a3raw,
            alpha_floor,
            alpha_ceil,
            mass0,
        )

        t += dt
        dt_last = dt
        steps += 1

    info = AlphaTransportInfo(steps=steps, dt_last=dt_last, limiter=limiter, bounded=bounded)
    return (alpha, info) if return_info else alpha
