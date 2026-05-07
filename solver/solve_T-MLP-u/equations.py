"""Governing equations: linear advection and Euler.

Each equation provides a uniform interface used by the FVM kernel:

    eq.nvar                        — number of conserved variables
    eq.var_names                   — human-readable names (cons. vars)
    eq.prim_names                  — primitive variable names (used by reconstruction)
    eq.prim_to_cons(W)  → U
    eq.cons_to_prim(U)  → W
    eq.physical_flux(U, normal) → F     # F · n at a state, dim-aware
    eq.max_wave_speed(U, normal) → λ_max
    eq.wave_speeds_lr(U_L, U_R, normal) → (S_L, S_R)   # for HLL/HLLC

Free parameters: 0 (γ for Euler is a *physical* constant, not a tunable).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np


_EPS = 1e-30


# ─── Linear advection ──────────────────────────────────────────────────────
@dataclass
class Advection:
    """Scalar advection in 1D or 2D, optionally with a spatially-varying
    velocity field.

        ∂u/∂t + a(x) · ∇u = 0             ∇·a = 0 (assumed)

    `velocity` may be either:
      - an ndarray of shape (dim,) for a constant velocity field, or
      - a callable f(x, y) → (u, v) returning the velocity components at
        Cartesian coordinates x, y (broadcast-compatible).  Use this for
        rotational / shear / vortical flows.

    Conservative form is identical for incompressible velocities:
        ∂u/∂t + ∇·(a u) = 0,
    so U = W = u (single variable).
    """
    velocity: object
    nvar: int = 1
    var_names: Sequence[str] = field(default_factory=lambda: ('u',))
    prim_names: Sequence[str] = field(default_factory=lambda: ('u',))

    def __post_init__(self):
        if callable(self.velocity):
            sample = self.velocity(np.asarray(0.0), np.asarray(0.0))
            if not isinstance(sample, (tuple, list)) or len(sample) not in (1, 2):
                raise ValueError(
                    "callable velocity must return a (u,) or (u, v) tuple")
            self._dim = len(sample)
            self._is_variable = True
        else:
            arr = np.atleast_1d(np.asarray(self.velocity, dtype=float))
            self.velocity = arr
            self._dim = arr.shape[0]
            self._is_variable = False

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def is_variable_velocity(self) -> bool:
        return self._is_variable

    def velocity_at(self, points):
        """Velocity at the given coordinates.  `points` has shape (..., dim)."""
        if self._is_variable:
            x = points[..., 0]
            if self._dim == 1:
                comps = self.velocity(x)
            else:
                y = points[..., 1]
                comps = self.velocity(x, y)
            return np.stack([np.asarray(c, dtype=float) for c in comps], axis=-1)
        v = self.velocity
        return np.broadcast_to(v, points.shape[:-1] + (self._dim,))

    def prim_to_cons(self, W):
        return W

    def cons_to_prim(self, U):
        return U

    def physical_flux(self, U, normal, points=None):
        """F·n = (a·n) u.  Pass `points` so variable velocities can sample
        the local field; ignored for constant-velocity advection."""
        if self._is_variable and points is not None:
            a = self.velocity_at(points)
            a_dot_n = np.einsum('...i,...i->...', a, normal)
        else:
            a_dot_n = np.einsum('i,...i->...', self.velocity, normal)
        return a_dot_n * U

    def max_wave_speed(self, U, normal, points=None):
        if self._is_variable and points is not None:
            a = self.velocity_at(points)
            a_dot_n = np.einsum('...i,...i->...', a, normal)
        else:
            a_dot_n = np.einsum('i,...i->...', self.velocity, normal)
        return np.abs(a_dot_n)

    def wave_speeds_lr(self, U_L, U_R, normal, points=None):
        a = self.max_wave_speed(U_L, normal, points=points)
        return -a, a


# ─── Euler 1D ──────────────────────────────────────────────────────────────
@dataclass
class Euler1D:
    """1D compressible Euler with γ-law gas:

        U = (ρ, ρu, ρE)^T
        W = (ρ, u, p)^T
        e = p / ((γ-1) ρ),   E = e + ½ u²,   c² = γ p / ρ
        F = (ρu, ρu² + p, (ρE + p) u)^T

    γ is a physical constant (1.4 for air, etc.), not a tuning knob.
    """
    gamma: float = 1.4
    nvar: int = 3
    var_names: Sequence[str] = field(
        default_factory=lambda: ('rho', 'rho_u', 'rho_E'))
    prim_names: Sequence[str] = field(
        default_factory=lambda: ('rho', 'u', 'p'))

    @property
    def dim(self) -> int:
        return 1

    def prim_to_cons(self, W):
        rho, u, p = W[0], W[1], W[2]
        rho = np.maximum(rho, _EPS)
        e   = p / ((self.gamma - 1.0) * rho)
        E   = e + 0.5 * u * u
        return np.stack([rho, rho * u, rho * E], axis=0)

    def cons_to_prim(self, U):
        rho = np.maximum(U[0], _EPS)
        u   = U[1] / rho
        E   = U[2] / rho
        p   = (self.gamma - 1.0) * rho * (E - 0.5 * u * u)
        return np.stack([rho, u, p], axis=0)

    def physical_flux(self, U, normal=None):
        """Return F·n at a state U with face normal n (1D: n=±1).

        Accepts U of shape (3,) or (3, N).  Returns same shape as U.
        Each conserved variable's flux is a 1D scalar; F·n is the
        component-wise projection F * n_x.
        """
        rho = np.maximum(U[0], _EPS)
        u = U[1] / rho
        E = U[2] / rho
        p = (self.gamma - 1.0) * rho * (E - 0.5 * u * u)
        if normal is None:
            n_x = 1.0
        else:
            n_x = np.asarray(normal)[..., 0]
        F = np.empty_like(U)
        F[0] = rho * u * n_x
        F[1] = (rho * u * u + p) * n_x
        F[2] = (U[2] + p) * u * n_x
        return F

    def sound_speed(self, U):
        rho = np.maximum(U[0], _EPS)
        u = U[1] / rho
        E = U[2] / rho
        p = (self.gamma - 1.0) * rho * (E - 0.5 * u * u)
        return np.sqrt(np.maximum(self.gamma * p / rho, _EPS))

    def max_wave_speed(self, U, normal=None):
        rho = np.maximum(U[0], _EPS)
        u = U[1] / rho
        c = self.sound_speed(U)
        return np.abs(u) + c

    def wave_speeds_lr(self, U_L, U_R, normal=None):
        """Davis estimate."""
        c_L = self.sound_speed(U_L); c_R = self.sound_speed(U_R)
        u_L = U_L[1] / np.maximum(U_L[0], _EPS)
        u_R = U_R[1] / np.maximum(U_R[0], _EPS)
        S_L = np.minimum(u_L - c_L, u_R - c_R)
        S_R = np.maximum(u_L + c_L, u_R + c_R)
        return S_L, S_R


# ─── Euler 2D — placeholder skeleton ───────────────────────────────────────
@dataclass
class Euler2D:
    """2D Euler — built on top of the same primitive (ρ, u, v, p).
    Skeleton only at this stage; physical_flux / wave_speeds_lr will be
    finalised when 2D test cases come online.
    """
    gamma: float = 1.4
    nvar: int = 4
    var_names: Sequence[str] = field(
        default_factory=lambda: ('rho', 'rho_u', 'rho_v', 'rho_E'))
    prim_names: Sequence[str] = field(
        default_factory=lambda: ('rho', 'u', 'v', 'p'))

    @property
    def dim(self) -> int:
        return 2

    def prim_to_cons(self, W):
        rho, u, v, p = W[0], W[1], W[2], W[3]
        rho = np.maximum(rho, _EPS)
        e   = p / ((self.gamma - 1.0) * rho)
        E   = e + 0.5 * (u * u + v * v)
        return np.stack([rho, rho * u, rho * v, rho * E], axis=0)

    def cons_to_prim(self, U):
        rho = np.maximum(U[0], _EPS)
        u   = U[1] / rho
        v   = U[2] / rho
        E   = U[3] / rho
        p   = (self.gamma - 1.0) * rho * (E - 0.5 * (u * u + v * v))
        return np.stack([rho, u, v, p], axis=0)

    def physical_flux(self, U, normal):
        """Project 2D flux onto an arbitrary face normal n=(nx, ny)."""
        n = np.asarray(normal, dtype=float)
        rho = np.maximum(U[0], _EPS)
        u = U[1] / rho;  v = U[2] / rho
        E = U[3] / rho
        p = (self.gamma - 1.0) * rho * (E - 0.5 * (u * u + v * v))
        un = u * n[..., 0] + v * n[..., 1]
        F = np.empty_like(U)
        F[0] = rho * un
        F[1] = rho * u * un + p * n[..., 0]
        F[2] = rho * v * un + p * n[..., 1]
        F[3] = (U[3] + p) * un
        return F

    def sound_speed(self, U):
        rho = np.maximum(U[0], _EPS)
        u = U[1] / rho;  v = U[2] / rho
        E = U[3] / rho
        p = (self.gamma - 1.0) * rho * (E - 0.5 * (u * u + v * v))
        return np.sqrt(np.maximum(self.gamma * p / rho, _EPS))

    def max_wave_speed(self, U, normal):
        n = np.asarray(normal, dtype=float)
        rho = np.maximum(U[0], _EPS)
        u = U[1] / rho;  v = U[2] / rho
        un = u * n[..., 0] + v * n[..., 1]
        c = self.sound_speed(U)
        return np.abs(un) + c

    def wave_speeds_lr(self, U_L, U_R, normal):
        n = np.asarray(normal, dtype=float)
        c_L = self.sound_speed(U_L); c_R = self.sound_speed(U_R)
        u_Ln = (U_L[1] * n[..., 0] + U_L[2] * n[..., 1]) / np.maximum(U_L[0], _EPS)
        u_Rn = (U_R[1] * n[..., 0] + U_R[2] * n[..., 1]) / np.maximum(U_R[0], _EPS)
        S_L = np.minimum(u_Ln - c_L, u_Rn - c_R)
        S_R = np.maximum(u_Ln + c_L, u_Rn + c_R)
        return S_L, S_R
