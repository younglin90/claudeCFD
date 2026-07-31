"""W = (α₁, T₁, T₂, u, p) primitive utilities.

Re-exports the Phase 2 helpers from `solver.He2024.primitive_W`:

  - `prim_to_cons_W(W, ph1, ph2)`     W → U with EOS-consistent ρ_k, e_k.
  - `cons_to_prim_W(U, ph1, ph2, ...)` U → W via 3×3 Newton on (p, T1, T2).
  - `dUdW_analytic(W, ph1, ph2)`       closed-form 5×5 dU/dW per cell.
  - `dUdW_numerical(W, ph1, ph2, rel)` centered FD reference (validation only).

Adds W-helpers used by the new IMEX solver but not by the legacy one.
"""
from __future__ import annotations
import numpy as np

from .he2024_compat import load_primitive_W

_primitive_W = load_primitive_W()
prim_to_cons_W = _primitive_W.prim_to_cons_W
_cons_to_prim_W_base = _primitive_W.cons_to_prim_W
dUdW_analytic = _primitive_W.dUdW_analytic
dUdW_numerical = _primitive_W.dUdW_numerical

_EPS = 1e-30


def _scalar_e(eos, rho, p):
    return float(np.asarray(eos.energy(np.array([rho]), np.array([p])))[0])


def _scalar_T(eos, rho, e):
    return float(np.asarray(eos.temperature(np.array([rho]), np.array([e])))[0])


def _scalar_pressure(eos, rho, e):
    return float(np.asarray(eos.pressure(np.array([rho]), np.array([e])))[0])


def _scalar_derivative(value):
    return float(np.asarray(value)[0])


def _de_dp_fixed_rho(eos, rho, p):
    """Return de/dp at fixed rho using the available (p,T) EOS derivatives."""
    e = _scalar_e(eos, rho, p)
    T = max(_scalar_T(eos, rho, e), 1.0e-12)
    rho_a = np.array([rho])
    T_a = np.array([T])
    try:
        rho_p = _scalar_derivative(eos.drhodp_T(rho_a, T_a))
        rho_T = _scalar_derivative(eos.drhodT_p(rho_a, T_a))
        e_p = _scalar_derivative(eos.dedp_T(rho_a, T_a))
        e_T = _scalar_derivative(eos.dedT_p(rho_a, T_a))
        if np.isfinite(rho_T) and abs(rho_T) > 1.0e-30:
            val = e_p - e_T * rho_p / rho_T
            if np.isfinite(val):
                return val
    except Exception:
        pass

    # Last-resort local slope. This path is rarely used but keeps diagnostics
    # finite for EOS corners where a derivative is singular.
    h = max(abs(p) * 1.0e-6, 1.0)
    pp = max(p + h, 1.0)
    pm = max(p - h, 1.0)
    if pp == pm:
        pp = p + h
    return (_scalar_e(eos, rho, pp) - _scalar_e(eos, rho, pm)) / (pp - pm)


def _positive_seed(arr, i, fallback):
    if arr is None:
        return float(fallback)
    val = float(np.asarray(arr)[i])
    return val if np.isfinite(val) and val > 0.0 else float(fallback)


def _recover_near_pure_cell(U, eos1, eos2, i, T1_init, T2_init,
                            W_seed, tol, max_iter):
    U1, U2, U3, U4, U5 = (np.asarray(U[k], dtype=float) for k in range(5))
    alpha = min(max(float(U5[i]), 0.0), 1.0)
    beta = 1.0 - alpha
    q1 = max(float(U1[i]), 0.0)
    q2 = max(float(U2[i]), 0.0)
    rho = max(q1 + q2, _EPS)
    u = float(U3[i]) / rho
    rho_e = float(U4[i]) - 0.5 * float(U3[i]) * u
    mass_floor = 1.0e-14 * max(rho, 1.0)
    use1 = q1 > mass_floor and alpha > 0.0
    use2 = q2 > mass_floor and beta > 0.0

    if not (use1 or use2):
        return None

    rho1 = q1 / max(alpha, _EPS) if use1 else 1.0
    rho2 = q2 / max(beta, _EPS) if use2 else 1.0
    p = float(np.asarray(W_seed[4])[i]) if W_seed is not None else np.nan
    if not (np.isfinite(p) and p > 0.0):
        if use1:
            p = _scalar_pressure(eos1, rho1, rho_e / max(q1, _EPS))
        elif use2:
            p = _scalar_pressure(eos2, rho2, rho_e / max(q2, _EPS))
    p = max(p if np.isfinite(p) else 1.0e5, 1.0)

    def residual_at(p_val):
        total = 0.0
        deriv = 0.0
        if use1:
            total += q1 * _scalar_e(eos1, rho1, p_val)
            deriv += q1 * _de_dp_fixed_rho(eos1, rho1, p_val)
        if use2:
            total += q2 * _scalar_e(eos2, rho2, p_val)
            deriv += q2 * _de_dp_fixed_rho(eos2, rho2, p_val)
        return total - rho_e, deriv

    for _ in range(max_iter):
        F, dF = residual_at(p)
        if not np.isfinite(F):
            return None
        if abs(F) <= tol * max(abs(rho_e), 1.0):
            break
        if not (np.isfinite(dF) and abs(dF) > 1.0e-30):
            h = max(abs(p) * 1.0e-6, 1.0)
            Fp, _ = residual_at(p + h)
            Fm, _ = residual_at(max(p - h, 1.0))
            dF = (Fp - Fm) / (p + h - max(p - h, 1.0))
        if not (np.isfinite(dF) and abs(dF) > 1.0e-30):
            return None

        dp = -F / dF
        damp = 1.0
        p_new = p
        for _line in range(12):
            trial = p + damp * dp
            if np.isfinite(trial) and trial > 1.0:
                p_new = trial
                break
            damp *= 0.5
        if abs(p_new - p) <= 1.0e-13 * max(abs(p), 1.0):
            p = max(p_new, 1.0)
            break
        p = max(p_new, 1.0)

    e1 = _scalar_e(eos1, rho1, p) if use1 else np.nan
    e2 = _scalar_e(eos2, rho2, p) if use2 else np.nan
    T1 = _scalar_T(eos1, rho1, e1) if use1 else np.nan
    T2 = _scalar_T(eos2, rho2, e2) if use2 else np.nan
    if not (np.isfinite(T1) and T1 > 0.0):
        T1 = _positive_seed(T1_init, i, T2 if np.isfinite(T2) else 300.0)
    if not (np.isfinite(T2) and T2 > 0.0):
        T2 = _positive_seed(T2_init, i, T1 if np.isfinite(T1) else 300.0)
    if not (np.isfinite(p) and np.isfinite(T1) and np.isfinite(T2)):
        return None
    return alpha, T1, T2, u, p


def cons_to_prim_W(U, ph1, ph2, T1_init=None, T2_init=None,
                   tol=1e-9, max_iter=30, alpha_pure_tol=0.0):
    """Recover W from U, with an optional fixed-density near-pure fallback.

    The default path is the validated Phase-2 3x3 Newton.  When
    ``alpha_pure_tol`` is positive, cells with alpha close to 0 or 1 are
    recovered by a scalar pressure solve at fixed phase densities.  That avoids
    the singular ghost-phase block without changing mixed-cell behavior.
    """
    W = _cons_to_prim_W_base(U, ph1, ph2, T1_init=T1_init, T2_init=T2_init,
                             tol=tol, max_iter=max_iter)
    if alpha_pure_tol <= 0.0:
        return W

    U5 = np.asarray(U[4], dtype=float)
    alpha = np.clip(U5, 0.0, 1.0)
    near = (alpha <= alpha_pure_tol) | (alpha >= 1.0 - alpha_pure_tol)
    if not np.any(near):
        return W

    out = [np.asarray(c, dtype=float).copy() for c in W]
    bad = np.zeros_like(alpha, dtype=bool)
    for c in out:
        bad |= ~np.isfinite(c)
    mask = near | bad
    for i in np.where(mask)[0]:
        recovered = _recover_near_pure_cell(
            U, ph1, ph2, int(i), T1_init, T2_init, W, tol, max_iter)
        if recovered is None:
            continue
        for k in range(5):
            out[k][i] = recovered[k]
    return tuple(out)

__all__ = [
    'prim_to_cons_W', 'cons_to_prim_W', 'dUdW_analytic', 'dUdW_numerical',
    'pack_W', 'unpack_W', 'uniform_W',
]


def pack_W(W):
    """Stack a 5-tuple of (N,) arrays into one (5N,) flat array (Newton state)."""
    return np.concatenate([np.asarray(c, dtype=float) for c in W])


def unpack_W(W_flat, N):
    """Inverse of `pack_W`: split (5N,) → 5-tuple of (N,) arrays."""
    return tuple(W_flat[i * N:(i + 1) * N] for i in range(5))


def uniform_W(N, alpha, T1, T2, u, p):
    """Build a uniform primitive state of length N."""
    return (np.full(N, float(alpha)),
            np.full(N, float(T1)),
            np.full(N, float(T2)),
            np.full(N, float(u)),
            np.full(N, float(p)))
