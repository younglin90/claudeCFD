"""Primitive ↔ conservative variable conversion (W ↔ U) for v2.

Re-uses the validated `solver.He2024.primitive_W` module:
    prim_to_cons_W(W, ph1, ph2)  →  (U_tuple, aux)
    cons_to_prim_W(U, ph1, ph2)  →  (W_tuple, aux)

Aliases here keep the v2 import path narrow so that swapping the underlying
implementation later (or providing a different W ordering) does not require
touching every consumer.

Variables:
    W = (α₁, T₁, T₂, u, p)                 — cell-centred, (N,)
    U = (α₁ρ₁, α₂ρ₂, ρu, ρE, α₁)           — cell-centred, (N,)
"""
from __future__ import annotations
import numpy as np

from .he2024_compat import load_primitive_W

_pw = load_primitive_W()
_prim_to_cons_W = _pw.prim_to_cons_W
_cons_to_prim_W = _pw.cons_to_prim_W


__all__ = ['prim_to_cons', 'cons_to_prim', 'W_to_arrays', 'U_to_arrays']


def prim_to_cons(W, eos1, eos2):
    """W → U.  Returns (U_tuple, aux dict with rho1, rho2, e1, e2, rho)."""
    return _prim_to_cons_W(W, eos1, eos2)


def cons_to_prim(U, eos1, eos2, T1_init=None, T2_init=None,
                 tol=1e-9, max_iter=30):
    """U → W via 3×3 Newton (T₁, T₂, p) with α₁ and u recovered first.

    Returns the W 5-tuple.  Falls back to per-phase 1-D Newton in the
    pure-phase limit; raises FloatingPointError if no admissible state is
    found (caller decides whether to revert the time step).

    Note (R4 attempt and rollback):
      Tolerance was tested at 1e-12 with max_iter=60 in R4 to address
      the S2-Case-A long-time NaN at step ~1000 (R3 result).  The result
      was *mixed*: PE-coupling accuracy in S4 improved by 2 decades, but
      stiff Newton (Air-Water Z≈3340) lost long-time stability — 07-1
      regressed from finite at t_end (R3) to NaN @ step 432 (R4).
      Reverted to the R3 defaults; the long-time NaN at PE static is now
      tracked as a known limitation requiring a structural fix
      (face-level PE projection or W-based update — R5+).
    """
    return _cons_to_prim_W(U, eos1, eos2,
                           T1_init=T1_init, T2_init=T2_init,
                           tol=tol, max_iter=max_iter)


def W_to_arrays(W):
    """Tuple-of-arrays accessor.  `W` may be a 5-tuple of (N,) arrays or a
    (5, N) ndarray.  Returns 5 separate (N,) arrays (no copy when possible).
    """
    if isinstance(W, np.ndarray) and W.ndim == 2 and W.shape[0] == 5:
        return W[0], W[1], W[2], W[3], W[4]
    return W[0], W[1], W[2], W[3], W[4]


def U_to_arrays(U):
    """Same accessor for U."""
    if isinstance(U, np.ndarray) and U.ndim == 2 and U.shape[0] == 5:
        return U[0], U[1], U[2], U[3], U[4]
    return U[0], U[1], U[2], U[3], U[4]
