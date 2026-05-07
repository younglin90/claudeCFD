"""Ghost-cell boundary conditions for the v2 explicit FVM solver.

Supported BCs (v2 R1 minimal set):
  'periodic'     — wrap-around
  'transmissive' — zero-gradient (Neumann)
  'reflective'   — even reflection for scalars; odd for velocity (u → −u)

extend(arr, bc_l, bc_r, ng, odd=False) returns (N + 2 ng,) padded array.
extend_W(W, bc_l, bc_r, ng) returns each component padded with the right
symmetry (velocity is odd-reflected at reflective walls).

NSCBC / inlet_acoustic / Dirichlet inlet variants are deliberately omitted
in R1 — they introduce free-parameter knobs (impedance Z₀ reference state)
that do not belong in the v2 baseline.  Re-add only when a validation case
(e.g. 04-B sinusoidal forcing) demands it.
"""
from __future__ import annotations
import numpy as np


__all__ = ['extend', 'extend_W', 'extend_U']


def extend(arr, bc_l, bc_r, ng=1, *, odd=False):
    """Pad `arr` with `ng` ghost cells on each side."""
    arr = np.asarray(arr, dtype=float)
    N = arr.shape[0]
    out = np.empty(N + 2 * ng, dtype=float)
    out[ng:ng + N] = arr

    # Left ghosts
    if bc_l == 'periodic':
        out[:ng] = arr[N - ng:N]
    elif bc_l == 'transmissive':
        out[:ng] = arr[0]
    elif bc_l == 'reflective':
        rev = arr[:ng][::-1]
        out[:ng] = -rev if odd else rev
    else:
        raise ValueError(f"Unknown bc_l='{bc_l}'.")

    # Right ghosts
    if bc_r == 'periodic':
        out[N + ng:] = arr[:ng]
    elif bc_r == 'transmissive':
        out[N + ng:] = arr[N - 1]
    elif bc_r == 'reflective':
        rev = arr[N - ng:N][::-1]
        out[N + ng:] = -rev if odd else rev
    else:
        raise ValueError(f"Unknown bc_r='{bc_r}'.")

    return out


def extend_W(W, bc_l, bc_r, ng=1):
    """Pad each component of W = (α₁, T₁, T₂, u, p) with the right symmetry.

    Velocity (index 3) is odd-reflected at reflective walls; all other
    components are even-reflected.
    """
    alpha, T1, T2, u, p = W
    a_ext  = extend(alpha, bc_l, bc_r, ng, odd=False)
    T1_ext = extend(T1,    bc_l, bc_r, ng, odd=False)
    T2_ext = extend(T2,    bc_l, bc_r, ng, odd=False)
    u_ext  = extend(u,     bc_l, bc_r, ng, odd=True)
    p_ext  = extend(p,     bc_l, bc_r, ng, odd=False)
    return a_ext, T1_ext, T2_ext, u_ext, p_ext


def extend_U(U, bc_l, bc_r, ng=1):
    """Pad each component of U = (α₁ρ₁, α₂ρ₂, ρu, ρE, α₁) with the right
    reflection symmetry.

    Momentum (index 2 = ρu) is odd-reflected at reflective walls; all other
    components (mass, energy, volume fraction) are even-reflected.
    """
    out = []
    for k, comp in enumerate(U):
        odd = (k == 2)        # only ρu flips sign at a reflective wall
        out.append(extend(comp, bc_l, bc_r, ng, odd=odd))
    return tuple(out)
