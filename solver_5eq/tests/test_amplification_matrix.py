"""One-step amplification matrix spectral analysis (ChatGPT v2 §1).

Constructs A_PE = ∂Φ_Δt / ∂W around a PE base state with α-jump, then
measures ρ(A_PE) = max_i |λ_i|.  The PE-violating eigenmode shows up as a
single (or few) dominant eigenvalues with |λ| > 1.

Toggles:
  - ARS222 / be1 / be_full
  - APEC mode='differential' / 'secant'
  - LO flux 'pe_preserving' / 'rusanov'
  - face_thermo='acid' / 'upwind'
  - pe_relax 'none' / 'pressure'

The first toggle that drops ρ(A) below ~1 + O(Δt) tells us which layer is
responsible for the spectral instability.

Run:  python3 tests/test_amplification_matrix.py
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solver.five_eq_IMEX.eos_facade import make_eos
from solver.five_eq_IMEX.time_integrator import ars222_step, be1_step, be_full_step


def _W_to_vec(W):
    return np.concatenate([np.asarray(c, dtype=float) for c in W])


def _vec_to_W(v, N):
    return tuple(v[i*N:(i+1)*N].copy() for i in range(5))


def amplification_matrix(step_fn, W0, eps_rel=1e-6):
    """Return A = ∂Φ/∂W (5N × 5N) by 1st-order FD."""
    N = W0[0].shape[0]
    n = 5 * N
    v0 = _W_to_vec(W0)
    W_base = _vec_to_W(v0, N)
    out_base = step_fn(W_base)
    v_base = _W_to_vec(out_base)
    A = np.empty((n, n), dtype=float)
    for j in range(n):
        # Component-wise step size
        comp = j // N
        scale = abs(v0[j])
        if comp == 0:                      # alpha
            eps = eps_rel
        else:
            eps = max(scale * eps_rel, eps_rel)
        v_pert = v0.copy()
        v_pert[j] += eps
        W_pert = _vec_to_W(v_pert, N)
        out_pert = step_fn(W_pert)
        v_pert_out = _W_to_vec(out_pert)
        A[:, j] = (v_pert_out - v_base) / eps
    return A


def _initial_state(N=8, alpha_floor=1e-3):
    eos1 = make_eos('ideal', gamma=1.4, kv=717.5)
    eos2 = make_eos('sg', gamma=4.1, pinf=4.4e8, kv=474.2)
    L = 1.0; dx = L / N
    x = (np.arange(N) + 0.5) * dx
    p0 = 1e5; u0 = 1.0; T0 = 300.0
    a1 = np.where((x >= 0.4) & (x < 0.6), alpha_floor, 1.0 - alpha_floor)
    W = (a1, np.full(N, T0), np.full(N, T0), np.full(N, u0), np.full(N, p0))
    return W, eos1, eos2, dx


def main():
    N = 8
    W0, eos1, eos2, dx = _initial_state(N=N)
    dt = 3.7e-5

    cases = [
        ('ARS222 raw',         lambda W: ars222_step(W, dt, eos1, eos2, dx, 'periodic', 'periodic',
                                                      newton_kwargs={'max_iter': 6, 'rtol': 1e-9, 'atol': 1e-12},
                                                      pe_relax='none')[0]),
        ('be1 raw',            lambda W: be1_step(W, dt, eos1, eos2, dx, 'periodic', 'periodic',
                                                   newton_kwargs={'max_iter': 6, 'rtol': 1e-9, 'atol': 1e-12})[0]),
        ('be1 schur=True',     lambda W: be1_step(W, dt, eos1, eos2, dx, 'periodic', 'periodic',
                                                   newton_kwargs={'max_iter': 6, 'rtol': 1e-9, 'atol': 1e-12},
                                                   schur=True)[0]),
        ('be1 pe_correct=True', lambda W: be1_step(W, dt, eos1, eos2, dx, 'periodic', 'periodic',
                                                    newton_kwargs={'max_iter': 6, 'rtol': 1e-9, 'atol': 1e-12},
                                                    pe_correct=True)[0]),
    ]
    print(f"Amplification spectral radius around α-jump PE state, N={N}, dt={dt}")
    print(f"{'-'*74}")
    print(f"  {'integrator':16s} {'ρ(A)':>14s} {'top 3 |λ|':>40s}")
    print(f"{'-'*74}")
    for label, step_fn in cases:
        try:
            A = amplification_matrix(step_fn, W0)
            eigs = np.linalg.eigvals(A)
            mags = np.abs(eigs)
            mags_sorted = np.sort(mags)[::-1]
            top3 = ' '.join(f'{m:.3e}' for m in mags_sorted[:3])
            print(f"  {label:16s} {float(mags_sorted[0]):14.4e} {top3:>40s}")
        except Exception as e:
            print(f"  {label:16s} ERROR: {e}")


if __name__ == '__main__':
    main()
