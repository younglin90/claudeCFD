"""PE-projection-restricted spectral radius (ChatGPT v3 §6.1 follow-up).

The full one-step amplification matrix A measures *all* eigenmodes including
ordinary advection (which has |λ|>1 under BE-style Newton when Δt is large
relative to the local material CFL).  Most of those eigenvalues are *not*
PE-violating — they are the natural transport modes.

To isolate the **PE-violating eigenmodes**, project the input perturbation
onto the *PE-tangent subspace* (subspace where p ≡ p₀, u ≡ u₀ are preserved
to first order) and measure how much the output deviates from PE.

Concretely:
  1. Pick base state W* with α-jump, uniform (u, p, T_k).
  2. For each PE-tangent direction δW (= e_α, e_T1, e_T2; *not* e_u, e_p),
     compute Φ(W* + ε δW) − Φ(W*) and read out the (p, u) components of
     the resulting δW^{n+1}.
  3. The amplification of (p, u) perturbations *from* PE-tangent inputs is
     the PE-violating eigenvalue spectrum.

If max |λ_PE-violating| ≈ 1 + O(Δt), the spatial scheme is PE-preserving
and ρ(A) ≈ 3.77 is just standard transport.  If |λ_PE-violating| ≫ 1, the
PE drift is real.

Run:  python3 tests/test_pe_projected_spectral.py
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


def measure_pe_response(step_fn, W0, N, dt, eps_rel=1e-6):
    """For each PE-tangent input direction (α, T1, T2 components only),
    measure the (p, u) response amplitudes after one step.

    Returns
    -------
    (max_dp_norm, max_du_norm) — relative response magnitudes
    """
    p0 = W0[4][0]; u0 = W0[3][0]
    base_out = step_fn(W0)
    p_base = base_out[4]; u_base = base_out[3]

    max_dp = 0.0
    max_du = 0.0

    # Loop over input perturbations on α (component 0), T1 (1), T2 (2) only
    for comp in (0, 1, 2):
        for cell in range(N):
            W_pert = list(np.asarray(c, dtype=float).copy() for c in W0)
            if comp == 0:
                eps = eps_rel
            else:
                eps = max(abs(W_pert[comp][cell]) * eps_rel, eps_rel)
            W_pert[comp][cell] += eps
            try:
                out = step_fn(tuple(W_pert))
            except Exception:
                continue
            dp = (out[4] - p_base) / eps
            du = (out[3] - u_base) / eps
            # Normalise: dp by p₀ (relative pressure perturbation per unit input)
            #            du by max(|u_0|, 1)
            max_dp = max(max_dp, float(np.max(np.abs(dp))) / max(p0, 1.0))
            max_du = max(max_du, float(np.max(np.abs(du))) / max(abs(u0), 1.0))
    return max_dp, max_du


def main():
    eos1 = make_eos('ideal', gamma=1.4, kv=717.5)
    eos2 = make_eos('sg', gamma=4.1, pinf=4.4e8, kv=474.2)
    N = 8
    dx = 1.0 / N
    x = (np.arange(N) + 0.5) * dx
    p0 = 1e5; u0 = 1.0; T0 = 300.0
    a1 = np.where((x >= 0.4) & (x < 0.6), 1e-3, 1.0 - 1e-3)
    W0 = (a1, np.full(N, T0), np.full(N, T0), np.full(N, u0), np.full(N, p0))
    dt = 3.7e-5

    cases = [
        ('ARS222', lambda W: ars222_step(W, dt, eos1, eos2, dx, 'periodic', 'periodic',
                                          newton_kwargs={'max_iter': 6, 'rtol': 1e-9, 'atol': 1e-12},
                                          pe_relax='none')[0]),
        ('be1',    lambda W: be1_step(W, dt, eos1, eos2, dx, 'periodic', 'periodic',
                                       newton_kwargs={'max_iter': 6, 'rtol': 1e-9, 'atol': 1e-12})[0]),
        ('be1+pe_correct', lambda W: be1_step(W, dt, eos1, eos2, dx, 'periodic', 'periodic',
                                              newton_kwargs={'max_iter': 6, 'rtol': 1e-9, 'atol': 1e-12},
                                              pe_correct=True)[0]),
        ('be_full', lambda W: be_full_step(W, dt, eos1, eos2, dx, 'periodic', 'periodic',
                                            newton_kwargs={'max_iter': 8, 'rtol': 1e-9, 'atol': 1e-12})[0]),
    ]
    print("PE-projection-restricted response (input perturbations only on α/T1/T2)")
    print(f"{'-'*78}")
    print(f"  {'integrator':24s} {'max |Δp/p₀|':>14s} {'max |Δu/u₀|':>14s}")
    print(f"  (per unit ε perturbation; values ≪ 1 → PE-preserving)")
    print(f"{'-'*78}")
    for label, step_fn in cases:
        try:
            dp, du = measure_pe_response(step_fn, W0, N, dt)
            print(f"  {label:24s} {dp:14.3e} {du:14.3e}")
        except Exception as e:
            print(f"  {label:24s} ERROR: {e}")


if __name__ == '__main__':
    main()
