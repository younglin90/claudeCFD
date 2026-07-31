"""PE-invariant one-step diagnostic (ChatGPT v2 §6 우선순위 1).

For a uniform-(p, u, T_k) state with α-jump, a single time step Φ(W^n) → W^{n+1}
should preserve

    p^{n+1} = p^n           (machine ε)
    u^{n+1} = u^n           (machine ε)
    g(q1, q2, α; p^n) ≡ ρe  (cell-update PE residual byte-zero)

This script measures the four quantities (ep, eu, ρe-residual, R_E) for a
matrix of toggles so the spectral PE-violating mode can be isolated to a
single sub-component.

Run:  python3 tests/test_pe_invariance.py
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solver.five_eq_IMEX.eos_facade import make_eos
from solver.five_eq_IMEX.pe_diagnostic import update_residual, face_consistency
from solver.five_eq_IMEX.time_integrator import ars222_step, be1_step, be_full_step
from solver.five_eq_IMEX.face_state import face_state
from solver.five_eq_IMEX.flux import advective_fluxes


def _initial_state(N=10, L=1.0):
    eos1 = make_eos('ideal', gamma=1.4, kv=717.5)
    eos2 = make_eos('sg', gamma=4.1, pinf=4.4e8, kv=474.2)
    dx = L / N
    x = (np.arange(N) + 0.5) * dx
    p0 = 1e5; u0 = 1.0; T0 = 300.0
    a1 = np.where((x >= 0.4) & (x < 0.6), 1e-3, 1.0 - 1e-3)
    W = (a1, np.full(N, T0), np.full(N, T0), np.full(N, u0), np.full(N, p0))
    return W, eos1, eos2, dx, p0, u0, T0


def _measure(W_n, W_new, eos1, eos2, p0, u0):
    ep = float(np.max(np.abs((W_new[4] - p0) / p0)))
    eu = float(np.max(np.abs(W_new[3] - u0)))
    R_E = update_residual(W_n, W_new, eos1, eos2)
    return ep, eu, float(np.max(np.abs(R_E)))


def _face_check(W, eos1, eos2):
    face = face_state(W, eos1, eos2, 'periodic', 'periodic')
    flx = advective_fluxes(face, eos1, eos2, energy_form='apec')
    Rq1, Rq2 = face_consistency(face, flx['F_a1r1'], flx['F_a2r2'], flx['F_alpha'])
    return float(np.max(np.abs(Rq1))), float(np.max(np.abs(Rq2)))


def main():
    W0, eos1, eos2, dx, p0, u0, T0 = _initial_state()
    dt = 3.7e-5
    print(f"PE-invariant one-step diagnostic — α-jump SG air-water, dt={dt}")
    print(f"{'-'*78}")
    Rq1, Rq2 = _face_check(W0, eos1, eos2)
    print(f"  face consistency (init):  R_q1={Rq1:.2e}  R_q2={Rq2:.2e}")
    print(f"{'-'*78}")

    cases = [
        ('ARS222 + acid + apec_diff + relax_none + lo_pe',
         lambda W: ars222_step(W, dt, eos1, eos2, dx, 'periodic', 'periodic',
                                newton_kwargs={'max_iter': 10, 'rtol': 1e-10, 'atol': 1e-13},
                                pe_relax='none')),
        ('ARS222 + relax_pressure',
         lambda W: ars222_step(W, dt, eos1, eos2, dx, 'periodic', 'periodic',
                                newton_kwargs={'max_iter': 10, 'rtol': 1e-10, 'atol': 1e-13},
                                pe_relax='pressure')),
        ('be1 + relax_none',
         lambda W: be1_step(W, dt, eos1, eos2, dx, 'periodic', 'periodic',
                            newton_kwargs={'max_iter': 10, 'rtol': 1e-10, 'atol': 1e-13})),
        ('be_full + relax_none',
         lambda W: be_full_step(W, dt, eos1, eos2, dx, 'periodic', 'periodic',
                                 newton_kwargs={'max_iter': 12, 'rtol': 1e-10, 'atol': 1e-13})),
    ]

    print(f"  {'case':52s} {'ep':>10s} {'eu':>10s} {'|R_E|':>10s}")
    print(f"{'-'*78}")
    for label, step_fn in cases:
        W = tuple(c.copy() for c in W0)
        try:
            W, _info = step_fn(W)
            ep, eu, RE = _measure(W0, W, eos1, eos2, p0, u0)
            print(f"  {label:52s} {ep:10.2e} {eu:10.2e} {RE:10.2e}")
        except Exception as e:
            print(f"  {label:52s} ERROR: {e}")


if __name__ == '__main__':
    main()
