"""Transport-mode dominant eigenvector analysis (ChatGPT v3 follow-up).

ρ(A_be1) ≈ 3.77 의 dominant eigenvector 가 어떤 grid mode 인지 확인:
  - nyquist (alternating sign) — odd-even decoupling indicator
  - long wavelength — physical mode (acoustic / advection)
  - interface-localised — α-jump 인접 cell 에 집중

dominant eigenvector 의 spatial pattern + (W component decomposition) 출력.

Run:  python3 tests/test_transport_eigenmode.py
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solver.five_eq_IMEX.eos_facade import make_eos
from solver.five_eq_IMEX.time_integrator import be1_step


def _W_to_vec(W):
    return np.concatenate([np.asarray(c, dtype=float) for c in W])


def _vec_to_W(v, N):
    return tuple(v[i*N:(i+1)*N].copy() for i in range(5))


def amplification_matrix(step_fn, W0, eps_rel=1e-6):
    N = W0[0].shape[0]
    n = 5 * N
    v0 = _W_to_vec(W0)
    out_base = step_fn(W0)
    v_base = _W_to_vec(out_base)
    A = np.empty((n, n), dtype=float)
    for j in range(n):
        comp = j // N
        scale = abs(v0[j])
        eps = eps_rel if comp == 0 else max(scale * eps_rel, eps_rel)
        v_pert = v0.copy(); v_pert[j] += eps
        out = step_fn(_vec_to_W(v_pert, N))
        A[:, j] = (_W_to_vec(out) - v_base) / eps
    return A


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

    def step_be1(W):
        return be1_step(W, dt, eos1, eos2, dx, 'periodic', 'periodic',
                        newton_kwargs={'max_iter': 6, 'rtol': 1e-9, 'atol': 1e-12})[0]

    print(f"Transport eigenmode analysis (be1, N={N}, dt={dt})")
    print(f"α profile: {a1}")
    print(f"{'-'*78}")

    A = amplification_matrix(step_be1, W0)
    eigvals, eigvecs = np.linalg.eig(A)
    mags = np.abs(eigvals)
    order = np.argsort(mags)[::-1]
    eigvals = eigvals[order]; eigvecs = eigvecs[:, order]

    labels = ['α₁  ', 'T₁  ', 'T₂  ', 'u   ', 'p   ']
    for k_mode in range(min(3, len(eigvals))):
        v = np.real(eigvecs[:, k_mode])
        # Normalise by component scale
        v = v / max(np.max(np.abs(v)), 1e-30)
        print(f"\nMode {k_mode}:  λ = {eigvals[k_mode]:.4e}, |λ| = {mags[order[k_mode]]:.4f}")
        for i_comp in range(5):
            comp_vec = v[i_comp*N:(i_comp+1)*N]
            sign_pat = ''.join('+' if x > 0.05 else ('-' if x < -0.05 else '·')
                                for x in comp_vec)
            mag_max = float(np.max(np.abs(comp_vec)))
            print(f"  {labels[i_comp]}  |max|={mag_max:.3f}  pattern: [{sign_pat}]")


if __name__ == '__main__':
    main()
