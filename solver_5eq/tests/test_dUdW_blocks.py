"""Sanity checks for jacobian.dUdW_blocks helper."""
from __future__ import annotations
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solver.five_eq_IMEX.eos_facade import make_eos
from solver.five_eq_IMEX.primitive import dUdW_analytic
from solver.five_eq_IMEX.jacobian import dUdW_blocks


def _build_state(N=7):
    x = (np.arange(N) + 0.5) / N
    a1 = 0.1 + 0.8 * (0.5 + 0.5 * np.sin(2.0 * np.pi * x))
    T1 = 280.0 + 40.0 * x
    T2 = 300.0 - 20.0 * x
    u = -2.0 + 4.0 * x
    p = 8.0e4 + 2.0e4 * np.cos(2.0 * np.pi * x)
    return (a1, T1, T2, u, p)


def _assert_close(name, a, b, rtol=1e-13, atol=1e-13):
    diff = float(np.max(np.abs(a - b)))
    ref = max(float(np.max(np.abs(b))), 1.0)
    rel = diff / ref
    if not (diff <= atol or rel <= rtol):
        raise AssertionError(f"{name}: max_abs={diff:.3e}, max_rel={rel:.3e}")


def main():
    eos1 = make_eos('ideal', gamma=1.4, kv=717.5)
    eos2 = make_eos('sg', gamma=4.1, pinf=4.4e8, kv=474.2)
    W = _build_state()
    J = dUdW_analytic(W, eos1, eos2)
    B = dUdW_blocks(W, eos1, eos2)

    required = (
        'A_pp', 'A_up', 'A_uu', 'A_ua', 'A_pa', 'A_pT1', 'A_pT2',
        'M_aa', 'M_au', 'M_ap', 'M_ua', 'M_pa', 'M_aa_inv',
        'Mtilde_uu', 'Mtilde_up', 'Mtilde_pu', 'Mtilde_pp', 'Sigma_pp',
    )
    for k in required:
        if k not in B:
            raise AssertionError(f"missing key: {k}")
        arr = np.asarray(B[k])
        if not np.all(np.isfinite(arr)):
            raise AssertionError(f"non-finite values in {k}")

    if B['M_aa'].shape != (3, 3, W[0].size):
        raise AssertionError(f"shape mismatch M_aa: {B['M_aa'].shape}")
    if B['M_aa_inv'].shape != (3, 3, W[0].size):
        raise AssertionError(f"shape mismatch M_aa_inv: {B['M_aa_inv'].shape}")
    for key in ('M_au', 'M_ap', 'M_ua', 'M_pa'):
        if B[key].shape != (3, W[0].size):
            raise AssertionError(f"shape mismatch {key}: {B[key].shape}")
    for key in ('A_pp', 'A_up', 'A_uu', 'A_ua', 'A_pa', 'A_pT1', 'A_pT2',
                'Mtilde_uu', 'Mtilde_up', 'Mtilde_pu', 'Mtilde_pp', 'Sigma_pp'):
        if B[key].shape != W[0].shape:
            raise AssertionError(f"shape mismatch {key}: {B[key].shape}")

    _assert_close('A_pp', B['A_pp'], J[3, 4])
    _assert_close('A_up', B['A_up'], J[2, 4])
    _assert_close('A_uu', B['A_uu'], J[2, 3])
    _assert_close('A_ua', B['A_ua'], J[2, 0])
    _assert_close('A_pa', B['A_pa'], J[3, 0])
    _assert_close('A_pT1', B['A_pT1'], J[3, 1])
    _assert_close('A_pT2', B['A_pT2'], J[3, 2])

    print("dUdW_blocks helper checks")
    print("  [OK] keys/shape/finite")
    print("  [OK] exact slice consistency vs dUdW_analytic")
    print("All tests passed.")


if __name__ == '__main__':
    main()
