"""Phase 3 unit test — uniform-flow residual.

For W = const everywhere:
  - explicit residual L_E should be 0 to machine ε.
  - implicit divergences (∇p, ∂(p u)/∂x) should be 0 to machine ε.
  - Newton must converge in 0 iterations (R is already 0).
  - One ARS222 step must leave W unchanged.

Run:  python3 tests/test_uniform_flow.py
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solver.five_eq_IMEX.eos_facade import make_eos
from solver.five_eq_IMEX.primitive import uniform_W
from solver.five_eq_IMEX.residual import explicit_residual, implicit_divergences
from solver.five_eq_IMEX.time_integrator import ars222_step


def _check_zero(label, arr, atol=1e-10):
    m = float(np.max(np.abs(arr)))
    ok = m <= atol
    flag = 'OK' if ok else 'FAIL'
    print(f"  [{flag}] {label:24s}  max|·| = {m:.3e}")
    return ok


def test_uniform_air_water():
    eos1 = make_eos('ideal', gamma=1.4, kv=717.5)
    eos2 = make_eos('sg', gamma=4.1, pinf=4.4e8, kv=474.2)
    N = 40
    W = uniform_W(N, alpha=0.6, T1=300.0, T2=300.0, u=10.0, p=1e5)
    dx = 0.025

    print("\n=== uniform W (Ideal air + SG water) ===")
    L_E, _ = explicit_residual(W, eos1, eos2, dx, 'periodic', 'periodic')
    impl = implicit_divergences(W, dx, 'periodic', 'periodic')
    ok = True
    ok &= _check_zero('L_E[α₁ρ₁]', L_E[0])
    ok &= _check_zero('L_E[α₂ρ₂]', L_E[1])
    ok &= _check_zero('L_E[ρu]',  L_E[2])
    ok &= _check_zero('L_E[ρE]',  L_E[3])
    ok &= _check_zero('L_E[α]',   L_E[4])
    ok &= _check_zero('grad_p',   impl['grad_p'])
    ok &= _check_zero('div_pu',   impl['div_pu'])
    ok &= _check_zero('div_u',    impl['div_u'])

    # One ARS222 step must reproduce W to high precision
    W_new, _ = ars222_step(W, dt=1e-3, eos1=eos1, eos2=eos2,
                           dx=dx, bc_l='periodic', bc_r='periodic')
    err_p  = float(np.max(np.abs((W_new[4] - W[4]) / W[4])))
    err_u  = float(np.max(np.abs(W_new[3] - W[3])))
    err_T1 = float(np.max(np.abs(W_new[1] - W[1])))
    err_T2 = float(np.max(np.abs(W_new[2] - W[2])))
    err_a  = float(np.max(np.abs(W_new[0] - W[0])))
    print(f"  one ARS222 step: err(p)={err_p:.2e}, err(u)={err_u:.2e}, "
          f"err(T1)={err_T1:.2e}, err(T2)={err_T2:.2e}, err(α)={err_a:.2e}")
    ok &= (err_p < 1e-10 and err_u < 1e-9 and err_a < 1e-10
           and err_T1 < 1e-8 and err_T2 < 1e-8)
    assert ok


def test_uniform_nasg_air():
    eos1 = make_eos('ideal', gamma=1.4, kv=717.5)
    eos2 = make_eos('nasg', gamma=1.187, pinf=7.028e8, kv=3610.0,
                    b=6.61e-4, eta=-1.177788e6)
    N = 30
    W = uniform_W(N, alpha=0.5, T1=300.0, T2=300.0, u=1.0, p=1e5)
    dx = 0.05

    print("\n=== uniform W (Ideal air + NASG water, 02-A like) ===")
    L_E, _ = explicit_residual(W, eos1, eos2, dx, 'periodic', 'periodic')
    impl = implicit_divergences(W, dx, 'periodic', 'periodic')
    ok = True
    for k, label in enumerate(['α₁ρ₁', 'α₂ρ₂', 'ρu', 'ρE', 'α']):
        ok &= _check_zero(f'L_E[{label}]', L_E[k])
    ok &= _check_zero('grad_p', impl['grad_p'])
    ok &= _check_zero('div_pu', impl['div_pu'])
    ok &= _check_zero('div_u',  impl['div_u'])
    assert ok


def main():
    print("Phase 3 uniform-flow regression\n")
    failed = []
    for fn in (test_uniform_air_water, test_uniform_nasg_air):
        try:
            fn()
            print(f"   *** {fn.__name__}: PASS")
        except AssertionError:
            failed.append(fn.__name__)
            print(f"   *** {fn.__name__}: FAILED")
        except Exception as exc:
            failed.append(f"{fn.__name__} ({exc})")
            print(f"   *** {fn.__name__}: ERROR {exc}")
    print("\n--------------------------------------------------------------------")
    if failed:
        print(f"FAILED ({len(failed)}): {failed}")
        sys.exit(1)
    print("All tests passed.")


if __name__ == '__main__':
    main()
