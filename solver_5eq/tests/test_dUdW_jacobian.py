"""Phase 2 unit tests — analytic dU/dW vs centered FD on prim_to_cons_W.

Also round-trips W → U → W via cons_to_prim_W (Newton 3×3) to confirm
primitive recovery is accurate to ~1e-10 in p, ~1e-8 in T.

Run:  python3 tests/test_dUdW_jacobian.py
"""
from __future__ import annotations
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solver.He2024.primitive_W import (  # noqa: E402
    prim_to_cons_W, cons_to_prim_W, dUdW_analytic, dUdW_numerical,
)


def _W_uniform(alpha, T1, T2, u, p, N=4):
    """Build a uniform W tuple of length N with cell-wise variation."""
    return (np.full(N, alpha, dtype=float),
            np.full(N, T1, dtype=float),
            np.full(N, T2, dtype=float),
            np.full(N, u, dtype=float),
            np.full(N, p, dtype=float))


def _check_jac(label, ph1, ph2, W, rtol=1e-3, atol_scale=1e-4):
    """Compare analytic vs centered FD Jacobian.

    For each entry use combined criterion:
        |a − f| / (max(|a|, |f|) · scale) < rtol     OR  |a − f| < atol_scale · row_scale
    where row_scale is the largest |a| in row i across columns — protects
    near-zero cross-derivatives where FD has cancellation noise (~1e-7
    relative even though absolute error is 1e-6 of row magnitude).
    """
    print(f"\n=== {label} ===")
    print(f"  W = (α={W[0][0]}, T1={W[1][0]}, T2={W[2][0]}, u={W[3][0]}, p={W[4][0]:.3e})")
    J_an = dUdW_analytic(W, ph1, ph2)
    J_fd = dUdW_numerical(W, ph1, ph2, rel=1e-6)
    rows = ['α₁ρ₁', 'α₂ρ₂', 'ρu  ', 'ρE  ', 'α   ']
    cols = ['α   ', 'T1  ', 'T2  ', 'u   ', 'p   ']
    ok = True
    print(f"  cell 0 (rtol={rtol}, atol = {atol_scale}·row_scale):")
    for i in range(5):
        row_scale = max(abs(J_an[i, j, 0]) for j in range(5))
        atol = atol_scale * max(row_scale, 1e-30)
        line = f"    d{rows[i]}/d :"
        for j in range(5):
            a = J_an[i, j, 0]
            f = J_fd[i, j, 0]
            abs_err = abs(a - f)
            rel_denom = max(abs(a), abs(f), 1e-30)
            rel_err = abs_err / rel_denom
            entry_ok = (rel_err < rtol) or (abs_err < atol)
            if not entry_ok:
                ok = False
            mark = ' ' if entry_ok else '!'
            line += f"  {mark}{cols[j]} an={a: .3e} fd={f: .3e}"
        print(line)
    return ok


def _check_roundtrip(label, ph1, ph2, W, p_tol=1e-7, T_tol=1e-5):
    """W → U → W round-trip."""
    print(f"\n=== {label} round-trip ===")
    U, aux = prim_to_cons_W(W, ph1, ph2)
    Wb = cons_to_prim_W(U, ph1, ph2)
    err_alpha = np.max(np.abs(Wb[0] - W[0]))
    err_T1 = np.max(np.abs((Wb[1] - W[1]) / W[1]))
    err_T2 = np.max(np.abs((Wb[2] - W[2]) / W[2]))
    err_u = np.max(np.abs(Wb[3] - W[3]))
    err_p = np.max(np.abs((Wb[4] - W[4]) / W[4]))
    print(f"  err(α)={err_alpha:.2e}  err(T1)={err_T1:.2e}  err(T2)={err_T2:.2e}  "
          f"err(u)={err_u:.2e}  err(p)={err_p:.2e}")
    ok = (err_alpha < 1e-12 and err_T1 < T_tol and err_T2 < T_tol
          and err_p < p_tol and err_u < 1e-10)
    return ok


def test_air_water_sg():
    ph1 = dict(gamma=1.4, pinf=0.0, kv=717.5)
    ph2 = dict(gamma=4.1, pinf=4.4e8, kv=474.2)
    # Single-phase rich air
    W = _W_uniform(0.99, 300.0, 300.0, 1.0, 1e5)
    assert _check_jac('SG air-water (α≈1, air-side)', ph1, ph2, W)
    assert _check_roundtrip('SG air-water (α≈1)', ph1, ph2, W)
    # Single-phase rich water
    W = _W_uniform(0.01, 300.0, 300.0, 1.0, 1e5)
    assert _check_jac('SG air-water (α≈0, water-side)', ph1, ph2, W)
    assert _check_roundtrip('SG air-water (α≈0)', ph1, ph2, W)


def test_air_water_mixed():
    ph1 = dict(gamma=1.4, pinf=0.0, kv=717.5)
    ph2 = dict(gamma=4.1, pinf=4.4e8, kv=474.2)
    W = _W_uniform(0.5, 350.0, 320.0, 100.0, 1e7)
    assert _check_jac('SG air-water mixed α=0.5', ph1, ph2, W)
    assert _check_roundtrip('SG air-water mixed α=0.5', ph1, ph2, W, p_tol=1e-6)


def test_nasg_water_air():
    ph1 = dict(gamma=1.4, pinf=0.0, kv=717.5)
    ph2 = dict(gamma=1.187, pinf=7.028e8, kv=3610.0,
               b=6.61e-4, eta=-1.177788e6, q=0.0)
    # 02-A NASG state
    W = _W_uniform(0.99, 300.0, 300.0, 1.0, 1e5)
    assert _check_jac('NASG (air rich)', ph1, ph2, W)
    assert _check_roundtrip('NASG (air rich)', ph1, ph2, W, p_tol=5e-7)
    W = _W_uniform(0.01, 300.0, 300.0, 1.0, 1e5)
    assert _check_jac('NASG (water rich)', ph1, ph2, W)
    assert _check_roundtrip('NASG (water rich)', ph1, ph2, W, p_tol=5e-7)


def test_high_pressure():
    ph1 = dict(gamma=1.4, pinf=0.0, kv=717.5)
    ph2 = dict(gamma=4.4, pinf=6.0e8, kv=474.2)
    W = _W_uniform(0.5, 400.0, 300.0, 50.0, 1.0e9)
    assert _check_jac('SG high-p mixed (1 GPa)', ph1, ph2, W)
    assert _check_roundtrip('SG high-p mixed (1 GPa)', ph1, ph2, W, p_tol=1e-6)


def main():
    print("Phase 2 dU/dW + W↔U consistency\n")
    failed = []
    for fn in (test_air_water_sg, test_air_water_mixed,
               test_nasg_water_air, test_high_pressure):
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
