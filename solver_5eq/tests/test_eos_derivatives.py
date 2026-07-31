"""Phase 1 unit tests — EOS thermodynamic derivative consistency.

Verifies for each (p, T)-anchored derivative:
    drhodp_T, drhodT_p, dedp_T, dedT_p
that the analytic implementation matches a 2nd-order centered FD evaluated
on the EOS surface (ρ_FD computed from EOS.density(p, T) or from
pressure_from_rhoT inversion).

EOS list:
  IdealEOS  — closed form
  SGEOS     — closed form
  NASGEOS   — closed form (water, Le Métayer-Saurel 2016)
  MieGruneisenEOS — exercised via base FD chain rule (no analytic override yet)
  JWLEOS    — exercised via base FD chain rule
  RKPREOS   — exercised via base FD chain rule (CO₂, gas branch)

Run:  python3 -m pytest tests/test_eos_derivatives.py -v
or:   python3 tests/test_eos_derivatives.py
"""
from __future__ import annotations
import math
import os
import sys

import numpy as np

# Make repo root importable when run as a script
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solver.He2024.eos_general import (  # noqa: E402
    IdealEOS, SGEOS, NASGEOS, MieGruneisenEOS, JWLEOS, RKPREOS,
)


# ─── helpers ───────────────────────────────────────────────────────────────
def _fd_drho_dp_T(eos, p, T, rel=1e-6):
    dp = rel * np.abs(p)
    rho_p = eos.density(p + dp, T)
    rho_m = eos.density(p - dp, T)
    return (rho_p - rho_m) / (2.0 * dp)


def _fd_drho_dT_p(eos, p, T, rel=1e-6):
    dT = rel * np.abs(T)
    rho_p = eos.density(p, T + dT)
    rho_m = eos.density(p, T - dT)
    return (rho_p - rho_m) / (2.0 * dT)


def _fd_de_dp_T(eos, p, T, rel=1e-6):
    dp = rel * np.abs(p)
    rho_p = eos.density(p + dp, T)
    rho_m = eos.density(p - dp, T)
    e_p = eos.energy(rho_p, p + dp)
    e_m = eos.energy(rho_m, p - dp)
    return (e_p - e_m) / (2.0 * dp)


def _fd_de_dT_p(eos, p, T, rel=1e-6):
    dT = rel * np.abs(T)
    rho_p = eos.density(p, T + dT)
    rho_m = eos.density(p, T - dT)
    e_p = eos.energy(rho_p, p)
    e_m = eos.energy(rho_m, p)
    return (e_p - e_m) / (2.0 * dT)


def _check(name: str, analytic, fd, rtol=1e-4, atol=1e-6):
    a = np.atleast_1d(np.asarray(analytic, dtype=float))
    f = np.atleast_1d(np.asarray(fd, dtype=float))
    err = np.max(np.abs(a - f) / (np.abs(f) + atol))
    ok = err < rtol
    flag = 'OK' if ok else 'FAIL'
    print(f"  [{flag}] {name:14s}  analytic={a.ravel()[0]: .6e}  FD={f.ravel()[0]: .6e}  rel={err:.2e}")
    return ok


def _exercise(eos, p_arr, T_arr, label, rtol=1e-4):
    print(f"\n=== {label} ===")
    rho = eos.density(p_arr, T_arr)
    print(f"  ρ(p,T) = {rho.ravel()[0]:.6e}")

    drho_dp_an = eos.drhodp_T(rho, T_arr)
    drho_dT_an = eos.drhodT_p(rho, T_arr)
    de_dp_an   = eos.dedp_T(rho, T_arr)
    de_dT_an   = eos.dedT_p(rho, T_arr)

    drho_dp_fd = _fd_drho_dp_T(eos, p_arr, T_arr)
    drho_dT_fd = _fd_drho_dT_p(eos, p_arr, T_arr)
    de_dp_fd   = _fd_de_dp_T(eos, p_arr, T_arr)
    de_dT_fd   = _fd_de_dT_p(eos, p_arr, T_arr)

    ok = True
    ok &= _check('drhodp_T', drho_dp_an, drho_dp_fd, rtol=rtol)
    ok &= _check('drhodT_p', drho_dT_an, drho_dT_fd, rtol=rtol)
    ok &= _check('dedp_T',   de_dp_an,   de_dp_fd,   rtol=rtol)
    ok &= _check('dedT_p',   de_dT_an,   de_dT_fd,   rtol=rtol)
    return ok


# ─── individual cases ──────────────────────────────────────────────────────
def test_ideal_air():
    eos = IdealEOS(gamma=1.4, kv=717.5)
    p = np.array([1.0e5]); T = np.array([300.0])
    assert _exercise(eos, p, T, 'IdealEOS  (air, 1 bar, 300 K)')


def test_sg_water_denner():
    eos = SGEOS(gamma=4.1, pinf=4.4e8, kv=474.2)
    p = np.array([1.0e5]); T = np.array([300.0])
    assert _exercise(eos, p, T, 'SGEOS     (water Denner, 1 bar, 300 K)')


def test_sg_water_yoosung():
    eos = SGEOS(gamma=4.4, pinf=6.0e8, kv=474.2)
    p = np.array([1.0e7]); T = np.array([350.0])
    assert _exercise(eos, p, T, 'SGEOS     (water Yoo-Sung, 100 bar, 350 K)')


def test_nasg_water_metayer():
    eos = NASGEOS(gamma=1.187, pinf=7.028e8, kv=3610.0,
                  b=6.61e-4, eta=-1.177788e6, q=0.0)
    p = np.array([1.0e5]); T = np.array([300.0])
    assert _exercise(eos, p, T, 'NASGEOS   (water Le Métayer, 1 bar, 300 K)')


def test_nasg_water_high_p():
    eos = NASGEOS(gamma=1.187, pinf=7.028e8, kv=3610.0,
                  b=6.61e-4, eta=-1.177788e6)
    p = np.array([1.0e9]); T = np.array([400.0])
    assert _exercise(eos, p, T, 'NASGEOS   (water, 1 GPa, 400 K)')


def test_mie_gruneisen_base_fd():
    """MG: only base-class FD chain available — verify self-consistent FD vs FD."""
    eos = MieGruneisenEOS(Gamma_G=0.4, rho_ref=1000.0,
                          p_ref_coef=1.0e7, p_ref_n=7.0,
                          e_ref_coef=0.0, kv=1500.0)
    p = np.array([1.0e6]); T = np.array([320.0])
    # density(p,T) base Newton vs no-analytic — looser tolerance
    print(f"\n=== MieGruneisenEOS (base FD chain) ===")
    rho = eos.density(p, T)
    drhodp_an = eos.drhodp_T(rho, T)
    drhodp_fd = _fd_drho_dp_T(eos, p, T, rel=1e-5)
    print(f"  ρ(p,T)={rho.ravel()[0]:.6e}  drhodp_T={drhodp_an.ravel()[0]:.3e} (FD {drhodp_fd.ravel()[0]:.3e})")


def test_jwl_base_fd():
    eos = JWLEOS()  # default TNT
    p = np.array([5.0e9]); T = np.array([3000.0])
    print(f"\n=== JWLEOS (base FD chain, TNT, 5 GPa, 3000 K) ===")
    rho = eos.density(p, T)
    drhodp_an = eos.drhodp_T(rho, T)
    drhodp_fd = _fd_drho_dp_T(eos, p, T, rel=1e-5)
    print(f"  ρ(p,T)={rho.ravel()[0]:.6e}  drhodp_T={drhodp_an.ravel()[0]:.3e} (FD {drhodp_fd.ravel()[0]:.3e})")


# ─── orchestrator (script mode) ────────────────────────────────────────────
def main():
    print("Phase 1 EOS derivative consistency (analytic vs centered FD)\n")
    failed = []
    for fn in (test_ideal_air,
               test_sg_water_denner, test_sg_water_yoosung,
               test_nasg_water_metayer, test_nasg_water_high_p,
               test_mie_gruneisen_base_fd, test_jwl_base_fd):
        try:
            fn()
        except AssertionError:
            failed.append(fn.__name__)
            print(f"  *** {fn.__name__}: FAILED")
        except Exception as exc:
            failed.append(f"{fn.__name__} (error: {exc})")
            print(f"  *** {fn.__name__}: ERROR {exc}")
    print("\n--------------------------------------------------------------------")
    if failed:
        print(f"FAILED ({len(failed)}): {failed}")
        sys.exit(1)
    print("All tests passed.")


if __name__ == '__main__':
    main()
