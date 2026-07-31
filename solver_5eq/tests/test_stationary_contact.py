"""Stationary-contact PE-residual test (ChatGPT v3 §6 우선순위 1).

For an α-jump base state with uniform (u, p, T_k), the explicit residual

    R_U = L_E(W) + L_I(W)        (excluding the time-derivative term)

must satisfy

    G_i = (∂p/∂U)_i · R_{U,i} ≈ 0     to machine precision.

If max|G| ≫ 0, the explicit advection / α-source / face thermodynamic state
or its blending is not PE-preserving — *before* time integration ever amplifies
the seed.  This is the key test ChatGPT v3 §6.2 prescribed: it isolates the
spatial PE-violating mode from the time integrator.

Scope of L_E / L_I evaluated here:
  L_E[k] = ∇·F_E[k] − S_E[k]      (advection + α-source, from explicit_residual)
  L_I[k] = (0, 0, ∂p/∂x, ∂(p·u)/∂x, 0)

In a uniform-(u, p) PE state, L_I ≡ 0 (∇p = 0, ∂(pu)/∂x = u·∇p = 0), so any
nonzero G_i comes entirely from L_E.

Run:  python3 tests/test_stationary_contact.py
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solver.five_eq_IMEX.eos_facade import make_eos
from solver.five_eq_IMEX.residual import explicit_residual, implicit_divergences
from solver.five_eq_IMEX.pe_correction import dpdU, apply_pe_tangent_projection


def _init(N=10, alpha_floor=1e-3):
    eos1 = make_eos('ideal', gamma=1.4, kv=717.5)
    eos2 = make_eos('sg', gamma=4.1, pinf=4.4e8, kv=474.2)
    L = 1.0; dx = L / N
    x = (np.arange(N) + 0.5) * dx
    p0 = 1e5; u0 = 1.0; T0 = 300.0
    a1 = np.where((x >= 0.4) & (x < 0.6), alpha_floor, 1.0 - alpha_floor)
    W = (a1, np.full(N, T0), np.full(N, T0),
         np.full(N, u0), np.full(N, p0))
    return W, eos1, eos2, dx


def _G_and_components(W, eos1, eos2, dx, *, energy_form='differential',
                      face_thermo='acid', positivity=False, dt_ref=1e-5,
                      kapila_closure=False, force_lo=False,
                      project_mode=None, label=None):
    """Compute residual L_E + L_I and its PE-projection G."""
    L_E, _ = explicit_residual(W, eos1, eos2, dx, 'periodic', 'periodic',
                                energy_form=energy_form,
                                face_thermo=face_thermo,
                                positivity=positivity, dt=dt_ref,
                                kapila_closure=kapila_closure,
                                force_lo=force_lo)
    impl = implicit_divergences(W, dx, 'periodic', 'periodic',
                                eos1=eos1, eos2=eos2)
    L_I = (np.zeros_like(L_E[0]), np.zeros_like(L_E[0]),
           impl['grad_p'], impl['div_pu'], np.zeros_like(L_E[0]))
    R_U = tuple(L_E[k] + L_I[k] for k in range(5))
    if project_mode is not None:
        R_U, _ = apply_pe_tangent_projection(
            R_U, W, eos1, eos2, mode=project_mode)
    dpdU_rows = dpdU(W, eos1, eos2)             # (5, N)
    G = np.zeros_like(R_U[0])
    for k in range(5):
        G = G + dpdU_rows[k] * R_U[k]
    return R_U, L_E, L_I, dpdU_rows, G


def main():
    print("Stationary-contact PE-residual test (ChatGPT v3 §6.2)")
    print(f"{'-'*78}")
    print(f"  base state: (u, p, T_k) uniform, α-jump 1e-3 ↔ 0.999")
    print(f"  uniform-(u,p) ⇒ L_I = 0 → any nonzero G comes from L_E (advection + α-source)")
    print(f"{'-'*78}")

    W, eos1, eos2, dx = _init()

    cases = [
        dict(label='APEC=differential, ACID, no positivity blending',
             energy_form='differential', face_thermo='acid', positivity=False),
        dict(label='APEC=secant, ACID, no positivity blending',
             energy_form='secant', face_thermo='acid', positivity=False),
        dict(label='APEC=differential, face_thermo=upwind',
             energy_form='differential', face_thermo='upwind', positivity=False),
        dict(label='APEC=differential, ACID, positivity blending ON',
             energy_form='differential', face_thermo='acid', positivity=True),
    ]
    print(f"  {'case':54s} {'max|G|':>10s} {'max|L_E[ρE]|':>14s}")
    print(f"{'-'*78}")
    for case in cases:
        try:
            R_U, L_E, L_I, dpdU_rows, G = _G_and_components(
                W, eos1, eos2, dx, **case)
            print(f"  {case['label']:54s} {float(np.max(np.abs(G))):10.2e} "
                  f"{float(np.max(np.abs(L_E[3]))):14.2e}")
        except Exception as e:
            print(f"  {case['label']:54s} ERROR: {e}")

    print(f"\n{'-'*78}")
    print("07-like active path diagnostic — Kapila + positivity LO + ACID:")
    R_raw, L_E_raw, _L_I_raw, _dpdU_raw, G_raw = _G_and_components(
        W, eos1, eos2, dx,
        energy_form='apec',
        face_thermo='acid',
        positivity=True,
        kapila_closure=True,
        force_lo=True,
        project_mode=None,
    )
    R_prj, _L_E_prj, _L_I_prj, _dpdU_prj, G_prj = _G_and_components(
        W, eos1, eos2, dx,
        energy_form='apec',
        face_thermo='acid',
        positivity=True,
        kapila_closure=True,
        force_lo=True,
        project_mode='interface',
    )
    max_raw = float(np.max(np.abs(G_raw)))
    max_prj = float(np.max(np.abs(G_prj)))
    print(f"  raw max|p_U·L_E|       = {max_raw:.3e}")
    print(f"  projected max|p_U·L_E| = {max_prj:.3e}")
    if not (np.isfinite(max_raw) and np.isfinite(max_prj)):
        raise AssertionError(
            "Non-finite 07-like explicit PE-normal residual: "
            f"raw={max_raw:.3e}, projected={max_prj:.3e}")
    if max_raw <= 1.0e-14:
        if max_prj > 1.0e-14:
            raise AssertionError(
                "PE tangent projection introduced a PE-normal component "
                f"on an already PE-preserving 07-like path: {max_prj:.3e}")
    elif max_prj >= max_raw:
        raise AssertionError(
            "PE tangent projection did not reduce the 07-like explicit "
            f"PE-normal residual: raw={max_raw:.3e}, projected={max_prj:.3e}")

    # Detailed component breakdown for the first case (ACID + differential)
    print(f"\n{'-'*78}")
    print("Component breakdown — APEC=differential, ACID, no positivity:")
    R_U, L_E, L_I, dpdU_rows, G = _G_and_components(
        W, eos1, eos2, dx,
        energy_form='differential', face_thermo='acid', positivity=False)
    labels = ['α₁ρ₁', 'α₂ρ₂', 'ρu  ', 'ρE  ', 'α   ']
    print(f"  i  {'cell':>6s}  " + "  ".join(f"{lab:>12s}" for lab in labels)
          + f"  {'G':>12s}")
    for i in range(W[0].shape[0]):
        row = [f"R_U[{labels[k]}][{i}]={R_U[k][i]:+10.3e}" for k in range(5)]
        # Concise: just R_U values + G
        rvals = [f"{R_U[k][i]:12.3e}" for k in range(5)]
        print(f"  {i:2d}        " + "  ".join(rvals) + f"  {G[i]:12.3e}")


if __name__ == '__main__':
    main()
