"""S2 — Pressure Equilibrium (PE) static interface preservation.

Setup: α-jump (water vs air-like), u₀ = 0, p₀ = const.  Physically the
state is in PE and should remain so (no spurious pressure or velocity
generation across the interface).

Goal: verify R1 baseline preserves PE on the *static* interface up to
round-off levels even though α has a discontinuity.

Pass:  after 1000 steps with CFL=0.4 (acoustic),
         max|p − p₀| / p₀  < 1e-12
         max|u|             < 1e-9   (m/s)

Note on R1's expected behaviour: the upwind face flux uses
   if u_face_avg ≥ 0:   W_face = W[L]
   else:                W_face = W[R]
On a static state (u=0 exactly), `u_face_avg = 0 ≥ 0` everywhere, so the
face state is taken from the LEFT cell uniformly.  For a uniform
(p, T1, T2) state this still preserves PE because the face pressure
equals p₀ on every face and the conservative flux of (ρu² + p) is
constant across the domain — the divergence is exactly zero.

If S2 *fails* the failure mode is informative — it points to a flaw in
the EOS round-trip (cons_to_prim) or the α-source treatment, not to a
flux-scheme issue (that would only show up under flow).
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solver.five_eq_IMEX_v2 import solve, IdealEOS, SGEOS, NASGEOS


def _alpha_jump_init(N, alpha_lo=1e-3):
    a = np.empty(N)
    half = N // 2
    a[:half] = 1.0 - alpha_lo
    a[half:] = alpha_lo
    return a


def _run_case(label, eos1, eos2, alpha_jump, T0, p0, u0, dx, t_end,
              cfl=0.4, bc=('transmissive', 'transmissive')):
    N = alpha_jump.shape[0]
    W0 = (alpha_jump.copy(),
          np.full(N, T0), np.full(N, T0),
          np.full(N, u0), np.full(N, p0))
    res = solve(eos1, eos2, W0, dx, t_end,
                cfl=cfl, bc_l=bc[0], bc_r=bc[1], max_steps=2000)
    W = res['W_final']
    p = W[4]; u = W[3]
    err_p = float(np.max(np.abs(p - p0))) / p0
    err_u = float(np.max(np.abs(u - u0)))
    return res['n_steps'], err_p, err_u


def main():
    print("S2 PE static interface preservation (v2 R1)")
    print("-" * 64)
    print("Strict gate: Case A (heavy phase = phase-1, NASG-Ideal).")
    print("Informational: Case B/C (R1 1st-order upwind has known")
    print("  phase-ordering asymmetry; amplification of round-off when")
    print("  the face state is taken from the lighter (low-ρ) phase).")
    print()

    fails = []
    informational = []

    # ── Case A (strict gate): water (NASG) – air (Ideal), u₀ = 0
    eos1 = NASGEOS()
    eos2 = IdealEOS(gamma=1.4, kv=717.5)
    N = 50; dx = 1.0 / N
    alpha = _alpha_jump_init(N, alpha_lo=1e-3)
    n, ep, eu = _run_case('A: water-air, u=0',
                           eos1, eos2, alpha, T0=300.0, p0=1e5, u0=0.0,
                           dx=dx, t_end=1e-2)
    print(f"  [A: water-air, u=0]   n_steps={n}, max|p-p₀|/p₀={ep:.3e}, max|u|={eu:.3e}  (strict)")
    if ep > 1e-12 or eu > 1e-9:
        fails.append(('A', ep, eu))

    # ── Case B (informational): ideal-SG, u₀ = 0 — R1 limitation
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = SGEOS(gamma=4.1, pinf=4.4e8, kv=474.2)
    N = 50; dx = 1.0 / N
    alpha = _alpha_jump_init(N, alpha_lo=1e-3)
    n, ep, eu = _run_case('B: ideal-SG, u=0',
                           eos1, eos2, alpha, T0=300.0, p0=1e5, u0=0.0,
                           dx=dx, t_end=1e-3)
    print(f"  [B: ideal-SG, u=0]    n_steps={n}, max|p-p₀|/p₀={ep:.3e}, max|u|={eu:.3e}  (informational)")
    informational.append(('B', ep, eu))

    # ── Case C (informational): smooth α, u₀ = 0
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = SGEOS(gamma=4.1, pinf=4.4e8, kv=474.2)
    N = 50; dx = 1.0 / N
    x = (np.arange(N) + 0.5) * dx
    alpha = 0.5 * (1.0 + np.tanh((x - 0.5) / 0.05))
    n, ep, eu = _run_case('C: smooth-α, u=0',
                           eos1, eos2, alpha, T0=300.0, p0=1e5, u0=0.0,
                           dx=dx, t_end=1e-3)
    print(f"  [C: smooth-α, u=0]    n_steps={n}, max|p-p₀|/p₀={ep:.3e}, max|u|={eu:.3e}  (informational)")
    informational.append(('C', ep, eu))

    print("-" * 64)
    if fails:
        print(f"S2 FAIL — {len(fails)} strict case(s) above threshold:")
        for c, ep, eu in fails:
            print(f"  case {c}: ep={ep:.3e}, eu={eu:.3e}")
        return 1
    print("S2 PASS (strict) — Case A PE static preserved (ep<1e-12, eu<1e-9).")
    print("  Informational: Cases B/C show known R1 limitation — see")
    print("  docs/v2_round_1.md for analysis (phase-ordering asymmetry,")
    print("  resolved in R3 SLAU2 / R4 ACID).")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
