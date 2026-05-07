"""Face state for v2 R2 final — central pressure & velocity, upwind advected scalars.

After investigating two variants in Round 2:
  - R2a  (p_face, u_face both central) — Galilean p/u/T 9-decade improvement,
    07-B Argon-Air finite to t_end, but advection unstable (S5 NaN @ 45).
  - R2.1a (p_face central, u_face upwind) — strictly *worse* than both R1
    and R2a (S2 Case A NaN @ 1342 vs R1 2000 PASS; all 07 sub-cases
    NaN within ≤300 steps).  Root cause: at PE state u_face_avg sits at
    round-off level so the upwind sign for u_face randomizes, injecting
    noise into every interior face.

We retain **R2a** (p_face + u_face central) as v2 R2 final because:
  (i) its PE-coupling improvement (S4 Galilean 9-decade) is real;
  (ii) its advection instability (S5 step 45) is the *known* limitation
      that R3 SLAU2 (or any Riemann-flavoured face flux) is designed to
      resolve via χ(M)-scaled mass-flux upwind dissipation.

R2 final face state (no free parameters):
    u_face = ½ (u_L + u_R)                ← central
    p_face = ½ (p_L + p_R)                ← central — PE-preserving
    α_face, T1_face, T2_face = upwind side (advected scalars)

References:
  Saurel, Petitpas, Berry 2009 §3.1 (Allaire-Massoni hybrid splitting)
  Coquel, Hérard, Saleh 2017 §4 (entropy-consistent face state)
"""
from __future__ import annotations
import numpy as np


__all__ = ['face_upwind_state']


def face_upwind_state(W_ext):
    """R2 final face state — pressure/velocity central, mass advection upwind.

    Parameters
    ----------
    W_ext : 5-tuple of (N+2,) arrays
        Ghost-extended primitive (α, T1, T2, u, p) with ng = 1.

    Returns
    -------
    face : dict with keys
        alpha, T1, T2 — upwind side                   (Nf,)
        u            — central average ½(u_L + u_R)   (Nf,)
        p            — central average ½(p_L + p_R)   (Nf,)
        upwind_left  — bool, True where u_face ≥ 0    (Nf,)
    """
    alpha_e, T1_e, T2_e, u_e, p_e = W_ext

    # Central averages — pressure and velocity (PE-preserving)
    u_face = 0.5 * (u_e[:-1] + u_e[1:])
    p_face = 0.5 * (p_e[:-1] + p_e[1:])

    # Upwind on the advected scalars (α, T_k)
    upwind_left = u_face >= 0.0
    a_f  = np.where(upwind_left, alpha_e[:-1], alpha_e[1:])
    T1_f = np.where(upwind_left, T1_e[:-1],   T1_e[1:])
    T2_f = np.where(upwind_left, T2_e[:-1],   T2_e[1:])

    return dict(alpha=a_f, T1=T1_f, T2=T2_f,
                u=u_face, p=p_face,
                upwind_left=upwind_left)
