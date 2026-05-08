"""TVD slope limiters and the T-MLP-u wrapper.

# ─── Classical TVD limiters ψ(r) ────────────────────────────────────────────
Each ψ has fixed mathematical form — no tuning parameters.  All accept the
slope ratio r = Δ_-/Δ_+ and return ψ(r) ∈ [0, 2].

  minmod     : ψ_MM(r)    = max(0, min(1, r))                        Roe 1986
  van_leer   : ψ_VL(r)    = (r + |r|)/(1 + |r|)  ≡ 2r/(1+r) for r>0  van Leer 1974
  superbee   : ψ_SB(r)    = max(0, min(2r,1), min(r,2))              Roe 1986
  van_albada : ψ_VA(r)    = (r²+r)/(r²+1)                            van Albada 1982
  mc         : ψ_MC(r)    = max(0, min(2r, ½(1+r), 2))               monotonized central
  umist      : ψ_UMIST(r) = max(0, min(2r, ¼+¾r, ¾+¼r, 2))           Lien-Leschziner 1994

# ─── T-MLP-u wrapper ────────────────────────────────────────────────────────
T-MLP-u takes any base TVD limiter ψ_TVD and adds a Local Maximum Principle
(LMP) bound on top.  Per face side (using the upstream / cell / downstream
triplet (UU, U, D)):

    Δ_+ = φ_D − φ_U
    δ   = ½ (1 − C_f) Δ_+               (Hancock factor C_f optional)
    r   = (φ_U − φ_UU) / Δ_+
    ψ_TVD = (chosen TVD limiter)(r)
    φ_min = min(φ_UU, φ_U, φ_D)
    φ_max = max(φ_UU, φ_U, φ_D)

LMP bound on δ:
    if δ > 0:  ψ_MLP = (φ_max − φ_U) / δ
    if δ < 0:  ψ_MLP = (φ_min − φ_U) / δ

Final limiter:
    ψ_final = max(0, min(2, ψ_TVD, ψ_MLP))

Reconstructed face value:
    φ_face = φ_U + ψ_final · δ

Free parameters: 0 (all formulae fixed).  C_f only enters when the user
plugs in a Hancock predictor-corrector — for plain MUSCL pass C_f = 0.
"""
from __future__ import annotations
import numpy as np


__all__ = [
    'minmod', 'van_leer', 'superbee', 'van_albada', 'mc', 'umist',
    'downwind',
    'minmod2',
    't_mlp_u_face_value',
    'TVD_LIMITERS',
]

_EPS = 1e-30


# ─── Two-argument helper ───────────────────────────────────────────────────
def minmod2(a, b):
    """Symmetric two-argument minmod (sign-agreement, smaller magnitude)."""
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))


# ─── Classical TVD limiters ψ(r) ───────────────────────────────────────────
def minmod(r):
    return np.maximum(0.0, np.minimum(1.0, r))


def van_leer(r):
    abs_r = np.abs(r)
    return (r + abs_r) / (1.0 + abs_r)


def superbee(r):
    a = np.minimum(2.0 * r, 1.0)
    b = np.minimum(r, 2.0)
    return np.maximum(0.0, np.maximum(a, b))


def van_albada(r):
    return (r * r + r) / (r * r + 1.0)


def mc(r):
    """Monotonised central: ψ_MC(r) = max(0, min(2r, ½(1+r), 2))."""
    return np.maximum(0.0,
                      np.minimum(np.minimum(2.0 * r, 0.5 * (1.0 + r)), 2.0))


def umist(r):
    """UMIST: ψ_UMIST(r) = max(0, min(2r, ¼+¾r, ¾+¼r, 2))."""
    a = 2.0 * r
    b = 0.25 + 0.75 * r
    c = 0.75 + 0.25 * r
    return np.maximum(0.0,
                      np.minimum(np.minimum(np.minimum(a, b), c), 2.0))


def downwind(r):
    """Roe ultrabee / fully-downwind TVD limiter:

        ψ_DW(r) = max(0, min(2r, 2))

    The most compressive symmetric limiter still inside Sweby's TVD
    region.  ψ_DW(0)=0, ψ_DW(½)=1, ψ_DW(1)=2 (the face value coincides
    with the *downstream* cell value), ψ_DW(r≥1)=2.  Used standalone it
    over-compresses (anti-diffusive); combined with the T-MLP-u wrapper
    the LMP bound clips just enough to stay monotone.
    """
    return np.maximum(0.0, np.minimum(2.0 * r, 2.0))


def hyper_c(r, courant=0.4):
    """Hyper-C (Leonard 1991) — compressive scheme used as the
    sharp-interface arm of CICSAM (Ubbink 1997).

        ψ_HC(r) = 2 · min(1, r · (1−Co)/Co)

    Derivation: in NVD, Hyper-C says φ̃_f = min(1, φ̃_C/Co).  Converting
    back to TVD gives a limiter MORE compressive than downwind for any
    Co < ½ — at Co=0.4 the slope is 3r (vs 2r for downwind).  Best
    paired with a robust LMP wrapper (T-MLP-u) since it sits well above
    the Sweby region; CICSAM's full version blends it with Ultimate-
    QUICKEST via cos²(2θ) for smooth-region anti-compression.

    `courant` is a fixed estimate; for true CICSAM Co should be the
    actual face Courant number per step.
    """
    factor = (1.0 - courant) / max(courant, 1e-10)
    return np.maximum(0.0, np.minimum(2.0 * r * factor, 2.0))


def hyper_c_co3(r):
    """Hyper-C at Co=0.3 — slope 7/3·r (extra compressive)."""
    return hyper_c(r, courant=0.3)


def hyper_c_co35(r):
    """Hyper-C at Co=0.35 — slope ≈ 1.857·r."""
    return hyper_c(r, courant=0.35)


def hyper_c_co45(r):
    """Hyper-C at Co=0.45 — slope ≈ 1.222·r (gentler than 0.4)."""
    return hyper_c(r, courant=0.45)


TVD_LIMITERS = {
    'minmod':       minmod,
    'van_leer':     van_leer,
    'superbee':     superbee,
    'van_albada':   van_albada,
    'mc':           mc,
    'umist':        umist,
    'downwind':     downwind,
    'hyper_c':      hyper_c,
    'cicsam':       hyper_c,    # alias — Hyper-C is CICSAM's compressive arm
    'cicsam_co3':   hyper_c_co3,
    'cicsam_co35':  hyper_c_co35,
    'cicsam_co45':  hyper_c_co45,
}


# ─── T-MLP-u face-value helper ─────────────────────────────────────────────
def t_mlp_u_face_value(phi_UU, phi_U, phi_D, psi_tvd_fn,
                       hancock_courant: float = 0.0):
    """Return the T-MLP-u-limited reconstructed face value.

    Inputs may be scalars, 1D arrays (per-face), or 2D arrays (nvar, n_faces).
    Broadcasting follows numpy rules — we operate component-wise.

    Steps:
        Δ_+ = φ_D − φ_U
        δ   = ½ (1 − C_f) Δ_+
        r   = (φ_U − φ_UU) / Δ_+    (safe-guarded against 0)
        ψ_TVD = psi_tvd_fn(r)
        ψ_MLP = (φ_max − φ_U)/δ if δ > 0 else (φ_min − φ_U)/δ
        ψ_final = max(0, min(2, ψ_TVD, ψ_MLP))
        φ_face = φ_U + ψ_final · δ

    Returns φ_face with shape determined by broadcasting.
    """
    delta_plus = phi_D - phi_U
    delta = 0.5 * (1.0 - hancock_courant) * delta_plus

    # Slope ratio for the TVD limiter — guard against |Δ_+| = 0.
    sign_dp = np.where(delta_plus >= 0.0, 1.0, -1.0)
    safe_dp = np.where(np.abs(delta_plus) > _EPS, delta_plus, sign_dp * _EPS)
    delta_minus = phi_U - phi_UU
    r = delta_minus / safe_dp
    psi_tvd = psi_tvd_fn(r)

    # MLP local bound — different sign branches for δ.
    phi_min = np.minimum(np.minimum(phi_UU, phi_U), phi_D)
    phi_max = np.maximum(np.maximum(phi_UU, phi_U), phi_D)

    # We need ψ such that  φ_U + ψ·δ ∈ [φ_min, φ_max].
    #   δ > 0:  ψ ≤ (φ_max − φ_U)/δ
    #   δ < 0:  ψ ≤ (φ_min − φ_U)/δ        (note (φ_min − φ_U) ≤ 0 and δ < 0 → positive)
    # δ ≈ 0: any ψ keeps φ_face ≈ φ_U; pick the universal cap (= 2).
    safe_pos = np.where(delta >  _EPS,  delta, _EPS)
    safe_neg = np.where(delta < -_EPS,  delta, -_EPS)
    psi_mlp_pos = (phi_max - phi_U) / safe_pos
    psi_mlp_neg = (phi_min - phi_U) / safe_neg
    psi_mlp = np.where(delta >  _EPS, psi_mlp_pos,
              np.where(delta < -_EPS, psi_mlp_neg,
                       np.full_like(delta + 0.0, 2.0)))

    psi_final = np.maximum(0.0,
                           np.minimum(2.0, np.minimum(psi_tvd, psi_mlp)))
    return phi_U + psi_final * delta
