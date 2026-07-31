"""TVD slope limiters and the T-MLP-u wrapper.

# ─── Classical TVD limiters ψ(r) ────────────────────────────────────────────
Each ψ has fixed mathematical form — no tuning parameters.  All accept the
slope ratio r = Δ_-/Δ_+ and return ψ(r) ∈ [0, 2].

  minmod     : ψ_MM(r)    = max(0, min(1, r))                        Roe 1986
  van_leer   : ψ_VL(r)    = (r + |r|)/(1 + |r|)  ≡ 2r/(1+r) for r>0  van Leer 1974
  superbee   : ψ_SB(r)    = max(0, min(2r,1), min(r,2))              Roe 1986
  modified_superbee:
                ψ_MSB(r) = max(0, min(1.5r,1), min(r,1.5))
  van_albada : ψ_VA(r)    = (r²+r)/(r²+1)                            van Albada 1982
  mc         : ψ_MC(r)    = max(0, min(2r, ½(1+r), 2))               monotonized central
  umist      : ψ_UMIST(r) = max(0, min(2r, ¼+¾r, ¾+¼r, 2))           Lien-Leschziner 1994
  koren      : ψ_K(r)     = max(0, min(2r, (1+2r)/3, 2))             Koren 1993
  bounded_cd : ψ_CD(r)    = 1                                        central difference

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
    'minmod', 'van_leer', 'superbee', 'modified_superbee',
    'van_albada', 'mc', 'umist', 'koren',
    'stoic', 'stacs', 'mstacs',
    'bounded_cd', 'downwind',
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


def modified_superbee(r):
    """Less-compressive SUPERBEE variant with Sweby cap beta=1.5."""
    beta = 1.5
    a = np.minimum(beta * r, 1.0)
    b = np.minimum(r, beta)
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


def koren(r):
    """Koren third-order upwind TVD limiter.

    ψ_K(r) = max(0, min(2r, (1+2r)/3, 2)).
    """
    a = 2.0 * r
    b = (1.0 + 2.0 * r) / 3.0
    return np.maximum(0.0, np.minimum(np.minimum(a, b), 2.0))


def _nvd_coord_from_tmlpu_r(r):
    """Return NVD donor coordinate for TMLP-u's face ratio convention.

    The unstructured TMLP-u paper form uses

        r = (phi_R - phi_L) / (phi_L - phi_LL).

    With phi_LL, phi_L, phi_R as upwind/current/downwind states, the
    normalized current-cell value is

        C_D = (phi_L - phi_LL) / (phi_R - phi_LL) = 1 / (1 + r).

    Non-monotone r<=0 states are mapped to the upwind fallback by the caller.
    """
    rr = np.asarray(r, dtype=float)
    rr_pos = np.maximum(rr, 0.0)
    return 1.0 / (1.0 + rr_pos)


def _psi_from_nvd_current(cd, cf):
    """Convert NVD face value to the TMLP-u limiter coefficient.

    For alpha_f=0.5, C_f = C_D + 0.5*psi*(1-C_D), hence

        psi = 2*(C_f-C_D)/(1-C_D).
    """
    denom = np.maximum(1.0 - cd, 1.0e-30)
    psi = 2.0 * (cf - cd) / denom
    return np.maximum(0.0, psi)


def _nvd_superbee(cd):
    return np.where(
        cd < 0.0, cd,
        np.where(
            cd < 1.0 / 3.0, 2.0 * cd,
            np.where(
                cd < 0.5, 0.5 + 0.5 * cd,
                np.where(
                    cd < 2.0 / 3.0, 1.5 * cd,
                    np.where(cd <= 1.0, 1.0, cd)))))


def _nvd_stoic(cd):
    return np.where(
        cd < 0.0, cd,
        np.where(
            cd < 0.2, 3.0 * cd,
            np.where(
                cd < 0.5, 0.5 + 0.5 * cd,
                np.where(
                    cd < 5.0 / 6.0, 0.375 + 0.75 * cd,
                    np.where(cd <= 1.0, 1.0, cd)))))


def stoic(r):
    """STOIC NVD high-resolution scheme as a TMLP-u psi(r) arm.

    STOIC is the HR arm used in STACS/MSTACS.  The piecewise NVD definition
    is converted to a face-limiter coefficient for the TMLP-u paper ratio.
    """
    rr = np.asarray(r, dtype=float)
    cd = _nvd_coord_from_tmlpu_r(rr)
    cf = _nvd_stoic(cd)
    return np.where(rr > 0.0, _psi_from_nvd_current(cd, cf), 0.0)


def stacs(r):
    """STACS compressive aligned-interface arm.

    Darwish-Moukalled STACS blends SUPERBEE and STOIC with a cos(theta)^4
    weight.  The scalar TVD limiter API has no face/interface angle, so this
    arm represents the sharp, aligned-interface limit used by the BVD sharp
    candidate.  The smooth branch is supplied separately by bounded CD.
    """
    rr = np.asarray(r, dtype=float)
    cd = _nvd_coord_from_tmlpu_r(rr)
    cf = _nvd_superbee(cd)
    return np.where(rr > 0.0, _psi_from_nvd_current(cd, cf), 0.0)


def mstacs(r, courant=0.4):
    """MSTACS compressive differencing arm in NVD form.

    For Co <= 0.33 MSTACS uses Hyper-C, C_f=min(C_D/Co, 1).  For larger
    Courant numbers it uses C_f=min(3*C_D, 1).  STOIC is the HR arm in the
    full scheme; here BVD supplies the smooth/sharp switching.
    """
    rr = np.asarray(r, dtype=float)
    cd = _nvd_coord_from_tmlpu_r(rr)
    if courant <= 0.33:
        cf = np.minimum(cd / max(courant, 1.0e-10), 1.0)
    else:
        cf = np.minimum(3.0 * cd, 1.0)
    cf = np.where((cd >= 0.0) & (cd <= 1.0), cf, cd)
    return np.where(rr > 0.0, _psi_from_nvd_current(cd, cf), 0.0)


def mstacs_co25(r):
    return mstacs(r, courant=0.25)


def mstacs_co5(r):
    return mstacs(r, courant=0.5)


def mstacs_co75(r):
    return mstacs(r, courant=0.75)


def bounded_cd(r):
    """Central-difference limiter arm: ψ_CD(r) = 1.

    With the generic T-MLP-u face formula

        phi_f = phi_U + psi * alpha_f * (phi_D - phi_U),

    this gives bounded central differencing when ``mlp_bound=True`` and
    plain central differencing when the MLP bound is disabled.
    """
    return np.ones_like(r, dtype=float)


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


def pure_downwind(r):
    """Pure downwind reconstruction: φ_L ≡ φ_D, φ_R ≡ φ_U (downstream).

        ψ ≡ 2  ∀ r  (no TVD constraint, no extremum cutoff)

    Anti-diffusive everywhere — even at local extrema where TVD
    schemes shut off (ψ=0).  Mathematically unstable; provided as a
    stress test to demonstrate the necessity of the LMP wrapper or
    a TVD constraint.
    """
    return np.full_like(r, 2.0)


def tmlpu_shape(r):
    """Smooth compressive arm for T-MLP-u shape preservation.

    The limiter is central at locally linear data (r=1 ⇒ ψ=1) and
    smoothly becomes downwind-compressive as r moves away from one:

        ψ(r) = 1 + tanh(|log(max(r, eps))|)

    Negative or zero r is treated as the compressive limit.  This is intended
    to be used inside the T-MLP-u vertex LMP wrapper, which supplies the
    monotonicity/boundedness constraint.
    """
    rr = np.maximum(np.asarray(r, dtype=float), 1.0e-12)
    return np.minimum(2.0, 1.0 + np.tanh(np.abs(np.log(rr))))


def tmlpu_preserve(r):
    """One-sided smooth-compressive arm for shape preservation.

    The symmetric ``tmlpu_shape`` arm compresses both r < 1 and r > 1.
    That is sharp for discontinuities but can squeeze smooth cone/hump
    profiles.  This arm keeps central differencing on the r >= 1 side and
    only moves toward downwind compression when r < 1:

        ψ(r) = 1 + tanh(max(0, -log(max(r, eps))))

    It is a single continuous ψ_TVD arm, not a smoothness-threshold switch.
    The T-MLP-u vertex bound supplies the multidimensional boundedness cap.
    """
    rr = np.maximum(np.asarray(r, dtype=float), 1.0e-12)
    return np.minimum(2.0, 1.0 + np.tanh(np.maximum(-np.log(rr), 0.0)))


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


def hyper_c_co38(r):
    """Hyper-C at Co=0.38 — slope ≈ 1.632·r (between 0.35 and 0.4)."""
    return hyper_c(r, courant=0.38)


def hyper_c_co42(r):
    """Hyper-C at Co=0.42 — slope ≈ 1.381·r (between 0.4 and 0.45)."""
    return hyper_c(r, courant=0.42)


def hyper_c_co34(r):
    """Hyper-C at Co=0.34 — slope ≈ 1.941·r (more compressive)."""
    return hyper_c(r, courant=0.34)


def hyper_c_co36(r):
    """Hyper-C at Co=0.36 — slope ≈ 1.778·r (between 0.34 and 0.38)."""
    return hyper_c(r, courant=0.36)


TVD_LIMITERS = {
    'minmod':       minmod,
    'van_leer':     van_leer,
    'vanleer':      van_leer,
    'van-leer':     van_leer,
    'superbee':     superbee,
    'modified_superbee': modified_superbee,
    'superbee15':   modified_superbee,
    'van_albada':   van_albada,
    'mc':           mc,
    'umist':        umist,
    'koren':        koren,
    'stoic':        stoic,
    'stacs':        stacs,
    'mstacs':       mstacs,
    'mstacs_co25':  mstacs_co25,
    'mstacs_co5':   mstacs_co5,
    'mstacs_co75':  mstacs_co75,
    'bounded_cd':   bounded_cd,
    'central':      bounded_cd,
    'cd':           bounded_cd,
    'downwind':     downwind,
    'pure_downwind': pure_downwind,
    'tmlpu_shape':  tmlpu_shape,
    'shape':        tmlpu_shape,
    'tmlpu_preserve': tmlpu_preserve,
    'preserve':     tmlpu_preserve,
    'hyper_c':      hyper_c,
    'cicsam':       hyper_c,    # alias — Hyper-C is CICSAM's compressive arm
    'cicsam_co3':   hyper_c_co3,
    'cicsam_co34':  hyper_c_co34,
    'cicsam_co35':  hyper_c_co35,
    'cicsam_co36':  hyper_c_co36,
    'cicsam_co38':  hyper_c_co38,
    'cicsam_co42':  hyper_c_co42,
    'cicsam_co45':  hyper_c_co45,
}


# ─── T-MLP-u face-value helper ─────────────────────────────────────────────
def t_mlp_u_face_value(phi_UU, phi_U, phi_D, psi_tvd_fn,
                       hancock_courant: float = 0.0,
                       alpha_f: float = 0.5):
    """Return the T-MLP-u-limited reconstructed face value.

    Inputs may be scalars, 1D arrays (per-face), or 2D arrays (nvar, n_faces).
    Broadcasting follows numpy rules — we operate component-wise.

    Steps:
        Δ_+ = φ_D − φ_U
        δ   = α_f (1 − C_f) Δ_+
        r   = (φ_U − φ_UU) / Δ_+    (safe-guarded against 0)
        ψ_TVD = psi_tvd_fn(r)
        ψ_MLP = (φ_max − φ_U)/δ if δ > 0 else (φ_min − φ_U)/δ
        ψ_final = max(0, min(2, ψ_TVD, ψ_MLP))
        φ_face = φ_U + ψ_final · δ

    Returns φ_face with shape determined by broadcasting.
    """
    delta_plus = phi_D - phi_U
    delta = alpha_f * (1.0 - hancock_courant) * delta_plus

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
