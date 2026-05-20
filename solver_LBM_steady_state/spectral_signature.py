"""Residual spectral signature indicators for ASH adaptive dispatch.

Inputs: case (has project=M, dof, shape), R (distribution residual f - L(f))
Output: dict with χ_low, H_k, χ_kin, q_k

Usage in dispatch:
    sig = compute_signature(case, R, R_prev)
    if sig['H_k'] < 0.65 and sig['chi_low'] > 0.7:  smooth periodic → SAN
    elif sig['chi_kin'] > 0.5:                       kinetic-heavy → Lean
    elif sig['chi_low'] > 0.5:                       mild wall → Safe-NN
    else:                                            stiff broad → TR/Lean
"""
import numpy as np


def compute_signature(case, R, R_prev=None, k_cut_frac=0.25):
    """Return spectral signature dict.

    χ_low : low-freq macro residual energy fraction (cutoff at k_cut_frac * Nyquist)
    H_k   : normalized spectral entropy (0=single mode, 1=uniform)
    χ_kin : kinetic residual energy fraction = ||(I-TM)R|| / ||R||
    q_k   : recent reduction ||R||/||R_prev||  (None if no prev)
    """
    # macro residual
    R_U = case.project(R)                # (3, N, N)
    N = R_U.shape[-1]
    R_U_hat = np.fft.fft2(R_U, axes=(1, 2))
    energy = np.sum(np.abs(R_U_hat) ** 2, axis=0)   # (N, N)
    total = energy.sum() + 1e-30

    kxv = np.fft.fftfreq(N) * N
    kyv = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(kxv, kyv, indexing="xy")
    kmag = np.sqrt(KX * KX + KY * KY) / (0.5 * N)   # normalized to Nyquist
    mask_low = kmag <= k_cut_frac
    chi_low = float(energy[mask_low].sum() / total)

    p = energy.ravel() / total
    p = p[p > 1e-30]
    H = -float((p * np.log(p)).sum())
    H_max = np.log(N * N)
    H_norm = H / H_max if H_max > 0 else 0.0

    # kinetic residual: distribute residual minus lift of macro residual
    R_macro_lifted = case.lift(R_U)
    R_kin = R - R_macro_lifted
    norm_R = float(np.sqrt(np.sum(R * R)) + 1e-30)
    norm_kin = float(np.sqrt(np.sum(R_kin * R_kin)))
    chi_kin = norm_kin / norm_R

    q_k = None
    if R_prev is not None:
        norm_prev = float(np.sqrt(np.sum(R_prev * R_prev)) + 1e-30)
        q_k = norm_R / norm_prev

    return {
        "chi_low": chi_low,
        "H_k": H_norm,
        "chi_kin": chi_kin,
        "q_k": q_k,
        "norm_R": norm_R,
    }


def classify_regime(sig, q_thresh=0.95):
    """Return one of: 'smooth_periodic', 'mild_wall', 'kinetic_heavy', 'stiff_broad'."""
    chi_low = sig["chi_low"]
    H = sig["H_k"]
    chi_kin = sig["chi_kin"]
    q = sig.get("q_k", None)

    if H < 0.45 and chi_low > 0.7 and chi_kin < 0.5:
        return "smooth_periodic"
    if chi_kin > 0.7:
        return "kinetic_heavy"
    if chi_low > 0.5 and H < 0.7:
        return "mild_wall"
    if q is not None and q > q_thresh:
        return "stagnant"
    return "stiff_broad"
