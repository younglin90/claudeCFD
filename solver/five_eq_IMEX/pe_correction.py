"""Residual-level PE correction (ChatGPT v2 §6.4).

For a uniform-(p, u) state with α-jump, the discrete residual

    R_U^raw = (U(W) − U^n)/Δt + L_E(W) + L_I(W)

projects onto the pressure-equilibrium normal direction with a small but
non-zero coefficient (the seed of the spectral PE-violating mode).  The
correction modifies *only the energy component* so that the projection
vanishes byte-exactly:

    R_E^new = R_E^raw  −  (∂p/∂U)·R_U^raw  /  (∂p/∂(ρE))

This makes the cell update consistent with the manifold

    p(U^{n+1}) = p(U^n)        whenever the W-state is in PE.

Mathematically this is equivalent to *absorbing* the PE-violating component
of R_U^raw into the energy update — the conserved variable U^{n+1} differs
from the raw IMEX update only along the energy direction.

The Jacobian rows (∂p/∂U_k) at fixed (W) are obtained by inverting the
analytic dU/dW (Phase 2 product) and extracting the row corresponding to p.
For the user-spec W-ordering W = (α, T₁, T₂, u, p) the p-row of W is index 4,
so (∂W/∂U) = (dU/dW)^{-1}, and (∂p/∂U) is its bottom row.
"""
from __future__ import annotations
import numpy as np

from .he2024_compat import load_primitive_W

dUdW_analytic = load_primitive_W().dUdW_analytic

_EPS = 1e-30


def dpdU(W, eos1, eos2):
    """Return (∂p/∂U_0, …, ∂p/∂U_4) at each cell — bottom row of (dU/dW)^{-1}.

    Shape: (5, N) — one row per conservative-component.
    """
    J = dUdW_analytic(W, eos1, eos2)        # (5, 5, N)
    N = J.shape[-1]
    dpdU_rows = np.empty((5, N), dtype=float)
    # Solve dU/dW · X = e_p (e_p picks the W=p row → bottom)
    # Then dW/dU = X.  We want the p-row of dW/dU = inv(dU/dW)[4, :].
    e_p = np.zeros((5, 1))
    e_p[4, 0] = 1.0
    for j in range(N):
        try:
            inv_row = np.linalg.solve(J[:, :, j].T, np.array([0., 0., 0., 0., 1.0]))
            # inv_row solves J^T · x = e_4 → x = (J^{-1})^T e_4 = bottom row of J^{-1}? No,
            # we want the row vector y = e_4^T · J^{-1}, i.e. (J^{-T} e_4)^T.
            dpdU_rows[:, j] = inv_row if np.all(np.isfinite(inv_row)) else 0.0
        except np.linalg.LinAlgError:
            dpdU_rows[:, j] = 0.0
    return dpdU_rows


def apply_pe_correction(R_tuple, W, eos1, eos2):
    """Project R_U^raw onto PE-tangent and rebalance the energy residual.

    Returns the corrected 5-tuple of (N,) arrays.  The correction touches
    only R[3] (energy row) by default — this is the simplest "well-balanced"
    PE projection.
    """
    dpdU_rows = dpdU(W, eos1, eos2)        # (5, N)
    # PE-normal projection of the residual:
    #   π = Σ_k (∂p/∂U_k) · R_U[k]
    pi = np.zeros_like(R_tuple[0])
    finite = np.ones_like(R_tuple[0], dtype=bool)
    with np.errstate(over='ignore', invalid='ignore'):
        for k in range(5):
            finite &= np.isfinite(dpdU_rows[k]) & np.isfinite(R_tuple[k])
            pi = pi + dpdU_rows[k] * R_tuple[k]
    # Energy-component sensitivity: ∂p/∂(ρE)
    dpdrhoE = dpdU_rows[3]
    # Avoid division by 0 — fall back to no correction
    safe = finite & np.isfinite(pi) & np.isfinite(dpdrhoE) & (np.abs(dpdrhoE) > _EPS)
    delta_RE = np.where(safe, -pi / np.where(safe, dpdrhoE, 1.0), 0.0)
    R_new = list(R_tuple)
    R_new[3] = R_tuple[3] + delta_RE
    return tuple(R_new), pi


def _contact_projection_mask(W, *,
                             alpha_grad_tol=1e-8,
                             pressure_contact_tol=1e-8,
                             velocity_contact_tol=1e-8):
    """Cells that look like material contacts: grad(alpha) without acoustic grad."""
    a, _, _, u, p = W
    da = np.maximum(np.abs(a - np.roll(a, 1)), np.abs(np.roll(a, -1) - a))
    dp = np.maximum(np.abs(p - np.roll(p, 1)), np.abs(np.roll(p, -1) - p))
    du = np.maximum(np.abs(u - np.roll(u, 1)), np.abs(np.roll(u, -1) - u))
    p_ref = max(float(np.nanmax(np.abs(p))), 1.0)
    u_ref = max(float(np.nanmax(np.abs(u))), 1.0)
    return ((da > alpha_grad_tol)
            & (dp / p_ref < pressure_contact_tol)
            & (du / u_ref < velocity_contact_tol))


def _interface_projection_mask(W, *, alpha_grad_tol=1e-8):
    """Cells adjacent to a material interface, independent of acoustic content."""
    a = W[0]
    da = np.maximum(np.abs(a - np.roll(a, 1)), np.abs(np.roll(a, -1) - a))
    return da > alpha_grad_tol


def _interface_band_projection_mask(W, *, alpha_grad_tol=1e-8, radius=6):
    """Cells in a narrow band around material interfaces."""
    mask = _interface_projection_mask(W, alpha_grad_tol=alpha_grad_tol)
    if radius <= 0 or not np.any(mask):
        return mask
    out = mask.copy()
    for s in range(1, radius + 1):
        out |= np.roll(mask, s) | np.roll(mask, -s)
    return out


def _impedance_projection_weight(W, eos1, eos2, *,
                                 alpha_grad_tol=1e-8,
                                 min_log10_ratio=2.0,
                                 max_strength=0.35):
    """Projection only at very strong impedance material interfaces."""
    a, T1, T2, _u, p = W
    interface = _interface_projection_mask(W, alpha_grad_tol=alpha_grad_tol)
    try:
        from .sound_speed import phase_sound_speed_sq
        rho1 = eos1.density(p, T1)
        rho2 = eos2.density(p, T2)
        c1 = np.sqrt(np.maximum(phase_sound_speed_sq(eos1, rho1, T1), 1.0e-30))
        c2 = np.sqrt(np.maximum(phase_sound_speed_sq(eos2, rho2, T2), 1.0e-30))
        Z1 = np.maximum(rho1 * c1, 1.0e-30)
        Z2 = np.maximum(rho2 * c2, 1.0e-30)
        ratio = np.maximum(Z1 / Z2, Z2 / Z1)
        strength = max_strength * np.clip(
            (np.log10(np.maximum(ratio, 1.0)) - min_log10_ratio) / 1.0,
            0.0, 1.0)
    except Exception:
        strength = np.zeros_like(a)
    return np.where(interface, strength, 0.0)


def _sensor_projection_weight(W, eos1, eos2, *,
                              alpha_grad_tol=1e-8,
                              pressure_contact_tol=1e-8,
                              velocity_contact_tol=1e-8):
    """Smooth PE-projection strength for 07 diagnostics.

    Pure material contacts get weight near one. Acoustic gas-gas waves get
    weight near zero. Strong impedance interfaces keep a small projection floor
    so Air-Water does not lose the PE stabilizer entirely.
    """
    a, T1, T2, u, p = W
    da = np.maximum(np.abs(a - np.roll(a, 1)), np.abs(np.roll(a, -1) - a))
    dp = np.maximum(np.abs(p - np.roll(p, 1)), np.abs(np.roll(p, -1) - p))
    du = np.maximum(np.abs(u - np.roll(u, 1)), np.abs(np.roll(u, -1) - u))
    p_ref = max(float(np.nanmax(np.abs(p))), 1.0)

    material = np.clip(da / max(alpha_grad_tol, 1.0e-12), 0.0, 1.0)

    try:
        from .sound_speed import phase_sound_speed_sq, mixture_sound_speed_sq
        rho1 = eos1.density(p, T1)
        rho2 = eos2.density(p, T2)
        c1_sq = phase_sound_speed_sq(eos1, rho1, T1)
        c2_sq = phase_sound_speed_sq(eos2, rho2, T2)
        c_mix = np.sqrt(np.maximum(
            mixture_sound_speed_sq(a, rho1, c1_sq, rho2, c2_sq, kind='kapila'),
            1.0e-30))
        rho = a * rho1 + (1.0 - a) * rho2
        Z = np.maximum(rho * c_mix, 1.0e-30)
        Z_l = np.roll(Z, 1)
        Z_r = np.roll(Z, -1)
        ratio_l = np.maximum(Z / Z_l, Z_l / Z)
        ratio_r = np.maximum(Z / Z_r, Z_r / Z)
        Z_ratio = np.maximum(ratio_l, ratio_r)
        c_ref = max(float(np.nanmax(c_mix)), 1.0)
    except Exception:
        Z_ratio = np.ones_like(a)
        c_ref = 1.0

    p_tol = max(pressure_contact_tol, 1.0e-6)
    u_tol = max(velocity_contact_tol, 1.0e-6)
    acoustic_flat = np.exp(-((dp / (p_tol * p_ref)) ** 2
                             + (du / (u_tol * c_ref)) ** 2))
    # Keep gas-gas acoustic interfaces effectively unprojected.  Only very
    # strong impedance jumps (Air-Water scale) retain a projection floor.
    strong_impedance = np.clip((np.log10(np.maximum(Z_ratio, 1.0)) - 2.0) / 1.0,
                               0.0, 1.0)
    impedance_floor = 0.80 * strong_impedance
    weight = material * np.maximum(acoustic_flat, impedance_floor)
    return np.where(np.isfinite(weight), np.clip(weight, 0.0, 1.0), 0.0)


def apply_pe_tangent_projection(R_tuple, W, eos1, eos2, *,
                                mode='always',
                                alpha_grad_tol=1e-8,
                                pressure_contact_tol=1e-8,
                                velocity_contact_tol=1e-8):
    """Project residual onto PE tangent by removing the dpdU-normal component.

    Unlike `apply_pe_correction` (energy-only rebalance), this adjusts all
    conservative components:

        R_new = R_raw - beta * g,   g = dpdU,   beta = (g·R)/(g·g)
    """
    g = dpdU(W, eos1, eos2)   # (5, N)
    num = np.zeros_like(R_tuple[0])
    den = np.zeros_like(R_tuple[0])
    finite = np.ones_like(R_tuple[0], dtype=bool)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        for k in range(5):
            finite &= np.isfinite(g[k]) & np.isfinite(R_tuple[k])
            num = num + g[k] * R_tuple[k]
            den = den + g[k] * g[k]
        safe = finite & np.isfinite(num) & np.isfinite(den) & (np.abs(den) > _EPS)
        weight = np.ones_like(num)
        if mode == 'contact':
            safe = safe & _contact_projection_mask(
                W,
                alpha_grad_tol=alpha_grad_tol,
                pressure_contact_tol=pressure_contact_tol,
                velocity_contact_tol=velocity_contact_tol)
        elif mode == 'interface':
            safe = safe & _interface_projection_mask(
                W, alpha_grad_tol=alpha_grad_tol)
        elif mode == 'interface_band':
            safe = safe & _interface_band_projection_mask(
                W, alpha_grad_tol=alpha_grad_tol)
        elif mode == 'impedance':
            weight = _impedance_projection_weight(
                W, eos1, eos2, alpha_grad_tol=alpha_grad_tol)
            safe = safe & (weight > 0.0)
        elif mode == 'sensor':
            weight = _sensor_projection_weight(
                W, eos1, eos2,
                alpha_grad_tol=alpha_grad_tol,
                pressure_contact_tol=pressure_contact_tol,
                velocity_contact_tol=velocity_contact_tol)
            safe = safe & (weight > 0.0)
        elif mode != 'always':
            raise ValueError(f"Unknown PE projection mode='{mode}'.")
        beta = np.where(safe, weight * num / np.where(safe, den, 1.0), 0.0)
    R_new = [None] * 5
    for k in range(5):
        R_new[k] = np.where(safe, R_tuple[k] - beta * g[k], R_tuple[k])
    return tuple(R_new), num
