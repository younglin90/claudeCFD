"""Shared macro low-order primitive for synthetic acceleration family.

Provides FFT-based linear Stokes inverse for periodic macro residual:

    Given macro residual R_U = (R_rho, R_jx, R_jy):
        - Solve div-free Stokes problem for δu, δv
        - Mass: δρ chosen to satisfy continuity (often zero-mean fix)
        - Return δU = (δρ, δjx, δjy) in lift-compatible form

For non-periodic walls, residual mode k=0 of momentum is the mean shear,
preserved (Schur PC handles this).
"""
import numpy as np


def fft_stokes_inverse(R_U, nu, omega=None):
    """Inverse Stokes: ν ∇² u = -R_jx,  ν ∇² v = -R_jy.

    Returns δU = (δρ=0, δu, δv) macro correction.
    """
    _, N, _ = R_U.shape
    R_jx_hat = np.fft.fft2(R_U[1])
    R_jy_hat = np.fft.fft2(R_U[2])
    kx = 2.0 * np.pi * np.fft.fftfreq(N) * N
    ky = 2.0 * np.pi * np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    k2 = KX * KX + KY * KY
    k2[0, 0] = 1.0
    du_hat = R_jx_hat / (nu * k2)
    dv_hat = R_jy_hat / (nu * k2)
    du_hat[0, 0] = 0.0
    dv_hat[0, 0] = 0.0
    # Project to div-free: subtract (k · u) k / k²
    div = (KX * du_hat + KY * dv_hat) / k2
    div[0, 0] = 0.0
    du_hat = du_hat - KX * div
    dv_hat = dv_hat - KY * div
    du = np.real(np.fft.ifft2(du_hat))
    dv = np.real(np.fft.ifft2(dv_hat))
    drho = np.zeros_like(du)
    return np.stack([drho, du, dv], axis=0)


def hot_stress_from_fneq(f_neq, CX, CY):
    """Extract 2nd-moment HoT stress tensor π_αβ = Σ c_α c_β f_neq."""
    pi_xx = (f_neq * (CX * CX)[:, None, None]).sum(axis=0)
    pi_xy = (f_neq * (CX * CY)[:, None, None]).sum(axis=0)
    pi_yy = (f_neq * (CY * CY)[:, None, None]).sum(axis=0)
    return pi_xx, pi_xy, pi_yy


def divergence_2tensor(pi_xx, pi_xy, pi_yy, dx=1.0):
    """∂_β π_αβ via centered diff, periodic."""
    dpi_x = (np.roll(pi_xx, -1, axis=1) - np.roll(pi_xx, 1, axis=1)) / (2 * dx) + \
            (np.roll(pi_xy, -1, axis=0) - np.roll(pi_xy, 1, axis=0)) / (2 * dx)
    dpi_y = (np.roll(pi_xy, -1, axis=1) - np.roll(pi_xy, 1, axis=1)) / (2 * dx) + \
            (np.roll(pi_yy, -1, axis=0) - np.roll(pi_yy, 1, axis=0)) / (2 * dx)
    return dpi_x, dpi_y
