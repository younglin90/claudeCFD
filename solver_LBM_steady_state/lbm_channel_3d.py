"""3D channel flow : periodic x+z, no-slip y-walls, constant body force F_x.

Analytical steady Poiseuille (independent of z) :
    u_x^*(y) = (F0 / 2nu) y (L - y),   L = N_y - 1
    u_y^* = u_z^* = 0
"""

import numpy as np

from lbm_3d import (CX, CY, CZ, W3D, CX_INT, CY_INT, CZ_INT,
                     equilibrium_3d, guo_source_3d)


def apply_channel3d_bc(f):
    """Full-way bounce-back on y=0 and y=N-1 walls."""
    # Bottom y=0: incoming pop indices with CY>0: 3, 7, 10, 15, 17
    bottom_in = [3, 7, 10, 15, 17]
    bottom_opp = [4, 8, 9, 16, 18]
    for i, ip in zip(bottom_in, bottom_opp):
        f[i, 0, :, :] = f[ip, 0, :, :]
    # Top y=N-1: incoming with CY<0: 4, 8, 9, 16, 18
    top_in = [4, 8, 9, 16, 18]
    top_opp = [3, 7, 10, 15, 17]
    for i, ip in zip(top_in, top_opp):
        f[i, -1, :, :] = f[ip, -1, :, :]
    return f


def stream_channel3d(f):
    """Periodic x+z, walls in y (no shift; BC handles)."""
    fn = np.empty_like(f)
    for i in range(19):
        fn[i] = np.roll(np.roll(np.roll(f[i], CZ_INT[i], axis=0),
                                  CY_INT[i], axis=1),
                          CX_INT[i], axis=2)
    return apply_channel3d_bc(fn)


class Channel3DCase:
    """3D channel : periodic x+z, walls in y, constant body force F_x."""

    def __init__(self, N, nu, F0=1e-6):
        self.N = N
        self.Nx = N; self.Ny = N; self.Nz = N
        self.nu = nu
        self.omega = 1.0 / (3.0 * nu + 0.5)
        self.F0 = F0
        self.shape = (19, N, N, N)
        self.dof = 19 * N * N * N
        self.macro_dof = 4 * N * N * N
        self.n_U = 4

        self.Fx = F0 * np.ones((N, N, N), dtype=np.float64)
        self.Fy = np.zeros((N, N, N), dtype=np.float64)
        self.Fz = np.zeros((N, N, N), dtype=np.float64)

        L = N - 1.0
        self.L_eff = L
        self.U_max = F0 * L * L / (8.0 * nu)
        self.Re = self.U_max * L / nu

    def lbe_step(self, f):
        rho = f.sum(axis=0)
        rhoux = (f * CX[:, None, None, None]).sum(axis=0)
        rhouy = (f * CY[:, None, None, None]).sum(axis=0)
        rhouz = (f * CZ[:, None, None, None]).sum(axis=0)
        ux = rhoux / rho
        uy = rhouy / rho
        uz = rhouz / rho
        ux_eq = ux + 0.5 * self.Fx / rho
        uy_eq = uy + 0.5 * self.Fy / rho
        uz_eq = uz + 0.5 * self.Fz / rho
        feq = equilibrium_3d(rho, ux_eq, uy_eq, uz_eq)
        S = guo_source_3d(rho, ux, uy, uz, self.Fx, self.Fy, self.Fz, self.omega)
        fstar = f - self.omega * (f - feq) + S
        return stream_channel3d(fstar)

    def residual(self, f):
        return f - self.lbe_step(f)

    def res_norm(self, f):
        R = self.residual(f)
        return float(np.sqrt((R * R).mean()))

    def initial_field(self):
        rho = np.ones((self.N, self.N, self.N), dtype=np.float64)
        ux = np.zeros((self.N, self.N, self.N), dtype=np.float64)
        uy = np.zeros((self.N, self.N, self.N), dtype=np.float64)
        uz = np.zeros((self.N, self.N, self.N), dtype=np.float64)
        return equilibrium_3d(rho, ux, uy, uz)

    def macro(self, f):
        rho = f.sum(axis=0)
        ux = (f * CX[:, None, None, None]).sum(axis=0) / rho
        uy = (f * CY[:, None, None, None]).sum(axis=0) / rho
        uz = (f * CZ[:, None, None, None]).sum(axis=0) / rho
        return rho, ux, uy, uz

    def analytical_ux(self):
        """u_x^*(y) Poiseuille profile (y varies along axis 1)."""
        y = np.arange(self.N, dtype=np.float64).reshape(1, self.N, 1)
        L = self.L_eff
        prof = self.F0 / (2.0 * self.nu) * y * (L - y)
        return prof * np.ones((self.N, self.N, self.N))

    def project(self, f):
        rho = f.sum(axis=0)
        rhoux = (f * CX[:, None, None, None]).sum(axis=0)
        rhouy = (f * CY[:, None, None, None]).sum(axis=0)
        rhouz = (f * CZ[:, None, None, None]).sum(axis=0)
        return np.stack([rho, rhoux, rhouy, rhouz], axis=0)

    def lift(self, dU):
        drho, drhoux, drhouy, drhouz = dU[0], dU[1], dU[2], dU[3]
        df = np.empty((19,) + drho.shape, dtype=np.float64)
        for i in range(19):
            df[i] = W3D[i] * (drho + 3.0 * (CX[i] * drhoux + CY[i] * drhouy + CZ[i] * drhouz))
        return df

    def _fast_norm(self, x):
        xr = x.ravel()
        return float(np.sqrt(xr @ xr))

    def jvp(self, w, f_base, R_base, norm_f_cached=None):
        if norm_f_cached is None:
            norm_f_cached = self._fast_norm(f_base)
        norm_w = self._fast_norm(w)
        if norm_w < 1e-30:
            return np.zeros_like(R_base)
        eps = 1e-7 * (norm_f_cached + 1.0) / norm_w
        R_pert = self.residual(f_base + eps * w)
        return (R_pert - R_base) / eps
