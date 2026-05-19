"""2D Couette flow case : periodic-x, bottom wall no-slip, top wall moves with U_wall.

Analytical steady (linear shear) :
    u_x^*(y) = U_wall * y / (N-1),  u_y^* = 0,  rho^* = 1
"""

import numpy as np

from lbm_periodic import CX, CY, W, CX_INT, CY_INT, equilibrium, stream_periodic


def apply_couette_bc(f, U_wall):
    """Bottom wall (j=0) : full-way bounce-back.
       Top wall (j=N-1)  : Ladd momentum-corrected bounce-back with u=(U_wall, 0).
    """
    # bottom (j=0): incoming 2, 5, 6
    f[2, 0, :] = f[4, 0, :]
    f[5, 0, :] = f[7, 0, :]
    f[6, 0, :] = f[8, 0, :]
    # top (j=N-1, moving lid): incoming 4, 7, 8
    rho_top = (
        f[0, -1, :] + f[1, -1, :] + f[3, -1, :]
        + 2.0 * (f[2, -1, :] + f[5, -1, :] + f[6, -1, :])
    )
    f[4, -1, :] = f[2, -1, :]
    f[7, -1, :] = f[5, -1, :] - 6.0 * W[5] * rho_top * U_wall
    f[8, -1, :] = f[6, -1, :] + 6.0 * W[6] * rho_top * U_wall
    return f


class CouetteCase:
    """Force-free Couette : periodic-x, bottom no-slip, top moving lid."""

    def __init__(self, N, nu, U_wall=0.05):
        self.N = N
        self.nu = nu
        self.omega = 1.0 / (3.0 * nu + 0.5)
        self.U_wall = U_wall
        self.shape = (9, N, N)
        self.dof = 9 * N * N
        self.macro_dof = 3 * N * N
        self.Re = U_wall * (N - 1) / nu

    def lbe_step(self, f):
        rho = f.sum(axis=0)
        rhoux = (f * CX[:, None, None]).sum(axis=0)
        rhouy = (f * CY[:, None, None]).sum(axis=0)
        ux = rhoux / rho
        uy = rhouy / rho
        feq = equilibrium(rho, ux, uy)
        fstar = f - self.omega * (f - feq)
        fnew = stream_periodic(fstar)
        return apply_couette_bc(fnew, self.U_wall)

    def residual(self, f):
        return f - self.lbe_step(f)

    def res_norm(self, f):
        R = self.residual(f)
        return float(np.sqrt((R * R).mean()))

    def initial_field(self):
        rho = np.ones((self.N, self.N), dtype=np.float64)
        ux = np.zeros((self.N, self.N), dtype=np.float64)
        uy = np.zeros((self.N, self.N), dtype=np.float64)
        ux[-1, :] = self.U_wall  # seed top row
        return equilibrium(rho, ux, uy)

    def macro(self, f):
        rho = f.sum(axis=0)
        ux = (f * CX[:, None, None]).sum(axis=0) / rho
        uy = (f * CY[:, None, None]).sum(axis=0) / rho
        return rho, ux, uy

    def analytical_ux(self):
        y = np.arange(self.N, dtype=np.float64)
        return (self.U_wall * y / (self.N - 1)).reshape(self.N, 1) * np.ones((self.N, self.N))

    def project(self, f):
        rho = f.sum(axis=0)
        rhoux = (f * CX[:, None, None]).sum(axis=0)
        rhouy = (f * CY[:, None, None]).sum(axis=0)
        return np.stack([rho, rhoux, rhouy], axis=0)

    def lift(self, dU):
        drho, drhoux, drhouy = dU[0], dU[1], dU[2]
        df = np.empty((9,) + drho.shape, dtype=np.float64)
        for i in range(9):
            df[i] = W[i] * (drho + 3.0 * CX[i] * drhoux + 3.0 * CY[i] * drhouy)
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
