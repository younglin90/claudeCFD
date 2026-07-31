"""Phase-2 testbed : 2D channel flow.

Geometry :
    Periodic in x.
    No-slip walls at j=0 (bottom) and j=N-1 (top) via full-way bounce-back.
    Constant body force F_x.

Analytical steady (incompressible NS Poiseuille) :
    u_x^*(y) = (F0 / (2 nu)) * y * (L - y),    L = N - 1
    u_y^*    = 0,    rho^* = 1
"""

import numpy as np

from lbm_periodic import (
    CX, CY, W, CX_INT, CY_INT,
    equilibrium, stream_periodic, guo_source,
)


def apply_channel_bc(f):
    """Full-way bounce-back on top and bottom rows.
    j=0 (bottom) : incoming pop indices 2, 5, 6
    j=N-1 (top) : incoming pop indices 4, 7, 8
    """
    # Bottom wall j=0
    f[2, 0, :] = f[4, 0, :]
    f[5, 0, :] = f[7, 0, :]
    f[6, 0, :] = f[8, 0, :]
    # Top wall j=N-1
    f[4, -1, :] = f[2, -1, :]
    f[7, -1, :] = f[5, -1, :]
    f[8, -1, :] = f[6, -1, :]
    return f


class ChannelCase:
    """Force-driven 2D channel : periodic-x, no-slip walls in y."""

    def __init__(self, N, nu, F0=1e-5):
        self.N = N
        self.nu = nu
        self.omega = 1.0 / (3.0 * nu + 0.5)
        self.F0 = F0
        self.shape = (9, N, N)
        self.dof = 9 * N * N
        self.macro_dof = 3 * N * N

        self.Fx = F0 * np.ones((N, N), dtype=np.float64)
        self.Fy = np.zeros((N, N), dtype=np.float64)

        L = N - 1.0
        self.L_eff = L
        self.U_max = F0 * L * L / (8.0 * nu)  # peak Poiseuille
        self.Re = self.U_max * L / nu

    def lbe_step(self, f):
        rho = f.sum(axis=0)
        rhoux = (f * CX[:, None, None]).sum(axis=0)
        rhouy = (f * CY[:, None, None]).sum(axis=0)
        ux = rhoux / rho
        uy = rhouy / rho
        ux_eq = ux + 0.5 * self.Fx / rho
        uy_eq = uy + 0.5 * self.Fy / rho
        feq = equilibrium(rho, ux_eq, uy_eq)
        S = guo_source(rho, ux, uy, self.Fx, self.Fy, self.omega)
        fstar = f - self.omega * (f - feq) + S
        fnew = stream_periodic(fstar)
        return apply_channel_bc(fnew)

    def residual(self, f):
        return f - self.lbe_step(f)

    def res_norm(self, f):
        R = self.residual(f)
        return float(np.sqrt((R * R).mean()))

    def initial_field(self):
        rho = np.ones((self.N, self.N), dtype=np.float64)
        ux = np.zeros((self.N, self.N), dtype=np.float64)
        uy = np.zeros((self.N, self.N), dtype=np.float64)
        return equilibrium(rho, ux, uy)

    def macro(self, f):
        rho = f.sum(axis=0)
        ux = (f * CX[:, None, None]).sum(axis=0) / rho
        uy = (f * CY[:, None, None]).sum(axis=0) / rho
        return rho, ux, uy

    def analytical_ux(self):
        y = np.arange(self.N, dtype=np.float64)
        L = self.L_eff
        prof = self.F0 / (2.0 * self.nu) * y * (L - y)
        return prof.reshape(self.N, 1) * np.ones((self.N, self.N))

    # ---- projection / lift / JVP (identical to KolmogorovCase) ----
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
