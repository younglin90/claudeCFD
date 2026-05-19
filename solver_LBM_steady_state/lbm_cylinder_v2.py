"""Proper cylinder flow benchmark with Zou-He inflow + extrapolation outflow.

Setup :
    Inflow (x=0)   : Zou-He velocity BC, u = (U_in, 0), parabolic optional
    Outflow (x=L)  : zero-gradient extrapolation
    Top/bottom     : free-slip (periodic-y or zero-gradient) ; here periodic-y
    Cylinder       : full-way bounce-back (voxel mask)

Validation : drag coefficient Cd at Re = U_in · D / nu, comparison with Henderson 1995 :
    Re=20 : Cd ≈ 2.05
    Re=40 : Cd ≈ 1.54
"""

import numpy as np

from lbm_periodic import CX, CY, W, CX_INT, CY_INT, equilibrium, stream_periodic
from lbm_voxel import build_cylinder_mask


# opposite pop index (for D2Q9 bounce-back)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])


def apply_zou_he_inflow(f, U_in):
    """Zou-He velocity BC at x=0 (leftmost column). Sets f[1, :, 0], f[5, :, 0], f[8, :, 0]
    from neighboring populations and prescribed u = (U_in, 0).

    From Zou-He 1997 derivation for D2Q9 left wall :
        rho_0 = (f[0]+f[2]+f[4] + 2*(f[3]+f[6]+f[7])) / (1 - U_in)
        f[1] = f[3] + (2/3) rho_0 U_in
        f[5] = f[7] + (1/2)(f[4] - f[2]) + (1/6) rho_0 U_in
        f[8] = f[6] + (1/2)(f[2] - f[4]) + (1/6) rho_0 U_in
    """
    # column 0
    f0c = f[0, :, 0]
    f2c = f[2, :, 0]
    f3c = f[3, :, 0]
    f4c = f[4, :, 0]
    f6c = f[6, :, 0]
    f7c = f[7, :, 0]
    rho_0 = (f0c + f2c + f4c + 2.0 * (f3c + f6c + f7c)) / (1.0 - U_in)
    f[1, :, 0] = f3c + (2.0 / 3.0) * rho_0 * U_in
    f[5, :, 0] = f7c + 0.5 * (f4c - f2c) + (1.0 / 6.0) * rho_0 * U_in
    f[8, :, 0] = f6c + 0.5 * (f2c - f4c) + (1.0 / 6.0) * rho_0 * U_in
    return f


def apply_extrap_outflow(f):
    """Zero-gradient extrapolation at x=N-1.
    Copy interior column N-2 into N-1 for unknown populations.
    Unknown at right wall : f[3], f[6], f[7] (those streaming from outside).
    """
    f[3, :, -1] = f[3, :, -2]
    f[6, :, -1] = f[6, :, -2]
    f[7, :, -1] = f[7, :, -2]
    return f


def stream_with_cylinder_bb(f, chi):
    """Periodic-y streaming, then bounce-back at fluid-solid links inside domain."""
    fn = np.empty_like(f)
    for i in range(9):
        # x-direction NOT periodic ; y-direction periodic
        # we do periodic shift in both then enforce x-BC after
        fn[i] = np.roll(np.roll(f[i], CY_INT[i], axis=0), CX_INT[i], axis=1)
    # bounce-back at fluid-solid links
    for i in range(1, 9):
        chi_src = np.roll(np.roll(chi, CY_INT[i], axis=0), CX_INT[i], axis=1)
        bad = (chi == 1) & (chi_src == 0)
        if bad.any():
            fn[i, bad] = f[OPP[i], bad]
    return fn * chi[None, :, :]


def lbe_step_cylinder(f, chi, omega, U_in):
    """LBE step : collision + streaming + Zou-He inflow + extrap outflow + cylinder bb."""
    rho = f.sum(axis=0)
    rho_safe = np.where(rho < 1e-12, 1.0, rho)
    rhoux = (f * CX[:, None, None]).sum(axis=0)
    rhouy = (f * CY[:, None, None]).sum(axis=0)
    ux = rhoux / rho_safe
    uy = rhouy / rho_safe
    feq = equilibrium(rho, ux, uy)
    fstar = f - omega * (f - feq)
    fnew = stream_with_cylinder_bb(fstar, chi)
    # apply velocity BC at inflow, extrapolation at outflow
    fnew = apply_zou_he_inflow(fnew, U_in)
    fnew = apply_extrap_outflow(fnew)
    return fnew


class CylinderInflowCase:
    """2D cylinder flow : Zou-He inflow + extrapolation outflow + periodic-y."""

    def __init__(self, Nx=128, Ny=64, D=12, U_in=0.05, Re=20, cx=None, cy=None):
        self.Nx = Nx
        self.Ny = Ny
        self.N = Nx  # for SCMK PC builder compatibility (uses single N)
        self.D = D
        self.U_in = U_in
        self.Re = Re
        self.nu = U_in * D / Re
        self.omega = 1.0 / (3.0 * self.nu + 0.5)
        if cx is None:
            cx = Nx // 4
        if cy is None:
            cy = Ny // 2
        self.cx = cx
        self.cy = cy
        yy, xx = np.meshgrid(np.arange(Ny), np.arange(Nx), indexing="ij")
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        self.chi = np.where(r2 < (D / 2) ** 2, 0.0, 1.0).astype(np.float64)

        self.shape = (9, Ny, Nx)
        self.dof = 9 * Ny * Nx
        self.macro_dof = 3 * Ny * Nx
        self.fluid_fraction = float(self.chi.mean())

    def lbe_step(self, f):
        return lbe_step_cylinder(f, self.chi, self.omega, self.U_in)

    def residual(self, f):
        return f - self.lbe_step(f)

    def res_norm(self, f):
        R = self.residual(f)
        return float(np.sqrt((R * R).mean()))

    def initial_field(self):
        rho = np.ones((self.Ny, self.Nx), dtype=np.float64)
        ux = np.full((self.Ny, self.Nx), self.U_in, dtype=np.float64)
        uy = np.zeros((self.Ny, self.Nx), dtype=np.float64)
        f0 = equilibrium(rho, ux, uy)
        return f0 * self.chi[None, :, :]

    def macro(self, f):
        rho = f.sum(axis=0)
        rho_safe = np.where(rho < 1e-12, 1.0, rho)
        ux = (f * CX[:, None, None]).sum(axis=0) / rho_safe
        uy = (f * CY[:, None, None]).sum(axis=0) / rho_safe
        return rho, ux * self.chi, uy * self.chi

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
        return df * self.chi[None, :, :]

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


def compute_drag_cd_momentum(case, f):
    """Drag coefficient via Ladd 1994 momentum-exchange method.

    For each fluid cell adjacent to solid in direction c_i :
        F_drag_link = c_i · (f_i(x) + f_{opp(i)}(x_b))
    where f_i is pre-collision distribution streaming toward solid,
    f_{opp(i)} is the bounced-back population. In full-way BB after streaming,
    these are 2 · f_i(x).

    Sum over all fluid-solid links gives total drag.
    Cd = 2 F_x / (rho · U_in² · D)
    """
    fx_total = 0.0
    chi = case.chi
    for i in range(1, 9):
        # Fluid cell x with solid neighbor at x + c_i :
        # chi_dest = chi shifted so position (j,k) gives chi(j+c_y, k+c_x)
        chi_dest = np.zeros_like(chi)
        chi_dest = np.roll(np.roll(chi, -CY_INT[i], axis=0), -CX_INT[i], axis=1)
        link = (chi == 1) & (chi_dest == 0)
        if link.any():
            # Momentum injected into fluid by bounce-back at this link :
            # 2 * c_i_x * f_i(x)  (Ladd 1994)
            # but f_i streams INTO solid -> bounce-back into f_{opp(i)}
            # net force on fluid in +x direction = - 2 * c_i_x * f_i
            fx_total += -2.0 * CX[i] * float(f[i][link].sum())
    # Force on cylinder = - force on fluid = - fx_total (Newton 3rd)
    F_drag = -fx_total
    rho_mean = 1.0
    Cd = 2.0 * abs(F_drag) / (rho_mean * case.U_in ** 2 * case.D + 1e-30)
    return Cd
