"""Force-free LBM case primitives with inlet/outlet BC.

This module is intentionally independent from forcing-based implementations.
All classes expose the same basic interface used by the existing solvers:
`lbe_step`, `residual`, `initial_field`, `macro`, `project`, `lift`, `jvp`.
"""

from __future__ import annotations

import numpy as np

from lbm_periodic import CX, CY, W, CX_INT, CY_INT, equilibrium


OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
FLUX_MATCHED_OUTLET_RELAXATION = 0.05


def _fast_norm(x: np.ndarray) -> float:
    r = x.ravel()
    return float(np.sqrt(float(r @ r)))


def _stream_periodic_open_x(f: np.ndarray) -> np.ndarray:
    fn = np.empty_like(f)
    for i in range(9):
        fn[i] = np.roll(np.roll(f[i], CY_INT[i], axis=0), CX_INT[i], axis=1)
    return fn


def _stream_with_mask_open_x(fstar: np.ndarray, chi: np.ndarray) -> np.ndarray:
    fn = _stream_periodic_open_x(fstar)
    for i in range(1, 9):
        chi_src = np.roll(np.roll(chi, CY_INT[i], axis=0), CX_INT[i], axis=1)
        src_fluid = (chi == 1.0) & (chi_src == 1.0)
        fn[i] = np.where(src_fluid, fn[i], fstar[OPP[i], :, :]) * chi
    return fn


def _apply_zou_he_inlet_left(f: np.ndarray, ux_in: float | np.ndarray, uy_in: float = 0.0, chi: np.ndarray | None = None):
    """Apply Zou-He velocity BC on left boundary x=0."""
    f0 = f[:, :, 0].copy()
    ny = f.shape[1]
    if np.isscalar(ux_in):
        uxs = np.full(ny, float(ux_in), dtype=np.float64)
    else:
        uxs = np.asarray(ux_in, dtype=np.float64)
        if uxs.ndim != 1:
            uxs = uxs.reshape(-1)
    if uxs.size != ny:
        uxs = np.full(ny, float(ux_in if np.isscalar(ux_in) else uxs.flat[0]), dtype=np.float64)

    if np.isscalar(uy_in):
        uys = np.full(ny, float(uy_in), dtype=np.float64)
    else:
        uys = np.asarray(uy_in, dtype=np.float64)
        if uys.ndim != 1:
            uys = uys.reshape(-1)
    if uys.size != ny:
        uys = np.full(ny, float(uy_in if np.isscalar(uy_in) else uys.flat[0]), dtype=np.float64)

    f0c = f0[0, :]
    f2c = f0[2, :]
    f3c = f0[3, :]
    f4c = f0[4, :]
    f6c = f0[6, :]
    f7c = f0[7, :]

    if chi is None:
        mask = np.ones(ny, dtype=bool)
    else:
        mask = chi[:, 0] > 0.0
    mask = mask & (np.abs(1.0 - uxs) > 1.0e-12)
    rho0 = np.zeros(ny, dtype=np.float64)
    rho0[mask] = (f0c[mask] + f2c[mask] + f4c[mask] + 2.0 * (f3c[mask] + f6c[mask] + f7c[mask])) / (1.0 - uxs[mask])

    f1 = f0[1, :]
    f5 = f0[5, :]
    f8 = f0[8, :]
    f1[mask] = f3c[mask] + (2.0 / 3.0) * rho0[mask] * uxs[mask]
    f5[mask] = f7c[mask] + 0.5 * (f4c[mask] - f2c[mask]) + (1.0 / 6.0) * rho0[mask] * uxs[mask] + 0.5 * rho0[mask] * uys[mask]
    f8[mask] = f6c[mask] + 0.5 * (f2c[mask] - f4c[mask]) + (1.0 / 6.0) * rho0[mask] * uxs[mask] - 0.5 * rho0[mask] * uys[mask]
    f[1, :, 0] = f1
    f[5, :, 0] = f5
    f[8, :, 0] = f8
    return f


def _apply_extrap_outlet_right(f: np.ndarray, chi: np.ndarray | None = None):
    """Zero-gradient outlet at x=N-1."""
    if f.shape[2] < 2:
        return f
    if chi is None:
        f[3, :, -1] = f[3, :, -2]
        f[6, :, -1] = f[6, :, -2]
        f[7, :, -1] = f[7, :, -2]
        return f
    outlet = chi[:, -1] > 0.0
    interior = chi[:, -2] > 0.0
    active = outlet & interior
    f[3, active, -1] = f[3, active, -2]
    f[6, active, -1] = f[6, active, -2]
    f[7, active, -1] = f[7, active, -2]
    return f


def _apply_flux_matched_outlet_right(f: np.ndarray, chi: np.ndarray | None = None):
    """Match integrated right-outlet mass flux to the left inlet flux."""
    if f.shape[2] < 1:
        return f
    ny = f.shape[1]
    if chi is None:
        inlet = np.ones(ny, dtype=bool)
        outlet = np.ones(ny, dtype=bool)
    else:
        inlet = chi[:, 0] > 0.0
        outlet = chi[:, -1] > 0.0
    if not np.any(inlet) or not np.any(outlet):
        return f

    rhoux_l = f[1, :, 0] + f[5, :, 0] + f[8, :, 0] - f[3, :, 0] - f[6, :, 0] - f[7, :, 0]
    rhoux_r = f[1, :, -1] + f[5, :, -1] + f[8, :, -1] - f[3, :, -1] - f[6, :, -1] - f[7, :, -1]
    fin = float(np.sum(rhoux_l[inlet]))
    fout = float(np.sum(rhoux_r[outlet]))
    diff = fout - fin
    if not np.isfinite(diff):
        return f
    n_active = int(np.sum(outlet))
    if n_active <= 0:
        return f
    corr = FLUX_MATCHED_OUTLET_RELAXATION * diff / float(n_active)
    f[3, outlet, -1] += (2.0 / 3.0) * corr
    f[6, outlet, -1] += (1.0 / 6.0) * corr
    f[7, outlet, -1] += (1.0 / 6.0) * corr
    return f


def _has_bounded_y_walls(chi: np.ndarray | None) -> bool:
    if chi is None or chi.ndim != 2:
        return False
    return not np.any(chi[0, :] > 0.0) and not np.any(chi[-1, :] > 0.0)


def _apply_extrap_outlet_top(f: np.ndarray, chi: np.ndarray | None = None):
    """Zero-gradient outlet at y=Ny-1."""
    if f.shape[1] < 2:
        return f
    if chi is None:
        f[4, -1, :] = f[4, -2, :]
        f[7, -1, :] = f[7, -2, :]
        f[8, -1, :] = f[8, -2, :]
        return f
    outlet = chi[-1, :] > 0.0
    interior = chi[-2, :] > 0.0
    active = outlet & interior
    f[4, -1, active] = f[4, -2, active]
    f[7, -1, active] = f[7, -2, active]
    f[8, -1, active] = f[8, -2, active]
    return f


def _apply_pressure_outlet_right(f: np.ndarray, rho_out: float = 1.0, uy_out: float = 0.0, chi: np.ndarray | None = None):
    """Zou-He pressure outlet on the right boundary."""
    if f.shape[2] < 1:
        return f
    ny = f.shape[1]
    if chi is None:
        mask = np.ones(ny, dtype=bool)
    else:
        mask = chi[:, -1] > 0.0
    if not np.any(mask):
        return f
    rho = float(rho_out)
    uy = float(uy_out)
    f0 = f[0, :, -1]
    f1 = f[1, :, -1]
    f2 = f[2, :, -1]
    f4 = f[4, :, -1]
    f5 = f[5, :, -1]
    f8 = f[8, :, -1]
    ux = np.zeros(ny, dtype=np.float64)
    ux[mask] = -1.0 + (f0[mask] + f2[mask] + f4[mask] + 2.0 * (f1[mask] + f5[mask] + f8[mask])) / rho
    f[3, mask, -1] = f1[mask] - (2.0 / 3.0) * rho * ux[mask]
    f[7, mask, -1] = f5[mask] + 0.5 * (f2[mask] - f4[mask]) - (1.0 / 6.0) * rho * ux[mask] - 0.5 * rho * uy
    f[6, mask, -1] = f8[mask] + 0.5 * (f4[mask] - f2[mask]) - (1.0 / 6.0) * rho * ux[mask] + 0.5 * rho * uy
    return f


def _apply_pressure_outlet_top(f: np.ndarray, rho_out: float = 1.0, ux_out: float = 0.0, chi: np.ndarray | None = None):
    """Zou-He pressure outlet on the top boundary."""
    if f.shape[1] < 1:
        return f
    nx = f.shape[2]
    if chi is None:
        mask = np.ones(nx, dtype=bool)
    else:
        mask = chi[-1, :] > 0.0
    if not np.any(mask):
        return f
    rho = float(rho_out)
    ux = float(ux_out)
    f0 = f[0, -1, :]
    f1 = f[1, -1, :]
    f2 = f[2, -1, :]
    f3 = f[3, -1, :]
    f5 = f[5, -1, :]
    f6 = f[6, -1, :]
    uy = np.zeros(nx, dtype=np.float64)
    uy[mask] = -1.0 + (f0[mask] + f1[mask] + f3[mask] + 2.0 * (f2[mask] + f5[mask] + f6[mask])) / rho
    f[4, -1, mask] = f2[mask] - (2.0 / 3.0) * rho * uy[mask]
    f[7, -1, mask] = f5[mask] + 0.5 * (f1[mask] - f3[mask]) - (1.0 / 6.0) * rho * uy[mask] - 0.5 * rho * ux
    f[8, -1, mask] = f6[mask] + 0.5 * (f3[mask] - f1[mask]) - (1.0 / 6.0) * rho * uy[mask] + 0.5 * rho * ux
    return f


def _stream_with_mask_bounce(fstar: np.ndarray, chi: np.ndarray) -> np.ndarray:
    """Nonperiodic masked streaming with bounce-back at solid/out-of-domain faces."""
    fn = np.empty_like(fstar)
    _stream_with_mask_bounce_into(fstar, chi, fn)
    return fn


def _stream_with_mask_bounce_into(fstar: np.ndarray, chi: np.ndarray, fn: np.ndarray, fluid: np.ndarray | None = None) -> np.ndarray:
    """Nonperiodic masked streaming into a caller-owned output buffer."""
    q, ny, nx = fstar.shape
    if fluid is None:
        fluid = chi > 0.0
    for i in range(q):
        cy = int(CY_INT[i])
        cx = int(CX_INT[i])
        opp = OPP[i]
        y_dst0 = max(0, cy)
        y_dst1 = ny + min(0, cy)
        x_dst0 = max(0, cx)
        x_dst1 = nx + min(0, cx)
        y_src0 = y_dst0 - cy
        y_src1 = y_dst1 - cy
        x_src0 = x_dst0 - cx
        x_src1 = x_dst1 - cx
        fn[i] = fstar[opp] * chi
        if y_dst0 < y_dst1 and x_dst0 < x_dst1:
            src_fluid = fluid[y_src0:y_src1, x_src0:x_src1]
            dst_fluid = fluid[y_dst0:y_dst1, x_dst0:x_dst1]
            valid = src_fluid & dst_fluid
            streamed = fstar[i, y_src0:y_src1, x_src0:x_src1]
            bounced = fstar[opp, y_dst0:y_dst1, x_dst0:x_dst1]
            fn[i, y_dst0:y_dst1, x_dst0:x_dst1] = np.where(valid, streamed, bounced) * chi[y_dst0:y_dst1, x_dst0:x_dst1]
    return fn


def _equilibrium_into(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray, out: np.ndarray) -> np.ndarray:
    u2 = 1.5 * (ux * ux + uy * uy)
    for i in range(9):
        cu = 3.0 * (CX[i] * ux + CY[i] * uy)
        out[i] = W[i] * rho * (1.0 + cu + 0.5 * cu * cu - u2)
    return out


def _apply_channel_wall_bc(f: np.ndarray):
    """Full-way bounce-back on top/bottom walls."""
    f[2, 0, :] = f[4, 0, :]
    f[5, 0, :] = f[7, 0, :]
    f[6, 0, :] = f[8, 0, :]
    f[4, -1, :] = f[2, -1, :]
    f[7, -1, :] = f[5, -1, :]
    f[8, -1, :] = f[6, -1, :]
    return f


def _poiseuille_profile(ny: int, u_mean: float):
    y = np.arange(ny, dtype=np.float64)
    if ny <= 1:
        return np.zeros(ny, dtype=np.float64)
    eta = y / float(ny - 1)
    ubar = float(u_mean) * (ny - 1.0) / ny
    return 6.0 * ubar * eta * (1.0 - eta)


class NoForceChannelCase:
    """Force-free channel with periodic x-direction and wall boundaries at y."""

    def __init__(
        self,
        N: int,
        nu: float,
        U_in: float = 0.05,
        x_bc: str = "periodic",
        initial_profile: str = "constant",
    ):
        self.N = int(N)
        self.nu = float(nu)
        self.U_in = float(U_in)
        self.x_bc = str(x_bc).lower()
        if self.x_bc not in {"periodic", "inlet_outlet"}:
            raise ValueError(f"unsupported x_bc: {x_bc}")
        self.initial_profile = str(initial_profile).lower()
        if self.initial_profile not in {"constant", "poiseuille", "zero"}:
            raise ValueError(f"unsupported initial_profile: {initial_profile}")
        self.omega = 1.0 / (3.0 * self.nu + 0.5)
        self.shape = (9, self.N, self.N)
        self.dof = 9 * self.N * self.N
        self.macro_dof = 3 * self.N * self.N
        self.chi = np.ones((self.N, self.N), dtype=np.float64)
        self.Fx = np.zeros((self.N, self.N), dtype=np.float64)
        self.Fy = np.zeros((self.N, self.N), dtype=np.float64)
        self.fluid_fraction = 1.0
        self.reference_kind = self._reference_kind()

    def _reference_kind(self) -> str:
        has_force = bool(np.any(np.abs(self.Fx) > 0.0) or np.any(np.abs(self.Fy) > 0.0))
        if self.x_bc == "periodic" and (not has_force):
            return "zero_flow"
        if self.x_bc == "inlet_outlet":
            return "inlet_outlet"
        return "unknown"

    def _initial_profile(self):
        y = np.arange(self.N, dtype=np.float64)
        if self.initial_profile == "poiseuille":
            if self.N <= 1:
                return np.zeros(self.N, dtype=np.float64)
            L = float(self.N - 1)
            # Average velocity is close to U_in for consistency with legacy convention.
            # ux_max (centerline) is 1.5 * Ubar.
            ubar = self.U_in * (self.N - 1.0) / self.N
            return 6.0 * ubar * (y / L) * (1.0 - y / L)
        if self.initial_profile == "zero":
            return np.zeros(self.N, dtype=np.float64)
        # default "constant" (legacy behavior with no-slip walls at top/bottom)
        ux = np.full(self.N, self.U_in, dtype=np.float64)
        ux[0] = 0.0
        ux[-1] = 0.0
        return ux

    def lbe_step(self, f: np.ndarray) -> np.ndarray:
        rho = f.sum(axis=0)
        rho_safe = np.where(rho < 1.0e-12, 1.0, rho)
        ux = (f * CX[:, None, None]).sum(axis=0) / rho_safe
        uy = (f * CY[:, None, None]).sum(axis=0) / rho_safe
        feq = equilibrium(rho, ux, uy)
        fstar = f - self.omega * (f - feq)
        fnew = _stream_periodic_open_x(fstar)
        if self.x_bc == "inlet_outlet":
            fnew = _apply_zou_he_inlet_left(fnew, self._initial_profile())
            fnew = _apply_extrap_outlet_right(fnew)
        fnew = _apply_channel_wall_bc(fnew)
        return fnew

    def residual(self, f: np.ndarray) -> np.ndarray:
        return f - self.lbe_step(f)

    def res_norm(self, f: np.ndarray) -> float:
        R = self.residual(f)
        return float(np.sqrt((R * R).mean()))

    def initial_field(self) -> np.ndarray:
        rho = np.ones((self.N, self.N), dtype=np.float64)
        ux = np.tile(self._initial_profile()[:, None], (1, self.N))
        uy = np.zeros((self.N, self.N), dtype=np.float64)
        return equilibrium(rho, ux, uy)

    def macro(self, f: np.ndarray):
        rho = f.sum(axis=0)
        rho_safe = np.where(rho < 1.0e-12, 1.0, rho)
        ux = (f * CX[:, None, None]).sum(axis=0) / rho_safe
        uy = (f * CY[:, None, None]).sum(axis=0) / rho_safe
        return rho, ux, uy

    def project(self, f: np.ndarray):
        rho = f.sum(axis=0)
        rhoux = (f * CX[:, None, None]).sum(axis=0)
        rhouy = (f * CY[:, None, None]).sum(axis=0)
        return np.stack([rho, rhoux, rhouy], axis=0)

    def lift(self, dU: np.ndarray):
        drho, drhoux, drhouy = dU[0], dU[1], dU[2]
        df = np.empty((9,) + drho.shape, dtype=np.float64)
        for i in range(9):
            df[i] = W[i] * (drho + 3.0 * CX[i] * drhoux + 3.0 * CY[i] * drhouy)
        return df

    def _fast_norm(self, x: np.ndarray) -> float:
        return _fast_norm(x)

    def jvp(self, w: np.ndarray, f_base: np.ndarray, R_base: np.ndarray, norm_f_cached: float | None = None):
        if norm_f_cached is None:
            norm_f_cached = self._fast_norm(f_base)
        norm_w = self._fast_norm(w)
        if norm_w < 1.0e-30:
            return np.zeros_like(R_base)
        eps = 1.0e-7 * (norm_f_cached + 1.0) / norm_w
        R_pert = self.residual(f_base + eps * w)
        return (R_pert - R_base) / eps


class NoForcePoiseuilleRectCase:
    """Force-free rectangular Poiseuille channel with velocity inlet/outlet."""

    def __init__(
        self,
        Ny: int,
        Nx: int,
        nu: float,
        U_in: float = 0.05,
        initial_profile: str = "poiseuille",
        outlet_bc: str = "pressure",
    ):
        self.Ny = int(Ny)
        self.Nx = int(Nx)
        self.N = max(self.Ny, self.Nx)
        self.nu = float(nu)
        self.U_in = float(U_in)
        self.initial_profile = str(initial_profile).lower()
        self.x_bc = "inlet_outlet"
        self.omega = 1.0 / (3.0 * self.nu + 0.5)
        self.shape = (9, self.Ny, self.Nx)
        self.dof = 9 * self.Ny * self.Nx
        self.macro_dof = 3 * self.Ny * self.Nx
        self.chi = np.ones((self.Ny, self.Nx), dtype=np.float64)
        self.Fx = np.zeros((self.Ny, self.Nx), dtype=np.float64)
        self.Fy = np.zeros((self.Ny, self.Nx), dtype=np.float64)
        self.fluid_fraction = 1.0
        self.reference_kind = "inlet_outlet"
        self.outlet_bc = str(outlet_bc).lower()
        if self.outlet_bc not in {"extrap", "pressure"}:
            raise ValueError(f"unsupported outlet_bc: {outlet_bc}")

    def _initial_profile(self):
        if self.initial_profile == "zero":
            return np.zeros(self.Ny, dtype=np.float64)
        if self.initial_profile == "constant":
            ux = np.full(self.Ny, self.U_in, dtype=np.float64)
            ux[0] = 0.0
            ux[-1] = 0.0
            return ux
        return _poiseuille_profile(self.Ny, self.U_in)

    def lbe_step(self, f: np.ndarray) -> np.ndarray:
        rho = f.sum(axis=0)
        rho_safe = np.where(rho < 1.0e-12, 1.0, rho)
        ux = (f * CX[:, None, None]).sum(axis=0) / rho_safe
        uy = (f * CY[:, None, None]).sum(axis=0) / rho_safe
        feq = equilibrium(rho, ux, uy)
        fstar = f - self.omega * (f - feq)
        fnew = _stream_with_mask_bounce(fstar, self.chi)
        fnew = _apply_zou_he_inlet_left(fnew, self._initial_profile())
        if self.outlet_bc == "pressure":
            fnew = _apply_pressure_outlet_right(fnew, rho_out=1.0, uy_out=0.0)
        else:
            fnew = _apply_extrap_outlet_right(fnew)
        fnew = _apply_channel_wall_bc(fnew)
        return fnew

    def residual(self, f: np.ndarray) -> np.ndarray:
        return f - self.lbe_step(f)

    def res_norm(self, f: np.ndarray) -> float:
        return float(np.sqrt(np.mean(self.residual(f) ** 2)))

    def initial_field(self) -> np.ndarray:
        rho = np.ones((self.Ny, self.Nx), dtype=np.float64)
        ux = np.tile(self._initial_profile()[:, None], (1, self.Nx))
        uy = np.zeros_like(ux)
        return equilibrium(rho, ux, uy)

    def analytical_ux(self):
        return np.tile(_poiseuille_profile(self.Ny, self.U_in)[:, None], (1, self.Nx))

    def macro(self, f: np.ndarray):
        rho = f.sum(axis=0)
        rho_safe = np.where(rho < 1.0e-12, 1.0, rho)
        ux = (f * CX[:, None, None]).sum(axis=0) / rho_safe
        uy = (f * CY[:, None, None]).sum(axis=0) / rho_safe
        return rho, ux, uy

    def project(self, f: np.ndarray):
        rho = f.sum(axis=0)
        rhoux = (f * CX[:, None, None]).sum(axis=0)
        rhouy = (f * CY[:, None, None]).sum(axis=0)
        return np.stack([rho, rhoux, rhouy], axis=0)

    def lift(self, dU: np.ndarray):
        drho, drhoux, drhouy = dU[0], dU[1], dU[2]
        df = np.empty((9,) + drho.shape, dtype=np.float64)
        for i in range(9):
            df[i] = W[i] * (drho + 3.0 * CX[i] * drhoux + 3.0 * CY[i] * drhouy)
        return df

    def _fast_norm(self, x: np.ndarray) -> float:
        return _fast_norm(x)

    def jvp(self, w: np.ndarray, f_base: np.ndarray, R_base: np.ndarray, norm_f_cached: float | None = None):
        if norm_f_cached is None:
            norm_f_cached = self._fast_norm(f_base)
        norm_w = self._fast_norm(w)
        if norm_w < 1.0e-30:
            return np.zeros_like(R_base)
        eps = 1.0e-7 * (norm_f_cached + 1.0) / norm_w
        return (self.residual(f_base + eps * w) - R_base) / eps


class NoForceMaskedCase:
    """Force-free masked-flow case with inlet/outlet BC and full-way BB on mask."""

    def __init__(self, chi: np.ndarray, nu: float, U_in: float = 0.05):
        self.chi = chi.astype(np.float64)
        if self.chi.ndim != 2 or self.chi.shape[0] != self.chi.shape[1]:
            raise ValueError("chi must be square 2D mask")
        self.N = int(self.chi.shape[0])
        self.nu = float(nu)
        self.U_in = float(U_in)
        self.omega = 1.0 / (3.0 * self.nu + 0.5)
        self.shape = (9, self.N, self.N)
        self.dof = 9 * self.N * self.N
        self.macro_dof = 3 * self.N * self.N
        self.Fx = np.zeros((self.N, self.N), dtype=np.float64)
        self.Fy = np.zeros((self.N, self.N), dtype=np.float64)
        self.fluid_fraction = float(self.chi.mean())

    def _inlet_profile(self):
        ux = np.full(self.N, self.U_in, dtype=np.float64)
        return ux

    def lbe_step(self, f: np.ndarray) -> np.ndarray:
        rho = f.sum(axis=0)
        rho_safe = np.where(rho < 1.0e-12, 1.0, rho)
        ux = (f * CX[:, None, None]).sum(axis=0) / rho_safe
        uy = (f * CY[:, None, None]).sum(axis=0) / rho_safe
        feq = equilibrium(rho, ux, uy)
        fstar = f - self.omega * (f - feq)

        fnew = _stream_with_mask_open_x(fstar, self.chi)
        fnew = _apply_zou_he_inlet_left(fnew, self._inlet_profile(), chi=self.chi)
        fnew = _apply_extrap_outlet_right(fnew, chi=self.chi)
        if _has_bounded_y_walls(self.chi):
            fnew = _apply_flux_matched_outlet_right(fnew, chi=self.chi)
        return fnew * self.chi[None, :, :]

    def residual(self, f: np.ndarray) -> np.ndarray:
        return f - self.lbe_step(f)

    def res_norm(self, f: np.ndarray) -> float:
        r = self.residual(f)
        fluid = self.chi > 0.0
        return float(np.sqrt(np.mean(r[:, fluid] * r[:, fluid])))

    def initial_field(self) -> np.ndarray:
        rho = np.ones((self.N, self.N), dtype=np.float64)
        ux = np.tile(self._inlet_profile()[:, None], (1, self.N))
        uy = np.zeros((self.N, self.N), dtype=np.float64)
        f0 = equilibrium(rho, ux, uy)
        return f0 * self.chi[None, :, :]

    def macro(self, f: np.ndarray):
        rho = f.sum(axis=0)
        rho_safe = np.where(rho < 1.0e-12, 1.0, rho)
        ux = (f * CX[:, None, None]).sum(axis=0) / rho_safe
        uy = (f * CY[:, None, None]).sum(axis=0) / rho_safe
        return rho, ux * self.chi, uy * self.chi

    def project(self, f: np.ndarray):
        rho = f.sum(axis=0)
        rhoux = (f * CX[:, None, None]).sum(axis=0)
        rhouy = (f * CY[:, None, None]).sum(axis=0)
        return np.stack([rho, rhoux, rhouy], axis=0)

    def lift(self, dU: np.ndarray):
        drho, drhoux, drhouy = dU[0], dU[1], dU[2]
        df = np.empty((9,) + drho.shape, dtype=np.float64)
        for i in range(9):
            df[i] = W[i] * (drho + 3.0 * CX[i] * drhoux + 3.0 * CY[i] * drhouy)
        return df * self.chi[None, :, :]

    def _fast_norm(self, x: np.ndarray) -> float:
        return _fast_norm(x)

    def jvp(self, w: np.ndarray, f_base: np.ndarray, R_base: np.ndarray, norm_f_cached: float | None = None):
        if norm_f_cached is None:
            norm_f_cached = self._fast_norm(f_base)
        norm_w = self._fast_norm(w)
        if norm_w < 1.0e-30:
            return np.zeros_like(R_base)
        eps = 1.0e-7 * (norm_f_cached + 1.0) / norm_w
        R_pert = self.residual(f_base + eps * w)
        return (R_pert - R_base) / eps


class NoForceTJunctionRectCase:
    """Rectangular T-junction with left inlet, right/top outlets, and wall mask."""

    def __init__(self, chi: np.ndarray, nu: float, U_in: float = 0.04, outlet_bc: str = "extrap"):
        self.chi = chi.astype(np.float64)
        if self.chi.ndim != 2:
            raise ValueError("chi must be 2D")
        self.Ny, self.Nx = map(int, self.chi.shape)
        self.N = max(self.Ny, self.Nx)
        self.nu = float(nu)
        self.U_in = float(U_in)
        self.omega = 1.0 / (3.0 * self.nu + 0.5)
        self.shape = (9, self.Ny, self.Nx)
        self.dof = 9 * self.Ny * self.Nx
        self.macro_dof = 3 * self.Ny * self.Nx
        self.Fx = np.zeros((self.Ny, self.Nx), dtype=np.float64)
        self.Fy = np.zeros((self.Ny, self.Nx), dtype=np.float64)
        self.fluid_fraction = float(self.chi.mean())
        self.reference_kind = "tight_ref"
        self.outlet_bc = str(outlet_bc).lower()
        if self.outlet_bc not in {"extrap", "pressure"}:
            raise ValueError(f"unsupported outlet_bc: {outlet_bc}")
        self.fluid = self.chi > 0.0
        self.inlet_profile = self._build_inlet_profile()
        self._work = None

    def _build_inlet_profile(self):
        ux = np.zeros(self.Ny, dtype=np.float64)
        inlet = self.chi[:, 0] > 0.0
        idx = np.where(inlet)[0]
        if idx.size == 0:
            return ux
        local = np.linspace(0.0, 1.0, idx.size, dtype=np.float64)
        prof = 6.0 * self.U_in * local * (1.0 - local)
        ux[idx] = prof
        return ux

    def _inlet_profile(self):
        return self.inlet_profile

    def _buffers(self):
        if self._work is None:
            self._work = {
                "rho": np.empty((self.Ny, self.Nx), dtype=np.float64),
                "rho_safe": np.empty((self.Ny, self.Nx), dtype=np.float64),
                "ux": np.empty((self.Ny, self.Nx), dtype=np.float64),
                "uy": np.empty((self.Ny, self.Nx), dtype=np.float64),
                "feq": np.empty(self.shape, dtype=np.float64),
                "fstar": np.empty(self.shape, dtype=np.float64),
            }
        return self._work

    def lbe_step_inplace(self, f: np.ndarray, out: np.ndarray) -> np.ndarray:
        work = self._buffers()
        rho = work["rho"]
        rho_safe = work["rho_safe"]
        ux = work["ux"]
        uy = work["uy"]
        feq = work["feq"]
        fstar = work["fstar"]
        np.sum(f, axis=0, out=rho)
        np.maximum(rho, 1.0e-12, out=rho_safe)
        np.add(f[1], f[5], out=ux)
        ux += f[8]
        ux -= f[3]
        ux -= f[6]
        ux -= f[7]
        ux /= rho_safe
        np.add(f[2], f[5], out=uy)
        uy += f[6]
        uy -= f[4]
        uy -= f[7]
        uy -= f[8]
        uy /= rho_safe
        _equilibrium_into(rho, ux, uy, feq)
        np.subtract(f, feq, out=fstar)
        fstar *= self.omega
        np.subtract(f, fstar, out=fstar)
        _stream_with_mask_bounce_into(fstar, self.chi, out, fluid=self.fluid)
        _apply_zou_he_inlet_left(out, self.inlet_profile, chi=self.chi)
        if self.outlet_bc == "pressure":
            _apply_pressure_outlet_right(out, rho_out=1.0, uy_out=0.0, chi=self.chi)
            _apply_pressure_outlet_top(out, rho_out=1.0, ux_out=0.0, chi=self.chi)
        else:
            _apply_extrap_outlet_right(out, chi=self.chi)
            _apply_extrap_outlet_top(out, chi=self.chi)
        out *= self.chi[None, :, :]
        return out

    def lbe_step(self, f: np.ndarray) -> np.ndarray:
        return self.lbe_step_inplace(f, np.empty_like(f))

    def residual(self, f: np.ndarray) -> np.ndarray:
        return f - self.lbe_step(f)

    def res_norm(self, f: np.ndarray) -> float:
        r = self.residual(f)
        return float(np.sqrt(np.mean(r[:, self.fluid] * r[:, self.fluid])))

    def initial_field(self) -> np.ndarray:
        rho = np.ones((self.Ny, self.Nx), dtype=np.float64)
        ux = np.tile(self._inlet_profile()[:, None], (1, self.Nx))
        uy = np.zeros((self.Ny, self.Nx), dtype=np.float64)
        return equilibrium(rho, ux, uy) * self.chi[None, :, :]

    def macro(self, f: np.ndarray):
        rho = f.sum(axis=0)
        rho_safe = np.where(rho < 1.0e-12, 1.0, rho)
        ux = (f * CX[:, None, None]).sum(axis=0) / rho_safe
        uy = (f * CY[:, None, None]).sum(axis=0) / rho_safe
        return rho, ux * self.chi, uy * self.chi

    def project(self, f: np.ndarray):
        rho = f.sum(axis=0)
        rhoux = (f * CX[:, None, None]).sum(axis=0)
        rhouy = (f * CY[:, None, None]).sum(axis=0)
        return np.stack([rho, rhoux, rhouy], axis=0)

    def lift(self, dU: np.ndarray):
        drho, drhoux, drhouy = dU[0], dU[1], dU[2]
        df = np.empty((9,) + drho.shape, dtype=np.float64)
        for i in range(9):
            df[i] = W[i] * (drho + 3.0 * CX[i] * drhoux + 3.0 * CY[i] * drhouy)
        return df * self.chi[None, :, :]

    def _fast_norm(self, x: np.ndarray) -> float:
        return _fast_norm(x)

    def jvp(self, w: np.ndarray, f_base: np.ndarray, R_base: np.ndarray, norm_f_cached: float | None = None):
        if norm_f_cached is None:
            norm_f_cached = self._fast_norm(f_base)
        norm_w = self._fast_norm(w)
        if norm_w < 1.0e-30:
            return np.zeros_like(R_base)
        eps = 1.0e-7 * (norm_f_cached + 1.0) / norm_w
        return (self.residual(f_base + eps * w) - R_base) / eps
