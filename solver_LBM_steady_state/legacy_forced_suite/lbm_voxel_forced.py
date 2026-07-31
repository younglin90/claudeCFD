"""Phase-5 testbed : periodic D2Q9 + voxel solid mask + Guo forcing.

Mimics vessel-like geometry with arbitrary interior solid voxels.
Bounce-back applied at every fluid-solid interface (full-way).

mask convention :  chi[j, k] = 1 (fluid),  chi[j, k] = 0 (solid)
"""

import numpy as np

from lbm_periodic import CX, CY, W, CX_INT, CY_INT, equilibrium, guo_source


def stream_with_mask(f, chi):
    """Periodic stream + bounce-back at fluid-solid links.

    For each direction i, if f streams from a fluid neighbor to a fluid cell : normal stream.
    If origin cell is solid : the post-collision distribution that would have streamed
    out gets bounced back at the destination via opposite-direction swap.

    Implementation : streaming then mask correction.
    """
    fn = np.empty_like(f)
    for i in range(9):
        fn[i] = np.roll(np.roll(f[i], CY_INT[i], axis=0), CX_INT[i], axis=1)

    # Bounce-back : at fluid cell x, if neighbor x - c_i is solid, then
    # f_i(x) should come from bounce-back : f_i(x) = f_opp(i)(x) (pre-stream value).
    # We need pre-stream values at fluid cells where streaming source is solid.
    # OPP map : 0->0, 1->3, 2->4, 3->1, 4->2, 5->7, 6->8, 7->5, 8->6
    OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

    # Mask of "fluid cell with solid neighbor in direction -c_i (source is solid)"
    # For each i, check if x - c_i is solid
    for i in range(1, 9):
        chi_src = np.roll(np.roll(chi, CY_INT[i], axis=0), CX_INT[i], axis=1)
        bad = (chi == 1) & (chi_src == 0)
        # at these cells, f_i should be bounce-back of f_{opp(i)} pre-stream
        # pre-stream f_{opp(i)} at the fluid cell x equals f[opp(i), x] BEFORE streaming
        # However we already streamed -- use the streamed value f[opp(i), x] which came from
        # x - c_opp(i) = x + c_i. We want the pre-stream value at x itself.
        # Trick : the value at fn[opp(i), x] post-stream came from f[opp(i), x + c_i].
        # We need pre-stream f[opp(i), x] which we lost. Solution : keep pre-stream array.
        pass

    return fn


def lbe_step_voxel(f, chi, omega, Fx, Fy):
    """LBE step on voxel mask geometry.

    Pipeline :
        (1) Compute moments, equilibrium, Guo source.
        (2) Collide : f* = f - omega(f - feq) + S         (only at fluid cells)
        (3) Stream  with bounce-back at fluid-solid links (using pre-stream f*).
        (4) Zero solid cells (cosmetic, not strictly needed).
    """
    rho = f.sum(axis=0)
    rhoux = (f * CX[:, None, None]).sum(axis=0)
    rhouy = (f * CY[:, None, None]).sum(axis=0)
    rho_safe = np.where(rho < 1e-12, 1.0, rho)
    ux = rhoux / rho_safe
    uy = rhouy / rho_safe
    ux_eq = ux + 0.5 * Fx / rho_safe
    uy_eq = uy + 0.5 * Fy / rho_safe
    feq = equilibrium(rho, ux_eq, uy_eq)
    S = guo_source(rho, ux, uy, Fx, Fy, omega)
    fstar = f - omega * (f - feq) + S

    # Stream + bounce-back  (full-way, on-grid)
    # For each direction i and fluid cell x :
    #   if neighbor x - c_i is fluid : f_i^new(x) = f*_i(x - c_i)
    #   else (solid)                  : f_i^new(x) = f*_{opp(i)}(x)
    OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
    fnew = np.empty_like(f)
    for i in range(9):
        streamed = np.roll(np.roll(fstar[i], CY_INT[i], axis=0), CX_INT[i], axis=1)
        # mask of source = fluid?
        chi_src = np.roll(np.roll(chi, CY_INT[i], axis=0), CX_INT[i], axis=1)
        src_fluid = (chi == 1) & (chi_src == 1)
        # bounce-back contribution : f*_{opp(i)}(x) (pre-stream, current cell)
        bb_val = fstar[OPP[i]]
        # use streamed where source fluid, bb_val otherwise (and only at fluid cells)
        fnew[i] = np.where(src_fluid, streamed, bb_val) * chi
    return fnew


def build_random_obstacle_mask(N, density, seed=0):
    """Random scattered obstacle voxels (vessel-like sparse occlusion).
    Returns chi shape (N, N), with `density` fraction of cells set solid (0).
    """
    rng = np.random.RandomState(seed)
    chi = np.ones((N, N), dtype=np.float64)
    n_solid = int(density * N * N)
    flat_idx = rng.choice(N * N, size=n_solid, replace=False)
    chi.flat[flat_idx] = 0.0
    return chi


def build_cylinder_mask(N, cx, cy, radius):
    """Single cylindrical obstacle inside periodic box."""
    chi = np.ones((N, N), dtype=np.float64)
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    chi[r2 < radius ** 2] = 0.0
    return chi


class VoxelCase:
    """Periodic D2Q9 + voxel mask + Guo body force."""

    def __init__(self, chi, nu, F0=1e-5, kf=0):
        self.chi = chi.astype(np.float64)
        self.N = chi.shape[0]
        self.nu = nu
        self.omega = 1.0 / (3.0 * nu + 0.5)
        self.F0 = F0
        self.kf = kf
        self.shape = (9, self.N, self.N)
        self.dof = 9 * self.N * self.N
        self.macro_dof = 3 * self.N * self.N

        # body force pattern : constant if kf=0, sinusoidal otherwise
        if kf == 0:
            self.Fx = F0 * np.ones((self.N, self.N), dtype=np.float64)
        else:
            y = np.arange(self.N, dtype=np.float64).reshape(self.N, 1)
            self.k_lat = 2.0 * np.pi * kf / self.N
            self.Fx = F0 * np.sin(self.k_lat * y) * np.ones((self.N, self.N))
        self.Fy = np.zeros((self.N, self.N), dtype=np.float64)
        # zero force at solid cells
        self.Fx *= self.chi
        self.Fy *= self.chi

        self.fluid_fraction = float(self.chi.mean())

    def lbe_step(self, f):
        return lbe_step_voxel(f, self.chi, self.omega, self.Fx, self.Fy)

    def residual(self, f):
        return f - self.lbe_step(f)

    def res_norm(self, f):
        R = self.residual(f)
        return float(np.sqrt((R * R).mean()))

    def initial_field(self):
        rho = np.ones((self.N, self.N), dtype=np.float64) * self.chi
        ux = np.zeros((self.N, self.N), dtype=np.float64)
        uy = np.zeros((self.N, self.N), dtype=np.float64)
        f0 = equilibrium(np.where(self.chi > 0, 1.0, 1.0), ux, uy)
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
        # mask : no correction inside solid
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
