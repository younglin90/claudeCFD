"""D2Q9 BGK LBM core for lid-driven cavity.

Provides:
  - Residual oracle  R_f(f) = f - L(f)
  - Macro projection M : (9,N,N) -> (3,N,N)
  - Linear lifting   T : (3,N,N) -> (9,N,N)  with  M T = I
  - Schur action     S_U v  (Galerkin and AP-corrected)
"""

import numpy as np


# D2Q9 lattice
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.float64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.float64)
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=np.float64)

CX_INT = CX.astype(int)
CY_INT = CY.astype(int)


def equilibrium(rho, ux, uy):
    """Compute f_i^eq for all 9 directions. rho,ux,uy shape (Ny,Nx)."""
    feq = np.empty((9,) + rho.shape, dtype=np.float64)
    u2 = 1.5 * (ux * ux + uy * uy)
    for i in range(9):
        cu = 3.0 * (CX[i] * ux + CY[i] * uy)
        feq[i] = W[i] * rho * (1.0 + cu + 0.5 * cu * cu - u2)
    return feq


def moments(f):
    rho = f.sum(axis=0)
    rhoux = (f * CX[:, None, None]).sum(axis=0)
    rhouy = (f * CY[:, None, None]).sum(axis=0)
    ux = rhoux / rho
    uy = rhouy / rho
    return rho, ux, uy


def stream(f):
    fnew = np.empty_like(f)
    for i in range(9):
        fnew[i] = np.roll(np.roll(f[i], CY_INT[i], axis=0), CX_INT[i], axis=1)
    return fnew


def apply_cavity_bc(f, U_wall):
    """Lid-driven cavity boundary conditions.

    Walls (left/right/bottom): no-slip via full-way bounce-back.
    Top: moving lid (u=U_wall, v=0) via Ladd momentum-corrected bounce-back.
    """
    # Left wall i=0 : incoming pop indices 1, 5, 8
    f[1, :, 0] = f[3, :, 0]
    f[5, :, 0] = f[7, :, 0]
    f[8, :, 0] = f[6, :, 0]

    # Right wall i=Nx-1 : incoming 3, 6, 7
    f[3, :, -1] = f[1, :, -1]
    f[6, :, -1] = f[8, :, -1]
    f[7, :, -1] = f[5, :, -1]

    # Bottom wall j=0 : incoming 2, 5, 6
    f[2, 0, :] = f[4, 0, :]
    f[5, 0, :] = f[7, 0, :]
    f[6, 0, :] = f[8, 0, :]

    # Top wall j=Ny-1 (moving lid) : incoming 4, 7, 8
    rho_top = (
        f[0, -1, :]
        + f[1, -1, :]
        + f[3, -1, :]
        + 2.0 * (f[2, -1, :] + f[5, -1, :] + f[6, -1, :])
    )
    f[4, -1, :] = f[2, -1, :]
    f[7, -1, :] = f[5, -1, :] - 6.0 * W[5] * rho_top * U_wall
    f[8, -1, :] = f[6, -1, :] + 6.0 * W[6] * rho_top * U_wall

    return f


class LBMCavity:
    """Lid-driven cavity case with D2Q9 BGK residual oracle."""

    def __init__(self, N, Re, U_wall=0.1):
        self.N = N
        self.U_wall = U_wall
        self.Re = Re
        self.nu = U_wall * (N - 1) / Re
        self.omega = 1.0 / (3.0 * self.nu + 0.5)
        self.shape = (9, N, N)
        self.dof = 9 * N * N
        self.macro_dof = 3 * N * N

    # ---------------------------------------------------------------- #
    # Native LBM operator and residual
    # ---------------------------------------------------------------- #
    def lbe_step(self, f):
        rho, ux, uy = moments(f)
        feq = equilibrium(rho, ux, uy)
        fstar = f - self.omega * (f - feq)
        fnew = stream(fstar)
        return apply_cavity_bc(fnew, self.U_wall)

    def residual(self, f):
        return f - self.lbe_step(f)

    def res_norm(self, f):
        R = self.residual(f)
        return np.sqrt((R * R).mean())

    def initial_field(self):
        rho0 = np.ones((self.N, self.N), dtype=np.float64)
        ux0 = np.zeros((self.N, self.N), dtype=np.float64)
        uy0 = np.zeros((self.N, self.N), dtype=np.float64)
        # seed top row near lid speed to break degeneracy
        ux0[-1, :] = self.U_wall
        return equilibrium(rho0, ux0, uy0)

    # ---------------------------------------------------------------- #
    # Macro projection M and linear lift T  (M T = I exactly)
    # ---------------------------------------------------------------- #
    def project(self, f):
        """M : (9,N,N) -> (3,N,N) extracting (rho, rho ux, rho uy)."""
        rho = f.sum(axis=0)
        rhoux = (f * CX[:, None, None]).sum(axis=0)
        rhouy = (f * CY[:, None, None]).sum(axis=0)
        return np.stack([rho, rhoux, rhouy], axis=0)

    def lift(self, dU):
        """T : (3,N,N) -> (9,N,N).  Linear lift, M T = I.

        df_i = w_i (drho + 3 c_ix d(rho u) + 3 c_iy d(rho v)).
        Independent of base state -> truly linear.
        """
        drho, drhoux, drhouy = dU[0], dU[1], dU[2]
        df = np.empty((9, *drho.shape), dtype=np.float64)
        for i in range(9):
            df[i] = W[i] * (drho + 3.0 * CX[i] * drhoux + 3.0 * CY[i] * drhouy)
        return df

    # ---------------------------------------------------------------- #
    # JVP and Schur actions
    # ---------------------------------------------------------------- #
    def _fast_norm(self, x):
        # ravel + dot is ~3-5x faster than np.linalg.norm for small arrays
        xr = x.ravel()
        return float(np.sqrt(xr @ xr))

    def _fd_eps(self, norm_f_cached, w):
        norm_w = self._fast_norm(w)
        if norm_w < 1e-30:
            return 1e-8
        return 1e-7 * (norm_f_cached + 1.0) / norm_w

    def jvp(self, w, f_base, R_base, norm_f_cached=None):
        """J_f w ≈ [R_f(f + eps w) - R_f(f)] / eps."""
        if norm_f_cached is None:
            norm_f_cached = self._fast_norm(f_base)
        eps = self._fd_eps(norm_f_cached, w)
        R_pert = self.residual(f_base + eps * w)
        return (R_pert - R_base) / eps

    def schur_galerkin(self, dU, f_base, R_base, norm_f_cached=None):
        """S_U^Gal v = M J_f T v  (1 residual probe)."""
        w = self.lift(dU)
        Jw = self.jvp(w, f_base, R_base, norm_f_cached=norm_f_cached)
        return self.project(Jw)

    def schur_apmnt(self, dU, f_base, R_base, norm_f_cached=None):
        """AP-corrected Schur :
              S_U^AP v = M J_f T v  -  (1/omega) M J_f (I-P_eq) J_f T v
           where P_eq = T M (rank-3 projector onto macro-equilibrium subspace).
           2 residual probes per matvec.
        """
        if norm_f_cached is None:
            norm_f_cached = self._fast_norm(f_base)
        w = self.lift(dU)
        Jw = self.jvp(w, f_base, R_base, norm_f_cached=norm_f_cached)
        SU_gal = self.project(Jw)

        # Non-equilibrium part of J_f T v
        null_part = Jw - self.lift(SU_gal)
        if self._fast_norm(null_part) < 1e-30:
            return SU_gal

        J_null = self.jvp(null_part, f_base, R_base, norm_f_cached=norm_f_cached)
        SU_corr = self.project(J_null)
        return SU_gal - (1.0 / self.omega) * SU_corr
