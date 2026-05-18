"""D3Q19 BGK + Guo forcing + fully periodic boundary.

Lattice :
    c_0  = (0, 0, 0)                          w_0 = 1/3
    c_1-6  = (±1, 0, 0), (0, ±1, 0), (0, 0, ±1)   w = 1/18
    c_7-18 = (±1, ±1, 0), (±1, 0, ±1), (0, ±1, ±1)  w = 1/36
"""

import numpy as np


# D3Q19 velocities
_C3D = np.array([
    [ 0,  0,  0],  # 0
    [ 1,  0,  0],  # 1
    [-1,  0,  0],  # 2
    [ 0,  1,  0],  # 3
    [ 0, -1,  0],  # 4
    [ 0,  0,  1],  # 5
    [ 0,  0, -1],  # 6
    [ 1,  1,  0],  # 7
    [-1, -1,  0],  # 8
    [ 1, -1,  0],  # 9
    [-1,  1,  0],  # 10
    [ 1,  0,  1],  # 11
    [-1,  0, -1],  # 12
    [ 1,  0, -1],  # 13
    [-1,  0,  1],  # 14
    [ 0,  1,  1],  # 15
    [ 0, -1, -1],  # 16
    [ 0,  1, -1],  # 17
    [ 0, -1,  1],  # 18
], dtype=np.float64)

CX = _C3D[:, 0]
CY = _C3D[:, 1]
CZ = _C3D[:, 2]
W3D = np.array([1/3] + [1/18] * 6 + [1/36] * 12, dtype=np.float64)

CX_INT = CX.astype(int)
CY_INT = CY.astype(int)
CZ_INT = CZ.astype(int)


def equilibrium_3d(rho, ux, uy, uz):
    """f_i^eq for D3Q19. Shape rho,u : (N,N,N). Returns (19,N,N,N)."""
    feq = np.empty((19,) + rho.shape, dtype=np.float64)
    u2 = 1.5 * (ux * ux + uy * uy + uz * uz)
    for i in range(19):
        cu = 3.0 * (CX[i] * ux + CY[i] * uy + CZ[i] * uz)
        feq[i] = W3D[i] * rho * (1.0 + cu + 0.5 * cu * cu - u2)
    return feq


def stream_3d_periodic(f):
    """3D periodic streaming. f shape (19,N,N,N) -> (19,N,N,N)."""
    fn = np.empty_like(f)
    for i in range(19):
        fn[i] = np.roll(np.roll(np.roll(f[i], CZ_INT[i], axis=0),
                                  CY_INT[i], axis=1),
                          CX_INT[i], axis=2)
    return fn


def guo_source_3d(rho, ux, uy, uz, Fx, Fy, Fz, omega):
    """Guo forcing source S_i for D3Q19."""
    coeff = (1.0 - 0.5 * omega)
    S = np.empty((19,) + rho.shape, dtype=np.float64)
    for i in range(19):
        cu = CX[i] * ux + CY[i] * uy + CZ[i] * uz
        term = ((CX[i] - ux) + 3.0 * cu * CX[i]) * Fx + \
               ((CY[i] - uy) + 3.0 * cu * CY[i]) * Fy + \
               ((CZ[i] - uz) + 3.0 * cu * CZ[i]) * Fz
        S[i] = coeff * W3D[i] * 3.0 * term
    return S


class Kolmogorov3DCase:
    """3D Kolmogorov flow : periodic + F_x(y) = F0 sin(kf · 2π y/N).

    Analytical steady (incompressible NS) :
        u_x^*(y) = F0/(nu kf^2) sin(kf y), u_y^* = u_z^* = 0
    """

    def __init__(self, N, nu, F0=1e-5, kf=1):
        self.N = N
        self.nu = nu
        self.omega = 1.0 / (3.0 * nu + 0.5)
        self.F0 = F0
        self.kf = kf
        self.shape = (19, N, N, N)
        self.dof = 19 * N * N * N
        self.macro_dof = 4 * N * N * N  # rho, ρux, ρuy, ρuz
        self.n_U = 4

        y = np.arange(N, dtype=np.float64).reshape(1, N, 1)
        self.k_lat = 2.0 * np.pi * kf / N
        self.Fx = F0 * np.sin(self.k_lat * y) * np.ones((N, N, N), dtype=np.float64)
        self.Fy = np.zeros((N, N, N), dtype=np.float64)
        self.Fz = np.zeros((N, N, N), dtype=np.float64)

        self.U_amp = F0 / (nu * self.k_lat * self.k_lat)
        self.Re = self.U_amp * N / nu

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
        return stream_3d_periodic(fstar)

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
        y = np.arange(self.N, dtype=np.float64).reshape(1, self.N, 1)
        return self.U_amp * np.sin(self.k_lat * y) * np.ones((self.N, self.N, self.N))

    def project(self, f):
        """M : f -> (rho, ρux, ρuy, ρuz). Shape (4, N, N, N)."""
        rho = f.sum(axis=0)
        rhoux = (f * CX[:, None, None, None]).sum(axis=0)
        rhouy = (f * CY[:, None, None, None]).sum(axis=0)
        rhouz = (f * CZ[:, None, None, None]).sum(axis=0)
        return np.stack([rho, rhoux, rhouy, rhouz], axis=0)

    def lift(self, dU):
        """T : (4,N,N,N) -> (19,N,N,N) linear lift, M T = I."""
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


def build_spectral_schur_3d(N, omega):
    """3D Fourier-Moment AP-Schur preconditioner for D3Q19.

    S_U^AP(k) = (I - MAT) - coeff * [MA²T - (MAT)²]   per Fourier mode
    coeff = clipped((1-ω)/ω)
    Adaptive Tikhonov regularization.

    Returns S_inv shape (N, N, N, 4, 4) complex.
    """
    n_U = 4

    M_mat = np.zeros((n_U, 19), dtype=np.float64)
    M_mat[0, :] = 1.0
    M_mat[1, :] = CX
    M_mat[2, :] = CY
    M_mat[3, :] = CZ

    T_mat = np.zeros((19, n_U), dtype=np.float64)
    for i in range(19):
        T_mat[i, 0] = W3D[i]
        T_mat[i, 1] = 3.0 * W3D[i] * CX[i]
        T_mat[i, 2] = 3.0 * W3D[i] * CY[i]
        T_mat[i, 3] = 3.0 * W3D[i] * CZ[i]

    kx = 2.0 * np.pi * np.fft.fftfreq(N)
    ky = 2.0 * np.pi * np.fft.fftfreq(N)
    kz = 2.0 * np.pi * np.fft.fftfreq(N)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")  # axes (z,y,x)

    phase = np.empty((19, N, N, N), dtype=np.complex128)
    for i in range(19):
        phase[i] = np.exp(-1j * (KX * CX[i] + KY * CY[i] + KZ * CZ[i]))

    # MAT
    MAT = np.einsum("ai,ib,izyx->abzyx", M_mat, T_mat, phase)
    # MA²T
    phase2 = phase * phase
    MA2T = np.einsum("ai,ib,izyx->abzyx", M_mat, T_mat, phase2)
    # (MAT)²
    MAT2 = np.einsum("abzyx,bczyx->aczyx", MAT, MAT)

    # Galerkin Schur
    S_U = -MAT.copy()
    for a in range(n_U):
        S_U[a, a] += 1.0

    # AP correction
    raw = (1.0 - omega) / omega
    coeff = 0.5 * np.sign(raw) * min(0.5, abs(raw))
    S_U = S_U - coeff * (MA2T - MAT2)

    # Adaptive Tikhonov
    S_U_t = np.transpose(S_U, (2, 3, 4, 0, 1))  # (N,N,N,4,4)
    sing = np.linalg.svd(S_U_t, compute_uv=False)
    sigma_max = float(sing.max())
    eta_auto = sigma_max / 50.0

    I_n = np.eye(n_U, dtype=np.complex128)
    S_U_reg = S_U_t + eta_auto * I_n[None, None, None, :, :]
    S_inv = np.linalg.pinv(S_U_reg)

    # Mode (0,0,0) = mass conservation (no Newton on rho mean), momentum free
    mode0 = np.zeros((n_U, n_U), dtype=np.complex128)
    mode0[1, 1] = 1.0
    mode0[2, 2] = 1.0
    mode0[3, 3] = 1.0
    S_inv[0, 0, 0] = mode0

    return S_inv


def apply_spectral_schur_3d(case, R_f, S_inv):
    """Apply 3D Fourier PC : df = T·IFFT(S_inv·FFT(M·R_f))."""
    R_U = case.project(R_f)                              # (4, N, N, N)
    R_U_hat = np.fft.fftn(R_U, axes=(1, 2, 3))           # (4, N, N, N) complex
    R_perm = np.transpose(R_U_hat, (1, 2, 3, 0))         # (N, N, N, 4)
    dU_perm = np.einsum("zyxab,zyxb->zyxa", S_inv, R_perm)
    dU_hat = np.transpose(dU_perm, (3, 0, 1, 2))         # (4, N, N, N)
    dU = np.real(np.fft.ifftn(dU_hat, axes=(1, 2, 3)))
    return case.lift(dU)
