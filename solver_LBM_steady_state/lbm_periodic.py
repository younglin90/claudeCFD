"""D2Q9 BGK + Guo forcing + fully periodic boundary.

Kolmogorov flow steady state :
    F_x(y) = F_0 sin(k_f y)              (body force, sinusoidal shear)
    u_x^*(y) = F_0/(nu k_f^2) sin(k_f y)  (analytical NS steady)
    u_y^*  = 0,    rho^* = 1
"""

import numpy as np


CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.float64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.float64)
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=np.float64)
CX_INT = CX.astype(int)
CY_INT = CY.astype(int)


def equilibrium(rho, ux, uy):
    feq = np.empty((9,) + rho.shape, dtype=np.float64)
    u2 = 1.5 * (ux * ux + uy * uy)
    for i in range(9):
        cu = 3.0 * (CX[i] * ux + CY[i] * uy)
        feq[i] = W[i] * rho * (1.0 + cu + 0.5 * cu * cu - u2)
    return feq


def stream_periodic(f):
    fn = np.empty_like(f)
    for i in range(9):
        fn[i] = np.roll(np.roll(f[i], CY_INT[i], axis=0), CX_INT[i], axis=1)
    return fn


def guo_source(rho, ux, uy, Fx, Fy, omega):
    """Guo et al. (2002) forcing source S_i."""
    coeff = (1.0 - 0.5 * omega)
    S = np.empty((9,) + rho.shape, dtype=np.float64)
    for i in range(9):
        cu = CX[i] * ux + CY[i] * uy
        term = ((CX[i] - ux) + 3.0 * cu * CX[i]) * Fx + (
            (CY[i] - uy) + 3.0 * cu * CY[i]
        ) * Fy
        S[i] = coeff * W[i] * 3.0 * term
    return S


class KolmogorovCase:
    """Periodic 2D Kolmogorov flow with Guo forcing."""

    def __init__(self, N, nu, F0=1e-5, kf=1):
        self.N = N
        self.nu = nu
        self.omega = 1.0 / (3.0 * nu + 0.5)
        self.F0 = F0
        self.kf = kf
        self.shape = (9, N, N)
        self.dof = 9 * N * N
        self.macro_dof = 3 * N * N

        # body force F_x(y) = F0 sin(kf * 2pi y / N)
        y = np.arange(N, dtype=np.float64).reshape(N, 1)
        self.k_lat = 2.0 * np.pi * kf / N
        self.Fx = F0 * np.sin(self.k_lat * y) * np.ones((N, N), dtype=np.float64)
        self.Fy = np.zeros((N, N), dtype=np.float64)

        # analytical steady velocity amplitude
        self.U_amp = F0 / (nu * self.k_lat * self.k_lat)
        self.Re = self.U_amp * N / nu

    def lbe_step(self, f):
        rho = f.sum(axis=0)
        rhoux = (f * CX[:, None, None]).sum(axis=0)
        rhouy = (f * CY[:, None, None]).sum(axis=0)
        # Guo velocity shift: u_eq = (rho u + 0.5 F)/rho
        ux = rhoux / rho
        uy = rhouy / rho
        ux_eq = ux + 0.5 * self.Fx / rho
        uy_eq = uy + 0.5 * self.Fy / rho
        feq = equilibrium(rho, ux_eq, uy_eq)
        S = guo_source(rho, ux, uy, self.Fx, self.Fy, self.omega)
        fstar = f - self.omega * (f - feq) + S
        return stream_periodic(fstar)

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
        y = np.arange(self.N, dtype=np.float64).reshape(self.N, 1)
        return self.U_amp * np.sin(self.k_lat * y) * np.ones((self.N, self.N))

    # ---------------- projection / lift / Schur ----------------
    def project(self, f):
        rho = f.sum(axis=0)
        rhoux = (f * CX[:, None, None]).sum(axis=0)
        rhouy = (f * CY[:, None, None]).sum(axis=0)
        return np.stack([rho, rhoux, rhouy], axis=0)

    def lift(self, dU):
        """Linear lift (state-independent), M T = I."""
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


# ---------------------------------------------------------------- #
#  Spectral Fourier-Moment Schur preconditioner
# ---------------------------------------------------------------- #
def build_spectral_schur(N, omega=None, mode="ap"):
    """Pre-compute 3x3 complex Schur S_U(k) and its inverse for every Fourier
    mode k = 2 pi (m, n)/N.

    Linearization around uniform base (rho_bar=1, u_bar=0) :

        Collision  C(omega) = (1-omega) I + omega T M
        Streaming  A(k)     = diag(exp(-i k . c_i))
        Linear LBE step  L'(k) = A(k) C
        Residual Jacobian J(k) = I - L'(k)

    Galerkin Schur :
        S_U^G(k) = M J(k) T = I - M A(k) T

    AP-corrected (kinetic null-space contribution, J_kk ~= omega) :
        S_U^AP(k) = S_U^G  -  (1 - omega)/omega [ M A^2 T  -  (M A T)^2 ]

    Mode (0,0) (mean) is singular -> inverse set to 0.

    Parameters
    ----------
    N       : grid size
    omega   : collision rate ; required when mode='ap'
    mode    : 'galerkin' or 'ap'

    Returns
    -------
    S_inv : ndarray shape (N, N, 3, 3) complex
    """
    if mode == "ap" and omega is None:
        raise ValueError("AP mode requires omega")
    M_mat = np.zeros((3, 9), dtype=np.float64)
    M_mat[0, :] = 1.0
    M_mat[1, :] = CX
    M_mat[2, :] = CY

    T_mat = np.zeros((9, 3), dtype=np.float64)
    for i in range(9):
        T_mat[i, 0] = W[i]
        T_mat[i, 1] = 3.0 * W[i] * CX[i]
        T_mat[i, 2] = 3.0 * W[i] * CY[i]

    kx = 2.0 * np.pi * np.fft.fftfreq(N)  # (N,)
    ky = 2.0 * np.pi * np.fft.fftfreq(N)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")  # KX shape (N, N) ; KY shape (N, N)
    # NOTE : in our convention, rows of f are along Y (axis 0), cols along X (axis 1).
    # We want phase[i, j_row, k_col] = exp(-i (KX*c_x + KY*c_y))
    # with j_row indexing rows (y), k_col indexing cols (x). np.meshgrid with
    # indexing='xy' gives KX varying along axis 1, KY along axis 0 -- correct.

    phase = np.empty((9, N, N), dtype=np.complex128)
    for i in range(9):
        phase[i] = np.exp(-1j * (KX * CX[i] + KY * CY[i]))

    # MAT[a, b, j, k] = sum_i M[a,i] T[i,b] phase[i,j,k]
    MAT = np.einsum("ai,ib,ijk->abjk", M_mat, T_mat, phase)

    # Galerkin Schur S_U^G = I - MAT
    S_U = -MAT.copy()
    for a in range(3):
        S_U[a, a] += 1.0

    if mode == "ap":
        phase2 = phase * phase
        MA2T = np.einsum("ai,ib,ijk->abjk", M_mat, T_mat, phase2)
        MAT2 = np.einsum("abjk,bcjk->acjk", MAT, MAT)
        coeff = 0.5 * (1.0 - omega) / omega
        S_U = S_U - coeff * (MA2T - MAT2)

    # iter17 : momentum eta larger
    S_U_t = np.transpose(S_U, (2, 3, 0, 1))
    eta_diag = np.diag([5e-2, 1e-1, 1e-1]).astype(np.complex128)
    S_U_reg = S_U_t + eta_diag[None, None, :, :]
    S_inv = np.linalg.inv(S_U_reg)
    # iter4 : mode (0,0) -> diag(0, 1, 1). mass mean is conserved, no correction;
    # only momentum mean gets passthrough (let kinetic LBE handle)
    mode00 = np.zeros((3, 3), dtype=np.complex128)
    mode00[1, 1] = 1.0
    mode00[2, 2] = 1.0
    S_inv[0, 0] = mode00
    return S_inv


def apply_spectral_schur(case, R_f, S_inv, k_low_cutoff=None):
    """Apply preconditioner :  df = T  IFFT( S_U^{-1}  FFT( M R_f ) ).

    Parameters
    ----------
    k_low_cutoff : if not None, zero macro residual modes with
        |k|/k_nyq < k_low_cutoff   (cutoff in normalized frequency [0,1]).
        Lets the LBE smoother handle low-frequency (wall-dominated) residual.
    """
    R_U = case.project(R_f)                       # (3, N, N) real
    R_U_hat = np.fft.fft2(R_U, axes=(1, 2))       # (3, N, N) complex

    if k_low_cutoff is not None:
        N = R_U.shape[-1]
        # |k| normalized to Nyquist = N/2
        kxv = np.fft.fftfreq(N) * N
        kyv = np.fft.fftfreq(N) * N
        KX, KY = np.meshgrid(kxv, kyv, indexing="xy")
        kmag = np.sqrt(KX * KX + KY * KY) / (0.5 * N)  # [0, ~sqrt(2)]
        mask = kmag >= k_low_cutoff
        R_U_hat = R_U_hat * mask[None, :, :]

    R_perm = np.transpose(R_U_hat, (1, 2, 0))     # (N, N, 3)
    dU_perm = np.einsum("jkab,jkb->jka", S_inv, R_perm)
    dU_hat = np.transpose(dU_perm, (2, 0, 1))     # (3, N, N)
    dU = np.real(np.fft.ifft2(dU_hat, axes=(1, 2)))
    return case.lift(dU)
