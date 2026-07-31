"""Newton Jacobian for the IMEX residual.

Two implementations are provided:

  - `assemble_jacobian_analytic` — closed-form (1/(γΔt))·dU/dW + central
    finite-difference cross blocks for ∇p and ∂(p·u)/∂x.  Cheap but tightly
    coupled to the central face stencil; brittle near pure-phase corners.

  - `assemble_jacobian_fd` — sparse FD on `R(W)` using a 3-cell stencil
    (each component perturbs only ±1 neighbours).  Robust and dependable;
    used as the default for Phase 6+7 sprint where the implicit operator
    starts to feel non-trivial.

Both return a `scipy.sparse.csr_matrix` of shape (5N, 5N) ordered cell-major
so that row/column index `5i + k` corresponds to cell i, component k.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

from .primitive import dUdW_analytic
from .residual import residual as _residual

_EPS = 1e-30


def _flatten(R_tuple):
    N = R_tuple[0].shape[0]
    out = np.empty(5 * N, dtype=float)
    for k in range(5):
        out[k::5] = R_tuple[k]
    return out


def _scaled_step(W_k_array, k):
    """Component-aware FD step size."""
    rel = 1e-6
    if k == 0:                         # alpha — always O(1)
        return np.full_like(W_k_array, rel)
    return np.maximum(np.abs(W_k_array) * rel, rel)


def dUdW_blocks(W, eos1, eos2):
    """Extract per-cell analytic dU/dW blocks used by Schur-style solvers.

    Returns a dict of (N,) arrays:
      - A_pp  = d(ρE)/dp    = J[3,4]
      - A_up  = d(ρu)/dp    = J[2,4]
      - A_uu  = d(ρu)/du    = J[2,3]
      - A_ua  = d(ρu)/dα    = J[2,0]
      - A_pa  = d(ρE)/dα    = J[3,0]
      - A_pT1 = d(ρE)/dT1   = J[3,1]
      - A_pT2 = d(ρE)/dT2   = J[3,2]
    """
    J = dUdW_analytic(W, eos1, eos2)
    n = W[0].shape[0]
    a_idx = (0, 1, 2)
    eq_a_idx = (0, 1, 4)     # rows for (a1*rho1, a2*rho2, a1)

    # Raw scalar blocks (u-row, p-row)
    A_pp = J[3, 4].copy()
    A_up = J[2, 4].copy()
    A_uu = J[2, 3].copy()
    A_pu = J[3, 3].copy()

    # Raw coupling vectors/matrices for a-block elimination
    M_aa = np.empty((3, 3, n), dtype=float)
    M_au = np.empty((3, n), dtype=float)
    M_ap = np.empty((3, n), dtype=float)
    M_ua = np.empty((3, n), dtype=float)
    M_pa = np.empty((3, n), dtype=float)
    M_aa_inv = np.empty((3, 3, n), dtype=float)

    Mtilde_uu = np.empty(n, dtype=float)
    Mtilde_up = np.empty(n, dtype=float)
    Mtilde_pu = np.empty(n, dtype=float)
    Mtilde_pp = np.empty(n, dtype=float)
    Sigma_pp = np.empty(n, dtype=float)

    for i in range(n):
        Ji = J[:, :, i]
        Aaa = Ji[np.ix_(eq_a_idx, a_idx)]           # (3,3)
        Aau = Ji[np.ix_(eq_a_idx, (3,))][:, 0]      # (3,)
        Aap = Ji[np.ix_(eq_a_idx, (4,))][:, 0]      # (3,)
        Aua = Ji[np.ix_((2,), a_idx)][0, :]         # (3,)
        Apa = Ji[np.ix_((3,), a_idx)][0, :]         # (3,)

        # Regularize near pure-phase corners.
        reg = 1e-14 * max(float(np.max(np.abs(Aaa))), 1.0)
        Aaa_inv = np.linalg.inv(Aaa + reg * np.eye(3))

        t_uu = A_uu[i] - Aua @ Aaa_inv @ Aau
        t_up = A_up[i] - Aua @ Aaa_inv @ Aap
        t_pu = A_pu[i] - Apa @ Aaa_inv @ Aau
        t_pp = A_pp[i] - Apa @ Aaa_inv @ Aap
        sig = t_pp - t_pu * t_up / max(abs(t_uu), _EPS)

        M_aa[:, :, i] = Aaa
        M_au[:, i] = Aau
        M_ap[:, i] = Aap
        M_ua[:, i] = Aua
        M_pa[:, i] = Apa
        M_aa_inv[:, :, i] = Aaa_inv
        Mtilde_uu[i] = t_uu
        Mtilde_up[i] = t_up
        Mtilde_pu[i] = t_pu
        Mtilde_pp[i] = t_pp
        Sigma_pp[i] = sig

    return {
        'A_pp': A_pp,
        'A_up': A_up,
        'A_uu': A_uu,
        'A_ua': J[2, 0].copy(),
        'A_pa': J[3, 0].copy(),
        'A_pT1': J[3, 1].copy(),
        'A_pT2': J[3, 2].copy(),
        # Schur-ready blocks
        'M_aa': M_aa,
        'M_au': M_au,
        'M_ap': M_ap,
        'M_ua': M_ua,
        'M_pa': M_pa,
        'M_aa_inv': M_aa_inv,
        'Mtilde_uu': Mtilde_uu,
        'Mtilde_up': Mtilde_up,
        'Mtilde_pu': Mtilde_pu,
        'Mtilde_pp': Mtilde_pp,
        'Sigma_pp': Sigma_pp,
    }


def assemble_jacobian_fd(W, U_target, gamma_dt, L_E, eos1, eos2, dx, bc_l, bc_r,
                         *, u_inlet=None, p_inlet=None,
                         alpha_source_explicit=True, kapila_source=None,
                         rhie_chow=False,
                         imp_dissipation=0.0,
                         imp_dissipation_form='biharmonic',
                         imp_compact_lap_coeff=0.0,
                         include_explicit_residual=False,
                         pe_correct=False):
    """Sparse FD Jacobian — colored FD with adaptive stencil width.

    Each (component k, offset s in {0,1,2}) is one color: only cells
    {s, s+3, s+6, …} are perturbed simultaneously, ensuring no two perturbed
    cells share a row in the residual.
    """
    N = W[0].shape[0]
    n_dof = 5 * N
    J = lil_matrix((n_dof, n_dof), dtype=float)

    R0_tuple, _ = _residual(W, U_target, gamma_dt, L_E, eos1, eos2, dx,
                            bc_l, bc_r, u_inlet=u_inlet, p_inlet=p_inlet,
                            alpha_source_explicit=alpha_source_explicit,
                            kapila_source=kapila_source,
                            rhie_chow=rhie_chow,
                            imp_dissipation=imp_dissipation,
                            imp_dissipation_form=imp_dissipation_form,
                            imp_compact_lap_coeff=imp_compact_lap_coeff,
                            include_explicit_residual=include_explicit_residual,
                            pe_correct=pe_correct)
    R0 = _flatten(R0_tuple)

    use_wide_stencil = (
        (imp_dissipation_form == 'biharmonic' and imp_dissipation != 0.0)
        or (imp_compact_lap_coeff != 0.0)
    )
    stride = 5 if use_wide_stencil else 3
    row_offsets = (-2, -1, 0, 1, 2) if use_wide_stencil else (-1, 0, 1)
    for comp in range(5):
        eps_full = _scaled_step(W[comp], comp)
        for offset in range(stride):
            cells = np.arange(offset, N, stride)
            if cells.size == 0:
                continue
            W_pert = list(np.asarray(c, dtype=float).copy() for c in W)
            W_pert[comp][cells] = W[comp][cells] + eps_full[cells]
            R1_tuple, _ = _residual(tuple(W_pert), U_target, gamma_dt, L_E,
                                    eos1, eos2, dx, bc_l, bc_r,
                                    u_inlet=u_inlet, p_inlet=p_inlet,
                                    alpha_source_explicit=alpha_source_explicit,
                                    kapila_source=kapila_source,
                                    rhie_chow=rhie_chow,
                                    imp_dissipation=imp_dissipation,
                                    imp_dissipation_form=imp_dissipation_form,
                                    imp_compact_lap_coeff=imp_compact_lap_coeff,
                                    include_explicit_residual=include_explicit_residual,
                                    pe_correct=pe_correct)
            R1 = _flatten(R1_tuple)
            dR = (R1 - R0)
            for ci in cells:
                col = 5 * ci + comp
                inv_eps = 1.0 / eps_full[ci]
                # Write only the local row blocks touched by the stencil.
                for di in row_offsets:
                    ri = ci + di
                    if ri < 0 or ri >= N:
                        # boundary — implicit cross block absorbed in residual
                        # via ghost extension; skip.
                        continue
                    for r in range(5):
                        J[5 * ri + r, col] = dR[5 * ri + r] * inv_eps
    return csr_matrix(J)


def assemble_jacobian_analytic(W, eos1, eos2, gamma_dt, dx, bc_l, bc_r):
    """Closed-form Jacobian: (1/(γΔt))·dU/dW only.

    Phase 3 baseline used a full analytic block including the central-face
    cross terms, but the resulting matrix can be rank-deficient near pure
    phases.  The simpler "diagonal-only" form here lets the Newton step
    treat the implicit acoustic block via the FD path inside
    `assemble_jacobian_fd`; we keep this routine for diagnostics.
    """
    N = W[0].shape[0]
    inv_gdt = 1.0 / gamma_dt
    J = lil_matrix((5 * N, 5 * N), dtype=float)
    Jud = dUdW_analytic(W, eos1, eos2)
    for i in range(N):
        for r in range(5):
            for c in range(5):
                J[5 * i + r, 5 * i + c] = inv_gdt * Jud[r, c, i]
    return csr_matrix(J)


# Default Jacobian builder (signature consistent with newton.py)
def assemble_jacobian(W, eos1, eos2, gamma_dt, dx, bc_l, bc_r):
    """Backward-compat shim: returns the analytic diagonal Jacobian.

    Newton solver in this branch uses `assemble_jacobian_fd` directly with
    full residual context.  The shim is kept for any external caller and
    simply returns the (1/(γΔt))·dU/dW block.
    """
    return assemble_jacobian_analytic(W, eos1, eos2, gamma_dt, dx, bc_l, bc_r)
