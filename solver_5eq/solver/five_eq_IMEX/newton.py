"""Newton-Krylov solver for one ARS-stage IMEX implicit equation.

Solves
    R(W) = 0
with `R` from `residual.residual` and Jacobian from `jacobian.assemble_jacobian`.
Uses scipy.sparse.linalg.spsolve as the linear solver (suitable for N ≤ 200);
larger problems should switch to GMRES + ILU in Phase 4.

Damped Newton with positivity guards:

    δW := solve(J, -R)
    λ  := 1
    while not admissible(W + λ δW) or ‖R(W+λδW)‖ > (1−η λ) ‖R(W)‖:
        λ /= 2

Admissibility: α∈[1e-12, 1−1e-12], T_k > T_floor, p > p_floor.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse.linalg import spsolve

from .primitive import pack_W, unpack_W
from .residual import residual as _residual_fn
from .residual import residual_full as _residual_full_fn
from .jacobian import assemble_jacobian_fd, dUdW_blocks
from .helmholtz import solve_helmholtz_periodic
from .sound_speed import phase_sound_speed_sq, mixture_sound_speed_sq


def _flatten_R(R_tuple):
    """Pack 5-tuple of (N,) into one (5N,) interleaved cell-major vector."""
    N = R_tuple[0].shape[0]
    out = np.empty(5 * N, dtype=float)
    for k in range(5):
        out[k::5] = R_tuple[k]
    return out


def _residual_norm(R_tuple, scales):
    """Scaled infinity-norm: max_k max_i |R_k(i)| / scale_k."""
    out = 0.0
    for k in range(5):
        s = max(scales[k], 1e-30)
        out = max(out, float(np.max(np.abs(R_tuple[k])) / s))
    return out


def _admissible(W, T_floor=1.0, p_floor=1.0):
    a, T1, T2, _, p = W
    return (np.all((a > 1e-12) & (a < 1.0 - 1e-12))
            and np.all(T1 > T_floor) and np.all(T2 > T_floor)
            and np.all(p > p_floor))


def _scales_from_W(W, eos1, eos2):
    """Component-wise reference scales for residual norm.

    Use U_ref values — gives the residual a comparable magnitude across α (~O(1)),
    α·ρ_k (~O(1)–O(10³)), ρ u (~O(ρ |u|)) and ρ E (~O(ρ e)).
    """
    a1, T1, T2, u, p = W
    rho1 = eos1.density(p, T1)
    rho2 = eos2.density(p, T2)
    e1 = eos1.energy(rho1, p)
    e2 = eos2.energy(rho2, p)
    rho = a1 * rho1 + (1.0 - a1) * rho2
    rho_e = a1 * rho1 * e1 + (1.0 - a1) * rho2 * e2
    return [
        max(np.max(np.abs(a1 * rho1)), 1.0),
        max(np.max(np.abs((1.0 - a1) * rho2)), 1.0),
        max(np.max(np.abs(rho * np.maximum(np.abs(u), 1.0))), 1.0),
        max(np.max(np.abs(rho_e + 0.5 * rho * u * u)), 1.0),
        1.0,
    ]


def newton_solve(W0, U_target, gamma_dt, L_E, eos1, eos2, dx, bc_l, bc_r, *,
                 max_iter=10, rtol=1e-6, atol=1e-10,
                 line_search_max=8, eta=1e-4,
                 u_inlet=None, p_inlet=None,
                 alpha_source_explicit=True,
                 kapila_source=None,
                 rhie_chow=False,
                 imp_dissipation=0.0,
                 imp_dissipation_form='biharmonic',
                 imp_compact_lap_coeff=0.0,
                 include_explicit_residual=False,
                 pe_correct=False,
                 verbose=False):
    """Solve R(W) = 0 starting from W0.  Returns (W, info)."""
    W = tuple(np.asarray(c, dtype=float).copy() for c in W0)
    N = W[0].shape[0]

    scales = _scales_from_W(W, eos1, eos2)
    R, _ = _residual_fn(W, U_target, gamma_dt, L_E, eos1, eos2, dx, bc_l, bc_r,
                       u_inlet=u_inlet, p_inlet=p_inlet,
                       alpha_source_explicit=alpha_source_explicit,
                       kapila_source=kapila_source,
                       rhie_chow=rhie_chow,
                       imp_dissipation=imp_dissipation,
                       imp_dissipation_form=imp_dissipation_form,
                       imp_compact_lap_coeff=imp_compact_lap_coeff,
                       include_explicit_residual=include_explicit_residual,
                       pe_correct=pe_correct)
    norm = _residual_norm(R, scales)
    norm0 = max(norm, atol)

    history = [norm]
    it = 0
    converged = (norm <= atol + rtol * norm0) or (norm == 0.0)

    while not converged and it < max_iter:
        J = assemble_jacobian_fd(W, U_target, gamma_dt, L_E,
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
        Rvec = _flatten_R(R)
        try:
            # Tikhonov regularization for tiny diagonals (pure-phase corners)
            from scipy.sparse import eye as speye
            n = J.shape[0]
            lam = 1e-12 * float(np.max(np.abs(J.diagonal())))
            dW_vec = spsolve(J + lam * speye(n, format='csr'), -Rvec)
        except Exception as exc:
            return W, dict(converged=False, iter=it, history=history, error=str(exc))

        # Cell-major unpack: dW_vec[5i + k] → component k, cell i
        dW = [dW_vec[k::5] for k in range(5)]

        lam = 1.0
        for _ in range(line_search_max):
            W_trial = tuple(W[k] + lam * dW[k] for k in range(5))
            if _admissible(W_trial):
                R_trial, _ = _residual_fn(W_trial, U_target, gamma_dt, L_E,
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
                norm_trial = _residual_norm(R_trial, scales)
                if norm_trial <= (1.0 - eta * lam) * norm:
                    W = W_trial
                    R = R_trial
                    norm = norm_trial
                    break
            lam *= 0.5
        else:
            return W, dict(converged=False, iter=it, history=history,
                            reason='line_search_failed', norm=norm)

        history.append(norm)
        it += 1
        converged = (norm <= atol + rtol * norm0)
        if verbose:
            print(f"    Newton iter {it}: ‖R‖∞/scale={norm:.3e}, λ={lam:.3f}")

    return W, dict(converged=converged, iter=it, history=history, norm=norm)


def _grad_central_periodic(phi, dx):
    return (np.roll(phi, -1) - np.roll(phi, 1)) / (2.0 * dx)


def _grad_implicit_periodic(phi, dx, *, dissipation=0.0, dissipation_form='biharmonic'):
    """Match the implicit-pressure gradient stencil used in residual.py."""
    if dissipation_form == 'biharmonic' and dissipation > 0.0:
        p_LL = np.roll(phi, 1)
        p_L = phi
        p_R = np.roll(phi, -1)
        p_RR = np.roll(phi, -2)
        bih = (-p_LL + 3.0 * p_L - 3.0 * p_R + p_RR) / 8.0
        p_face = 0.5 * (p_L + p_R) - dissipation * bih
        return (p_face - np.roll(p_face, 1)) / dx
    return _grad_central_periodic(phi, dx)


def newton_solve_schur(W0, U_target, gamma_dt, L_E, eos1, eos2, dx, bc_l, bc_r, *,
                       max_iter=10, rtol=1e-6, atol=1e-10,
                       line_search_max=8, eta=1e-4,
                       u_inlet=None, p_inlet=None,
                       alpha_source_explicit=True,
                       kapila_source=None,
                       rhie_chow=False,
                       imp_dissipation=0.0,
                       imp_dissipation_form='biharmonic',
                       imp_compact_lap_coeff=0.0,
                       include_explicit_residual=False,
                       pe_correct=False,
                       verbose=False):
    """Approximate Schur-Helmholtz Newton update for periodic (u,p)-block.

    Current scope: periodic BC only. For other BC types, this function
    transparently falls back to `newton_solve`.
    """
    if (not (bc_l == 'periodic' and bc_r == 'periodic')) or kapila_source is not None:
        return newton_solve(
            W0, U_target, gamma_dt, L_E, eos1, eos2, dx, bc_l, bc_r,
            max_iter=max_iter, rtol=rtol, atol=atol,
            line_search_max=line_search_max, eta=eta,
            u_inlet=u_inlet, p_inlet=p_inlet,
            alpha_source_explicit=alpha_source_explicit,
            kapila_source=kapila_source,
            rhie_chow=rhie_chow,
            imp_dissipation=imp_dissipation,
            imp_dissipation_form=imp_dissipation_form,
            imp_compact_lap_coeff=imp_compact_lap_coeff,
            include_explicit_residual=include_explicit_residual,
            pe_correct=pe_correct,
            verbose=verbose
        )

    W = tuple(np.asarray(c, dtype=float).copy() for c in W0)
    scales = _scales_from_W(W, eos1, eos2)
    R, _ = _residual_fn(W, U_target, gamma_dt, L_E, eos1, eos2, dx, bc_l, bc_r,
                        u_inlet=u_inlet, p_inlet=p_inlet,
                       alpha_source_explicit=alpha_source_explicit,
                       kapila_source=kapila_source,
                       rhie_chow=rhie_chow,
                       imp_dissipation=imp_dissipation,
                       imp_dissipation_form=imp_dissipation_form,
                       imp_compact_lap_coeff=imp_compact_lap_coeff,
                       include_explicit_residual=include_explicit_residual,
                       pe_correct=pe_correct)
    norm = _residual_norm(R, scales)
    norm0 = max(norm, atol)

    history = [norm]
    it = 0
    converged = (norm <= atol + rtol * norm0) or (norm == 0.0)

    while not converged and it < max_iter:
        blk = dUdW_blocks(W, eos1, eos2)

        # Residual partitions for a-block Schur elimination.
        r_a = np.vstack((R[0], R[1], R[4]))         # shape (3, N)
        r_u = R[2]
        r_p = R[3]
        n = W[0].shape[0]

        r_tilde_u = np.empty(n, dtype=float)
        r_tilde_p = np.empty(n, dtype=float)
        for i in range(n):
            Aaa_inv = blk['M_aa_inv'][:, :, i]
            corr_a = Aaa_inv @ r_a[:, i]
            r_tilde_u[i] = r_u[i] - blk['M_ua'][:, i] @ corr_a
            r_tilde_p[i] = r_p[i] - blk['M_pa'][:, i] @ corr_a

        a1, T1, T2, _, p = W
        rho1 = eos1.density(p, T1)
        rho2 = eos2.density(p, T2)
        c1_sq = phase_sound_speed_sq(eos1, rho1, T1)
        c2_sq = phase_sound_speed_sq(eos2, rho2, T2)
        c_mix_sq = mixture_sound_speed_sq(a1, rho1, c1_sq, rho2, c2_sq, kind='kapila')
        # k_face = gamma_dt/(rho_eff*dx^2) ≈ gamma_dt*c^2/dx^2
        rho_eff = 1.0 / np.maximum(c_mix_sq, 1e-30)
        sigma_pp = np.where(np.abs(blk['Sigma_pp']) > 1e-30,
                            blk['Sigma_pp'],
                            np.sign(blk['Sigma_pp'] + 1e-300) * 1e-30)

        # Pressure correction from reduced p-row.
        rhs_p = -(r_tilde_p - (blk['Mtilde_pu'] / np.maximum(blk['Mtilde_uu'], 1e-30)) * r_tilde_u)
        dp = solve_helmholtz_periodic(
            sigma_pp, rho_eff, gamma_dt, dx, rhs_p
        )
        # Reduced u-row:
        # (Mtilde_uu/gdt) du + (Mtilde_up/gdt) dp + grad(dp) = -r_tilde_u
        grad_dp = _grad_implicit_periodic(
            dp, dx,
            dissipation=imp_dissipation,
            dissipation_form=imp_dissipation_form
        )
        du = (-r_tilde_u - (blk['Mtilde_up'] / gamma_dt) * dp - grad_dp) / np.maximum(
            blk['Mtilde_uu'] / gamma_dt, 1e-30
        )

        # Back-substitute (a, T1, T2) from a-block row.
        da = np.empty_like(r_a)
        rhs_scale = gamma_dt
        for i in range(n):
            Aaa_inv = blk['M_aa_inv'][:, :, i]
            rhs_a = (-rhs_scale * r_a[:, i]
                     - blk['M_au'][:, i] * du[i]
                     - blk['M_ap'][:, i] * dp[i])
            da[:, i] = Aaa_inv @ rhs_a

        dW = [np.zeros_like(W[k]) for k in range(5)]
        dW[0] = da[0, :]
        dW[1] = da[1, :]
        dW[2] = da[2, :]
        dW[3] = du
        dW[4] = dp

        lam = 1.0
        for _ in range(line_search_max):
            W_trial = tuple(W[k] + lam * dW[k] for k in range(5))
            if _admissible(W_trial):
                R_trial, _ = _residual_fn(
                    W_trial, U_target, gamma_dt, L_E, eos1, eos2, dx, bc_l, bc_r,
                    u_inlet=u_inlet, p_inlet=p_inlet,
                    alpha_source_explicit=alpha_source_explicit,
                    kapila_source=kapila_source,
                    rhie_chow=rhie_chow,
                    imp_dissipation=imp_dissipation,
                    imp_dissipation_form=imp_dissipation_form,
                    imp_compact_lap_coeff=imp_compact_lap_coeff,
                    include_explicit_residual=include_explicit_residual,
                    pe_correct=pe_correct
                )
                norm_trial = _residual_norm(R_trial, scales)
                if norm_trial <= (1.0 - eta * lam) * norm:
                    W = W_trial
                    R = R_trial
                    norm = norm_trial
                    break
            lam *= 0.5
        else:
            return W, dict(converged=False, iter=it, history=history,
                           reason='line_search_failed', norm=norm,
                           solver='schur')

        history.append(norm)
        it += 1
        converged = (norm <= atol + rtol * norm0)
        if verbose:
            print(f"    Newton-schur iter {it}: ‖R‖∞/scale={norm:.3e}, λ={lam:.3f}")

    return W, dict(converged=converged, iter=it, history=history,
                   norm=norm, solver='schur')


def newton_solve_full(W0, U_n, dt, eos1, eos2, dx, bc_l, bc_r, *,
                      max_iter=15, rtol=1e-8, atol=1e-12,
                      line_search_max=10, eta=1e-4,
                      u_inlet=None, p_inlet=None,
                      alpha_scheme='upwind', primitive_scheme='upwind',
                      energy_form='apec', kapila_closure=False,
                      positivity=True,
                      verbose=False):
    """Solve fully-implicit BE residual R_full(W) = 0 with FD-sparse Newton.

    The Jacobian is built via 3-cell-stride FD on `R_full`, so it captures
    the W-dependence of L_E (mass / momentum / energy advection) on top of
    the ARS-style L_I — at the cost of more FD perturbations per Newton
    iteration but with machine-ε PE preservation in return.
    """
    from scipy.sparse import lil_matrix, csr_matrix, eye as speye

    W = tuple(np.asarray(c, dtype=float).copy() for c in W0)
    N = W[0].shape[0]

    def R_full(W_in):
        R_t, _, _ = _residual_full_fn(W_in, U_n, dt, eos1, eos2, dx, bc_l, bc_r,
                                       u_inlet=u_inlet, p_inlet=p_inlet,
                                       alpha_scheme=alpha_scheme,
                                       primitive_scheme=primitive_scheme,
                                       energy_form=energy_form,
                                       kapila_closure=kapila_closure,
                                       positivity=positivity)
        return R_t

    def fd_jacobian(W_curr, R0_tuple):
        n_dof = 5 * N
        J = lil_matrix((n_dof, n_dof), dtype=float)
        R0 = _flatten_R(R0_tuple)
        stride = 3
        for comp in range(5):
            wc = W_curr[comp]
            if comp == 0:
                eps_full = np.full_like(wc, 1e-7)
            else:
                eps_full = np.maximum(np.abs(wc) * 1e-7, 1e-7)
            for offset in range(stride):
                cells = np.arange(offset, N, stride)
                if cells.size == 0:
                    continue
                W_pert = list(np.asarray(c, dtype=float).copy() for c in W_curr)
                W_pert[comp][cells] = wc[cells] + eps_full[cells]
                R1 = _flatten_R(R_full(tuple(W_pert)))
                dR = R1 - R0
                for ci in cells:
                    col = 5 * ci + comp
                    inv_eps = 1.0 / eps_full[ci]
                    for di in (-1, 0, 1):
                        ri = ci + di
                        if ri < 0 or ri >= N:
                            continue
                        for r in range(5):
                            J[5 * ri + r, col] = dR[5 * ri + r] * inv_eps
        return csr_matrix(J)

    scales = _scales_from_W(W, eos1, eos2)
    R = R_full(W)
    norm = _residual_norm(R, scales)
    norm0 = max(norm, atol)
    history = [norm]
    it = 0
    converged = (norm <= atol + rtol * norm0)

    while not converged and it < max_iter:
        J = fd_jacobian(W, R)
        Rvec = _flatten_R(R)
        try:
            n = J.shape[0]
            lam = 1e-12 * float(np.max(np.abs(J.diagonal())))
            dW_vec = spsolve(J + lam * speye(n, format='csr'), -Rvec)
        except Exception as exc:
            return W, dict(converged=False, iter=it, history=history, error=str(exc))

        dW = [dW_vec[k::5] for k in range(5)]
        lam_ls = 1.0
        for _ in range(line_search_max):
            W_trial = tuple(W[k] + lam_ls * dW[k] for k in range(5))
            if _admissible(W_trial):
                R_trial = R_full(W_trial)
                norm_trial = _residual_norm(R_trial, scales)
                if norm_trial <= (1.0 - eta * lam_ls) * norm:
                    W = W_trial
                    R = R_trial
                    norm = norm_trial
                    break
            lam_ls *= 0.5
        else:
            return W, dict(converged=False, iter=it, history=history,
                            reason='line_search_failed', norm=norm)

        history.append(norm)
        it += 1
        converged = (norm <= atol + rtol * norm0)
        if verbose:
            print(f"    Newton-full iter {it}: ‖R‖={norm:.3e}, λ={lam_ls:.3f}")

    return W, dict(converged=converged, iter=it, history=history, norm=norm)
