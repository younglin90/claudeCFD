"""IMEX time integrator — standard IMEX-RK Butcher tableau.

ARS(2,2,2) — Ascher-Ruuth-Spiteri 1997 / Pareschi-Russo 2005 Type II,
γ = 1 − 1/√2 ≈ 0.292893,  δ = 1 − 1/(2γ) = 1 − √2/2 / (1−1/√2)·... :

    explicit (Ã)            implicit (A)
    ----------              ------------
    0 |                     0 |
    γ | γ                   γ |  γ
    1 | δ   1-δ             1 |  1-γ   γ
    ---------------         ----------------
    b̃: δ   1-δ              b:  1-γ   γ

Sign convention:  dU/dt + L_E(W) + L_I(W) = 0,  so the stage equation is

    U(W^(i)) = U_i^*  −  Δt · a_ii · L_I(W^(i))

with the anchor

    U_i^* = U^n  −  Δt · Σ_{j<i} ( ã_ij · L_E(W^(j)) + a_ij · L_I(W^(j)) )

and final update

    U^{n+1} = U^n  −  Δt · Σ_i ( b̃_i · L_E(W^(i)) + b_i · L_I(W^(i)) ).

This is the form ChatGPT diagnosis §E recommends.  The earlier construction
(`target = U(W^(1))` + `final = average`) is non-standard and induces an
algorithmic round-off path; replaced here.
"""
from __future__ import annotations
import math
import os
import numpy as np

from .primitive import prim_to_cons_W, cons_to_prim_W
from .residual import explicit_residual, implicit_divergences
from .newton import newton_solve, newton_solve_full, newton_solve_schur
from .reconstruction import normalise_primitive_scheme
from .relaxation import relax_pressure, relax_pT

GAMMA = 1.0 - 1.0 / math.sqrt(2.0)         # γ ≈ 0.292893

# Ascher-Ruuth-Spiteri 1997 ARS(2,2,2) — original "L-stable, stiffly accurate"
# tableau, all weights non-negative.
#
#   explicit (Ã)            implicit (A)
#   ----------              ------------
#   0  |                    0  |
#   γ  | γ                  γ  |  γ                 (note: stage-2 explicit
#   1  | 0   1              1  |  1−γ   γ           anchor uses only L_E^(1))
#   ---------------         ----------------
#   b̃: 0   1   0           b:  1−γ   γ   0
#                                                    b̃_3 = 0, b_3 unused since
#   (Ascher–Ruuth–Spiteri 1997 §2.3 form 2).         stage 3 == final (sa).
#
# This avoids the Pareschi-Russo Type II δ = 1 − 1/(2γ) ≈ −0.7071 negative
# weight, which amplifies PE round-off across α-jumps (ChatGPT diag §E).

A_E = (
    (0.0, 0.0, 0.0),
    (GAMMA, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)
A_I = (
    (0.0, 0.0, 0.0),
    (0.0, GAMMA, 0.0),
    (0.0, 1.0 - GAMMA, GAMMA),
)
B_E = (0.0, 1.0, 0.0)
B_I = (0.0, 1.0 - GAMMA, GAMMA)

# Pareschi-Russo IMEX-SSP3(4,3,3) L-stable tableau.
#
# References:
#   Pareschi & Russo, J. Sci. Comput. 25 (2005), Table 6.
#   Boscheri & Pareschi, JCP 434 (2021), SSP3(4,3,3) option.
#
# Sign convention follows this file: dU/dt + L_E(W) + L_I(W) = 0.
_SSP3_A = 0.24169426078821
_SSP3_BETA = 0.06042356519705
_SSP3_ETA = 0.12915286960590
_SSP3_DELTA = 0.5 - _SSP3_BETA - _SSP3_ETA - _SSP3_A

SSP3_A_E = (
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.25, 0.25, 0.0),
)
SSP3_A_I = (
    (_SSP3_A, 0.0, 0.0, 0.0),
    (-_SSP3_A, _SSP3_A, 0.0, 0.0),
    (0.0, 1.0 - _SSP3_A, _SSP3_A, 0.0),
    (_SSP3_BETA, _SSP3_ETA, _SSP3_DELTA, _SSP3_A),
)
SSP3_B_E = (0.0, 1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0)
SSP3_B_I = SSP3_B_E


def _L_I(W, dx, bc_l, bc_r, u_inlet=None, p_inlet=None,
         eos1=None, eos2=None, rhie_chow=False, gamma_dt=None,
         imp_dissipation=0.0, imp_dissipation_form='biharmonic',
         imp_compact_lap_coeff=0.0,
         kapila_source=None):
    """Implicit residual L_I(W) as a 5-tuple of (N,) arrays."""
    N = W[0].shape[0]
    impl = implicit_divergences(W, dx, bc_l, bc_r,
                                u_inlet=u_inlet, p_inlet=p_inlet,
                                eos1=eos1, eos2=eos2,
                                rhie_chow=rhie_chow, gamma_dt=gamma_dt,
                                dissipation=imp_dissipation,
                                dissipation_form=imp_dissipation_form,
                                compact_lap_coeff=imp_compact_lap_coeff)
    L = [np.zeros(N) for _ in range(5)]
    L[2] = impl['grad_p']
    L[3] = impl['div_pu']
    if kapila_source is not None:
        L[4] = -kapila_source * impl['div_u']
    return tuple(L)


def _accumulate_target(U_n, dt, A_E_row, A_I_row, L_E_list, L_I_list):
    """U_i^* = U^n − Δt · Σ_{j<i}(ã_ij · L_E^(j) + a_ij · L_I^(j))."""
    out = list(np.asarray(c).copy() for c in U_n)
    for j in range(len(L_E_list)):
        coef = dt * A_E_row[j]
        if coef != 0.0:
            for k in range(5):
                out[k] = out[k] - coef * L_E_list[j][k]
        coef = dt * A_I_row[j]
        if coef != 0.0:
            for k in range(5):
                out[k] = out[k] - coef * L_I_list[j][k]
    return tuple(out)


def _finite_admissible_W(W):
    """Lightweight guard for post-update primitive recovery."""
    if any(not np.all(np.isfinite(c)) for c in W):
        return False
    a, T1, T2, _, p = W
    return (np.all((a > -1e-10) & (a < 1.0 + 1e-10))
            and np.all(T1 > 0.0) and np.all(T2 > 0.0)
            and np.all(p > 0.0))


def _relative_update_norm(U_ref, dU):
    """Max relative conservative update size, used to skip exact no-op steps."""
    vals = []
    for ref, delta in zip(U_ref, dU):
        scale = np.maximum(np.abs(ref), 1.0)
        vals.append(float(np.max(np.abs(delta) / scale)))
    return max(vals) if vals else 0.0


def _explicit_stage_operator(W_anchor, dt, eos1, eos2, dx, bc_l, bc_r, *,
                             u_inlet=None, p_inlet=None,
                             mixture_kind='kapila',
                             kapila_closure=False,
                             alpha_pure_tol=0.0,
                             alpha_scheme='upwind',
                             primitive_scheme='upwind',
                             energy_form='apec',
                             energy_alpha_pure_tol=1.0e-12,
                             face_thermo='acid',
                             positivity=True,
                             lo_flux='pe_preserving',
                             explicit_operator='residual'):
    """Return L_E(W_anchor) for RK stages.

    ``residual`` is the clean Phase-3 operator.  ``imex_ad_material`` reuses
    the current production material/advection update and converts its
    conservative one-step delta into an RK residual.  The latter is the
    relevant ablation for the current solver because it includes SLAU2,
    THINC-BVD/MSTACS alpha handling, density reconstruction, and rho-alpha
    preservation that are not present in the older Phase-3 residual path.
    """
    key = str(explicit_operator or 'residual').strip().lower().replace('-', '_')
    if key in {'residual', 'phase3', 'clean'}:
        L_E, face = explicit_residual(
            W_anchor, eos1, eos2, dx, bc_l, bc_r,
            alpha_scheme=alpha_scheme,
            primitive_scheme=primitive_scheme,
            energy_form=energy_form,
            energy_alpha_pure_tol=energy_alpha_pure_tol,
            face_thermo=face_thermo,
            kapila_closure=kapila_closure,
            positivity=positivity,
            dt=dt,
            force_lo=False,
            lo_flux=lo_flux)
        return L_E, face
    if key in {'imex_ad_material', 'material', 'production_material'}:
        from .imex_ad import _material_update
        U_anchor, _ = prim_to_cons_W(W_anchor, eos1, eos2)
        mat = _material_update(
            W_anchor, dt, eos1, eos2, dx, bc_l, bc_r,
            u_inlet=u_inlet, p_inlet=p_inlet,
            mixture_kind=mixture_kind,
            kapila_closure=kapila_closure,
            alpha_pure_tol=alpha_pure_tol,
            alpha_scheme=alpha_scheme,
            primitive_scheme=primitive_scheme,
            kapila_source_mode=os.environ.get(
                "FIVE_EQ_IMEX_KAPILA_SOURCE_MODE", "mixed_path"),
            material_energy_form=energy_form,
            return_aux=False)
        U_mat = tuple(np.asarray(c, dtype=float) for c in mat)
        L_E = tuple((np.asarray(U_anchor[k], dtype=float) - U_mat[k]) / dt
                    for k in range(5))
        return L_E, {}
    raise ValueError(
        "FIVE_EQ_IMEX_SSP3_EXPLICIT_OPERATOR must be 'residual' or "
        "'imex_ad_material'.")


def _max_adjacent_impedance_ratio(W, eos1, eos2):
    from .residual import _mixture_impedance
    Z = np.asarray(_mixture_impedance(W, eos1, eos2), dtype=float)
    if Z.size < 2:
        return 1.0
    ZL = np.maximum(Z[:-1], 1.0e-30)
    ZR = np.maximum(Z[1:], 1.0e-30)
    return float(np.max(np.maximum(ZL / ZR, ZR / ZL)))


def _pe_projection_allowed(W, eos1=None, eos2=None, *,
                           p_tol=1.0e-10, u_tol=1.0e-10,
                           liquid_gas_impedance_ratio=100.0):
    """Allow PE projection only for pressure/velocity-flat material contacts.

    The PE tangent projection is designed for stationary/moving material
    contacts.  In acoustic R/T cases it removes part of the physical pressure
    and velocity perturbation and can destabilize gas-gas interfaces.  Gate it
    with a strict flatness sensor; 02-A contacts have round-off level p/u
    gradients, while 07 acoustic pulses do not.
    """
    p = np.asarray(W[4], dtype=float)
    u = np.asarray(W[3], dtype=float)
    a = np.asarray(W[0], dtype=float)
    if p.size < 2:
        return True
    if a.size >= 2 and float(np.max(np.abs(np.diff(a)))) <= 1.0e-8:
        return False
    p_scale = max(float(np.max(np.abs(p))), 1.0)
    if eos1 is not None and eos2 is not None:
        try:
            from .sound_speed import phase_sound_speed_sq, mixture_sound_speed_sq
            a1, T1, T2, _u, pp = W
            rho1 = eos1.density(pp, T1)
            rho2 = eos2.density(pp, T2)
            c1_sq = phase_sound_speed_sq(eos1, rho1, T1)
            c2_sq = phase_sound_speed_sq(eos2, rho2, T2)
            c_ref = float(np.max(np.sqrt(np.maximum(
                mixture_sound_speed_sq(a1, rho1, c1_sq, rho2, c2_sq, kind='kapila'),
                1.0e-30))))
        except Exception:
            c_ref = max(float(np.max(np.abs(u))), 1.0)
    else:
        c_ref = max(float(np.max(np.abs(u))), 1.0)
    p_jump = float(np.max(np.abs(np.diff(p)))) / p_scale
    u_jump = float(np.max(np.abs(np.diff(u)))) / max(c_ref, 1.0)
    return p_jump <= p_tol and u_jump <= u_tol


def _project_conservative_target_to_pe(U_target, W_ref, eos1, eos2, *,
                                       alpha_pure_tol=0.0):
    """Put a p/u-flat material-contact target back on the PE manifold.

    Sharp-interface material advection updates phase masses and volume
    fraction.  The corresponding energy row must be thermodynamically
    consistent with the same uniform pressure; otherwise the following
    implicit acoustic stage is asked to solve a spurious EOS pressure jump.
    This projection is activated only by the same pressure/velocity-flat
    material-contact sensor used by the existing PE tangent correction.
    """
    if not _pe_projection_allowed(W_ref, eos1, eos2):
        return U_target

    q1 = np.maximum(np.asarray(U_target[0], dtype=float), 0.0)
    q2 = np.maximum(np.asarray(U_target[1], dtype=float), 0.0)
    alpha = np.clip(
        np.asarray(U_target[4], dtype=float),
        max(float(alpha_pure_tol), 1.0e-12),
        1.0 - max(float(alpha_pure_tol), 1.0e-12))
    rho = np.maximum(q1 + q2, 1.0e-30)
    u0 = float(np.mean(np.asarray(W_ref[3], dtype=float)))
    p0 = float(np.mean(np.asarray(W_ref[4], dtype=float)))
    if not (np.isfinite(u0) and np.isfinite(p0) and p0 > 0.0):
        return U_target

    rho1 = np.maximum(q1 / np.maximum(alpha, 1.0e-12), 1.0e-30)
    rho2 = np.maximum(q2 / np.maximum(1.0 - alpha, 1.0e-12), 1.0e-30)
    p_arr = np.full_like(alpha, p0)
    e1 = eos1.energy(rho1, p_arr)
    e2 = eos2.energy(rho2, p_arr)
    kinetic = 0.5 * rho * u0 * u0
    U_pe = (
        q1,
        q2,
        rho * u0,
        q1 * e1 + q2 * e2 + kinetic,
        alpha,
    )
    if all(np.all(np.isfinite(c)) for c in U_pe):
        return U_pe
    return U_target


def ars222_step(W_n, dt, eos1, eos2, dx, bc_l, bc_r, *,
                u_inlet=None, p_inlet=None,
                newton_kwargs=None,
                kapila_closure=False,
                rhie_chow=False,
                imp_dissipation=1.0,
                imp_dissipation_form='biharmonic',
                imp_compact_lap_coeff=0.0,
                pe_relax='pressure',  # 'none' | 'pressure' | 'pT'
                verbose=False):
    """Standard ARS(γ,γ,2) IMEX-RK step.

    Stage residual at internal stage i (a_ii > 0):

        R_i(W) = (U(W) − U_i^*) / (a_ii Δt) + L_I(W) = 0,
        a_ii Δt = γ Δt for both internal stages.

    `U_i^*` accumulates the explicit and previous implicit contributions.
    """
    newton_kwargs = newton_kwargs or {}
    U_n, _ = prim_to_cons_W(W_n, eos1, eos2)

    # ---- Stage 1: free node (W^(1) = W_n; A_I[1,1] = 0) ----
    W_stages = [tuple(np.asarray(c).copy() for c in W_n)]
    L_E_list = [None] * 3
    L_I_list = [None] * 3
    L_E_list[0], _ = explicit_residual(W_n, eos1, eos2, dx, bc_l, bc_r,
                                        kapila_closure=kapila_closure,
                                        positivity=True, dt=dt)
    L_I_list[0] = _L_I(W_n, dx, bc_l, bc_r,
                       u_inlet=u_inlet, p_inlet=p_inlet,
                       eos1=eos1, eos2=eos2,
                       rhie_chow=rhie_chow, gamma_dt=GAMMA * dt,
                       imp_dissipation=imp_dissipation,
                       imp_dissipation_form=imp_dissipation_form,
                       imp_compact_lap_coeff=imp_compact_lap_coeff)

    # ---- Stage 2: a_22 = γ, anchor U_2^* uses (ã_21, a_21) ----
    U_star_2 = _accumulate_target(U_n, dt, A_E[1], A_I[1],
                                   L_E_list[:1], L_I_list[:1])
    W2, info2 = newton_solve(W_n, U_star_2, GAMMA * dt, L_E_list[0],
                             eos1, eos2, dx, bc_l, bc_r,
                             u_inlet=u_inlet, p_inlet=p_inlet,
                             rhie_chow=rhie_chow,
                             imp_dissipation=imp_dissipation,
                             imp_dissipation_form=imp_dissipation_form,
                             imp_compact_lap_coeff=imp_compact_lap_coeff,
                             verbose=verbose, **newton_kwargs)
    W_stages.append(W2)
    L_E_list[1], _ = explicit_residual(W2, eos1, eos2, dx, bc_l, bc_r,
                                        kapila_closure=kapila_closure,
                                        positivity=True, dt=dt)
    L_I_list[1] = _L_I(W2, dx, bc_l, bc_r,
                       u_inlet=u_inlet, p_inlet=p_inlet,
                       eos1=eos1, eos2=eos2,
                       rhie_chow=rhie_chow, gamma_dt=GAMMA * dt,
                       imp_dissipation=imp_dissipation,
                       imp_dissipation_form=imp_dissipation_form,
                       imp_compact_lap_coeff=imp_compact_lap_coeff)

    # ---- Stage 3: a_33 = γ, anchor U_3^* uses 2 previous stages ----
    U_star_3 = _accumulate_target(U_n, dt, A_E[2], A_I[2],
                                   L_E_list[:2], L_I_list[:2])
    W3, info3 = newton_solve(W2, U_star_3, GAMMA * dt, L_E_list[1],
                             eos1, eos2, dx, bc_l, bc_r,
                             u_inlet=u_inlet, p_inlet=p_inlet,
                             rhie_chow=rhie_chow,
                             imp_dissipation=imp_dissipation,
                             imp_dissipation_form=imp_dissipation_form,
                             imp_compact_lap_coeff=imp_compact_lap_coeff,
                             verbose=verbose, **newton_kwargs)
    W_stages.append(W3)
    L_E_list[2], _ = explicit_residual(W3, eos1, eos2, dx, bc_l, bc_r,
                                        kapila_closure=kapila_closure,
                                        positivity=True, dt=dt)
    L_I_list[2] = _L_I(W3, dx, bc_l, bc_r,
                       u_inlet=u_inlet, p_inlet=p_inlet,
                       eos1=eos1, eos2=eos2,
                       rhie_chow=rhie_chow, gamma_dt=GAMMA * dt,
                       imp_dissipation=imp_dissipation,
                       imp_dissipation_form=imp_dissipation_form,
                       imp_compact_lap_coeff=imp_compact_lap_coeff)

    # ---- Final: U^{n+1} = U^n − Δt · Σ_i (b̃_i · L_E^(i) + b_i · L_I^(i)) ----
    U_next = list(np.asarray(c).copy() for c in U_n)
    for i in range(3):
        be = dt * B_E[i]; bi_ = dt * B_I[i]
        if be != 0.0:
            for k in range(5):
                U_next[k] = U_next[k] - be * L_E_list[i][k]
        if bi_ != 0.0:
            for k in range(5):
                U_next[k] = U_next[k] - bi_ * L_I_list[i][k]
    U_next = tuple(U_next)

    W_new = cons_to_prim_W(U_next, eos1, eos2,
                            T1_init=W3[1], T2_init=W3[2],
                            tol=1e-13, max_iter=50)

    # DC λ_k pressure-equilibrium projection (He & Tan 2024 Eq. A.19 idea):
    # The IMEX advance + primitive recovery preserves U exactly, but the W
    # may drift off the PE manifold due to face_state EOS round-off and
    # cons_to_prim Newton tolerance.  Project W back so p_1 = p_2 (always)
    # and optionally T_1 = T_2.  Conservative: U(W_relax) = U_next holds for
    # `pressure` mode (only T_k re-distributed); for 'pT' mode mass is also
    # rebalanced — slightly looser conservation but stronger PE damping.
    if pe_relax == 'pressure':
        W_new = relax_pressure(W_new, eos1, eos2)
    elif pe_relax == 'pT':
        W_new = relax_pT(W_new, eos1, eos2)
    elif pe_relax != 'none':
        raise ValueError(f"Unknown pe_relax='{pe_relax}'.")
    return W_new, dict(stage2=info2, stage3=info3,
                        L_E=L_E_list, L_I=L_I_list)


def imex_ssp3_step(W_n, dt, eos1, eos2, dx, bc_l, bc_r, *,
                   u_inlet=None, p_inlet=None, p_outlet=None,
                   newton_kwargs=None,
                   mixture_kind='kapila',
                   kapila_closure=False,
                   rhie_chow=False,
                   imp_dissipation=0.02,
                   imp_dissipation_form='biharmonic',
                   imp_compact_lap_coeff=0.0,
                   schur=True,
                   alpha_scheme='muscl',
                   primitive_scheme='upwind',
                   energy_form='apec',
                   energy_alpha_pure_tol=1.0e-12,
                   face_thermo='acid',
                   positivity=True,
                   lo_flux='pe_preserving',
                   kapila_acoustic_source=False,
                   alpha_pure_tol=1.0e-8,
                   explicit_operator='imex_ad_material',
                   stage_pe_relax='none',
                   pe_relax='none',
                   verbose=False):
    """Pareschi-Russo IMEX-SSP3(4,3,3) additive RK step.

    This is the stage-residual split form, not a wrapper around
    ``imex_ad_step``:

        U_i^* = U^n - dt * sum_{j<i}(aE_ij L_E(W_j)
                                    +aI_ij L_I(W_j))
        R_i(W_i) = (U(W_i) - U_i^*)/(aI_ii dt) + L_I(W_i) = 0

    The final update uses the published equal explicit/implicit weights.
    Limiters remain inside ``explicit_residual`` and are evaluated at every
    explicit RK stage, so this is the clean split path to test whether a real
    high-order IMEX time discretization improves the current composite solver.
    """
    newton_kwargs = newton_kwargs or {}
    primitive_scheme = normalise_primitive_scheme(primitive_scheme)
    U_n, _ = prim_to_cons_W(W_n, eos1, eos2)
    solver_fn = newton_solve_schur if schur else newton_solve

    W_stages = []
    L_E_list = []
    L_I_list = []
    info_stages = []

    for i in range(4):
        U_star = _accumulate_target(
            U_n, dt, SSP3_A_E[i], SSP3_A_I[i], L_E_list, L_I_list)

        kapila_source = None
        if kapila_closure and kapila_acoustic_source:
            from .source_d1 import D_K_kapila
            source_anchor = W_stages[-1] if W_stages else W_n
            kapila_source = D_K_kapila(source_anchor, eos1, eos2)

        if i == 0:
            guess = W_n
        else:
            guess = W_stages[-1]
        pe_target = _pe_projection_allowed(guess, eos1, eos2)
        U_star = _project_conservative_target_to_pe(
            U_star, guess, eos1, eos2, alpha_pure_tol=alpha_pure_tol)
        gamma_dt = SSP3_A_I[i][i] * dt
        if pe_target:
            W_i = cons_to_prim_W(
                U_star, eos1, eos2,
                T1_init=guess[1], T2_init=guess[2],
                alpha_pure_tol=alpha_pure_tol,
                tol=1e-13,
                max_iter=50)
            info_i = dict(
                converged=True,
                iter=0,
                history=[0.0],
                solver='pe_target_recovery')
        else:
            W_i, info_i = solver_fn(
                guess, U_star, gamma_dt,
                tuple(np.zeros_like(W_n[0]) for _ in range(5)),
                eos1, eos2, dx, bc_l, bc_r,
                u_inlet=u_inlet, p_inlet=p_inlet,
                alpha_source_explicit=(kapila_source is None),
                kapila_source=kapila_source,
                rhie_chow=rhie_chow,
                imp_dissipation=imp_dissipation,
                imp_dissipation_form=imp_dissipation_form,
                imp_compact_lap_coeff=imp_compact_lap_coeff,
                include_explicit_residual=False,
                pe_correct=False,
                verbose=verbose, **newton_kwargs)
        W_stages.append(W_i)
        if stage_pe_relax == 'pressure':
            W_i = relax_pressure(W_i, eos1, eos2)
            W_stages[-1] = W_i
        elif stage_pe_relax == 'pT':
            W_i = relax_pT(W_i, eos1, eos2)
            W_stages[-1] = W_i
        elif stage_pe_relax != 'none':
            raise ValueError(f"Unknown stage_pe_relax='{stage_pe_relax}'.")
        info_stages.append(info_i)

        L_E_i, _ = _explicit_stage_operator(
            W_i, dt, eos1, eos2, dx, bc_l, bc_r,
            u_inlet=u_inlet, p_inlet=p_inlet,
            mixture_kind=mixture_kind,
            kapila_closure=kapila_closure,
            alpha_pure_tol=alpha_pure_tol,
            alpha_scheme=alpha_scheme,
            primitive_scheme=primitive_scheme,
            energy_form=energy_form,
            energy_alpha_pure_tol=energy_alpha_pure_tol,
            face_thermo=face_thermo,
            positivity=positivity,
            lo_flux=lo_flux,
            explicit_operator=explicit_operator)
        L_I_i = _L_I(
            W_i, dx, bc_l, bc_r,
            u_inlet=u_inlet, p_inlet=p_inlet,
            eos1=eos1, eos2=eos2,
            rhie_chow=rhie_chow,
            gamma_dt=gamma_dt,
            imp_dissipation=imp_dissipation,
            imp_dissipation_form=imp_dissipation_form,
            imp_compact_lap_coeff=imp_compact_lap_coeff,
            kapila_source=kapila_source)
        L_E_list.append(L_E_i)
        L_I_list.append(L_I_i)

    U_next = list(np.asarray(c).copy() for c in U_n)
    for i in range(4):
        be = dt * SSP3_B_E[i]
        bi = dt * SSP3_B_I[i]
        if be != 0.0:
            for k in range(5):
                U_next[k] = U_next[k] - be * L_E_list[i][k]
        if bi != 0.0:
            for k in range(5):
                U_next[k] = U_next[k] - bi * L_I_list[i][k]
    U_next = tuple(U_next)
    U_next = _project_conservative_target_to_pe(
        U_next, W_stages[-1], eos1, eos2, alpha_pure_tol=alpha_pure_tol)

    W_new = cons_to_prim_W(
        U_next, eos1, eos2,
        T1_init=W_stages[-1][1],
        T2_init=W_stages[-1][2],
        alpha_pure_tol=alpha_pure_tol,
        tol=1e-13,
        max_iter=50)
    if pe_relax == 'pressure':
        W_new = relax_pressure(W_new, eos1, eos2)
    elif pe_relax == 'pT':
        W_new = relax_pT(W_new, eos1, eos2)
    elif pe_relax != 'none':
        raise ValueError(f"Unknown pe_relax='{pe_relax}'.")

    return W_new, dict(
        time_integrator='imex_ssp3_433_stage_residual',
        tableau='Pareschi-Russo IMEX-SSP3(4,3,3)',
        stages=info_stages,
        L_E=L_E_list,
        L_I=L_I_list,
        primitive_scheme=primitive_scheme,
        alpha_scheme=alpha_scheme)


def _blend_conservative_states(W_a, W_b, theta, eos1, eos2, *,
                               alpha_pure_tol=0.0,
                               pe_reference=None):
    """Convex blend in conservative variables and recover primitives."""
    U_a, _ = prim_to_cons_W(W_a, eos1, eos2)
    U_b, _ = prim_to_cons_W(W_b, eos1, eos2)
    U = tuple(theta * np.asarray(U_a[k]) + (1.0 - theta) * np.asarray(U_b[k])
              for k in range(5))
    if pe_reference is not None:
        U = _project_conservative_target_to_pe(
            U, pe_reference, eos1, eos2, alpha_pure_tol=alpha_pure_tol)
    return cons_to_prim_W(
        U, eos1, eos2,
        T1_init=theta * np.asarray(W_a[1]) + (1.0 - theta) * np.asarray(W_b[1]),
        T2_init=theta * np.asarray(W_a[2]) + (1.0 - theta) * np.asarray(W_b[2]),
        alpha_pure_tol=alpha_pure_tol,
        tol=1e-13,
        max_iter=50)


def _imex_ad_ssp3_transport_acoustic_cn(
        W_n, dt, eos1, eos2, dx, bc_l, bc_r, *,
        u_inlet=None, p_inlet=None, p_outlet=None,
        alpha_inlet=None, T1_inlet=None, T2_inlet=None,
        mixture_kind='kapila',
        kapila_closure=False,
        alpha_pure_tol=1.0e-8,
        alpha_scheme='muscl',
        primitive_scheme='upwind',
        pressure_closure='regime_auto'):
    """SSP3 material transport followed by one production acoustic CN solve."""
    from .imex_ad import (
        _material_update,
        _normalise_pressure_closure,
        _pressure_jump_high_to_low_impedance,
        _solve_acoustic_ad,
        _solve_acoustic_energy_ad,
        _phase_acoustic,
        _acoustic_faces_np,
        _recover_pressure_from_total_energy,
        _compressive_pressure_mask,
        _pure_material_cell_mask,
        _entropy_pressure_estimate,
        _regularize_near_vacuum_velocity,
        _primitive_lmp_effective_mode,
        _primitive_lmp_enabled,
        _primitive_lmp_clip,
        _primitive_global_pressure_clip,
        _primitive_led_filter,
        _env_enabled,
    )

    primitive_scheme = normalise_primitive_scheme(primitive_scheme)
    pressure_closure = _normalise_pressure_closure(pressure_closure)
    energy_momentum_refresh = pressure_closure == 'implicit_energy_momentum'
    pure_tol_auto = max(float(alpha_pure_tol), np.finfo(float).eps ** 0.25)
    alpha_now = np.asarray(W_n[0], dtype=float)
    single_phase_limit = (
        float(np.min(alpha_now)) >= 1.0 - pure_tol_auto
        or float(np.max(alpha_now)) <= pure_tol_auto
    )
    if single_phase_limit:
        from .imex_ad import imex_ad_step
        W_new, info = imex_ad_step(
            W_n, dt, eos1, eos2, dx, bc_l, bc_r,
            u_inlet=u_inlet, p_inlet=p_inlet, p_outlet=p_outlet,
            alpha_inlet=alpha_inlet, T1_inlet=T1_inlet, T2_inlet=T2_inlet,
            mixture_kind=mixture_kind,
            kapila_closure=kapila_closure,
            alpha_pure_tol=alpha_pure_tol,
            alpha_scheme=alpha_scheme,
            primitive_scheme=primitive_scheme,
            pressure_closure=pressure_closure)
        info = dict(info)
        info['time_integrator'] = 'imex_ad_ssp3_single_phase_acoustic_cn'
        info['ssp3_single_phase_limit'] = True
        return W_new, info
    p_now = np.asarray(W_n[4], dtype=float)
    if p_now.size > 1 and np.all(np.isfinite(p_now)):
        p_den = np.maximum(np.maximum(np.abs(p_now[:-1]), np.abs(p_now[1:])), 1.0)
        max_rel_p_jump = float(np.max(np.abs(np.diff(p_now)) / p_den))
    else:
        max_rel_p_jump = 0.0
    if max_rel_p_jump > np.finfo(float).eps ** 0.25:
        from .imex_ad import imex_ad_step
        W_new, info = imex_ad_step(
            W_n, dt, eos1, eos2, dx, bc_l, bc_r,
            u_inlet=u_inlet, p_inlet=p_inlet, p_outlet=p_outlet,
            alpha_inlet=alpha_inlet, T1_inlet=T1_inlet, T2_inlet=T2_inlet,
            mixture_kind=mixture_kind,
            kapila_closure=kapila_closure,
            alpha_pure_tol=alpha_pure_tol,
            alpha_scheme=alpha_scheme,
            primitive_scheme=primitive_scheme,
            pressure_closure=pressure_closure)
        info = dict(info)
        info['time_integrator'] = 'imex_ad_ssp3_pressure_jump_acoustic_cn'
        info['ssp3_pressure_jump_limit'] = True
        info['max_rel_p_jump'] = max_rel_p_jump
        return W_new, info
    if pressure_closure == 'regime_auto':
        has_immiscible_interface = (
            float(np.min(alpha_now)) <= pure_tol_auto
            and float(np.max(alpha_now)) >= 1.0 - pure_tol_auto
        )
        if has_immiscible_interface:
            if _pressure_jump_high_to_low_impedance(
                    W_n, eos1, eos2, mixture_kind=mixture_kind,
                    alpha_pure_tol=alpha_pure_tol):
                pressure_closure = 'implicit_energy'
                energy_momentum_refresh = True
            else:
                pressure_closure = 'pressure_work_consistent'
        else:
            pressure_closure = 'implicit_energy'

    kapila_source_mode = os.environ.get(
        "FIVE_EQ_IMEX_KAPILA_SOURCE_MODE", "mixed_path")
    kapila_source_mode = str(kapila_source_mode).strip().lower().replace("-", "_")
    if kapila_source_mode in ('mixture_path', 'true_mixture_path'):
        kapila_source_mode = 'mixed_path'
    elif kapila_source_mode in ('mixed', 'true_mixture', 'mixture_trapezoid'):
        kapila_source_mode = 'mixed_trapezoid'
    elif kapila_source_mode in ('immiscible', 'interface_preserving'):
        kapila_source_mode = 'immiscible_trapezoid'
    material_energy_form = 'apec' if pressure_closure == 'apec_pe' else 'allaire'

    def material_euler(W_anchor):
        U_mat = _material_update(
            W_anchor, dt, eos1, eos2, dx, bc_l, bc_r,
            u_inlet=u_inlet, p_inlet=p_inlet, p_outlet=p_outlet,
            alpha_inlet=alpha_inlet, T1_inlet=T1_inlet, T2_inlet=T2_inlet,
            mixture_kind=mixture_kind,
            kapila_closure=kapila_closure,
            alpha_pure_tol=alpha_pure_tol,
            alpha_scheme=alpha_scheme,
            primitive_scheme=primitive_scheme,
            kapila_source_mode=kapila_source_mode,
            material_energy_form=material_energy_form,
            return_aux=False)
        U_mat = tuple(np.asarray(c, dtype=float) for c in U_mat)
        U_mat = _project_conservative_target_to_pe(
            U_mat, W_anchor, eos1, eos2, alpha_pure_tol=alpha_pure_tol)
        return cons_to_prim_W(
            U_mat, eos1, eos2,
            T1_init=W_anchor[1],
            T2_init=W_anchor[2],
            alpha_pure_tol=alpha_pure_tol,
            tol=1e-13,
            max_iter=50)

    W1 = material_euler(W_n)
    W2_euler = material_euler(W1)
    pe_ref = W_n if _pe_projection_allowed(W_n, eos1, eos2) else None
    W2 = _blend_conservative_states(
        W_n, W2_euler, 0.75, eos1, eos2,
        alpha_pure_tol=alpha_pure_tol,
        pe_reference=pe_ref)
    W3_euler = material_euler(W2)
    W_adv = _blend_conservative_states(
        W_n, W3_euler, 1.0 / 3.0, eos1, eos2,
        alpha_pure_tol=alpha_pure_tol,
        pe_reference=pe_ref)

    U_adv, _ = prim_to_cons_W(W_adv, eos1, eos2)
    q1_new = np.asarray(U_adv[0], dtype=float)
    q2_new = np.asarray(U_adv[1], dtype=float)
    m_adv = np.asarray(U_adv[2], dtype=float)
    rhoE_new = np.asarray(U_adv[3], dtype=float)
    alpha_new = np.clip(np.asarray(U_adv[4], dtype=float), 1.0e-12, 1.0 - 1.0e-12)

    if pressure_closure in ('implicit_energy', 'implicit_energy_momentum', 'apec_pe'):
        u_new, p_new = _solve_acoustic_energy_ad(
            W_n, q1_new, q2_new, m_adv, rhoE_new, alpha_new, dt,
            eos1, eos2, dx, bc_l, bc_r,
            u_inlet=u_inlet, p_inlet=p_inlet, p_outlet=p_outlet,
            mixture_kind=mixture_kind,
            alpha_pure_tol=alpha_pure_tol,
            primitive_scheme=primitive_scheme,
            momentum_refresh=energy_momentum_refresh)
        _, _, Z = _phase_acoustic(
            W_n, eos1, eos2, mixture_kind=mixture_kind,
            alpha_pure_tol=alpha_pure_tol)
        p_f, u_f = _acoustic_faces_np(
            u_new, p_new, Z, bc_l, bc_r,
            u_inlet=u_inlet, p_inlet=p_inlet, p_outlet=p_outlet)
        rhoE_new = rhoE_new - dt * (
            p_f[1:] * u_f[1:] - p_f[:-1] * u_f[:-1]) / dx
    else:
        u_new, p_new = _solve_acoustic_ad(
            W_n, q1_new, q2_new, m_adv, alpha_new, dt,
            eos1, eos2, dx, bc_l, bc_r,
            u_inlet=u_inlet, p_inlet=p_inlet, p_outlet=p_outlet,
            mixture_kind=mixture_kind,
            alpha_pure_tol=alpha_pure_tol,
            primitive_scheme=primitive_scheme)
        if pressure_closure == 'compressive_recovery':
            recovery_mask = _compressive_pressure_mask(W_n)
            if alpha_pure_tol > 0.0:
                pure_tol = max(float(alpha_pure_tol), np.finfo(float).eps ** 0.25)
                recovery_mask = recovery_mask & ~_pure_material_cell_mask(
                    W_n[0], pure_tol)
            if np.any(recovery_mask):
                p_recovered = _recover_pressure_from_total_energy(
                    q1_new, q2_new, rhoE_new, alpha_new, u_new, p_new,
                    eos1, eos2)
                p_new = np.where(recovery_mask, p_recovered, p_new)
        elif pressure_closure == 'pressure_work_consistent':
            _, _, Z = _phase_acoustic(
                W_n, eos1, eos2, mixture_kind=mixture_kind,
                alpha_pure_tol=alpha_pure_tol)
            p_f, u_f = _acoustic_faces_np(
                u_new, p_new, Z, bc_l, bc_r,
                u_inlet=u_inlet, p_inlet=p_inlet, p_outlet=p_outlet)
            rhoE_new = rhoE_new - dt * (
                p_f[1:] * u_f[1:] - p_f[:-1] * u_f[:-1]) / dx
            if _env_enabled("FIVE_EQ_IMEX_PW_PURE_SHOCK_RECOVERY", "1"):
                recovery_mask = _compressive_pressure_mask(W_n)
                pure_tol = max(float(alpha_pure_tol), np.finfo(float).eps ** 0.25)
                recovery_mask = recovery_mask & ~_pure_material_cell_mask(
                    W_n[0], pure_tol)
                if np.any(recovery_mask):
                    p_recovered = _recover_pressure_from_total_energy(
                        q1_new, q2_new, rhoE_new, alpha_new, u_new, p_new,
                        eos1, eos2)
                    p_new = np.where(recovery_mask, p_recovered, p_new)
        elif pressure_closure == 'dual_entropy':
            entropy_mask = _compressive_pressure_mask(W_n)
            p_entropy = _entropy_pressure_estimate(
                W_n, q1_new, q2_new, alpha_new, p_new, eos1, eos2)
            p_new = np.where(entropy_mask, p_entropy, p_new)

    u_new, vacuum_velocity_mask = _regularize_near_vacuum_velocity(
        W_n, q1_new, q2_new, u_new, p_new, eos1, eos2,
        mixture_kind=mixture_kind,
        alpha_pure_tol=alpha_pure_tol,
        bc_l=bc_l, bc_r=bc_r)
    lmp_mode = _primitive_lmp_effective_mode(W_n)
    if _primitive_lmp_enabled(lmp_mode):
        if lmp_mode in {"stencil", "local_stencil", "old"}:
            u_new, p_new = _primitive_lmp_clip(W_n, u_new, p_new, bc_l, bc_r)
        elif lmp_mode in {"global_p", "pressure_global", "p_global"}:
            u_new, p_new = _primitive_global_pressure_clip(W_n, u_new, p_new)
        else:
            u_new, p_new = _primitive_led_filter(
                u_new, p_new, bc_l, bc_r, mode=lmp_mode)

    rho1_new = q1_new / np.maximum(alpha_new, 1.0e-12)
    rho2_new = q2_new / np.maximum(1.0 - alpha_new, 1.0e-12)
    e1_new = eos1.energy(rho1_new, p_new)
    e2_new = eos2.energy(rho2_new, p_new)
    T1_new = eos1.temperature(rho1_new, e1_new)
    T2_new = eos2.temperature(rho2_new, e2_new)
    return (
        alpha_new, T1_new, T2_new, u_new, p_new
    ), {
        'time_integrator': 'imex_ad_ssp3_transport_acoustic_cn',
        'scheme': 'ssp3_material_transport_plus_cn_acoustic',
        'pressure_closure': pressure_closure,
        'primitive_scheme': primitive_scheme,
        'alpha_scheme': alpha_scheme,
        'kapila_source_mode': kapila_source_mode,
        'material_energy_form': material_energy_form,
        'vacuum_velocity_cells': int(np.count_nonzero(vacuum_velocity_mask)),
    }


def imex_ad_ssp3_step(W_n, dt, eos1, eos2, dx, bc_l, bc_r, *,
                      u_inlet=None, p_inlet=None, p_outlet=None,
                      alpha_inlet=None, T1_inlet=None, T2_inlet=None,
                      mixture_kind='kapila',
                      kapila_closure=False,
                      alpha_pure_tol=1.0e-8,
                      alpha_scheme='muscl',
                      primitive_scheme='upwind',
                      pressure_closure='regime_auto',
                      **_unused):
    scope = os.environ.get("FIVE_EQ_IMEX_SSP3_SCOPE", "transport_acoustic_cn")
    scope = str(scope).strip().lower().replace("-", "_")
    if scope in {"transport", "material", "transport_acoustic_cn",
                 "material_acoustic_cn", "single_acoustic"}:
        return _imex_ad_ssp3_transport_acoustic_cn(
            W_n, dt, eos1, eos2, dx, bc_l, bc_r,
            u_inlet=u_inlet, p_inlet=p_inlet, p_outlet=p_outlet,
            alpha_inlet=alpha_inlet, T1_inlet=T1_inlet, T2_inlet=T2_inlet,
            mixture_kind=mixture_kind,
            kapila_closure=kapila_closure,
            alpha_pure_tol=alpha_pure_tol,
            alpha_scheme=alpha_scheme,
            primitive_scheme=primitive_scheme,
            pressure_closure=pressure_closure)
    if scope not in {"full_step", "shu_osher_full", "full_imex"}:
        raise ValueError(
            "FIVE_EQ_IMEX_SSP3_SCOPE must be 'transport_acoustic_cn' or "
            "'full_step'.")
    """Shu-Osher SSP3 composition of the production all-Mach IMEX step.

    The stage map ``G`` is the same numerically stable all-Mach update used by
    the accepted solver path: material/advection fluxes plus the implicit
    acoustic pressure solve.  Convex combinations are done in conservative
    variables, preserving the SSP/TVD contract of the stage map.  On p/u-flat
    material-contact states, the conservative blends are projected back onto
    the pressure-equilibrium manifold so that SSP sub-staging does not create
    an EOS pressure defect.
    """
    from .imex_ad import imex_ad_step

    primitive_scheme = normalise_primitive_scheme(primitive_scheme)

    def G(W):
        W_next, info = imex_ad_step(
            W, dt, eos1, eos2, dx, bc_l, bc_r,
            u_inlet=u_inlet, p_inlet=p_inlet, p_outlet=p_outlet,
            alpha_inlet=alpha_inlet, T1_inlet=T1_inlet, T2_inlet=T2_inlet,
            mixture_kind=mixture_kind,
            kapila_closure=kapila_closure,
            alpha_pure_tol=alpha_pure_tol,
            alpha_scheme=alpha_scheme,
            primitive_scheme=primitive_scheme,
            pressure_closure=pressure_closure)
        return W_next, info

    pe_ref = W_n if _pe_projection_allowed(W_n, eos1, eos2) else None
    W1, info1 = G(W_n)
    W2_star, info2 = G(W1)
    W2 = _blend_conservative_states(
        W_n, W2_star, 0.75, eos1, eos2,
        alpha_pure_tol=alpha_pure_tol,
        pe_reference=pe_ref)
    W3_star, info3 = G(W2)
    W_new = _blend_conservative_states(
        W_n, W3_star, 1.0 / 3.0, eos1, eos2,
        alpha_pure_tol=alpha_pure_tol,
        pe_reference=pe_ref)
    return W_new, dict(
        time_integrator='imex_ad_ssp3_shu_osher',
        tableau='Shu-Osher SSPRK3 conservative composition of imex_ad_step',
        stages=[info1, info2, info3],
        primitive_scheme=primitive_scheme,
        alpha_scheme=alpha_scheme)


def strang_step(W_n, dt, eos1, eos2, dx, bc_l, bc_r, **kwargs):
    """Strang fallback: explicit-only (debug)."""
    L_E, _ = explicit_residual(W_n, eos1, eos2, dx, bc_l, bc_r,
                               positivity=True, dt=dt)
    U_n, _ = prim_to_cons_W(W_n, eos1, eos2)
    U_next = tuple(U_n[k] - dt * L_E[k] for k in range(5))
    return cons_to_prim_W(U_next, eos1, eos2,
                          T1_init=W_n[1], T2_init=W_n[2]), {}


def be_full_step(W_n, dt, eos1, eos2, dx, bc_l, bc_r, *,
                 u_inlet=None, p_inlet=None,
                 newton_kwargs=None,
                 kapila_closure=False,
                 verbose=False):
    """Fully-implicit Backward Euler — mass / momentum / energy / α / ∇p / p·u
    all inside one Newton solve.  Equivalent in form to He2024 imex_5n; the
    only choice that achieves machine-ε PE preservation across sharp α-jumps
    for arbitrary EOS (verified on 02-A NASG).

    The user-task spec ARS222 split is preserved as `ars222_step`; this is
    an additional, more-stable integrator for stiff interface cases.
    """
    newton_kwargs = newton_kwargs or {}
    U_n, _ = prim_to_cons_W(W_n, eos1, eos2)
    W_new, info = newton_solve_full(W_n, U_n, dt, eos1, eos2, dx, bc_l, bc_r,
                                     u_inlet=u_inlet, p_inlet=p_inlet,
                                     kapila_closure=kapila_closure,
                                     verbose=verbose, **newton_kwargs)
    return W_new, dict(stage=info)


def split_step(W_n, dt, eos1, eos2, dx, bc_l, bc_r, *,
               u_inlet=None, p_inlet=None,
               newton_kwargs=None,
               kapila_closure=False,
               rhie_chow=False,
               imp_dissipation=1.0,
               imp_dissipation_form='biharmonic',
               imp_compact_lap_coeff=0.0,
               pe_correct=False,
               adv_substeps=4,
               verbose=False):
    """Strang-like split (ChatGPT v3 §6.3): explicit advection sub-cycle,
    then implicit pressure projection.

    Stage A — advection sub-cycle (explicit, multi-step Euler, stable for
              material CFL <= 1):
        for s in range(adv_substeps):
            U_a = U_a − (dt/adv_substeps) · L_E(W_a)
            W_a = cons_to_prim_W(U_a)

    Stage B — implicit pressure projection (Backward-Euler on L_I only):
        Solve R(W) = (U(W) − U_a) / dt + L_I(W) = 0
        Then U_next = U(W^{n+1}) = U_a − dt · L_I(W^{n+1})

    Compared to be1, advection is now sub-cycled with material CFL bound
    (no forward-Euler |λ|>1 amplification), and pressure block remains
    implicit (acoustic-stable).
    """
    newton_kwargs = newton_kwargs or {}
    U_a, _ = prim_to_cons_W(W_n, eos1, eos2)
    W_a = tuple(np.asarray(c, dtype=float).copy() for c in W_n)
    dt_sub = dt / max(adv_substeps, 1)
    for s in range(adv_substeps):
        L_E_s, _ = explicit_residual(W_a, eos1, eos2, dx, bc_l, bc_r,
                                      kapila_closure=kapila_closure,
                                      positivity=True, dt=dt_sub)
        U_a = tuple(U_a[k] - dt_sub * L_E_s[k] for k in range(5))
        W_a = cons_to_prim_W(U_a, eos1, eos2,
                              T1_init=W_a[1], T2_init=W_a[2],
                              tol=1e-13, max_iter=50)

    # Stage B — implicit pressure projection
    L_E_dummy = tuple(np.zeros_like(W_n[0]) for _ in range(5))
    W_imp, info = newton_solve(W_a, U_a, dt, L_E_dummy,
                                eos1, eos2, dx, bc_l, bc_r,
                                u_inlet=u_inlet, p_inlet=p_inlet,
                                rhie_chow=rhie_chow,
                                imp_dissipation=imp_dissipation,
                                imp_dissipation_form=imp_dissipation_form,
                                imp_compact_lap_coeff=imp_compact_lap_coeff,
                                pe_correct=pe_correct,
                                verbose=verbose, **newton_kwargs)
    return W_imp, dict(stage_A=W_a, stage_B=info)


def be1_step(W_n, dt, eos1, eos2, dx, bc_l, bc_r, *,
             u_inlet=None, p_inlet=None,
             newton_kwargs=None,
             kapila_closure=False,
             rhie_chow=False,
             imp_dissipation=0.02,
             imp_dissipation_form='biharmonic',
             imp_compact_lap_coeff=0.0,
             schur=True,
             pe_project_explicit=True,
             pe_projection_mode='always',
             explicit_force_lo=True,
             alpha_scheme='muscl',
             primitive_scheme='upwind',
             energy_form='apec',
             energy_alpha_pure_tol=1.0e-12,
             face_thermo='acid',
             positivity=True,
             lo_flux='pe_preserving',
             final_update_backtracking=True,
             final_update_backtracking_steps=12,
             zero_update_tol=1e-13,
             pure_branch=False,
             alpha_pure_tol=1.0e-8,
             implicit_include_explicit_residual=False,
             kapila_acoustic_source=False,
             pe_correct=False,
             verbose=False):
    """Single-stage Backward-Euler (Abgrall-consistent: anchor evaluated
    only once at W_n, eliminates stage-2 round-off drift seen in ARS222).

    Solves
        R(W) = (U(W) − U^n)/Δt + L_I(W) = 0
    by Newton, then accumulates explicit operator at the *single* anchor:
        U^{n+1} = U^n − Δt · (L_E(W^n) + L_I(W^{n+1}))

    PE-preserving by construction in the user spec sense — both L_E and L_I
    are evaluated on a single, deterministic anchor pair (W^n for explicit,
    W^{n+1} for implicit).  Equivalent to a 1-stage L-stable IMEX scheme.
    """
    newton_kwargs = newton_kwargs or {}
    primitive_scheme = normalise_primitive_scheme(primitive_scheme)
    U_n, _ = prim_to_cons_W(W_n, eos1, eos2)
    kapila_source = None
    if kapila_closure and kapila_acoustic_source:
        from .source_d1 import D_K_kapila
        kapila_source = D_K_kapila(W_n, eos1, eos2)
    L_E1, _ = explicit_residual(W_n, eos1, eos2, dx, bc_l, bc_r,
                                 alpha_scheme=alpha_scheme,
                                 primitive_scheme=primitive_scheme,
                                 energy_form=energy_form,
                                 energy_alpha_pure_tol=energy_alpha_pure_tol,
                                 face_thermo=face_thermo,
                                 kapila_closure=kapila_closure,
                                 positivity=positivity, dt=dt,
                                 force_lo=explicit_force_lo,
                                 lo_flux=lo_flux,
                                 kapila_source_in_implicit=(kapila_source is not None))
    solver_fn = newton_solve_schur if schur else newton_solve
    W_imp, info = solver_fn(W_n, U_n, dt, L_E1,
                            eos1, eos2, dx, bc_l, bc_r,
                            u_inlet=u_inlet, p_inlet=p_inlet,
                            alpha_source_explicit=(kapila_source is None),
                            kapila_source=kapila_source,
                            rhie_chow=rhie_chow,
                            imp_dissipation=imp_dissipation,
                            imp_dissipation_form=imp_dissipation_form,
                            imp_compact_lap_coeff=imp_compact_lap_coeff,
                            include_explicit_residual=implicit_include_explicit_residual,
                            # Keep Newton objective conservative; applying
                            # PE correction inside residual can destabilize
                            # the linearized amplification on alpha-jumps.
                            pe_correct=False,
                            verbose=verbose, **newton_kwargs)
    L_I1 = _L_I(W_imp, dx, bc_l, bc_r, u_inlet=u_inlet, p_inlet=p_inlet,
                eos1=eos1, eos2=eos2,
                rhie_chow=rhie_chow, gamma_dt=dt,
                imp_dissipation=imp_dissipation,
                imp_dissipation_form=imp_dissipation_form,
                imp_compact_lap_coeff=imp_compact_lap_coeff,
                kapila_source=kapila_source)
    L_total = tuple(L_E1[k] + L_I1[k] for k in range(5))
    if pe_project_explicit and str(pe_projection_mode).endswith('_explicit'):
        # Acoustic cases need the implicit pressure block to propagate physical
        # p/u waves.  These diagnostic modes project only the explicit material
        # transport residual and then add the implicit acoustic residual back.
        if _pe_projection_allowed(W_n, eos1, eos2):
            from .pe_correction import apply_pe_tangent_projection
            base_mode = str(pe_projection_mode)[:-len('_explicit')]
            L_E1, _ = apply_pe_tangent_projection(
                L_E1, W_n, eos1, eos2, mode=base_mode)
            L_total = tuple(L_E1[k] + L_I1[k] for k in range(5))
    elif pe_project_explicit:
        # Project the combined residual at the implicit state.
        from .pe_correction import apply_pe_tangent_projection
        L_total, _ = apply_pe_tangent_projection(
            L_total, W_imp, eos1, eos2, mode=pe_projection_mode)
    elif pe_correct:
        # Legacy energy-only PE correction on the combined residual.
        from .pe_correction import apply_pe_correction
        L_total, _ = apply_pe_correction(L_total, W_imp, eos1, eos2)
    dU_total = tuple(dt * L_total[k] for k in range(5))
    rel_update = _relative_update_norm(U_n, dU_total)
    if rel_update <= zero_update_tol:
        return tuple(np.asarray(c).copy() for c in W_n), dict(
            stage=info, L_E=L_E1, L_I=L_I1, final_theta=1.0,
            zero_update=True, rel_update=rel_update,
            primitive_scheme=primitive_scheme)
    theta = 1.0
    max_trials = final_update_backtracking_steps if final_update_backtracking else 1
    last_error = None
    for _ in range(max_trials):
        U_next = tuple(U_n[k] - theta * dt * L_total[k] for k in range(5))
        try:
            W_new = cons_to_prim_W(U_next, eos1, eos2,
                                    T1_init=W_imp[1], T2_init=W_imp[2],
                                    alpha_pure_tol=(alpha_pure_tol
                                                    if pure_branch else 0.0))
            if _finite_admissible_W(W_new):
                return W_new, dict(stage=info, L_E=L_E1, L_I=L_I1,
                                   final_theta=theta,
                                   primitive_scheme=primitive_scheme)
            last_error = 'non_finite_or_inadmissible_recovery'
        except Exception as exc:
            last_error = f'{type(exc).__name__}: {exc}'
        theta *= 0.5

    # Keep the march finite for diagnostics; the reported theta marks fallback.
    return tuple(np.asarray(c).copy() for c in W_n), dict(
        stage=info, L_E=L_E1, L_I=L_I1, final_theta=0.0,
        final_update_error=last_error,
        primitive_scheme=primitive_scheme)
