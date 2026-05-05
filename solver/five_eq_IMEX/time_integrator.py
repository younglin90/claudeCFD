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
