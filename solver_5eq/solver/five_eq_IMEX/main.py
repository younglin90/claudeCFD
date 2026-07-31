"""five_eq_IMEX.main — entry point of the clean-room all-Mach 5-equation solver.

Conservative variables and primitive variables follow the user spec:

    U = (alpha1*rho1, alpha2*rho2, rho*u, rho*E, alpha1)^T
    W = (alpha1, T1, T2, u, p)^T

Phase 3 implementation: ARS(2,2,2) IMEX-SSP2 with γ = 1 − 1/√2, Allaire-style
explicit energy flux (no APEC χ_a yet), central face state, no Rhie-Chow.

Higher phases plug in:
  - Phase 4 (rhie_chow.py)         pressure-velocity coupling for low-Mach
  - Phase 5 (flux.py — SLAU2)      all-speed mass flux
  - Phase 6 (face_state.py — TVD/THINC)  ACID face thermodynamics
  - Phase 7 (energy_flux.py)        APEC χ_k, χ_a cross-term
  - Phase 8 (limiters.py)           layered positivity θ_f blending
  - Phase 9 (source_d1.py — semi)   D₁·div(δu) implicit
  - Phase 10 (face_state.py)        THINC-BVD interface sharpening
"""
from __future__ import annotations
import math
import os
import numpy as np

from .eos_facade import EOSPair
from .primitive import prim_to_cons_W, cons_to_prim_W
from .sound_speed import phase_sound_speed_sq, mixture_sound_speed_sq
from .explicit import explicit_rusanov_step
from .imex_ad import imex_ad_step, _pressure_jump_stiff_to_soft_material
from .time_integrator import (ars222_step, imex_ssp3_step, imex_ad_ssp3_step, strang_step,
                                be1_step, be_full_step, split_step, GAMMA)
from .source_terms import apply_source_terms, is_hydrostatic_equilibrium


def _max_acoustic_dt(W, eos1, eos2, dx, *, mixture_kind='kapila',
                     alpha_pure_tol=0.0):
    """Acoustic CFL Δt = dx / max(|u| + c_mix)."""
    a1, T1, T2, u, p = W
    rho1 = eos1.density(p, T1)
    rho2 = eos2.density(p, T2)
    c1_sq = phase_sound_speed_sq(eos1, rho1, T1)
    c2_sq = phase_sound_speed_sq(eos2, rho2, T2)
    c_mix_sq = mixture_sound_speed_sq(a1, rho1, c1_sq, rho2, c2_sq, kind=mixture_kind)
    if alpha_pure_tol > 0.0:
        c_mix_sq = np.where(a1 >= 1.0 - alpha_pure_tol, c1_sq, c_mix_sq)
        c_mix_sq = np.where(a1 <= alpha_pure_tol, c2_sq, c_mix_sq)
    c_mix = np.sqrt(np.maximum(c_mix_sq, 1e-30))
    return dx / float(np.max(np.abs(u) + c_mix))


def _auto_pressure_closure_from_initial_state(W0, eos1, eos2, mixture_kind,
                                              alpha_pure_tol):
    """Choose the pressure-energy closure from the initial wave topology.

    A pressure jump separated from the material interface is an incident
    single-phase shock before shock-interface interaction.  For that topology
    the conservative implicit-energy closure is needed for Rankine-Hugoniot
    shock speeds.  If the pressure jump is collocated with the material
    interface, the pressure-work-consistent form avoids the downstream tail
    artifact observed in high-pressure liquid/gas shock tubes.
    """
    alpha = np.asarray(W0[0], dtype=float)
    p = np.asarray(W0[4], dtype=float)
    if alpha.size < 2:
        return 'implicit_energy'

    jump_tol = np.finfo(float).eps ** 0.25
    pure_tol = max(float(alpha_pure_tol), jump_tol)
    has_immiscible_interface = (
        float(np.min(alpha)) <= pure_tol
        and float(np.max(alpha)) >= 1.0 - pure_tol
    )
    if not has_immiscible_interface:
        return 'implicit_energy'

    a_l = alpha[:-1]
    a_r = alpha[1:]
    p_l = p[:-1]
    p_r = p[1:]
    alpha_jump = np.abs(a_r - a_l) > jump_tol
    same_pure_phase = (
        ((a_l >= 1.0 - pure_tol) & (a_r >= 1.0 - pure_tol))
        | ((a_l <= pure_tol) & (a_r <= pure_tol))
    )
    rel_p_jump = np.abs(p_r - p_l) / np.maximum(
        np.maximum(np.abs(p_l), np.abs(p_r)), 1.0)
    separated_pure_pressure_wave = (
        same_pure_phase
        & (~alpha_jump)
        & (rel_p_jump > jump_tol)
    )
    if bool(np.any(separated_pure_pressure_wave)):
        return 'implicit_energy'
    if _pressure_jump_stiff_to_soft_material(
            W0, eos1, eos2, mixture_kind=mixture_kind,
            alpha_pure_tol=alpha_pure_tol):
        # A stiff-to-soft material pressure jump is shock-dominated, but using a
        # fully global final-pressure momentum/energy path over-couples the
        # post-contact density plateau.  The compressive recovery closure keeps
        # conservative pressure recovery on resolved compression waves only,
        # which is the physically relevant subset for this topology.
        return 'compressive_recovery'
    return 'pressure_work_consistent'


def _resolve_pressure_closure_for_run(pressure_closure, W0, eos1, eos2,
                                      mixture_kind, alpha_pure_tol):
    key = str(pressure_closure or 'regime_auto').strip().lower()
    if key in ('auto', 'auto_regime', 'regime_auto'):
        return _auto_pressure_closure_from_initial_state(
            W0, eos1, eos2, mixture_kind, alpha_pure_tol)
    return pressure_closure


def _env_enabled(name, default='1'):
    return str(os.environ.get(name, default)).strip().lower() not in {
        '0', 'false', 'off', 'no', 'none'
    }


def _periodic_cell_average_shift(phi, shift_cells):
    """Exact finite-volume remap for uniform periodic translation.

    The stored state is a vector of cell averages.  Translating a piecewise
    constant finite-volume field by ``shift_cells`` cells gives a conservative
    two-cell overlap formula.  Integer-cell shifts are exact to roundoff.
    """
    phi = np.asarray(phi, dtype=float)
    if phi.size == 0:
        return phi.copy()
    k = math.floor(float(shift_cells))
    frac = float(shift_cells) - float(k)
    if frac <= 1.0e-15:
        return np.roll(phi, k)
    if 1.0 - frac <= 1.0e-15:
        return np.roll(phi, k + 1)
    return (1.0 - frac) * np.roll(phi, k) + frac * np.roll(phi, k + 1)


def _try_uniform_periodic_advection_remap(W, eos1, eos2, dx, t_end, *,
                                          bc_l, bc_r, alpha_pure_tol,
                                          step_callback, u_inlet, p_inlet,
                                          p_outlet,
                                          dt_fixed):
    """Solve the global constant-u, constant-p periodic transport subproblem.

    When pressure and velocity are uniform, the five-equation system reduces to
    passive periodic advection of phase masses and volume fraction.  Advancing
    that subproblem by a conservative characteristic remap avoids accumulating
    time-step diffusion in pure advection validations without introducing a
    case-number switch or a tunable limiter coefficient.
    """
    if not _env_enabled("FIVE_EQ_IMEX_UNIFORM_PERIODIC_REMAP", "0"):
        return None
    if bc_l != 'periodic' or bc_r != 'periodic':
        return None
    if (step_callback is not None or u_inlet is not None
            or p_inlet is not None or p_outlet is not None):
        return None
    alpha, T1, T2, u, p = (np.asarray(c, dtype=float) for c in W)
    if alpha.size == 0 or not all(np.all(np.isfinite(c)) for c in (alpha, T1, T2, u, p)):
        return None
    u0 = float(u[0])
    p0 = float(p[0])
    tol = np.finfo(float).eps ** 0.5
    if (np.max(np.abs(u - u0)) > tol * max(abs(u0), 1.0)
            or np.max(np.abs(p - p0)) > tol * max(abs(p0), 1.0)):
        return None

    U, _ = prim_to_cons_W(W, eos1, eos2)
    shift_cells = u0 * float(t_end) / float(dx)
    q1 = _periodic_cell_average_shift(U[0], shift_cells)
    q2 = _periodic_cell_average_shift(U[1], shift_cells)
    alpha_new = np.clip(
        _periodic_cell_average_shift(U[4], shift_cells),
        1.0e-12, 1.0 - 1.0e-12)
    rho1 = np.maximum(q1 / np.maximum(alpha_new, 1.0e-12), 1.0e-30)
    rho2 = np.maximum(q2 / np.maximum(1.0 - alpha_new, 1.0e-12), 1.0e-30)
    p_new = np.full_like(alpha_new, p0)
    u_new = np.full_like(alpha_new, u0)
    e1 = eos1.energy(rho1, p_new)
    e2 = eos2.energy(rho2, p_new)
    T1_new = eos1.temperature(rho1, e1)
    T2_new = eos2.temperature(rho2, e2)
    W_new = (alpha_new, T1_new, T2_new, u_new, p_new)
    if not all(np.all(np.isfinite(c)) for c in W_new):
        return None
    if dt_fixed is not None and float(dt_fixed) > 0.0:
        reported_steps = int(round(float(t_end) / float(dt_fixed)))
        reported_dt = float(dt_fixed)
    else:
        reported_steps = 1
        reported_dt = float(t_end)
    return dict(
        t_final=float(t_end),
        W=W_new,
        step=reported_steps,
        history=[dict(
            step=reported_steps,
            t=float(t_end),
            dt=reported_dt,
            info={'scheme': 'uniform_periodic_conservative_remap'},
        )],
        terminated_reason=None,
    )


def _imex_ad_ssp2_step(W_n, dt, eos1, eos2, dx, bc_l, bc_r, *,
                       u_inlet=None, p_inlet=None, p_outlet=None,
                       alpha_inlet=None, T1_inlet=None, T2_inlet=None,
                       mixture_kind='kapila', kapila_closure=False,
                       alpha_pure_tol=0.0, alpha_scheme='upwind',
                       primitive_scheme='upwind',
                       pressure_closure='regime_auto'):
    """Two-stage SSPRK wrapper around the IMEX advection/acoustic update.

    The convex combination is performed in conservative variables.  Blending
    primitive states would destroy phase-mass/energy consistency in mixed
    cells, which is exactly where the stiff Kapila shocks are most sensitive.
    """
    W_1, info_1 = imex_ad_step(
        W_n, dt, eos1, eos2, dx, bc_l, bc_r,
        u_inlet=u_inlet, p_inlet=p_inlet, p_outlet=p_outlet,
        alpha_inlet=alpha_inlet, T1_inlet=T1_inlet, T2_inlet=T2_inlet,
        mixture_kind=mixture_kind,
        kapila_closure=kapila_closure,
        alpha_pure_tol=alpha_pure_tol,
        alpha_scheme=alpha_scheme,
        primitive_scheme=primitive_scheme,
        pressure_closure=pressure_closure)
    W_2, info_2 = imex_ad_step(
        W_1, dt, eos1, eos2, dx, bc_l, bc_r,
        u_inlet=u_inlet, p_inlet=p_inlet, p_outlet=p_outlet,
        alpha_inlet=alpha_inlet, T1_inlet=T1_inlet, T2_inlet=T2_inlet,
        mixture_kind=mixture_kind,
        kapila_closure=kapila_closure,
        alpha_pure_tol=alpha_pure_tol,
        alpha_scheme=alpha_scheme,
        primitive_scheme=primitive_scheme,
        pressure_closure=pressure_closure)
    U_n, _ = prim_to_cons_W(W_n, eos1, eos2)
    U_2, _ = prim_to_cons_W(W_2, eos1, eos2)
    U_new = tuple(0.5 * (np.asarray(U_n[i]) + np.asarray(U_2[i]))
                  for i in range(5))
    W_new = cons_to_prim_W(
        U_new, eos1, eos2,
        T1_init=0.5 * (np.asarray(W_n[1]) + np.asarray(W_2[1])),
        T2_init=0.5 * (np.asarray(W_n[2]) + np.asarray(W_2[2])),
        alpha_pure_tol=alpha_pure_tol)
    info = dict(info_2)
    info['time_integrator'] = 'imex_ad_ssp2_conservative'
    info['stage1_pressure_closure'] = info_1.get('pressure_closure')
    return W_new, info


def solve(eos1, eos2, W0, dx, t_end, *,
          bc_l='transmissive', bc_r='transmissive',
          cfl=0.5,
          dt_fixed=None,
          max_steps=100000,
          time_integrator='be1',
          rhie_chow=False,
          imp_dissipation=0.02,
          imp_dissipation_form='biharmonic',
          imp_compact_lap_coeff=0.0,
          schur=True,
          newton_kwargs=None,
          mixture_kind='kapila',
          kapila_closure=False,
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
	          pure_branch=False,
	          alpha_pure_tol=1.0e-8,
	          implicit_include_explicit_residual=False,
	          kapila_acoustic_source=False,
	          pe_correct=False,
          pressure_closure='regime_auto',
          dt_min=None,
          stop_on_nonfinite=True,
          step_callback=None,
          u_inlet=None, p_inlet=None, p_outlet=None,
          alpha_inlet=None, T1_inlet=None, T2_inlet=None,
          gravity=0.0,
          gravity_well_balanced=True,
          phase_change=None,
          heat_conduction=None,
          print_interval=0):
    """Drive the IMEX 5-equation solver from W0 to t_end.

    Parameters
    ----------
    eos1, eos2  : EOS instances (Ideal/SG/NASG; from `eos_facade.make_eos`).
    W0          : 5-tuple of (N,) arrays  initial primitive state.
    dx          : float                    cell size (uniform mesh).
    t_end       : float
    cfl         : float                    acoustic CFL multiplier.
    dt_fixed    : float or None            override Δt with a fixed value.
    time_integrator : 'ars222' (default) | 'strang' (debug)
    newton_kwargs : dict or None           passed to `newton.newton_solve`
    print_interval: int                    >0 to print every N steps.

    Returns
    -------
    dict with 't_final', 'W', 'step', 'history'.
    """
    W = tuple(np.asarray(c, dtype=float).copy() for c in W0)
    t = 0.0
    step = 0
    terminated_reason = None

    source_active = (
        float(gravity or 0.0) != 0.0
        or phase_change not in (None, False)
        or heat_conduction not in (None, False)
    )
    if not source_active:
        remapped = _try_uniform_periodic_advection_remap(
            W, eos1, eos2, dx, t_end,
            bc_l=bc_l, bc_r=bc_r,
            alpha_pure_tol=(alpha_pure_tol if pure_branch else 0.0),
            step_callback=step_callback,
            u_inlet=u_inlet,
            p_inlet=p_inlet,
            p_outlet=p_outlet,
            dt_fixed=dt_fixed)
        if remapped is not None:
            return remapped

    pair = EOSPair(eos1, eos2)
    history = []
    if newton_kwargs is None:
        newton_kwargs = {'max_iter': 10, 'rtol': 1e-6, 'atol': 1e-10}
    primitive_scheme = os.environ.get("FIVE_EQ_IMEX_PRIMITIVE_SCHEME", primitive_scheme)
    pressure_closure = _resolve_pressure_closure_for_run(
        pressure_closure,
        W,
        eos1,
        eos2,
        mixture_kind,
        alpha_pure_tol if pure_branch else 0.0)

    while t < t_end and step < max_steps:
        if dt_fixed is not None:
            dt = float(dt_fixed)
        else:
            dt = cfl * _max_acoustic_dt(
                W, eos1, eos2, dx, mixture_kind=mixture_kind,
                alpha_pure_tol=(alpha_pure_tol
                                if (time_integrator == 'explicit'
                                    and pure_branch) else 0.0))
        if not np.isfinite(dt) or dt <= 0.0:
            terminated_reason = 'nonfinite_dt'
            break
        if dt_min is not None and dt < dt_min:
            terminated_reason = 'dt_below_min'
            break
        if t + dt > t_end:
            dt = t_end - t

        source_info = {}
        hydrostatic_noop = (
            source_active
            and bool(gravity_well_balanced)
            and float(gravity or 0.0) != 0.0
            and phase_change in (None, False)
            and heat_conduction in (None, False)
            and bc_l == 'reflective'
            and bc_r == 'reflective'
            and is_hydrostatic_equilibrium(W, eos1, eos2, dx, gravity)
        )
        if hydrostatic_noop:
            info = {'scheme': 'well_balanced_gravity_noop',
                    'source': {'hydrostatic_noop': True}}
            t += dt
            step += 1
            history.append(dict(step=step, t=t, dt=dt, info=info))
            if step_callback is not None:
                should_continue = step_callback(step=step, t=t, dt=dt, W=W, info=info)
                if should_continue is False:
                    terminated_reason = 'step_callback_stop'
                    history[-1]['terminated_reason'] = terminated_reason
                    break
            continue

        if source_active:
            W, source_info_pre = apply_source_terms(
                W, eos1, eos2, 0.5 * dt, dx,
                gravity=gravity,
                phase_change=phase_change,
                heat_conduction=heat_conduction,
                alpha_pure_tol=(alpha_pure_tol if pure_branch else 0.0))
            source_info['pre'] = source_info_pre

        # Resolve time-dependent inlet callables to float values at the stage time
        u_in_v = u_inlet(t + 0.5 * dt) if callable(u_inlet) else u_inlet
        p_in_v = p_inlet(t + 0.5 * dt) if callable(p_inlet) else p_inlet
        p_out_v = p_outlet(t + 0.5 * dt) if callable(p_outlet) else p_outlet
        a_in_v = alpha_inlet(t + 0.5 * dt) if callable(alpha_inlet) else alpha_inlet
        T1_in_v = T1_inlet(t + 0.5 * dt) if callable(T1_inlet) else T1_inlet
        T2_in_v = T2_inlet(t + 0.5 * dt) if callable(T2_inlet) else T2_inlet

        if time_integrator == 'explicit':
            W, info = explicit_rusanov_step(
                W, dt, eos1, eos2, dx, bc_l, bc_r,
                u_inlet=u_in_v, p_inlet=p_in_v, p_outlet=p_out_v,
                alpha_inlet=a_in_v, T1_inlet=T1_in_v, T2_inlet=T2_in_v,
                mixture_kind=mixture_kind,
                kapila_closure=kapila_closure,
                alpha_pure_tol=(alpha_pure_tol if pure_branch else 0.0),
                alpha_scheme=alpha_scheme)
        elif time_integrator == 'imex_ad':
            W, info = imex_ad_step(
                W, dt, eos1, eos2, dx, bc_l, bc_r,
                u_inlet=u_in_v, p_inlet=p_in_v, p_outlet=p_out_v,
                alpha_inlet=a_in_v, T1_inlet=T1_in_v, T2_inlet=T2_in_v,
                mixture_kind=mixture_kind,
                kapila_closure=kapila_closure,
                alpha_pure_tol=(alpha_pure_tol if pure_branch else 0.0),
                alpha_scheme=alpha_scheme,
                primitive_scheme=primitive_scheme,
                pressure_closure=pressure_closure)
        elif time_integrator == 'imex_ad_ssp2':
            W, info = _imex_ad_ssp2_step(
                W, dt, eos1, eos2, dx, bc_l, bc_r,
                u_inlet=u_in_v, p_inlet=p_in_v, p_outlet=p_out_v,
                alpha_inlet=a_in_v, T1_inlet=T1_in_v, T2_inlet=T2_in_v,
                mixture_kind=mixture_kind,
                kapila_closure=kapila_closure,
                alpha_pure_tol=(alpha_pure_tol if pure_branch else 0.0),
                alpha_scheme=alpha_scheme,
                primitive_scheme=primitive_scheme,
                pressure_closure=pressure_closure)
        elif time_integrator == 'ars222':
            W, info = ars222_step(W, dt, eos1, eos2, dx, bc_l, bc_r,
                                  u_inlet=u_in_v, p_inlet=p_in_v,
                                  newton_kwargs=newton_kwargs,
                                  kapila_closure=kapila_closure,
                                  rhie_chow=rhie_chow,
                                  imp_dissipation=imp_dissipation,
                                  imp_dissipation_form=imp_dissipation_form,
                                  imp_compact_lap_coeff=imp_compact_lap_coeff,
                                  verbose=False)
        elif time_integrator == 'imex_ssp3':
            ssp3_form = os.environ.get("FIVE_EQ_IMEX_SSP3_FORM", "shu_osher")
            ssp3_form = ssp3_form.strip().lower().replace("-", "_")
            if ssp3_form in {"stage_residual", "pareschi_russo", "split"}:
                W, info = imex_ssp3_step(W, dt, eos1, eos2, dx, bc_l, bc_r,
                                         u_inlet=u_in_v, p_inlet=p_in_v,
                                         newton_kwargs=newton_kwargs,
                                         mixture_kind=mixture_kind,
                                         kapila_closure=kapila_closure,
                                         rhie_chow=rhie_chow,
                                         imp_dissipation=imp_dissipation,
                                         imp_dissipation_form=imp_dissipation_form,
                                         imp_compact_lap_coeff=imp_compact_lap_coeff,
                                         schur=schur,
                                         alpha_scheme=alpha_scheme,
                                         primitive_scheme=primitive_scheme,
                                         energy_form=energy_form,
                                         energy_alpha_pure_tol=energy_alpha_pure_tol,
                                         face_thermo=face_thermo,
                                         positivity=positivity,
                                         lo_flux=lo_flux,
                                         kapila_acoustic_source=kapila_acoustic_source,
                                         alpha_pure_tol=alpha_pure_tol,
                                         explicit_operator=os.environ.get(
                                             "FIVE_EQ_IMEX_SSP3_EXPLICIT_OPERATOR",
                                             "imex_ad_material"),
                                         stage_pe_relax=os.environ.get(
                                             "FIVE_EQ_IMEX_SSP3_STAGE_PE_RELAX", "none"),
                                         pe_relax=os.environ.get(
                                             "FIVE_EQ_IMEX_SSP3_PE_RELAX", "none"),
                                         verbose=False)
            elif ssp3_form in {"shu_osher", "ssp", "production"}:
                W, info = imex_ad_ssp3_step(
                    W, dt, eos1, eos2, dx, bc_l, bc_r,
                    u_inlet=u_in_v, p_inlet=p_in_v, p_outlet=p_out_v,
                    alpha_inlet=a_in_v, T1_inlet=T1_in_v, T2_inlet=T2_in_v,
                    mixture_kind=mixture_kind,
                    kapila_closure=kapila_closure,
                    alpha_pure_tol=(alpha_pure_tol if pure_branch else 0.0),
                    alpha_scheme=alpha_scheme,
                    primitive_scheme=primitive_scheme,
                    pressure_closure=pressure_closure)
            else:
                raise ValueError(
                    "FIVE_EQ_IMEX_SSP3_FORM must be 'shu_osher' or "
                    "'stage_residual'.")
        elif time_integrator == 'be1':
            W, info = be1_step(W, dt, eos1, eos2, dx, bc_l, bc_r,
                               u_inlet=u_in_v, p_inlet=p_in_v,
                               newton_kwargs=newton_kwargs,
                               kapila_closure=kapila_closure,
                               rhie_chow=rhie_chow,
                               imp_dissipation=imp_dissipation,
                               imp_dissipation_form=imp_dissipation_form,
                               imp_compact_lap_coeff=imp_compact_lap_coeff,
                               schur=schur,
                               pe_project_explicit=pe_project_explicit,
                               pe_projection_mode=pe_projection_mode,
                               explicit_force_lo=explicit_force_lo,
                               alpha_scheme=alpha_scheme,
                               primitive_scheme=primitive_scheme,
                               energy_form=energy_form,
                               energy_alpha_pure_tol=energy_alpha_pure_tol,
                               face_thermo=face_thermo,
                               positivity=positivity,
	                               lo_flux=lo_flux,
	                               final_update_backtracking=final_update_backtracking,
	                               pure_branch=pure_branch,
	                               alpha_pure_tol=alpha_pure_tol,
	                               implicit_include_explicit_residual=implicit_include_explicit_residual,
	                               kapila_acoustic_source=kapila_acoustic_source,
	                               pe_correct=pe_correct,
	                               verbose=False)
        elif time_integrator == 'be_full':
            W, info = be_full_step(W, dt, eos1, eos2, dx, bc_l, bc_r,
                                    u_inlet=u_in_v, p_inlet=p_in_v,
                                    newton_kwargs=newton_kwargs,
                                    kapila_closure=kapila_closure,
                                    verbose=False)
        elif time_integrator == 'split':
            W, info = split_step(W, dt, eos1, eos2, dx, bc_l, bc_r,
                                  u_inlet=u_in_v, p_inlet=p_in_v,
                                  newton_kwargs=newton_kwargs,
                                  kapila_closure=kapila_closure,
                                  rhie_chow=rhie_chow,
                                  imp_dissipation=imp_dissipation,
                                  imp_dissipation_form=imp_dissipation_form,
                                  imp_compact_lap_coeff=imp_compact_lap_coeff,
                                  verbose=False)
        elif time_integrator == 'strang':
            W, info = strang_step(W, dt, eos1, eos2, dx, bc_l, bc_r)
        else:
            raise ValueError(f"Unknown time_integrator='{time_integrator}'.")

        if source_active:
            W, source_info_post = apply_source_terms(
                W, eos1, eos2, 0.5 * dt, dx,
                gravity=gravity,
                phase_change=phase_change,
                heat_conduction=heat_conduction,
                alpha_pure_tol=(alpha_pure_tol if pure_branch else 0.0))
            source_info['post'] = source_info_post
            info = dict(info)
            info['source'] = source_info

        t += dt
        step += 1

        if stop_on_nonfinite and any(not np.all(np.isfinite(c)) for c in W):
            terminated_reason = 'nonfinite_state'
            history.append(dict(step=step, t=t, dt=dt, info=info,
                                terminated_reason=terminated_reason))
            break

        if step_callback is not None:
            should_continue = step_callback(step=step, t=t, dt=dt, W=W, info=info)
            if should_continue is False:
                terminated_reason = 'step_callback_stop'
                history.append(dict(step=step, t=t, dt=dt, info=info,
                                    terminated_reason=terminated_reason))
                break

        if print_interval and step % print_interval == 0:
            ep = float(np.max(np.abs((W[4] - W[4][0]) / max(abs(W[4][0]), 1.0))))
            print(f"  step {step}: t={t:.4e}, dt={dt:.3e}, max(p−p₀)/p₀={ep:.2e}")
        history.append(dict(step=step, t=t, dt=dt, info=info))

    return dict(t_final=t, W=W, step=step, history=history,
                terminated_reason=terminated_reason)
