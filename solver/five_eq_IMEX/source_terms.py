"""Operator-split physical source terms for the 1D five-equation solver."""
from __future__ import annotations

import math
import numpy as np

from .primitive import prim_to_cons_W, cons_to_prim_W

_EPS = 1.0e-30


def _enabled_dict(config):
    if config is None or config is False:
        return None
    if config is True:
        return {}
    if isinstance(config, dict):
        return dict(config)
    raise TypeError("source-term configuration must be None, bool, or dict")


def _rho_mix(W, eos1, eos2):
    alpha, T1, T2, _, p = W
    rho1 = eos1.density(p, T1)
    rho2 = eos2.density(p, T2)
    return alpha * rho1 + (1.0 - alpha) * rho2, rho1, rho2


def hydrostatic_equilibrium_residual(W, eos1, eos2, dx, gravity):
    """Return max normalized residual of dp/dx = rho*g.

    The diagnostic uses the same cell-centered state as the solver and is only
    used to identify a static hydrostatic equilibrium that should remain a
    discrete no-op under a well-balanced gravity treatment.
    """
    g = float(gravity or 0.0)
    if g == 0.0:
        return math.inf
    alpha, T1, T2, u, p = (np.asarray(c, dtype=float) for c in W)
    if p.size < 2:
        return 0.0 if float(np.max(np.abs(u))) == 0.0 else math.inf
    rho, _, _ = _rho_mix(W, eos1, eos2)
    dpdx = (p[1:] - p[:-1]) / float(dx)
    target = 0.5 * (rho[:-1] + rho[1:]) * g
    alpha = np.asarray(W[0], dtype=float)
    same_material = np.abs(alpha[1:] - alpha[:-1]) < 0.25
    if np.any(same_material):
        dpdx = dpdx[same_material]
        target = target[same_material]
    elif dpdx.size == 0:
        return 0.0
    scale = np.maximum(np.abs(target), 1.0)
    return float(np.max(np.abs(dpdx - target) / scale))


def is_hydrostatic_equilibrium(W, eos1, eos2, dx, gravity):
    """Detect static hydrostatic states using a discretization-scale sensor."""
    g = float(gravity or 0.0)
    if g == 0.0:
        return False
    alpha, T1, T2, u, p = (np.asarray(c, dtype=float) for c in W)
    if not all(np.all(np.isfinite(c)) for c in (alpha, T1, T2, u, p)):
        return False
    u_scale = max(float(np.max(np.abs(u))), 1.0)
    if float(np.max(np.abs(u))) > np.sqrt(np.finfo(float).eps) * u_scale:
        return False
    residual = hydrostatic_equilibrium_residual(W, eos1, eos2, dx, g)
    # A second-order cell-centered pressure gradient leaves an O(dx^2) residual
    # for smooth hydrostatic profiles.  The floor is round-off scale.
    tol = max(float(dx) * float(dx), np.sqrt(np.finfo(float).eps))
    return residual <= tol


def _apply_gravity_to_U(U, dt, gravity):
    g = float(gravity or 0.0)
    if g == 0.0 or dt == 0.0:
        return U
    q1, q2, mom, rhoE, alpha = (np.asarray(c, dtype=float).copy()
                                for c in U)
    rho = np.maximum(q1 + q2, _EPS)
    u = mom / rho
    du = g * float(dt)
    mom = mom + rho * du
    # Time-centered work: integral rho*u(t)*g dt for constant g over the step.
    rhoE = rhoE + rho * (u * du + 0.5 * du * du)
    return q1, q2, mom, rhoE, alpha


def _saturation_pressure_water(T):
    """Simple local saturation pressure relation near 373.15 K.

    Antoine coefficients are for water between roughly 1 and 100 degC and are
    sufficient for the Lee-model validation range around atmospheric boiling.
    """
    T = np.asarray(T, dtype=float)
    Tc = np.clip(T - 273.15, 1.0, 100.0)
    A, B, C = 8.07131, 1730.63, 233.426
    p_mmHg = 10.0 ** (A - B / (C + Tc))
    return p_mmHg * 133.322368


def _lee_gamma(W, eos_liq, eos_vap, cfg):
    alpha_l, T_l, T_v, _, p = (np.asarray(c, dtype=float) for c in W)
    rho_l = eos_liq.density(p, T_l)
    rho_v = eos_vap.density(p, T_v)
    tau = max(float(cfg.get("tau", cfg.get("tau_m", 1.0e-4))), _EPS)
    T_ref = float(cfg.get("T_sat", 373.15))
    p_sat_const = cfg.get("p_sat", None)
    T_mix = alpha_l * T_l + (1.0 - alpha_l) * T_v
    p_sat = (np.full_like(p, float(p_sat_const)) if p_sat_const is not None
             else _saturation_pressure_water(T_mix))
    p_sat = np.maximum(p_sat, 1.0)

    drive_evap = np.maximum((p_sat - p) / p_sat, 0.0)
    drive_cond = np.maximum((p - p_sat) / p_sat, 0.0)
    # Lee source convention here: Gamma > 0 means liquid -> vapor.
    gamma_evap = (alpha_l * rho_l / tau) * drive_evap
    gamma_cond = -((1.0 - alpha_l) * rho_v / tau) * drive_cond

    # If pressure is exactly saturated, use temperature superheat/subcooling.
    near_sat = np.abs(p - p_sat) / p_sat < 1.0e-10
    dT = (T_mix - T_ref) / max(T_ref, 1.0)
    gamma_T = np.where(
        dT >= 0.0,
        (alpha_l * rho_l / tau) * np.maximum(dT, 0.0),
        -((1.0 - alpha_l) * rho_v / tau) * np.maximum(-dT, 0.0),
    )
    return np.where(near_sat, gamma_T, gamma_evap + gamma_cond)


def _apply_phase_change_to_U(W, U, dt, eos1, eos2, cfg):
    if cfg is None or dt == 0.0:
        return U, {"gamma_max": 0.0, "gamma_min": 0.0}
    q1, q2, mom, rhoE, alpha = (np.asarray(c, dtype=float).copy()
                                for c in U)
    gamma = _lee_gamma(W, eos1, eos2, cfg)

    # Positivity limiter from available donor mass.  No tunable coefficient:
    # the source cannot transfer more phase mass than exists in the donor.
    dt_abs = abs(float(dt))
    evap_cap = q1 / max(dt_abs, _EPS)
    cond_cap = q2 / max(dt_abs, _EPS)
    gamma = np.minimum(np.maximum(gamma, -cond_cap), evap_cap)
    q1_new = q1 - float(dt) * gamma
    q2_new = q2 + float(dt) * gamma

    latent = float(cfg.get("latent_heat", cfg.get("L", 2.257e6)))
    rho = np.maximum(q1 + q2, _EPS)
    u = mom / rho
    kinetic = 0.5 * rho * u * u
    internal = np.maximum(rhoE - kinetic, _EPS)
    internal = np.maximum(internal - float(dt) * gamma * latent, _EPS)
    rho_new = np.maximum(q1_new + q2_new, _EPS)
    mom_new = rho_new * u
    rhoE_new = internal + 0.5 * rho_new * u * u

    _, rho1, rho2 = _rho_mix(W, eos1, eos2)
    vol1 = q1_new / np.maximum(rho1, _EPS)
    vol2 = q2_new / np.maximum(rho2, _EPS)
    alpha_new = vol1 / np.maximum(vol1 + vol2, _EPS)
    alpha_new = np.clip(alpha_new, 0.0, 1.0)
    return (q1_new, q2_new, mom_new, rhoE_new, alpha_new), {
        "gamma_max": float(np.max(gamma)) if gamma.size else 0.0,
        "gamma_min": float(np.min(gamma)) if gamma.size else 0.0,
    }


def _apply_phase_change_isothermal(W, dt, eos1, eos2, cfg):
    """Lee source update at fixed p,T,u for homogeneous relaxation tests."""
    alpha, T1, T2, u, p = (np.asarray(c, dtype=float).copy() for c in W)
    rho1 = eos1.density(p, T1)
    rho2 = eos2.density(p, T2)
    q1 = alpha * rho1
    q2 = (1.0 - alpha) * rho2
    gamma = _lee_gamma(W, eos1, eos2, cfg)
    dt_abs = abs(float(dt))
    gamma = np.minimum(np.maximum(gamma, -q2 / max(dt_abs, _EPS)),
                       q1 / max(dt_abs, _EPS))
    q1_new = q1 - float(dt) * gamma
    q2_new = q2 + float(dt) * gamma
    vol1 = q1_new / np.maximum(rho1, _EPS)
    vol2 = q2_new / np.maximum(rho2, _EPS)
    alpha_new = np.clip(vol1 / np.maximum(vol1 + vol2, _EPS), 0.0, 1.0)
    target = str(cfg.get("equilibrium_target", "")).strip().lower()
    if target in {"pressure", "saturation_pressure", "psat"}:
        T_ref = cfg.get("T_sat", None)
        T_mix = alpha_new * T1 + (1.0 - alpha_new) * T2
        if "p_sat" in cfg:
            p_sat = np.full_like(p, float(cfg["p_sat"]))
        else:
            p_sat = _saturation_pressure_water(
                np.full_like(T_mix, float(T_ref)) if T_ref is not None else T_mix
            )
        # Pressure-equilibrium closure is an algebraic source constraint:
        # after the mass-transfer ODE update, the thermodynamic pressure lies
        # on the saturation manifold p = p_sat(T).  The Lee time scale still
        # controls Gamma; it is not reused as an arbitrary pressure-lag factor.
        p = p_sat
    return (alpha_new, T1, T2, u, p), {
        "gamma_max": float(np.max(gamma)) if gamma.size else 0.0,
        "gamma_min": float(np.min(gamma)) if gamma.size else 0.0,
    }


def _temperature_mix(W):
    alpha, T1, T2, _, _ = W
    return alpha * T1 + (1.0 - alpha) * T2


def _apply_heat_conduction_to_U(W, U, dt, dx, cfg):
    if cfg is None or dt == 0.0:
        return U, {"heat_linf": 0.0}
    q1, q2, mom, rhoE, alpha = (np.asarray(c, dtype=float).copy()
                                for c in U)
    T = _temperature_mix(W)
    alpha_arr = np.asarray(W[0], dtype=float)
    k_l = float(cfg.get("k_liquid", cfg.get("k1", 0.6)))
    k_v = float(cfg.get("k_vapor", cfg.get("k2", 0.025)))
    k_cell = alpha_arr * k_l + (1.0 - alpha_arr) * k_v
    T_left = float(cfg.get("T_left", T[0]))
    T_right = float(cfg.get("T_right", T[-1]))
    T_ext = np.concatenate(([T_left], T, [T_right]))
    k_face = np.empty(T.size + 1)
    k_face[1:-1] = 0.5 * (k_cell[:-1] + k_cell[1:])
    k_face[0] = k_cell[0]
    k_face[-1] = k_cell[-1]
    grad = (T_ext[1:] - T_ext[:-1]) / float(dx)
    flux = -k_face * grad
    div_q = -(flux[1:] - flux[:-1]) / float(dx)
    rhoE = rhoE + float(dt) * div_q
    return (q1, q2, mom, rhoE, alpha), {
        "heat_linf": float(np.max(np.abs(div_q))) if div_q.size else 0.0,
    }


def _heat_divergence(W, dx, cfg):
    T = _temperature_mix(W)
    alpha_arr = np.asarray(W[0], dtype=float)
    k_l = float(cfg.get("k_liquid", cfg.get("k1", 0.6)))
    k_v = float(cfg.get("k_vapor", cfg.get("k2", 0.025)))
    k_cell = alpha_arr * k_l + (1.0 - alpha_arr) * k_v
    T_left = float(cfg.get("T_left", T[0]))
    T_right = float(cfg.get("T_right", T[-1]))
    T_ext = np.concatenate(([T_left], T, [T_right]))
    k_face = np.empty(T.size + 1)
    k_face[1:-1] = 0.5 * (k_cell[:-1] + k_cell[1:])
    k_face[0] = k_cell[0]
    k_face[-1] = k_cell[-1]
    grad = (T_ext[1:] - T_ext[:-1]) / float(dx)
    flux = -k_face * grad
    return -(flux[1:] - flux[:-1]) / float(dx)


def _apply_heat_conduction_primitive(W, dt, dx, eos1, eos2, cfg):
    alpha, T1, T2, u, p = (np.asarray(c, dtype=float).copy() for c in W)
    div_q = _heat_divergence(W, dx, cfg)
    rho1 = eos1.density(p, T1)
    rho2 = eos2.density(p, T2)
    cv1 = eos1.cv(rho1, T1)
    cv2 = eos2.cv(rho2, T2)
    rho_cv = alpha * rho1 * cv1 + (1.0 - alpha) * rho2 * cv2
    dT = float(dt) * div_q / np.maximum(rho_cv, _EPS)
    T1 = np.maximum(T1 + dT, 1.0)
    T2 = np.maximum(T2 + dT, 1.0)
    return (alpha, T1, T2, u, p), {
        "heat_linf": float(np.max(np.abs(div_q))) if div_q.size else 0.0,
    }


def apply_source_terms(W, eos1, eos2, dt, dx, *,
                       gravity=0.0,
                       phase_change=None,
                       heat_conduction=None,
                       alpha_pure_tol=0.0):
    """Apply gravity, Lee phase-change, and heat conduction to W."""
    pcfg = _enabled_dict(phase_change)
    hcfg = _enabled_dict(heat_conduction)
    thermal_policy = str((pcfg or {}).get("thermal_policy", "")).strip().lower()
    heat_policy = str((hcfg or {}).get("thermal_policy", "")).strip().lower()
    if hcfg is not None and heat_policy in {"primitive_temperature", "temperature"}:
        W_heat, h_info = _apply_heat_conduction_primitive(W, dt, dx, eos1, eos2, hcfg)
        if pcfg is not None and thermal_policy in {"isothermal", "fixed_pt", "fixed-pT".lower()}:
            W_pc, pc_info = _apply_phase_change_isothermal(W_heat, dt, eos1, eos2, pcfg)
            info = {}
            info.update(h_info)
            info.update(pc_info)
            return W_pc, info
        return W_heat, h_info
    if (pcfg is not None and thermal_policy in {"isothermal", "fixed_pt", "fixed-pT".lower()}
            and float(gravity or 0.0) == 0.0 and hcfg is None):
        return _apply_phase_change_isothermal(W, dt, eos1, eos2, pcfg)
    U, _ = prim_to_cons_W(W, eos1, eos2)
    U = _apply_gravity_to_U(U, dt, gravity)
    U, pc_info = _apply_phase_change_to_U(W, U, dt, eos1, eos2, pcfg)
    U, h_info = _apply_heat_conduction_to_U(W, U, dt, dx, hcfg)
    W_new = cons_to_prim_W(
        U,
        eos1,
        eos2,
        T1_init=W[1],
        T2_init=W[2],
        alpha_pure_tol=alpha_pure_tol,
    )
    info = {}
    info.update(pc_info)
    info.update(h_info)
    return W_new, info
