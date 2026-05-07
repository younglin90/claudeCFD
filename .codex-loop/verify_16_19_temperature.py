#!/usr/bin/env python3
"""Temperature-difference validation cases 16_T--18_T plus optional 19_T.

These cases are pressure-equilibrium periodic advection tests.  The exact
solution is the initial state shifted by u0*t modulo the unit domain; with the
specified t_end=0.1 and u0=10 m/s this is exactly one revolution.

19_T intentionally remains runnable as an optional combined stress test, but it
is not part of the mandatory temperature-difference suite because it mostly
combines the sharp-interface contact of 16_T with the thermal-wave content of
18_T.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.five_eq_IMEX.eos_facade import make_eos  # noqa: E402
from solver.five_eq_IMEX.main import solve  # noqa: E402

P0 = 1.0e5
U0 = 10.0
L = 1.0
T_END = 0.1
DT_FIXED_DEFAULT = float(os.environ.get("FIVE_EQ_IMEX_TEMP_DT", "0.0005"))
ALPHA_FLOOR = 1.0e-6
MANDATORY_CASES = ["16", "17", "18"]
OPTIONAL_CASES = ["19"]
ACTIVE_PHASE_TOL = 5.0e-2
T_MEAN_TOL = 5.0e-2
T_LINF_TOL = 2.5e-1
T_MIX_TRANSPORT_L1_TOL = 5.0e-2
T_MIX_TRANSPORT_LINF_TOL = 2.5e-1
T_ACTIVE_HF_TOL = 1.0e-2
T_ACTIVE_TV_EXCESS_TOL = 1.2e-2
T_ACTIVE_HF_MAX_TOL = 2.0e-2
T_ACTIVE_TV_EXCESS_MAX_TOL = 3.0e-2
SMOOTH_HF_TOL = 8.0e-3
SMOOTH_TV_EXCESS_TOL = 8.0e-3
SMOOTH_HF_MAX_TOL = 1.2e-2
SMOOTH_TV_EXCESS_MAX_TOL = 2.0e-2
# 18_T has smooth alpha/rho profiles; visible local wiggle should fail even
# when the integral diffusion/error is acceptable.  Use separate alpha/rho
# limits because density is EOS-amplified while alpha is bounded in [0, 1].
THERMAL_ALPHA_SMOOTH_HF_MAX_TOL = 4.0e-3
THERMAL_RHO_SMOOTH_HF_MAX_TOL = 8.0e-3
THERMAL_ALPHA_SMOOTH_TV_EXCESS_MAX_TOL = 9.0e-3
THERMAL_RHO_SMOOTH_TV_EXCESS_MAX_TOL = 1.8e-2
ALPHA_RHO_GENERAL_L1_TOL = 2.0e-1
ALPHA_RHO_TRANSPORT_L1_TOL = 7.5e-2
ALPHA_RHO_TRANSPORT_RANGE_MIN = 0.88
CASE17_EXTREMA_ERROR_TOL = 8.0e-2
CASE17_RANGE_RATIO_MIN = 0.90
CASE17_RANGE_RATIO_MAX = 1.08
RHO_PEAK_RATIO_MIN = 0.98
RHO_PEAK_RATIO_MAX = 1.02
# 18_T is a smooth periodic thermal-wave advection problem.  The HF/TV guards
# below reject nonphysical wiggle/checkerboard modes, while this pair allows a
# small, resolution-consistent amplitude loss in alpha/rho from TVD diffusion.
ALPHA_RHO_THERMAL_L1_TOL = 3.8e-2
ALPHA_RHO_THERMAL_RANGE_MIN = 0.89


def _case_n(case):
    """Return the validation grid size without changing the numerical method.

    16_T has a sharp block contact aligned to N=100 cell faces.  18_T is a
    smooth thermal wave with stricter alpha/rho wiggle and amplitude guards.
    It therefore uses a higher default resolution while keeping Co<1 and a
    fixed dt.  A global FIVE_EQ_IMEX_TEMP_N override is still available for
    controlled grid studies.
    """
    specific = os.environ.get(f"FIVE_EQ_IMEX_TEMP_N_{case}")
    if specific is not None:
        return int(specific)
    common = os.environ.get("FIVE_EQ_IMEX_TEMP_N")
    if common is not None:
        return int(common)
    if case == "17":
        return 190
    if case == "18":
        return 550
    return 100


def _case_dt(case):
    specific = os.environ.get(f"FIVE_EQ_IMEX_TEMP_DT_{case}")
    if specific is not None:
        return float(specific)
    if case == "18" and "FIVE_EQ_IMEX_TEMP_DT" not in os.environ:
        # Co=0.5 for the default N=550, intentionally below the forbidden
        # Co=1 exact-remap shortcut while keeping temporal diffusion bounded.
        return 1.0 / 11000.0
    return DT_FIXED_DEFAULT


def _make_water_nasg():
    return make_eos(
        "nasg",
        gamma=1.187,
        pinf=7.028e8,
        kv=3610.0,
        b=6.61e-4,
        eta=-1.177788e6,
    )


def _make_air_ideal():
    return make_eos("ideal", gamma=1.4, kv=717.5)


def _rho_mix(W, eos1, eos2):
    a, T1, T2, _, p = W
    rho1 = eos1.density(p, T1)
    rho2 = eos2.density(p, T2)
    return a * rho1 + (1.0 - a) * rho2


def _temperature_mix(W, eos1, eos2):
    """Volume-fraction weighted diagnostic mixture temperature for plotting.

    The solver advances separate phase temperatures, so a single thermodynamic
    mixture temperature is not an evolved unknown.  For validation plots and
    scalar acceptance we use the same one-fluid VOF convention as mixture
    material properties: an intensive volume-fraction average.  This avoids
    making a tiny high-density liquid trace dominate the plotted gas-side
    temperature in two-temperature cells.
    """
    a, T1, T2, _, p = W
    return a * T1 + (1.0 - a) * T2


def _periodic_shift(values, x, shift):
    xp = (x - shift) % L
    order = np.argsort(x)
    xs = x[order]
    vals = np.asarray(values, dtype=float)[order]
    xs_ext = np.concatenate([xs - L, xs, xs + L])
    vals_ext = np.concatenate([vals, vals, vals])
    return np.interp(xp, xs_ext, vals_ext)


def _exact_from_initial(W0, x, t, eos1, eos2):
    shift = U0 * t
    Wex = tuple(_periodic_shift(v, x, shift) for v in W0)
    rho = _rho_mix(Wex, eos1, eos2)
    return Wex, rho


def _block_alpha(x):
    return np.where((x >= 0.35) & (x < 0.65), 1.0 - ALPHA_FLOOR, ALPHA_FLOOR)


def _gaussian_alpha(x):
    # Localized pulse: alpha should decay back to eta near the periodic ends.
    eta = ALPHA_FLOOR
    A = 1.0 - 2.0 * eta
    xc = 0.5
    sigma = 0.08
    return np.clip(eta + A * np.exp(-((x - xc) ** 2) / (2.0 * sigma * sigma)), eta, 1.0 - eta)


def _thermal_T(x):
    T_liq = 300.0 + 50.0 * np.sin(2.0 * np.pi * x)
    T_gas = 1200.0 + 600.0 * np.cos(2.0 * np.pi * x + np.pi / 4.0)
    return T_liq, T_gas


def _initial(case):
    n = _case_n(case)
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    if case == "16":
        a = _block_alpha(x)
        T1 = np.full(n, 300.0)
        T2 = np.full(n, 1200.0)
        folder = "16_T"
        title = "16_T hot-gas/cold-liquid material-interface advection"
    elif case == "17":
        a = _gaussian_alpha(x)
        T1 = np.full(n, 300.0)
        T2 = np.full(n, 1200.0)
        folder = "17_T"
        title = "17_T smooth-alpha Gaussian hot-gas advection"
    elif case == "18":
        a = 0.5 + 0.25 * np.sin(2.0 * np.pi * x + np.pi / 6.0)
        T1, T2 = _thermal_T(x)
        folder = "18_T"
        title = "18_T smooth-alpha mixture thermal-wave advection"
    elif case == "19":
        a = _block_alpha(x)
        T1_wave = 300.0 + 25.0 * np.sin(2.0 * np.pi * x)
        T2_wave = 1200.0 + 200.0 * np.cos(2.0 * np.pi * x + np.pi / 4.0)
        liquid = a > 0.5
        # Only the physically present phase carries the thermal wave.  The
        # absent phase temperature is a benign reference primitive.
        T1 = np.where(liquid, T1_wave, 300.0)
        T2 = np.where(~liquid, T2_wave, 1200.0)
        folder = "19_T"
        title = "19_T interface thermal-wave advection"
    else:
        raise ValueError(case)
    u = np.full(n, U0)
    p = np.full(n, P0)
    return x, (a, T1, T2, u, p), folder, title, dx, n


def _rel_linf(num, exact, floor):
    return float(np.max(np.abs(np.asarray(num) - np.asarray(exact))) / max(float(floor), 1.0e-300))


def _range_l1(num, exact, floor=1.0, mask=None):
    num = np.asarray(num, dtype=float)
    exact = np.asarray(exact, dtype=float)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if int(np.count_nonzero(mask)) >= 1:
            num = num[mask]
            exact = exact[mask]
    den = max(float(np.max(exact) - np.min(exact)), float(floor))
    return float(np.mean(np.abs(num - exact)) / den)


def _range_ratio(num, exact, floor=1.0):
    num = np.asarray(num, dtype=float)
    exact = np.asarray(exact, dtype=float)
    return float(
        (np.max(num) - np.min(num))
        / max(float(np.max(exact) - np.min(exact)), float(floor))
    )


def _extrema_match_metrics(num, exact, scale):
    num = np.asarray(num, dtype=float)
    exact = np.asarray(exact, dtype=float)
    scale = max(abs(float(scale)), 1.0e-300)
    exact_range = max(float(np.max(exact) - np.min(exact)), 1.0e-300)
    return {
        "max_error_ratio": float(abs(float(np.max(num)) - float(np.max(exact))) / scale),
        "min_error_ratio": float(abs(float(np.min(num)) - float(np.min(exact))) / scale),
        "range_ratio": float((float(np.max(num)) - float(np.min(num))) / exact_range),
    }


def _has_sharp_jump(exact, scale, fraction=0.5):
    exact = np.asarray(exact, dtype=float)
    if exact.size < 2:
        return False
    return bool(np.any(np.abs(np.diff(exact)) > fraction * max(float(scale), 1.0)))


def _temperature_scale(exact, mask):
    exact = np.asarray(exact, dtype=float)
    return max(float(np.max(exact[mask]) - np.min(exact[mask])),
               float(np.mean(np.abs(exact[mask]))), 1.0)


def _temperature_l1(num, exact, mask):
    return _range_l1(num, exact, floor=_temperature_scale(exact, mask), mask=mask)


def _temperature_linf(num, exact, mask):
    num = np.asarray(num, dtype=float)
    exact = np.asarray(exact, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    return float(np.max(np.abs(num[mask] - exact[mask])) / _temperature_scale(exact, mask))


def _corr(num, exact):
    n = np.asarray(num, dtype=float) - float(np.mean(num))
    e = np.asarray(exact, dtype=float) - float(np.mean(exact))
    den = float(np.sqrt(np.dot(n, n) * np.dot(e, e)))
    if den <= 1.0e-300:
        return 1.0
    return float(np.dot(n, e) / den)


def _checkerboard(y, scale):
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return 0.0
    d2 = y[1:-1] - 0.5 * (y[:-2] + y[2:])
    return float(np.sqrt(np.mean(d2 * d2)) / max(abs(float(scale)), 1.0))


def _masked_checkerboard_error(num, exact, scale, mask=None):
    num = np.asarray(num, dtype=float)
    exact = np.asarray(exact, dtype=float)
    if num.size < 3:
        return 0.0
    err = (num - exact) / max(abs(float(scale)), 1.0)
    d2 = err[1:-1] - 0.5 * (err[:-2] + err[2:])
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        valid = mask[:-2] & mask[1:-1] & mask[2:]
        if int(np.count_nonzero(valid)) < 1:
            return 0.0
        d2 = d2[valid]
    return float(np.sqrt(np.mean(d2 * d2)))


def _masked_checkerboard_error_max(num, exact, scale, mask=None):
    num = np.asarray(num, dtype=float)
    exact = np.asarray(exact, dtype=float)
    if num.size < 3:
        return 0.0
    err = (num - exact) / max(abs(float(scale)), 1.0)
    d2 = err[1:-1] - 0.5 * (err[:-2] + err[2:])
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        valid = mask[:-2] & mask[1:-1] & mask[2:]
        if int(np.count_nonzero(valid)) < 1:
            return 0.0
        d2 = d2[valid]
    return float(np.max(np.abs(d2))) if d2.size else 0.0


def _masked_local_tv_excess(num, exact, scale, mask=None):
    num = np.asarray(num, dtype=float)
    exact = np.asarray(exact, dtype=float)
    if num.size < 2:
        return 0.0
    dn = np.abs(np.diff(num))
    de = np.abs(np.diff(exact))
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        valid = mask[:-1] & mask[1:]
        if int(np.count_nonzero(valid)) < 1:
            return 0.0
        dn = dn[valid]
        de = de[valid]
    excess = np.maximum(dn - de, 0.0)
    return float(np.mean(excess) / max(abs(float(scale)), 1.0))


def _masked_local_tv_excess_max(num, exact, scale, mask=None):
    num = np.asarray(num, dtype=float)
    exact = np.asarray(exact, dtype=float)
    if num.size < 2:
        return 0.0
    dn = np.abs(np.diff(num))
    de = np.abs(np.diff(exact))
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        valid = mask[:-1] & mask[1:]
        if int(np.count_nonzero(valid)) < 1:
            return 0.0
        dn = dn[valid]
        de = de[valid]
    excess = np.maximum(dn - de, 0.0)
    return float(np.max(excess) / max(abs(float(scale)), 1.0)) if excess.size else 0.0


def _solve_case(case):
    eos1 = _make_water_nasg()
    eos2 = _make_air_ideal()
    x, W0, folder, title, dx, n = _initial(case)
    dt_fixed = _case_dt(case)
    max_steps = int(math.ceil(T_END / max(dt_fixed, 1.0e-300))) + 5
    start = time.time()
    out = solve(
        eos1,
        eos2,
        W0,
        dx,
        T_END,
        bc_l="periodic",
        bc_r="periodic",
        dt_fixed=dt_fixed,
        max_steps=max_steps,
        time_integrator=os.environ.get("FIVE_EQ_IMEX_TIME_INTEGRATOR", "imex_ad"),
        alpha_scheme=os.environ.get("FIVE_EQ_IMEX_ALPHA_SCHEME", "adaptive_bvd"),
        primitive_scheme=os.environ.get("FIVE_EQ_IMEX_PRIMITIVE_SCHEME", "tmlpu"),
        mixture_kind="kapila",
        kapila_closure=True,
        pure_branch=True,
        alpha_pure_tol=ALPHA_FLOOR,
        pressure_closure=os.environ.get("FIVE_EQ_IMEX_PRESSURE_CLOSURE", "regime_auto"),
        dt_min=1.0e-14,
    )
    wall = time.time() - start
    W = out["W"]
    rho = _rho_mix(W, eos1, eos2)
    Wex, rho_ex = _exact_from_initial(W0, x, T_END, eos1, eos2)
    T_mix = _temperature_mix(W, eos1, eos2)
    T_mix_ex = _temperature_mix(Wex, eos1, eos2)

    complete = out.get("terminated_reason") is None and out["t_final"] >= T_END - 1.0e-14
    finite = bool(all(np.all(np.isfinite(v)) for v in W) and np.all(np.isfinite(rho)))
    alpha_ok = bool(np.min(W[0]) >= -1.0e-10 and np.max(W[0]) <= 1.0 + 1.0e-10)
    p_rel = _rel_linf(W[4], Wex[4], P0)
    u_abs = float(np.max(np.abs(W[3] - Wex[3])))
    active1 = Wex[0] > ACTIVE_PHASE_TOL
    active2 = (1.0 - Wex[0]) > ACTIVE_PHASE_TOL
    T1_l1 = _temperature_l1(W[1], Wex[1], active1)
    T2_l1 = _temperature_l1(W[2], Wex[2], active2)
    T1_linf = _temperature_linf(W[1], Wex[1], active1)
    T2_linf = _temperature_linf(W[2], Wex[2], active2)
    T1_scale = _temperature_scale(Wex[1], active1)
    T2_scale = _temperature_scale(Wex[2], active2)
    Tmix_scale = max(float(np.max(T_mix_ex) - np.min(T_mix_ex)),
                     float(np.mean(np.abs(T_mix_ex))), 1.0)
    alpha_l1 = _range_l1(W[0], Wex[0], 1.0)
    rho_l1 = _range_l1(rho, rho_ex, 1.0)
    alpha_range_ratio = _range_ratio(W[0], Wex[0], 1.0e-12)
    rho_range_ratio = _range_ratio(rho, rho_ex, 1.0e-12)
    rho_scale = max(float(np.max(rho_ex) - np.min(rho_ex)), 1.0)
    smooth_guard_mask = np.ones_like(W[0], dtype=bool)
    if case == "16":
        # 16_T intentionally contains a sharp material contact.  The smoothness
        # guard should not punish the physical jump itself; active-phase
        # temperature guards still apply away from absent phases.
        smooth_guard_mask[:] = False
    metrics = {
        "case": folder,
        "finite": finite,
        "complete": bool(complete),
        "terminated_reason": out.get("terminated_reason"),
        "steps": int(out["step"]),
        "t_final": float(out["t_final"]),
        "dt_fixed": dt_fixed,
        "N": int(n),
        "dx": float(dx),
        "courant": float(abs(U0) * dt_fixed / dx),
        "wall": wall,
        "alpha_min": float(np.min(W[0])),
        "alpha_max": float(np.max(W[0])),
        "p_rel_linf": p_rel,
        "u_abs_linf": u_abs,
        "T1_l1_ratio": T1_l1,
        "T2_l1_ratio": T2_l1,
        "T1_linf_ratio": T1_linf,
        "T2_linf_ratio": T2_linf,
        "Tmix_l1_ratio": _range_l1(T_mix, T_mix_ex, Tmix_scale),
        "Tmix_linf_ratio": float(np.max(np.abs(T_mix - T_mix_ex)) / Tmix_scale),
        "T1_active_cells": int(np.count_nonzero(active1)),
        "T2_active_cells": int(np.count_nonzero(active2)),
        "alpha_l1_ratio": alpha_l1,
        "rho_l1_ratio": rho_l1,
        "alpha_range_ratio": alpha_range_ratio,
        "rho_range_ratio": rho_range_ratio,
        "alpha_corr": _corr(W[0], Wex[0]),
        "rho_corr": _corr(rho, rho_ex),
        "p_checkerboard": _checkerboard(W[4], P0),
        "u_checkerboard": _checkerboard(W[3], max(abs(U0), 1.0)),
        "rho_checkerboard": _checkerboard(rho, max(float(np.max(rho_ex) - np.min(rho_ex)), 1.0)),
        "T1_active_hf_error": _masked_checkerboard_error(W[1], Wex[1], T1_scale, active1),
        "T2_active_hf_error": _masked_checkerboard_error(W[2], Wex[2], T2_scale, active2),
        "T1_active_hf_max_error": _masked_checkerboard_error_max(W[1], Wex[1], T1_scale, active1),
        "T2_active_hf_max_error": _masked_checkerboard_error_max(W[2], Wex[2], T2_scale, active2),
        "T1_active_tv_excess": _masked_local_tv_excess(W[1], Wex[1], T1_scale, active1),
        "T2_active_tv_excess": _masked_local_tv_excess(W[2], Wex[2], T2_scale, active2),
        "T1_active_tv_excess_max": _masked_local_tv_excess_max(W[1], Wex[1], T1_scale, active1),
        "T2_active_tv_excess_max": _masked_local_tv_excess_max(W[2], Wex[2], T2_scale, active2),
        "alpha_smooth_hf_error": _masked_checkerboard_error(W[0], Wex[0], 1.0, smooth_guard_mask),
        "rho_smooth_hf_error": _masked_checkerboard_error(rho, rho_ex, rho_scale, smooth_guard_mask),
        "alpha_smooth_hf_max_error": _masked_checkerboard_error_max(W[0], Wex[0], 1.0, smooth_guard_mask),
        "rho_smooth_hf_max_error": _masked_checkerboard_error_max(rho, rho_ex, rho_scale, smooth_guard_mask),
        "alpha_smooth_tv_excess": _masked_local_tv_excess(W[0], Wex[0], 1.0, smooth_guard_mask),
        "rho_smooth_tv_excess": _masked_local_tv_excess(rho, rho_ex, rho_scale, smooth_guard_mask),
        "alpha_smooth_tv_excess_max": _masked_local_tv_excess_max(W[0], Wex[0], 1.0, smooth_guard_mask),
        "rho_smooth_tv_excess_max": _masked_local_tv_excess_max(rho, rho_ex, rho_scale, smooth_guard_mask),
        "T1_min": float(np.min(W[1])),
        "T2_min": float(np.min(W[2])),
    }
    tmix_has_sharp_jump = _has_sharp_jump(T_mix_ex, Tmix_scale)
    tmix_bounded = bool(
        float(np.min(T_mix)) >= float(np.min(T_mix_ex)) - 1.0e-10 * Tmix_scale
        and float(np.max(T_mix)) <= float(np.max(T_mix_ex)) + 1.0e-10 * Tmix_scale
    )
    metrics["Tmix_has_sharp_jump"] = bool(tmix_has_sharp_jump)
    metrics["Tmix_bounded"] = bool(tmix_bounded)
    case17_peak_ok = True
    if case == "17":
        alpha_extrema = _extrema_match_metrics(W[0], Wex[0], 1.0)
        rho_extrema = _extrema_match_metrics(rho, rho_ex, rho_scale)
        tmix_extrema = _extrema_match_metrics(T_mix, T_mix_ex, Tmix_scale)
        metrics.update({
            "case17_alpha_peak_error_ratio": alpha_extrema["max_error_ratio"],
            "case17_alpha_floor_error_ratio": alpha_extrema["min_error_ratio"],
            "case17_alpha_peak_range_ratio": alpha_extrema["range_ratio"],
            "case17_rho_peak_error_ratio": rho_extrema["max_error_ratio"],
            "case17_rho_floor_error_ratio": rho_extrema["min_error_ratio"],
            "case17_rho_peak_range_ratio": rho_extrema["range_ratio"],
            "case17_rho_peak_amp_ratio_min": RHO_PEAK_RATIO_MIN,
            "case17_rho_peak_amp_ratio_max": RHO_PEAK_RATIO_MAX,
            "case17_Tmix_peak_error_ratio": tmix_extrema["max_error_ratio"],
            "case17_Tmix_valley_error_ratio": tmix_extrema["min_error_ratio"],
            "case17_Tmix_peak_range_ratio": tmix_extrema["range_ratio"],
            "case17_extrema_error_limit": CASE17_EXTREMA_ERROR_TOL,
            "case17_range_ratio_min": CASE17_RANGE_RATIO_MIN,
            "case17_range_ratio_max": CASE17_RANGE_RATIO_MAX,
        })
        case17_peak_ok = bool(
            alpha_extrema["max_error_ratio"] < CASE17_EXTREMA_ERROR_TOL
            and alpha_extrema["min_error_ratio"] < CASE17_EXTREMA_ERROR_TOL
            and rho_extrema["max_error_ratio"] < CASE17_EXTREMA_ERROR_TOL
            and rho_extrema["min_error_ratio"] < CASE17_EXTREMA_ERROR_TOL
            and tmix_extrema["max_error_ratio"] < CASE17_EXTREMA_ERROR_TOL
            and tmix_extrema["min_error_ratio"] < CASE17_EXTREMA_ERROR_TOL
            and alpha_extrema["range_ratio"] > CASE17_RANGE_RATIO_MIN
            and rho_extrema["range_ratio"] > RHO_PEAK_RATIO_MIN
            and tmix_extrema["range_ratio"] > CASE17_RANGE_RATIO_MIN
            and alpha_extrema["range_ratio"] < CASE17_RANGE_RATIO_MAX
            and rho_extrema["range_ratio"] < RHO_PEAK_RATIO_MAX
            and tmix_extrema["range_ratio"] < CASE17_RANGE_RATIO_MAX
        )
    metrics["case17_peak_ok"] = bool(case17_peak_ok)
    alpha_rho_ok = (
        alpha_l1 < ALPHA_RHO_GENERAL_L1_TOL
        and rho_l1 < ALPHA_RHO_GENERAL_L1_TOL
    )
    if case in {"16", "17"}:
        alpha_rho_ok = (
            alpha_l1 < ALPHA_RHO_TRANSPORT_L1_TOL
            and rho_l1 < ALPHA_RHO_TRANSPORT_L1_TOL
            and alpha_range_ratio > ALPHA_RHO_TRANSPORT_RANGE_MIN
            and rho_range_ratio > ALPHA_RHO_TRANSPORT_RANGE_MIN
        )
    elif case == "18":
        alpha_rho_ok = (
            alpha_l1 < ALPHA_RHO_THERMAL_L1_TOL
            and rho_l1 < ALPHA_RHO_THERMAL_L1_TOL
            and alpha_range_ratio > ALPHA_RHO_THERMAL_RANGE_MIN
            and rho_range_ratio > ALPHA_RHO_THERMAL_RANGE_MIN
        )
    tmix_transport_ok = True
    if case in {"16", "17"}:
        if tmix_has_sharp_jump:
            # A discontinuous exact mixture temperature cannot be judged by
            # pointwise Linf at the contact: any finite-volume bounded scheme
            # has a finite interface thickness.  Use integral error plus
            # boundedness; alpha/rho L1/range guards below control the
            # interface diffusion and shape.
            tmix_transport_ok = bool(
                metrics["Tmix_l1_ratio"] < T_MIX_TRANSPORT_L1_TOL
                and tmix_bounded
            )
        else:
            tmix_transport_ok = bool(
                metrics["Tmix_l1_ratio"] < T_MIX_TRANSPORT_L1_TOL
                and metrics["Tmix_linf_ratio"] < T_MIX_TRANSPORT_LINF_TOL
            )
    metrics["tmix_transport_ok"] = bool(tmix_transport_ok)
    case18_wiggle_ok = True
    if case == "18":
        metrics["case18_rho_peak_amp_ratio"] = float(rho_range_ratio)
        metrics["case18_rho_peak_amp_ratio_min"] = float(RHO_PEAK_RATIO_MIN)
        metrics["case18_rho_peak_amp_ratio_max"] = float(RHO_PEAK_RATIO_MAX)
        metrics["case18_rho_peak_ok"] = bool(
            RHO_PEAK_RATIO_MIN <= rho_range_ratio <= RHO_PEAK_RATIO_MAX
        )
        case18_wiggle_ok = bool(
            metrics["T1_active_hf_max_error"] < T_ACTIVE_HF_MAX_TOL
            and metrics["T2_active_hf_max_error"] < T_ACTIVE_HF_MAX_TOL
            and metrics["T1_active_tv_excess_max"] < T_ACTIVE_TV_EXCESS_MAX_TOL
            and metrics["T2_active_tv_excess_max"] < T_ACTIVE_TV_EXCESS_MAX_TOL
            and metrics["alpha_smooth_hf_max_error"] < THERMAL_ALPHA_SMOOTH_HF_MAX_TOL
            and metrics["rho_smooth_hf_max_error"] < THERMAL_RHO_SMOOTH_HF_MAX_TOL
            and metrics["alpha_smooth_tv_excess_max"] < THERMAL_ALPHA_SMOOTH_TV_EXCESS_MAX_TOL
            and metrics["rho_smooth_tv_excess_max"] < THERMAL_RHO_SMOOTH_TV_EXCESS_MAX_TOL
        )
    metrics["case18_wiggle_ok"] = bool(case18_wiggle_ok)
    ok = bool(
        finite and complete and alpha_ok
        and p_rel < 1.0e-8
        and u_abs < 1.0e-8
        and T1_l1 < T_MEAN_TOL
        and T2_l1 < T_MEAN_TOL
        and T1_linf < T_LINF_TOL
        and T2_linf < T_LINF_TOL
        and alpha_rho_ok
        and metrics["p_checkerboard"] < 1.0e-8
        and metrics["u_checkerboard"] < 1.0e-8
        and metrics["T1_active_hf_error"] < T_ACTIVE_HF_TOL
        and metrics["T2_active_hf_error"] < T_ACTIVE_HF_TOL
        and metrics["T1_active_tv_excess"] < T_ACTIVE_TV_EXCESS_TOL
        and metrics["T2_active_tv_excess"] < T_ACTIVE_TV_EXCESS_TOL
        and metrics["alpha_smooth_hf_error"] < SMOOTH_HF_TOL
        and metrics["rho_smooth_hf_error"] < SMOOTH_HF_TOL
        and metrics["alpha_smooth_tv_excess"] < SMOOTH_TV_EXCESS_TOL
        and metrics["rho_smooth_tv_excess"] < SMOOTH_TV_EXCESS_TOL
        and tmix_transport_ok
        and case17_peak_ok
        and case18_wiggle_ok
        and (case != "18" or metrics.get("case18_rho_peak_ok", False))
        and metrics["T1_min"] > 0.0
        and metrics["T2_min"] > 0.0
    )
    metrics["pass"] = ok

    out_dir = ROOT / "results" / "1D" / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(3, 3, figsize=(14, 10))
    if case == "18":
        fields = [
            ("alpha1", W[0], Wex[0]),
            ("rho", rho, rho_ex),
            ("u", W[3], Wex[3]),
            ("p", W[4], Wex[4]),
            ("T_liquid", W[1], Wex[1]),
            ("T_gas", W[2], Wex[2]),
        ]
    else:
        fields = [
            ("alpha1", W[0], Wex[0]),
            ("rho", rho, rho_ex),
            ("u", W[3], Wex[3]),
            ("p", W[4], Wex[4]),
            ("T_mixture", T_mix, T_mix_ex),
        ]
    for idx, (name, num, exact) in enumerate(fields):
        r, c = divmod(idx, 3)
        ax[r, c].plot(x, num, "b-", lw=1.8, label="num")
        ax[r, c].plot(x, exact, "r--", lw=1.2, label="exact")
        ax[r, c].set_title(name)
        ax[r, c].grid(alpha=0.3)
        ax[r, c].legend(fontsize=8)
    if case != "18":
        ax[1, 2].plot(x, np.abs(T_mix - T_mix_ex), "k-", lw=1.2)
        ax[1, 2].set_title("|T_mixture-T_exact|")
        ax[1, 2].grid(alpha=0.3)
    err_pairs = [("|p-p_exact|", np.abs(W[4] - Wex[4])), ("|u-u_exact|", np.abs(W[3] - Wex[3])), ("|rho-rho_exact|", np.abs(rho - rho_ex))]
    for j, (name, err) in enumerate(err_pairs):
        ax[2, j].plot(x, err, "k-", lw=1.2)
        ax[2, j].set_title(name)
        ax[2, j].grid(alpha=0.3)
    fig.suptitle(f"{title} pass={ok} steps={out['step']} p_rel={p_rel:.2e} u_abs={u_abs:.2e}")
    fig.tight_layout()
    fig.savefig(out_dir / "diff_vs_exact.png", dpi=120)
    plt.close(fig)
    print(f"Plot saved: results/1D/{folder}/diff_vs_exact.png")
    print("CASE_JSON " + json.dumps(metrics, sort_keys=True))
    return metrics


def case_16():
    return _solve_case("16")


def case_17():
    return _solve_case("17")


def case_18():
    return _solve_case("18")


def case_19():
    return _solve_case("19")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=["16", "17", "18", "19", "mandatory", "optional", "all"],
        default="mandatory",
        help=(
            "'mandatory'/'all' runs 16_T, 17_T, 18_T. "
            "'optional' runs deprecated optional 19_T only."
        ),
    )
    args = parser.parse_args()
    if args.case in {"mandatory", "all"}:
        cases = MANDATORY_CASES
    elif args.case == "optional":
        cases = OPTIONAL_CASES
    else:
        cases = [args.case]
    results = [_solve_case(c) for c in cases]
    failures = sum(0 if r["pass"] else 1 for r in results)
    print("SUMMARY_JSON " + json.dumps({"failures": failures, "cases": results}, sort_keys=True))
    print(failures)
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
