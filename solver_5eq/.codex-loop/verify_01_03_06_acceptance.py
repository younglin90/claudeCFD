#!/usr/bin/env python3
"""Sequential non-shock acceptance checks for five_eq_IMEX IMEX_AD.

Run one case at a time:

    python3 .codex-loop/verify_01_03_06_acceptance.py --case 01

The cases follow validation/1D specs for 01, 03, 04, 05, and 06.  Plots are
always overwritten at results/1D/{case}/diff_vs_exact.png.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solver.five_eq_IMEX.eos_facade import make_eos  # noqa: E402
from solver.five_eq_IMEX.main import solve  # noqa: E402
from solver.five_eq_IMEX.sound_speed import phase_sound_speed_sq, mixture_sound_speed_sq  # noqa: E402
from oscillation_guards import high_frequency_oscillation_guard  # noqa: E402


P0 = 1.0e5
CASE05_PEAK_RATIO_MIN = 0.98
CASE05_PEAK_RATIO_MAX = 1.02


def _make_water_nasg():
    return make_eos(
        "nasg",
        gamma=1.187,
        pinf=7.028e8,
        kv=3610.0,
        b=6.61e-4,
        eta=-1.177788e6,
    )


def _out_dir(case_name: str) -> str:
    path = os.path.join(ROOT, "results", "1D", case_name)
    os.makedirs(path, exist_ok=True)
    return path


def _temperature_for_rho_p(eos, rho: np.ndarray | float, p: np.ndarray | float) -> np.ndarray:
    rho_a = np.asarray(rho, dtype=float)
    p_a = np.asarray(p, dtype=float)
    return eos.temperature(rho_a, eos.energy(rho_a, p_a))


def _sound_speed_for_rho_p(eos, rho: float, p: float) -> float:
    rho_a = np.asarray([float(rho)], dtype=float)
    p_a = np.asarray([float(p)], dtype=float)
    e_a = eos.energy(rho_a, p_a)
    return float(np.sqrt(np.maximum(eos.sound_speed_sq(rho_a, e_a, p_a)[0], 1.0e-300)))


def _rho_mix(W, eos1, eos2) -> np.ndarray:
    a, T1, T2, _, p = W
    rho1 = eos1.density(p, T1)
    rho2 = eos2.density(p, T2)
    return a * rho1 + (1.0 - a) * rho2


def _finite(W) -> bool:
    return bool(all(np.all(np.isfinite(c)) for c in W))


def _checkerboard(p: np.ndarray, p0: float, mask: np.ndarray | None = None) -> float:
    if mask is None or int(np.sum(mask)) < 4:
        arr = np.asarray(p, dtype=float)
    else:
        arr = np.asarray(p, dtype=float)[mask]
    if arr.size < 4:
        return 0.0
    d2 = arr[1:-1] - 0.5 * (arr[:-2] + arr[2:])
    return float(np.sqrt(np.mean(d2 * d2)) / max(abs(p0), 1.0))


def _measure_lambda(x: np.ndarray, signal: np.ndarray, mask: np.ndarray) -> float:
    idx = np.flatnonzero(mask)
    if idx.size < 20:
        return 0.0
    xx = x[idx]
    yy = np.asarray(signal, dtype=float)[idx]
    yy = yy - float(np.mean(yy))
    if float(np.max(np.abs(yy))) <= 1.0e-14:
        return 0.0
    zc = np.flatnonzero(np.diff(np.signbit(yy)))
    if zc.size < 2:
        return 0.0
    crossings = 0.5 * (xx[zc] + xx[zc + 1])
    return float(2.0 * np.mean(np.diff(crossings)))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float) - float(np.mean(a))
    bb = np.asarray(b, dtype=float) - float(np.mean(b))
    den = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
    if den <= 1.0e-300:
        return 1.0
    return float(np.sum(aa * bb) / den)


def _shape_metrics(num: np.ndarray, exact: np.ndarray, mask: np.ndarray,
                   amp_floor: float) -> dict[str, float]:
    """Profile similarity allowing amplitude damping.

    We require a non-negligible wave, correct phase/shape correlation, and a
    small residual after least-squares amplitude rescaling.  This intentionally
    does not require the numerical amplitude to match the exact amplitude.
    """
    if int(np.sum(mask)) < 8:
        return {"corr": 0.0, "scaled_l2": float("inf"), "amp_ratio": 0.0, "scale": 0.0}
    n = np.asarray(num, dtype=float)[mask]
    e = np.asarray(exact, dtype=float)[mask]
    n = n - float(np.mean(n))
    e = e - float(np.mean(e))
    e_norm2 = float(np.dot(e, e))
    n_amp = float(np.max(np.abs(n)))
    e_amp = float(np.max(np.abs(e)))
    if e_norm2 <= 1.0e-300 or e_amp <= 1.0e-300:
        return {"corr": 0.0, "scaled_l2": float("inf"), "amp_ratio": 0.0, "scale": 0.0}
    scale = float(np.dot(n, e) / e_norm2)
    residual = n - scale * e
    scaled_l2 = float(np.sqrt(np.mean(residual * residual)) / max(abs(scale) * e_amp, amp_floor))
    return {
        "corr": _pearson(n, e),
        "scaled_l2": scaled_l2,
        "amp_ratio": n_amp / e_amp,
        "scale": scale,
    }


def _amp_ratio_ok(value: float, *, lo: float = CASE05_PEAK_RATIO_MIN,
                  hi: float = CASE05_PEAK_RATIO_MAX) -> bool:
    return bool(lo <= float(value) <= hi)


def _save_plot(case_name: str, x: np.ndarray, W, rho: np.ndarray,
               exact: dict[str, np.ndarray], title: str) -> None:
    out = _out_dir(case_name)
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    fields = [
        ("rho", rho, exact["rho"]),
        ("u", W[3], exact["u"]),
        ("p", W[4], exact["p"]),
    ]
    for j, (name, num, ex) in enumerate(fields):
        ax[0, j].plot(x, num, "b-", lw=1.4, label="num")
        ax[0, j].plot(x, ex, "r--", lw=1.1, label="exact")
        ax[0, j].set_title(name)
        ax[0, j].grid(alpha=0.3)
        ax[0, j].legend(fontsize=8)
        ax[1, j].plot(x, np.abs(np.asarray(num) - np.asarray(ex)), "k-", lw=1.0)
        ax[1, j].set_title(f"abs error {name}")
        ax[1, j].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "diff_vs_exact.png"), dpi=300)
    fig.savefig(os.path.join(out, "diff_vs_exact.pdf"))
    plt.close(fig)
    print(f"Plot saved: results/1D/{case_name}/diff_vs_exact.png")


def _solve_imex(eos1, eos2, W0, dx, t_end, *, bc_l, bc_r, cfl=0.4,
                dt_fixed=None, u_inlet=None, p_inlet=None, max_steps=100000,
                alpha_pure_tol=1.0e-8, kapila_closure=False):
    pressure_closure = os.environ.get("FIVE_EQ_IMEX_PRESSURE_CLOSURE", "regime_auto")
    alpha_scheme = os.environ.get("FIVE_EQ_IMEX_ALPHA_SCHEME", "adaptive_bvd")
    primitive_scheme = os.environ.get("FIVE_EQ_IMEX_PRIMITIVE_SCHEME", "weno3")
    return solve(
        eos1,
        eos2,
        W0,
        dx,
        t_end,
        bc_l=bc_l,
        bc_r=bc_r,
        cfl=cfl,
        dt_fixed=dt_fixed,
        max_steps=max_steps,
        time_integrator=os.environ.get("FIVE_EQ_IMEX_TIME_INTEGRATOR", "imex_ssp3"),
        alpha_scheme=alpha_scheme,
        mixture_kind="kapila",
        kapila_closure=kapila_closure,
        pure_branch=True,
        alpha_pure_tol=alpha_pure_tol,
        primitive_scheme=primitive_scheme,
        pressure_closure=pressure_closure,
        u_inlet=u_inlet,
        p_inlet=p_inlet,
    )


def case_01() -> dict:
    eos_air = make_eos("ideal", gamma=1.4, kv=717.5)
    eos_water = _make_water_nasg()
    n = 100
    dx = 1.0 / n
    x = (np.arange(n) + 0.5) * dx
    p0 = P0
    T0 = 293.0
    a = np.where(x < 0.5, 1.0 - 1.0e-6, 1.0e-6)
    W0 = (a, np.full(n, T0), np.full(n, T0), np.zeros(n), np.full(n, p0))
    t_end = 1.0
    dt_fixed = 0.01
    t0 = time.time()
    out = _solve_imex(
        eos_air, eos_water, W0, dx, t_end,
        bc_l="transmissive", bc_r="transmissive", cfl=0.4,
        dt_fixed=dt_fixed,
        alpha_pure_tol=1.0e-6)
    wall = time.time() - t0
    W = out["W"]
    rho = _rho_mix(W, eos_air, eos_water)
    exact = {"rho": _rho_mix(W0, eos_air, eos_water), "u": W0[3], "p": W0[4]}
    p_rel = float(np.max(np.abs(W[4] - p0)) / p0)
    u_abs = float(np.max(np.abs(W[3])))
    osc = _checkerboard(W[4], p0)
    hf = high_frequency_oscillation_guard(
        x,
        {
            "rho": (rho, exact["rho"], max(float(np.ptp(exact["rho"])), 1.0)),
            "u": (W[3], exact["u"], 1.0),
            "p": (W[4], exact["p"], p0),
        },
        sharp_centers=(0.5,),
    )
    complete = out.get("terminated_reason") is None and out["t_final"] >= t_end - 1.0e-12
    ok = bool(
        _finite(W) and complete
        and p_rel < 1.0e-10 and u_abs < 1.0e-6 and osc < 1.0e-4
        and hf["hf_oscillation_ok"]
    )
    _save_plot("01_A", x, W, rho, exact,
               f"01_A static PE pass={ok} p_rel={p_rel:.2e} u={u_abs:.2e} osc={osc:.2e}")
    return {
        "case": "01_A",
        "pass": ok,
        "wall": wall,
        "steps": int(out["step"]),
        "t_final": float(out["t_final"]),
        "t_end": t_end,
        "dt_fixed": dt_fixed,
        "complete": bool(complete),
        "p_rel": p_rel,
        "u_abs": u_abs,
        "osc": osc,
        **hf,
    }


def case_03() -> dict:
    eos_air = make_eos("ideal", gamma=1.4, kv=717.5)
    eos_water = make_eos("sg", gamma=4.4, pinf=6.0e8, kv=474.2)
    n = 200
    dx = 1.0 / n
    x = (np.arange(n) + 0.5) * dx
    p0 = P0
    T0 = 293.0
    dp0 = 1.0
    a = np.full(n, 1.0e-6)
    p_init = np.where(np.abs(x - 0.5) < 0.1, p0 + dp0, p0)
    W0 = (a, np.full(n, T0), np.full(n, T0), np.zeros(n), p_init)
    rho2_bg = float(eos_water.density(np.array([p0]), np.array([T0]))[0])
    c0 = float(np.sqrt(4.4 * (p0 + 6.0e8) / rho2_bg))
    t_end = 3.0e-4
    t0 = time.time()
    out = _solve_imex(
        eos_air, eos_water, W0, dx, t_end,
        bc_l="transmissive", bc_r="transmissive", cfl=0.4,
        alpha_pure_tol=1.0e-6)
    wall = time.time() - t0
    W = out["W"]
    rho = _rho_mix(W, eos_air, eos_water)
    t = out["t_final"]
    p_exact = np.full(n, p0)
    left = np.abs(x - (0.5 - c0 * t)) < 0.1
    right = np.abs(x - (0.5 + c0 * t)) < 0.1
    p_exact[left] += 0.5 * dp0
    p_exact[right] += 0.5 * dp0
    u_exact = np.zeros(n)
    u_exact[right] += 0.5 * dp0 / (rho2_bg * c0)
    u_exact[left] -= 0.5 * dp0 / (rho2_bg * c0)
    rho_exact = _rho_mix(W0, eos_air, eos_water) + (p_exact - p_init) / max(c0 * c0, 1.0)
    exact = {"rho": rho_exact, "u": u_exact, "p": p_exact}
    dp_max = float(np.max(np.abs(W[4] - p0)))
    osc = _checkerboard(W[4], p0)
    complete = out.get("terminated_reason") is None and out["t_final"] >= t_end - 1.0e-14
    p_l2 = float(np.sqrt(np.mean((W[4] - p_exact) ** 2)))
    ok = bool(_finite(W) and complete and 0.45 <= dp_max <= 0.55 and osc < 1.0e-4)
    _save_plot("03_B", x, W, rho, exact,
               f"03_B low-Mach pulse pass={ok} dp_max={dp_max:.3f}Pa osc={osc:.2e}")
    return {
        "case": "03_B",
        "pass": ok,
        "wall": wall,
        "steps": int(out["step"]),
        "complete": bool(complete),
        "dp_max": dp_max,
        "p_l2": p_l2,
        "osc": osc,
    }


def _single_phase_acoustic(case_name: str, *, material: str) -> dict:
    if material == "air":
        eos1 = make_eos("ideal", gamma=1.4, kv=717.5)
        eos2 = make_eos("ideal", gamma=1.4, kv=717.5)
        n = int(os.environ.get("FIVE_EQ_CASE04_N", "500"))
        p0, rho0, u0, f, t_end = P0, 1.157, 1.0, 2000.0, 2.3e-3
        alpha = 1.0 - 1.0e-6
        phase1 = True
        amp_floor_ratio = 0.10
        osc_limit = 1.0e-3
    elif material == "water":
        eos_air = make_eos("ideal", gamma=1.4, kv=717.5)
        eos_water = _make_water_nasg()
        eos1, eos2 = eos_air, eos_water
        n = int(os.environ.get("FIVE_EQ_CASE05_N", "400"))
        # Match 04_B's layout: the injected wave front remains inside the tube
        # with about 20% undisturbed length at the right boundary.
        p0, rho0, u0, f, t_end = P0, 998.0, 1.0, 6000.0, 5.10e-4
        alpha = 1.0e-6
        phase1 = False
        amp_floor_ratio = 0.10
        osc_limit = 5.0e-2
    else:
        raise ValueError(material)

    L = 1.0
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    c0 = _sound_speed_for_rho_p(eos1 if phase1 else eos2, rho0, p0)
    du = 0.01 * u0
    dp_amp = rho0 * c0 * du
    lam0 = c0 / f
    T_main = _temperature_for_rho_p(eos1 if phase1 else eos2, rho0, p0)
    if phase1:
        T1 = np.full(n, float(T_main))
        T2 = T1.copy()
    else:
        T1 = np.full(n, 293.0)
        T2 = np.full(n, float(T_main))
    a = np.full(n, alpha)
    W0 = (a, T1, T2, np.full(n, u0), np.full(n, p0))

    def u_in(t: float) -> float:
        return u0 + du * math.sin(2.0 * math.pi * f * t)

    def p_in(t: float) -> float:
        return p0 + dp_amp * math.sin(2.0 * math.pi * f * t)

    t0 = time.time()
    out = _solve_imex(
        eos1, eos2, W0, dx, t_end,
        bc_l="inlet", bc_r="transmissive", cfl=0.4,
        u_inlet=u_in, p_inlet=p_in,
        alpha_pure_tol=1.0e-6)
    wall = time.time() - t0
    W = out["W"]
    rho = _rho_mix(W, eos1, eos2)
    tau = out["t_final"] - x / c0
    touched = tau > 0.0
    p_exact = np.full(n, p0)
    u_exact = np.full(n, u0)
    rho_phase_exact = np.full(n, rho0)
    phase = np.sin(2.0 * math.pi * f * tau[touched])
    p_exact[touched] = p0 + dp_amp * phase
    u_exact[touched] = u0 + du * phase
    rho_phase_exact[touched] = rho0 + rho0 * du / c0 * phase
    if phase1:
        rho_exact = rho_phase_exact
    else:
        # The plotted numerical density is rho_mix, not the pure-water phase
        # density.  Include the alpha_air floor in the reference for a
        # like-for-like comparison.
        rho_air_exact = eos1.density(p_exact, T1)
        rho_exact = alpha * rho_air_exact + (1.0 - alpha) * rho_phase_exact
    exact = {"rho": rho_exact, "u": u_exact, "p": p_exact}
    rho_floor = max(float(np.max(rho_exact) - np.min(rho_exact)), rho0 * du / max(c0, 1.0e-300))
    hf = high_frequency_oscillation_guard(
        x,
        {
            "rho": (rho, exact["rho"], rho_floor),
            "u": (W[3], exact["u"], du),
            "p": (W[4], exact["p"], dp_amp),
        },
        # The exact solution is a physical sinusoidal wave.  Its normal
        # extrema should not be counted as nonphysical shock/contact ringing;
        # residual HF, overshoot, and excess TV checks remain active.
        sharp_turn_limit=12,
        smooth_local_turn_limit=12,
    )
    dp_meas = 0.5 * float(np.max(W[4] - p0) - np.min(W[4] - p0))
    wave_region = touched & (x < 0.95 * c0 * out["t_final"])
    lam = _measure_lambda(x, W[4] - p0, wave_region)
    osc = _checkerboard(W[4], p0, ~touched)
    complete = out.get("terminated_reason") is None and out["t_final"] >= t_end - 1.0e-14
    lambda_ok = lam > 0.0 and abs(lam - lam0) / lam0 < 0.10
    p_l2 = float(np.sqrt(np.mean((W[4][touched] - p_exact[touched]) ** 2)) / max(dp_amp, 1.0))
    u_l2 = float(np.sqrt(np.mean((W[3][touched] - u_exact[touched]) ** 2)) / max(du, 1.0e-30))
    p_shape = _shape_metrics(W[4] - p0, p_exact - p0, wave_region, amp_floor_ratio * dp_amp)
    u_shape = _shape_metrics(W[3] - u0, u_exact - u0, wave_region, amp_floor_ratio * du)
    rho_shape = _shape_metrics(rho, rho_exact, wave_region, amp_floor_ratio * rho_floor)
    peak_amp_ok = True
    if material == "water":
        peak_amp_ok = bool(
            _amp_ratio_ok(rho_shape["amp_ratio"])
            and _amp_ratio_ok(u_shape["amp_ratio"])
            and _amp_ratio_ok(p_shape["amp_ratio"])
        )
    profile_ok = (
        p_shape["amp_ratio"] >= amp_floor_ratio
        and u_shape["amp_ratio"] >= amp_floor_ratio
        and p_shape["corr"] > 0.60
        and u_shape["corr"] > 0.60
        and p_shape["scaled_l2"] < 1.00
        and u_shape["scaled_l2"] < 1.00
        and peak_amp_ok
    )
    ok = bool(
        _finite(W) and complete and profile_ok and lambda_ok and osc < osc_limit
        and hf["hf_oscillation_ok"]
    )
    _save_plot(case_name, x, W, rho, exact,
               f"{case_name} {material} sinusoidal pass={ok} dp={dp_meas:.3g} "
               f"lambda={lam:.3g}/{lam0:.3g} corr={p_shape['corr']:.2f}/{u_shape['corr']:.2f} "
               f"peak={rho_shape['amp_ratio']:.2f}/{u_shape['amp_ratio']:.2f}/{p_shape['amp_ratio']:.2f} "
               f"osc={osc:.2e}")
    return {
        "case": case_name,
        "pass": ok,
        "wall": wall,
        "steps": int(out["step"]),
        "complete": bool(complete),
        "dp_meas": dp_meas,
        "dp_exact": dp_amp,
        "lambda_meas": lam,
        "lambda_exact": lam0,
        "osc": osc,
        "L2p_amp": p_l2,
        "L2u_amp": u_l2,
        "p_corr": p_shape["corr"],
        "u_corr": u_shape["corr"],
        "rho_corr": rho_shape["corr"],
        "p_scaled_l2": p_shape["scaled_l2"],
        "u_scaled_l2": u_shape["scaled_l2"],
        "rho_scaled_l2": rho_shape["scaled_l2"],
        "p_amp_ratio": p_shape["amp_ratio"],
        "u_amp_ratio": u_shape["amp_ratio"],
        "rho_amp_ratio": rho_shape["amp_ratio"],
        "peak_amp_ratio_min": CASE05_PEAK_RATIO_MIN if material == "water" else amp_floor_ratio,
        "peak_amp_ratio_max": CASE05_PEAK_RATIO_MAX if material == "water" else float("inf"),
        "peak_amp_ok": bool(peak_amp_ok),
        "profile_ok": bool(profile_ok),
        **hf,
    }


def case_04() -> dict:
    return _single_phase_acoustic("04_B", material="air")


def case_05() -> dict:
    return _single_phase_acoustic("05_B", material="water")


def case_06() -> dict:
    rho_L, gamma_L = 1.2650, 1.40
    rho_R, gamma_R = 1.7537, 1.01
    p0, T0, u0 = P0, 293.0, 0.30886
    kv_L = p0 / ((gamma_L - 1.0) * rho_L * T0)
    kv_R = p0 / ((gamma_R - 1.0) * rho_R * T0)
    eos1 = make_eos("ideal", gamma=gamma_L, kv=kv_L)
    eos2 = make_eos("ideal", gamma=gamma_R, kv=kv_R)
    n = 200
    dx = 1.0 / n
    x = (np.arange(n) + 0.5) * dx
    a = np.where(x < 0.5, 1.0 - 1.0e-6, 1.0e-6)
    T1 = np.full(n, T0)
    T2 = np.full(n, T0)
    W0 = (a, T1, T2, np.full(n, u0), np.full(n, p0))
    t_end = 3.3e-3
    t0 = time.time()
    out = _solve_imex(
        eos1, eos2, W0, dx, t_end,
        bc_l="transmissive", bc_r="transmissive", cfl=0.4,
        alpha_pure_tol=1.0e-6)
    wall = time.time() - t0
    W = out["W"]
    rho = _rho_mix(W, eos1, eos2)
    exact = {"rho": _rho_mix(W0, eos1, eos2), "u": np.full(n, u0), "p": np.full(n, p0)}
    p_rel = float(np.max(np.abs(W[4] - p0)) / p0)
    u_abs = float(np.max(np.abs(W[3] - u0)))
    osc = _checkerboard(W[4], p0)
    complete = out.get("terminated_reason") is None and out["t_final"] >= t_end - 1.0e-14
    ok = bool(_finite(W) and complete and p_rel < 1.0e-9 and u_abs < 1.0e-6 and osc < 1.0e-4)
    _save_plot("06_B", x, W, rho, exact,
               f"06_B impedance PE pass={ok} p_rel={p_rel:.2e} u={u_abs:.2e} osc={osc:.2e}")
    return {
        "case": "06_B",
        "pass": ok,
        "wall": wall,
        "steps": int(out["step"]),
        "complete": bool(complete),
        "p_rel": p_rel,
        "u_abs": u_abs,
        "osc": osc,
        "corr_alpha": _pearson(W[0], W0[0]),
    }


CASES = {
    "01": case_01,
    "03": case_03,
    "04": case_04,
    "05": case_05,
    "06": case_06,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    args = parser.parse_args()
    result = CASES[args.case]()
    print("CASE_JSON " + json.dumps(result, sort_keys=True))
    print(0 if result["pass"] else 1)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
