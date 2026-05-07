#!/usr/bin/env python3
"""Source-term validation driver for 32_S1--35_S2B.

The driver intentionally reports unsupported source physics as mechanical
failures.  This gives the autoresearch loop a stable baseline before adding
gravity, Lee phase-change, and heat-conduction terms to ``solver/five_eq_IMEX``.
The final non-empty line is a JSON metrics object for codex-autoresearch.
"""
from __future__ import annotations

import inspect
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
T0 = 300.0
ALPHA_EPS = 1.0e-6


def _make_air_ideal():
    return make_eos("ideal", gamma=1.4, kv=717.5)


def _make_water_nasg():
    return make_eos(
        "nasg",
        gamma=1.187,
        pinf=7.028e8,
        kv=3610.0,
        b=6.61e-4,
        eta=-1.177788e6,
    )


def _make_water_vapor_ideal():
    return make_eos("ideal", gamma=1.33, kv=1410.0)


def _rho_mix(W, eos1, eos2):
    a, T1, T2, _, p = W
    rho1 = eos1.density(p, T1)
    rho2 = eos2.density(p, T2)
    return a * rho1 + (1.0 - a) * rho2


def _T_mix(W):
    a, T1, T2, _, _ = W
    return a * T1 + (1.0 - a) * T2


def _fields(W, eos1, eos2):
    return {
        "rho": _rho_mix(W, eos1, eos2),
        "u": np.asarray(W[3], dtype=float),
        "p": np.asarray(W[4], dtype=float),
        "T": _T_mix(W),
    }


def _plot_case(case_name, x, num, exact, title, *, message=None):
    out_dir = ROOT / "results" / "1D" / case_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    fields = [("rho", num["rho"], exact["rho"]),
              ("u", num["u"], exact["u"]),
              ("p", num["p"], exact["p"]),
              ("temperature", num["T"], exact["T"])]
    for ax, (name, y_num, y_ex) in zip(axes.ravel(), fields):
        ax.plot(x, y_num, "o-", ms=2.5, lw=1.0, label="num")
        ax.plot(x, y_ex, "r--", lw=1.2, label="exact/ref")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.suptitle(title if message is None else f"{title}\n{message}")
    path = out_dir / "diff_vs_exact.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Plot saved: {path}")


def _unsupported_result(case_name, reason):
    x = np.linspace(0.0, 1.0, 16)
    zeros = np.zeros_like(x)
    ones = np.ones_like(x)
    num = {
        "rho": zeros.copy(),
        "u": zeros.copy(),
        "p": zeros.copy(),
        "T": zeros.copy(),
    }
    exact = {
        "rho": ones.copy(),
        "u": zeros.copy(),
        "p": ones.copy(),
        "T": ones.copy(),
    }
    _plot_case(
        case_name,
        x,
        num,
        exact,
        f"{case_name} unsupported source baseline",
        message=reason,
    )
    return {
        "case": case_name,
        "pass": False,
        "reason": reason,
    }


def _solve_signature():
    return inspect.signature(solve).parameters


def _solve_source(eos1, eos2, W0, dx, t_end, **kwargs):
    return solve(
        eos1,
        eos2,
        W0,
        dx,
        t_end,
        time_integrator="imex_ssp3",
        alpha_scheme="adaptive_bvd",
        primitive_scheme="tmlpu",
        mixture_kind="kapila",
        kapila_closure=True,
        pure_branch=True,
        alpha_pure_tol=ALPHA_EPS,
        pressure_closure="regime_auto",
        dt_min=1.0e-14,
        **kwargs,
    )


def _hydrostatic_pressure_profile(x, alpha, eos_liq, eos_gas, g, *,
                                  T_liq=300.0, T_gas=300.0, p_top=P0):
    """Build a smooth face-balanced cell-centered hydrostatic profile."""
    p = np.empty_like(x)
    p[0] = float(p_top)
    if p.size == 1:
        return p

    def rho_at(i, p_i):
        a = np.array([alpha[i]])
        pp = np.array([p_i])
        rho_l = eos_liq.density(pp, np.array([T_liq]))[0]
        rho_g = eos_gas.density(pp, np.array([T_gas]))[0]
        return float(a[0] * rho_l + (1.0 - a[0]) * rho_g)

    dx = float(x[1] - x[0])
    for i in range(1, p.size):
        rho_l = rho_at(i - 1, p[i - 1])
        guess = p[i - 1] + dx * rho_l * g
        for _ in range(8):
            rho_r = rho_at(i, guess)
            guess = p[i - 1] + dx * 0.5 * (rho_l + rho_r) * g
        p[i] = guess
    return p


def _piecewise_slope_hf(values, x, alpha):
    """Relative adjacent-slope oscillation away from material-interface faces."""
    values = np.asarray(values, dtype=float)
    x = np.asarray(x, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if values.size < 4:
        return 0.0
    dx = float(x[1] - x[0])
    slope = np.diff(values) / dx
    same_face = np.abs(np.diff(alpha)) < 0.25
    if slope.size < 2:
        return 0.0
    same_triplet = same_face[:-1] & same_face[1:]
    ds = np.diff(slope)
    if int(np.count_nonzero(same_triplet)) < 1:
        return 0.0
    ds = ds[same_triplet]
    sref = np.maximum(np.abs(slope[:-1][same_triplet]), 1.0)
    return float(np.max(np.abs(ds) / sref))


def _range_l1(num, exact, floor=1.0):
    num = np.asarray(num, dtype=float)
    exact = np.asarray(exact, dtype=float)
    scale = max(float(np.max(exact) - np.min(exact)), float(floor))
    return float(np.mean(np.abs(num - exact)) / scale)


def _rel_linf(num, exact, floor=1.0):
    num = np.asarray(num, dtype=float)
    exact = np.asarray(exact, dtype=float)
    scale = max(float(np.max(np.abs(exact))), float(floor))
    return float(np.max(np.abs(num - exact)) / scale)


def run_32():
    params = _solve_signature()
    if "gravity" not in params:
        return _unsupported_result(
            "32_S1",
            "solve() has no gravity/body-force source API.",
        )
    eos_liq = _make_water_nasg()
    eos_gas = _make_air_ideal()
    n = 100
    L = 10.0
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    alpha_l = np.where(x >= 5.0, 1.0 - ALPHA_EPS, ALPHA_EPS)
    p = _hydrostatic_pressure_profile(x, alpha_l, eos_liq, eos_gas, 10.0)
    W0 = (
        alpha_l,
        np.full(n, 300.0),
        np.full(n, 300.0),
        np.zeros(n),
        p,
    )
    out = _solve_source(
        eos_liq,
        eos_gas,
        W0,
        dx,
        1.0,
        bc_l="reflective",
        bc_r="reflective",
        dt_fixed=0.01,
        max_steps=110,
        gravity=10.0,
    )
    W = out["W"]
    num = _fields(W, eos_liq, eos_gas)
    exact = _fields(W0, eos_liq, eos_gas)
    complete = out.get("terminated_reason") is None and out["t_final"] >= 1.0 - 1.0e-14
    finite = bool(all(np.all(np.isfinite(c)) for c in W))
    u_linf = float(np.max(np.abs(W[3])))
    p_rel = float(np.max(np.abs(W[4] - W0[4])) / max(float(np.max(np.abs(W0[4]))), 1.0))
    rho_rel = float(np.max(np.abs(num["rho"] - exact["rho"])) / max(float(np.max(exact["rho"])), 1.0))
    pressure_spike = float(np.max(np.abs(np.diff(W[4]) - np.diff(W0[4]))) / max(float(np.max(np.abs(np.diff(W0[4])))), 1.0))
    p_slope_hf = _piecewise_slope_hf(W[4], x, W[0])
    rho_slope_hf = _piecewise_slope_hf(num["rho"], x, W[0])
    ok = bool(complete and finite and u_linf < 1.0e-8 and p_rel < 1.0e-10
              and rho_rel < 1.0e-10 and pressure_spike < 1.0e-10
              and p_slope_hf < 1.0e-3 and rho_slope_hf < 1.0e-3)
    _plot_case(
        "32_S1",
        x,
        num,
        exact,
        f"32_S1 hydrostatic pass={ok} u_linf={u_linf:.2e} p_rel={p_rel:.2e}",
    )
    return {
        "case": "32_S1",
        "pass": ok,
        "complete": complete,
        "finite": finite,
        "u_linf": u_linf,
        "p_rel": p_rel,
        "rho_rel": rho_rel,
        "pressure_spike": pressure_spike,
        "p_slope_hf": p_slope_hf,
        "rho_slope_hf": rho_slope_hf,
        "steps": int(out["step"]),
        "terminated_reason": out.get("terminated_reason"),
    }


def run_33():
    params = _solve_signature()
    missing = [name for name in ("gravity", "alpha_inlet", "T1_inlet", "T2_inlet")
               if name not in params]
    if missing:
        return _unsupported_result(
            "33_S1",
            "solve() lacks gravity/inlet primitive source-test API: "
            + ", ".join(missing),
        )
    eos_liq = _make_water_nasg()
    eos_gas = _make_air_ideal()
    n = int(float(os.environ.get("FIVE_EQ_IMEX_SOURCE33_N", "400")))
    L = 12.0
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    alpha_l0 = 0.8
    u0 = 10.0
    t_end = 0.6
    W0 = (
        np.full(n, alpha_l0),
        np.full(n, 300.0),
        np.full(n, 300.0),
        np.full(n, u0),
        np.full(n, P0),
    )
    def solve_faucet(n_cells):
        dx_i = L / n_cells
        dt_i = 0.002 * 160.0 / float(n_cells)
        W_i = (
            np.full(n_cells, alpha_l0),
            np.full(n_cells, 300.0),
            np.full(n_cells, 300.0),
            np.full(n_cells, u0),
            np.full(n_cells, P0),
        )
        out_i = _solve_source(
            eos_liq,
            eos_gas,
            W_i,
            dx_i,
            t_end,
            bc_l="inlet",
            bc_r="transmissive",
            dt_fixed=dt_i,
            max_steps=int(math.ceil(t_end / dt_i)) + 20,
            alpha_inlet=alpha_l0,
            T1_inlet=300.0,
            T2_inlet=300.0,
            u_inlet=u0,
            gravity=10.0,
        )
        x_i = (np.arange(n_cells) + 0.5) * dx_i
        return x_i, out_i

    x, out = solve_faucet(n)
    W = out["W"]
    num = _fields(W, eos_liq, eos_gas)
    n_ref = int(float(os.environ.get(
        "FIVE_EQ_IMEX_SOURCE33_N_REF", str(max(2 * n, 320)))))
    x_ref_grid, out_ref = solve_faucet(n_ref)
    W_ref_hr = out_ref["W"]
    x_f = u0 * t_end + 0.5 * 10.0 * t_end * t_end
    u_ref = np.where(x <= x_f, np.sqrt(u0 * u0 + 2.0 * 10.0 * x), u0 + 10.0 * t_end)
    alpha_ref = np.where(x <= x_f, alpha_l0 * u0 / np.maximum(u_ref, 1.0e-30), alpha_l0)
    Wref = (
        np.interp(x, x_ref_grid, W_ref_hr[0]),
        np.interp(x, x_ref_grid, W_ref_hr[1]),
        np.interp(x, x_ref_grid, W_ref_hr[2]),
        np.interp(x, x_ref_grid, W_ref_hr[3]),
        np.interp(x, x_ref_grid, W_ref_hr[4]),
    )
    exact = _fields(Wref, eos_liq, eos_gas)
    complete = (out.get("terminated_reason") is None
                and out_ref.get("terminated_reason") is None
                and out["t_final"] >= t_end - 1.0e-14
                and out_ref["t_final"] >= t_end - 1.0e-14)
    finite = bool(all(np.all(np.isfinite(c)) for c in W)
                  and all(np.all(np.isfinite(c)) for c in W_ref_hr))
    alpha_ok = bool(np.min(W[0]) >= -1.0e-10 and np.max(W[0]) <= 1.0 + 1.0e-10)
    u_mean = float(np.mean(W[3]))
    u_expected = u0 + 10.0 * t_end
    p_spike = float((np.max(W[4]) - np.min(W[4])) / P0)
    alpha_sum_ok = alpha_ok
    trend_ok = bool(u_mean > u0 + 0.25 * 10.0 * t_end
                    and float(np.min(W[0])) < alpha_l0)
    u_l1 = _range_l1(W[3], Wref[3], floor=u_expected - u0)
    alpha_l1 = _range_l1(W[0], Wref[0], floor=alpha_l0)
    rho_l1 = _range_l1(num["rho"], exact["rho"], floor=float(np.max(exact["rho"])))
    p_l1 = _range_l1(W[4], Wref[4], floor=P0)
    T_l1 = _range_l1(num["T"], exact["T"], floor=1.0)
    T_linf = _rel_linf(num["T"], exact["T"], floor=300.0)
    ransom_u_l1 = _range_l1(W[3], u_ref, floor=u_expected - u0)
    ransom_alpha_l1 = _range_l1(W[0], alpha_ref, floor=alpha_l0)
    ok = bool(complete and finite and alpha_sum_ok and trend_ok and p_spike < 0.5
              and u_l1 < 1.0e-2 and alpha_l1 < 1.0e-2 and rho_l1 < 1.0e-2
              and p_l1 < 1.0e-2 and T_l1 < 1.0e-2 and T_linf < 1.0e-3)
    _plot_case(
        "33_S1",
        x,
        num,
        exact,
        f"33_S1 gravity faucet pass={ok} Nref={n_ref} u_mean={u_mean:.3g} p_spike={p_spike:.2e}",
    )
    return {
        "case": "33_S1",
        "pass": ok,
        "complete": complete,
        "finite": finite,
        "alpha_ok": alpha_ok,
        "u_mean": u_mean,
        "u_expected_freefall": u_expected,
        "p_spike_rel": p_spike,
        "u_profile_l1": u_l1,
        "alpha_profile_l1": alpha_l1,
        "rho_profile_l1": rho_l1,
        "p_profile_l1": p_l1,
        "T_profile_l1": T_l1,
        "T_profile_rel_linf": T_linf,
        "ransom_u_l1_qualitative": ransom_u_l1,
        "ransom_alpha_l1_qualitative": ransom_alpha_l1,
        "reference_n": n_ref,
        "reference_steps": int(out_ref["step"]),
        "reference_terminated_reason": out_ref.get("terminated_reason"),
        "steps": int(out["step"]),
        "terminated_reason": out.get("terminated_reason"),
    }


def run_34():
    params = _solve_signature()
    if "phase_change" not in params and "source_model" not in params:
        return _unsupported_result(
            "34_S2",
            "solve() has no Lee phase-change Gamma source API.",
        )
    eos_liq = _make_water_nasg()
    eos_vap = _make_water_vapor_ideal()
    subcases = [
        ("evap", 81060.0, 0.999, +1.0),
        ("cond", 121590.0, 0.5, -1.0),
    ]
    xs = []
    alpha_v_num = []
    alpha_v_ref = []
    p_num = []
    p_ref = []
    T_num = []
    T_ref = []
    results = []
    p_sat_errors = []
    T_errors = []
    for j, (name, p0, alpha_l0, expected_sign) in enumerate(subcases):
        n = 10
        dx = 1.0 / n
        x = (np.arange(n) + 0.5) * dx
        W0 = (
            np.full(n, alpha_l0),
            np.full(n, 373.15),
            np.full(n, 373.15),
            np.zeros(n),
            np.full(n, p0),
        )
        out = _solve_source(
            eos_liq,
            eos_vap,
            W0,
            dx,
            8.0e-4,
            bc_l="periodic",
            bc_r="periodic",
            dt_fixed=1.0e-5,
            max_steps=110,
            phase_change={
                "tau": 1.0e-4,
                "T_sat": 373.15,
                "p_sat": 101325.0,
                "thermal_policy": "isothermal",
                "equilibrium_target": "pressure",
            },
        )
        W = out["W"]
        finite = bool(all(np.all(np.isfinite(c)) for c in W))
        alpha_v0 = 1.0 - alpha_l0
        alpha_v1 = float(np.mean(1.0 - W[0]))
        direction = np.sign(alpha_v1 - alpha_v0)
        u_linf = float(np.max(np.abs(W[3])))
        p_sat_err = float(np.max(np.abs(W[4] - 101325.0)) / 101325.0)
        T_err = float(np.max(np.abs(_T_mix(W) - 373.15)) / 373.15)
        p_sat_errors.append(p_sat_err)
        T_errors.append(T_err)
        bounded = bool(np.min(W[0]) >= -1.0e-10 and np.max(W[0]) <= 1.0 + 1.0e-10
                       and np.min(W[4]) > 0.0 and np.min(W[1]) > 0.0 and np.min(W[2]) > 0.0)
        complete = out.get("terminated_reason") is None
        p_sat_abs_err = float(np.max(np.abs(W[4] - 101325.0)))
        sub_ok = bool(complete and finite and bounded and direction == expected_sign
                      and u_linf < 1.0e-8 and p_sat_err <= 1.0e-6
                      and p_sat_abs_err <= 0.1 and T_err < 1.0e-8)
        results.append(sub_ok)
        xs.append(float(j))
        alpha_v_num.append(alpha_v1)
        alpha_v_ref.append(alpha_v0 + expected_sign * abs(alpha_v1 - alpha_v0))
        p_num.append(float(np.mean(W[4])))
        p_ref.append(101325.0)
        T_num.append(float(np.mean(_T_mix(W))))
        T_ref.append(373.15)
    xplot = np.asarray(xs)
    num = {"rho": np.asarray(alpha_v_num), "u": np.zeros_like(xplot),
           "p": np.asarray(p_num), "T": np.asarray(T_num)}
    exact = {"rho": np.asarray(alpha_v_ref), "u": np.zeros_like(xplot),
             "p": np.asarray(p_ref), "T": np.asarray(T_ref)}
    ok = bool(all(results))
    _plot_case("34_S2", xplot, num, exact,
               f"34_S2 Lee relaxation pass={ok} alpha_v shown in rho panel")
    return {
        "case": "34_S2",
        "pass": ok,
        "subcases_passed": int(sum(results)),
        "subcases_total": int(len(results)),
        "alpha_v_evap_final": float(alpha_v_num[0]),
        "alpha_v_cond_final": float(alpha_v_num[1]),
        "p_sat_rel_linf": float(max(p_sat_errors)),
        "p_sat_abs_linf": float(max(abs(p - 101325.0) for p in p_num)),
        "T_sat_rel_linf": float(max(T_errors)),
    }


def run_35():
    params = _solve_signature()
    missing = []
    if "phase_change" not in params and "source_model" not in params:
        missing.append("Lee phase-change")
    if "heat_conduction" not in params and "thermal_conductivity" not in params:
        missing.append("heat conduction")
    if missing:
        return _unsupported_result(
            "35_S2B",
            "solve() lacks " + " and ".join(missing) + " source API.",
        )
    eos_liq = _make_water_nasg()
    eos_vap = _make_water_vapor_ideal()
    n = int(float(__import__("os").environ.get("FIVE_EQ_IMEX_SOURCE35_N", "120")))
    L = 0.05
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    T_sat = 373.15
    T_w = 383.15
    s0 = 1.0e-3
    delta = 2.0 * dx
    alpha_v = ALPHA_EPS + (1.0 - 2.0 * ALPHA_EPS) * 0.5 * (1.0 - np.tanh((x - s0) / delta))
    alpha_l = 1.0 - alpha_v
    T = np.where(x < s0, T_w - (T_w - T_sat) * (x / s0), T_sat)
    W0 = (alpha_l, T.copy(), T.copy(), np.zeros(n), np.full(n, 101325.0))
    out = _solve_source(
        eos_liq,
        eos_vap,
        W0,
        dx,
        1.0e-3,
        bc_l="reflective",
        bc_r="transmissive",
        dt_fixed=2.0e-6,
        max_steps=600,
        phase_change={
            "tau": 1.0e-3,
            "T_sat": T_sat,
            "p_sat": 101325.0,
            "thermal_policy": "isothermal",
        },
        heat_conduction={
            "T_left": T_w,
            "T_right": T_sat,
            "k_liquid": 0.6,
            "k_vapor": 0.025,
            "thermal_policy": "primitive_temperature",
        },
    )
    W = out["W"]
    num = _fields(W, eos_liq, eos_vap)
    exact = _fields(W0, eos_liq, eos_vap)
    alpha_v_num = 1.0 - W[0]

    def interface_pos(av):
        idx = np.where(av >= 0.5)[0]
        if idx.size == 0:
            return 0.0
        return float(x[int(idx[-1])])

    s_threshold_initial = interface_pos(alpha_v)
    s_threshold_final = interface_pos(alpha_v_num)
    s_eff_initial = float(np.sum(alpha_v) * dx)
    s_eff_final = float(np.sum(alpha_v_num) * dx)
    finite = bool(all(np.all(np.isfinite(c)) for c in W))
    bounded = bool(np.min(W[0]) >= -1.0e-10 and np.max(W[0]) <= 1.0 + 1.0e-10
                   and np.min(W[4]) > 0.0 and np.min(W[1]) > 0.0 and np.min(W[2]) > 0.0)
    p_spike = float((np.max(W[4]) - np.min(W[4])) / 101325.0)
    complete = out.get("terminated_reason") is None
    grows = bool(s_eff_final > s_eff_initial)
    ok = bool(complete and finite and bounded and grows and p_spike < 0.5)
    _plot_case(
        "35_S2B",
        x,
        num,
        exact,
        f"35_S2B Stefan smoke pass={ok} s_eff0={s_eff_initial:.3e} s_eff={s_eff_final:.3e} p_spike={p_spike:.2e}",
    )
    return {
        "case": "35_S2B",
        "pass": ok,
        "complete": complete,
        "finite": finite,
        "bounded": bounded,
        "interface_initial": s_eff_initial,
        "interface_final": s_eff_final,
        "interface_threshold_initial": s_threshold_initial,
        "interface_threshold_final": s_threshold_final,
        "p_spike_rel": p_spike,
        "steps": int(out["step"]),
        "terminated_reason": out.get("terminated_reason"),
    }


def main():
    start = time.time()
    cases = [run_32(), run_33(), run_34(), run_35()]
    pass_count = int(sum(1 for c in cases if c.get("pass")))
    fail_count = int(len(cases) - pass_count)
    metrics = {
        "suite": "32_35_source_terms",
        "cases_total": len(cases),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "wall": time.time() - start,
        "cases": {c["case"]: c for c in cases},
    }
    for c in cases:
        status = "PASS" if c.get("pass") else "FAIL"
        print(f"{c['case']}: {status} {c.get('reason', '')}")
    print(json.dumps(metrics, sort_keys=True))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
