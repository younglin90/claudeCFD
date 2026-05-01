#!/usr/bin/env python3
"""Strict-but-diffusion-aware 02/07 acceptance verifier.

Final stdout line is an integer failure count for codex-autoresearch.
The criteria intentionally separate material-contact preservation from
acoustic-profile similarity:

* 02-A: pressure is checked by relative Linf, velocity by absolute Linf.
  Alpha/rho diffusion is allowed, but profile collapse is not.
* 07-B: moderate acoustic diffusion is allowed, but pressure/velocity profiles
  must remain correlated with the linear acoustic reference and must not show a
  strong alternating checkerboard residual around the material interface.
"""
from __future__ import annotations

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
from oscillation_guards import high_frequency_oscillation_guard  # noqa: E402

P0 = 1.0e5
U_PEAK_07 = 0.02
P_TOL_02 = 1.0e-6
U_TOL_02 = 1.0e-6
MIN_RANGE_RATIO_02 = 0.10
MIN_CORR_02 = 0.50
PEAK_CELL_TOL_07 = 3


@dataclass(frozen=True)
class Case07:
    name: str
    left: str
    right: str
    x_intf: float
    x_src: float
    sigma: float
    t_end: float


EOS_07 = {
    "Air": {"kind": "ideal", "gamma": 1.400, "pinf": 0.0, "kv": 717.5, "rho": 1.157, "c": 347.8},
    "Helium": {"kind": "ideal", "gamma": 1.667, "pinf": 0.0, "kv": 3120.0, "rho": 0.164, "c": 1008.2},
    "Argon": {"kind": "ideal", "gamma": 1.660, "pinf": 0.0, "kv": 312.0, "rho": 1.748, "c": 308.2},
    "Water": {
        "kind": "nasg",
        "gamma": 1.187,
        "pinf": 7.028e8,
        "kv": 3610.0,
        "b": 6.61e-4,
        "eta": -1.177788e6,
        "rho": 998.0,
        "c": 1567.3350584385664,
    },
}

CASES_07 = (
    # NASG water has a much larger acoustic speed than air, so the transmitted
    # Gaussian broadens in x.  Stop earlier to keep about six transmitted-wave
    # sigmas before the right boundary, similar to the visual separation in the
    # Helium-Air subcase.
    Case07("Air-Water", "Air", "Water", 0.5, 0.1, 0.014, 1.55e-3),
    Case07("Helium-Air", "Helium", "Air", 1.0, 0.2, 0.049, 1.513e-3),
    Case07("Argon-Air", "Argon", "Air", 0.5, 0.1, 0.038, 2.02e-3),
)


def _make_eos_07(name: str):
    spec = EOS_07[name]
    if spec["kind"] == "ideal":
        return make_eos("ideal", gamma=spec["gamma"], kv=spec["kv"])
    if spec["kind"] == "sg":
        return make_eos("sg", gamma=spec["gamma"], pinf=spec["pinf"], kv=spec["kv"])
    if spec["kind"] == "nasg":
        return make_eos(
            "nasg",
            gamma=spec["gamma"],
            pinf=spec["pinf"],
            kv=spec["kv"],
            b=spec["b"],
            eta=spec["eta"],
        )
    raise ValueError(f"unknown 07 EOS kind for {name}: {spec['kind']}")


def _temperature_07(name: str) -> float:
    spec = EOS_07[name]
    eos = _make_eos_07(name)
    rho = np.asarray([spec["rho"]], dtype=float)
    p = np.asarray([P0], dtype=float)
    return float(eos.temperature(rho, eos.energy(rho, p))[0])


def _rho_mix(W, eos1, eos2) -> np.ndarray:
    a, T1, T2, _, p = W
    rho1 = eos1.density(p, T1)
    rho2 = eos2.density(p, T2)
    return a * rho1 + (1.0 - a) * rho2


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float) - float(np.mean(a))
    bb = np.asarray(b, dtype=float) - float(np.mean(b))
    den = float(np.sqrt(np.dot(aa, aa) * np.dot(bb, bb)))
    if den <= 1.0e-300:
        return 1.0
    return float(np.dot(aa, bb) / den)


def _theta_from_eos(eos, p: float, T: float) -> float:
    """Small-signal dT/dp along an isentrope from the EOS p,T derivatives."""
    pp = np.asarray([p], dtype=float)
    TT = np.asarray([T], dtype=float)
    rho = eos.density(pp, TT)
    rho_p = eos.drhodp_T(rho, TT)
    rho_T = eos.drhodT_p(rho, TT)
    e_p = eos.dedp_T(rho, TT)
    e_T = eos.dedT_p(rho, TT)
    pr2 = pp / np.maximum(rho * rho, 1.0e-300)
    den = e_T - pr2 * rho_T
    theta = (pr2 * rho_p - e_p) / np.where(np.abs(den) > 1.0e-300, den, 1.0e-300)
    return float(theta[0])


def _exact_07(case: Case07, x: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear acoustic d'Alembert solution for the Gaussian interface tests."""
    left = EOS_07[case.left]
    right = EOS_07[case.right]
    cL = float(left["c"])
    cR = float(right["c"])
    ZL = float(left["rho"] * left["c"])
    ZR = float(right["rho"] * right["c"])
    R = (ZR - ZL) / (ZR + ZL)
    Tu = 2.0 * ZL / (ZL + ZR)
    Tp = 2.0 * ZR / (ZL + ZR)
    t_hit = (case.x_intf - case.x_src) / cL

    x = np.asarray(x, dtype=float)
    left_mask = x < case.x_intf

    def gauss(center: float, sigma: float) -> np.ndarray:
        return np.exp(-((x - center) ** 2) / (2.0 * sigma * sigma))

    u_inc = U_PEAK_07 * gauss(case.x_src + cL * t, case.sigma) * left_mask
    u_ref_shape = U_PEAK_07 * gauss(2.0 * case.x_intf - case.x_src - cL * t, case.sigma) * left_mask
    if t > t_hit:
        sigR = case.sigma * cR / cL
        u_tr_shape = U_PEAK_07 * gauss(case.x_intf + cR * (t - t_hit), sigR) * (~left_mask)
    else:
        u_tr_shape = np.zeros_like(x)

    u_exact = u_inc - R * u_ref_shape + Tu * u_tr_shape
    p_exact = P0 + ZL * u_inc + R * ZL * u_ref_shape + Tp * ZL * u_tr_shape
    return p_exact, u_exact


def _metrics_profile(x, p_num, u_num, p_exact, u_exact, dp_wave, du_wave) -> dict[str, float]:
    p_err = np.asarray(p_num, dtype=float) - np.asarray(p_exact, dtype=float)
    u_err = np.asarray(u_num, dtype=float) - np.asarray(u_exact, dtype=float)
    p_sig = np.asarray(p_exact, dtype=float) - P0
    u_sig = np.asarray(u_exact, dtype=float)
    dp = max(float(dp_wave), 1.0e-300)
    du = max(float(du_wave), 1.0e-300)
    den_l1p = max(float(np.sum(np.abs(p_sig))), 1.0e-300)
    den_l1u = max(float(np.sum(np.abs(u_sig))), 1.0e-300)
    return {
        "L2p": float(np.sqrt(np.mean(p_err * p_err)) / dp),
        "Lip": float(np.max(np.abs(p_err)) / dp),
        "L2u": float(np.sqrt(np.mean(u_err * u_err)) / du),
        "Liu": float(np.max(np.abs(u_err)) / du),
        "frac_p": float(np.mean(np.abs(p_err) < 0.30 * dp)),
        "frac_u": float(np.mean(np.abs(u_err) < 0.30 * du)),
        "L1p": float(np.sum(np.abs(p_err)) / den_l1p),
        "L1u": float(np.sum(np.abs(u_err)) / den_l1u),
        "corr_p": _pearson(np.asarray(p_num, dtype=float) - P0, p_sig),
        "corr_u": _pearson(np.asarray(u_num, dtype=float), u_sig),
    }


def _ensure_dir(case_name: str) -> str:
    out = os.path.join(ROOT, "results", "1D", case_name)
    os.makedirs(out, exist_ok=True)
    return out


def _range_ratio(num: np.ndarray, exact: np.ndarray) -> float:
    den = float(np.max(exact) - np.min(exact))
    if den <= 1.0e-300:
        return 1.0
    return float((np.max(num) - np.min(num)) / den)


def _finite_W(W) -> bool:
    return bool(all(np.all(np.isfinite(v)) for v in W))


def _save_02_plot(x, W, W0, rho, rho_exact, metrics: dict) -> None:
    out_dir = _ensure_dir("02_A")
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    fields = [
        ("rho", rho, rho_exact),
        ("u", W[3], W0[3]),
        ("p", W[4], W0[4]),
    ]
    for j, (name, num, exact) in enumerate(fields):
        ax[0, j].plot(x, num, "b-o", ms=4, label="num")
        ax[0, j].plot(x, exact, "r--", label="exact")
        ax[0, j].set_title(name)
        ax[0, j].grid(alpha=0.3)
        ax[0, j].legend(fontsize=8)
        ax[1, j].plot(x, np.abs(num - exact), "k-o", ms=4)
        ax[1, j].set_title(f"abs error {name}")
        ax[1, j].grid(alpha=0.3)
    fig.suptitle(
        "02_A NASG strict acceptance "
        f"pass={metrics['pass']} p_rel={metrics['p_rel_linf']:.3e} "
        f"u_linf={metrics['u_abs_linf']:.3e}"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "diff_vs_exact.png"), dpi=120)
    plt.close(fig)
    print("Plot saved: results/1D/02_A/diff_vs_exact.png")


def verify_02_A() -> dict:
    eos1 = make_eos("ideal", gamma=1.4, kv=717.5)
    eos2 = make_eos(
        "nasg",
        gamma=1.187,
        pinf=7.028e8,
        kv=3610.0,
        b=6.61e-4,
        eta=-1.177788e6,
    )
    n = 10
    length = 1.0
    dx = length / n
    x = (np.arange(n) + 0.5) * dx
    p0 = P0
    u0 = 1.0
    alpha_floor = 1.0e-3
    a1 = np.where((x >= 0.4) & (x < 0.6), alpha_floor, 1.0 - alpha_floor)
    W0 = (
        a1,
        np.full(n, 300.0),
        np.full(n, 300.0),
        np.full(n, u0),
        np.full(n, p0),
    )

    t0 = time.time()
    pressure_closure = os.environ.get("FIVE_EQ_IMEX_PRESSURE_CLOSURE", "regime_auto")
    alpha_scheme = os.environ.get("FIVE_EQ_IMEX_ALPHA_SCHEME", "mstacs")
    primitive_scheme = os.environ.get("FIVE_EQ_IMEX_PRIMITIVE_SCHEME", "weno3")
    out = solve(
        eos1,
        eos2,
        W0,
        dx,
        1.0,
        bc_l="periodic",
        bc_r="periodic",
        cfl=0.5,
        max_steps=50000,
        dt_fixed=0.01,
        time_integrator="imex_ad",
        alpha_scheme=alpha_scheme,
        kapila_closure=True,
        pure_branch=True,
        alpha_pure_tol=alpha_floor,
        primitive_scheme=primitive_scheme,
        pressure_closure=pressure_closure,
    )
    W = out["W"]
    rho = _rho_mix(W, eos1, eos2)
    rho_exact = _rho_mix(W0, eos1, eos2)
    finite = _finite_W(W)
    complete = out.get("terminated_reason") is None and out["t_final"] >= 1.0 - 1.0e-12

    p_rel_linf = float(np.max(np.abs(W[4] - W0[4])) / max(np.max(np.abs(W0[4])), 1.0))
    u_abs_linf = float(np.max(np.abs(W[3] - W0[3])))
    alpha_range_ratio = _range_ratio(W[0], W0[0])
    rho_range_ratio = _range_ratio(rho, rho_exact)
    corr_alpha = _pearson(W[0], W0[0])
    corr_rho = _pearson(rho, rho_exact)
    hf = high_frequency_oscillation_guard(
        x,
        {
            "rho": (rho, rho_exact, 1.0),
            "u": (W[3], W0[3], 1.0),
            "p": (W[4], W0[4], P0),
        },
    )
    admissible = (
        np.min(W[0]) >= -1.0e-10
        and np.max(W[0]) <= 1.0 + 1.0e-10
        and np.min(rho) > 0.0
        and np.min(W[4]) > 0.0
    )
    profile_ok = (
        alpha_range_ratio >= MIN_RANGE_RATIO_02
        and rho_range_ratio >= MIN_RANGE_RATIO_02
        and corr_alpha >= MIN_CORR_02
        and corr_rho >= MIN_CORR_02
    )
    ok = (
        finite
        and complete
        and admissible
        and p_rel_linf <= P_TOL_02
        and u_abs_linf <= U_TOL_02
        and profile_ok
        and hf["hf_oscillation_ok"]
    )
    metrics = {
        "case": "02_A_NASG",
        "pass": bool(ok),
        "finite": finite,
        "complete": complete,
        "admissible": bool(admissible),
        "p_rel_linf": p_rel_linf,
        "u_abs_linf": u_abs_linf,
        "alpha_range_ratio": alpha_range_ratio,
        "rho_range_ratio": rho_range_ratio,
        "corr_alpha": corr_alpha,
        "corr_rho": corr_rho,
        "steps": int(out["step"]),
        "wall": time.time() - t0,
        **hf,
    }
    _save_02_plot(x, W, W0, rho, rho_exact, metrics)
    print(
        "ACCEPT 02_A_NASG "
        f"pass={ok} p_rel_linf={p_rel_linf:.3e} "
        f"u_abs_linf={u_abs_linf:.3e} alpha_range={alpha_range_ratio:.3f} "
        f"rho_range={rho_range_ratio:.3f} corr_alpha={corr_alpha:.3f} "
        f"corr_rho={corr_rho:.3f} finite={finite} complete={complete}"
    )
    return metrics


def _checkerboard_metric(x: np.ndarray, residual: np.ndarray, center: float, dx: float) -> dict:
    width = max(10.0 * dx, 0.06)
    idx = np.flatnonzero(np.abs(x - center) <= width)
    if idx.size < 6:
        return {"alt_ratio": 0.0, "amp": 0.0, "n": int(idx.size)}
    r = residual[idx].astype(float)
    r = r - float(np.mean(r))
    amp = float(np.max(np.abs(r)))
    if amp <= 1.0e-14:
        return {"alt_ratio": 0.0, "amp": amp, "n": int(idx.size)}
    signs = np.where(np.arange(idx.size) % 2 == 0, 1.0, -1.0)
    alt_ratio = float(abs(np.dot(r, signs)) / (np.sum(np.abs(r)) + 1.0e-300))
    return {"alt_ratio": alt_ratio, "amp": amp, "n": int(idx.size)}


def _oscillation_ok(x, W, p_exact, u_exact, case, dp_wave, dx) -> tuple[bool, dict]:
    rp = (np.asarray(W[4]) - np.asarray(p_exact)) / max(dp_wave, 1.0e-30)
    ru = (np.asarray(W[3]) - np.asarray(u_exact)) / max(U_PEAK_07, 1.0e-30)
    pchk = _checkerboard_metric(x, rp, case.x_intf, dx)
    uchk = _checkerboard_metric(x, ru, case.x_intf, dx)
    # The first version only caught near-perfect odd-even checkerboards.  The
    # 07-B Air-Water failure mode is broader: pressure ringing near the
    # material interface can be lower-frequency but still visually and
    # numerically unacceptable.  Reject both strong alternating modes and large
    # normalized interface residuals.
    bad_p = (
        (pchk["alt_ratio"] > 0.60 and pchk["amp"] > 0.20)
        or pchk["amp"] > 0.30
    )
    bad_u = (
        (uchk["alt_ratio"] > 0.60 and uchk["amp"] > 0.20)
        or uchk["amp"] > 0.45
    )
    return (not (bad_p or bad_u)), {
        "p_alt_ratio": pchk["alt_ratio"],
        "p_alt_amp": pchk["amp"],
        "u_alt_ratio": uchk["alt_ratio"],
        "u_alt_amp": uchk["amp"],
    }


def _profile_pass_07(m: dict) -> bool:
    """Diffusion-aware 07 gate aligned with the validation spec.

    The strict metric is intentionally not used here: 07 allows moderate
    amplitude diffusion, but rejects wrong phase/shape and visible oscillation.
    """
    return (
        m["L2p"] < 0.50 and m["L2u"] < 0.20
        and m["Lip"] < 2.0 and m["Liu"] < 1.0
        and m["frac_p"] >= 0.70 and m["frac_u"] >= 0.70
        and m["corr_p"] > 0.50 and m["corr_u"] > 0.50
        and m["L1p"] < 2.0 and m["L1u"] < 8.0
    )


def _peak_location_ok(x, W, p_exact, u_exact,
                      cell_tol: int = PEAK_CELL_TOL_07,
                      require_abs_peak: bool = True) -> tuple[bool, dict]:
    """Require 07-B acoustic peak locations to match within a few cells."""
    x = np.asarray(x)
    fields = {
        "p": (np.asarray(W[4], dtype=float) - P0,
              np.asarray(p_exact, dtype=float) - P0),
        "u": (np.asarray(W[3], dtype=float),
              np.asarray(u_exact, dtype=float)),
    }
    details: dict[str, float | int | bool] = {}
    ok = True
    for name, (num, exact) in fields.items():
        exact_abs_amp = float(np.max(np.abs(exact)))
        num_abs_amp = float(np.max(np.abs(num)))
        if exact_abs_amp <= 1.0e-30:
            details[f"{name}_abs_idx"] = -1
            details[f"{name}_exact_abs_idx"] = -1
            details[f"{name}_abs_required"] = bool(require_abs_peak)
            details[f"{name}_abs_ok"] = bool(num_abs_amp <= 1.0e-12)
            ok = ok and ((not require_abs_peak) or bool(details[f"{name}_abs_ok"]))
            continue

        idx_abs = int(np.argmax(np.abs(num)))
        idx_abs_exact = int(np.argmax(np.abs(exact)))
        abs_delta = abs(idx_abs - idx_abs_exact)
        abs_ok = abs_delta <= cell_tol
        ok = ok and ((not require_abs_peak) or abs_ok)
        details[f"{name}_abs_idx"] = idx_abs
        details[f"{name}_exact_abs_idx"] = idx_abs_exact
        details[f"{name}_abs_delta_cells"] = abs_delta
        details[f"{name}_abs_tol_cells"] = int(cell_tol)
        details[f"{name}_abs_required"] = bool(require_abs_peak)
        details[f"{name}_abs_x"] = float(x[idx_abs])
        details[f"{name}_exact_abs_x"] = float(x[idx_abs_exact])
        details[f"{name}_abs_ok"] = bool(abs_ok)
        details[f"{name}_abs_amp"] = num_abs_amp
        details[f"{name}_exact_abs_amp"] = exact_abs_amp

        for extremum, fn in (("max", np.argmax), ("min", np.argmin)):
            idx_exact = int(fn(exact))
            exact_amp = abs(float(exact[idx_exact]))
            require_signed = exact_amp >= 0.10 * exact_abs_amp
            idx = int(fn(num))
            signed_delta = abs(idx - idx_exact)
            signed_ok = (not require_signed) or (signed_delta <= cell_tol)
            ok = ok and signed_ok
            details[f"{name}_{extremum}_idx"] = idx
            details[f"{name}_exact_{extremum}_idx"] = idx_exact
            details[f"{name}_{extremum}_delta_cells"] = signed_delta
            details[f"{name}_{extremum}_tol_cells"] = int(cell_tol)
            details[f"{name}_{extremum}_x"] = float(x[idx])
            details[f"{name}_exact_{extremum}_x"] = float(x[idx_exact])
            details[f"{name}_{extremum}_required"] = bool(require_signed)
            details[f"{name}_{extremum}_ok"] = bool(signed_ok)

    return bool(ok), details


def _exact_07_rho(case, x: np.ndarray, p_exact: np.ndarray) -> np.ndarray:
    left = EOS_07[case.left]
    right = EOS_07[case.right]
    mask_L = x < case.x_intf
    rho0 = np.where(mask_L, left["rho"], right["rho"])
    c = np.where(mask_L, left["c"], right["c"])
    return rho0 + (np.asarray(p_exact) - P0) / np.maximum(c * c, 1.0e-300)


def _save_07_plot(rows, alpha_floor: float) -> None:
    out_dir = _ensure_dir("07_B")
    fig, ax = plt.subplots(len(rows), 3, figsize=(14, 4.2 * len(rows)))
    if len(rows) == 1:
        ax = ax.reshape(1, -1)
    for i, row in enumerate(rows):
        case = row["case_obj"]
        x = row["x"]
        W = row.get("W")
        if W is None:
            for j in range(3):
                ax[i, j].set_title(f"{case.name} ERROR")
                ax[i, j].grid(alpha=0.3)
            continue
        fields = [
            ("rho", row["rho"], row["rho_exact"]),
            ("u", W[3], row["u_exact"]),
            ("p-P0", W[4] - P0, row["p_exact"] - P0),
        ]
        for j, (name, num, exact) in enumerate(fields):
            ax[i, j].plot(x, num, "b-", lw=1.4, label="num")
            if exact is not None:
                ax[i, j].plot(x, exact, "r--", lw=1.1, label="exact")
            ax[i, j].set_title(f"{case.name} pass={row['pass']} {name}")
            ax[i, j].grid(alpha=0.3)
            ax[i, j].legend(fontsize=8)
    fig.suptitle("07_B five_eq_IMEX strict acceptance")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "diff_vs_exact.png"), dpi=120)
    plt.close(fig)
    print("Plot saved: results/1D/07_B/diff_vs_exact.png")


def verify_07_B() -> dict:
    n = 200
    length = 1.5
    dx = length / n
    x = (np.arange(n) + 0.5) * dx
    alpha_floor = 1.0e-5
    rows = []
    failures = 0
    for case in CASES_07:
        t0 = time.time()
        row = {"case": case.name, "case_obj": case, "pass": False, "error": None}
        eos1 = _make_eos_07(case.left)
        eos2 = _make_eos_07(case.right)
        eL = EOS_07[case.left]
        ZL = eL["rho"] * eL["c"]
        dp_wave = ZL * U_PEAK_07
        mask_L = x < case.x_intf
        a1 = np.where(mask_L, 1.0 - alpha_floor, alpha_floor)
        T1 = np.full(n, _temperature_07(case.left))
        T2 = np.full(n, _temperature_07(case.right))
        u = np.where(
            mask_L,
            U_PEAK_07 * np.exp(-((x - case.x_src) ** 2) / (2.0 * case.sigma**2)),
            0.0,
        )
        p = P0 + ZL * u
        theta_L = _theta_from_eos(eos1, P0, float(T1[0]))
        T1 = T1 + theta_L * (p - P0) * mask_L
        W0 = (a1, T1, T2, u, p)
        try:
            pressure_closure = os.environ.get("FIVE_EQ_IMEX_PRESSURE_CLOSURE", "regime_auto")
            alpha_scheme = os.environ.get("FIVE_EQ_IMEX_ALPHA_SCHEME", "mstacs")
            primitive_scheme = os.environ.get("FIVE_EQ_IMEX_PRIMITIVE_SCHEME", "weno3")
            out = solve(
                eos1,
                eos2,
                W0,
                dx,
                case.t_end,
                bc_l="reflective",
                bc_r="transmissive",
                cfl=0.4,
                max_steps=5000,
                time_integrator="imex_ad",
                alpha_scheme=alpha_scheme,
                kapila_closure=True,
                dt_min=1.0e-10,
                pure_branch=True,
                alpha_pure_tol=max(alpha_floor, 1.0e-8),
                primitive_scheme=primitive_scheme,
                pressure_closure=pressure_closure,
            )
            W = out["W"]
            p_exact, u_exact = _exact_07(case, x, out["t_final"])
            rho_exact = _exact_07_rho(case, x, p_exact)
            rho_num = _rho_mix(W, eos1, eos2)
            m = _metrics_profile(x, W[4], W[3], p_exact, u_exact, dp_wave, U_PEAK_07)
            finite = _finite_W(W)
            complete = out.get("terminated_reason") is None and out["t_final"] >= case.t_end
            profile_ok = _profile_pass_07(m)
            osc_ok, osc = _oscillation_ok(x, W, p_exact, u_exact, case, dp_wave, dx)
            peak_ok, peak = _peak_location_ok(
                x,
                W,
                p_exact,
                u_exact,
                require_abs_peak=(case.name == "Air-Water"),
            )
            hf = high_frequency_oscillation_guard(
                x,
                {
                    "rho": (rho_num, rho_exact, 1.0),
                    "u": (W[3], u_exact, U_PEAK_07),
                    "p": (W[4], p_exact, dp_wave),
                },
                sharp_centers=(case.x_intf,),
            )
            ok = bool(
                finite and complete and profile_ok and osc_ok and peak_ok
                and hf["hf_oscillation_ok"]
            )
            failures += 0 if ok else 1
            row.update(
                {
                    "pass": ok,
                    "x": x.copy(),
                    "W": W,
                    "rho": rho_num,
                    "rho_exact": rho_exact,
                    "p_exact": p_exact,
                    "u_exact": u_exact,
                    "metrics": m,
                    "finite": finite,
                    "complete": bool(complete),
                    "profile_ok": bool(profile_ok),
                    "osc_ok": bool(osc_ok),
                    "osc": osc,
                    "hf": hf,
                    "peak_ok": bool(peak_ok),
                    "peak": peak,
                    "steps": int(out["step"]),
                    "wall": time.time() - t0,
                }
            )
            print(
                f"ACCEPT 07_B {case.name} pass={ok} "
                f"L2p={m['L2p']:.3e} Lip={m['Lip']:.3e} "
                f"L2u={m['L2u']:.3e} Liu={m['Liu']:.3e} "
                f"frac_p={m['frac_p']:.2f} frac_u={m['frac_u']:.2f} "
                f"corr_p={m['corr_p']:.2f} corr_u={m['corr_u']:.2f} "
                f"p_alt={osc['p_alt_ratio']:.2f}/{osc['p_alt_amp']:.2f} "
                f"u_alt={osc['u_alt_ratio']:.2f}/{osc['u_alt_amp']:.2f} "
                f"p_peak={peak['p_abs_idx']}/{peak['p_exact_abs_idx']} "
                f"u_peak={peak['u_abs_idx']}/{peak['u_exact_abs_idx']} "
                f"finite={finite} complete={complete} profile={profile_ok} "
                f"osc={osc_ok} hf={hf['hf_oscillation_ok']} peak={peak_ok}"
            )
        except Exception as exc:  # pragma: no cover - verifier diagnostics
            failures += 1
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"ACCEPT 07_B {case.name} pass=False error={row['error']}")
        rows.append(row)
    _save_07_plot(rows, alpha_floor)
    return {
        "case": "07_B_all",
        "pass": failures == 0,
        "failures": failures,
        "subcases": [
            {
                k: v
                for k, v in row.items()
                if k
                not in {
                    "case_obj",
                    "x",
                    "W",
                    "rho",
                    "rho_exact",
                    "p_exact",
                    "u_exact",
                }
            }
            for row in rows
        ],
    }


def main() -> int:
    results = {"02": verify_02_A(), "07": verify_07_B()}
    failures = 0
    failures += 0 if results["02"]["pass"] else 1
    failures += int(results["07"]["failures"])
    compact = {
        "failures": failures,
        "goal_reached": failures == 0,
        "results": results,
    }
    print("ACCEPTANCE_JSON " + json.dumps(compact, sort_keys=True, default=str))
    print(failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
