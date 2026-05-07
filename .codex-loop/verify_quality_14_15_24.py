#!/usr/bin/env python3
"""Quality gate for dense two-phase problem cases 14, 15, and 24.

This script is intentionally separate from the broad 08-26 verifier so the
autoresearch loop can track one scalar objective: the number of remaining
quality failures reported by the user.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFY_PATH = os.path.join(ROOT, ".codex-loop", "verify_08_26_acceptance.py")


def _load_verify_module():
    spec = importlib.util.spec_from_file_location("verify_08_26_acceptance", VERIFY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {VERIFY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case15_quality(v):
    eos_air = v.make_eos("ideal", gamma=1.4, kv=717.5)
    eos_water = v.make_eos("sg", gamma=4.4, pinf=6.0e8, kv=474.2)
    n = 100
    L = 1.0
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    x0 = 0.5
    t_end = 1.0e-3
    p0 = np.full(n, 1.0e5)
    u0 = np.where(x < x0, -100.0, 100.0)
    rho_air = np.full(n, 1.3)
    rho_water = np.full(n, 1000.0)
    T1 = v._temperature_for_rho_p(eos_air, rho_air, p0)
    T2 = v._temperature_for_rho_p(eos_water, rho_water, p0)
    W0 = (np.full(n, 0.01), T1, T2, u0, p0)

    start = time.time()
    out = v._solve_same_scheme(
        eos_air,
        eos_water,
        W0,
        dx,
        t_end,
        bc_l="transmissive",
        bc_r="transmissive",
        cfl=0.01,
        alpha_pure_tol=1.0e-6,
        max_steps=250000,
    )
    wall = time.time() - start
    W = out["W"]
    rho = v._rho_mix(W, eos_air, eos_water)
    exact = v._case15_reference(x)

    p = np.asarray(W[4], dtype=float)
    p_ref = np.asarray(exact["p"], dtype=float)
    p_ref_amp = max(float(np.max(p_ref) - np.min(p_ref)), 1.0)
    p_ref_l2 = float(np.sqrt(np.mean((p - p_ref) ** 2)) / p_ref_amp)
    center = (x >= 0.43) & (x <= 0.57)
    p_center_linf = float(np.max(np.abs(p[center] - p_ref[center])) / 1.0e5)
    p_center_max = float(np.max(p[center]))
    p_center_min = float(np.min(p[center]))
    p_d2 = p[1:-1] - 0.5 * (p[:-2] + p[2:])
    p_hf_osc = float(np.max(np.abs(p_d2)) / 1.0e5)

    complete = out.get("terminated_reason") is None and out["t_final"] >= t_end - 1.0e-14
    finite = v._finite_admissible(W, rho)
    p_osc = v._checkerboard(p, 1.0e5)
    rho_osc = v._checkerboard(rho, 1000.0)
    alpha_peak = float(np.max(W[0]))
    rho_min = float(np.min(rho))
    p_min = float(np.min(p))
    u_min = float(np.min(W[3]))
    u_max = float(np.max(W[3]))

    cavitation_like = alpha_peak > 0.20 and rho_min < 700.0 and p_min < 8.0e4
    ok = bool(
        finite
        and complete
        and cavitation_like
        and p_min > 0.0
        and u_min < -50.0
        and u_max > 50.0
        and p_osc < 0.03
        and p_hf_osc < 0.015
        and p_ref_l2 < 0.20
        and p_center_linf < 0.35
        and rho_osc < 0.25
    )

    row = {
        "name": "15E cavitation",
        "pass": ok,
        "W": W,
        "rho": rho,
        "x": x,
        "exact": exact,
        "alpha_num": W[0],
        "metrics": {},
    }
    out_dir = v._ensure_dir("15_E")
    np.savetxt(
        os.path.join(out_dir, "reference_digitized_15.csv"),
        np.column_stack([x, exact["alpha"], exact["rho"], exact["u"], exact["p"]]),
        delimiter=",",
        header="x,alpha1_ref,rho_ref,u_ref,p_ref",
        comments="",
    )
    v._save_multi_plot("15_E", [row], f"15_E cavitation pass={ok}")
    return {
        "case": "15_E",
        "pass": ok,
        "finite": bool(finite),
        "complete": bool(complete),
        "terminated_reason": out.get("terminated_reason"),
        "wall": wall,
        "steps": int(out["step"]),
        "alpha_peak": alpha_peak,
        "rho_min": rho_min,
        "p_min": p_min,
        "u_min": u_min,
        "u_max": u_max,
        "p_osc": p_osc,
        "p_hf_osc": p_hf_osc,
        "p_ref_l2": p_ref_l2,
        "p_center_linf": p_center_linf,
        "p_center_min": p_center_min,
        "p_center_max": p_center_max,
        "rho_osc": rho_osc,
    }


def main():
    v = _load_verify_module()
    rows = [v.case_14(), _case15_quality(v), v.case_24()]
    failures = int(sum(0 if bool(row.get("pass")) else 1 for row in rows))
    result = {
        "metric": failures,
        "quality_failures": failures,
        "pass": failures == 0,
        "cases": rows,
    }
    print("QUALITY_JSON " + json.dumps(result, sort_keys=True))
    print(failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
