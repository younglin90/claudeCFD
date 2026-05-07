#!/usr/bin/env python3
"""Quality metric for 15_E cavitation profile matching.

The final stdout line is JSON for codex-autoresearch.  The script intentionally
uses the existing low-order 15_E validation setup; it does not enable high-order
interface reconstruction.
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / ".codex-loop" / "verify_08_26_acceptance.py"


def _load_base():
    spec = importlib.util.spec_from_file_location("verify_08_26_acceptance", BASE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _scaled_l2(a, b, scale):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)) / max(float(scale), 1.0e-30))


def main() -> int:
    v = _load_base()
    eos_air = v.make_eos("ideal", gamma=1.4, kv=717.5)
    eos_water = v.make_eos("sg", gamma=4.4, pinf=6.0e8, kv=474.2)
    n = int(os.environ.get("CASE15_N", "100"))
    t_end = float(os.environ.get("CASE15_T_END", "9.5e-4"))
    pressure_closure = os.environ.get("FIVE_EQ_IMEX_PRESSURE_CLOSURE", "compressive_recovery")
    x, dx, W0 = v._case15_initial_state(n, eos_air, eos_water)
    start = time.time()
    out = v.solve(
        eos_air,
        eos_water,
        W0,
        dx,
        t_end,
        bc_l="transmissive",
        bc_r="transmissive",
        cfl=0.01,
        time_integrator=os.environ.get("FIVE_EQ_IMEX_TIME_INTEGRATOR", "imex_ssp3"),
        mixture_kind="kapila",
        kapila_closure=True,
        alpha_scheme="cicsam",
        alpha_pure_tol=1.0e-6,
        max_steps=250000,
        pressure_closure=pressure_closure,
    )
    W = out["W"]
    rho = v._rho_mix(W, eos_air, eos_water)
    exact = v._case15_computed_reference(x, eos_air, eos_water)
    complete = out.get("terminated_reason") is None and out["t_final"] >= t_end - 1.0e-14
    finite = v._finite_admissible(W, rho)
    i_center = max(0, min(len(x) - 2, int(np.searchsorted(x, 0.5) - 1)))
    metrics = {
        "alpha_l2": _scaled_l2(W[0], exact["alpha"], 1.0),
        "rho_l2": _scaled_l2(rho, exact["rho"], 1000.0),
        "u_l2": _scaled_l2(W[3], exact["u"], 100.0),
        "p_l2": _scaled_l2(W[4], exact["p"], 1.0e5),
        "alpha_corr": float(v._pearson(W[0], exact["alpha"])),
        "rho_corr": float(v._pearson(rho, exact["rho"])),
        "u_corr": float(v._pearson(W[3], exact["u"])),
        "p_corr": float(v._pearson(W[4], exact["p"])),
        "center_u_jump": float(abs(W[3][i_center + 1] - W[3][i_center])),
        "center_ref_jump": float(abs(exact["u"][i_center + 1] - exact["u"][i_center])),
        "p_osc": float(v._checkerboard(W[4], 1.0e5)),
        "rho_osc": float(v._checkerboard(rho, 1000.0)),
        "alpha_peak": float(np.max(W[0])),
        "rho_min": float(np.min(rho)),
        "p_min": float(np.min(W[4])),
        "u_min": float(np.min(W[3])),
        "u_max": float(np.max(W[3])),
        "steps": int(out["step"]),
        "wall": float(time.time() - start),
        "complete": bool(complete),
        "finite": bool(finite),
        "terminated_reason": out.get("terminated_reason"),
    }
    metrics["composite_l2"] = float(
        metrics["alpha_l2"] + metrics["rho_l2"] + metrics["u_l2"] + metrics["p_l2"]
    )
    metrics["admissible_guard"] = bool(
        finite
        and complete
        and metrics["p_min"] > 0.0
        and metrics["alpha_peak"] > 0.2
        and metrics["rho_min"] < 700.0
        and metrics["p_min"] < 8.0e4
        and metrics["center_u_jump"] < max(20.0, 3.0 * metrics["center_ref_jump"])
        and metrics["p_osc"] < 0.03
        and metrics["rho_osc"] < 0.25
    )
    row = {
        "name": "15E cavitation quality",
        "pass": metrics["admissible_guard"],
        "W": W,
        "rho": rho,
        "x": x,
        "exact": exact,
        "alpha_num": W[0],
        "metrics": metrics,
    }
    v._save_multi_plot(
        "15_E",
        [row],
        f"15_E quality composite={metrics['composite_l2']:.4f} guard={metrics['admissible_guard']}",
    )
    print(json.dumps(metrics, sort_keys=True))
    return 0 if metrics["admissible_guard"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
