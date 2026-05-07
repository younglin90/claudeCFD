#!/usr/bin/env python3
"""Fast metrics wrapper for case 24 at N=100.

The regular acceptance verifier is intentionally broad.  This wrapper narrows
case 24 into a single machine-readable score for autoresearch iterations:
shock-location error, post-shock density hump, exact error, and oscillation
guards.  The final stdout line is JSON for the autoresearch helper.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODEX_LOOP = ROOT / ".codex-loop"
VERIFY = CODEX_LOOP / "verify_08_26_acceptance.py"


def _load_verifier():
    for path in (str(ROOT), str(CODEX_LOOP)):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location("verify_08_26_acceptance", VERIFY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {VERIFY}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _rho_hump_metrics(row):
    x = np.asarray(row["x"], dtype=float)
    rho = np.asarray(row["rho"], dtype=float)
    rho_ref = np.asarray(row["exact"]["rho"], dtype=float)
    plateau = (x > 0.15) & (x < 0.75)
    if int(np.count_nonzero(plateau)) < 3:
        plateau = np.ones_like(x, dtype=bool)
    residual = rho - rho_ref
    ids = np.flatnonzero(plateau)
    j_max = int(ids[np.argmax(residual[ids])])
    j_min = int(ids[np.argmin(residual[ids])])
    ref_jump = max(float(np.max(rho_ref) - np.min(rho_ref)), 1.0)
    ref_post = max(float(np.max(np.abs(rho_ref[plateau]))), 1.0)
    hump_pos = max(0.0, float(residual[j_max]))
    hump_neg = max(0.0, float(-residual[j_min]))
    return {
        "rho_hump_x": float(x[j_max]),
        "rho_hump_abs": hump_pos,
        "rho_hump_rel_jump": hump_pos / ref_jump,
        "rho_dip_abs": hump_neg,
        "rho_dip_rel_jump": hump_neg / ref_jump,
        "rho_hump_rel_post": hump_pos / ref_post,
    }


def main() -> int:
    os.environ["FIVE_EQ_CASE24_N"] = os.environ.get("FIVE_EQ_CASE24_N", "100")
    os.environ["FIVE_EQ_IMEX_CHARACTERISTIC_RECON"] = "0"
    os.environ["MPLCONFIGDIR"] = os.environ.get("MPLCONFIGDIR", "/tmp/mpl")

    v = _load_verifier()
    rows = [v._case24_subcase(psi) for psi in (0.0, 0.25, 0.5, 0.75, 1.0)]
    ok_native = bool(all(row["pass"] for row in rows))
    v._save_multi_plot("24_H", rows, f"24_H N=100 no-characteristic pass={ok_native}")

    subcases = []
    for row in rows:
        m = dict(row["metrics"])
        m.update(_rho_hump_metrics(row))
        m["pass"] = bool(row["pass"])
        m["name"] = row["name"]
        subcases.append(m)

    fail_count = int(sum(0 if m["pass"] else 1 for m in subcases))
    hf_fail_count = int(sum(0 if m.get("hf_oscillation_ok", False) else 1 for m in subcases))
    max_shock_cells = max(float(m["shock_cells"]) for m in subcases)
    max_rho_hump = max(float(m["rho_hump_rel_jump"]) for m in subcases)
    max_rho_dip = max(float(m["rho_dip_rel_jump"]) for m in subcases)
    max_rho_l2 = max(float(m["rho_profile_l2"]) for m in subcases)
    max_peak = max(
        max(float(m.get(f"{field}_sharp_overshoot", 0.0)) for field in ("rho", "u", "p"))
        for m in subcases
    )
    max_l2 = max(
        max(float(m[f"{field}_profile_l2"]) for field in ("rho", "u", "p"))
        for m in subcases
    )
    characteristic_used = int(os.environ.get("FIVE_EQ_IMEX_CHARACTERISTIC_RECON", "0") not in {"0", "false", "off", "no"})

    # Lower is better. Large weights make hard failures dominate soft quality.
    score = (
        1000.0 * fail_count
        + 1000.0 * hf_fail_count
        + 500.0 * characteristic_used
        + 25.0 * max(0.0, max_shock_cells - 0.5)
        + 350.0 * max_rho_hump
        + 180.0 * max_rho_dip
        + 120.0 * max_rho_l2
        + 50.0 * max_peak
        + 10.0 * max_l2
    )

    eps = 1.0e-12
    accepted = bool(
        fail_count == 0
        and hf_fail_count == 0
        and characteristic_used == 0
        and max_shock_cells <= 1.5 + eps
        and max_rho_hump <= 0.010 + eps
        and max_rho_dip <= 0.050 + eps
        and max_rho_l2 <= 0.050 + eps
        and max_peak <= 0.05 + eps
        and max_l2 <= 0.20 + eps
    )
    out = {
        "score": float(score),
        "accepted": accepted,
        "fail_count": fail_count,
        "hf_fail_count": hf_fail_count,
        "characteristic_used": characteristic_used,
        "max_shock_cells": float(max_shock_cells),
        "max_rho_hump_rel_jump": float(max_rho_hump),
        "max_rho_dip_rel_jump": float(max_rho_dip),
        "max_rho_profile_l2": float(max_rho_l2),
        "max_peak_overshoot": float(max_peak),
        "max_profile_l2": float(max_l2),
        "subcases": subcases,
    }
    print(json.dumps(out, sort_keys=True))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
