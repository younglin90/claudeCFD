"""Paper-faithful PLBE check for lid-driven cavity Re=1000.

The Guo-Zhao-Shi PLBE paper chooses a reference Mach number M*=0.058 and
sets gamma=(M/M*)^2. This script tests that policy instead of forcing the
existing high-Mach local cavity setting U_wall=0.1 into PLBE.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from lbm_plbe_cavity import PLBECavity
from solver_unified_safe_nn import _residual_norm
from verify_plbe_re1000 import BOUNCEBACK_REF_VTK, read_vtk_velocity, run_plbe


OUT = Path("paper_revision_data") / "plbe_re1000_paper_mach"


def scaled_velocity_error(ux_ref, uy_ref, ref_u, ux, uy, u_wall):
    ux_ref_s = ux_ref / ref_u
    uy_ref_s = uy_ref / ref_u
    ux_s = ux / u_wall
    uy_s = uy / u_wall
    dux = ux_s - ux_ref_s
    duy = uy_s - uy_ref_s
    num = float(np.sqrt(np.sum(dux * dux + duy * duy)))
    den = max(float(np.sqrt(np.sum(ux_ref_s * ux_ref_s + uy_ref_s * uy_ref_s))), 1.0e-30)
    mag = np.sqrt(dux * dux + duy * duy)
    return {
        "rel_l2_scaled_vs_highmach_picard": num / den,
        "linf_scaled_vs_highmach_picard": float(np.max(mag)),
        "rms_scaled_vs_highmach_picard": float(np.sqrt(np.mean(dux * dux + duy * duy))),
    }


def run_config(label, mach, gamma, ux_ref, uy_ref, max_steps):
    cs = 1.0 / math.sqrt(3.0)
    u_wall = mach * cs
    case = PLBECavity(N=129, Re=1000, U_wall=u_wall, gamma=gamma)
    f, hist = run_plbe(case, max_steps=max_steps, tol=1.0e-6, check_every=100, verbose=False)
    final_change = float(hist[-1][1]) if hist else float("nan")
    lbe = int(hist[-1][2]) if hist else 0
    finite = bool(np.all(np.isfinite(f)))
    if finite:
        _, native_res = _residual_norm(case, f)
        _, ux, uy = case.macro(f)
        err = scaled_velocity_error(ux_ref, uy_ref, 0.1, ux, uy, u_wall)
    else:
        native_res = float("nan")
        err = {
            "rel_l2_scaled_vs_highmach_picard": float("inf"),
            "linf_scaled_vs_highmach_picard": float("inf"),
            "rms_scaled_vs_highmach_picard": float("inf"),
        }
    return {
        "label": label,
        "mach": mach,
        "u_wall": u_wall,
        "gamma": gamma,
        "tau": case.tau,
        "omega": case.omega,
        "lbe_calls": lbe,
        "velocity_change_100": final_change,
        "native_residual": float(native_res),
        "finite": finite,
        "paper_converged": bool(np.isfinite(final_change) and final_change < 1.0e-6),
        **err,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ux_ref, uy_ref = read_vtk_velocity(BOUNCEBACK_REF_VTK, 129)
    mstar = 0.058
    configs = []
    for ratio in [0.25, 0.354, 0.5, 0.707, 0.9]:
        mach = mstar * ratio
        configs.append((f"paper_gamma_{ratio * ratio:.4f}", mach, ratio * ratio))
    configs.append(("standard_lbe_same_mstar", mstar, 1.0))
    configs.append(("local_highmach_plbe_gamma1", 0.1 / (1.0 / math.sqrt(3.0)), 1.0))

    rows = []
    for label, mach, gamma in configs:
        row = run_config(label, mach, gamma, ux_ref, uy_ref, max_steps=80000)
        rows.append(row)
        rows_sorted = sorted(rows, key=lambda r: (not r["paper_converged"], r["rel_l2_scaled_vs_highmach_picard"], r["lbe_calls"]))
        with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows_sorted[0].keys()))
            wr.writeheader()
            wr.writerows(rows_sorted)
        metrics = {
            "mstar": mstar,
            "variant_count": len(rows_sorted),
            "paper_converged_count": sum(int(r["paper_converged"]) for r in rows_sorted),
            "best": rows_sorted[0],
        }
        (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(row, sort_keys=True), flush=True)
    print((OUT / "metrics.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
