"""Finalize the best paper-style PLBE Re=1000 cavity run."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from ghia_validation import get_ghia_data
from lbm_plbe_cavity import PLBECavity
from paper_60case_benchmark import write_history_csv, write_vtk
from solver_unified_safe_nn import _residual_norm
from verify_plbe_re1000 import run_plbe


OUT = Path("paper_revision_data") / "plbe_re1000_paper_mach"


def ghia_metrics(case, f):
    _, ux, uy = case.macro(f)
    y_g, u_g, x_g, v_g = get_ghia_data(1000)
    n = case.N
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    u_center = ux[:, n // 2] / case.U_wall
    v_center = uy[n // 2, :] / case.U_wall
    u_interp = np.interp(y_g, y, u_center)
    v_interp = np.interp(x_g, x, v_center)
    return {
        "u_rms": float(np.sqrt(np.mean((u_interp - u_g) ** 2))),
        "v_rms": float(np.sqrt(np.mean((v_interp - v_g) ** 2))),
        "u_linf": float(np.max(np.abs(u_interp - u_g))),
        "v_linf": float(np.max(np.abs(v_interp - v_g))),
        "centerline_max": float(max(np.max(np.abs(u_interp - u_g)), np.max(np.abs(v_interp - v_g)))),
    }


def write_centerline_csv(path, case, f):
    _, ux, uy = case.macro(f)
    n = case.N
    grid = np.linspace(0.0, 1.0, n)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["s", "u_vertical_over_ulid", "v_horizontal_over_ulid"])
        for i in range(n):
            wr.writerow([grid[i], ux[i, n // 2] / case.U_wall, uy[n // 2, i] / case.U_wall])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mach_star = 0.058
    ratio = 0.25
    gamma = ratio * ratio
    mach = mach_star * ratio
    u_wall = mach / math.sqrt(3.0)
    case = PLBECavity(N=129, Re=1000, U_wall=u_wall, gamma=gamma)
    f, hist = run_plbe(case, max_steps=240000, tol=1.0e-6, check_every=100, verbose=False)
    _, native_res = _residual_norm(case, f)
    metrics = {
        "case": "lid_driven_cavity_Re1000_N129",
        "implementation": "Guo-Zhao-Shi PLBE EDF + unknown-only nonequilibrium extrapolation BC",
        "mach_star": mach_star,
        "mach": mach,
        "gamma": gamma,
        "u_wall": u_wall,
        "tau": case.tau,
        "omega": case.omega,
        "lbe_calls": int(hist[-1][2]),
        "velocity_change_100": float(hist[-1][1]),
        "native_residual": float(native_res),
        "paper_converged": bool(np.isfinite(hist[-1][1]) and hist[-1][1] < 1.0e-6),
        "ghia": ghia_metrics(case, f),
    }
    write_history_csv(OUT / "final_gamma00625_history.csv", hist)
    write_centerline_csv(OUT / "final_gamma00625_centerline.csv", case, f)
    write_vtk(OUT / "final_gamma00625_field.vtk", case, f)
    (OUT / "final_gamma00625_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
