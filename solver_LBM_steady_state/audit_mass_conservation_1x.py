from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from paper_60case_benchmark_no_force import macro_of
from paper_60case_benchmark_no_force_scaling import CASE_IDS, _fluid_mask, case_factory_scaled


def _flux_metrics(case, rho, ux, uy, fluid):
    has_open = hasattr(case, "U_in")
    if not has_open:
        return "", np.nan, np.nan, np.nan, np.nan
    fin = 0.0
    fout = 0.0
    if fluid.ndim == 2 and fluid.shape[1] > 1:
        inlet = fluid[:, 0]
        right = fluid[:, -1]
        if np.any(inlet):
            fin += float(np.sum(rho[inlet, 0] * ux[inlet, 0]))
        if np.any(right):
            fout += float(np.sum(rho[right, -1] * ux[right, -1]))
    if type(case).__name__ == "NoForceTJunctionRectCase" and fluid.shape[0] > 1:
        top = fluid[-1, :]
        if np.any(top):
            fout += float(np.sum(rho[-1, top] * uy[-1, top]))
    scale = max(abs(fin), abs(fout), 1.0e-30)
    closure = abs(fin - fout) / scale
    return "open", fin, fout, fin - fout, closure


def audit(out_dir: Path):
    rows = []
    for base_id in CASE_IDS:
        case_id, _label, _tol, factory = case_factory_scaled(base_id, 1)
        npz_path = out_dir / "npz" / f"{case_id}__proposed.npz"
        if not npz_path.exists():
            rows.append({"base_case_id": base_id, "case_id": case_id, "status": "missing"})
            continue
        case = factory()
        fluid = _fluid_mask(case)
        f0 = case.initial_field()
        f = np.load(npz_path, allow_pickle=False)["f"]
        rho0, ux0, uy0 = macro_of(case, f0)
        rho, ux, uy = macro_of(case, f)
        mass0 = float(np.sum(rho0[fluid]))
        mass = float(np.sum(rho[fluid]))
        mass_rel_drift = abs(mass - mass0) / max(abs(mass0), abs(mass), 1.0e-30)
        speed = np.sqrt(ux * ux + uy * uy)
        kind, fin, fout, net, closure = _flux_metrics(case, rho, ux, uy, fluid)
        if kind == "":
            kind = "closed_or_periodic"
        rows.append(
            {
                "base_case_id": base_id,
                "case_id": case_id,
                "status": "ok",
                "mass_kind": kind,
                "mass_initial": mass0,
                "mass_final": mass,
                "mass_rel_drift": mass_rel_drift,
                "rho_min": float(np.min(rho[fluid])),
                "rho_mean": float(np.mean(rho[fluid])),
                "rho_max": float(np.max(rho[fluid])),
                "speed_mean": float(np.mean(speed[fluid])),
                "speed_max": float(np.max(speed[fluid])),
                "inflow": fin,
                "outflow": fout,
                "net_flux": net,
                "flux_closure_rel": closure,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()
    rows = audit(args.out_dir)
    fieldnames = [
        "base_case_id",
        "case_id",
        "status",
        "mass_kind",
        "mass_initial",
        "mass_final",
        "mass_rel_drift",
        "rho_min",
        "rho_mean",
        "rho_max",
        "speed_mean",
        "speed_max",
        "inflow",
        "outflow",
        "net_flux",
        "flux_closure_rel",
    ]
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(",".join(fieldnames))
    for row in rows:
        print(",".join(str(row.get(k, "")) for k in fieldnames))


if __name__ == "__main__":
    main()
