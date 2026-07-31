"""Paper-grade strict T-junction benchmark with tight reference metadata.

Outputs are written under ``papers_data/t_junction_rect_strict`` and are kept
separate from relaxed/smoke results.  The benchmark uses a force-free
left-inlet/right-top-outlet T-junction with exact cell-count channel widths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "24")
os.environ.setdefault("OMP_NUM_THREADS", "24")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from no_force_suite.no_force_lb_core import NoForceTJunctionRectCase
from no_force_suite.no_force_masks import make_t_junction_rect_lattice_mask
from paper_60case_benchmark_no_force import macro_of, write_history_csv, write_vtk
from paper_60case_benchmark_no_force_scaling import run_method_with_wall
from solver_baseline import reference_residual_norm


OUT = Path("papers_data") / "t_junction_rect_strict"
METHODS = [
    "picard_lbm",
    "anderson_lbm",
    "preconditioned_lbm",
    "inexact_newton_lbe",
    "dual_time_mg_lbm",
    "proposed",
]


def level_spec(level: int):
    level = int(level)
    if level not in {1, 2, 3}:
        raise ValueError("level must be 1, 2, or 3")
    ny = 128 * level
    nx = 192 * level
    width = 32 * level
    u_in = 0.04 / float(level)
    tol = 1.0e-7 / float(level)
    return ny, nx, width, u_in, tol


def make_case(level: int):
    ny, nx, width, u_in, _tol = level_spec(level)
    chi = make_t_junction_rect_lattice_mask(ny, nx, width)
    return NoForceTJunctionRectCase(chi, nu=0.05, U_in=u_in, outlet_bc="pressure")


def case_id(level: int):
    ny, nx, width, _u_in, _tol = level_spec(level)
    return f"t_junction_rect_strict_Nx{nx}_Ny{ny}_W{width}__{level}x"


def _file_hash(paths):
    h = hashlib.sha256()
    for path in paths:
        p = Path(path)
        h.update(str(p).encode("utf-8"))
        if p.exists():
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()[:16]


def geometry_hash(case):
    h = hashlib.sha256()
    h.update(np.asarray(case.chi, dtype=np.float64).tobytes())
    h.update(str((case.Ny, case.Nx, case.U_in, case.nu)).encode("utf-8"))
    return h.hexdigest()[:16]


def operator_hash():
    return _file_hash([
        "no_force_suite/no_force_lb_core.py",
        "no_force_suite/no_force_masks.py",
        "solver_baseline.py",
        "solver_anderson.py",
        "solver_proposed_single.py",
        "paper_faithful_baselines.py",
        "paper_tjunction_strict_benchmark.py",
    ])


def flux_metrics(case, f):
    rho, ux, uy = case.macro(f)
    fluid = case.chi > 0.0
    inlet = fluid[:, 0]
    right = fluid[:, -1]
    top = fluid[-1, :]
    fin = float(np.sum(rho[inlet, 0] * ux[inlet, 0]))
    fright = float(np.sum(rho[right, -1] * ux[right, -1]))
    ftop = float(np.sum(rho[-1, top] * uy[-1, top]))
    imbalance = abs(fin - fright - ftop) / max(abs(fin), abs(fright) + abs(ftop), 1.0e-30)
    speed = np.sqrt(ux[fluid] * ux[fluid] + uy[fluid] * uy[fluid])
    return {
        "flux_inlet": fin,
        "flux_right_outlet": fright,
        "flux_top_outlet": ftop,
        "flux_net_imbalance": float(imbalance),
        "rho_min": float(np.min(rho[fluid])),
        "rho_max": float(np.max(rho[fluid])),
        "max_speed": float(np.max(speed)) if speed.size else 0.0,
        "max_mach_lattice": float(np.max(speed) / (1.0 / np.sqrt(3.0))) if speed.size else 0.0,
        "finite": int(np.all(np.isfinite(f))),
    }


def velocity_error(case, f_ref, f):
    _rho_ref, ux_ref, uy_ref = case.macro(f_ref)
    _rho, ux, uy = case.macro(f)
    fluid = case.chi > 0.0
    du = ux[fluid] - ux_ref[fluid]
    dv = uy[fluid] - uy_ref[fluid]
    ref = np.sqrt(ux_ref[fluid] ** 2 + uy_ref[fluid] ** 2)
    den = max(float(np.sqrt(np.sum(ref * ref))), 1.0e-30)
    return {
        "vel_rel_l2_vs_ref": float(np.sqrt(np.sum(du * du + dv * dv)) / den),
        "vel_linf_vs_ref": float(max(np.max(np.abs(du)), np.max(np.abs(dv))) if du.size else 0.0),
        "vel_rms_vs_ref": float(np.sqrt(np.mean(du * du + dv * dv)) if du.size else 0.0),
    }


def vorticity(case, f):
    _rho, ux, uy = case.macro(f)
    dvdx = np.gradient(uy, axis=1)
    dudy = np.gradient(ux, axis=0)
    return (dvdx - dudy) * (case.chi > 0.0)


def write_field_csv(path, case, f):
    path.parent.mkdir(parents=True, exist_ok=True)
    rho, ux, uy = case.macro(f)
    vort = vorticity(case, f)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["y", "x", "fluid", "rho", "ux", "uy", "speed", "vorticity"])
        for y in range(case.Ny):
            for x in range(case.Nx):
                wr.writerow([
                    y,
                    x,
                    int(case.chi[y, x] > 0.0),
                    rho[y, x],
                    ux[y, x],
                    uy[y, x],
                    float(np.hypot(ux[y, x], uy[y, x])),
                    vort[y, x],
                ])


def plot_histories(path, histories, xkey):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.0, 4.0), dpi=180)
    for method, hist in histories.items():
        if not hist:
            continue
        if xkey == "lbe":
            x = [max(float(row[2]), 1.0) for row in hist]
            xlabel = "LBE calls"
        else:
            x = [max(float(row[3]), 1.0e-6) for row in hist]
            xlabel = "Wall seconds"
        y = [max(float(row[1]), 1.0e-16) for row in hist]
        plt.loglog(x, y, marker="o", markersize=2, linewidth=1.1, label=method)
    plt.xlabel(xlabel)
    plt.ylabel("Native residual norm")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_field(path, case, f, quantity):
    path.parent.mkdir(parents=True, exist_ok=True)
    rho, ux, uy = case.macro(f)
    if quantity == "speed":
        arr = np.sqrt(ux * ux + uy * uy)
        title = "Velocity magnitude"
    elif quantity == "vorticity":
        arr = vorticity(case, f)
        title = "Vorticity"
    else:
        raise ValueError(quantity)
    arr = np.where(case.chi > 0.0, arr, np.nan)
    plt.figure(figsize=(7.0, 4.0), dpi=180)
    plt.imshow(arr, origin="lower", cmap="viridis", aspect="equal")
    plt.colorbar(label=quantity)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_error(path, case, f_ref, f):
    path.parent.mkdir(parents=True, exist_ok=True)
    _rho_ref, ux_ref, uy_ref = case.macro(f_ref)
    _rho, ux, uy = case.macro(f)
    err = np.sqrt((ux - ux_ref) ** 2 + (uy - uy_ref) ** 2)
    err = np.where(case.chi > 0.0, err, np.nan)
    plt.figure(figsize=(7.0, 4.0), dpi=180)
    plt.imshow(err, origin="lower", cmap="magma", aspect="equal")
    plt.colorbar(label="velocity error")
    plt.title("Velocity error vs tight reference")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def picard_polish(case, f, target_tol, max_steps, check_every=200):
    hist = []
    t0 = time.perf_counter()
    lbe = 0
    step_fn = getattr(case, "lbe_step_inplace", None)
    work = np.empty_like(f) if callable(step_fn) else None
    r = f - case.lbe_step(f)
    lbe += 1
    rn = reference_residual_norm(case, r)
    hist.append((0, rn, lbe, time.perf_counter() - t0))
    for step in range(1, max_steps + 1):
        if callable(step_fn):
            f_new = step_fn(f, work)
        else:
            f_new = case.lbe_step(f)
        lbe += 1
        if step % check_every == 0:
            r = f_new - case.lbe_step(f_new)
            lbe += 1
            rn = reference_residual_norm(case, r)
            hist.append((step, rn, lbe, time.perf_counter() - t0))
            if not np.isfinite(rn) or rn <= target_tol:
                f = f_new
                break
        if callable(step_fn):
            f, work = f_new, f
        else:
            f = f_new
    return f, hist, time.perf_counter() - t0


def build_reference(level, out_dir, ref_tol, polish_steps, force=False):
    case = make_case(level)
    meta_path = out_dir / "reference_meta.json"
    npz_path = out_dir / "reference_tight.npz"
    op_hash = operator_hash()
    geom_hash = geometry_hash(case)
    if (not force) and meta_path.exists() and npz_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("operator_hash") == op_hash and meta.get("geometry_hash") == geom_hash and float(meta.get("reference_tol", 1.0)) <= ref_tol:
            data = np.load(npz_path, allow_pickle=True)
            return case, data["f"], meta

    # Trusted reference: inexact Newton to get close, followed by native Picard
    # polish.  It is not included as a comparison method.
    f0, hist_newton, wall_newton = run_method_with_wall(
        "inexact_newton_lbe", case, tol=max(ref_tol * 10.0, 1.0e-10), max_steps=polish_steps, verbose=False
    )
    f_ref, hist_polish, wall_polish = picard_polish(case, f0, target_tol=ref_tol, max_steps=polish_steps, check_every=200)
    final_res = float(hist_polish[-1][1]) if hist_polish else float("inf")
    meta = {
        "case_id": case_id(level),
        "level": int(level),
        "reference_solver": "inexact_newton_lbe_plus_picard_polish",
        "reference_tol": float(ref_tol),
        "reference_final_residual": final_res,
        "reference_converged": int(np.isfinite(final_res) and final_res <= ref_tol),
        "newton_wall_seconds": float(wall_newton),
        "picard_polish_wall_seconds": float(wall_polish),
        "newton_history_rows": len(hist_newton),
        "picard_polish_history_rows": len(hist_polish),
        "geometry_hash": geom_hash,
        "operator_hash": op_hash,
        **flux_metrics(case, f_ref),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, f=f_ref)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_history_csv(out_dir / "histories" / "reference_newton.csv", hist_newton)
    write_history_csv(out_dir / "histories" / "reference_picard_polish.csv", hist_polish)
    write_vtk(out_dir / "fields" / "reference_tight.vtk", case, f_ref)
    write_field_csv(out_dir / "fields" / "reference_tight.csv", case, f_ref)
    return case, f_ref, meta


def run_level(level, methods, force_reference=False, no_vtk=False):
    out_dir = OUT / f"{level}x"
    out_dir.mkdir(parents=True, exist_ok=True)
    _ny, _nx, _width, _u_in, tol = level_spec(level)
    ref_tol = min(tol * 0.1, 1.0e-8)
    polish_steps = {1: 120000, 2: 180000, 3: 240000}[int(level)]
    case_ref, f_ref, ref_meta = build_reference(level, out_dir, ref_tol, polish_steps, force=force_reference)
    if not ref_meta.get("reference_converged"):
        print(f"[warn] reference not tight for {case_id(level)}: residual={ref_meta.get('reference_final_residual'):.3e}", flush=True)

    rows = []
    histories = {}
    fields_dir = out_dir / "fields"
    hist_dir = out_dir / "histories"
    fig_dir = out_dir / "figures"
    for method in methods:
        case = make_case(level)
        max_steps = {1: 120000, 2: 180000, 3: 240000}[int(level)]
        f, hist, wall = run_method_with_wall(method, case, tol=tol, max_steps=max_steps, verbose=False)
        final_res = float(hist[-1][1]) if hist else float("inf")
        row = {
            "case_id": case_id(level),
            "level": int(level),
            "method": method,
            "tol": float(tol),
            "lbe_calls": int(hist[-1][2]) if hist else 0,
            "wall_seconds": float(wall),
            "final_residual": final_res,
            "converged": int(np.isfinite(final_res) and final_res <= 5.0 * tol),
            "reference_solver": ref_meta["reference_solver"],
            "reference_tol": float(ref_meta["reference_tol"]),
            "reference_final_residual": float(ref_meta["reference_final_residual"]),
            "reference_converged": int(ref_meta["reference_converged"]),
            "geometry_hash": ref_meta["geometry_hash"],
            "operator_hash": ref_meta["operator_hash"],
        }
        row.update(velocity_error(case, f_ref, f))
        row.update(flux_metrics(case, f))
        rows.append(row)
        histories[method] = hist
        write_history_csv(hist_dir / f"{method}.csv", hist)
        if not no_vtk:
            write_vtk(fields_dir / f"{method}.vtk", case, f)
        write_field_csv(fields_dir / f"{method}.csv", case, f)
        if method == "proposed":
            plot_field(fig_dir / "velocity_magnitude_proposed.png", case, f, "speed")
            plot_field(fig_dir / "vorticity_proposed.png", case, f, "vorticity")
            plot_error(fig_dir / "error_contour_proposed.png", case, f_ref, f)
        print(
            f"  {method:22s} wall={wall:8.3f} lbe={row['lbe_calls']:8d} "
            f"res={final_res:.3e} rel={row['vel_rel_l2_vs_ref']:.3e} flux={row['flux_net_imbalance']:.3e}",
            flush=True,
        )

    fields = list(rows[0].keys()) if rows else []
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps({"rows": rows, "reference": ref_meta}, indent=2), encoding="utf-8")
    plot_histories(fig_dir / "residual_vs_lbe_calls.png", histories, "lbe")
    plot_histories(fig_dir / "residual_vs_wall_seconds.png", histories, "wall")
    with (fig_dir / "flux_balance.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["method", "flux_inlet", "flux_right_outlet", "flux_top_outlet", "flux_net_imbalance"])
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row[k] for k in ["method", "flux_inlet", "flux_right_outlet", "flux_top_outlet", "flux_net_imbalance"]})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default="1", help="comma-separated levels: 1,2,3")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--force-reference", action="store_true")
    parser.add_argument("--no-vtk", action="store_true")
    args = parser.parse_args()
    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        raise ValueError(f"unknown methods: {bad}")

    all_rows = []
    for level in levels:
        print(f"[case] {case_id(level)}", flush=True)
        all_rows.extend(run_level(level, methods, force_reference=args.force_reference, no_vtk=args.no_vtk))

    OUT.mkdir(parents=True, exist_ok=True)
    if all_rows:
        fields = list(all_rows[0].keys())
        with (OUT / "summary_all_levels.csv").open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            wr.writeheader()
            wr.writerows(all_rows)
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
