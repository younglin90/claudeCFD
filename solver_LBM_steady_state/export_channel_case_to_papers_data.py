from __future__ import annotations

"""Export one no-force scaling case into papers_data/<case_id>.

- Source data: paper_revision_data/no_force_scaling_benchmark
- Outputs: per-case summary, histories, fields, vtk, figures
  (residual plots, contour/profile, and accuracy comparison)

Usage:
    python3 export_channel_case_to_papers_data.py --case-id channel_n32__3x
"""

import argparse
import csv
import json
from pathlib import Path
import shutil
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_60case_benchmark_no_force import macro_of
from paper_60case_benchmark_no_force_scaling import (
    _load_cached,
    case_factory_scaled,
)

SOURCE_DIR = Path("paper_revision_data") / "no_force_scaling_benchmark"
SOURCE_HIST = SOURCE_DIR / "histories"
SOURCE_VTK = SOURCE_DIR / "vtk"
CACHE_DIR = SOURCE_DIR / "npz_cache"
DST_ROOT = Path("papers_data")

METHOD_LABELS = {
    "picard_lbm": "Picard",
    "anderson_lbm": "Anderson",
    "preconditioned_lbm": "Preconditioned",
    "inexact_newton_lbe": "Inexact Newton",
    "dual_time_mg_lbm": "Dual-time MG",
    "proposed": "SafeNN",
}


def safe_float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return float(default)


def parse_case_id(value: str) -> Tuple[str, int]:
    if "__" not in value or not value.endswith("x"):
        raise ValueError(f"bad case id '{value}', expected '<base>__<level>x'")
    base, level_token = value.rsplit("__", 1)
    try:
        level = int(level_token[:-1])
    except Exception as exc:
        raise ValueError(f"bad scaling level token '{level_token}'") from exc
    if level < 1 or level > 3:
        raise ValueError(f"scaling level must be 1..3, got {level}")
    return base, level


def read_case_rows(case_id: str):
    src = SOURCE_DIR / "summary.csv"
    rows = []
    with src.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("case_id") == case_id:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"no summary rows for {case_id}")
    return rows


def _load_cache_flexible(case_id: str, method: str):
    cached = _load_cached(case_id, method)
    if cached is not None:
        return cached
    candidates = sorted(
        CACHE_DIR.glob(f"{case_id}__{method}__*.npz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    with np.load(candidates[0], allow_pickle=False) as data:
        return data["f"], [tuple(row) for row in data["hist"].tolist()], float(data["wall"])


def vorticity_from_velocity(ux: np.ndarray, uy: np.ndarray, dx: float = 1.0, dy: float = 1.0):
    du_dy = np.gradient(ux, axis=0, edge_order=1) / dy
    dv_dx = np.gradient(uy, axis=1, edge_order=1) / dx
    # ω = du/dy - dv/dx
    return du_dy - dv_dx


def analytic_channel_profile(case) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = int(case.N)
    U_in = float(getattr(case, "U_in", 0.05))
    y = np.arange(N, dtype=np.float64)
    L = float(N - 1)
    if L <= 0:
        raise ValueError("invalid channel N")
    # laminar Poiseuille profile with approximately matching average velocity.
    ubar = U_in * (N - 1.0) / N
    ux_1d = 6.0 * ubar * (y / L) * (1.0 - y / L)
    ux = np.broadcast_to(ux_1d[:, None], (N, N)).copy()
    uy = np.zeros((N, N), dtype=np.float64)
    rho = np.ones((N, N), dtype=np.float64)
    return rho, ux, uy


def compute_accuracy(ref: Tuple[np.ndarray, np.ndarray, np.ndarray], case, f: np.ndarray):
    rho_ref, ux_ref, uy_ref = ref
    rho, ux, uy = macro_of(case, f)
    mask = getattr(case, "chi", np.ones((case.N, case.N), dtype=bool)) > 0

    du = (ux - ux_ref)[mask]
    dv = (uy - uy_ref)[mask]
    speed = np.sqrt(ux * ux + uy * uy)
    speed_ref = np.sqrt(ux_ref * ux_ref + uy_ref * uy_ref)
    ds = (speed - speed_ref)[mask]

    rel_den = float(np.sqrt(np.sum(ux_ref[mask] ** 2 + uy_ref[mask] ** 2)))
    rel_den = max(rel_den, 1.0e-30)
    rel_l2 = float(np.sqrt(np.sum(du * du + dv * dv)) / rel_den)
    linf = float(max(np.max(np.abs(du)) if du.size else 0.0, np.max(np.abs(dv)) if dv.size else 0.0))
    rms = float(np.sqrt(np.mean(du * du + dv * dv)) if du.size else 0.0)
    speed_rms = float(np.sqrt(np.mean(ds * ds)) if ds.size else 0.0)
    speed_l2 = float(np.sqrt(np.sum(ds * ds) / max(1, ds.size))) if ds.size else 0.0
    return rel_l2, linf, rms, speed_rms, speed_l2


def write_field_csvs(case_id: str, method: str, case, f: np.ndarray, ref: Tuple[np.ndarray, np.ndarray, np.ndarray], out_dir: Path):
    rho_ref, ux_ref, uy_ref = ref
    rho, ux, uy = macro_of(case, f)
    speed = np.sqrt(ux * ux + uy * uy)
    speed_ref = np.sqrt(ux_ref * ux_ref + uy_ref * uy_ref)
    vort = vorticity_from_velocity(ux, uy)
    vort_ref = vorticity_from_velocity(ux_ref, uy_ref)
    chi = getattr(case, "chi", np.ones_like(ux, dtype=np.float64))
    ny, nx = ux.shape

    mask = chi > 0
    base = f"{case_id}__{method}"

    macro_csv = out_dir / f"{base}__macro.csv"
    vort_csv = out_dir / f"{base}__vorticity.csv"
    err_csv = out_dir / f"{base}__error.csv"

    with macro_csv.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iy", "ix", "x", "y", "rho", "ux", "uy", "speed", "chi"])
        for j in range(ny):
            y = float(j)
            for i in range(nx):
                wr.writerow([j, i, float(i), y, rho[j, i], ux[j, i], uy[j, i], speed[j, i], int(chi[j, i] > 0)])

    with vort_csv.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iy", "ix", "x", "y", "vorticity", "vorticity_ref", "vorticity_error", "chi"])
        for j in range(ny):
            y = float(j)
            for i in range(nx):
                wr.writerow([
                    j,
                    i,
                    float(i),
                    y,
                    vort[j, i],
                    vort_ref[j, i],
                    vort[j, i] - vort_ref[j, i],
                    int(mask[j, i]),
                ])

    with err_csv.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow([
            "iy",
            "ix",
            "x",
            "y",
            "ux_err",
            "uy_err",
            "speed_err",
            "speed_ref",
            "speed",
            "chi",
        ])
        for j in range(ny):
            y = float(j)
            for i in range(nx):
                if mask[j, i]:
                    ux_err = ux[j, i] - ux_ref[j, i]
                    uy_err = uy[j, i] - uy_ref[j, i]
                    speed_err = speed[j, i] - speed_ref[j, i]
                else:
                    ux_err = float("nan")
                    uy_err = float("nan")
                    speed_err = float("nan")
                wr.writerow([
                    j,
                    i,
                    float(i),
                    y,
                    ux_err,
                    uy_err,
                    speed_err,
                    speed_ref[j, i],
                    speed[j, i],
                    int(mask[j, i]),
                ])

    # Optional exact/analytic error file
    base = out_dir / f"{case_id}__{method}__analytic_error.csv"
    with base.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iy", "ix", "x", "y", "ux_err", "uy_err", "speed_err", "vort_err", "mask"])
        for j in range(ny):
            y = float(j)
            for i in range(nx):
                if mask[j, i]:
                    ux_err = ux[j, i] - ux_ref[j, i]
                    uy_err = uy[j, i] - uy_ref[j, i]
                    speed_err = speed[j, i] - speed_ref[j, i]
                    vort_err = vort[j, i] - vort_ref[j, i]
                else:
                    ux_err = float("nan")
                    uy_err = float("nan")
                    speed_err = float("nan")
                    vort_err = float("nan")
                wr.writerow([j, i, float(i), y, ux_err, uy_err, speed_err, vort_err, int(mask[j, i])])

    # centerline profiles
    mid_x = nx // 2
    mid_y = ny // 2
    with (out_dir / f"{case_id}__{method}__centerline_x.csv").open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["x", "uy", "uy_ref", "uy_err", "ux", "ux_ref", "ux_err", "speed", "speed_ref", "speed_err"])
        for i in range(nx):
            wr.writerow([
                i,
                uy[mid_y, i],
                uy_ref[mid_y, i],
                uy[mid_y, i] - uy_ref[mid_y, i],
                ux[mid_y, i],
                ux_ref[mid_y, i],
                ux[mid_y, i] - ux_ref[mid_y, i],
                speed[mid_y, i],
                speed_ref[mid_y, i],
                speed[mid_y, i] - speed_ref[mid_y, i],
            ])

    with (out_dir / f"{case_id}__{method}__centerline_y.csv").open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["y", "ux", "ux_ref", "ux_err", "uy", "uy_ref", "uy_err", "speed", "speed_ref", "speed_err"])
        for j in range(ny):
            wr.writerow([
                j,
                ux[j, mid_x],
                ux_ref[j, mid_x],
                ux[j, mid_x] - ux_ref[j, mid_x],
                uy[j, mid_x],
                uy_ref[j, mid_x],
                uy[j, mid_x] - uy_ref[j, mid_x],
                speed[j, mid_x],
                speed_ref[j, mid_x],
                speed[j, mid_x] - speed_ref[j, mid_x],
            ])


def plot_residual_histories(case_id: str, methods: Iterable[str], rows: Iterable[dict], out_dir: Path):
    rows_by_method = {r["method"]: r for r in rows}

    def _load_history(method: str):
        path = SOURCE_HIST / f"{case_id}__{method}.csv"
        if not path.exists():
            return []
        out = []
        with path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                it = safe_float(row.get("iter"))
                res = safe_float(row.get("residual"))
                lbe = safe_float(row.get("lbe_calls"), 0.0)
                wall_raw = safe_float(row.get("wall_seconds_raw"), np.nan)
                wall = safe_float(row.get("wall_seconds"), np.nan)
                accepted = int(safe_float(row.get("accepted"), 1.0) > 0.5)
                phase = safe_float(row.get("phase"), 0)
                if not np.isfinite(res) or not np.isfinite(it) or not np.isfinite(lbe):
                    continue
                out.append({
                    "iter": int(it),
                    "residual": max(res, 1e-30),
                    "lbe_calls": int(lbe),
                    "wall_seconds_raw": wall_raw,
                    "wall_seconds": wall,
                    "accepted": accepted,
                    "phase": phase,
                })
        return out

    # Residual vs iteration
    fig, ax = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
    fig_a, ax_a = plt.figure(figsize=(6.4, 4.5), constrained_layout=True), None

    any_line = False
    for method in methods:
        hist = _load_history(method)
        if not hist:
            continue
        any_line = True
        xs = [r["iter"] for r in hist]
        ys = [r["residual"] for r in hist]
        ax.plot(xs, ys, label=METHOD_LABELS.get(method, method), linewidth=1.2)

        xs_ok = [r["iter"] for r in hist if r["accepted"] == 1]
        ys_ok = [r["residual"] for r in hist if r["accepted"] == 1]
        if xs_ok:
            if ax_a is None:
                fig_a = plt.figure(figsize=(6.4, 4.5), constrained_layout=True)
                ax_a = fig_a.add_subplot(1, 1, 1)
            ax_a.plot(xs_ok, ys_ok, label=METHOD_LABELS.get(method, method), linewidth=1.2)

    if any_line:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("native residual")
        ax.set_title(f"{case_id} residual")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
        fig.savefig(out_dir / "residual_vs_iteration.png", dpi=220)
        fig.savefig(out_dir / "residual_vs_iteration.pdf")

    if ax_a is not None:
        ax_a.set_xscale("log")
        ax_a.set_yscale("log")
        ax_a.set_xlabel("iteration")
        ax_a.set_ylabel("native residual")
        ax_a.set_title(f"{case_id} residual (accepted only)")
        ax_a.grid(True, which="both", alpha=0.3)
        ax_a.legend(fontsize=7)
        fig_a.savefig(out_dir / "residual_vs_iteration_accepted_only.png", dpi=220)
        fig_a.savefig(out_dir / "residual_vs_iteration_accepted_only.pdf")

    plt.close(fig)
    if fig_a is not None:
        plt.close(fig_a)

    # Residual vs wall time (both raw-local and cumulative monotonic)
    for accepted_only in [False, True]:
        suffix = "accepted_only" if accepted_only else "all"

        # raw wall plot
        fig_r, ax_r = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
        any_line = False
        for method in methods:
            hist = _load_history(method)
            if not hist:
                continue
            xs = []
            ys = []
            for h in hist:
                if accepted_only and not h["accepted"]:
                    continue
                if not np.isfinite(h["wall_seconds_raw"]):
                    continue
                xs.append(h["wall_seconds_raw"])
                ys.append(h["residual"])
            if not xs:
                continue
            any_line = True
            ax_r.plot(xs, ys, label=METHOD_LABELS.get(method, method), linewidth=1.2)
        if any_line:
            ax_r.set_xscale("log")
            ax_r.set_yscale("log")
            ax_r.set_xlabel("wall seconds (raw)")
            ax_r.set_ylabel("native residual")
            ax_r.set_title(f"{case_id} residual vs raw wall")
            ax_r.grid(True, which="both", alpha=0.3)
            ax_r.legend(fontsize=7)
            fig_r.savefig(out_dir / f"residual_vs_wall_raw{'_accepted' if accepted_only else ''}.png", dpi=220)
            fig_r.savefig(out_dir / f"residual_vs_wall_raw{'_accepted' if accepted_only else ''}.pdf")
        plt.close(fig_r)

        # cumulative monotonic wall
        fig_c, ax_c = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
        any_line = False
        for method in methods:
            hist = _load_history(method)
            if not hist:
                continue
            xs = []
            ys = []
            for h in hist:
                if accepted_only and not h["accepted"]:
                    continue
                if not np.isfinite(h["wall_seconds"]):
                    continue
                xs.append(h["wall_seconds"])
                ys.append(h["residual"])
            if not xs:
                continue
            any_line = True
            ax_c.plot(xs, ys, label=METHOD_LABELS.get(method, method), linewidth=1.2)
        if any_line:
            ax_c.set_xscale("log")
            ax_c.set_yscale("log")
            ax_c.set_xlabel("wall seconds (cumulative)")
            ax_c.set_ylabel("native residual")
            ax_c.set_title(f"{case_id} residual vs wall")
            ax_c.grid(True, which="both", alpha=0.3)
            ax_c.legend(fontsize=7)
            fig_c.savefig(out_dir / f"residual_vs_wall_cumulative{'_accepted' if accepted_only else ''}.png", dpi=220)
            fig_c.savefig(out_dir / f"residual_vs_wall_cumulative{'_accepted' if accepted_only else ''}.pdf")
        plt.close(fig_c)


def plot_method_fields(case_id: str, case, methods_data: Dict[str, np.ndarray], ref: Tuple[np.ndarray, np.ndarray, np.ndarray], out_dir: Path):
    rho_ref, ux_ref, uy_ref = ref
    chi = getattr(case, "chi", np.ones((case.N, case.N), dtype=np.float64))

    for method, f in methods_data.items():
        rho, ux, uy = macro_of(case, f)
        speed = np.sqrt(ux * ux + uy * uy)
        speed_ref = np.sqrt(ux_ref * ux_ref + uy_ref * uy_ref)
        vort = vorticity_from_velocity(ux, uy)
        vort_ref = vorticity_from_velocity(ux_ref, uy_ref)
        label = METHOD_LABELS.get(method, method)

        fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
        im = ax.imshow(np.ma.array(speed, mask=(chi <= 0)), origin="lower", cmap="viridis")
        ax.set_title(f"{case_id} {label}: speed")
        fig.colorbar(im, ax=ax, label="|u|")
        fig.savefig(out_dir / f"{case_id}__{method}__speed.png", dpi=220)
        fig.savefig(out_dir / f"{case_id}__{method}__speed.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
        im = ax.imshow(np.ma.array(vort, mask=(chi <= 0)), origin="lower", cmap="coolwarm")
        ax.set_title(f"{case_id} {label}: vorticity")
        fig.colorbar(im, ax=ax, label="vorticity")
        fig.savefig(out_dir / f"{case_id}__{method}__vorticity.png", dpi=220)
        fig.savefig(out_dir / f"{case_id}__{method}__vorticity.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
        im = ax.imshow(np.ma.array(speed - speed_ref, mask=(chi <= 0)), origin="lower", cmap="bwr")
        ax.set_title(f"{case_id} {label}: speed error")
        fig.colorbar(im, ax=ax, label="speed_err")
        fig.savefig(out_dir / f"{case_id}__{method}__speed_error.png", dpi=220)
        fig.savefig(out_dir / f"{case_id}__{method}__speed_error.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
        im = ax.imshow(np.ma.array(vort - vort_ref, mask=(chi <= 0)), origin="lower", cmap="bwr")
        ax.set_title(f"{case_id} {label}: vorticity error")
        fig.colorbar(im, ax=ax, label="vorticity_err")
        fig.savefig(out_dir / f"{case_id}__{method}__vorticity_error.png", dpi=220)
        fig.savefig(out_dir / f"{case_id}__{method}__vorticity_error.pdf")
        plt.close(fig)

    # centerline profile comparison (all methods)
    j_mid = int(case.N // 2)
    i_mid = int(case.N // 2)
    x = np.arange(case.N, dtype=np.float64)
    y = np.arange(case.N, dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), constrained_layout=True)
    for method, f in methods_data.items():
        _, ux, uy = macro_of(case, f)
        axes[0].plot(x, ux[j_mid, :], label=METHOD_LABELS.get(method, method), lw=1.1)
        axes[1].plot(y, uy[:, i_mid], label=METHOD_LABELS.get(method, method), lw=1.1)

    axes[0].plot(x, ux_ref[j_mid, :], "k--", lw=1.7, label="reference ux")
    axes[1].plot(y, uy_ref[:, i_mid], "k--", lw=1.7, label="reference uy")
    axes[0].set_title(f"{case_id}: centerline profiles")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("ux")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("uy")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)

    fig.savefig(out_dir / f"{case_id}__centerline_ux_uy_methods.png", dpi=220)
    fig.savefig(out_dir / f"{case_id}__centerline_ux_uy_methods.pdf")
    plt.close(fig)


def write_accuracy_summary(case_id: str, acc_rows: Iterable[Tuple[str, float, float, float, float, float]], ref_name: str, out_dir: Path):
    out = out_dir / f"{case_id}__accuracy_summary.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow([
            "case_id",
            "method",
            "label",
            "reference",
            "rel_l2",
            "linf",
            "rms",
            "speed_rms",
            "speed_l2",
        ])
        for m, rel_l2, linf, rms, speed_rms, speed_l2 in acc_rows:
            wr.writerow([case_id, m, METHOD_LABELS.get(m, m), ref_name, rel_l2, linf, rms, speed_rms, speed_l2])


def plot_accuracy_bars(case_id: str, acc_rows: Iterable[Tuple[str, float, float, float, float, float]], out_dir: Path):
    acc_rows = list(acc_rows)
    if not acc_rows:
        return

    labels = [METHOD_LABELS.get(m, m) for m, *_ in acc_rows]
    rel_l2 = [v[1] for v in acc_rows]

    fig, ax = plt.subplots(figsize=(7.0, 3.9), constrained_layout=True)
    x = np.arange(len(labels), dtype=np.int64)
    ax.bar(x, rel_l2, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18)
    ax.set_yscale("log")
    ax.set_ylabel("relative L2")
    ax.set_title(f"{case_id}: relative L2 vs reference")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_dir / f"{case_id}__accuracy_bar_relL2.png", dpi=220)
    fig.savefig(out_dir / f"{case_id}__accuracy_bar_relL2.pdf")
    plt.close(fig)


def export_case(case_id: str, force_refresh: bool = False):
    base_id, level = parse_case_id(case_id)
    rows = read_case_rows(case_id)
    # keep method ordering deterministic as defined above if present
    methods = [r["method"] for r in rows]
    methods = sorted(dict.fromkeys(methods).keys())

    dst = DST_ROOT / case_id
    if force_refresh and dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    dst_hist = dst / "histories"
    dst_fields = dst / "fields"
    dst_figure = dst / "figure"
    dst_vtk = dst / "vtk"
    for p in (dst_hist, dst_fields, dst_figure, dst_vtk):
        p.mkdir(parents=True, exist_ok=True)

    # copy full dataset-level artifacts for reproducibility
    shutil.copy2(SOURCE_DIR / "metrics.json", dst / "metrics.json")
    shutil.copy2(SOURCE_DIR / "summary.json", dst / "summary.json")
    shutil.copy2(SOURCE_DIR / "summary.csv", dst / "summary_all_methods.csv")

    # write case-only summary
    with (dst / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for row in rows:
            wr.writerow(row)

    # copy history and vtk files from source
    for row in rows:
        method = row["method"]
        sh_src = SOURCE_HIST / f"{case_id}__{method}.csv"
        if sh_src.exists():
            shutil.copy2(sh_src, dst_hist / sh_src.name)

        vt_src = SOURCE_VTK / f"{case_id}__{method}.vtk"
        if vt_src.exists():
            shutil.copy2(vt_src, dst_vtk / vt_src.name)

    # build case
    _, _, _, factory = case_factory_scaled(base_id, level)
    case = factory()

    # load all method fields
    methods_data: Dict[str, np.ndarray] = {}
    for method in methods:
        cached = _load_cache_flexible(case_id, method)
        if cached is None:
            print(f"[warn] missing cache for {method}")
            continue
        methods_data[method] = cached[0]

    if "picard_lbm" not in methods_data:
        raise RuntimeError(f"missing picard cache for {case_id}")

    # reference selection: channel analytic requested first
    if base_id == "channel_n32":
        ref_name = "analytic"
        ref = analytic_channel_profile(case)
    else:
        ref_name = "picard"
        ref = macro_of(case, methods_data["picard_lbm"])

    # write fields and collect accuracy
    acc_rows = []
    for method, f in methods_data.items():
        write_field_csvs(case_id, method, case, f, ref, dst_fields)
        rel_l2, linf, rms, speed_rms, speed_l2 = compute_accuracy(ref, case, f)
        acc_rows.append((method, rel_l2, linf, rms, speed_rms, speed_l2))

    # method profiles + plots
    plot_residual_histories(case_id, methods_data.keys(), rows, dst_figure)
    plot_method_fields(case_id, case, methods_data, ref, dst_figure)
    write_accuracy_summary(case_id, acc_rows, ref_name, dst)
    plot_accuracy_bars(case_id, acc_rows, dst_figure)

    manifest = {
        "case_id": case_id,
        "base_case_id": base_id,
        "level": int(level),
        "methods": methods,
        "reference": ref_name,
        "source_dir": str(SOURCE_DIR),
        "output_dir": str(dst),
        "threads": {
            "numba": 24,
            "omp": 24,
            "openblas": 1,
            "mkl": 1,
        },
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (dst / "reproducibility_config.json").write_text(json.dumps({
        "case_id": case_id,
        "source_root": str(SOURCE_DIR),
        "method_order": methods,
        "reference": ref_name,
        "threads": {
            "NUMBA_NUM_THREADS": 24,
            "OMP_NUM_THREADS": 24,
            "OPENBLAS_NUM_THREADS": 1,
            "MKL_NUM_THREADS": 1,
        },
        "export_script": "export_channel_case_to_papers_data.py",
    }, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case-id", default="channel_n32__3x", help="case id, e.g., channel_n32__3x")
    p.add_argument("--force-refresh", action="store_true", help="delete destination folder first")
    args = p.parse_args()
    export_case(args.case_id, force_refresh=args.force_refresh)


if __name__ == "__main__":
    main()
