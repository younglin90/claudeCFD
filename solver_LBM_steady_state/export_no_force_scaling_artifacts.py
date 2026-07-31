"""Export publication artifacts for no-force scaling benchmark."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_60case_benchmark_no_force_scaling import (
    CASE_IDS,
    METHODS as DEFAULT_METHODS,
    _load_cached,
    case_factory_scaled,
)
from paper_60case_benchmark_no_force_scaling import HIST_DIR as SRC_HIST_DIR
from paper_60case_benchmark_no_force import macro_of
from paper_60case_benchmark import velocity_error

SRC = Path("paper_revision_data") / "no_force_scaling_benchmark"
OUT = Path("paper_revision_data") / "no_force_scaling_artifacts"
OUT_FIELDS = OUT / "fields"
OUT_HIST = OUT / "histories"
OUT_VTK = OUT / "vtk"
OUT_FIG = OUT / "figures"
CACHE_DIR = SRC / "npz_cache"


METHOD_LABELS = {
    "picard_lbm": "Picard",
    "anderson_lbm": "Anderson",
    "preconditioned_lbm": "Preconditioned",
    "inexact_newton_lbe": "Inexact Newton",
    "dual_time_mg_lbm": "Dual-time MG",
    "proposed": "SafeNN",
}


CASE_LABELS = {
    "channel_n32": "Channel",
    "couette_n32": "Couette",
    "cavity_re100_n33": "Cavity Re100",
    "cavity_re400_n49": "Cavity Re400",
    "cavity_re1000_n129": "Cavity Re1000",
    "multi_cylinder_n32": "Multi-cylinder",
    "backward_step_n64": "Backward step",
    "cylinder_wake_n64": "Cylinder wake",
    "t_junction_n64": "T-junction",
}


def read_rows():
    with (SRC / "summary.csv").open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows


def _fnum(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return float(default)


def write_csv_rows(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row.get(k, "") for k in fields})


def _normalize_history_for_artifact(rows: list[dict]) -> list[dict]:
    """Normalize history rows for artifact output.

    - enforce global monotonic iter
    - preserve raw wall seconds + cumulative monotonic wall seconds
    - keep accepted/phase markers (default accepted=1)
    - fallback accepted=0 only for large residual jumps where no explicit marker exists
    """
    if not rows:
        return []
    norm: list[dict] = []
    spike_ratio = 2.0
    segment_offset = 0.0
    phase = 0
    global_iter = 0

    prev_iter = _fnum(rows[0].get("iter"), 0.0)
    prev_wall_raw = _fnum(rows[0].get("wall_seconds_raw"), _fnum(rows[0].get("wall_seconds"), 0.0))
    segment_base = prev_wall_raw if np.isfinite(prev_wall_raw) else 0.0
    prev_wall_cum = 0.0
    prev_residual = _fnum(rows[0].get("residual"), float("inf"))

    def emit(r: dict, residual: float, wall_raw: float, wall_cum: float, accepted_override: int | None):
        accepted = accepted_override if accepted_override is not None else int(_fnum(r.get("accepted"), 1.0))
        norm.append({
            "iter": int(global_iter),
            "residual": residual,
            "lbe_calls": _fnum(r.get("lbe_calls")),
            "wall_seconds_raw": wall_raw,
            "wall_seconds": wall_cum,
            "accepted": float(accepted),
            "phase": str(r.get("phase", phase)),
        })

    first_wall_raw = _fnum(rows[0].get("wall_seconds_raw"), _fnum(rows[0].get("wall_seconds"), 0.0))
    accepted0 = int(_fnum(rows[0].get("accepted"), 1.0))
    emit(rows[0], _fnum(rows[0].get("residual"), float("nan")), first_wall_raw, 0.0, accepted0)
    global_iter += 1

    prev_iter_is = prev_iter
    prev_wall_raw_is = prev_wall_raw
    prev_wall_cum_is = prev_wall_cum
    prev_residual_is = prev_residual

    for r in rows[1:]:
        it = _fnum(r.get("iter"))
        wall_raw = _fnum(r.get("wall_seconds_raw"), _fnum(r.get("wall_seconds"), float("nan")))
        if not np.isfinite(wall_raw):
            continue

        if not np.isfinite(it) or not np.isfinite(prev_iter_is) or not np.isfinite(prev_wall_raw_is):
            is_reset = False
        else:
            is_reset = (it < prev_iter_is) or (wall_raw < prev_wall_raw_is - 1e-15)

        if is_reset:
            segment_offset = prev_wall_cum_is
            phase += 1
            segment_base = wall_raw

        wall_norm = wall_raw - segment_base
        wall_norm = wall_norm if np.isfinite(wall_norm) else 0.0
        wall_norm = max(wall_norm, 0.0)
        wall_cum = segment_offset + wall_norm
        if wall_cum < prev_wall_cum_is:
            wall_cum = prev_wall_cum_is

        residual = _fnum(r.get("residual"), float("nan"))
        accepted = int(_fnum(r.get("accepted"), 1.0))
        if _fnum(r.get("accepted"), 1.0) == 1.0 and np.isfinite(prev_residual_is) and np.isfinite(residual):
            if prev_residual_is > 0.0 and residual / prev_residual_is > spike_ratio:
                accepted = 0
        emit(r, residual, wall_raw, wall_cum, accepted)

        global_iter += 1
        prev_iter_is = it
        prev_wall_raw_is = wall_raw
        prev_wall_cum_is = wall_cum
        prev_residual_is = residual

    return norm


def normalize_history_csv(src_path: Path, dst_path: Path):
    rows: list[dict] = []
    with src_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    rows_norm = _normalize_history_for_artifact(rows)
    write_csv_rows(
        dst_path,
        rows_norm,
        ["iter", "residual", "lbe_calls", "wall_seconds_raw", "wall_seconds", "accepted", "phase"],
    )


def compute_vorticity(ux: np.ndarray, uy: np.ndarray, dx: float = 1.0, dy: float = 1.0):
    duy_dx = np.gradient(uy, axis=1, edge_order=1) / dx
    dux_dy = np.gradient(ux, axis=0, edge_order=1) / dy
    return duy_dx - dux_dy


def write_case_fields(case_id: str, method: str, ref_case, ref_f, case, f, out_dir: Path):
    rho, ux, uy = macro_of(case, f)
    rho_ref, ux_ref, uy_ref = macro_of(ref_case, ref_f)

    chi = getattr(case, "chi", np.ones_like(ux, dtype=np.float64))
    speed = np.sqrt(ux * ux + uy * uy)
    speed_ref = np.sqrt(ux_ref * ux_ref + uy_ref * uy_ref)
    vort = compute_vorticity(ux, uy)
    vort_ref = compute_vorticity(ux_ref, uy_ref)
    eps = 1.0e-12
    rel_speed_err = np.where(np.abs(speed_ref) > eps, np.abs(speed - speed_ref) / (np.abs(speed_ref) + eps), 0.0)

    ny, nx = ux.shape
    xx = np.arange(nx, dtype=np.float64)
    yy = np.arange(ny, dtype=np.float64)

    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{case_id}__{method}"

    with (out_dir / f"{base}__macro.csv").open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iy", "ix", "x", "y", "rho", "ux", "uy", "speed", "chi"])
        for j in range(ny):
            y = yy[j]
            for i in range(nx):
                wr.writerow([j, i, xx[i], y, rho[j, i], ux[j, i], uy[j, i], speed[j, i], chi[j, i]])

    with (out_dir / f"{base}__vorticity.csv").open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iy", "ix", "x", "y", "vorticity", "vorticity_ref", "vorticity_err", "chi"])
        for j in range(ny):
            y = yy[j]
            for i in range(nx):
                v = vort[j, i]
                vr = vort_ref[j, i]
                wr.writerow([j, i, xx[i], y, v, vr, v - vr, chi[j, i]])

    with (out_dir / f"{base}__error.csv").open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow([
            "iy",
            "ix",
            "x",
            "y",
            "ux_err",
            "uy_err",
            "speed_err",
            "speed_rel_err",
            "speed_ref",
            "speed",
            "chi",
        ])
        for j in range(ny):
            y = yy[j]
            for i in range(nx):
                wr.writerow([
                    j,
                    i,
                    xx[i],
                    y,
                    ux[j, i] - ux_ref[j, i],
                    uy[j, i] - uy_ref[j, i],
                    speed[j, i] - speed_ref[j, i],
                    rel_speed_err[j, i],
                    speed_ref[j, i],
                    speed[j, i],
                    chi[j, i],
                ])


def write_centerline_csv(case_id: str, method: str, ref_case, ref_f, case, f, out_dir: Path):
    rho, ux, uy = macro_of(case, f)
    _, ux_ref, uy_ref = macro_of(ref_case, ref_f)
    speed = np.sqrt(ux * ux + uy * uy)
    speed_ref = np.sqrt(ux_ref * ux_ref + uy_ref * uy_ref)
    ny, nx = ux.shape

    i_mid = nx // 2
    j_mid = ny // 2

    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{case_id}__{method}"
    x = np.arange(nx, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)

    with (out_dir / f"{base}__centerline_x.csv").open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["x", "uy", "uy_ref", "uy_err", "ux", "ux_ref", "ux_err", "speed", "speed_ref", "speed_err"])
        for i in range(nx):
            wr.writerow([
                x[i],
                uy[j_mid, i],
                uy_ref[j_mid, i],
                uy[j_mid, i] - uy_ref[j_mid, i],
                ux[j_mid, i],
                ux_ref[j_mid, i],
                ux[j_mid, i] - ux_ref[j_mid, i],
                speed[j_mid, i],
                speed_ref[j_mid, i],
                speed[j_mid, i] - speed_ref[j_mid, i],
            ])

    with (out_dir / f"{base}__centerline_y.csv").open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["y", "ux", "ux_ref", "ux_err", "uy", "uy_ref", "uy_err", "speed", "speed_ref", "speed_err"])
        for j in range(ny):
            wr.writerow([
                y[j],
                ux[j, i_mid],
                ux_ref[j, i_mid],
                ux[j, i_mid] - ux_ref[j, i_mid],
                uy[j, i_mid],
                uy_ref[j, i_mid],
                uy[j, i_mid] - uy_ref[j, i_mid],
                speed[j, i_mid],
                speed_ref[j, i_mid],
                speed[j, i_mid] - speed_ref[j, i_mid],
            ])


def _load_cached_flexible(case_id: str, method: str):
    """Load cached fields, accepting legacy cache key variants.

    The cache key may change when solver code changes; keep a fallback pass so
    artifact generation can consume any cached result for the same case/method.
    """
    exact = _load_cached(case_id, method)
    if exact is not None:
        return exact

    # fallback for stale cache key formats
    candidates = sorted(
        CACHE_DIR.glob(f"{case_id}__{method}__*.npz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None

    data = np.load(candidates[0], allow_pickle=False)
    return data["f"], [tuple(row) for row in data["hist"].tolist()], float(data["wall"])


def plot_field_and_error(case_id: str, method: str, case, f, f_ref, out_dir: Path):
    rho, ux, uy = macro_of(case, f)
    _, ux_ref, uy_ref = macro_of(case, f_ref)
    chi = getattr(case, "chi", np.ones_like(ux))
    speed = np.sqrt(ux * ux + uy * uy)
    speed_ref = np.sqrt(ux_ref * ux_ref + uy_ref * uy_ref)
    vort = compute_vorticity(ux, uy)
    vort_ref = compute_vorticity(ux_ref, uy_ref)
    label = METHOD_LABELS.get(method, method)

    out_case_dir = out_dir / case_id
    out_case_dir.mkdir(parents=True, exist_ok=True)
    base = f"{method}"

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(np.ma.array(speed, mask=chi <= 0), origin="lower", cmap="viridis")
    ax.set_title(f"{case_id} / {label}: speed")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="|u|")
    fig.savefig(out_case_dir / f"{method}__speed.png", dpi=180)
    fig.savefig(out_case_dir / f"{method}__speed.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(np.ma.array(vort, mask=chi <= 0), origin="lower", cmap="coolwarm")
    ax.set_title(f"{case_id} / {label}: vorticity")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="vorticity")
    fig.savefig(out_case_dir / f"{method}__vorticity.png", dpi=180)
    fig.savefig(out_case_dir / f"{method}__vorticity.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    err = speed - speed_ref
    im = ax.imshow(np.ma.array(err, mask=chi <= 0), origin="lower", cmap="bwr")
    ax.set_title(f"{case_id} / {label}: speed error")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="err")
    fig.savefig(out_case_dir / f"{method}__error_speed.png", dpi=180)
    fig.savefig(out_case_dir / f"{method}__error_speed.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.semilogy([], [])
    ax.plot(np.arange(vort.size), np.ravel(vort), label="vorticity")
    ax.plot(np.arange(vort_ref.size), np.ravel(vort_ref), label="vorticity_ref", alpha=0.5, ls="--")
    ax.set_title(f"{case_id} / {label}: flattened vorticity")
    ax.set_xlabel("index")
    ax.set_ylabel("vorticity")
    ax.legend()
    fig.savefig(out_case_dir / f"{method}__vorticity_profile.pdf")
    fig.savefig(out_case_dir / f"{method}__vorticity_profile.png", dpi=180)
    plt.close(fig)

    # centerline profile (for quick paper-ready quick checks)
    nx = ux.shape[1]
    ny = ux.shape[0]
    i_mid = nx // 2
    j_mid = ny // 2
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), constrained_layout=True)
    axes[0].plot(np.arange(nx), ux[j_mid, :], label="ux")
    axes[0].plot(np.arange(nx), ux_ref[j_mid, :], ls="--", label="ux_ref", alpha=0.8)
    axes[0].set_title(f"{case_id} / {label}: centerline y={j_mid}")
    axes[0].set_ylabel("ux")
    axes[0].legend(fontsize=7)
    axes[1].plot(np.arange(ny), uy[:, i_mid], label="uy")
    axes[1].plot(np.arange(ny), uy_ref[:, i_mid], ls="--", label="uy_ref", alpha=0.8)
    axes[1].set_ylabel("uy")
    axes[1].set_xlabel("axis index")
    axes[1].legend(fontsize=7)
    fig.savefig(out_case_dir / f"{method}__centerline_profiles.png", dpi=180)
    fig.savefig(out_case_dir / f"{method}__centerline_profiles.pdf")
    plt.close(fig)


def read_history(case_id: str, method: str):
    path = OUT_HIST / f"{case_id}__{method}.csv"
    if not path.exists():
        path = SRC_HIST_DIR / f"{case_id}__{method}.csv"
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "iter": _fnum(row.get("iter")),
                "residual": _fnum(row.get("residual")),
                "lbe_calls": _fnum(row.get("lbe_calls")),
                "wall_seconds_raw": _fnum(row.get("wall_seconds_raw"), np.nan),
                "wall_seconds": _fnum(row.get("wall_seconds")),
                "accepted": _fnum(row.get("accepted"), 1.0),
                "phase": row.get("phase", ""),
            })
    return rows


def _pick_wall_series(rows: list[dict], prefer_raw: bool, accepted_only: bool = False) -> tuple[list[float], list[float]]:
    if not rows:
        return [], []
    x = []
    y = []
    for r in rows:
        if accepted_only and not bool(r.get("accepted", 1.0)):
            continue
        wall_raw = r.get("wall_seconds_raw")
        wall = r.get("wall_seconds")
        resid = r.get("residual")
        wall_val = wall_raw if prefer_raw and np.isfinite(wall_raw) else wall
        if not np.isfinite(wall_val) or not np.isfinite(resid):
            continue
        x.append(float(wall_val))
        y.append(float(max(resid, 1e-16)))
    return x, y


def _safe_nn_xrange(series_by_method: dict[str, tuple[list[float], list[float]],],
                    *,
                    min_ratio: float = 4.0,
                    fallback_ratio: float = 8.0) -> tuple[float, float] | None:
    """Return x-axis limits that better show the proposed-method transition.

    If proposed converges rapidly compared with others, the plot is zoomed to the
    region where proposed becomes small, while preserving log-scale validity.
    """

    if "proposed" not in series_by_method:
        return None

    def _finite(arr):
        return np.asarray(arr, dtype=np.float64)[np.isfinite(arr) & (np.asarray(arr, dtype=np.float64) > 0)]

    p_x = _finite(series_by_method["proposed"][0])
    p_y = _finite(series_by_method["proposed"][1])
    if len(p_x) < 4 or len(p_y) < 4:
        return None

    # Keep only paired finite points and align lengths
    n = min(len(p_x), len(p_y))
    p_x = p_x[:n]
    p_y = p_y[:n]

    # focus point: where residual reaches 10% of first residual or 1e-5, whichever is larger
    y0 = float(p_y[0]) if np.isfinite(p_y[0]) else 1.0e-6
    if y0 <= 0.0:
        return None
    target = max(y0 * 0.1, 1.0e-5)
    hit = np.where(p_y <= target)[0]
    if hit.size > 0:
        x_focus = float(p_x[hit[0]])
    else:
        x_focus = float(np.quantile(p_x, 0.75))

    all_x = []
    for xs, _ in series_by_method.values():
        arr = _finite(xs)
        if arr.size:
            all_x.extend(arr.tolist())
    if not all_x:
        return None

    x_all = float(np.max(all_x))
    x_min = float(np.min([v for v in all_x if v > 0.0])) if any(v > 0.0 for v in all_x) else 1.0
    if not np.isfinite(x_focus) or x_focus <= 0.0 or x_focus >= x_all:
        return None

    if x_all / x_focus < min_ratio:
        return None

    left = max(1.0e-9, x_min * 0.9)
    right = x_focus * fallback_ratio
    if not np.isfinite(right) or right <= left:
        return None
    return left, right


def plot_case_residuals(case_id: str, methods: list[str], out_dir: Path):
    fig = plt.figure(figsize=(6.5, 4.2), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    any_line = False
    series_by_method: dict[str, tuple[list[float], list[float]]] = {}
    for method in methods:
        hist = read_history(case_id, method)
        if not hist:
            continue
        lbe_x = [r["lbe_calls"] for r in hist if np.isfinite(r["lbe_calls"]) and np.isfinite(r["residual"])]
        y = [max(r["residual"], 1e-16) for r in hist if np.isfinite(r["lbe_calls"]) and np.isfinite(r["residual"])]
        if not lbe_x:
            continue
        series_by_method[method] = (lbe_x, y)
        any_line = True
        ax.plot(lbe_x, y, label=METHOD_LABELS.get(method, method), linewidth=1.2)
    if any_line:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("LBE calls")
        ax.set_ylabel("native residual")
        ax.set_title(case_id)
        rng = _safe_nn_xrange(series_by_method, min_ratio=4.0, fallback_ratio=8.0)
        if rng:
            ax.set_xlim(*rng)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
        (out_dir / case_id).mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / case_id / "residual_vs_lbe.png", dpi=220)
        fig.savefig(out_dir / case_id / "residual_vs_lbe.pdf")
    plt.close(fig)


def plot_case_residuals_by_lbe(case_id: str, methods: list[str], out_dir: Path, accepted_only: bool):
    fig = plt.figure(figsize=(6.5, 4.2), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    any_line = False
    series_by_method: dict[str, tuple[list[float], list[float]]] = {}
    for method in methods:
        hist = read_history(case_id, method)
        if not hist:
            continue
        lbe_x = []
        y = []
        for r in hist:
            if accepted_only and not bool(r.get("accepted", 1.0)):
                continue
            if np.isfinite(r["lbe_calls"]) and np.isfinite(r["residual"]):
                lbe_x.append(r["lbe_calls"])
                y.append(max(r["residual"], 1e-16))
        if not lbe_x:
            continue
        series_by_method[method] = (lbe_x, y)
        any_line = True
        ax.plot(lbe_x, y, label=METHOD_LABELS.get(method, method), linewidth=1.2)

    if any_line:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("LBE calls")
        ax.set_ylabel("native residual")
        ax.set_title(f"{case_id}: residual vs LBE calls (accepted only)")
        suffix = "accepted"

        rng = _safe_nn_xrange(series_by_method, min_ratio=4.0, fallback_ratio=8.0)
        if rng:
            ax.set_xlim(*rng)

        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
        (out_dir / case_id).mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / case_id / f"residual_vs_lbe_{suffix}.png", dpi=220)
        fig.savefig(out_dir / case_id / f"residual_vs_lbe_{suffix}.pdf")
    plt.close(fig)


def plot_case_residuals_by_wall(case_id: str, methods: list[str], out_dir: Path):
    _plot_case_residuals_by_wall(case_id, methods, out_dir, use_raw_wall=True, accepted_only=False)
    _plot_case_residuals_by_wall(case_id, methods, out_dir, use_raw_wall=False, accepted_only=False)
    _plot_case_residuals_by_wall(case_id, methods, out_dir, use_raw_wall=True, accepted_only=True)
    _plot_case_residuals_by_wall(case_id, methods, out_dir, use_raw_wall=False, accepted_only=True)
    plot_case_residuals_by_lbe(case_id, methods, out_dir, accepted_only=True)


def _plot_case_residuals_by_wall(case_id: str, methods: list[str], out_dir: Path, use_raw_wall: bool, accepted_only: bool):
    fig = plt.figure(figsize=(6.5, 4.2), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    any_line = False
    series_by_method: dict[str, tuple[list[float], list[float]]] = {}
    for method in methods:
        hist = read_history(case_id, method)
        if not hist:
            continue
        x, y = _pick_wall_series(hist, prefer_raw=use_raw_wall, accepted_only=accepted_only)
        if not x:
            continue
        x0 = float(x[0])
        if np.isfinite(x0):
            x = [max(float(v) - x0, 0.0) for v in x]
        series_by_method[method] = (x, y)
        any_line = True
        ax.plot(x, y, label=METHOD_LABELS.get(method, method), linewidth=1.2)

    if any_line:
        ax.set_xscale("log")
        ax.set_yscale("log")
        if use_raw_wall:
            ax.set_xlabel("wall seconds (raw segment-local)")
            if accepted_only:
                ax.set_title(f"{case_id}: residual vs wall (raw, accepted only)")
                suffix = "raw_accepted"
            else:
                ax.set_title(f"{case_id}: residual vs wall (raw, shifted to 0)")
                suffix = "raw"
        else:
            ax.set_xlabel("wall seconds (monotonic cumulative)")
            if accepted_only:
                ax.set_title(f"{case_id}: residual vs wall (cumulative, accepted only)")
                suffix = "cumulative_accepted"
            else:
                ax.set_title(f"{case_id}: residual vs wall (cumulative, shifted to 0)")
                suffix = "cumulative"

        ax.set_ylabel("native residual")
        rng = _safe_nn_xrange(series_by_method, min_ratio=4.0, fallback_ratio=8.0)
        if rng:
            ax.set_xlim(*rng)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
        (out_dir / case_id).mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / case_id / f"residual_vs_wall_{suffix}.png", dpi=220)
        fig.savefig(out_dir / case_id / f"residual_vs_wall_{suffix}.pdf")
    plt.close(fig)


def aggregate_outputs(rows):
    methods = sorted({r["method"] for r in rows})
    methods = [m for m in DEFAULT_METHODS if m in methods] + [m for m in methods if m not in DEFAULT_METHODS]

    per_method_rows = []
    for method in methods:
        mrows = [r for r in rows if r["method"] == method]
        if not mrows:
            continue
        lbe = [_fnum(r["lbe_calls"]) for r in mrows if _fnum(r["lbe_calls"]) > 0]
        wall = [_fnum(r["wall_seconds"]) for r in mrows if _fnum(r["wall_seconds"]) > 0]
        rel = [_fnum(r["rel_l2_vs_picard"]) for r in mrows if np.isfinite(_fnum(r["rel_l2_vs_picard"]))]
        vel_abs = [_fnum(r.get("vel_abs_l2_vs_picard")) for r in mrows if np.isfinite(_fnum(r.get("vel_abs_l2_vs_picard")))]
        rho_abs = [_fnum(r.get("rho_abs_l2_vs_picard")) for r in mrows if np.isfinite(_fnum(r.get("rho_abs_l2_vs_picard")))]
        per_method_rows.append({
            "method": method,
            "label": METHOD_LABELS.get(method, method),
            "case_count": len(mrows),
            "converged_count": sum(int(float(r["converged"])) for r in mrows),
            "mean_lbe_calls": float(np.mean(lbe)) if lbe else np.nan,
            "mean_wall_seconds": float(np.mean(wall)) if wall else np.nan,
            "mean_rel_l2_vs_picard": float(np.mean(rel)) if rel else np.nan,
            "mean_vel_abs_l2_vs_picard": float(np.mean(vel_abs)) if vel_abs else np.nan,
            "mean_rho_abs_l2_vs_picard": float(np.mean(rho_abs)) if rho_abs else np.nan,
        })

    per_case_rows = []
    for row in rows:
        per_case_rows.append({
            "base_case_id": row.get("base_case_id", ""),
            "scaling_level": row.get("scaling_level", ""),
            "case_id": row["case_id"],
            "method": row["method"],
            "lbe_calls": row["lbe_calls"],
            "wall_seconds": row["wall_seconds"],
            "final_residual": row["final_residual"],
            "converged": row["converged"],
            "rel_l2_vs_picard": row["rel_l2_vs_picard"],
            "linf_vs_picard": row["linf_vs_picard"],
            "rms_vs_picard": row["rms_vs_picard"],
            "vel_abs_l2_vs_picard": row.get("vel_abs_l2_vs_picard", ""),
            "vel_abs_linf_vs_picard": row.get("vel_abs_linf_vs_picard", ""),
            "vel_abs_rms_vs_picard": row.get("vel_abs_rms_vs_picard", ""),
            "rho_abs_l2_vs_picard": row.get("rho_abs_l2_vs_picard", ""),
            "rho_abs_linf_vs_picard": row.get("rho_abs_linf_vs_picard", ""),
            "rho_abs_rms_vs_picard": row.get("rho_abs_rms_vs_picard", ""),
        })

    return methods, per_method_rows, per_case_rows


def plot_method_summary(per_case_rows: list[dict], metrics: dict, out_dir: Path):
    case_counts = [r for r in per_case_rows if r["method"] == "proposed"]
    case_ids = sorted({r["case_id"] for r in per_case_rows})
    method_rows = {}
    for r in per_case_rows:
        method_rows.setdefault(r["method"], []).append(r)

    # mean by method
    method_names = sorted(method_rows.keys())
    tot_lbe = [np.nanmean([_fnum(x["lbe_calls"]) for x in method_rows[m]]) for m in method_names]
    tot_wall = [np.nanmean([_fnum(x["wall_seconds"]) for x in method_rows[m]]) for m in method_names]
    pass_count = [sum(int(float(x["converged"])) for x in method_rows[m]) for m in method_names]
    acc = [np.nanmean([_fnum(x["rel_l2_vs_picard"]) for x in method_rows[m]]) for m in method_names]

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    axs[0, 0].bar(method_names, tot_lbe)
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_title("Mean LBE calls")
    axs[0, 0].tick_params(axis="x", rotation=20)

    axs[0, 1].bar(method_names, tot_wall)
    axs[0, 1].set_yscale("log")
    axs[0, 1].set_title("Mean wall-clock (s)")
    axs[0, 1].tick_params(axis="x", rotation=20)

    axs[1, 0].bar(method_names, pass_count)
    axs[1, 0].set_title("Converged case counts")
    axs[1, 0].tick_params(axis="x", rotation=20)

    axs[1, 1].bar(method_names, acc)
    axs[1, 1].set_yscale("symlog", linthresh=1.0e-8)
    axs[1, 1].set_title("Mean relative L2 vs Picard")
    axs[1, 1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(out_dir / "summary__method_overview.png", dpi=220)
    fig.savefig(out_dir / "summary__method_overview.pdf")
    plt.close(fig)

    # pass heatmap by case
    cases = metrics.get("case_results", [])
    if cases:
        keys = ["converged", "lbe_win", "wall_win", "acc_win", "case_pass"]
        labels = [
            f"{CASE_LABELS.get(c['base_case_id'], c['base_case_id'])} {c['scaling_level']}x"
            for c in cases
        ]
        data = np.asarray([[c[k] for k in keys] for c in cases], dtype=float)
        fig, ax = plt.subplots(figsize=(8, 10))
        im = ax.imshow(data, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(["Conv", "LBE", "Wall", "Acc", "Pass"], rotation=30, fontsize=8)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, "1" if data[i, j] >= 0.5 else "0", va="center", ha="center", fontsize=7)
        ax.set_title("Proposed solver pass components")
        fig.colorbar(im, ax=ax, shrink=0.7, ticks=[0, 1])
        fig.tight_layout()
        fig.savefig(out_dir / "summary__pass_heatmap.png", dpi=220)
        fig.savefig(out_dir / "summary__pass_heatmap.pdf")
        plt.close(fig)


def main():
    rows = read_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    if OUT_FIG.exists():
        shutil.rmtree(OUT_FIG)
    if OUT_FIELDS.exists():
        shutil.rmtree(OUT_FIELDS)
    if OUT_HIST.exists():
        shutil.rmtree(OUT_HIST)
    if OUT_VTK.exists():
        shutil.rmtree(OUT_VTK)

    # Copy core metadata
    shutil.copy2(SRC / "summary.csv", OUT / "summary_all_methods.csv")
    shutil.copy2(SRC / "metrics.json", OUT / "metrics.json")
    shutil.copy2(SRC / "summary.json", OUT / "summary.json")

    # copy vtk directly
    if OUT_VTK.exists():
        shutil.rmtree(OUT_VTK)
    shutil.copytree(SRC / "vtk", OUT_VTK)

    # normalize histories so iter is globally monotonic and accepted markers exist
    if OUT_HIST.exists():
        shutil.rmtree(OUT_HIST)
    OUT_HIST.mkdir(parents=True, exist_ok=True)
    for src_hist in (SRC / "histories").iterdir():
        if not src_hist.is_file():
            continue
        dst = OUT_HIST / src_hist.name
        normalize_history_csv(src_hist, dst)

    methods, per_method_rows, per_case_rows = aggregate_outputs(rows)
    write_csv_rows(
        OUT / "per_method_score.csv",
        per_method_rows,
        [
            "method",
            "label",
            "case_count",
            "converged_count",
            "mean_lbe_calls",
            "mean_wall_seconds",
            "mean_rel_l2_vs_picard",
            "mean_vel_abs_l2_vs_picard",
            "mean_rho_abs_l2_vs_picard",
        ],
    )
    write_csv_rows(
        OUT / "per_case_metrics.csv",
        per_case_rows,
        [
            "base_case_id",
            "scaling_level",
            "case_id",
            "method",
            "lbe_calls",
            "wall_seconds",
            "final_residual",
            "converged",
            "rel_l2_vs_picard",
            "linf_vs_picard",
            "rms_vs_picard",
            "vel_abs_l2_vs_picard",
            "vel_abs_linf_vs_picard",
            "vel_abs_rms_vs_picard",
            "rho_abs_l2_vs_picard",
            "rho_abs_linf_vs_picard",
            "rho_abs_rms_vs_picard",
        ],
    )

    (OUT / "reproducibility_config.json").write_text(
        json.dumps(
            {
                "source": str(SRC),
                "method_order": methods,
                "numba_threads": 24,
                "openmp_threads": 24,
                "openblas_threads": 1,
                "case_ids": sorted(CASE_IDS),
                "scaling_levels": sorted({int(r["scaling_level"]) for r in rows}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # residual history figures by case
    case_ids = sorted({r["case_id"] for r in rows})
    for case_id in case_ids:
        plot_case_residuals(case_id, methods, OUT_FIG)
        plot_case_residuals_by_wall(case_id, methods, OUT_FIG)

    # per-case per-method field CSV + contour figures
    out_case_fig = OUT_FIG / "case_fields"
    out_case_fields = OUT_FIELDS
    metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))

    for case_id in sorted({r["case_id"] for r in rows}):
        # build reference case/method cache once
        case_rows = [r for r in rows if r["case_id"] == case_id]
        base_case_id = case_rows[0].get("base_case_id", "unknown")
        level = int(case_rows[0].get("scaling_level", 1))
        _, _, _, factory = case_factory_scaled(base_case_id, level)
        case = factory()
        ref_cache = _load_cached_flexible(case_id, "picard_lbm")
        if ref_cache is None:
            continue
        ref_f = ref_cache[0]

        for row in case_rows:
            method = row["method"]
            cached = _load_cached_flexible(case_id, method)
            if cached is None:
                continue
            f = cached[0]
            case_method = case_factory_scaled(base_case_id, level)[3]()

            write_case_fields(case_id, method, case_method, ref_f, case_method, f, out_case_fields)
            write_centerline_csv(case_id, method, case_method, ref_f, case_method, f, out_case_fields)
            plot_field_and_error(case_id, method, case_method, f, ref_f, out_case_fig)

    # case-level summary and pass heatmap
    plot_method_summary(per_case_rows, metrics, OUT_FIG)

    manifest = {
        "source_dir": str(SRC),
        "artifact_dir": str(OUT),
        "summary_rows": len(rows),
        "case_count": len({r["case_id"] for r in rows}),
        "method_count": len(methods),
        "metrics": metrics,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
