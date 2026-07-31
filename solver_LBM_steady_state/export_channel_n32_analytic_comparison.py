"""Generate analytic-reference comparison artifacts for channel_n32__1x.

This script compares all solved methods in the no-force 1x channel case against a
laminar Poiseuille-style analytical profile and writes CSV + contour/graph outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lbm_periodic import CX, CY
from paper_60case_benchmark_no_force_scaling import case_factory_scaled, _load_cached, HIST_DIR


OUT_ROOT = Path("paper_revision_data") / "no_force_scaling_artifacts"
CASE_ID = "channel_n32"
LEVEL = 1
CASE_KEY = f"{CASE_ID}__{LEVEL}x"
METHODS = [
    "picard_lbm",
    "anderson_lbm",
    "preconditioned_lbm",
    "inexact_newton_lbe",
    "dual_time_mg_lbm",
    "proposed",
]
METHOD_LABELS = {
    "picard_lbm": "Picard",
    "anderson_lbm": "Anderson",
    "preconditioned_lbm": "Preconditioned",
    "inexact_newton_lbe": "Inexact Newton",
    "dual_time_mg_lbm": "Dual-time MG",
    "proposed": "SafeNN",
}


def _macro(case, f):
    rho = f.sum(axis=0)
    rho_safe = np.where(rho < 1.0e-12, 1.0, rho)
    ux = (f * CX[:, None, None]).sum(axis=0) / rho_safe
    uy = (f * CY[:, None, None]).sum(axis=0) / rho_safe
    return rho, ux, uy


def _analytic_channel(case):
    N = int(case.N)
    u_in = float(getattr(case, "U_in", 0.05))
    L = float(N - 1)
    y = np.arange(N, dtype=np.float64)
    ubar = u_in * (N - 1.0) / N
    ux = 6.0 * ubar * (y / L) * (1.0 - y / L)
    ux = ux[:, None] * np.ones((N, N), dtype=np.float64)
    uy = np.zeros_like(ux)
    rho = np.ones((N, N), dtype=np.float64)
    return rho, ux, uy


def _vorticity(ux: np.ndarray, uy: np.ndarray):
    duy_dx = np.gradient(uy, axis=1, edge_order=1)
    dux_dy = np.gradient(ux, axis=0, edge_order=1)
    return duy_dx - dux_dy


def _load_cache_flexible(case_id: str, method: str, method_cache_file: Path):
    cached = _load_cached(case_id, method)
    if cached is not None:
        return cached[0]

    # Fallback: latest available cache entry.
    candidates = sorted(
        method_cache_file.glob(f"{case_id}__{method}__*.npz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    data = np.load(candidates[0], allow_pickle=False)
    return data["f"]


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    _, _, _, factory = case_factory_scaled(CASE_ID, LEVEL)
    case = factory()
    cache_dir = HIST_DIR.parent / "npz_cache"
    f_ref = _load_cache_flexible(CASE_KEY, "picard_lbm", cache_dir)
    if f_ref is None:
        raise RuntimeError(f"missing reference cache for {CASE_KEY} picard_lbm")

    # Build analytical reference
    rho_a, ux_a, uy_a = _analytic_channel(case)
    speed_a = np.sqrt(ux_a * ux_a + uy_a * uy_a)
    vort_a = _vorticity(ux_a, uy_a)
    mask = getattr(case, "chi", np.ones_like(ux_a, dtype=np.float64)) > 0.0

    # Directories
    out_fig_dir = OUT_ROOT / "figures" / CASE_KEY / "analytic_compare"
    out_csv_dir = OUT_ROOT / "fields" / "analytic_channel_n32__1x"
    _ensure_dir(out_fig_dir)
    _ensure_dir(out_csv_dir)

    # Write analytic fields once
    with (out_csv_dir / "analytical_field.csv").open("w", encoding="utf-8") as fh:
        fh.write("iy,ix,y,x,rho,ux,uy,speed,vort\n")
        for iy in range(case.N):
            for ix in range(case.N):
                fh.write(
                    f"{iy},{ix},{iy:.8f},{ix:.8f},"
                    f"{rho_a[iy,ix]:.10e},{ux_a[iy,ix]:.10e},{uy_a[iy,ix]:.10e},"
                    f"{speed_a[iy,ix]:.10e},{vort_a[iy,ix]:.10e}\n"
                )

    rows = []
    center_y = np.arange(case.N, dtype=np.float64)
    method_fields = []
    for m in METHODS:
        f = _load_cache_flexible(CASE_KEY, m, cache_dir)
        if f is None:
            continue

        rho, ux, uy = _macro(case, f)
        speed = np.sqrt(ux * ux + uy * uy)
        vort = _vorticity(ux, uy)

        du = ux[mask] - ux_a[mask]
        dv = uy[mask] - uy_a[mask]
        d_speed = speed[mask] - speed_a[mask]
        den = float(np.sqrt(np.sum(ux_a[mask] ** 2 + uy_a[mask] ** 2)))
        rel_l2 = float(np.sqrt(np.sum(du * du + dv * dv)) / max(den, 1.0e-30))
        linf = float(max(np.max(np.abs(du)) if du.size else 0.0, np.max(np.abs(dv)) if dv.size else 0.0))
        rms = float(np.sqrt(np.mean(du * du + dv * dv)) if du.size else 0.0)
        rows.append((m, rel_l2, linf, rms))
        method_fields.append((m, f, rho, ux, uy, speed, vort))

        # Field CSV + error CSV (excluding masked obstacles)
        with (out_csv_dir / f"{CASE_KEY}__{m}__analytic.csv").open("w", encoding="utf-8") as fh:
            fh.write(
                "iy,ix,x,y,rho,ux,uy,speed,vort,ux_ana,uy_ana,"
                "speed_ana,vort_ana,ux_err,uy_err,speed_err,vort_err\n"
            )
            for iy in range(case.N):
                for ix in range(case.N):
                    if mask[iy, ix]:
                        uxe = ux[iy, ix] - ux_a[iy, ix]
                        uye = uy[iy, ix] - uy_a[iy, ix]
                        se = speed[iy, ix] - speed_a[iy, ix]
                        ve = vort[iy, ix] - vort_a[iy, ix]
                    else:
                        uxe = np.nan
                        uye = np.nan
                        se = np.nan
                        ve = np.nan
                    fh.write(
                        f"{iy},{ix},{ix:.8f},{iy:.8f},"
                        f"{rho[iy,ix]:.10e},{ux[iy,ix]:.10e},{uy[iy,ix]:.10e},"
                        f"{speed[iy,ix]:.10e},{vort[iy,ix]:.10e},"
                        f"{ux_a[iy,ix]:.10e},{uy_a[iy,ix]:.10e},"
                        f"{speed_a[iy,ix]:.10e},{vort_a[iy,ix]:.10e},"
                        f"{uxe:.10e},{uye:.10e},{se:.10e},{ve:.10e}\n"
                    )

        # Overwrite error CSV with explicit loop to keep row-per-cell error
        with (out_csv_dir / f"{CASE_KEY}__{m}__analytic_error.csv").open("w", encoding="utf-8") as fh:
            fh.write("iy,ix,x,y,ux_err,uy_err,speed_err,vort_err,mask\n")
            for iy in range(case.N):
                for ix in range(case.N):
                    if mask[iy, ix]:
                        uxe = ux[iy, ix] - ux_a[iy, ix]
                        uye = uy[iy, ix] - uy_a[iy, ix]
                        se = speed[iy, ix] - speed_a[iy, ix]
                        ve = vort[iy, ix] - vort_a[iy, ix]
                    else:
                        uxe = np.nan
                        uye = np.nan
                        se = np.nan
                        ve = np.nan
                    fh.write(
                        f"{iy},{ix},{ix:.8f},{iy:.8f},"
                        f"{uxe:.10e},{uye:.10e},{se:.10e},{ve:.10e},{int(mask[iy,ix])}\n"
                    )

        # Field contour figures
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        im = ax.imshow(np.ma.array(speed, mask=~mask), origin="lower", cmap="viridis")
        ax.set_title(f"{METHOD_LABELS.get(m, m)} speed")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, label="|u|")
        fig.savefig(out_fig_dir / f"speed__{m}.png", dpi=180)
        fig.savefig(out_fig_dir / f"speed__{m}.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        im = ax.imshow(np.ma.array(speed - speed_a, mask=~mask), origin="lower", cmap="bwr")
        ax.set_title(f"{METHOD_LABELS.get(m, m)} speed error vs analytic")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, label="speed error")
        fig.savefig(out_fig_dir / f"speed_error__{m}.png", dpi=180)
        fig.savefig(out_fig_dir / f"speed_error__{m}.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        im = ax.imshow(np.ma.array(vort, mask=~mask), origin="lower", cmap="coolwarm")
        ax.set_title(f"{METHOD_LABELS.get(m, m)} vorticity")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, label="vorticity")
        fig.savefig(out_fig_dir / f"vorticity__{m}.png", dpi=180)
        fig.savefig(out_fig_dir / f"vorticity__{m}.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        im = ax.imshow(np.ma.array(vort - vort_a, mask=~mask), origin="lower", cmap="bwr")
        ax.set_title(f"{METHOD_LABELS.get(m, m)} vorticity error vs analytic")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, label="vorticity error")
        fig.savefig(out_fig_dir / f"vorticity_error__{m}.png", dpi=180)
        fig.savefig(out_fig_dir / f"vorticity_error__{m}.pdf")
        plt.close(fig)

    # summary CSV
    with (out_csv_dir / f"{CASE_KEY}__analytic_summary.csv").open("w", encoding="utf-8") as fh:
        fh.write("method,rel_L2_vs_analytic,linf_vs_analytic,rms_vs_analytic\n")
        for m, rel_l2, linf, rms in rows:
            fh.write(f"{m},{rel_l2:.15e},{linf:.15e},{rms:.15e}\n")

    # Comparison profile plots
    with (out_csv_dir / f"{CASE_KEY}__analytic_summary.csv").open("r", encoding="utf-8") as fh:
        pass

    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    # Centerline reference: channel은 y에 따른 x-평균 ux 프로파일.
    # 축 방향이 반대로 들어가서 실제 해석값이 뒤바뀐 그래프로 보이던 문제를 수정.
    ax.plot(center_y, ux_a[:, 0], "k--", lw=2.0, label="analytic")
    for m, _rl2, _, _ in rows:
        f_pair = next((x for x in method_fields if x[0] == m), None)
        if f_pair is None:
            continue
        _, _, ux_m, _, _, _, _ = f_pair
        row_profile = []
        for iy in range(case.N):
            row = ux_m[iy, :]
            row_mask = mask[iy, :]
            if np.any(row_mask):
                row_profile.append(float(np.mean(row[row_mask])))
            else:
                row_profile.append(float("nan"))
        ax.plot(center_y, row_profile, lw=1.2, label=METHOD_LABELS.get(m, m))
    ax.set_title("Channel N=32, 1x: x-averaged ux vs y")
    ax.set_xlabel("y index")
    ax.set_ylabel("x-averaged ux")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_fig_dir / "profile_xmean_ux_vs_analytic.png", dpi=220)
    fig.savefig(out_fig_dir / "profile_xmean_ux_vs_analytic.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    labels = [METHOD_LABELS.get(m, m) for m, *_ in rows]
    x = np.arange(len(labels))
    rel = [r[1] for r in rows]
    ax.bar(x, rel, width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Relative L2 vs analytic")
    ax.set_title("Channel N=32, 1x: analytic error by method")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_fig_dir / "analytic_relL2_bar.png", dpi=220)
    fig.savefig(out_fig_dir / "analytic_relL2_bar.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
