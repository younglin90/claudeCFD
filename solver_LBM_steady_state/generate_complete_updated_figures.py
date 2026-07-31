from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    import numba

    numba.set_num_threads(32)
except Exception:
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ghia_validation import extract_centerline, get_ghia_data
from lbm_optimized_2d import (
    OptimizedCavityCase,
    OptimizedChannelCase,
    OptimizedKolmogorovCase,
    solve_baseline_fast,
)
from paper_extra_benchmarks import make_case, solve_baseline_generic
from paper_remaining_calculations import ghia_metrics, solve_safe_nn_stats, velocity_rel_l2
from paper_revision_metrics import build_cases, macro_of, run_baseline
from solver_safe_nn import solve_safe_nn


OUT = Path("paper_revision_data/figures_complete")
JSON_OUT = Path("paper_revision_data/complete_updated_figure_runs.json")


def hist_to_json(hist):
    return [[float(a), float(b), int(c), float(d)] for a, b, c, d in hist]


def macro_speed(case, f):
    _, ux, uy = macro_of(case, f) if not hasattr(case, "macro") else case.macro(f)
    return np.sqrt(ux * ux + uy * uy)


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(path, flush=True)
    return path


def run_core_cases():
    runs = []
    fields = {}
    for spec in build_cases():
        print(f"[core] {spec['label']}", flush=True)
        case_b = spec["factory"]()
        case_s = spec["factory"]()
        baseline_spec = dict(spec)
        safe_kw = {}
        if spec["key"] == "cavity_Re400_N49":
            baseline_spec["tol"] = 5e-8
            baseline_spec["max_baseline"] = 200000
            safe_kw = {
                "tol": spec["tol"],
                "final_polish_tol": 5e-8,
                "final_polish_max_steps": 50000,
                "final_polish_check_every": 500,
            }
        else:
            safe_kw = {"tol": spec["tol"]}

        t0 = time.perf_counter()
        f_b, h_b = run_baseline(case_b, baseline_spec)
        wall_b = time.perf_counter() - t0
        t0 = time.perf_counter()
        f_s, h_s = solve_safe_nn(
            case_s,
            max_outer=220,
            krylov_max=10,
            krylov_tol=1e-3,
            kinetic_substeps=15,
            beta_max=0.7,
            eps_accept=0.10,
            verbose=False,
            **safe_kw,
        )
        wall_s = time.perf_counter() - t0
        runs.append(
            {
                "group": "core",
                "key": spec["key"],
                "label": spec["label"] + (" polished" if spec["key"] == "cavity_Re400_N49" else ""),
                "tol": float(baseline_spec["tol"] if spec["key"] == "cavity_Re400_N49" else spec["tol"]),
                "baseline_history": hist_to_json(h_b),
                "safe_history": hist_to_json(h_s),
                "baseline_lbe": int(h_b[-1][2]),
                "safe_lbe": int(h_s[-1][2]),
                "baseline_residual": float(h_b[-1][1]),
                "safe_residual": float(h_s[-1][1]),
                "baseline_wall": float(wall_b),
                "safe_wall": float(wall_s),
                "speedup_lbe": float(h_b[-1][2] / max(h_s[-1][2], 1)),
                "speedup_wall": float(wall_b / max(wall_s, 1e-12)),
            }
        )
        fields[spec["key"]] = (case_b, f_b, case_s, f_s)
    return runs, fields


def make_scaling_case(kind, n):
    nu = 0.05
    if kind == "kolmogorov":
        k_lat = 2.0 * np.pi / n
        f0 = 0.05 * nu * k_lat * k_lat
        return OptimizedKolmogorovCase(N=n, nu=nu, F0=f0, kf=1)
    if kind == "channel":
        f0 = 8.0 * nu * 0.05 / ((n - 1.0) ** 2)
        return OptimizedChannelCase(N=n, nu=nu, F0=f0)
    raise ValueError(kind)


def run_scaling_cases():
    specs = [
        ("Kolmogorov N=64", "kolmogorov", 64, 1e-7, 120000, 500),
        ("Channel N=64", "channel", 64, 1e-7, 160000, 500),
        ("Kolmogorov N=128", "kolmogorov", 128, 1e-9, 220000, 1000),
        ("Channel N=128", "channel", 128, 1e-9, 260000, 1000),
        ("Kolmogorov N=256", "kolmogorov", 256, 1e-9, 320000, 1000),
        ("Channel N=256", "channel", 256, 1e-9, 620000, 1000),
    ]
    runs = []
    fields = {}
    for label, kind, n, tol, max_steps, check_every in specs:
        print(f"[scaling] {label}", flush=True)
        case_b = make_scaling_case(kind, n)
        case_s = make_scaling_case(kind, n)
        t0 = time.perf_counter()
        f_b, h_b = solve_baseline_fast(
            case_b, max_steps=max_steps, tol=tol, check_every=check_every, verbose=True
        )
        wall_b = time.perf_counter() - t0
        t0 = time.perf_counter()
        f_s, h_s, stats = solve_safe_nn_stats(
            case_s,
            max_outer=300,
            tol=tol,
            kinetic_substeps=15,
            verbose=True,
        )
        wall_s = time.perf_counter() - t0
        key = f"{kind}_N{n}"
        runs.append(
            {
                "group": "scaling",
                "key": key,
                "label": label,
                "tol": tol,
                "baseline_history": hist_to_json(h_b),
                "safe_history": hist_to_json(h_s),
                "baseline_lbe": int(h_b[-1][2]),
                "safe_lbe": int(h_s[-1][2]),
                "baseline_residual": float(h_b[-1][1]),
                "safe_residual": float(h_s[-1][1]),
                "baseline_wall": float(wall_b),
                "safe_wall": float(wall_s),
                "safe_converged": bool(h_s[-1][1] < tol),
                "speedup_lbe": float(h_b[-1][2] / max(h_s[-1][2], 1)),
                "speedup_wall": float(wall_b / max(wall_s, 1e-12)),
                "safe_stats": stats,
            }
        )
        if n in (64,):
            fields[key] = (case_b, f_b, case_s, f_s)
    return runs, fields


def run_high_re_cavity_cases():
    specs = [
        ("Cavity Re=400 N=65 polished", 400, 65, 5e-8, 300000, 500, True),
        ("Cavity Re=1000 N=129", 1000, 129, 5e-7, 700000, 1000, False),
    ]
    runs = []
    fields = {}
    for label, re, n, report_tol, max_steps, check_every, polish in specs:
        print(f"[cavity] {label}", flush=True)
        case_b = OptimizedCavityCase(N=n, Re=re, U_wall=0.1)
        case_s = OptimizedCavityCase(N=n, Re=re, U_wall=0.1)
        t0 = time.perf_counter()
        f_b, h_b = solve_baseline_fast(
            case_b, max_steps=max_steps, tol=report_tol, check_every=check_every, verbose=True
        )
        wall_b = time.perf_counter() - t0
        t0 = time.perf_counter()
        f_s, h_s, stats = solve_safe_nn_stats(
            case_s,
            max_outer=420,
            tol=5e-7,
            kinetic_substeps=20 if re >= 1000 else 15,
            eps_accept=0.15 if re >= 1000 else 0.10,
            line_search=re >= 1000,
            line_search_max=4,
            final_polish_tol=report_tol if polish else None,
            final_polish_max_steps=60000 if polish else 0,
            final_polish_check_every=500,
            verbose=True,
        )
        wall_s = time.perf_counter() - t0
        key = f"cavity_Re{re}_N{n}"
        runs.append(
            {
                "group": "high_re_cavity",
                "key": key,
                "label": label,
                "tol": report_tol,
                "baseline_history": hist_to_json(h_b),
                "safe_history": hist_to_json(h_s),
                "baseline_lbe": int(h_b[-1][2]),
                "safe_lbe": int(h_s[-1][2]),
                "baseline_residual": float(h_b[-1][1]),
                "safe_residual": float(h_s[-1][1]),
                "baseline_wall": float(wall_b),
                "safe_wall": float(wall_s),
                "safe_converged": bool(h_s[-1][1] < report_tol if polish else h_s[-1][1] < 5e-7),
                "speedup_lbe": float(h_b[-1][2] / max(h_s[-1][2], 1)),
                "speedup_wall": float(wall_b / max(wall_s, 1e-12)),
                "rel_l2_vs_baseline": velocity_rel_l2(case_b, f_b, f_s),
                "safe_ghia": ghia_metrics(f_s, case_s, re),
                "safe_stats": stats,
            }
        )
        fields[key] = (case_b, f_b, case_s, f_s)
    return runs, fields


def run_extra_cases():
    runs = []
    fields = {}
    for name in ["backward_step", "cylinder_wake", "t_junction"]:
        label, case_b, _ = make_case(name)
        _, case_s, _ = make_case(name)
        print(f"[extra] {label}", flush=True)
        t0 = time.perf_counter()
        f_b, h_b = solve_baseline_generic(case_b, max_steps=70000, tol=1e-7, check_every=500, verbose=True)
        wall_b = time.perf_counter() - t0
        t0 = time.perf_counter()
        f_s, h_s, stats = solve_safe_nn_stats(
            case_s,
            max_outer=220,
            tol=1e-7,
            kinetic_substeps=15,
            eps_accept=0.12,
            line_search=True,
            line_search_max=4,
            verbose=True,
        )
        wall_s = time.perf_counter() - t0
        runs.append(
            {
                "group": "extra_mask",
                "key": name,
                "label": label,
                "tol": 1e-7,
                "baseline_history": hist_to_json(h_b),
                "safe_history": hist_to_json(h_s),
                "baseline_lbe": int(h_b[-1][2]),
                "safe_lbe": int(h_s[-1][2]),
                "baseline_residual": float(h_b[-1][1]),
                "safe_residual": float(h_s[-1][1]),
                "baseline_wall": float(wall_b),
                "safe_wall": float(wall_s),
                "safe_converged": bool(h_s[-1][1] < 1e-7),
                "speedup_lbe": float(h_b[-1][2] / max(h_s[-1][2], 1)),
                "speedup_wall": float(wall_b / max(wall_s, 1e-12)),
                "safe_stats": stats,
            }
        )
        fields[name] = (case_b, f_b, case_s, f_s)
    return runs, fields


def plot_history_grid(runs, path, title, ncols=3):
    n = len(runs)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), squeeze=False)
    for ax, run in zip(axes.ravel(), runs):
        hb = np.array(run["baseline_history"], dtype=float)
        hs = np.array(run["safe_history"], dtype=float)
        ax.semilogy(hb[:, 2], hb[:, 1], "--", color="0.45", lw=1.8, label="Picard")
        ax.semilogy(hs[:, 2], hs[:, 1], "-", color="#b84040", lw=2.0, label="Safe-NN")
        ax.axhline(run["tol"], color="0.2", ls=":", lw=1.0)
        ax.set_title(run["label"], fontsize=10)
        ax.set_xlabel("LBE calls")
        ax.set_ylabel("residual")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title, y=1.01, fontsize=13)
    return savefig(path)


def plot_speedup_status(runs, path):
    labels = [r["label"].replace(" ", "\n", 2) for r in runs]
    lbe = [max(r["speedup_lbe"], 1e-6) for r in runs]
    wall = [max(r["speedup_wall"], 1e-6) for r in runs]
    x = np.arange(len(runs))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(12, len(runs) * 0.75), 5.4))
    ax.bar(x - width / 2, lbe, width, label="LBE-call speedup", color="#2f6f9f")
    ax.bar(x + width / 2, wall, width, label="wall-clock speedup", color="#c95f3b")
    ax.axhline(1.0, color="0.25", lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_ylabel("Speedup over Picard")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("All 2D validation cases: LBE-call and wall-clock speedups")
    return savefig(path)


def plot_safeguard_diagnostics(runs):
    rows = [r for r in runs if r.get("safe_stats")]
    labels = [r["label"].replace(" ", "\n", 2) for r in rows]
    evals = [r["safe_stats"].get("lookahead_evaluations", 0) for r in rows]
    rejected = [r["safe_stats"].get("lookahead_rejections", 0) for r in rows]
    restarts = [r["safe_stats"].get("residual_increase_restarts", 0) for r in rows]
    linesearch = [r["safe_stats"].get("line_search_rejections", 0) for r in rows]
    polish = [r["safe_stats"].get("final_polish_steps", 0) for r in rows]
    x = np.arange(len(rows))
    width = 0.16
    fig, ax = plt.subplots(figsize=(max(12, len(rows) * 0.75), 5.4))
    ax.bar(x - 2 * width, evals, width, label="lookahead eval", color="#6b8fb3")
    ax.bar(x - width, rejected, width, label="lookahead rejected", color="#b84040")
    ax.bar(x, restarts, width, label="residual restarts", color="#8a6fb0")
    ax.bar(x + width, linesearch, width, label="line-search rejected", color="#c95f3b")
    ax.bar(x + 2 * width, polish, width, label="final-polish steps", color="#2d8a57")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("Count (symlog; polish shown in LBE steps)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    ax.set_title("Safeguard, fallback, line-search, and final-polish activity")
    return savefig(OUT / "figU12_safeguard_diagnostics_updated.png")


def plot_residual_error_scatter(runs):
    rows = [r for r in runs if "rel_l2_vs_baseline" in r]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for r in rows:
        ax.scatter(r["safe_residual"], r["rel_l2_vs_baseline"], s=60)
        ax.annotate(r["label"].replace(" Cavity", "\nCavity"), (r["safe_residual"], r["rel_l2_vs_baseline"]), fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Final Safe-NN residual")
    ax.set_ylabel("Relative L2 velocity difference vs Picard")
    ax.set_title("Residual tolerance vs field discrepancy for limitation cases")
    ax.grid(True, which="both", alpha=0.25)
    return savefig(OUT / "figS1_residual_error_scatter_updated.png")


def plot_mass_conservation(core_runs, core_fields):
    labels = []
    drift_b = []
    drift_s = []
    for run in core_runs:
        key = run["key"]
        case_b, f_b, case_s, f_s = core_fields[key]
        initial_mass = float(np.sum(macro_of(case_b, case_b.initial_field())[0]))
        labels.append(run["label"].replace(" ", "\n", 2))
        drift_b.append(abs(float(np.sum(macro_of(case_b, f_b)[0])) - initial_mass) / max(abs(initial_mass), 1e-30))
        drift_s.append(abs(float(np.sum(macro_of(case_s, f_s)[0])) - initial_mass) / max(abs(initial_mass), 1e-30))
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(x - width / 2, drift_b, width, label="Picard", color="0.55")
    ax.bar(x + width / 2, drift_s, width, label="Safe-NN", color="#b84040")
    ax.set_yscale("log")
    ax.set_ylabel("Relative mass drift")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("Mass conservation across core validation cases")
    return savefig(OUT / "figS2_mass_conservation_updated.png")


def plot_final_polish_diagnostic():
    diag_path = Path("paper_revision_data/cavity_oscillation_diagnostic.json")
    if not diag_path.exists():
        return None
    diag = json.loads(diag_path.read_text(encoding="utf-8"))
    before = diag["safe_before_post_relax"]
    after = diag["safe_after_picard_post_relax"]
    targets = after["targets_reached"]
    steps = [0]
    residuals = [before["safe_residual"]]
    rel = [before["rel_l2_vs_tight_baseline"]]
    for key in ["5e-07", "1e-07", "5e-08"]:
        if key in targets:
            steps.append(targets[key]["steps"])
            residuals.append(targets[key]["residual"])
            if key == "5e-08":
                rel.append(after["rel_l2_vs_tight_baseline"])
            else:
                rel.append(np.nan)
    fig, ax1 = plt.subplots(figsize=(6.8, 4.6))
    ax1.plot(steps, residuals, "o-", color="#2f6f9f", label="residual")
    ax1.set_yscale("log")
    ax1.set_xlabel("Picard final-polish steps")
    ax1.set_ylabel("Residual")
    ax1.grid(True, which="both", alpha=0.25)
    ax2 = ax1.twinx()
    valid_steps = [s for s, v in zip(steps, rel) if np.isfinite(v)]
    valid_rel = [v for v in rel if np.isfinite(v)]
    ax2.plot(valid_steps, valid_rel, "s--", color="#b84040", label="rel L2 vs tight baseline")
    ax2.set_yscale("log")
    ax2.set_ylabel("Relative L2")
    ax1.set_title("Cavity Re=400 final-polish removes residual oscillatory modes")
    lines, labs = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labs + labs2, frameon=False)
    return savefig(OUT / "figS3_cavity_final_polish_diagnostic_updated.png")


def plot_core_profiles(fields):
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    for ax, key, title in [
        (axes[0], "kolmogorov_N32", "Kolmogorov N=32"),
        (axes[1], "channel_N32", "Channel N=32"),
        (axes[2], "couette_N32", "Couette N=32"),
    ]:
        case_b, f_b, case_s, f_s = fields[key]
        _, ux_b, _ = macro_of(case_b, f_b)
        _, ux_s, _ = macro_of(case_s, f_s)
        ref = case_b.analytical_ux()
        y = np.arange(case_b.N)
        ax.plot(ref[:, case_b.N // 2], y, "k-", lw=1.8, label="analytical")
        ax.plot(ux_b[:, case_b.N // 2], y, "o", ms=3, color="0.45", label="Picard")
        ax.plot(ux_s[:, case_b.N // 2], y, "s", ms=3, color="#b84040", label="Safe-NN")
        ax.set_title(title)
        ax.set_xlabel("u_x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    return savefig(OUT / "figU2_core_profiles_updated.png")


def plot_cavity_centerlines(fields):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.4))
    for row, key, re, title in [
        (0, "cavity_Re100_N33", 100, "Cavity Re=100 N=33"),
        (1, "cavity_Re400_N49", 400, "Cavity Re=400 N=49 polished"),
    ]:
        case_b, f_b, case_s, f_s = fields[key]
        y_g, u_g, x_g, v_g = get_ghia_data(re)
        y_b, u_b, x_b, v_b = extract_centerline(f_b, case_b, case_b.U_wall)
        y_s, u_s, x_s, v_s = extract_centerline(f_s, case_s, case_s.U_wall)
        axes[row, 0].plot(u_g, y_g, "ko", ms=4, label="Ghia")
        axes[row, 0].plot(u_b, y_b, "--", color="0.45", label="Picard")
        axes[row, 0].plot(u_s, y_s, "-", color="#b84040", label="Safe-NN")
        axes[row, 0].set_title(title + ": u vertical")
        axes[row, 0].set_xlabel("u/U_lid")
        axes[row, 0].set_ylabel("y")
        axes[row, 1].plot(x_g, v_g, "ko", ms=4, label="Ghia")
        axes[row, 1].plot(x_b, v_b, "--", color="0.45", label="Picard")
        axes[row, 1].plot(x_s, v_s, "-", color="#b84040", label="Safe-NN")
        axes[row, 1].set_title(title + ": v horizontal")
        axes[row, 1].set_xlabel("x")
        axes[row, 1].set_ylabel("v/U_lid")
    axes[0, 0].legend(frameon=False, fontsize=8)
    return savefig(OUT / "figU3_cavity_centerlines_updated.png")


def plot_field_triplet(fields, key, title, path):
    case_b, f_b, case_s, f_s = fields[key]
    sb = macro_speed(case_b, f_b)
    ss = macro_speed(case_s, f_s)
    diff = np.abs(ss - sb)
    vmax = max(float(sb.max()), float(ss.max()), 1e-16)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, arr, label in [(axes[0], sb, "Picard |u|"), (axes[1], ss, "Safe-NN |u|")]:
        im = ax.imshow(arr, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    im = axes[2].imshow(diff, origin="lower", cmap="magma", vmin=0, vmax=max(float(diff.max()), 1e-16))
    axes[2].set_title("|Safe - Picard|")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03)
    fig.suptitle(title)
    return savefig(path)


def plot_scaling_speedups(runs):
    return plot_speedup_status(runs, OUT / "figU6_scaling_speedups_updated.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_runs = []
    all_fields = {}

    core_runs, core_fields = run_core_cases()
    all_runs += core_runs
    all_fields.update(core_fields)
    plot_history_grid(core_runs, OUT / "figU1_core6_convergence_updated.png", "Core 6 validation convergence histories")
    plot_core_profiles(core_fields)
    plot_cavity_centerlines(core_fields)
    plot_field_triplet(core_fields, "multi_cylinder_N32", "Multi-cylinder N=32 updated field", OUT / "figU4_multicylinder_field_updated.png")
    plot_mass_conservation(core_runs, core_fields)

    scaling_runs, scaling_fields = run_scaling_cases()
    all_runs += scaling_runs
    all_fields.update(scaling_fields)
    plot_history_grid(scaling_runs, OUT / "figU5_scaling_convergence_updated.png", "Grid-scaling convergence histories", ncols=3)
    plot_scaling_speedups(scaling_runs)
    for key, title in [
        ("kolmogorov_N64", "Kolmogorov N=64 updated field"),
        ("channel_N64", "Channel N=64 updated field"),
    ]:
        plot_field_triplet(scaling_fields, key, title, OUT / f"figU_{key}_field_updated.png")

    cavity_runs, cavity_fields = run_high_re_cavity_cases()
    all_runs += cavity_runs
    all_fields.update(cavity_fields)
    plot_history_grid(cavity_runs, OUT / "figU7_high_re_cavity_convergence_updated.png", "High-Re cavity convergence histories", ncols=2)
    plot_field_triplet(cavity_fields, "cavity_Re400_N65", "Cavity Re=400 N=65 polished field", OUT / "figU8_cavity_re400_n65_polished_field_updated.png")

    extra_runs, extra_fields = run_extra_cases()
    all_runs += extra_runs
    all_fields.update(extra_fields)
    plot_history_grid(extra_runs, OUT / "figU9_extra_mask_convergence_updated.png", "Additional mask-flow convergence histories", ncols=3)
    for key, title in [
        ("backward_step", "Backward-facing step updated field"),
        ("cylinder_wake", "Cylinder wake analogue updated field"),
        ("t_junction", "T-junction updated field"),
    ]:
        plot_field_triplet(extra_fields, key, title, OUT / f"figU10_{key}_field_updated.png")

    plot_speedup_status(all_runs, OUT / "figU11_all_2d_cases_speedup_status_updated.png")
    plot_safeguard_diagnostics(all_runs)
    plot_residual_error_scatter(all_runs)
    plot_final_polish_diagnostic()
    JSON_OUT.write_text(json.dumps({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": all_runs}, indent=2), encoding="utf-8")
    print(JSON_OUT, flush=True)


if __name__ == "__main__":
    main()
