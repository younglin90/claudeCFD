"""Generate field and convergence figures missing from the revision DOCX."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import numba

    numba.set_num_threads(32)
except Exception:
    pass

from paper_extra_benchmarks import make_case, solve_baseline_generic
from paper_remaining_calculations import solve_safe_nn_stats


OUT = Path("paper_revision_data/figures")
DATA_OUT = Path("paper_revision_data/field_figure_runs.json")


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    print(path)


def speed(case, f):
    _, ux, uy = case.macro(f)
    return np.sqrt(ux * ux + uy * uy) * case.chi


def run_field_case(name):
    label, case_b, forcing = make_case(name)
    _, case_s, _ = make_case(name)
    print(f"[field] {label}", flush=True)
    f_b, h_b = solve_baseline_generic(case_b, max_steps=70000, tol=1e-7, check_every=500, verbose=False)
    f_s, h_s, stats = solve_safe_nn_stats(
        case_s,
        max_outer=220,
        tol=1e-7,
        kinetic_substeps=15,
        eps_accept=0.12,
        line_search=True,
        line_search_max=4,
        verbose=False,
    )
    sb = speed(case_b, f_b)
    ss = speed(case_s, f_s)
    diff = np.abs(ss - sb) * case_b.chi
    return {
        "name": name,
        "label": label,
        "forcing": forcing,
        "case": case_b,
        "baseline_history": h_b,
        "safe_history": h_s,
        "stats": stats,
        "speed_baseline": sb,
        "speed_safe": ss,
        "speed_diff": diff,
        "baseline_residual": float(h_b[-1][1]),
        "safe_residual": float(h_s[-1][1]),
        "baseline_lbe": int(h_b[-1][2]),
        "safe_lbe": int(h_s[-1][2]),
    }


def plot_fields(run):
    label = run["label"].replace(" N=64", "")
    fields = [run["speed_baseline"], run["speed_safe"], run["speed_diff"]]
    titles = ["Picard |u|", "Safe-NN |u|", "|Safe - Picard|"]
    vmax = max(float(fields[0].max()), float(fields[1].max()), 1e-16)
    diffmax = max(float(fields[2].max()), 1e-16)
    path = OUT / f"fig_field_{run['name']}.png"
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for ax, arr, title in zip(axes[:2], fields[:2], titles[:2]):
        im = ax.imshow(arr, origin="lower", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    im = axes[2].imshow(fields[2], origin="lower", cmap="magma", vmin=0.0, vmax=diffmax)
    axes[2].set_title(titles[2])
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03)
    fig.suptitle(label)
    savefig(path)
    return path


def plot_convergence(runs):
    path = OUT / "fig_extra_benchmark_convergence.png"
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for run in runs:
        hb = np.array(run["baseline_history"], dtype=float)
        hs = np.array(run["safe_history"], dtype=float)
        short = run["label"].replace(" mask N=64", "").replace(" analogue N=64", "").replace(" N=64", "")
        ax.semilogy(hb[:, 2], hb[:, 1], "--", lw=1.6, label=f"{short} Picard")
        ax.semilogy(hs[:, 2], hs[:, 1], "-", lw=2.0, label=f"{short} Safe-NN")
    ax.set_xlabel("LBE calls")
    ax.set_ylabel("Native residual norm")
    ax.set_title("Convergence histories for additional 2D benchmarks")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    savefig(path)
    return path


def plot_speedup_from_history(runs):
    path = OUT / "fig_extra_benchmark_lbe_to_tolerance.png"
    labels = [
        r["label"].replace(" mask N=64", "").replace(" analogue N=64", "").replace(" N=64", "")
        for r in runs
    ]
    base = [r["baseline_lbe"] for r in runs]
    safe = [r["safe_lbe"] for r in runs]
    x = np.arange(len(runs))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.bar(x - width / 2, base, width, label="Picard LBE", color="#777777")
    ax.bar(x + width / 2, safe, width, label="Safe-NN LBE", color="#b84040")
    ax.set_yscale("log")
    ax.set_ylabel("LBE calls to tolerance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_title("LBE-call cost for additional 2D benchmarks")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(frameon=False)
    savefig(path)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [run_field_case(name) for name in ["backward_step", "cylinder_wake", "t_junction"]]
    field_paths = [str(plot_fields(run)) for run in runs]
    conv_path = str(plot_convergence(runs))
    lbe_path = str(plot_speedup_from_history(runs))
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "figures": field_paths + [conv_path, lbe_path],
        "runs": [
            {
                "name": r["name"],
                "label": r["label"],
                "baseline_lbe": r["baseline_lbe"],
                "safe_lbe": r["safe_lbe"],
                "baseline_residual": r["baseline_residual"],
                "safe_residual": r["safe_residual"],
                "max_speed_diff": float(r["speed_diff"].max()),
            }
            for r in runs
        ],
    }
    DATA_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(DATA_OUT)


if __name__ == "__main__":
    main()
