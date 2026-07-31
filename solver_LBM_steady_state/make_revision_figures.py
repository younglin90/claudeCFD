"""Generate figure assets for the Safe-NN LBM revision DOCX."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_extra_benchmarks import backward_step_mask, cylinder_wake_mask, t_junction_mask


OUT = Path("paper_revision_data/figures")
REMAINING = Path("paper_revision_data/remaining_2d_calculations.json")
EXTRA = Path("paper_revision_data/extra_2d_benchmarks.json")


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    print(path)


def plot_scaling(remaining: dict) -> Path:
    rows = remaining["scaling"]
    labels = [r["label"].replace(" Uamp=0.05", "").replace(" Umax=0.05", "") for r in rows]
    lbe = [r["safe_speedup_lbe"] for r in rows]
    wall = [r["safe_speedup_wall"] for r in rows]
    conv = [r["safe_converged"] for r in rows]
    x = np.arange(len(rows))
    width = 0.38

    path = OUT / "fig_revision_n_scaling_speedups.png"
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bars1 = ax.bar(x - width / 2, lbe, width, label="LBE-call speedup", color="#2f6f9f")
    bars2 = ax.bar(x + width / 2, wall, width, label="Wall-clock speedup", color="#c95f3b")
    ax.axhline(1.0, color="0.25", lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_ylabel("Speedup over Picard-LBM (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_title("N-scaling additions at tightened residual tolerance")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(frameon=False)
    for i, (b, ok) in enumerate(zip(bars1, conv)):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() * 1.08,
            "conv" if ok else "plateau",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )
    savefig(path)
    return path


def plot_cavity(remaining: dict) -> Path:
    rows = remaining["cavity"]
    labels = [r["label"].replace("Cavity ", "") for r in rows]
    lbe = [r["safe_speedup_lbe"] for r in rows]
    wall = [r["safe_speedup_wall"] for r in rows]
    ghia = [r["safe_ghia"]["centerline_max"] for r in rows]
    x = np.arange(len(rows))
    width = 0.35

    path = OUT / "fig_revision_cavity_stiff_summary.png"
    fig, ax1 = plt.subplots(figsize=(7.4, 4.5))
    ax1.bar(x - width / 2, lbe, width, label="LBE-call speedup", color="#2f6f9f")
    ax1.bar(x + width / 2, wall, width, label="Wall-clock speedup", color="#c95f3b")
    ax1.axhline(1.0, color="0.25", lw=1, ls="--")
    ax1.set_yscale("log")
    ax1.set_ylabel("Speedup over Picard-LBM (log scale)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", which="both", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, ghia, "o-", color="#2d8a57", lw=2, label="Ghia max deviation")
    ax2.set_ylabel("Ghia centerline max deviation")
    ax1.set_title("High-Re cavity: LBE-call gain vs wall-clock cost")
    lines, labs = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labs + labs2, frameon=False, loc="upper left")
    savefig(path)
    return path


def plot_extra_benchmarks(extra: dict) -> Path:
    rows = extra["benchmarks"]
    labels = [
        r["label"].replace(" mask N=64", "").replace(" analogue N=64", "").replace(" N=64", "")
        for r in rows
    ]
    lbe = [r["safe_speedup_lbe"] for r in rows]
    wall = [r["safe_speedup_wall"] for r in rows]
    rel = [r["velocity_metrics"]["rel_l2"] for r in rows]
    x = np.arange(len(rows))
    width = 0.35

    path = OUT / "fig_revision_extra_benchmarks.png"
    fig, ax1 = plt.subplots(figsize=(8.2, 4.6))
    ax1.bar(x - width / 2, lbe, width, label="LBE-call speedup", color="#2f6f9f")
    ax1.bar(x + width / 2, wall, width, label="Wall-clock speedup", color="#c95f3b")
    ax1.axhline(1.0, color="0.25", lw=1, ls="--")
    ax1.set_ylabel("Speedup over Picard-LBM")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=12, ha="right")
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, rel, "s-", color="#2d8a57", lw=2, label="rel L2 vs baseline")
    ax2.set_yscale("log")
    ax2.set_ylabel("Relative L2 velocity difference")
    ax1.set_title("Additional 2D masked-flow benchmarks")
    lines, labs = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labs + labs2, frameon=False, loc="upper right")
    savefig(path)
    return path


def plot_masks() -> Path:
    masks = [
        ("Backward step", backward_step_mask(64)),
        ("Cylinder wake analogue", cylinder_wake_mask(64)),
        ("T-junction", t_junction_mask(64)),
    ]
    path = OUT / "fig_revision_extra_benchmark_masks.png"
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2))
    for ax, (title, mask) in zip(axes, masks):
        ax.imshow(mask, origin="lower", cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"fluid fraction={mask.mean():.3f}")
    fig.suptitle("Additional benchmark geometries (white=fluid, black=solid)", y=1.02)
    savefig(path)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    remaining = json.loads(REMAINING.read_text(encoding="utf-8"))
    extra = json.loads(EXTRA.read_text(encoding="utf-8"))
    plot_scaling(remaining)
    plot_cavity(remaining)
    plot_extra_benchmarks(extra)
    plot_masks()


if __name__ == "__main__":
    main()
