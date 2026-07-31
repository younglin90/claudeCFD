"""Generate manuscript figures from the latest fixed30 scaling benchmark.

The script uses only cached outputs produced by verify_fixed30_scaling_strict.py.
It is intentionally tolerant of failed/non-finite cases so that manuscript
figures can still be generated from an honest benchmark table.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_60case_benchmark import macro_of
from verify_fixed30_scaling_strict import _load_cached, case_factory_scaled


SRC = Path("paper_revision_data") / "fixed30_scaling_strict"
OUT = Path("paper_revision_data") / "fixed30_manuscript_figures"
OUT.mkdir(parents=True, exist_ok=True)

METHOD_LABELS = {
    "picard_lbm": "Picard",
    "anderson_lbm": "Anderson",
    "preconditioned_lbm": "PLBE",
    "inexact_newton_lbe": "Newton",
    "dual_time_mg_lbm": "DTS-MG",
    "proposed": "SafeNN-Final",
}

CASE_LABELS = {
    "kolmogorov_n32": "Kolmogorov",
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


def _finite_float(value, default=np.nan):
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def load_data():
    with (SRC / "summary.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    metrics = json.loads((SRC / "metrics.json").read_text(encoding="utf-8"))
    return rows, metrics


def method_aggregate(rows):
    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    out = []
    for method, rs in by_method.items():
        lbes = [_finite_float(r["lbe_calls"], 0.0) for r in rs]
        walls = [_finite_float(r["wall_seconds"], 0.0) for r in rs]
        rels = [_finite_float(r["rel_l2_vs_picard"]) for r in rs]
        rels = [r for r in rels if np.isfinite(r)]
        out.append(
            {
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                "converged": sum(int(r["converged"]) for r in rs),
                "total_lbe": float(np.sum(lbes)),
                "total_wall": float(np.sum(walls)),
                "median_rel": float(np.median(rels)) if rels else np.nan,
                "mean_rel": float(np.mean(rels)) if rels else np.nan,
            }
        )
    return out


def plot_method_summary(rows, metrics):
    agg = method_aggregate(rows)
    labels = [a["label"] for a in agg]
    colors = ["#707070" if a["method"] != "proposed" else "#1f77b4" for a in agg]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.8))

    ax[0, 0].bar(labels, [a["total_lbe"] for a in agg], color=colors)
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_title("Total LBE calls")
    ax[0, 0].tick_params(axis="x", rotation=35)

    ax[0, 1].bar(labels, [a["total_wall"] for a in agg], color=colors)
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_title("Total wall-clock time (s)")
    ax[0, 1].tick_params(axis="x", rotation=35)

    ax[1, 0].bar(labels, [a["converged"] for a in agg], color=colors)
    ax[1, 0].axhline(30, color="k", lw=0.8, ls="--")
    ax[1, 0].set_ylim(0, 31)
    ax[1, 0].set_title("Converged cases out of 30")
    ax[1, 0].tick_params(axis="x", rotation=35)

    ax[1, 1].bar(labels, [a["median_rel"] for a in agg], color=colors)
    ax[1, 1].set_yscale("symlog", linthresh=1.0e-5)
    ax[1, 1].set_title("Median relative L2 vs Picard")
    ax[1, 1].tick_params(axis="x", rotation=35)

    fig.suptitle(
        "Fixed30 benchmark summary: "
        f"pass={metrics['pass_count']}/30, "
        f"LBE wins={metrics['lbe_win_count']}/30, "
        f"wall wins={metrics['wall_win_count']}/30, "
        f"accuracy wins={metrics['accuracy_win_count']}/30",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT / "fig_fixed30_method_summary.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_pass_heatmap(metrics):
    cases = metrics["case_results"]
    labels = [
        f"{CASE_LABELS.get(c['base_case_id'], c['base_case_id'])} {c['scaling_level']}x"
        for c in cases
    ]
    keys = ["converged", "lbe_win", "wall_win", "accuracy_win", "case_pass"]
    data = np.asarray([[c[k] for k in keys] for c in cases], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 9.5))
    im = ax.imshow(data, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(["Conv.", "LBE", "Wall", "Acc.", "Pass"], rotation=30, ha="right")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, "1" if data[i, j] > 0.5 else "0", ha="center", va="center", fontsize=7)
    ax.set_title("Proposed solver pass components across 30 scaled cases")
    fig.colorbar(im, ax=ax, shrink=0.7, ticks=[0, 1])
    fig.tight_layout()
    path = OUT / "fig_fixed30_pass_heatmap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_tradeoff(metrics):
    cases = metrics["case_results"]
    labels = [
        f"{CASE_LABELS.get(c['base_case_id'], c['base_case_id'])}\n{c['scaling_level']}x"
        for c in cases
    ]
    lbe_speed = [
        (c["best_fixed_lbe"] / max(c["proposed_lbe"], 1)) if c["best_fixed_lbe"] else np.nan
        for c in cases
    ]
    wall_speed = [
        (c["best_fixed_wall"] / max(c["proposed_wall"], 1.0e-30))
        if np.isfinite(c["best_fixed_wall"]) else np.nan
        for c in cases
    ]
    acc_ratio = []
    for c in cases:
        p = c["proposed_rel_l2"]
        b = c["best_fixed_rel_l2"]
        if np.isfinite(p) and np.isfinite(b) and p > 0:
            acc_ratio.append(b / p)
        else:
            acc_ratio.append(np.nan)
    x = np.arange(len(cases))
    fig, ax = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    ax[0].bar(x, lbe_speed, color="#1f77b4")
    ax[0].axhline(1, color="k", lw=0.8)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("LBE speedup")
    ax[0].set_title("Proposed solver tradeoff against the best converged fixed method")

    ax[1].bar(x, wall_speed, color="#2ca02c")
    ax[1].axhline(1, color="k", lw=0.8)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("Wall speedup")

    ax[2].bar(x, acc_ratio, color="#d62728")
    ax[2].axhline(1, color="k", lw=0.8)
    ax[2].set_yscale("symlog", linthresh=0.2)
    ax[2].set_ylabel("Accuracy ratio")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(labels, rotation=80, fontsize=7)
    ax[2].set_xlabel("Case and mesh level")
    fig.tight_layout()
    path = OUT / "fig_fixed30_speed_accuracy_tradeoff.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def read_history(case_id, method):
    path = SRC / "histories" / f"{case_id}__{method}.csv"
    if not path.exists():
        return None
    xs, ys = [], []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            x = _finite_float(row["lbe_calls"])
            y = _finite_float(row["residual"])
            if np.isfinite(x) and np.isfinite(y) and y > 0:
                xs.append(x)
                ys.append(y)
    return np.asarray(xs), np.asarray(ys)


def plot_residual_histories():
    base_cases = list(CASE_LABELS)
    fig, axes = plt.subplots(5, 2, figsize=(11, 12), sharex=False)
    for ax, base_id in zip(axes.ravel(), base_cases):
        case_id = f"{base_id}__1x"
        for method in ["picard_lbm", "anderson_lbm", "preconditioned_lbm", "proposed"]:
            hist = read_history(case_id, method)
            if hist is None:
                continue
            x, y = hist
            if len(x) == 0:
                continue
            ax.plot(x, y, lw=1.2, label=METHOD_LABELS.get(method, method))
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title(CASE_LABELS[base_id], fontsize=9)
        ax.grid(True, which="both", alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.supxlabel("LBE calls")
    fig.supylabel("Native residual")
    fig.suptitle("Residual histories for the 1x fixed30 benchmark cases", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = OUT / "fig_fixed30_residual_histories_1x.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def velocity_speed(case, f):
    _, ux, uy = macro_of(case, f)
    speed = np.sqrt(np.nan_to_num(ux * ux + uy * uy, nan=0.0, posinf=0.0, neginf=0.0))
    return speed


def plot_field_grid(base_cases, filename, title):
    fig, axes = plt.subplots(len(base_cases), 3, figsize=(10.5, 2.3 * len(base_cases)))
    if len(base_cases) == 1:
        axes = axes[None, :]
    for row_idx, base_id in enumerate(base_cases):
        case_id, _, _, factory = case_factory_scaled(base_id, 1)
        case_p = factory()
        case_s = factory()
        picard = _load_cached(case_id, "picard_lbm")
        proposed = _load_cached(case_id, "proposed")
        if picard is None or proposed is None:
            for j in range(3):
                axes[row_idx, j].axis("off")
            continue
        f_p = picard[0]
        f_s = proposed[0]
        sp = velocity_speed(case_p, f_p)
        ss = velocity_speed(case_s, f_s)
        err = np.abs(ss - sp)
        vmax = np.nanpercentile(np.concatenate([sp.ravel(), ss.ravel()]), 99.0)
        vmax = max(float(vmax), 1.0e-12)
        for j, (arr, label, cm, vm) in enumerate(
            [
                (sp, "Picard |u|", "viridis", vmax),
                (ss, "SafeNN |u|", "viridis", vmax),
                (err, "|difference|", "magma", max(float(np.nanpercentile(err, 99.0)), 1.0e-12)),
            ]
        ):
            im = axes[row_idx, j].imshow(arr, origin="lower", cmap=cm, vmin=0, vmax=vm)
            axes[row_idx, j].set_xticks([])
            axes[row_idx, j].set_yticks([])
            if row_idx == 0:
                axes[row_idx, j].set_title(label, fontsize=9)
            fig.colorbar(im, ax=axes[row_idx, j], fraction=0.046, pad=0.02)
        axes[row_idx, 0].set_ylabel(CASE_LABELS[base_id], fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = OUT / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main():
    rows, metrics = load_data()
    paths = [
        plot_method_summary(rows, metrics),
        plot_pass_heatmap(metrics),
        plot_tradeoff(metrics),
        plot_residual_histories(),
        plot_field_grid(
            [
                "kolmogorov_n32",
                "channel_n32",
                "couette_n32",
                "cavity_re100_n33",
                "cavity_re400_n49",
                "cavity_re1000_n129",
            ],
            "fig_fixed30_core_fields_1x.png",
            "Core 1x velocity fields and proposed-vs-Picard differences",
        ),
        plot_field_grid(
            [
                "multi_cylinder_n32",
                "backward_step_n64",
                "cylinder_wake_n64",
                "t_junction_n64",
            ],
            "fig_fixed30_mask_fields_1x.png",
            "Masked-flow 1x velocity fields and proposed-vs-Picard differences",
        ),
    ]
    manifest = {"figures": [str(p) for p in paths]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
