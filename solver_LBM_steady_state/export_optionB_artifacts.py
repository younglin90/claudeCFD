"""Export Option-B faithful-baseline paper artifacts.

This script is intentionally post-processing only: it reads the current
``verify_fixed10_strict.py`` summary/cache outputs and writes publication
working artifacts under ``paper_revision_data/optionB_faithful_benchmark``.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_60case_benchmark import CASE_IDS, METHODS, case_factory, macro_of, velocity_error, write_vtk
from verify_fixed10_strict import _cache_path, _load_cached


SRC = Path("paper_revision_data") / "fixed10_strict"
OUT = Path("paper_revision_data") / "optionB_faithful_benchmark"


def read_rows():
    with (SRC / "summary.csv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(row, key):
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row.get(k, "") for k in fields})


def aggregate_methods(rows):
    out = []
    for method in METHODS:
        mr = [r for r in rows if r["method"] == method]
        if not mr:
            continue
        out.append({
            "method": method,
            "case_count": len(mr),
            "converged_count": sum(int(float(r["converged"])) for r in mr),
            "mean_lbe_calls": np.nanmean([fnum(r, "lbe_calls") for r in mr]),
            "mean_wall_seconds": np.nanmean([fnum(r, "wall_seconds") for r in mr]),
            "mean_final_residual": np.nanmean([fnum(r, "final_residual") for r in mr]),
            "mean_rel_l2_vs_picard": np.nanmean([fnum(r, "rel_l2_vs_picard") for r in mr]),
            "max_rel_l2_vs_picard": np.nanmax([fnum(r, "rel_l2_vs_picard") for r in mr]),
        })
    return out


def write_reports(metrics: dict, rows: list[dict]):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reference_audit_report.md").write_text(
        "\n".join([
            "# Option-B Reference Method Audit",
            "",
            "## Current implementation status",
            "- `picard_lbm`: native fixed-point LBM reference.",
            "- `anderson_lbm`: Walker--Ni style fixed-point Anderson acceleration with residual safeguard.",
            "- `preconditioned_lbm`: Guo--Zhao--Shi-style PLBE equilibrium distribution function with explicit Picard iteration; this is more faithful than the previous AP-Schur proxy, but boundary/mask PLBE handling remains an implementation approximation.",
            "- `inexact_newton_lbe`: NE-preconditioned matrix-free JFNK with GMRES, line search, and Picard smoothing; still single-domain rather than full two-level ASPIN.",
            "- `dual_time_mg_lbm`: 2-level dual-time V-cycle proxy with coarse restriction/prolongation and LBE smoothing; failures/NaNs are reported, not hidden.",
            "",
            "## Important deviations to disclose",
            "- Backward step, cylinder wake, and T-junction remain periodic masked-flow analogues unless true inlet/outlet cases are later implemented.",
            "- The strict verifier still compares velocity error against Picard reference; a tighter independent reference should be generated for final manuscript tables.",
            "- Dual-time MG is improved from the old spectral smoother proxy but remains a compact two-level implementation, not a full production multigrid code.",
        ]),
        encoding="utf-8",
    )
    (OUT / "proposed_method_report.md").write_text(
        "\n".join([
            "# SafeNN-Final Current Status",
            "",
            f"- Strict score: `{metrics.get('score')}`",
            f"- Pass count: `{metrics.get('pass_count')}/{metrics.get('case_count')}`",
            f"- LBE win count: `{metrics.get('lbe_win_count')}/{metrics.get('case_count')}`",
            f"- Wall-clock win count: `{metrics.get('wall_win_count')}/{metrics.get('case_count')}`",
            f"- Accuracy win count: `{metrics.get('accuracy_win_count')}/{metrics.get('case_count')}`",
            f"- Anti-cheat pass: `{metrics.get('anti_cheat_pass')}`",
            "",
            "SafeNN-Final is currently fast in LBE calls on most cases, but accuracy-vs-Picard remains the main failure mode for channel, Couette, multi-cylinder, backward-step, and cylinder-wake analogue cases.",
        ]),
        encoding="utf-8",
    )
    failures = [c for c in metrics.get("case_results", []) if not c.get("case_pass")]
    lines = ["# Failure and Limitation Report", ""]
    for c in failures:
        lines.append(
            f"- `{c['case_id']}`: pass={c['case_pass']}, lbe_win={c['lbe_win']}, "
            f"wall_win={c['wall_win']}, accuracy_win={c['accuracy_win']}, "
            f"proposed_lbe={c['proposed_lbe']}, best_fixed_lbe={c['best_fixed_lbe']}, "
            f"proposed_rel_l2={c['proposed_rel_l2']:.6g}, best_fixed_rel_l2={c['best_fixed_rel_l2']:.6g}."
        )
    lines += [
        "",
        "Primary limitation: the current strict gate takes the best LBE, best wall-clock, and best accuracy from possibly different reference methods. This is a useful stress test, but final paper claims should also include Pareto or accuracy-constrained comparisons.",
    ]
    (OUT / "failure_and_limitation_report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_case(case_id: str, rows: list[dict]):
    fig_dir = OUT / "figures" / case_id
    fig_dir.mkdir(parents=True, exist_ok=True)
    case_rows = [r for r in rows if r["case_id"] == case_id]

    # residual histories
    plt.figure(figsize=(6, 4))
    for r in case_rows:
        hpath = SRC / "histories" / f"{case_id}__{r['method']}.csv"
        if not hpath.exists():
            continue
        data = np.genfromtxt(hpath, delimiter=",", names=True)
        if data.size == 0:
            continue
        x = np.atleast_1d(data["lbe_calls"])
        y = np.atleast_1d(data["residual"])
        plt.semilogy(x, y, label=r["method"])
    plt.xlabel("LBE calls")
    plt.ylabel("native residual")
    plt.title(case_id)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_dir / "residual_vs_lbe.png", dpi=220)
    plt.savefig(fig_dir / "residual_vs_lbe.pdf")
    plt.close()

    plt.figure(figsize=(6, 4))
    methods = [r["method"] for r in case_rows]
    lbe = [fnum(r, "lbe_calls") for r in case_rows]
    err = [fnum(r, "rel_l2_vs_picard") for r in case_rows]
    plt.bar(methods, lbe)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.ylabel("LBE calls")
    plt.title(f"{case_id}: work")
    plt.tight_layout()
    plt.savefig(fig_dir / "lbe_bar.png", dpi=220)
    plt.savefig(fig_dir / "lbe_bar.pdf")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(methods, err)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.ylabel("relative L2 vs Picard")
    plt.title(f"{case_id}: field error")
    plt.tight_layout()
    plt.savefig(fig_dir / "accuracy_bar.png", dpi=220)
    plt.savefig(fig_dir / "accuracy_bar.pdf")
    plt.close()


def export_vtk(rows: list[dict]):
    vtk_dir = OUT / "vtk"
    vtk_dir.mkdir(parents=True, exist_ok=True)
    for case_id in CASE_IDS:
        _, _, factory = case_factory(case_id)
        for method in METHODS:
            cached = _load_cached(case_id, method)
            if cached is None:
                continue
            f, _, _ = cached
            case = factory()
            write_vtk(vtk_dir / f"{case_id}__{method}.vtk", case, f)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    metrics = json.loads((SRC / "metrics.json").read_text(encoding="utf-8"))
    shutil.copy2(SRC / "summary.csv", OUT / "summary_all_methods.csv")
    shutil.copy2(SRC / "metrics.json", OUT / "metrics.json")
    write_csv(
        OUT / "per_case_metrics.csv",
        metrics["case_results"],
        [
            "case_id", "case_pass", "converged", "lbe_win", "wall_win", "accuracy_win",
            "proposed_lbe", "best_fixed_lbe", "proposed_wall", "best_fixed_wall",
            "proposed_rel_l2", "best_fixed_rel_l2",
        ],
    )
    method_rows = aggregate_methods(rows)
    write_csv(
        OUT / "per_method_score.csv",
        method_rows,
        [
            "method", "case_count", "converged_count", "mean_lbe_calls",
            "mean_wall_seconds", "mean_final_residual", "mean_rel_l2_vs_picard",
            "max_rel_l2_vs_picard",
        ],
    )
    (OUT / "reproducibility_config.json").write_text(
        json.dumps({
            "source_summary": str(SRC / "summary.csv"),
            "source_metrics": str(SRC / "metrics.json"),
            "methods": METHODS,
            "cases": CASE_IDS,
            "numba_threads": 24,
            "cache_policy": "verify_fixed10_strict source-hash cache key",
        }, indent=2),
        encoding="utf-8",
    )
    hist_out = OUT / "histories"
    if hist_out.exists():
        shutil.rmtree(hist_out)
    shutil.copytree(SRC / "histories", hist_out)
    write_reports(metrics, rows)
    for case_id in CASE_IDS:
        plot_case(case_id, rows)
    export_vtk(rows)
    print(json.dumps({
        "status": "ok",
        "out": str(OUT),
        "summary_all_methods": str(OUT / "summary_all_methods.csv"),
        "vtk_count": len(list((OUT / "vtk").glob("*.vtk"))),
        "figure_count": len(list((OUT / "figures").glob("**/*.png"))),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
