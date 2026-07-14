#!/usr/bin/env python3
"""Generate Euler-only 1D manuscript evidence.

This driver excludes source-term validations (32--35) and builds the missing
paper artifacts requested for a 1D Euler-equation numerical-method manuscript:

* core validation table from the latest evidence run,
* grid-refinement metrics,
* baseline/ablation comparisons,
* pressure-equilibrium preservation diagnostics,
* acoustic-CFL / all-speed diagnostics,
* PNG plots and CSV raw metric tables.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "1D" / "paper_euler_evidence"
PLOTS = OUT / "plots"
CSV_DIR = OUT / "csv"

BASE_ENV = {
    "MPLCONFIGDIR": "/tmp/mpl",
    "PYTHONPATH": ".codex-loop",
    "FIVE_EQ_IMEX_TIME_INTEGRATOR": "imex_ssp3",
    "FIVE_EQ_IMEX_ALPHA_SCHEME": "adaptive_bvd",
    "FIVE_EQ_IMEX_PRIMITIVE_SCHEME": "tmlpu",
    "FIVE_EQ_IMEX_TMLPU_TVD": "superbee",
    "FIVE_EQ_IMEX_MATERIAL_FLUX": "slau2",
    "FIVE_EQ_IMEX_PRESSURE_CLOSURE": "regime_auto",
    "FIVE_EQ_IMEX_CHARACTERISTIC_RECON": "1",
    "FIVE_EQ_IMEX_RUSANOV_FALLBACK": "0",
    "FIVE_EQ_IMEX_UNIFORM_PERIODIC_REMAP": "0",
    # WENO5-JS acoustic-face reconstruction (autoresearch 2026-07-14): removes
    # the air-water interface pressure ringing without tuned constants; the
    # full strict suite passes under this single configuration.
    "FIVE_EQ_IMEX_ACOUSTIC_RECON": "weno5",
    # Keep the costly hypersonic mixture case reproducible under the paper
    # budget while staying within the user's N<=800 ceiling.
    "FIVE_EQ_CASE24_N": "400",
}

CASE_SCRIPT = {
    "01_A": "results/1D/cases/01_A_PE_static_interface.py",
    "02_A": "results/1D/cases/02_A_PE_advection_unified.py",
    "03_B": "results/1D/cases/03_B_acoustic_ultra_low_mach_pulse.py",
    "04_B": "results/1D/cases/04_B_acoustic_sinusoidal_air_2000Hz.py",
    "05_B": "results/1D/cases/05_B_acoustic_sinusoidal_water_6000Hz.py",
    "07_B": "results/1D/cases/07_B_acoustic_reflection_transmission.py",
    "13_E": "results/1D/cases/13_E_shocktube_hp_air_lp_water.py",
    "14_E": "results/1D/cases/14_E_shocktube_hp_water_lp_air.py",
    "15_E": "results/1D/cases/15_E_cavitation.py",
    "16_T": "results/1D/cases/16_T_advection_hot_gas_cold_liquid.py",
    "17_T": "results/1D/cases/17_T_smooth_alpha_gaussian_hot_gas.py",
    "18_T": "results/1D/cases/18_T_thermal_wave_advection_p_equil.py",
    "24_H": "results/1D/cases/24_H_hypersonic_mixture_ms10.py",
    "25_H": "results/1D/cases/25_H_hypersonic_mach10_air_water.py",
}

CORE_EULER = (
    "01_A", "02_A", "04_B", "05_B", "07_B", "13_E", "14_E",
    "15_E", "16_T", "17_T", "18_T", "24_H", "25_H",
)

GRID_SWEEPS = (
    ("07_B", "FIVE_EQ_CASE07_N", (100, 200, 400),
     {"FIVE_EQ_CASE07_ONLY": "Air-Water"}),
    ("13_E", "FIVE_EQ_CASE13_N", (200, 400, 800), {}),
    ("14_E", "FIVE_EQ_CASE14_N", (200, 400, 800), {}),
    ("18_T", "FIVE_EQ_IMEX_TEMP_N_18", (200, 400, 550), {}),
    ("24_H", "FIVE_EQ_CASE24_N", (100, 200, 400), {}),
    ("25_H", "FIVE_EQ_CASE25_N", (200, 400, 800), {}),
)

VARIANTS = (
    ("production", {}),
    ("upwind_primitive", {"FIVE_EQ_IMEX_PRIMITIVE_SCHEME": "upwind"}),
    ("superbee_only", {"FIVE_EQ_IMEX_PRIMITIVE_SCHEME": "superbee"}),
    ("tmlpu_vanleer", {
        "FIVE_EQ_IMEX_PRIMITIVE_SCHEME": "tmlpu",
        "FIVE_EQ_IMEX_TMLPU_TVD": "vanleer",
    }),
    ("tmlpu_minmod", {
        "FIVE_EQ_IMEX_PRIMITIVE_SCHEME": "tmlpu",
        "FIVE_EQ_IMEX_TMLPU_TVD": "minmod",
    }),
    ("alpha_cicsam", {"FIVE_EQ_IMEX_ALPHA_SCHEME": "cicsam"}),
    ("alpha_mstacs", {"FIVE_EQ_IMEX_ALPHA_SCHEME": "mstacs"}),
    ("hllc_flux", {"FIVE_EQ_IMEX_MATERIAL_FLUX": "hllc_split"}),
)

BASELINE_TARGETS = (
    ("02_A", {}),
    ("07_B", {
        "FIVE_EQ_CASE07_ONLY": "Air-Water",
        "FIVE_EQ_CASE07_N": "200",
        "FIVE_EQ_CASE07_N_AIR_WATER": "200",
    }),
    ("13_E", {"FIVE_EQ_CASE13_N": "400"}),
    ("18_T", {"FIVE_EQ_IMEX_TEMP_N_18": "400"}),
)

CFL_SWEEP = (
    ("07_AirWater_CFL0p2", "07_B", {
        "FIVE_EQ_CASE07_ONLY": "Air-Water",
        "FIVE_EQ_CASE07_N": "200",
        "FIVE_EQ_CASE07_N_AIR_WATER": "200",
        "FIVE_EQ_CASE07_CFL": "0.2",
    }),
    ("07_AirWater_CFL0p4", "07_B", {
        "FIVE_EQ_CASE07_ONLY": "Air-Water",
        "FIVE_EQ_CASE07_N": "200",
        "FIVE_EQ_CASE07_N_AIR_WATER": "200",
        "FIVE_EQ_CASE07_CFL": "0.4",
    }),
    ("07_AirWater_CFL0p6", "07_B", {
        "FIVE_EQ_CASE07_ONLY": "Air-Water",
        "FIVE_EQ_CASE07_N": "200",
        "FIVE_EQ_CASE07_N_AIR_WATER": "200",
        "FIVE_EQ_CASE07_CFL": "0.6",
    }),
)

ALL_SPEED = (
    ("ultra_low_mach_03_B", "03_B", {}),
    ("low_mach_air_04_B", "04_B", {"FIVE_EQ_CASE04_N": "200"}),
    ("interface_acoustic_07_B", "07_B", {
        "FIVE_EQ_CASE07_ONLY": "Air-Water",
        "FIVE_EQ_CASE07_N": "400",
        "FIVE_EQ_CASE07_N_AIR_WATER": "400",
    }),
    ("hypersonic_25_H", "25_H", {"FIVE_EQ_CASE25_N": "200"}),
)


def _env(extra: dict[str, str] | None = None) -> dict[str, str]:
    out = os.environ.copy()
    out.update(BASE_ENV)
    if extra:
        out.update({k: str(v) for k, v in extra.items()})
    return out


def _extract_json(text: str) -> dict[str, Any]:
    found: dict[str, Any] | None = None
    for line in text.splitlines():
        for prefix in ("CASE_JSON ", "ACCEPTANCE_JSON ", "QUALITY_JSON "):
            if line.startswith(prefix):
                try:
                    found = json.loads(line[len(prefix):])
                except json.JSONDecodeError:
                    pass
    if found is not None:
        return found
    return {}


def _case_plot_path(case: str) -> Path:
    return ROOT / "results" / "1D" / case / "diff_vs_exact.png"


def _copy_plot(case: str, name: str) -> str:
    src = _case_plot_path(case)
    if not src.exists():
        return ""
    dst = PLOTS / f"{name}.png"
    shutil.copy2(src, dst)
    src_pdf = src.with_suffix(".pdf")
    if src_pdf.exists():
        shutil.copy2(src_pdf, PLOTS / f"{name}.pdf")
    return str(dst.relative_to(ROOT))


def _run(label: str, case: str, extra_env: dict[str, str] | None = None,
         timeout: int = 1800) -> dict[str, Any]:
    cmd = [sys.executable, CASE_SCRIPT[case]]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=_env(extra_env),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        payload = _extract_json(proc.stdout)
        ok = bool(payload.get("pass", False)) and proc.returncode == 0
        row = {
            "label": label,
            "case": case,
            "pass": ok,
            "returncode": proc.returncode,
            "wall_s": time.time() - t0,
            "env": extra_env or {},
            "json": payload,
            "tail": "\n".join(proc.stdout.splitlines()[-12:]),
        }
    except subprocess.TimeoutExpired as exc:
        row = {
            "label": label,
            "case": case,
            "pass": False,
            "returncode": 124,
            "wall_s": time.time() - t0,
            "env": extra_env or {},
            "json": {},
            "tail": str(exc),
        }
    row["plot"] = _copy_plot(case, label.replace("/", "_").replace(" ", "_"))
    return row


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
        return out
    if isinstance(obj, (list, tuple)):
        # Store complex lists as JSON; subcases are expanded separately.
        return {prefix: json.dumps(obj, sort_keys=True, default=str)}
    return {prefix: obj}


def _expand_metric_rows(group: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        base = {
            "group": group,
            "label": row["label"],
            "case": row["case"],
            "pass": row["pass"],
            "returncode": row["returncode"],
            "wall_s": row["wall_s"],
            "plot": row.get("plot", ""),
            "env": json.dumps(row.get("env", {}), sort_keys=True),
        }
        js = row.get("json", {})
        subcases = js.get("subcases")
        if isinstance(subcases, list) and subcases:
            for sub in subcases:
                sub_base = dict(base)
                sub_base["subcase"] = sub.get("case") or sub.get("name", "")
                sub_base.update(_flatten({k: v for k, v in sub.items()
                                          if k != "subcases"}))
                out.append(sub_base)
        else:
            base["subcase"] = ""
            base.update(_flatten(js))
            out.append(base)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_core_from_latest() -> list[dict[str, Any]]:
    src = ROOT / "results" / "1D" / "paper_evidence" / "paper_1d_evidence.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    rows = []
    for row in data.get("core", []):
        case = row["label"].split("/", 1)[1]
        if case not in CORE_EULER:
            continue
        copied = dict(row)
        copied["case"] = case
        copied["plot"] = _copy_plot(case, f"core_{case}")
        rows.append(copied)
    return rows


def _representative_metric(row: dict[str, Any]) -> float:
    js = row.get("json", {})
    case = row.get("case")
    if case == "07_B":
        subs = js.get("subcases", [])
        sub = subs[0] if subs else {}
        return float(sub.get("metrics", {}).get("L2p", float("nan")))
    if case == "13_E":
        return float(js.get("case13_rho_smooth_l2_rel", js.get("p_osc", float("nan"))))
    if case == "14_E":
        return float(js.get("case14_rho_plateau085_089_linf_ratio", js.get("p_osc", float("nan"))))
    if case == "18_T":
        return float(js.get("rho_l1_ratio", js.get("Tmix_l1_ratio", float("nan"))))
    if case == "24_H":
        vals = [float(s.get("rho_profile_l2", float("nan")))
                for s in js.get("subcases", []) if "rho_profile_l2" in s]
        return max(vals) if vals else float("nan")
    if case == "25_H":
        return float(js.get("p_scaled_l2", js.get("p_osc", float("nan"))))
    if "p_rel_linf" in js:
        return float(js["p_rel_linf"])
    if "p_scaled_l2" in js:
        return float(js["p_scaled_l2"])
    if "p_osc" in js:
        return float(js["p_osc"])
    return float("nan")


def _plot_convergence(rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    by_case: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        n = None
        for k, v in row.get("env", {}).items():
            if k.endswith("_N") or "_N_" in k:
                try:
                    n = int(v)
                except ValueError:
                    pass
        if n is None:
            continue
        by_case.setdefault(row["case"], []).append((n, _representative_metric(row)))
    for case, vals in sorted(by_case.items()):
        vals = sorted(vals)
        ax.plot([v[0] for v in vals], [v[1] for v in vals], marker="o", label=case)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("representative normalized error")
    ax.set_title("Grid-refinement evidence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "grid_refinement_errors.png", dpi=300)
    fig.savefig(PLOTS / "grid_refinement_errors.pdf")
    plt.close(fig)


def _plot_variant_bars(rows: list[dict[str, Any]]) -> None:
    metric_rows = []
    for row in rows:
        variant = row["label"].split("/", 1)[0]
        metric_rows.append((variant, row["case"], _representative_metric(row), row["pass"]))
    variants = list(dict.fromkeys(v for v, _, _, _ in metric_rows))
    cases = list(dict.fromkeys(c for _, c, _, _ in metric_rows))
    fig, axes = plt.subplots(len(cases), 1, figsize=(10, max(3, 2.6 * len(cases))), sharex=True)
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        vals = [m for v, c, m, _ in metric_rows if c == case for vv in [v]]
        labels = [v for v, c, _, _ in metric_rows if c == case]
        colors = ["#2a9d8f" if p else "#e76f51" for v, c, m, p in metric_rows if c == case]
        ax.bar(labels, vals, color=colors)
        ax.set_yscale("log")
        ax.set_ylabel(case)
        ax.grid(True, axis="y", which="both", alpha=0.25)
    axes[-1].tick_params(axis="x", rotation=35)
    fig.suptitle("Baseline / ablation comparison")
    fig.tight_layout()
    fig.savefig(PLOTS / "baseline_ablation_metrics.png", dpi=300)
    fig.savefig(PLOTS / "baseline_ablation_metrics.pdf")
    plt.close(fig)


def _plot_pass_heatmap(rows: list[dict[str, Any]]) -> None:
    variants = sorted(set(row["label"].split("/", 1)[0] for row in rows))
    cases = sorted(set(row["case"] for row in rows))
    mat = [[1.0 if any(r["label"].startswith(v + "/") and r["case"] == c and r["pass"]
                       for r in rows) else 0.0 for v in variants] for c in cases]
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(variants)), variants, rotation=35, ha="right")
    ax.set_yticks(range(len(cases)), cases)
    ax.set_title("Ablation PASS map")
    for i, c in enumerate(cases):
        for j, v in enumerate(variants):
            ax.text(j, i, "P" if mat[i][j] else "F", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(PLOTS / "ablation_pass_heatmap.png", dpi=300)
    fig.savefig(PLOTS / "ablation_pass_heatmap.pdf")
    plt.close(fig)


def _plot_pe(rows: list[dict[str, Any]]) -> None:
    pe = [r for r in rows if r["case"] in {"01_A", "02_A", "16_T", "17_T", "18_T"}]
    labels = [r["case"] for r in pe]
    p = [float(r.get("json", {}).get("p_rel_linf", 0.0)) for r in pe]
    u = [float(r.get("json", {}).get("u_abs_linf", 0.0)) for r in pe]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(labels))
    ax.bar([i - 0.18 for i in x], p, width=0.36, label="p rel Linf")
    ax.bar([i + 0.18 for i in x], u, width=0.36, label="u abs Linf")
    ax.set_yscale("log")
    ax.set_xticks(list(x), labels)
    ax.set_title("Pressure-equilibrium preservation")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "pressure_equilibrium_preservation.png", dpi=300)
    fig.savefig(PLOTS / "pressure_equilibrium_preservation.pdf")
    plt.close(fig)


def _plot_cfl(rows: list[dict[str, Any]]) -> None:
    xs, l2p, lip, ok = [], [], [], []
    for row in rows:
        cfl = float(row["env"].get("FIVE_EQ_CASE07_CFL", "nan"))
        sub = (row.get("json", {}).get("subcases") or [{}])[0]
        m = sub.get("metrics", {})
        xs.append(cfl)
        l2p.append(float(m.get("L2p", float("nan"))))
        lip.append(float(m.get("Lip", float("nan"))))
        ok.append(row["pass"])
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([xs[i] for i in order], [l2p[i] for i in order], marker="o", label="L2p")
    ax.plot([xs[i] for i in order], [lip[i] for i in order], marker="s", label="Linf p")
    for i in order:
        ax.text(xs[i], l2p[i], "P" if ok[i] else "F", fontsize=8)
    ax.set_xlabel("CFL")
    ax.set_ylabel("normalized error")
    ax.set_title("Acoustic CFL sensitivity: 07_B Air-Water")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "acoustic_cfl_sweep.png", dpi=300)
    fig.savefig(PLOTS / "acoustic_cfl_sweep.pdf")
    plt.close(fig)


def _write_markdown(result: dict[str, Any]) -> None:
    lines = [
        "# Euler-only 1D Paper Evidence",
        "",
        f"Generated: {result['generated_at']}",
        "",
        "## Scope",
        "",
        "This package excludes source-term cases 32--35 and focuses on the",
        "homogeneous Euler part of the five-equation model.",
        "",
        "## Fixed Production Method",
        "",
    ]
    for k, v in BASE_ENV.items():
        lines.append(f"- `{k}` = `{v}`")
    lines += [
        "",
        "## Artifacts",
        "",
        "- Raw metric CSV files: `csv/*.csv`",
        "- Manuscript PNG figures: `plots/*.png`",
        "- Machine-readable summary: `paper_euler_evidence.json`",
        "",
        "## Core Euler Sweep",
        "",
        "| Case | PASS | Wall(s) | Plot |",
        "|---|---:|---:|---|",
    ]
    for row in result["core"]:
        lines.append(
            f"| `{row['case']}` | {row['pass']} | {row['wall_s']:.1f} | `{row.get('plot','')}` |"
        )
    lines += [
        "",
        "## Required Manuscript Evidence Added",
        "",
        "1. Grid refinement: `csv/grid_metrics.csv`, `plots/grid_refinement_errors.png`.",
        "2. Baseline comparisons: `csv/baseline_metrics.csv`, `plots/baseline_ablation_metrics.png`.",
        "3. Ablation study: same CSV plus `plots/ablation_pass_heatmap.png`.",
        "4. Pressure-equilibrium preservation: `csv/core_metrics.csv`, `plots/pressure_equilibrium_preservation.png`.",
        "5. Acoustic CFL / all-speed diagnostics: `csv/cfl_metrics.csv`, `csv/all_speed_metrics.csv`, `plots/acoustic_cfl_sweep.png`.",
        "6. Source-term validation intentionally excluded for a future paper.",
        "7. Readiness summary updated separately in `docs/1d_method_paper_readiness.md`.",
        "",
    ]
    (OUT / "paper_euler_evidence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-core", action="store_true",
                        help="Rerun core Euler cases instead of reusing latest full evidence.")
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-cfl", action="store_true")
    parser.add_argument("--only-all-speed", action="store_true",
                        help="Update only the all-speed group in an existing output package.")
    args = parser.parse_args()

    PLOTS.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    result_path = OUT / "paper_euler_evidence.json"
    if args.only_all_speed and result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        result["all_speed"] = []
    else:
        result: dict[str, Any] = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "core": [],
            "grid": [],
            "baseline": [],
            "cfl": [],
            "all_speed": [],
        }

    if args.only_all_speed:
        pass
    elif args.run_core:
        for case in CORE_EULER:
            print(f"RUN core/{case}", flush=True)
            result["core"].append(_run(f"core/{case}", case, timeout=1800))
    else:
        result["core"] = _load_core_from_latest()

    if (not args.only_all_speed) and not args.skip_grid:
        for case, key, values, extra in GRID_SWEEPS:
            for n in values:
                env = dict(extra)
                env[key] = str(n)
                if case == "07_B":
                    env["FIVE_EQ_CASE07_N_AIR_WATER"] = str(n)
                print(f"RUN grid/{case}/N{n}", flush=True)
                result["grid"].append(_run(f"grid/{case}/N{n}", case, env, timeout=1800))

    if (not args.only_all_speed) and not args.skip_baseline:
        for variant, venv in VARIANTS:
            for case, cenv in BASELINE_TARGETS:
                env = dict(venv)
                env.update(cenv)
                print(f"RUN {variant}/{case}", flush=True)
                result["baseline"].append(_run(f"{variant}/{case}", case, env, timeout=1800))

    if (not args.only_all_speed) and not args.skip_cfl:
        for label, case, env in CFL_SWEEP:
            print(f"RUN cfl/{label}", flush=True)
            result["cfl"].append(_run(f"cfl/{label}", case, env, timeout=1800))
    if args.only_all_speed or not args.skip_cfl:
        for label, case, env in ALL_SPEED:
            print(f"RUN all_speed/{label}", flush=True)
            result["all_speed"].append(_run(f"all_speed/{label}", case, env, timeout=1800))

    _write_csv(CSV_DIR / "core_metrics.csv", _expand_metric_rows("core", result["core"]))
    _write_csv(CSV_DIR / "grid_metrics.csv", _expand_metric_rows("grid", result["grid"]))
    _write_csv(CSV_DIR / "baseline_metrics.csv", _expand_metric_rows("baseline", result["baseline"]))
    _write_csv(CSV_DIR / "cfl_metrics.csv", _expand_metric_rows("cfl", result["cfl"]))
    _write_csv(CSV_DIR / "all_speed_metrics.csv", _expand_metric_rows("all_speed", result["all_speed"]))

    _plot_pe(result["core"])
    if result["grid"]:
        _plot_convergence(result["grid"])
    if result["baseline"]:
        _plot_variant_bars(result["baseline"])
        _plot_pass_heatmap(result["baseline"])
    if result["cfl"]:
        _plot_cfl(result["cfl"])

    all_rows = result["core"] + result["grid"] + result["baseline"] + result["cfl"] + result["all_speed"]
    result["pass_count"] = int(sum(1 for r in all_rows if r["pass"]))
    result["fail_count"] = int(sum(1 for r in all_rows if not r["pass"]))
    result["total"] = len(all_rows)
    (OUT / "paper_euler_evidence.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    _write_markdown(result)
    print(json.dumps({
        "out_dir": str(OUT),
        "pass_count": result["pass_count"],
        "fail_count": result["fail_count"],
        "total": result["total"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
