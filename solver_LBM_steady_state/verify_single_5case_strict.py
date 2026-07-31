"""Strict 5-case verifier for one proposed LBM method.

This verifier is intentionally stricter than the earlier competitive helper:
the proposed solver may not use analytic target/equilibrium lifting. Analytic
solutions are allowed only as references for error measurement.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from paper_60case_benchmark import MULTI_CYLINDER_GEOMETRY_VERSION, make_multi_cylinder_mask
from solver_proposed_single import solve_proposed_single
from solver_unified_safe_nn import _residual_norm
from v2_run_all import case_macro, make_case, picard_tail


CASES = [
    ("kolmogorov", 32),
    ("channel", 32),
    ("couette", 32),
    ("cavity_re100", 33),
    ("multi_cylinder", 32),
]
FIXED_METHODS = {
    "picard_lbm",
    "anderson_lbm",
    "preconditioned_lbm",
    "inexact_newton_lbe",
    "dual_time_mg_lbm",
}
BASELINE_CACHE = Path("paper_revision_data") / "v2_final" / "compare_5case_11method.json"
OUT = Path("paper_revision_data") / "single5_strict"
VTK_ROOT = Path("paper_revision_data") / "v2_final"
BENCH60_VTK = Path("paper_revision_data") / "bench60" / "vtk"
SOLVER_PATH = Path("solver_proposed_single.py")
FORBIDDEN_SOLVER_TOKENS = (
    "analytical_ux",
    "_analytic_equilibrium",
    "target_converged",
    "target-deflated",
    "target deflated",
    "solve_anderson",
    "low_amplitude_unmasked",
)
ACCURACY_REL_TOL = 2.0e-6
ACCURACY_ABS_TOL = 1.0e-12


def anti_cheat_report():
    text = SOLVER_PATH.read_text(encoding="utf-8")
    hits = [token for token in FORBIDDEN_SOLVER_TOKENS if token in text]
    return {
        "anti_cheat_pass": int(not hits),
        "anti_cheat_hits": hits,
    }


def read_vtk_velocity(path: Path, n: int):
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("VECTORS velocity"):
            start = i + 1
            break
    if start is None:
        raise ValueError(f"missing velocity vectors in {path}")
    data = np.loadtxt(lines[start : start + n * n], dtype=np.float64)
    return data[:, 0].reshape(n, n), data[:, 1].reshape(n, n)


def analytic_reference(case_id, case):
    if not hasattr(case, "analytical_ux"):
        return None
    ux = case.analytical_ux()
    if ux.ndim == 1:
        if case_id == "channel":
            ux = np.tile(ux[:, None], (1, case.N))
        else:
            ux = np.tile(ux[None, :], (case.N, 1))
    return ux, np.zeros_like(ux)


def stored_picard_reference(case_id, n):
    if case_id == "cavity_re100":
        path = VTK_ROOT / "cavity_re100" / f"N{n}" / "picard_lbm" / "field.vtk"
    else:
        return None
    return read_vtk_velocity(path, n)


def reference_for(case_id, n, case):
    ref = analytic_reference(case_id, case)
    if ref is not None:
        return ref
    if case_id == "multi_cylinder":
        cache = OUT / f"{case_id}_{MULTI_CYLINDER_GEOMETRY_VERSION}_N{n}_tight_picard_ref.npz"
        if cache.exists():
            data = np.load(cache)
            return data["ux"], data["uy"]
        from solver_baseline import solve_baseline

        f_ref, _ = solve_baseline(
            case,
            max_steps=200000,
            tol=1.0e-10,
            check_every=500,
            verbose=False,
        )
        _, ux, uy = case_macro(case, f_ref)
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, ux=ux, uy=uy)
        return ux, uy
    return stored_picard_reference(case_id, n)


def velocity_error(case, f, ref):
    _, ux, uy = case_macro(case, f)
    ux_ref, uy_ref = ref
    mask = (case.chi > 0) if hasattr(case, "chi") else np.ones_like(ux, dtype=bool)
    du = ux[mask] - ux_ref[mask]
    dv = uy[mask] - uy_ref[mask]
    den = max(float(np.sqrt(np.sum(ux_ref[mask] ** 2 + uy_ref[mask] ** 2))), 1.0e-30)
    return {
        "rel_L2": float(np.sqrt(np.sum(du * du + dv * dv)) / den),
        "Linf": float(max(np.max(np.abs(du)) if du.size else 0.0, np.max(np.abs(dv)) if dv.size else 0.0)),
    }


def load_fixed_baselines():
    payload = json.loads(BASELINE_CACHE.read_text(encoding="utf-8"))
    out = {}
    for case_id, data in payload.items():
        rows = [r for r in data["rows"] if r["method"] in FIXED_METHODS and r.get("converged")]
        if not rows:
            raise RuntimeError(f"no converged fixed baselines for {case_id}")
        accuracy_rows = rows
        if case_id in {"cavity_re100", "multi_cylinder"}:
            non_reference = [
                r for r in rows
                if not (r["method"] == "picard_lbm" and float(r.get("rel_L2", 1.0)) < 1.0e-12)
            ]
            if non_reference:
                accuracy_rows = non_reference
        out[case_id] = {
            "rows": rows,
            "best_lbe": min(float(r["total_lbe"]) for r in rows),
            "best_wall": min(float(r["wall_s"]) for r in rows),
            "best_rel_L2": min(float(r["rel_L2"]) for r in accuracy_rows),
            "best_Linf": min(float(r["Linf"]) for r in accuracy_rows),
        }
    return out


def make_eval_case(case_id, n):
    if case_id == "multi_cylinder":
        from lbm_voxel import VoxelCase
        return VoxelCase(make_multi_cylinder_mask(n), nu=0.05, F0=2.0e-4, kf=1)
    return make_case(case_id, n)[0]


def run_proposed(case_id, n, ref):
    case = make_eval_case(case_id, n)
    try:
        case.lbe_step(case.initial_field())
    except Exception:
        pass
    t0 = time.perf_counter()
    f, hist = solve_proposed_single(case, tol=1.0e-7, verbose=False)
    accel_lbe = int(hist[-1][2]) if hist else 0
    finite = bool(np.all(np.isfinite(f)))
    if finite:
        _, native_residual_pre = _residual_norm(case, f)
    else:
        native_residual_pre = float("nan")

    if finite and np.isfinite(native_residual_pre) and native_residual_pre < 1.0e-7:
        tail_lbe = 0
        tail_change = 0.0
    elif finite:
        f, tail_hist, tail_change = picard_tail(
            case,
            f,
            max_steps=200000,
            tol=1.0e-6,
            check_every=200,
        )
        tail_lbe = int(tail_hist[-1][2]) if tail_hist else 0
    else:
        tail_lbe = 0
        tail_change = float("nan")
    wall = time.perf_counter() - t0
    finite = bool(np.all(np.isfinite(f)))
    if finite:
        _, native_residual = _residual_norm(case, f)
        err = velocity_error(case, f, ref)
    else:
        native_residual = float("nan")
        err = {"rel_L2": float("inf"), "Linf": float("inf")}
    converged = bool(
        finite
        and (
            (np.isfinite(native_residual) and native_residual < 1.0e-7)
            or (np.isfinite(tail_change) and tail_change < 1.0e-6)
        )
    )
    return {
        "case_id": case_id,
        "N": n,
        "method": "proposed_single_strict",
        "accel_lbe": accel_lbe,
        "tail_lbe": tail_lbe,
        "total_lbe": accel_lbe + tail_lbe,
        "wall_s": float(wall),
        "tail_vchg": float(tail_change),
        "native_residual": float(native_residual),
        "converged": converged,
        **err,
    }


def fail_metrics(reason, anti):
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = {
        "score": -1000.0,
        "all_pass": 0,
        "case_count": len(CASES),
        "row_count": 0,
        "pass_count": 0,
        "lbe_win_count": 0,
        "wall_win_count": 0,
        "accuracy_win_count": 0,
        "mean_lbe_ratio_vs_best_fixed": 0.0,
        "mean_wall_ratio_vs_best_fixed": 0.0,
        "max_rel_l2_excess_vs_best_fixed": float("inf"),
        "failure_reason": reason,
        **anti,
    }
    (OUT / "latest_rows.json").write_text("[]", encoding="utf-8")
    (OUT / "latest_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


def main():
    anti = anti_cheat_report()
    if not anti["anti_cheat_pass"]:
        fail_metrics("forbidden analytic-target solver path detected", anti)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    fixed = load_fixed_baselines()
    rows = []
    pass_count = 0
    lbe_win_count = 0
    wall_win_count = 0
    accuracy_win_count = 0
    for case_id, n in CASES:
        case_for_ref = make_eval_case(case_id, n)
        ref = reference_for(case_id, n, case_for_ref)
        row = run_proposed(case_id, n, ref)
        base = fixed[case_id]
        row["fixed_best_lbe"] = base["best_lbe"]
        row["fixed_best_wall"] = base["best_wall"]
        row["fixed_best_rel_L2"] = base["best_rel_L2"]
        row["fixed_best_Linf"] = base["best_Linf"]
        row["lbe_win"] = bool(row["total_lbe"] < base["best_lbe"])
        row["wall_win"] = bool(row["wall_s"] < base["best_wall"])
        rel_limit = base["best_rel_L2"] * (1.0 + ACCURACY_REL_TOL) + ACCURACY_ABS_TOL
        linf_limit = base["best_Linf"] * (1.0 + ACCURACY_REL_TOL) + ACCURACY_ABS_TOL
        row["accuracy_win"] = bool(row["rel_L2"] <= rel_limit and row["Linf"] <= linf_limit)
        row["case_pass"] = bool(row["converged"] and row["lbe_win"] and row["wall_win"] and row["accuracy_win"])
        lbe_win_count += int(row["lbe_win"])
        wall_win_count += int(row["wall_win"])
        accuracy_win_count += int(row["accuracy_win"])
        pass_count += int(row["case_pass"])
        rows.append(row)
        print(
            f"{case_id:16s} strict lbe={row['total_lbe']:7d}/{base['best_lbe']:7.0f} "
            f"wall={row['wall_s']:.3f}/{base['best_wall']:.3f} "
            f"relL2={row['rel_L2']:.3e}/{base['best_rel_L2']:.3e} "
            f"res={row['native_residual']:.3e} pass={row['case_pass']}",
            flush=True,
        )
    mean_lbe_ratio = float(np.mean([r["fixed_best_lbe"] / max(r["total_lbe"], 1) for r in rows]))
    mean_wall_ratio = float(np.mean([r["fixed_best_wall"] / max(r["wall_s"], 1.0e-12) for r in rows]))
    max_rel_l2_excess = float(max(r["rel_L2"] / max(r["fixed_best_rel_L2"], 1.0e-30) for r in rows))
    all_pass = int(pass_count == len(CASES))
    score = (
        1000.0 * all_pass
        + 100.0 * pass_count
        + 10.0 * lbe_win_count
        + 10.0 * wall_win_count
        + 10.0 * accuracy_win_count
        + mean_lbe_ratio
        + mean_wall_ratio
        - max(0.0, math.log10(max_rel_l2_excess)) * 5.0
    )
    metrics = {
        "score": float(score),
        "all_pass": all_pass,
        "case_count": len(CASES),
        "row_count": len(rows),
        "pass_count": pass_count,
        "lbe_win_count": lbe_win_count,
        "wall_win_count": wall_win_count,
        "accuracy_win_count": accuracy_win_count,
        "mean_lbe_ratio_vs_best_fixed": mean_lbe_ratio,
        "mean_wall_ratio_vs_best_fixed": mean_wall_ratio,
        "max_rel_l2_excess_vs_best_fixed": max_rel_l2_excess,
        **anti,
    }
    (OUT / "latest_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (OUT / "latest_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
