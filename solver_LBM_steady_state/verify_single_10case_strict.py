"""Extended strict verifier for the single proposed LBM solver.

Validation set:
  1. Kolmogorov N=32
  2. Plane Poiseuille channel N=32
  3. Couette N=32
  4. Lid-driven cavity Re=100 N=33
  5. Lid-driven cavity Re=400 N=49
  6. Lid-driven cavity Re=1000 N=129
  7. Multi-cylinder voxel N=32
  8. Backward-facing step N=64
  9. Cylinder wake analogue N=64
 10. T-junction N=64

This verifier intentionally measures the proposed method against native
residuals and cached Picard/analytic references.  It is an autoresearch
surface, not a claim that every case is already paper-ready.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from lbm_core import LBMCavity, moments as cavity_moments
from lbm_voxel import VoxelCase
from paper_60case_benchmark import (
    MULTI_CYLINDER_GEOMETRY_VERSION,
    backward_step_mask,
    cylinder_wake_mask,
    make_multi_cylinder_mask,
    make_t_junction_case,
)
from paper_extra_benchmarks import solve_baseline_generic
from solver_baseline import solve_baseline
from solver_proposed_single import solve_proposed_single
from solver_scmk import solve_baseline_periodic
from solver_unified_safe_nn import _residual_norm
from v2_run_all import make_case


OUT = Path("paper_revision_data") / "single10_strict"
REF_DIR = OUT / "refs"
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

CASES = [
    ("kolmogorov_n32", "kolmogorov", 32, 1.0e-7),
    ("channel_n32", "channel", 32, 1.0e-7),
    ("couette_n32", "couette", 32, 1.0e-7),
    ("cavity_re100_n33", "cavity_re100", 33, 5.0e-7),
    ("cavity_re400_n49", "cavity_re400", 49, 5.0e-7),
    ("cavity_re1000_n129", "cavity_re1000", 129, 5.0e-7),
    ("multi_cylinder_n32", "multi_cylinder", 32, 1.0e-7),
    ("backward_step_n64", "backward_step", 64, 1.0e-7),
    ("cylinder_wake_n64", "cylinder_wake", 64, 1.0e-7),
    ("t_junction_n64", "t_junction", 64, 1.0e-7),
]


def anti_cheat_report():
    text = SOLVER_PATH.read_text(encoding="utf-8")
    hits = [token for token in FORBIDDEN_SOLVER_TOKENS if token in text]
    return {"anti_cheat_pass": int(not hits), "anti_cheat_hits": hits}


def make_eval_case(kind: str, n: int):
    if kind == "multi_cylinder":
        return VoxelCase(make_multi_cylinder_mask(n), nu=0.05, F0=2.0e-4, kf=1)
    if kind == "backward_step":
        return VoxelCase(backward_step_mask(n), nu=0.05, F0=1.5e-5, kf=0)
    if kind == "cylinder_wake":
        return VoxelCase(cylinder_wake_mask(n), nu=0.04, F0=1.0e-5, kf=0)
    if kind == "t_junction":
        return make_t_junction_case(n)
    return make_case(kind, n)[0]


def macro(case, f):
    if hasattr(case, "macro"):
        return case.macro(f)
    return cavity_moments(f)


def analytic_reference(kind: str, case):
    if not hasattr(case, "analytical_ux"):
        return None
    ux = case.analytical_ux()
    if ux.ndim == 1:
        if kind == "channel":
            ux = np.tile(ux[:, None], (1, case.N))
        else:
            ux = np.tile(ux[None, :], (case.N, 1))
    return ux, np.zeros_like(ux), {"lbe": 0, "wall": 0.0, "residual": 0.0}


def baseline_runner(case):
    if isinstance(case, LBMCavity):
        return solve_baseline
    if isinstance(case, VoxelCase):
        return solve_baseline_generic
    return solve_baseline_periodic


def max_steps_for(case_id: str):
    if case_id == "cavity_re1000_n129":
        return 900000
    if "cavity" in case_id:
        return 300000
    if "t_junction" in case_id:
        return 120000
    return 200000


def reference_cache_path(case_id: str, n: int):
    version = MULTI_CYLINDER_GEOMETRY_VERSION if "multi_cylinder" in case_id else "v1"
    return REF_DIR / f"{case_id}_{version}_N{n}_picard_ref.npz"


def reference_for(case_id: str, kind: str, n: int, tol: float, case):
    ref = analytic_reference(kind, case)
    if ref is not None:
        return ref
    path = reference_cache_path(case_id, n)
    if path.exists():
        data = np.load(path)
        return (
            data["ux"],
            data["uy"],
            {
                "lbe": int(data["lbe"]),
                "wall": float(data["wall"]),
                "residual": float(data["residual"]),
            },
        )
    runner = baseline_runner(case)
    check_every = 50 if case_id == "cavity_re1000_n129" else (500 if n >= 64 else 200)
    t0 = time.perf_counter()
    f_ref, hist = runner(
        case,
        max_steps=max_steps_for(case_id),
        tol=tol,
        check_every=check_every,
        verbose=False,
    )
    wall = time.perf_counter() - t0
    _, ux, uy = macro(case, f_ref)
    residual = float(_residual_norm(case, f_ref)[1])
    lbe = int(hist[-1][2]) if hist else 0
    REF_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ux=ux, uy=uy, lbe=lbe, wall=wall, residual=residual)
    return ux, uy, {"lbe": lbe, "wall": wall, "residual": residual}


def velocity_error(case, f, ux_ref, uy_ref):
    _, ux, uy = macro(case, f)
    mask = (case.chi > 0) if hasattr(case, "chi") else np.ones_like(ux, dtype=bool)
    du = ux[mask] - ux_ref[mask]
    dv = uy[mask] - uy_ref[mask]
    den = max(float(np.sqrt(np.sum(ux_ref[mask] ** 2 + uy_ref[mask] ** 2))), 1.0e-30)
    return {
        "rel_L2": float(np.sqrt(np.sum(du * du + dv * dv)) / den),
        "Linf": float(max(np.max(np.abs(du)) if du.size else 0.0, np.max(np.abs(dv)) if dv.size else 0.0)),
    }


def case_accuracy_limit(case_id: str):
    if case_id in {"kolmogorov_n32", "channel_n32", "couette_n32"}:
        return 1.0e-2
    if "cavity" in case_id:
        return 5.0e-2
    if "t_junction" in case_id:
        return 5.0e-2
    if "backward_step" in case_id:
        return 2.5e-2
    return 1.0e-2


def run_case(case_id: str, kind: str, n: int, tol: float):
    case = make_eval_case(kind, n)
    ux_ref, uy_ref, ref_meta = reference_for(case_id, kind, n, tol, case)
    t0 = time.perf_counter()
    f, hist = solve_proposed_single(case, tol=tol, verbose=False)
    wall = time.perf_counter() - t0
    finite = bool(np.all(np.isfinite(f)))
    residual = float(_residual_norm(case, f)[1]) if finite else float("inf")
    err = velocity_error(case, f, ux_ref, uy_ref) if finite else {"rel_L2": float("inf"), "Linf": float("inf")}
    lbe = int(hist[-1][2]) if hist else 0
    ref_lbe = int(ref_meta["lbe"])
    ref_wall = float(ref_meta["wall"])
    converged = finite and np.isfinite(residual) and residual <= max(2.0 * tol, 1.0e-6)
    accurate = err["rel_L2"] <= case_accuracy_limit(case_id)
    speed_ok = ref_lbe == 0 or lbe <= ref_lbe or wall <= ref_wall
    return {
        "case_id": case_id,
        "kind": kind,
        "N": n,
        "tol": tol,
        "proposed_lbe": lbe,
        "proposed_wall": float(wall),
        "proposed_residual": residual,
        "reference_lbe": ref_lbe,
        "reference_wall": ref_wall,
        "reference_residual": float(ref_meta["residual"]),
        "lbe_speedup_vs_picard_ref": float(ref_lbe / lbe) if lbe > 0 and ref_lbe > 0 else None,
        "wall_speedup_vs_picard_ref": float(ref_wall / wall) if wall > 0 and ref_wall > 0 else None,
        "converged": bool(converged),
        "accurate": bool(accurate),
        "speed_ok": bool(speed_ok),
        "case_pass": bool(converged and accurate and speed_ok),
        **err,
    }


def main():
    anti = anti_cheat_report()
    OUT.mkdir(parents=True, exist_ok=True)
    if not anti["anti_cheat_pass"]:
        metrics = {
            "score": -1000.0,
            "all_pass": 0,
            "case_count": len(CASES),
            "pass_count": 0,
            "anti_cheat_pass": 0,
            "anti_cheat_hits": anti["anti_cheat_hits"],
        }
        print(json.dumps(metrics, sort_keys=True))
        return

    rows = [run_case(*spec) for spec in CASES]
    pass_count = sum(1 for r in rows if r["case_pass"])
    converged_count = sum(1 for r in rows if r["converged"])
    accurate_count = sum(1 for r in rows if r["accurate"])
    speed_count = sum(1 for r in rows if r["speed_ok"])
    speedups = [r["lbe_speedup_vs_picard_ref"] for r in rows if r["lbe_speedup_vs_picard_ref"] is not None]
    wall_speedups = [r["wall_speedup_vs_picard_ref"] for r in rows if r["wall_speedup_vs_picard_ref"] is not None]
    worst_rel = max(float(r["rel_L2"]) for r in rows)
    score = (
        100.0 * pass_count
        + 20.0 * converged_count
        + 10.0 * accurate_count
        + 5.0 * speed_count
        + 10.0 * (sum(speedups) / max(len(speedups), 1))
        + 5.0 * (sum(wall_speedups) / max(len(wall_speedups), 1))
        - 50.0 * math.log10(max(worst_rel, 1.0e-12) / 1.0e-4)
    )
    metrics = {
        "score": float(score),
        "all_pass": int(pass_count == len(CASES)),
        "case_count": len(CASES),
        "pass_count": pass_count,
        "converged_count": converged_count,
        "accurate_count": accurate_count,
        "speed_ok_count": speed_count,
        "worst_rel_L2": float(worst_rel),
        "mean_lbe_speedup_vs_picard_ref": float(sum(speedups) / max(len(speedups), 1)),
        "mean_wall_speedup_vs_picard_ref": float(sum(wall_speedups) / max(len(wall_speedups), 1)),
        "anti_cheat_pass": 1,
        "anti_cheat_hits": [],
    }
    (OUT / "latest_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (OUT / "latest_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
