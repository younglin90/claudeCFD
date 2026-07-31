"""Run force-free benchmark comparisons with the same solver interface.

This module intentionally excludes force-driven Kolmogorov and any forcing-
dependent baseline variants. It is a clean starting point for a no-force
verification protocol.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from solver_baseline import solve_baseline
from solver_proposed_single import solve_proposed_single
from solver_unified_safe_nn import _residual_norm

from no_force_suite.no_force_cases import SUPPORTED_CASES, make_case, supported_case_ids


OUT = Path("paper_revision_data") / "no_force_suite"
REF_DIR = OUT / "refs"


def _macro(case, f):
    if hasattr(case, "macro"):
        return case.macro(f)
    if hasattr(case, "project"):
        rho, rhoux, rhouy = case.project(f)
        rho_safe = np.where(rho < 1.0e-12, 1.0, rho)
        return rho, rhoux / rho_safe, rhouy / rho_safe
    raise RuntimeError("case has no macro method")


def _baseline_reference(case_id: str, case, tol: float):
    ref_path = REF_DIR / f"{case_id}_no_force_ref.npz"
    if ref_path.exists():
        data = np.load(ref_path, allow_pickle=False)
        ref_meta = {
            "lbe_calls": int(data["lbe_calls"]),
            "wall": float(data["wall"]),
            "residual": float(data["residual"]),
        }
        return data["f"], ref_meta

    check_every = 200
    max_steps = 70000
    t0 = time.perf_counter()
    f_ref, hist = solve_baseline(case, max_steps=max_steps, tol=tol, check_every=check_every, verbose=False)
    wall = time.perf_counter() - t0

    ref_meta = {
        "lbe_calls": int(hist[-1][2]) if hist else 0,
        "wall": float(wall),
        "residual": float(hist[-1][1]) if hist else float("inf"),
    }
    REF_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = ref_path.with_suffix(".npz.tmp.npz")
    np.savez_compressed(
        tmp_path,
        f=f_ref,
        lbe_calls=ref_meta["lbe_calls"],
        wall=ref_meta["wall"],
        residual=ref_meta["residual"],
    )
    tmp_path.replace(ref_path)
    return f_ref, ref_meta


def _velocity_error(case, f, f_ref):
    _, ux, uy = _macro(case, f)
    _, ux_ref, uy_ref = _macro(case, f_ref)
    if hasattr(case, "chi"):
        mask = case.chi > 0.0
    else:
        mask = np.ones_like(ux, dtype=bool)
    du = ux[mask] - ux_ref[mask]
    dv = uy[mask] - uy_ref[mask]
    mag_ref = np.sqrt(ux_ref[mask] ** 2 + uy_ref[mask] ** 2)
    den = max(float(np.sqrt(np.sum(mag_ref * mag_ref))), 1.0e-30)
    du_inf = float(np.max(np.abs(du))) if du.size else 0.0
    dv_inf = float(np.max(np.abs(dv))) if dv.size else 0.0
    return {
        "rel_L2": float(np.sqrt(np.sum(du * du + dv * dv)) / den),
        "Linf": float(max(du_inf, dv_inf)),
    }


def _run_case(case_id: str):
    label, n, tol, _ = SUPPORTED_CASES[case_id]
    case = make_case(case_id)

    f_ref, ref_meta = _baseline_reference(case_id, case, tol)
    _, ref_res = _residual_norm(case, f_ref)
    ref_meta["residual"] = float(ref_res)

    t0 = time.perf_counter()
    f_prop, hist_prop = solve_proposed_single(case, tol=tol, verbose=False)
    wall_prop = time.perf_counter() - t0

    _, prop_res = _residual_norm(case, f_prop)
    prop_res = float(prop_res)
    err = _velocity_error(case, f_prop, f_ref)

    lbe_prop = int(hist_prop[-1][2]) if hist_prop else 0

    return {
        "case_id": case_id,
        "label": label,
        "N": int(n),
        "tol": float(tol),
        "reference_lbe": int(ref_meta["lbe_calls"]),
        "reference_wall": float(ref_meta["wall"]),
        "reference_residual": float(ref_meta["residual"]),
        "proposed_lbe": lbe_prop,
        "proposed_wall": float(wall_prop),
        "proposed_residual": prop_res,
        "converged": bool(np.isfinite(prop_res) and prop_res <= max(2.0 * tol, 1.0e-6)),
        "lbe_speedup": float(ref_meta["lbe_calls"] / lbe_prop) if lbe_prop > 0 and ref_meta["lbe_calls"] > 0 else None,
        "wall_speedup": float(ref_meta["wall"] / wall_prop) if wall_prop > 0 and ref_meta["wall"] > 0 else None,
        **err,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_id in supported_case_ids():
        rows.append(_run_case(case_id))

    pass_count = sum(1 for r in rows if r["converged"])
    pass_ratio = pass_count / max(len(rows), 1)
    metrics = {
        "case_count": len(rows),
        "pass_count": pass_count,
        "pass_ratio": float(pass_ratio),
        "all_pass": int(pass_count == len(rows)),
    }

    (OUT / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
