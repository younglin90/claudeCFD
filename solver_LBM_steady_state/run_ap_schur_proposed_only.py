"""Run only the upgraded proposed solver with AP-Schur JFNK candidates.

This runner intentionally does not recompute fixed reference methods.  It is
for proposal-only revalidation after the benchmark reference pool already
exists from previous rounds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from paper_60case_benchmark import velocity_error
from paper_60case_benchmark_no_force import macro_of, write_history_csv
from paper_60case_benchmark_no_force_scaling import (
    CASE_IDS,
    _analytic_channel_reference,
    _analytic_couette_reference,
    _case_grid_shape,
    _f_rms_residual_value,
    _fluid_mask,
    _ghia_centerline_error,
    _load_tight_reference,
    _macro_l2_residual_components,
    _macro_l2_residual_value,
    _resample_reference_field,
    _tol_scale,
    _relative_macro_l2_convergence_tail,
    _unified_macro_l2_convergence_audit,
    case_factory_scaled,
    max_steps_for_scaled,
    write_diagnostic_csv,
)
from solver_proposed_single import (
    _cfg_float,
    _cfg_int,
    _fast_norm,
    _jvp_native,
    _native_residual,
    _picard_sweep,
    _state_is_admissible,
    solve_proposed_single,
)


def _parse_csv(value: str, allowed):
    if value == "all":
        return list(allowed)
    out = [x.strip() for x in value.split(",") if x.strip()]
    bad = [x for x in out if x not in allowed]
    if bad:
        raise ValueError(f"unknown values: {bad}")
    return out


def _accuracy_against_available_reference(base_case_id, level, case, f):
    if base_case_id in {"channel_n32", "channel_poiseuille_rect"}:
        ref_f = _analytic_channel_reference(case)
        err = velocity_error(case, ref_f, case, f, fluid_mask=_fluid_mask(case))
        return "analytic_poiseuille", err
    if base_case_id == "couette_n32":
        ref_f = _analytic_couette_reference(case)
        err = velocity_error(case, ref_f, case, f, fluid_mask=_fluid_mask(case))
        return "analytic_couette", err
    if base_case_id in {"cavity_re100_n33", "cavity_re400_n49", "cavity_re1000_n129"}:
        err = _ghia_centerline_error(case, f, case, f)
        return "ghia_centerline", err
    if base_case_id in {"multi_cylinder_n32", "backward_step_n64", "cylinder_wake_n64", "t_junction_rect"}:
        try:
            _case_id, _label, _tol, factory = case_factory_scaled(base_case_id, int(level))
            ref = _load_tight_reference(base_case_id, factory)
            if ref is not None and ref[0] is not None:
                ref_case, ref_f, _ref_path = ref
                ref_fit = _resample_reference_field(ref_case, ref_f, case)
                err = velocity_error(case, ref_fit, case, f, fluid_mask=_fluid_mask(case))
                return "tight_ref", err
        except Exception:
            pass
        if base_case_id == "t_junction_rect" and int(level) == 1:
            try:
                ref_path = (
                    Path(__file__).resolve().parent
                    / "paper_channel_n32_results"
                    / "_coord_round103_min_tjunction_1x2x"
                    / "t_junction_rect"
                    / "npz_cache"
                    / "t_junction_Nx96_Ny64_W16__1x__picard_lbm__5fd5add6db4e.npz"
                )
                if ref_path.exists():
                    ref_f = np.load(ref_path, allow_pickle=False)["f"]
                    if tuple(ref_f.shape[-2:]) == tuple(_case_grid_shape(case)):
                        err = velocity_error(case, ref_f, case, f, fluid_mask=_fluid_mask(case))
                        return "picard_ref_min_tjunction_1x", err
            except Exception:
                pass
    return "not_computed_proposed_only", {
        "rel_l2": float("nan"),
        "linf": float("nan"),
        "rms": float("nan"),
        "vel_abs_l2": float("nan"),
        "vel_abs_linf": float("nan"),
        "vel_abs_rms": float("nan"),
    }


def _history_final_row(case, f, hist, wall):
    final_res = float(_macro_l2_residual_value(case, f))
    lbe = int(hist[-1][2]) if hist else 0
    if hist and abs(float(hist[-1][1]) - final_res) <= max(1.0e-14, 1.0e-8 * max(abs(final_res), 1.0)):
        return hist
    hist.append((len(hist), final_res, lbe + 1, float(wall)))
    return hist


def _diagnostic_counts(hist):
    diags = getattr(hist, "diagnostics", []) or []
    ap = [d for d in diags if "ap_schur" in str(d.get("phase", ""))]
    return len(ap), sum(int(d.get("accepted", 0)) for d in ap)


def _has_accepted_phase(hist, phase: str) -> bool:
    for row in getattr(hist, "diagnostics", []) or []:
        if str(row.get("phase", "")) == phase and int(row.get("accepted", 0) or 0):
            return True
    return False


def _has_phase(hist, phase: str) -> bool:
    for row in getattr(hist, "diagnostics", []) or []:
        if str(row.get("phase", "")) == phase:
            return True
    return False


def _require_plateau_convergence() -> bool:
    return os.environ.get("SAFE_NN_REQUIRE_PLATEAU_CONVERGENCE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _strict_convergence_flags(base_id, hist, final_res, tol):
    rel_stats = getattr(hist, "relative_macro_l2_stats", {}) or {}
    absolute_converged = bool(np.isfinite(final_res) and final_res < 5.0 * float(tol))
    if not _require_plateau_convergence():
        return absolute_converged, absolute_converged, False, "macro_l2_final_threshold", rel_stats

    plateau_converged = bool(
        (
            int(rel_stats.get("relative_plateau", 0) or 0)
            or int(rel_stats.get("relative_floor_pass", 0) or 0)
        )
        and int(rel_stats.get("min_lbe_pass", 0) or 0)
    )
    converged = bool(absolute_converged and plateau_converged)
    return (
        converged,
        absolute_converged,
        plateau_converged,
        "macro_l2_final_threshold_and_relative_plateau",
        rel_stats,
    )


def _relative_tail_for_quick_screen(base_id, level, case, f, hist, t0, tol=None):
    max_steps = int(max_steps_for_scaled(base_id, level))
    if not hasattr(case, "chi"):
        old_abs = os.environ.get("SAFE_NN_RELATIVE_MACRO_ABS_THRESHOLD")
        if tol is not None and _require_plateau_convergence():
            os.environ["SAFE_NN_RELATIVE_MACRO_ABS_THRESHOLD"] = str(5.0 * float(tol))
        try:
            return _relative_macro_l2_convergence_tail("proposed", case, f, hist, t0, max_steps=max_steps)
        finally:
            if old_abs is None:
                os.environ.pop("SAFE_NN_RELATIVE_MACRO_ABS_THRESHOLD", None)
            else:
                os.environ["SAFE_NN_RELATIVE_MACRO_ABS_THRESHOLD"] = old_abs

    old = os.environ.get("SAFE_NN_RELATIVE_MACRO_MAX_LBE")
    old_abs = os.environ.get("SAFE_NN_RELATIVE_MACRO_ABS_THRESHOLD")
    if old is None or not _require_plateau_convergence():
        os.environ["SAFE_NN_RELATIVE_MACRO_MAX_LBE"] = str(max_steps)
    if tol is not None and _require_plateau_convergence():
        os.environ["SAFE_NN_RELATIVE_MACRO_ABS_THRESHOLD"] = str(5.0 * float(tol))
    try:
        return _relative_macro_l2_convergence_tail("proposed", case, f, hist, t0, max_steps=max_steps)
    finally:
        if old is None:
            os.environ.pop("SAFE_NN_RELATIVE_MACRO_MAX_LBE", None)
        else:
            os.environ["SAFE_NN_RELATIVE_MACRO_MAX_LBE"] = old
        if old_abs is None:
            os.environ.pop("SAFE_NN_RELATIVE_MACRO_ABS_THRESHOLD", None)
        else:
            os.environ["SAFE_NN_RELATIVE_MACRO_ABS_THRESHOLD"] = old_abs


def _post_tail_block_rre(case, f, hist, t0, tol):
    if hasattr(case, "Re") and hasattr(case, "U_wall") and not hasattr(case, "chi"):
        if float(getattr(case, "Re", 0.0)) > 600.0:
            return f, hist
        block = 1024
    elif hasattr(case, "chi"):
        block = 128
    else:
        return f, hist

    best_res = float(_macro_l2_residual_value(case, f))
    if not np.isfinite(best_res) or best_res <= 5.0 * float(tol):
        return f, hist

    states = [np.array(f, copy=True)]
    cur = np.array(f, copy=True)
    for _ in range(4):
        cur = _picard_sweep(case, cur, block)
        states.append(np.array(cur, copy=True))

    residuals = [states[i + 1] - states[i] for i in range(len(states) - 1)]
    n_m = len(residuals) - 1
    try:
        dR = np.stack([residuals[i + 1] - residuals[i] for i in range(n_m)], axis=-1).reshape(-1, n_m)
        dG = np.stack([states[i + 2] - states[i + 1] for i in range(n_m)], axis=-1).reshape(-1, n_m)
        gram = dR.T @ dR
        rhs = dR.T @ residuals[-1].ravel()
        reg = 1.0e-12 * max(float(np.trace(gram)) / max(n_m, 1), 1.0)
        gamma = np.linalg.solve(gram + reg * np.eye(n_m), rhs)
        candidate = (states[-1].ravel() - dG @ gamma).reshape(case.shape)
    except Exception:
        return f, hist

    if not _state_is_admissible(case, candidate):
        return f, hist
    rn = float(_macro_l2_residual_value(case, candidate))
    used_lbe = 4 * block + 1
    if not (np.isfinite(rn) and rn < best_res):
        return f, hist

    lbe = (int(hist[-1][2]) if hist else 0) + used_lbe
    wall_now = max(time.perf_counter() - t0, 1.0e-6)
    hist.append((len(hist), rn, lbe, wall_now))
    return candidate, hist


def _uniform_post_tail_rre_settle(case, f, hist, t0):
    best_f = np.array(f, copy=True)
    best_res = float(_macro_l2_residual_value(case, best_f))
    lbe0 = int(hist[-1][2]) if hist else 0
    best_lbe = lbe0
    scale = max(float(getattr(case, "N", 0)) / 32.0, 1.0)
    base = int(np.clip(round(256.0 * scale), 256, 2048))
    blocks = sorted({max(128, base // 2), base, min(4096, 2 * base)})
    local_lbe = 0
    for block in blocks:
        for depth in (3, 4, 5):
            states = [np.array(f, copy=True)]
            state = np.array(f, copy=True)
            for _ in range(depth):
                state = _picard_sweep(case, state, block)
                states.append(np.array(state, copy=True))
            local_lbe += block * depth
            residuals = [states[i + 1] - states[i] for i in range(len(states) - 1)]
            n_m = len(residuals) - 1
            try:
                dR = np.stack([residuals[i + 1] - residuals[i] for i in range(n_m)], axis=-1).reshape(-1, n_m)
                dG = np.stack([states[i + 2] - states[i + 1] for i in range(n_m)], axis=-1).reshape(-1, n_m)
                gram = dR.T @ dR
                rhs = dR.T @ residuals[-1].ravel()
                reg = 1.0e-12 * max(float(np.trace(gram)) / max(n_m, 1), 1.0)
                gamma = np.linalg.solve(gram + reg * np.eye(n_m), rhs)
                candidate = (states[-1].ravel() - dG @ gamma).reshape(case.shape)
            except Exception:
                continue
            if not np.all(np.isfinite(candidate)):
                continue
            settle_steps = min(1024, max(64, block))
            candidate = _picard_sweep(case, candidate, settle_steps)
            local_lbe += settle_steps
            if not _state_is_admissible(case, candidate):
                continue
            rn = float(_macro_l2_residual_value(case, candidate))
            local_lbe += 1
            if np.isfinite(rn) and rn < best_res:
                best_res = rn
                best_f = np.array(candidate, copy=True)
                best_lbe = lbe0 + local_lbe
    if best_res < float(_macro_l2_residual_value(case, f)):
        hist.append((len(hist), best_res, best_lbe, max(time.perf_counter() - t0, 1.0e-6)))
        return best_f, hist
    return f, hist


def _uniform_native_jfnk_polish(case, f, hist, t0, tol, require_threshold=False):
    try:
        from scipy.sparse.linalg import LinearOperator, gmres
    except Exception:
        return f, hist
    base_res = float(_macro_l2_residual_value(case, f))
    if not np.isfinite(base_res) or base_res <= 5.0 * float(tol):
        return f, hist
    try:
        r_base = _native_residual(case, f)
        norm_f = _fast_norm(case, f)
        probes = [0]

        def matvec(v_flat):
            probes[0] += 1
            return _jvp_native(case, v_flat.reshape(case.shape), f, r_base, norm_f).ravel()

        op = LinearOperator((f.size, f.size), matvec=matvec, dtype=np.float64)
        rhs = -r_base.ravel()
        df, info = gmres(
            op,
            rhs,
            rtol=_cfg_float("SAFE_NN_UNIFORM_JFNK_RTOL", 1.0e-2),
            atol=_cfg_float("SAFE_NN_UNIFORM_JFNK_RTOL", 1.0e-2) * np.linalg.norm(rhs) * 1.0e-3,
            maxiter=1,
            restart=max(2, _cfg_int("SAFE_NN_UNIFORM_JFNK_RESTART", 4)),
        )
    except Exception:
        return f, hist
    if info < 0 or not np.all(np.isfinite(df)):
        return f, hist
    best_f = np.array(f, copy=True)
    best_res = base_res
    best_lbe = (int(hist[-1][2]) if hist else 0) + max(1, probes[0])
    df = df.reshape(case.shape)
    for alpha in (1.0, 0.5, 0.25, 0.125):
        candidate = f + alpha * df
        if not _state_is_admissible(case, candidate):
            continue
        for settle_steps in (0, 16, 64):
            settled = candidate if settle_steps == 0 else _picard_sweep(case, candidate, settle_steps)
            rn = float(_macro_l2_residual_value(case, settled))
            if np.isfinite(rn) and rn < best_res:
                best_res = rn
                best_f = np.array(settled, copy=True)
                best_lbe += settle_steps + 1
    if best_res < base_res and ((not require_threshold) or best_res <= 5.0 * float(tol)):
        hist.append((len(hist), best_res, best_lbe, max(time.perf_counter() - t0, 1.0e-6)))
        return best_f, hist
    return f, hist


def _uniform_relative_floor_jfnk_polish(case, f, hist, t0):
    if not _require_plateau_convergence():
        return f, hist
    try:
        from scipy.sparse.linalg import LinearOperator, gmres
    except Exception:
        return f, hist

    initial_res = max(float(_macro_l2_residual_value(case, case.initial_field())), 1.0e-300)
    target_res = initial_res * _cfg_float("SAFE_NN_RELATIVE_MACRO_FLOOR", 1.0e-10)
    base_res = float(_macro_l2_residual_value(case, f))
    if not np.isfinite(base_res) or base_res <= target_res:
        return f, hist
    try:
        r_base = _native_residual(case, f)
        norm_f = _fast_norm(case, f)
        probes = [0]

        def matvec(v_flat):
            probes[0] += 1
            return _jvp_native(case, v_flat.reshape(case.shape), f, r_base, norm_f).ravel()

        op = LinearOperator((f.size, f.size), matvec=matvec, dtype=np.float64)
        rhs = -r_base.ravel()
        df, info = gmres(
            op,
            rhs,
            rtol=_cfg_float("SAFE_NN_UNIFORM_JFNK_RTOL", 1.0e-2),
            atol=_cfg_float("SAFE_NN_UNIFORM_JFNK_RTOL", 1.0e-2) * np.linalg.norm(rhs) * 1.0e-3,
            maxiter=1,
            restart=max(2, _cfg_int("SAFE_NN_UNIFORM_JFNK_RESTART", 4)),
        )
    except Exception:
        return f, hist
    if info < 0 or not np.all(np.isfinite(df)):
        return f, hist

    best_f = np.array(f, copy=True)
    best_res = base_res
    best_lbe = (int(hist[-1][2]) if hist else 0) + max(1, probes[0])
    df = df.reshape(case.shape)
    for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
        candidate = f + alpha * df
        if not _state_is_admissible(case, candidate) or not _open_masked_state_sane(case, candidate):
            continue
        for settle_steps in (0, 8, 32, 128):
            settled = candidate if settle_steps == 0 else _picard_sweep(case, candidate, settle_steps)
            if not _state_is_admissible(case, settled) or not _open_masked_state_sane(case, settled):
                continue
            rn = float(_macro_l2_residual_value(case, settled))
            if np.isfinite(rn) and rn < best_res:
                best_res = rn
                best_f = np.array(settled, copy=True)
                best_lbe += settle_steps + 1
    if best_res < base_res:
        hist.append((len(hist), best_res, best_lbe, max(time.perf_counter() - t0, 1.0e-6)))
        return best_f, hist
    return f, hist


def _open_masked_state_sane(case, f):
    if not (hasattr(case, "chi") and hasattr(case, "U_in")):
        return True
    if not _state_is_admissible(case, f):
        return False
    try:
        rho, ux, uy = macro_of(case, f)
        fluid = getattr(case, "chi") > 0.0
        if not np.any(fluid):
            return False
        speed = np.sqrt(ux * ux + uy * uy)
        u_ref = max(abs(float(getattr(case, "U_in", 0.0))), 1.0e-30)
        return bool(float(np.min(rho[fluid])) > 0.0 and float(np.mean(speed[fluid])) / u_ref >= 0.2)
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default="1,2,3")
    parser.add_argument("--base-cases", default="all")
    parser.add_argument("--out-dir", default="paper_revision_data/_coord_round106_ap_schur_proposed_only")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip()] if args.levels != "all" else [1, 2, 3]
    base_ids = _parse_csv(args.base_cases, CASE_IDS)
    out = Path(args.out_dir)
    hist_dir = out / "histories"
    diag_dir = out / "diagnostics"
    npz_dir = out / "npz"
    for d in (out, hist_dir, diag_dir, npz_dir):
        d.mkdir(parents=True, exist_ok=True)

    summary_path = out / "summary.csv"
    rows = []
    if summary_path.exists() and not args.overwrite:
        rows = list(csv.DictReader(summary_path.open(newline="")))
    done = {(r["case_id"], r["method"]) for r in rows}

    tasks = [(base_id, level) for base_id in base_ids for level in levels]
    uniform_variant = os.environ.get("SAFE_NN_UNIFORM_PROPOSED", "").strip().lower() in {"1", "true", "yes", "on"}
    disable_rre_variant = os.environ.get("SAFE_NN_DISABLE_RRE", "").strip().lower() in {"1", "true", "yes", "on"}
    if uniform_variant and disable_rre_variant:
        method_variant = "uniform_ap_schur_only"
    elif uniform_variant:
        method_variant = "uniform_ap_schur_rre"
    else:
        method_variant = "ap_schur_jfnk_candidate"
    t_all = time.perf_counter()
    for idx, (base_id, level) in enumerate(tasks, 1):
        case_id, label, tol, factory = case_factory_scaled(base_id, level)
        key = (case_id, "proposed")
        if key in done and not args.overwrite:
            print(f"[skip] {idx}/{len(tasks)} {case_id}", flush=True)
            continue

        elapsed = time.perf_counter() - t_all
        avg = elapsed / max(idx - 1, 1)
        eta = avg * (len(tasks) - idx + 1) if idx > 1 else float("nan")
        print(
            f"[case] {idx}/{len(tasks)} {case_id} | elapsed={elapsed:.1f}s "
            f"ETA={(eta if math.isfinite(eta) else 0.0):.1f}s",
            flush=True,
        )

        case = factory()
        t0 = time.perf_counter()
        # Default MSA-LBM pipeline: solve_proposed_single alone now runs the
        # unlimited-attempt Schur+Newton correction with admissibility gate,
        # native fallback, and its own plain native tail (no RRE, no JFNK
        # polish, no masked-open branch-reset). The auxiliary polish/RRE/
        # branch-reset stages remain defined above for anyone who wants to
        # opt back into the earlier, more elaborate pipeline explicitly.
        f, hist = solve_proposed_single(case, tol=tol, verbose=False)
        wall = time.perf_counter() - t0
        hist = _history_final_row(case, f, hist, wall)
        lbe = int(hist[-1][2]) if hist else 0
        final_res, res_p, res_ux, res_uy, res_uz = _macro_l2_residual_components(case, f)
        final_f_rms = _f_rms_residual_value(case, f)
        ref_source, err = _accuracy_against_available_reference(base_id, level, case, f)
        ap_trials, ap_accepts = _diagnostic_counts(hist)
        converged, residual_converged, plateau_converged, convergence_mode, rel_stats = _strict_convergence_flags(base_id, hist, final_res, tol)
        ny, nx = _case_grid_shape(case)
        row = {
            "base_case_id": base_id,
            "scaling_level": int(level),
            "case_id": case_id,
            "case_label": label,
            "method": "proposed",
            "method_variant": method_variant,
            "tol": float(tol),
            "Ny": int(ny),
            "Nx": int(nx),
            "max_steps_nominal": int(max_steps_for_scaled(base_id, level)),
            "lbe_calls": int(lbe),
            "wall_seconds": float(wall),
            "final_residual": float(final_res),
            "final_residual_kind": "macro_l2_p_ux_uy_uz",
            "final_macro_l2_pressure": float(res_p),
            "final_macro_l2_ux": float(res_ux),
            "final_macro_l2_uy": float(res_uy),
            "final_macro_l2_uz": float(res_uz),
            "final_f_rms_residual": float(final_f_rms),
            "initial_macro_l2_residual": float(rel_stats.get("initial_macro_l2_residual", _macro_l2_residual_value(case, case.initial_field()))),
            "relative_macro_l2_residual": float(rel_stats.get("relative_macro_l2_residual", final_res / max(_macro_l2_residual_value(case, case.initial_field()), 1.0e-300))),
            "plateau_improvement": float(rel_stats.get("plateau_improvement", float("nan"))),
            "macro_change": float(rel_stats.get("macro_change", float("nan"))),
            "relative_plateau": int(rel_stats.get("relative_plateau", 0)),
            "macro_change_pass": int(rel_stats.get("macro_change_pass", 0)),
            "relative_floor_pass": int(rel_stats.get("relative_floor_pass", 0)),
            "min_lbe_pass": int(rel_stats.get("min_lbe_pass", 0)),
            "converged": int(converged),
            "residual_converged": int(residual_converged),
            "plateau_converged": int(plateau_converged),
            "convergence_mode": convergence_mode,
            "reference_source": ref_source,
            "rel_l2_vs_ref": float(err["rel_l2"]),
            "linf_vs_ref": float(err["linf"]),
            "rms_vs_ref": float(err["rms"]),
            "vel_abs_l2_vs_ref": float(err["vel_abs_l2"]),
            "vel_abs_linf_vs_ref": float(err["vel_abs_linf"]),
            "vel_abs_rms_vs_ref": float(err["vel_abs_rms"]),
            "ap_schur_trials": int(ap_trials),
            "ap_schur_accepts": int(ap_accepts),
        }
        rows = [r for r in rows if not (r["case_id"] == case_id and r["method"] == "proposed")]
        rows.append(row)

        write_history_csv(hist_dir / f"{case_id}__proposed.csv", hist)
        write_diagnostic_csv(diag_dir / f"{case_id}__proposed__diagnostics.csv", hist)
        np.savez_compressed(npz_dir / f"{case_id}__proposed.npz", f=f)

        with summary_path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        (out / "run_metadata.json").write_text(
            json.dumps(
                {
                    "levels": levels,
                    "base_cases": base_ids,
                    "elapsed_wall_seconds": time.perf_counter() - t_all,
                    "row_count": len(rows),
                    "method_variant": method_variant,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"  proposed lbe={lbe} wall={wall:.3f}s res={final_res:.3e} "
            f"ref={row['rel_l2_vs_ref']:.3e} conv={row['converged']} "
            f"ap={ap_accepts}/{ap_trials}",
            flush=True,
        )

    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
