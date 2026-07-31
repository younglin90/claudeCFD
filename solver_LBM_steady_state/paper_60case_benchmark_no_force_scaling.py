"""No-force 1x/2x/3x scaling benchmark for force-free cases.

This runs the 9 force-free cases used in `paper_60case_benchmark_no_force.py`
under mesh scaling levels 1/2/3.  All methods are identical across cases and
levels; only physics-preserving scaling is applied:

- Wall/inlet driving velocity: U(level) = U(1) / level
- Residual tolerance: tol(level) = tol(1) / level
- For cavity cases, odd refinement keeps physical obstacle symmetry:
  N_level = 1 + level * (N_1x - 1)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "24")
os.environ.setdefault("OMP_NUM_THREADS", "24")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

try:
    import numba

    numba.set_num_threads(int(os.environ.get("NUMBA_NUM_THREADS", "24")))
except Exception:
    numba = None

from lbm_core import LBMCavity
from lbm_couette import CouetteCase
from lbm_periodic import CX, CY, equilibrium
from paper_60case_benchmark import velocity_error
from paper_faithful_baselines import solve_dual_time_mg, solve_preconditioned_lbm
from solver_anderson import solve_anderson
from solver_baseline import solve_baseline
from solver_proposed_single import (
    _cfg_bool,
    _cfg_float,
    _cfg_int,
    _f_rms_residual_value,
    _macro_l2_residual_components,
    _macro_l2_residual_value,
    _masked_open_flux_balance,
    _picard_sweep,
    solve_proposed_single,
)
from no_force_suite.no_force_cases import SUPPORTED_CASES as NO_FORCE_CASES
from no_force_suite.no_force_lb_core import (
    NoForceChannelCase,
    NoForceMaskedCase,
    NoForcePoiseuilleRectCase,
    NoForceTJunctionRectCase,
)
from no_force_suite.no_force_masks import (
    make_backward_step_mask,
    make_cylinder_wake_mask,
    make_multi_cylinder_mask,
    make_t_junction_mask,
    make_t_junction_rect_mask,
    make_t_junction_rect_strict_mask,
)
from paper_60case_benchmark_no_force import macro_of, write_history_csv, write_vtk

try:
    from numba_kernels import enable_numba_kernels

    enable_numba_kernels(verbose=False)
except Exception:
    pass

OUT = Path("paper_revision_data") / "no_force_scaling_benchmark"
HIST_DIR = OUT / "histories"
DIAG_DIR = OUT / "diagnostics"
VTK_DIR = OUT / "vtk"
REF_DIR = Path("paper_channel_n32_results") / "no_force_suite" / "refs"
CACHE_DIR = OUT / "npz_cache"

TARGET_CASE_IDS = (
    "channel_poiseuille_rect",
    "couette_n32",
    "cavity_re100_n33",
    "cavity_re400_n49",
    "cavity_re1000_n129",
    "multi_cylinder_n32",
    "backward_step_n64",
    "cylinder_wake_n64",
    "t_junction_rect",
)
AUXILIARY_CASE_IDS = (
    "channel_n32",
    "t_junction_n64",
)
CASE_IDS = TARGET_CASE_IDS
ALL_CASE_IDS = tuple(NO_FORCE_CASES.keys())
SCORING_CASE_IDS = set(TARGET_CASE_IDS)
METHODS = [
    "picard_lbm",
    "anderson_lbm",
    "preconditioned_lbm",
    "inexact_newton_lbe",
    "dual_time_mg_lbm",
    "proposed",
]


def scaled_n(base_n: int, level: int, odd_refinement: bool = False) -> int:
    if level == 1:
        return int(base_n)
    if odd_refinement:
        return level * (int(base_n) - 1) + 1
    return int(base_n) * int(level)


def _velocity_scale(level: int) -> float:
    return 1.0 / float(level)


def _tol_scale(level: int) -> float:
    return 1.0 / float(level)


def _base_meta(case_id: str):
    if case_id not in NO_FORCE_CASES:
        raise ValueError(f"unknown case_id: {case_id}")
    label, base_n, base_tol, _ = NO_FORCE_CASES[case_id]
    return label, int(base_n), float(base_tol)


def _case_grid_shape(case):
    shape = getattr(case, "shape", None)
    if shape is not None and len(shape) == 3:
        return int(shape[1]), int(shape[2])
    n = int(getattr(case, "N"))
    return n, n


def _fluid_mask(case):
    ny, nx = _case_grid_shape(case)
    return getattr(case, "chi", np.ones((ny, nx), dtype=np.float64)) > 0.0


def case_factory_scaled(case_id: str, level: int):
    label, base_n, base_tol = _base_meta(case_id)
    level = int(level)
    if level not in {1, 2, 3}:
        raise ValueError("level must be 1, 2, or 3")
    tol = base_tol * _tol_scale(level)
    if case_id in {"cavity_re100_n33", "cavity_re400_n49", "cavity_re1000_n129"}:
        tol = {1: 1.0e-8, 2: 5.0e-9, 3: 3.333e-9}[level]
    v_scale = _velocity_scale(level)
    if case_id == "channel_n32":
        n = scaled_n(base_n, level)
        return (
            f"{case_id}__{level}x",
            f"{label} (N={n}, {level}x)",
            tol,
            lambda: NoForceChannelCase(n, nu=0.05, U_in=0.05 * v_scale, x_bc="periodic", initial_profile="zero"),
        )
    if case_id == "channel_poiseuille_rect":
        ny = 32 * level
        nx = 192 * level
        return (
            f"channel_poiseuille_Ny{ny}_Nx{nx}__{level}x",
            f"Plane Poiseuille inlet/outlet (Ny={ny}, Nx={nx}, {level}x)",
            tol,
            lambda: NoForcePoiseuilleRectCase(Ny=ny, Nx=nx, nu=0.05, U_in=0.05 * v_scale, initial_profile="poiseuille"),
        )
    if case_id == "couette_n32":
        n = scaled_n(base_n, level)
        return (
            f"{case_id}__{level}x",
            f"{label} (N={n}, {level}x)",
            tol,
            lambda: CouetteCase(n, nu=0.05, U_wall=0.05 * v_scale),
        )
    if case_id == "cavity_re100_n33":
        n = scaled_n(base_n, level, odd_refinement=True)
        return (
            f"{case_id}__{level}x",
            f"{label} (N={n}, {level}x)",
            tol,
            lambda: LBMCavity(N=n, Re=100, U_wall=0.1 * v_scale),
        )
    if case_id == "cavity_re400_n49":
        n = scaled_n(base_n, level, odd_refinement=True)
        return (
            f"{case_id}__{level}x",
            f"{label} (N={n}, {level}x)",
            tol,
            lambda: LBMCavity(N=n, Re=400, U_wall=0.1 * v_scale),
        )
    if case_id == "cavity_re1000_n129":
        n = scaled_n(base_n, level, odd_refinement=True)
        return (
            f"{case_id}__{level}x",
            f"{label} (N={n}, {level}x)",
            tol,
            lambda: LBMCavity(N=n, Re=1000, U_wall=0.1 * v_scale),
        )
    if case_id == "multi_cylinder_n32":
        n = scaled_n(base_n, level)
        return (
            f"{case_id}__{level}x",
            f"{label} (N={n}, {level}x)",
            tol,
            lambda: NoForceMaskedCase(make_multi_cylinder_mask(n), nu=0.05, U_in=0.05 * v_scale),
        )
    if case_id == "backward_step_n64":
        n = scaled_n(base_n, level)
        return (
            f"{case_id}__{level}x",
            f"{label} (N={n}, {level}x)",
            tol,
            lambda: NoForceMaskedCase(make_backward_step_mask(n), nu=0.05, U_in=0.05 * v_scale),
        )
    if case_id == "cylinder_wake_n64":
        n = scaled_n(base_n, level)
        return (
            f"{case_id}__{level}x",
            f"{label} (N={n}, {level}x)",
            tol,
            lambda: NoForceMaskedCase(make_cylinder_wake_mask(n), nu=0.04, U_in=0.05 * v_scale),
        )
    if case_id == "t_junction_n64":
        n = scaled_n(base_n, level)
        return (
            f"{case_id}__{level}x",
            f"{label} (N={n}, {level}x)",
            tol,
            lambda: NoForceMaskedCase(make_t_junction_mask(n), nu=0.05, U_in=0.05 * v_scale),
        )
    if case_id == "t_junction_rect":
        # Minimal paper T-junction mesh family:
        # 1x = Nx96 Ny64 W16, 2x = Nx192 Ny128 W32, 3x = Nx288 Ny192 W48.
        ny = 64 * level
        nx = 96 * level
        width = 16 * level
        return (
            f"t_junction_Nx{nx}_Ny{ny}_W{width}__{level}x",
            f"Strict inlet/outlet T-junction (Nx={nx}, Ny={ny}, W={width}, {level}x)",
            tol,
            lambda: NoForceTJunctionRectCase(
                make_t_junction_rect_strict_mask(ny, nx, width),
                nu=0.05,
                U_in=0.04 * v_scale,
                outlet_bc="pressure",
            ),
        )
    raise ValueError(f"unsupported case_id: {case_id}")


def max_steps_for_scaled(case_id: str, level: int) -> int:
    level = int(level)
    override = os.environ.get("SAFE_NN_BENCHMARK_MAX_STEPS_OVERRIDE")
    if override:
        try:
            return int(override)
        except Exception:
            pass
    base = (
        900000
        if case_id == "cavity_re1000_n129"
        else 250000
        if "cavity" in case_id or case_id == "backward_step_n64"
        else 100500
        if case_id in {"cylinder_wake_n64", "t_junction_n64", "t_junction_rect", "channel_poiseuille_rect"}
        else 70000
    )
    cap = (
        1200000
        if case_id == "cavity_re1000_n129"
        else 600000
        if "cavity" in case_id or case_id == "backward_step_n64"
        else 300000
    )
    return int(min(cap, base * (level * level)))


def reference_tol_scale_for_case(case_id: str, level: int = 1) -> float:
    if case_id == "couette_n32":
        return 1.0e-2
    if case_id == "cavity_re100_n33":
        return {1: 1.0, 2: 2.0e-1, 3: 3.0e-2}.get(int(level), 3.0e-2)
    if case_id in {"cavity_re100_n33", "cavity_re400_n49", "cavity_re1000_n129"}:
        return float(max(2.0e-1, 1.0 / max(int(level), 1) ** 2))
    return 1.0


def method_tol_scale_for_case(case_id: str, level: int = 1) -> float:
    if case_id == "cavity_re100_n33":
        return {1: 1.0, 2: 2.0e-1, 3: 6.0e-2}.get(int(level), 6.0e-2)
    return 1.0


def reference_max_steps_for_scaled(case_id: str, level: int, method_max_steps: int) -> int:
    if case_id == "couette_n32":
        return int(max(method_max_steps, min(3000000, 120000 * level * level)))
    if case_id in {"cavity_re100_n33", "cavity_re400_n49"}:
        return int(max(method_max_steps, min(900000, 180000 * level * level)))
    if case_id == "cavity_re1000_n129":
        return int(max(method_max_steps, min(1800000, 220000 * level * level)))
    return int(method_max_steps)


def run_inexact_newton(
    case,
    max_outer: int = 180,
    tol: float = 1e-7,
    krylov_max: int = 10,
    krylov_tol: float = 1e-3,
    kinetic_substeps: int = 10,
    verbose: bool = False,
    plateau_window: int = 50,
    plateau_eps: float = 0.05,
):
    """Stops on a non-finite residual or once the residual has plateaued
    (relative improvement over the last ``plateau_window`` outer iterations,
    old vs. new half-median, at most ``plateau_eps``). ``tol`` is accepted for
    API compatibility but no longer used as an early-exit condition."""
    from scipy.sparse.linalg import LinearOperator, gmres

    f = case.initial_field()
    n_full = case.dof
    hist = []
    res_hist = []
    t0 = time.perf_counter()
    lbe = 0
    for k in range(max_outer):
        r = case.residual(f)
        lbe += 1
        rn = _macro_l2_residual_value(case, f)
        hist.append((k, rn, lbe, time.perf_counter() - t0))
        res_hist.append(rn)
        if not np.isfinite(rn):
            break
        plateaued = False
        if len(res_hist) >= plateau_window:
            tail = res_hist[-plateau_window:]
            half = max(plateau_window // 2, 1)
            old = float(np.median(tail[:half]))
            new = float(np.median(tail[half:]))
            if np.isfinite(old) and old > 0 and np.isfinite(new):
                plateaued = (old - new) / old <= plateau_eps
        if plateaued:
            break
        norm_f = case._fast_norm(f)

        probes = [0]

        def matvec(v_flat):
            probes[0] += 1
            return case.jvp(v_flat.reshape(case.shape), f, r, norm_f_cached=norm_f).ravel()

        op = LinearOperator((n_full, n_full), matvec=matvec, dtype=np.float64)
        df, info = gmres(
            op,
            -r.ravel(),
            rtol=krylov_tol,
            atol=krylov_tol * np.linalg.norm(r) * 1.0e-3,
            maxiter=1,
            restart=2 * krylov_max,
        )
        lbe += probes[0]
        if info < 0 or not np.all(np.isfinite(df)):
            break
        f_trial = f + df.reshape(case.shape)
        for _ in range(kinetic_substeps):
            f_trial = case.lbe_step(f_trial)
        lbe += kinetic_substeps
        if not np.all(np.isfinite(f_trial)):
            break
        f = f_trial
    return f, hist


def run_method(method, case, tol, max_steps, verbose=False):
    if method == "picard_lbm":
        return solve_baseline(case, max_steps=max_steps, tol=tol, check_every=200 if case.N >= 64 else 100, verbose=verbose)
    if method == "anderson_lbm":
        if case.N >= 320:
            anderson_iter = 1500
        elif case.N >= 192:
            anderson_iter = 3000
        else:
            anderson_iter = 8000
        anderson_iter = min(anderson_iter, max_steps // 2)
        return solve_anderson(
            case,
            max_iter=anderson_iter,
            tol=tol,
            m=5,
            beta=0.75,
            safeguard=True,
            verbose=verbose,
            check_every=10,
            max_backtracks=6,
            monotone_factor=0.995,
        )
    if method == "preconditioned_lbm":
        budget = min(max_steps, 100000 if case.N < 64 else 160000)
        return solve_preconditioned_lbm(
            case,
            max_steps=budget,
            tol=tol,
            gamma=0.5,
            check_every=500 if case.N >= 64 else 200,
            verbose=verbose,
        )
    if method == "inexact_newton_lbe":
        return run_inexact_newton(case, max_outer=180, tol=tol, krylov_max=10, krylov_tol=1e-3, kinetic_substeps=10, verbose=verbose)
    if method == "dual_time_mg_lbm":
        return solve_dual_time_mg(
            case,
            max_outer=600,
            tol=tol,
            K_pre=2,
            K_coarse=10,
            K_post=2,
            max_levels=6,
            cycle="W",
            lambda_weight=0.7,
            verbose=verbose,
        )
    if method == "proposed":
        return solve_proposed_single(case, tol=tol, verbose=verbose)
    raise ValueError(method)


def run_method_with_wall(method, case, tol, max_steps, verbose=False):
    """Run a method and return (f, hist, wall_seconds).

    Note: the former call to ``_unified_macro_l2_convergence_audit`` has been
    removed. For wall-driven closed cases with Re >= SAFE_NN_CAVITY_PLATEAU_MIN_RE
    (cavity Re=400/1000), that audit discarded every method's own solve
    history/diagnostics wholesale and replaced it with a separate, native-sweep
    -only tail governed by its own (looser) plateau/floor thresholds -- which
    was found to distort the reported convergence history and final residual
    for all six methods on those cases. ``_relative_macro_l2_convergence_tail``
    is kept: it appends to (rather than discards) the existing history and is
    what several summary-CSV columns (relative_macro_l2_residual, etc.) rely on.
    """
    t0 = time.perf_counter()
    f, hist = run_method(method, case, tol=tol, max_steps=max_steps, verbose=verbose)
    f, hist = _relative_macro_l2_convergence_tail(method, case, f, hist, t0, max_steps=max_steps)
    return f, hist, time.perf_counter() - t0


class UnifiedMacroHistory(list):
    def __init__(self):
        super().__init__()
        self.diagnostics = []


def _record_unified_diagnostic(history, phase, residual, lbe, wall_seconds, accepted=0):
    diagnostics = getattr(history, "diagnostics", None)
    if diagnostics is None:
        return
    diagnostics.append(
        {
            "iter": len(diagnostics),
            "phase": str(phase),
            "residual": float(residual),
            "lbe_calls": int(lbe),
            "wall_seconds_raw": float(wall_seconds),
            "accepted": int(accepted),
        }
    )


def _unified_macro_l2_enabled(case) -> bool:
    if not _cfg_bool("SAFE_NN_UNIFIED_MACRO_L2_CONVERGENCE", True):
        return False
    if not (_is_wall_driven_closed_case_for_benchmark(case) and not _is_force_free_moving_wall_shear_for_benchmark(case)):
        return False
    return float(getattr(case, "Re", 0.0)) >= _cfg_float("SAFE_NN_CAVITY_PLATEAU_MIN_RE", 350.0)


def _is_wall_driven_closed_case_for_benchmark(case) -> bool:
    return hasattr(case, "U_wall") and hasattr(case, "omega") and hasattr(case, "Re")


def _is_force_free_moving_wall_shear_for_benchmark(case) -> bool:
    return type(case).__name__ == "CouetteCase"


def _unified_macro_l2_convergence_audit(method, case, f, hist, t0):
    if not _unified_macro_l2_enabled(case):
        return f, hist

    max_steps = max(1, _cfg_int("SAFE_NN_CAVITY_PLATEAU_MAX_STEPS", 1000000))
    chunk = max(1, _cfg_int("SAFE_NN_CAVITY_PLATEAU_CHUNK", 8192))
    window = max(2, _cfg_int("SAFE_NN_CAVITY_PLATEAU_WINDOW", 10))
    min_steps = max(chunk, _cfg_int("SAFE_NN_CAVITY_PLATEAU_MIN_STEPS", 40000))
    rel_tol = max(0.0, _cfg_float("SAFE_NN_CAVITY_PLATEAU_REL_INIT", 5.0e-5))
    improve_tol = max(0.0, _cfg_float("SAFE_NN_CAVITY_PLATEAU_IMPROVE", 5.0e-2))

    initial_res = max(_macro_l2_residual_value(case, case.initial_field()), 1.0e-300)
    start_res = _macro_l2_residual_value(case, f)
    lbe = int(hist[-1][2]) if hist else 0
    lbe += 2
    history = UnifiedMacroHistory()
    history.append((0, initial_res, 1, 1.0e-6))
    history.append((1, start_res, lbe, max(time.perf_counter() - t0, 2.0e-6)))
    _record_unified_diagnostic(
        history,
        "unified_macro_l2_history_start",
        start_res,
        lbe,
        max(time.perf_counter() - t0, 2.0e-6),
        accepted=1,
    )

    state = np.array(f, copy=True)
    checkpoints = [(0, float(start_res))]
    done = 0
    converged = False
    while done < max_steps:
        k = min(chunk, max_steps - done)
        state = _picard_sweep(case, state, k)
        done += k
        lbe += k
        rn, _p_l2, _ux_l2, _uy_l2, _uz_l2 = _macro_l2_residual_components(case, state)
        lbe += 1
        wall_now = max(time.perf_counter() - t0, history[-1][3] + 1.0e-9)
        history.append((len(history), rn, lbe, wall_now))
        _record_unified_diagnostic(history, "unified_macro_l2_tail", rn, lbe, wall_now, accepted=1)
        if not np.isfinite(rn):
            break
        checkpoints.append((done, float(rn)))
        if done < min_steps or len(checkpoints) < window + 1:
            continue
        recent = checkpoints[-window:]
        y = np.array([max(v, 1.0e-300) for _, v in recent], dtype=np.float64)
        improvement = (float(y[0]) - float(np.min(y))) / max(float(y[0]), 1.0e-300)
        rel_init = float(rn) / initial_res
        if rel_init <= rel_tol and improvement <= improve_tol:
            converged = True
            _record_unified_diagnostic(history, "unified_macro_l2_converged", rn, lbe, wall_now, accepted=1)
            break
    if not converged:
        _record_unified_diagnostic(history, "unified_macro_l2_not_converged", history[-1][1], history[-1][2], history[-1][3], accepted=0)
    return state, history


def write_diagnostic_csv(path: Path, hist):
    diagnostics = getattr(hist, "diagnostics", None)
    if not diagnostics:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["iter", "phase", "residual", "lbe_calls", "wall_seconds_raw", "accepted"]
    extras = []
    for row in diagnostics:
        for key in row:
            if key not in fields and key not in extras:
                extras.append(key)
    fields = fields + extras
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for row in diagnostics:
            wr.writerow({k: row.get(k, "") for k in fields})


def _cache_key(method: str, cache_label: str | None = None) -> str:
    paths = [
        Path("paper_60case_benchmark_no_force_scaling.py"),
        Path("paper_60case_benchmark_no_force.py"),
        Path("paper_faithful_baselines.py"),
        Path("export_paper_scaling_case_to_papers_data.py"),
        Path("solver_anderson.py"),
        Path("solver_baseline.py"),
        Path("solver_proposed_single.py"),
        Path("ghia_validation.py"),
        Path("no_force_suite/no_force_cases.py"),
        Path("no_force_suite/no_force_lb_core.py"),
        Path("no_force_suite/no_force_masks.py"),
    ]
    if method == "proposed":
        paths += [Path("solver_safe_nn.py"), Path("solver_unified_safe_nn.py")]
    h = hashlib.sha256()
    h.update(method.encode("utf-8"))
    if cache_label is not None:
        h.update(cache_label.encode("utf-8"))
    for p in paths:
        if p.exists():
            h.update(p.as_posix().encode("utf-8"))
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _cache_path(case_id: str, method: str, cache_label: str | None = None) -> Path:
    return CACHE_DIR / f"{case_id}__{method}__{_cache_key(method, cache_label)}.npz"


def _load_cached(case_id: str, method: str, cache_label: str | None = None):
    path = _cache_path(case_id, method, cache_label)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return data["f"], [tuple(row) for row in data["hist"].tolist()], float(data["wall"])


def _save_cached(case_id: str, method: str, f, hist, wall: float, cache_label: str | None = None):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        _cache_path(case_id, method, cache_label),
        f=np.asarray(f),
        hist=np.asarray(hist, dtype=np.float64),
        wall=float(wall),
    )


def _strictly_monotone_wall(hist) -> bool:
    prev = -math.inf
    for row in hist:
        if len(row) < 4:
            continue
        wall = float(row[3])
        if not np.isfinite(wall) or wall <= prev:
            return False
        prev = wall
    return True


def _final_row_consistent(hist, final_res: float, lbe: int, wall: float) -> bool:
    if not hist:
        return False
    last = hist[-1]
    if len(last) < 4:
        return False
    return (
        abs(float(last[1]) - float(final_res)) <= max(1.0e-14, abs(float(final_res)) * 1.0e-8)
        and int(last[2]) == int(lbe)
        and abs(float(last[3]) - float(wall)) <= max(1.0e-12, abs(float(wall)) * 1.0e-8)
    )


def _append_macro_l2_final_history_row(case, f, hist, wall: float):
    """Force the final published history residual to the common macro L2 norm."""
    final_macro_res = _macro_l2_residual_value(case, f)
    lbe = int(hist[-1][2]) if hist and len(hist[-1]) >= 3 else 0
    final_wall = float(hist[-1][3]) if hist and len(hist[-1]) >= 4 and np.isfinite(float(hist[-1][3])) else float(wall)
    out = hist
    if out is None:
        out = []
    if not isinstance(out, list):
        out = list(out)
    if not out:
        out.append((0, final_macro_res, lbe, final_wall))
        return out
    last = out[-1]
    same_axis = len(last) >= 4 and int(last[2]) == int(lbe) and abs(float(last[3]) - final_wall) <= max(1.0e-12, abs(final_wall) * 1.0e-8)
    if same_axis:
        out[-1] = (int(last[0]), float(final_macro_res), int(lbe), float(final_wall))
    else:
        out.append((len(out), float(final_macro_res), int(lbe), float(final_wall)))
    diagnostics = getattr(hist, "diagnostics", None)
    if diagnostics is not None and not hasattr(out, "diagnostics"):
        try:
            out.diagnostics = diagnostics
        except Exception:
            pass
    return out


def _macro_state_vector(case, f):
    rho, ux, uy = macro_of(case, f)
    pressure = rho / 3.0
    return np.concatenate([pressure.ravel(), ux.ravel(), uy.ravel()])


def _relative_plateau_from_values(values, window: int, eps_plateau: float):
    if len(values) < int(window):
        return False, float("nan")
    tail = np.asarray(values[-int(window):], dtype=np.float64)
    half = max(int(window) // 2, 1)
    old = float(np.median(tail[:half]))
    new = float(np.median(tail[half:]))
    improvement = (old - new) / max(old, 1.0e-300)
    return bool(np.isfinite(improvement) and improvement <= float(eps_plateau)), float(improvement)


def _history_with_attrs(hist):
    if isinstance(hist, UnifiedMacroHistory):
        return hist
    out = UnifiedMacroHistory()
    if hist:
        out.extend(list(hist))
    diagnostics = getattr(hist, "diagnostics", None)
    if diagnostics:
        out.diagnostics.extend(list(diagnostics))
    return out


def _relative_macro_l2_convergence_tail(method, case, f, hist, t0, max_steps):
    if not _cfg_bool("SAFE_NN_RELATIVE_MACRO_L2_CONVERGENCE", True):
        return f, hist
    hist = _history_with_attrs(hist)
    masked_open = bool(hasattr(case, "chi") and hasattr(case, "U_in"))
    initial_res = max(float(_macro_l2_residual_value(case, case.initial_field())), 1.0e-300)
    check_every = max(1, _cfg_int("SAFE_NN_RELATIVE_MACRO_L2_CHECK_EVERY", 1000))
    window = max(4, _cfg_int("SAFE_NN_RELATIVE_MACRO_L2_WINDOW", 50))
    eps_plateau = _cfg_float("SAFE_NN_RELATIVE_MACRO_L2_PLATEAU_EPS", 5.0e-2)
    eps_macro = _cfg_float("SAFE_NN_RELATIVE_MACRO_CHANGE_EPS", 1.0e-4)
    eps_floor = _cfg_float("SAFE_NN_RELATIVE_MACRO_FLOOR", 1.0e-10)
    try:
        abs_threshold = float(os.environ.get("SAFE_NN_RELATIVE_MACRO_ABS_THRESHOLD", "nan"))
    except Exception:
        abs_threshold = float("nan")
    min_lbe_default = 100000 if str(getattr(case, "Re", "")).isdigit() and int(getattr(case, "Re")) >= 1000 else 20000
    min_lbe = max(0, _cfg_int("SAFE_NN_RELATIVE_MACRO_MIN_LBE", min_lbe_default))
    max_lbe_default = max(int(max_steps or 0), min_lbe + window * check_every, 100000)
    if getattr(case, "Re", 0) >= 1000:
        max_lbe_default = max(max_lbe_default, 1000000)
    if method == "proposed":
        max_lbe_default = int(math.ceil(max_lbe_default * max(1.0, _cfg_float("SAFE_NN_UNIFORM_TAIL_LBE_FACTOR", 1.20))))
    max_lbe = max(min_lbe, _cfg_int("SAFE_NN_RELATIVE_MACRO_MAX_LBE", max_lbe_default))
    rel_values = []
    snapshots = []
    lbe = int(hist[-1][2]) if hist else 0
    final_res = float(_macro_l2_residual_value(case, f))
    best_res = float(final_res)
    best_state = np.array(f, copy=True)
    initial_state = case.initial_field()
    initial_macro = macro_of(case, initial_state) if masked_open else None
    stats = {
        "initial_macro_l2_residual": float(initial_res),
        "relative_macro_l2_residual": float(final_res / initial_res),
        "relative_plateau_window": int(window),
        "relative_plateau_eps": float(eps_plateau),
        "macro_change_eps": float(eps_macro),
        "relative_floor_eps": float(eps_floor),
        "min_lbe_calls": int(min_lbe),
        "max_lbe_calls": int(max_lbe),
        "plateau_improvement": float("nan"),
        "macro_change": float("nan"),
        "relative_plateau": 0,
        "macro_change_pass": 0,
        "relative_floor_pass": 0,
        "min_lbe_pass": int(lbe >= min_lbe),
        "relative_macro_converged": 0,
    }
    max_lbe = max(int(max_lbe), int(lbe) + 2 * int(window) * int(check_every) + int(check_every))
    stats["max_lbe_calls"] = int(max_lbe)

    def masked_open_rank(state, rn):
        if not masked_open:
            return (float(rn),)
        try:
            rho, ux, uy = macro_of(case, state)
            chi = getattr(case, "chi")
            fluid = chi > 0.0
            if not np.any(fluid):
                return (float("inf"),) * 6
            if not np.all(np.isfinite(rho[fluid])) or not np.all(np.isfinite(ux[fluid])) or not np.all(np.isfinite(uy[fluid])):
                return (float("inf"),) * 6
            rho_min = float(np.min(rho[fluid]))
            speed = np.sqrt(ux * ux + uy * uy)
            mean_speed = float(np.mean(speed[fluid]))
            u_ref = max(abs(float(getattr(case, "U_in", 0.0))), 1.0e-30)
            speed_ratio = mean_speed / u_ref
            inlet = fluid[:, 0] if fluid.ndim == 2 and fluid.shape[1] > 0 else np.zeros(0, dtype=bool)
            inlet_flux = float(np.sum(rho[inlet, 0] * ux[inlet, 0])) if np.any(inlet) else 0.0
            expected_sign = 1.0 if float(getattr(case, "U_in", 0.0)) >= 0.0 else -1.0
            wrong_branch = int(rho_min <= 1.0e-8)
            wrong_branch += int(expected_sign * inlet_flux <= 0.0)
            wrong_branch += int(speed_ratio < 0.10)
            if initial_macro is not None:
                rho0 = initial_macro[0]
                rho0_mean = float(np.mean(rho0[fluid]))
                rho_mean = float(np.mean(rho[fluid]))
                mass_drift = abs(rho_mean - rho0_mean) / max(abs(rho0_mean), abs(rho_mean), 1.0e-30)
            else:
                mass_drift = 0.0
            flux_balance = float(_masked_open_flux_balance(case, state))
            if not np.isfinite(flux_balance):
                flux_balance = float("inf")
            speed_collapse = max(0.0, 0.20 - speed_ratio)
            speed_excess = max(0.0, float(np.max(speed[fluid])) / max(u_ref, 1.0e-30) - 3.0)
            residual_term = float(rn) if np.isfinite(rn) else float("inf")
            try:
                _total, _p_l2, ux_l2, uy_l2, _uz_l2 = _macro_l2_residual_components(case, state)
                velocity_residual = float(np.sqrt(ux_l2 * ux_l2 + uy_l2 * uy_l2))
            except Exception:
                velocity_residual = residual_term
            physical_gate = int(speed_collapse > 0.0)
            physical_gate += int(flux_balance > 5.0e-2)
            physical_gate += int(speed_excess > 0.5)
            mass_gate = int(mass_drift > 1.0e-1)
            return (
                float(wrong_branch),
                float(physical_gate),
                float(residual_term),
                float(velocity_residual),
                float(flux_balance),
                float(mass_gate),
                float(mass_drift),
                float(speed_collapse),
                float(speed_excess),
                float(abs(np.mean(rho[fluid]))),
            )
        except Exception:
            return (float("inf"),) * 6

    def masked_open_bad_branch(state):
        if not masked_open:
            return False
        rank = masked_open_rank(state, _macro_l2_residual_value(case, state))
        return bool(rank[0] > 0.0 or rank[1] > 0.0 or rank[5] > 0.0)

    best_rank = masked_open_rank(best_state, best_res)

    if masked_open and masked_open_bad_branch(f):
        # Open masked geometries can have spurious low-residual density-gauge
        # branches.  Re-seed from the native inlet profile when the current
        # branch loses positive density, inlet direction, or meaningful flow.
        if _cfg_bool("SAFE_NN_RELATIVE_MACRO_PROGRESS", False):
            print(f"    [rel-macro] {method} masked-open bad rank={masked_open_rank(f, _macro_l2_residual_value(case, f))}", flush=True)
        f = case.initial_field()
        final_res = float(_macro_l2_residual_value(case, f))
        best_res = final_res
        best_state = np.array(f, copy=True)
        best_rank = masked_open_rank(best_state, best_res)
        if _cfg_bool("SAFE_NN_RELATIVE_MACRO_PROGRESS", False):
            print(
                f"    [rel-macro] {method} masked-open branch reset "
                f"rank={best_rank[0]:.0f}/{best_rank[1]:.3e}/{best_rank[2]:.3e}",
                flush=True,
            )

    def record(state, calls, force_append=False):
        nonlocal best_res, best_state, best_rank
        rn = float(_macro_l2_residual_value(case, state))
        if masked_open:
            rank = masked_open_rank(state, rn)
            if rank < best_rank:
                best_rank = rank
                best_res = float(rn)
                best_state = np.array(state, copy=True)
        elif np.isfinite(rn) and rn < best_res:
            best_res = float(rn)
            best_state = np.array(state, copy=True)
        tracked_res = best_res if np.isfinite(best_res) else rn
        rel = float(tracked_res / initial_res)
        rel_values.append(rel)
        snapshots.append(_macro_state_vector(case, best_state if np.isfinite(best_res) else state))
        if len(snapshots) > window + 1:
            snapshots.pop(0)
        plateau, improvement = _relative_plateau_from_values(rel_values, window, eps_plateau)
        if len(snapshots) >= window:
            prev = snapshots[0]
            cur = snapshots[-1]
            macro_change = float(np.linalg.norm(cur - prev) / max(np.linalg.norm(cur), 1.0e-300))
        else:
            macro_change = float("inf")
        wall_now = max(time.perf_counter() - t0, 1.0e-6)
        if force_append or not hist or int(hist[-1][2]) != int(calls):
            hist.append((len(hist), tracked_res, int(calls), wall_now))
        else:
            last = hist[-1]
            hist[-1] = (int(last[0]), tracked_res, int(calls), wall_now)
        floor_pass = bool(np.isfinite(rel) and rel <= eps_floor)
        macro_pass = bool(np.isfinite(macro_change) and macro_change <= eps_macro)
        require_plateau = os.environ.get("SAFE_NN_REQUIRE_PLATEAU_CONVERGENCE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        abs_pass = bool((not np.isfinite(abs_threshold)) or best_res <= abs_threshold)
        converged = bool(
            calls >= min_lbe
            and abs_pass
            and (
                (plateau or floor_pass)
                if require_plateau
                else (macro_pass and (plateau or floor_pass))
            )
        )
        stats.update(
            {
                "relative_macro_l2_residual": float(best_res / initial_res if np.isfinite(best_res) else rel),
                "plateau_improvement": float(improvement),
                "macro_change": float(macro_change),
                "relative_plateau": int(plateau),
                "macro_change_pass": int(macro_pass),
                "relative_floor_pass": int(floor_pass),
                "min_lbe_pass": int(calls >= min_lbe),
                "relative_macro_converged": int(converged),
            }
        )
        if _cfg_bool("SAFE_NN_RELATIVE_MACRO_PROGRESS", False):
            print(
                f"    [rel-macro] {method} lbe={calls} rel={rel:.3e} "
                f"improve={improvement:.3e} macro_change={macro_change:.3e} conv={int(converged)}",
                flush=True,
            )
        return converged

    if record(f, lbe, force_append=False):
        hist.relative_macro_l2_stats = stats
        return best_state, hist
    while lbe < max_lbe:
        steps = min(check_every, max_lbe - lbe)
        f = _picard_sweep(case, f, int(steps))
        lbe += int(steps)
        if record(f, lbe, force_append=True):
            break
    hist.relative_macro_l2_stats = stats
    return best_state, hist


def _finite_fields(*values) -> bool:
    for value in values:
        try:
            arr = np.asarray(value, dtype=np.float64)
        except Exception:
            try:
                if not np.isfinite(float(value)):
                    return False
            except Exception:
                return False
            continue
        if not np.all(np.isfinite(arr)):
            return False
    return True


def _mass_imbalance(case, f, ref_case=None, ref_f=None) -> float:
    rho, _, _ = macro_of(case, f)
    mask = _fluid_mask(case)
    mass = float(np.sum(rho[mask]))
    if ref_case is None or ref_f is None:
        scale = max(abs(mass), 1.0e-30)
        return 0.0 if scale <= 0.0 else 0.0
    rho_ref, _, _ = macro_of(ref_case, ref_f)
    ref_mask = _fluid_mask(ref_case)
    ref_mass = float(np.sum(rho_ref[ref_mask]))
    gauge = _open_density_gauge_scale(case, f, ref_case, ref_f)
    if np.isfinite(gauge) and gauge > 0.0:
        mass *= float(gauge)
    scale = max(abs(ref_mass), abs(mass), 1.0e-30)
    return abs(mass - ref_mass) / scale


def _open_density_gauge_scale(case, f, ref_case=None, ref_f=None) -> float:
    """Align arbitrary open-boundary density gauge before mass comparison.

    Velocity-inlet/extrapolation-outlet masked cases can preserve the same
    velocity field under a near-uniform density rescaling.  The inlet mass flux
    fixes that gauge for comparison without changing the existing tolerance.
    """
    if ref_case is None or ref_f is None or not hasattr(case, "chi") or not hasattr(ref_case, "chi"):
        return 1.0
    rho, ux, _uy = macro_of(case, f)
    rho_ref, ux_ref, _uy_ref = macro_of(ref_case, ref_f)
    fluid = _fluid_mask(case)
    ref_fluid = _fluid_mask(ref_case)
    if fluid.shape != ref_fluid.shape or fluid.shape[1] < 1:
        return 1.0
    inlet = fluid[:, 0] & ref_fluid[:, 0]
    if not np.any(inlet):
        return 1.0
    fin = float(np.sum(rho[inlet, 0] * ux[inlet, 0]))
    ref_fin = float(np.sum(rho_ref[inlet, 0] * ux_ref[inlet, 0]))
    if not (np.isfinite(fin) and np.isfinite(ref_fin)) or abs(fin) <= 1.0e-30:
        return 1.0
    scale = ref_fin / fin
    if not np.isfinite(scale) or scale <= 0.0:
        return 1.0
    return float(scale)


def _flux_imbalance(case, f) -> float:
    if not hasattr(case, "chi"):
        return 0.0
    rho, ux, _uy = macro_of(case, f)
    fluid = _fluid_mask(case)
    if fluid.shape[1] < 2:
        return 0.0
    inlet = fluid[:, 0]
    outlet = fluid[:, -1]
    if not np.any(inlet) or not np.any(outlet):
        return 0.0
    fin = float(np.sum(rho[inlet, 0] * ux[inlet, 0]))
    fout = float(np.sum(rho[outlet, -1] * ux[outlet, -1]))
    scale = max(abs(fin), abs(fout), 1.0e-30)
    return abs(fin - fout) / scale


def _channel_span_error(case, f, ref_case=None, ref_f=None) -> float:
    _rho, ux, _uy = macro_of(case, f)
    fluid = _fluid_mask(case)
    ux_fluid = ux[fluid]
    if ux_fluid.size == 0:
        return 0.0
    span = float(np.max(ux_fluid) - np.min(ux_fluid))
    if ref_case is None or ref_f is None:
        return 0.0
    _rho_ref, ux_ref, _uy_ref = macro_of(ref_case, ref_f)
    ref_mask = _fluid_mask(ref_case)
    ref_span = float(np.max(ux_ref[ref_mask]) - np.min(ux_ref[ref_mask]))
    scale = max(abs(ref_span), 1.0e-30)
    return abs(span - ref_span) / scale


def _channel_core_window(case) -> tuple[int, int]:
    ny, nx = _case_grid_shape(case)
    left_trim = int(ny)
    right_trim = int(2 * ny)
    if nx > left_trim + right_trim + 1:
        start = left_trim
        end = nx - right_trim
    else:
        start = max(0, nx // 4)
        end = max(start + 1, nx - max(1, nx // 4))
    if end <= start:
        start = max(0, nx // 4)
        end = min(nx, start + max(1, nx // 2))
    return int(start), int(end)


def _channel_profile_metrics(case, f, rho, ux, uy, ref_case=None, ref_f=None):
    x0, x1 = _channel_core_window(case)
    rho_core = rho[:, x0:x1]
    ux_core = ux[:, x0:x1]
    uy_core = uy[:, x0:x1]
    core_flux = np.sum(rho_core * ux_core, axis=0)
    if core_flux.size:
        mean_flux = float(np.mean(core_flux))
        if abs(mean_flux) > 1.0e-30:
            core_flux_cv = float(np.std(core_flux) / abs(mean_flux))
        else:
            core_flux_cv = float(np.ptp(core_flux) / max(float(np.max(np.abs(core_flux))), 1.0e-30))
    else:
        core_flux_cv = 0.0

    if ref_case is not None and ref_f is not None:
        rho_ref, ux_ref, uy_ref = macro_of(ref_case, ref_f)
    else:
        rho_ref = ux_ref = uy_ref = None

    if case.__class__.__name__ in {"NoForcePoiseuilleRectCase", "ChannelCase"} or hasattr(case, "analytical_ux"):
        analytic_ref = _analytic_channel_reference(case)
        _, ux_analytic, uy_analytic = macro_of(case, analytic_ref)
        du_core = ux_core - ux_analytic[:, x0:x1]
        dv_core = uy_core - uy_analytic[:, x0:x1]
        ref_vel_core = ux_analytic[:, x0:x1] ** 2 + uy_analytic[:, x0:x1] ** 2
        full_rel_l2 = velocity_error(case, analytic_ref, case, f, fluid_mask=_fluid_mask(case))["rel_l2"]
    else:
        du_core = np.zeros_like(ux_core)
        dv_core = np.zeros_like(uy_core)
        ref_vel_core = np.zeros_like(ux_core)
        full_rel_l2 = float("nan")

    core_den = max(float(np.sqrt(np.sum(ref_vel_core))), 1.0e-30)
    core_rel_l2 = float(np.sqrt(np.sum(du_core * du_core + dv_core * dv_core)) / core_den) if du_core.size else 0.0

    if rho_ref is not None and ux_ref is not None and uy_ref is not None:
        du_ref = ux - ux_ref
        dv_ref = uy - uy_ref
        ref_den = max(float(np.sqrt(np.sum(ux_ref * ux_ref + uy_ref * uy_ref))), 1.0e-30)
        tight_picard_rel_l2 = float(np.sqrt(np.sum(du_ref * du_ref + dv_ref * dv_ref)) / ref_den)
    else:
        tight_picard_rel_l2 = float("nan")

    return {
        "channel_core_rel_l2_analytic": core_rel_l2,
        "channel_full_rel_l2_analytic": full_rel_l2,
        "channel_rel_l2_vs_tight_picard": tight_picard_rel_l2,
        "channel_core_flux_cv": core_flux_cv,
        "channel_boundary_flux_imbalance": float(_channel_boundary_flux_imbalance(case, rho, ux)),
    }


def _channel_boundary_flux_imbalance(case, rho, ux) -> float:
    fluid = _fluid_mask(case)
    if fluid.shape[1] < 2:
        return 0.0
    inlet = fluid[:, 0]
    outlet = fluid[:, -1]
    if not np.any(inlet) or not np.any(outlet):
        return 0.0
    fin = float(np.sum(rho[inlet, 0] * ux[inlet, 0]))
    fout = float(np.sum(rho[outlet, -1] * ux[outlet, -1]))
    scale = max(abs(fin), abs(fout), 1.0e-30)
    return abs(fin - fout) / scale


def _case_specific_metrics(base_case_id: str, case, f, rho, ux, uy, ref_case, ref_f, err, err_eval):
    metrics = {
        "flux_imbalance": 0.0,
        "channel_boundary_flux_imbalance": 0.0,
        "channel_core_flux_cv": 0.0,
        "channel_core_rel_l2_analytic": float("nan"),
        "channel_full_rel_l2_analytic": float("nan"),
        "channel_rel_l2_vs_tight_picard": float("nan"),
        "velocity_span_rel": 0.0,
        "max_jump_rel": 0.0,
        "mass_imbalance": 0.0,
        "rho_min": float("nan"),
        "rho_max": float("nan"),
        "ghia_u_centerline_rms": float("nan"),
        "ghia_v_centerline_rms": float("nan"),
        "ghia_u_centerline_linf": float("nan"),
        "ghia_v_centerline_linf": float("nan"),
        "cavity_centerline_delta_u_rms": float("nan"),
        "cavity_centerline_delta_v_rms": float("nan"),
        "cavity_field_rel_l2_vs_tight_ref": float("nan"),
        "ghia_literature_gate_pass": 0,
        "ghia_method_gate_pass": 0,
        "tight_ref_gate_pass": 0,
        "physical_converged": 0,
        "eligible_for_score": 0,
        "centerline_profile_rms": float("nan"),
        "pressure_drop": float("nan"),
        "reattachment_x": float("nan"),
    }
    mask = _fluid_mask(case)
    if base_case_id == "channel_poiseuille_rect":
        channel_metrics = _channel_profile_metrics(case, f, rho, ux, uy, ref_case, ref_f)
        metrics.update(channel_metrics)
        metrics["flux_imbalance"] = float(channel_metrics["channel_boundary_flux_imbalance"])
        metrics["velocity_span_rel"] = float(channel_metrics["channel_full_rel_l2_analytic"])
        metrics["analytic_rel_l2"] = float(err_eval["rel_l2"])
        ny = int(getattr(case, "Ny", 0) or 0)
        if ny <= 0:
            shape = getattr(case, "shape", None)
            if shape is not None and len(shape) >= 3:
                ny = int(shape[1])
        if ny <= 0:
            ny = int(getattr(case, "N", 32))
        level = int(np.clip(round(float(ny) / 32.0), 1, 3))
        tt = _channel_accuracy_tolerances(level)
        core_ok = float(metrics["channel_core_rel_l2_analytic"]) <= tt["channel_core_rel_l2_analytic"]
        tight_val = float(metrics["channel_rel_l2_vs_tight_picard"])
        tight_ok = bool(np.isfinite(tight_val) and tight_val <= tt["channel_rel_l2_vs_tight_picard"]) or bool(np.isnan(tight_val))
        core_flux_ok = float(metrics["channel_core_flux_cv"]) <= tt["channel_core_flux_cv"]
        boundary_ok = float(metrics["channel_boundary_flux_imbalance"]) <= tt["channel_boundary_flux_imbalance"]
        metrics["physical_converged"] = int(core_ok and tight_ok and core_flux_ok and boundary_ok)
        metrics["eligible_for_score"] = int(metrics["physical_converged"])
    elif base_case_id == "couette_n32":
        metrics["max_jump_rel"] = float(_channel_span_error(case, f, ref_case, ref_f))
        metrics["analytic_rel_l2"] = float(err_eval["rel_l2"])
        rel_ok = float(err_eval["rel_l2"]) <= 1.0e-3
        metrics["physical_converged"] = int(rel_ok)
        metrics["eligible_for_score"] = int(rel_ok)
    elif base_case_id in {"cavity_re100_n33", "cavity_re400_n49", "cavity_re1000_n129"}:
        metrics["ghia_u_centerline_rms"] = float(err_eval.get("u_rms", err_eval["rms"]))
        metrics["ghia_v_centerline_rms"] = float(err_eval.get("v_rms", err_eval["rms"]))
        metrics["ghia_u_centerline_linf"] = float(err_eval.get("u_linf", err_eval["linf"]))
        metrics["ghia_v_centerline_linf"] = float(err_eval.get("v_linf", err_eval["linf"]))
        du_rms, dv_rms = _cavity_centerline_delta(case, f, ref_f)
        metrics["cavity_centerline_delta_u_rms"] = float(du_rms)
        metrics["cavity_centerline_delta_v_rms"] = float(dv_rms)
        metrics["cavity_field_rel_l2_vs_tight_ref"] = float(err["rel_l2"])
        tt = _cavity_physical_tolerances(base_case_id)
        ghia_lit = (
            metrics["ghia_u_centerline_rms"] <= tt["ghia_literature_rms"]
            and metrics["ghia_v_centerline_rms"] <= tt["ghia_literature_rms"]
        )
        ghia_method = (
            metrics["ghia_u_centerline_rms"] <= tt["ghia_method_rms"]
            and metrics["ghia_v_centerline_rms"] <= tt["ghia_method_rms"]
        )
        tight_ref = (
            metrics["cavity_centerline_delta_u_rms"] <= tt["centerline_delta_rms"]
            and metrics["cavity_centerline_delta_v_rms"] <= tt["centerline_delta_rms"]
            and metrics["cavity_field_rel_l2_vs_tight_ref"] <= tt["field_rel_l2"]
        )
        metrics["ghia_literature_gate_pass"] = int(ghia_lit)
        metrics["ghia_method_gate_pass"] = int(ghia_method)
        metrics["tight_ref_gate_pass"] = int(tight_ref)
        metrics["physical_converged"] = int(ghia_lit and ghia_method and tight_ref)
        metrics["eligible_for_score"] = int(metrics["physical_converged"])
    elif base_case_id in {"multi_cylinder_n32", "t_junction_rect", "cylinder_wake_n64", "backward_step_n64"}:
        metrics["mass_imbalance"] = float(_mass_imbalance(case, f, ref_case, ref_f))
        rel_ok = float(err["rel_l2"]) <= 5.0e-4
        mass_ok = abs(float(metrics["mass_imbalance"])) <= 1.0e-4
        metrics["physical_converged"] = int(rel_ok and mass_ok)
        metrics["eligible_for_score"] = int(metrics["physical_converged"])
    else:
        metrics["mass_imbalance"] = float(_mass_imbalance(case, f, ref_case, ref_f))
        metrics["rho_min"] = float(np.min(rho[mask]) if np.any(mask) else np.nan)
        metrics["rho_max"] = float(np.max(rho[mask]) if np.any(mask) else np.nan)
    return metrics


def _channel_accuracy_tolerances(level: int) -> dict[str, float]:
    table = {
        1: {
            "channel_core_rel_l2_analytic": 6.0e-3,
            "channel_rel_l2_vs_tight_picard": 3.0e-3,
            "channel_core_flux_cv": 5.0e-4,
            "channel_boundary_flux_imbalance": 1.0e-3,
        },
        2: {
            "channel_core_rel_l2_analytic": 2.0e-3,
            "channel_rel_l2_vs_tight_picard": 2.0e-3,
            "channel_core_flux_cv": 4.0e-4,
            "channel_boundary_flux_imbalance": 7.5e-4,
        },
        3: {
            "channel_core_rel_l2_analytic": 1.0e-3,
            "channel_rel_l2_vs_tight_picard": 1.5e-3,
            "channel_core_flux_cv": 3.0e-4,
            "channel_boundary_flux_imbalance": 5.0e-4,
        },
    }
    return table.get(int(level), table[3])


def _cavity_physical_tolerances(base_case_id: str) -> dict[str, float]:
    if base_case_id == "cavity_re100_n33":
        return {
            "ghia_literature_rms": 3.0e-2,
            "ghia_method_rms": 1.0e-2,
            "centerline_delta_rms": 5.0e-3,
            "field_rel_l2": 4.0e-3,
        }
    if base_case_id == "cavity_re400_n49":
        return {
            "ghia_literature_rms": 6.0e-2,
            "ghia_method_rms": 3.0e-2,
            "centerline_delta_rms": 1.0e-2,
            "field_rel_l2": 3.0e-3,
        }
    return {
        "ghia_literature_rms": 8.0e-2,
        "ghia_method_rms": 5.0e-2,
        "centerline_delta_rms": 1.5e-2,
        "field_rel_l2": 5.0e-3,
    }


def _accuracy_components(base_case_id: str, row: dict):
    tol = max(float(row["tol"]), 1.0e-30)
    components = []
    rel_l2 = float(row.get("rel_l2_vs_ref", row.get("rel_l2_vs_picard", float("inf"))))
    if base_case_id == "channel_poiseuille_rect":
        level = int(float(row.get("scaling_level", 3)))
        tt = _channel_accuracy_tolerances(level)
        components.append(("channel_core_rel_l2_analytic", abs(float(row.get("channel_core_rel_l2_analytic", rel_l2))), tt["channel_core_rel_l2_analytic"]))
        tight_component = row.get("channel_rel_l2_vs_tight_picard", rel_l2)
        try:
            tight_component = float(tight_component)
            if np.isnan(tight_component):
                tight_component = 0.0
        except Exception:
            tight_component = 0.0
        components.append(("channel_rel_l2_vs_tight_picard", abs(tight_component), tt["channel_rel_l2_vs_tight_picard"]))
        components.append(("channel_core_flux_cv", abs(float(row.get("channel_core_flux_cv", 0.0))), tt["channel_core_flux_cv"]))
        components.append(("channel_boundary_flux_imbalance", abs(float(row.get("channel_boundary_flux_imbalance", row.get("flux_imbalance", 0.0)))), tt["channel_boundary_flux_imbalance"]))
    elif base_case_id == "couette_n32":
        components.append(("rel_l2", rel_l2, 1.0e-3))
    elif base_case_id in {"cavity_re100_n33", "cavity_re400_n49", "cavity_re1000_n129"}:
        tt = _cavity_physical_tolerances(base_case_id)
        components.append(("ghia_u_centerline_rms", abs(float(row.get("ghia_u_centerline_rms", rel_l2))), tt["ghia_method_rms"]))
        components.append(("ghia_v_centerline_rms", abs(float(row.get("ghia_v_centerline_rms", rel_l2))), tt["ghia_method_rms"]))
        components.append(("cavity_centerline_delta_u_rms", abs(float(row.get("cavity_centerline_delta_u_rms", rel_l2))), tt["centerline_delta_rms"]))
        components.append(("cavity_centerline_delta_v_rms", abs(float(row.get("cavity_centerline_delta_v_rms", rel_l2))), tt["centerline_delta_rms"]))
        components.append(("cavity_field_rel_l2_vs_tight_ref", abs(float(row.get("cavity_field_rel_l2_vs_tight_ref", row.get("rel_l2_vs_picard", rel_l2)))), tt["field_rel_l2"]))
    elif base_case_id in {"multi_cylinder_n32", "t_junction_rect", "cylinder_wake_n64", "backward_step_n64"}:
        components.append(("rel_l2", rel_l2, 5.0e-4))
        components.append(("mass_imbalance", abs(float(row.get("mass_imbalance", 0.0))), 1.0e-4))
    else:
        components.append(("rel_l2", rel_l2, max(20.0 * tol, 1.0e-3)))
    return components


def _load_tight_reference(base_case_id: str, factory):
    ref_path = REF_DIR / f"{base_case_id}_no_force_ref.npz"
    if not ref_path.exists():
        return None, None, None
    data = np.load(ref_path, allow_pickle=False)
    case = factory()
    ref_f = data["f"]
    try:
        if getattr(ref_f, "shape", None) is None or tuple(ref_f.shape[-2:]) != tuple(_case_grid_shape(case)):
            return None, None, None
    except Exception:
        return None, None, None
    return case, ref_f, ref_path


def _hist_to_list(hist):
    return [[int(a), float(b), int(c), float(d)] for a, b, c, d in hist]


def _has_accepted_phase(hist, phase: str) -> bool:
    diagnostics = getattr(hist, "diagnostics", None)
    if not diagnostics:
        return False
    for row in diagnostics:
        if str(row.get("phase", "")) == phase and int(row.get("accepted", 0) or 0):
            return True
    return False


def _has_phase(hist, phase: str) -> bool:
    diagnostics = getattr(hist, "diagnostics", None)
    if not diagnostics:
        return False
    return any(str(row.get("phase", "")) == phase for row in diagnostics)


def _read_float(row: dict, key: str, default=float("nan")):
    try:
        return float(row[key])
    except Exception:
        return float(default)


def _resample_reference_field(src_case, src_f, dst_case):
    if getattr(src_f, "shape", None) is None:
        return src_f
    dst_ny, dst_nx = _case_grid_shape(dst_case)
    if tuple(src_f.shape[-2:]) == (dst_ny, dst_nx):
        return src_f
    f = np.asarray(src_f, dtype=np.float64)
    rho_src = np.sum(f, axis=0)
    ux_src = np.tensordot(CX, f, axes=(0, 0)) / np.maximum(rho_src, 1e-15)
    uy_src = np.tensordot(CY, f, axes=(0, 0)) / np.maximum(rho_src, 1e-15)
    src_ny, src_nx = int(rho_src.shape[0]), int(rho_src.shape[1])
    y_src = np.linspace(0.0, 1.0, src_ny, dtype=np.float64)
    x_src = np.linspace(0.0, 1.0, src_nx, dtype=np.float64)
    y_dst = np.linspace(0.0, 1.0, dst_ny, dtype=np.float64)
    x_dst = np.linspace(0.0, 1.0, dst_nx, dtype=np.float64)

    def interp2(field):
        tmp = np.empty((dst_ny, src_nx), dtype=np.float64)
        for j in range(src_nx):
            tmp[:, j] = np.interp(y_dst, y_src, field[:, j])
        out = np.empty((dst_ny, dst_nx), dtype=np.float64)
        for i in range(dst_ny):
            out[i, :] = np.interp(x_dst, x_src, tmp[i, :])
        return out

    rho_i = interp2(rho_src)
    ux_i = interp2(ux_src)
    uy_i = interp2(uy_src)
    return equilibrium(rho_i, ux_i, uy_i)


def _ghia_centerline_error(case_ref, f_ref, case, f):
    from ghia_validation import get_ghia_data

    _, ux_ref, uy_ref = macro_of(case_ref, f_ref)
    _, ux, uy = macro_of(case, f)
    y_g, u_g, x_g, v_g = get_ghia_data(int(round(getattr(case_ref, "Re", getattr(case, "Re", 0)))))
    y_grid = np.linspace(0.0, 1.0, case.N, dtype=np.float64)
    x_grid = np.linspace(0.0, 1.0, case.N, dtype=np.float64)
    mid = case.N // 2
    u_line = ux[:, mid] / max(float(getattr(case_ref, "U_wall", 1.0)), 1.0e-30)
    v_line = uy[mid, :] / max(float(getattr(case_ref, "U_wall", 1.0)), 1.0e-30)
    u_interp = np.interp(y_g, y_grid, u_line)
    v_interp = np.interp(x_g, x_grid, v_line)
    du = u_interp - u_g
    dv = v_interp - v_g
    ref_u = np.asarray(u_g, dtype=np.float64)
    ref_v = np.asarray(v_g, dtype=np.float64)
    den_u = max(float(np.sqrt(np.sum(ref_u * ref_u))), 1.0e-30)
    den_v = max(float(np.sqrt(np.sum(ref_v * ref_v))), 1.0e-30)
    u_l2 = float(np.sqrt(np.sum(du * du)) / den_u)
    v_l2 = float(np.sqrt(np.sum(dv * dv)) / den_v)
    u_linf = float(np.max(np.abs(du)) if du.size else 0.0)
    v_linf = float(np.max(np.abs(dv)) if dv.size else 0.0)
    u_rms = float(np.sqrt(np.mean(du * du)) if du.size else 0.0)
    v_rms = float(np.sqrt(np.mean(dv * dv)) if dv.size else 0.0)
    return {
        "vel_abs_l2": float(np.sqrt(np.sum(du * du + dv * dv))),
        "vel_abs_linf": float(max(u_linf, v_linf)),
        "vel_abs_rms": float(np.sqrt(np.mean(np.concatenate([du * du, dv * dv])) if du.size else 0.0)),
        "rel_l2": float(0.5 * (u_l2 + v_l2)),
        "linf": float(max(u_linf, v_linf)),
        "rms": float(np.sqrt(0.5 * (u_rms * u_rms + v_rms * v_rms))),
        "u_rel_l2": float(u_l2),
        "v_rel_l2": float(v_l2),
        "u_linf": float(u_linf),
        "v_linf": float(v_linf),
        "u_rms": float(u_rms),
        "v_rms": float(v_rms),
    }


def _cavity_centerline_delta(case, f, ref_f) -> tuple[float, float]:
    _rho_ref, ux_ref, uy_ref = macro_of(case, ref_f)
    _rho, ux, uy = macro_of(case, f)
    ny, nx = _case_grid_shape(case)
    mid_x = nx // 2
    mid_y = ny // 2
    u_scale = max(float(getattr(case, "U_wall", 1.0)), 1.0e-30)
    du = (ux[:, mid_x] - ux_ref[:, mid_x]) / u_scale
    dv = (uy[mid_y, :] - uy_ref[mid_y, :]) / u_scale
    return (
        float(np.sqrt(np.mean(du * du)) if du.size else 0.0),
        float(np.sqrt(np.mean(dv * dv)) if dv.size else 0.0),
    )


def _analytic_channel_reference(case):
    ny, nx = _case_grid_shape(case)
    rho = np.ones((ny, nx), dtype=np.float64)
    if hasattr(case, "analytical_ux"):
        ux = np.asarray(case.analytical_ux(), dtype=np.float64)
        uy = np.zeros_like(ux)
        return equilibrium(rho, ux, uy)
    if hasattr(case, "_initial_profile"):
        ux_prof = np.asarray(case._initial_profile(), dtype=np.float64).reshape(ny, 1)
    else:
        y = np.arange(ny, dtype=np.float64)
        L = float(max(ny - 1, 1))
        ubar = float(getattr(case, "U_in", 0.0)) * (ny - 1.0) / ny
        ux_prof = (6.0 * ubar * (y / L) * (1.0 - y / L)).reshape(ny, 1)
    ux = np.tile(ux_prof, (1, nx))
    uy = np.zeros_like(ux)
    return equilibrium(rho, ux, uy)


def _channel_reference_kind(case) -> str:
    x_bc = str(getattr(case, "x_bc", "periodic")).lower()
    fx = np.asarray(getattr(case, "Fx", 0.0))
    fy = np.asarray(getattr(case, "Fy", 0.0))
    has_force = bool(np.any(np.abs(fx) > 0.0) or np.any(np.abs(fy) > 0.0))
    if x_bc == "periodic" and (not has_force):
        return "zero_flow"
    if x_bc == "inlet_outlet":
        return "inlet_outlet"
    return "unknown"


def _analytic_couette_reference(case):
    ny, nx = _case_grid_shape(case)
    rho = np.ones((ny, nx), dtype=np.float64)
    ux = np.asarray(case.analytical_ux(), dtype=np.float64)
    uy = np.zeros_like(ux)
    return equilibrium(rho, ux, uy)




def load_existing_rows(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open('r', encoding='utf-8') as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            rows.append({
                'base_case_id': row['base_case_id'],
                'scaling_level': int(row['scaling_level']),
                'case_id': row['case_id'],
                'case_label': row['case_label'],
                'method': row['method'],
                'tol': float(row['tol']),
                'N': int(row['N']),
                'lbe_calls': int(float(row['lbe_calls'])),
                'wall_seconds': _read_float(row, "wall_seconds"),
                'final_residual': _read_float(row, "final_residual", float("inf")),
                'final_residual_kind': row.get("final_residual_kind", ""),
                'final_macro_l2_residual': _read_float(row, "final_macro_l2_residual", _read_float(row, "final_residual")),
                'final_macro_l2_pressure': _read_float(row, "final_macro_l2_pressure"),
                'final_macro_l2_ux': _read_float(row, "final_macro_l2_ux"),
                'final_macro_l2_uy': _read_float(row, "final_macro_l2_uy"),
                'final_macro_l2_uz': _read_float(row, "final_macro_l2_uz"),
                'final_f_rms_residual': _read_float(row, "final_f_rms_residual"),
                'initial_macro_l2_residual': _read_float(row, "initial_macro_l2_residual"),
                'relative_macro_l2_residual': _read_float(row, "relative_macro_l2_residual"),
                'relative_plateau_window': int(float(row.get("relative_plateau_window", 0) or 0)),
                'relative_plateau_eps': _read_float(row, "relative_plateau_eps"),
                'macro_change_eps': _read_float(row, "macro_change_eps"),
                'relative_floor_eps': _read_float(row, "relative_floor_eps"),
                'min_lbe_calls': int(float(row.get("min_lbe_calls", 0) or 0)),
                'max_lbe_calls': int(float(row.get("max_lbe_calls", 0) or 0)),
                'plateau_improvement': _read_float(row, "plateau_improvement"),
                'macro_change': _read_float(row, "macro_change"),
                'relative_plateau': int(float(row.get("relative_plateau", 0) or 0)),
                'macro_change_pass': int(float(row.get("macro_change_pass", 0) or 0)),
                'relative_floor_pass': int(float(row.get("relative_floor_pass", 0) or 0)),
                'min_lbe_pass': int(float(row.get("min_lbe_pass", 0) or 0)),
                'converged': int(row['converged']),
                'rel_l2_vs_picard': _read_float(row, "rel_l2_vs_picard"),
                'linf_vs_picard': _read_float(row, "linf_vs_picard"),
                'rms_vs_picard': _read_float(row, "rms_vs_picard"),
                'vel_abs_l2_vs_picard': _read_float(row, "vel_abs_l2_vs_picard"),
                'vel_abs_linf_vs_picard': _read_float(row, "vel_abs_linf_vs_picard"),
                'vel_abs_rms_vs_picard': _read_float(row, "vel_abs_rms_vs_picard"),
                'rho_abs_l2_vs_picard': _read_float(row, "rho_abs_l2_vs_picard"),
                'rho_abs_linf_vs_picard': _read_float(row, "rho_abs_linf_vs_picard"),
                'rho_abs_rms_vs_picard': _read_float(row, "rho_abs_rms_vs_picard"),
                'reference_source': row.get("reference_source", ""),
                'ref_velocity_l2': _read_float(row, "ref_velocity_l2"),
                'run_mode': row.get("run_mode", "legacy"),
                'method_tol': _read_float(row, "method_tol", _read_float(row, "tol")),
                'reference_tol': _read_float(row, "reference_tol", _read_float(row, "tol")),
                'reference_converged': int(float(row.get("reference_converged", row.get("converged", 0)) or 0)),
                'reference_final_residual': _read_float(row, "reference_final_residual", _read_float(row, "final_residual")),
                'cache_label_picard': row.get("cache_label_picard", ""),
                'cache_label_method': row.get("cache_label_method", ""),
                'cache_hash_picard': row.get("cache_hash_picard", ""),
                'cache_hash_method': row.get("cache_hash_method", ""),
                'finite_fields': int(float(row.get("finite_fields", 1) or 1)),
                'history_wall_monotone': int(float(row.get("history_wall_monotone", 1) or 1)),
                'history_final_consistent': int(float(row.get("history_final_consistent", 1) or 1)),
                'flux_imbalance': _read_float(row, "flux_imbalance"),
                'channel_boundary_flux_imbalance': _read_float(row, "channel_boundary_flux_imbalance", _read_float(row, "flux_imbalance")),
                'channel_core_flux_cv': _read_float(row, "channel_core_flux_cv"),
                'channel_core_rel_l2_analytic': _read_float(row, "channel_core_rel_l2_analytic"),
                'channel_full_rel_l2_analytic': _read_float(row, "channel_full_rel_l2_analytic"),
                'channel_rel_l2_vs_tight_picard': _read_float(row, "channel_rel_l2_vs_tight_picard", _read_float(row, "rel_l2_vs_ref")),
                'velocity_span_rel': _read_float(row, "velocity_span_rel"),
                'max_jump_rel': _read_float(row, "max_jump_rel"),
                'mass_imbalance': _read_float(row, "mass_imbalance"),
                'rho_min': _read_float(row, "rho_min"),
                'rho_max': _read_float(row, "rho_max"),
                'ghia_u_centerline_rms': _read_float(row, "ghia_u_centerline_rms"),
                'ghia_v_centerline_rms': _read_float(row, "ghia_v_centerline_rms"),
                'ghia_u_centerline_linf': _read_float(row, "ghia_u_centerline_linf"),
                'ghia_v_centerline_linf': _read_float(row, "ghia_v_centerline_linf"),
                'centerline_profile_rms': _read_float(row, "centerline_profile_rms"),
                'pressure_drop': _read_float(row, "pressure_drop"),
                'reattachment_x': _read_float(row, "reattachment_x"),
            })
    return rows


def row_for(
    base_case_id,
    case_id,
    case_label,
    tol,
    ref_case,
    ref_f,
    method,
    case,
    f,
    hist,
    wall,
    eval_case=None,
    eval_f=None,
    *,
    cache_label_picard: str | None = None,
    cache_label_method: str | None = None,
):
    final_res = float(_macro_l2_residual_value(case, f))
    final_f_rms_res = float(_f_rms_residual_value(case, f))
    macro_res_total, macro_res_p, macro_res_ux, macro_res_uy, macro_res_uz = _macro_l2_residual_components(case, f)
    final_res = float(macro_res_total)
    lbe = int(hist[-1][2]) if hist else 0
    if hist and len(hist[-1]) >= 4 and np.isfinite(float(hist[-1][3])):
        wall = float(hist[-1][3])
    ref_f_fit = _resample_reference_field(ref_case, ref_f, case)
    fluid = _fluid_mask(case)
    err = velocity_error(case, ref_f_fit, case, f, fluid_mask=fluid)
    if base_case_id in {"channel_n32", "channel_poiseuille_rect"}:
        kind = _channel_reference_kind(ref_case)
        if kind == "inlet_outlet":
            ref_f_eval = _analytic_channel_reference(ref_case)
            ref_f_eval = _resample_reference_field(ref_case, ref_f_eval, case)
            err_eval = velocity_error(case, ref_f_eval, case, f, fluid_mask=fluid)
            ref_source = "analytic_poiseuille"
        elif kind == "zero_flow":
            err_eval = err
            ref_source = "analytic_zero_flow"
        else:
            err_eval = err
            ref_source = "tight_ref"
    elif base_case_id == "couette_n32":
        ref_f_eval = _analytic_couette_reference(ref_case)
        ref_f_eval = _resample_reference_field(ref_case, ref_f_eval, case)
        err_eval = velocity_error(case, ref_f_eval, case, f, fluid_mask=fluid)
        ref_source = "analytic_couette"
    elif base_case_id in {"cavity_re100_n33", "cavity_re400_n49", "cavity_re1000_n129"}:
        ref_f_eval = _resample_reference_field(ref_case, ref_f, case)
        err_eval = _ghia_centerline_error(case, ref_f_eval, case, f)
        ref_source = "ghia_centerline"
    elif eval_case is None or eval_f is None:
        err_eval = err
        ref_source = "picard"
    else:
        eval_f_fit = _resample_reference_field(eval_case, eval_f, case)
        err_eval = velocity_error(case, eval_f_fit, case, f, fluid_mask=fluid)
        ref_source = "tight_ref"
    rho_ref, ux_ref, uy_ref = macro_of(case, ref_f_fit)
    rho, ux, uy = macro_of(case, f)
    rho_err = rho[fluid] - rho_ref[fluid]
    ref_velocity_l2 = float(np.sqrt(np.sum(ux_ref[fluid] * ux_ref[fluid] + uy_ref[fluid] * uy_ref[fluid])))
    finite_fields = bool(_finite_fields(f, final_res, wall, err["rel_l2"], err["linf"], err["rms"], err_eval["rel_l2"], err_eval["linf"], err_eval["rms"]))
    history_monotone = bool(_strictly_monotone_wall(hist))
    history_consistent = bool(_final_row_consistent(hist, final_res, lbe, wall))
    cache_label_picard = str(cache_label_picard or f"tol{float(tol):.2e}")
    cache_label_method = str(cache_label_method or f"tol{float(tol):.2e}")
    cache_hash_picard = _cache_key("picard_lbm", cache_label_picard)
    cache_hash_method = _cache_key(method, cache_label_method)
    case_specific = _case_specific_metrics(base_case_id, case, f, rho, ux, uy, ref_case, ref_f_fit, err, err_eval)
    unified_macro_mode = bool(_has_phase(hist, "unified_macro_l2_tail"))
    unified_macro_converged = bool(_has_accepted_phase(hist, "unified_macro_l2_converged"))
    rel_macro_stats = getattr(hist, "relative_macro_l2_stats", {}) or {}
    rel_macro_mode = bool(rel_macro_stats)
    plateau_converged = bool(
        unified_macro_converged
        or (
            method == "proposed"
            and base_case_id in {"cavity_re400_n49", "cavity_re1000_n129"}
            and _has_accepted_phase(hist, "cavity_residual_plateau_converged")
        )
    )
    macro_plateau_mode = bool(
        unified_macro_mode
        or (
            method == "proposed"
            and base_case_id in {"cavity_re400_n49", "cavity_re1000_n129"}
            and _has_phase(hist, "cavity_residual_plateau_tail")
        )
    )
    residual_converged = bool(int(rel_macro_stats.get("relative_macro_converged", 0))) if rel_macro_mode else bool(np.isfinite(final_res) and final_res < 5.0 * tol)
    converged = bool(residual_converged) if rel_macro_mode else bool(plateau_converged if macro_plateau_mode else (residual_converged or plateau_converged))
    row = {
        "base_case_id": base_case_id,
        "scaling_level": int(case_id.split("__")[-1].rstrip("x")) if "__" in case_id else 1,
        "case_id": case_id,
        "case_label": case_label,
        "method": method,
        "tol": float(tol),
        "N": int(getattr(case, "N", max(_case_grid_shape(case)))),
        "lbe_calls": int(lbe),
        "wall_seconds": float(wall),
        "final_residual": float(final_res),
        "final_residual_kind": "macro_l2_p_ux_uy_uz",
        "final_macro_l2_residual": float(final_res),
        "final_macro_l2_pressure": float(macro_res_p),
        "final_macro_l2_ux": float(macro_res_ux),
        "final_macro_l2_uy": float(macro_res_uy),
        "final_macro_l2_uz": float(macro_res_uz),
        "final_f_rms_residual": float(final_f_rms_res),
        "initial_macro_l2_residual": float(rel_macro_stats.get("initial_macro_l2_residual", _macro_l2_residual_value(case, case.initial_field()))),
        "relative_macro_l2_residual": float(rel_macro_stats.get("relative_macro_l2_residual", final_res / max(_macro_l2_residual_value(case, case.initial_field()), 1.0e-300))),
        "relative_plateau_window": int(rel_macro_stats.get("relative_plateau_window", 0)),
        "relative_plateau_eps": float(rel_macro_stats.get("relative_plateau_eps", float("nan"))),
        "macro_change_eps": float(rel_macro_stats.get("macro_change_eps", float("nan"))),
        "relative_floor_eps": float(rel_macro_stats.get("relative_floor_eps", float("nan"))),
        "min_lbe_calls": int(rel_macro_stats.get("min_lbe_calls", 0)),
        "max_lbe_calls": int(rel_macro_stats.get("max_lbe_calls", 0)),
        "plateau_improvement": float(rel_macro_stats.get("plateau_improvement", float("nan"))),
        "macro_change": float(rel_macro_stats.get("macro_change", float("nan"))),
        "relative_plateau": int(rel_macro_stats.get("relative_plateau", 0)),
        "macro_change_pass": int(rel_macro_stats.get("macro_change_pass", 0)),
        "relative_floor_pass": int(rel_macro_stats.get("relative_floor_pass", 0)),
        "min_lbe_pass": int(rel_macro_stats.get("min_lbe_pass", 0)),
        "converged": int(converged),
        "residual_converged": int(residual_converged),
        "plateau_converged": int(plateau_converged),
        "convergence_mode": (
            "unified_macro_l2_initial_relative_plateau"
            if unified_macro_converged
            else (
                "unified_macro_l2_plateau_not_reached"
                if unified_macro_mode
                else (
                    "macro_l2_initial_relative_plateau"
                    if plateau_converged
                    else ("macro_l2_plateau_not_reached" if macro_plateau_mode else "residual_threshold")
                )
            )
        ),
        "finite_fields": int(finite_fields),
        "history_wall_monotone": int(history_monotone),
        "history_final_consistent": int(history_consistent),
        "cache_label_picard": cache_label_picard,
        "cache_label_method": cache_label_method,
        "cache_hash_picard": cache_hash_picard,
        "cache_hash_method": cache_hash_method,
        "rel_l2_vs_picard": float(err["rel_l2"]),
        "linf_vs_picard": float(err["linf"]),
        "rms_vs_picard": float(err["rms"]),
        "vel_abs_l2_vs_picard": float(err["vel_abs_l2"]),
        "vel_abs_linf_vs_picard": float(err["vel_abs_linf"]),
        "vel_abs_rms_vs_picard": float(err["vel_abs_rms"]),
        "rel_l2_vs_ref": float(err_eval["rel_l2"]),
        "linf_vs_ref": float(err_eval["linf"]),
        "rms_vs_ref": float(err_eval["rms"]),
        "vel_abs_l2_vs_ref": float(err_eval["vel_abs_l2"]),
        "vel_abs_linf_vs_ref": float(err_eval["vel_abs_linf"]),
        "vel_abs_rms_vs_ref": float(err_eval["vel_abs_rms"]),
        "reference_source": ref_source,
        "ref_velocity_l2": ref_velocity_l2,
        "rho_abs_l2_vs_picard": float(np.sqrt(np.sum(rho_err * rho_err))),
        "rho_abs_linf_vs_picard": float(max(np.max(np.abs(rho_err)), 0.0) if rho_err.size else 0.0),
        "rho_abs_rms_vs_picard": float(np.sqrt(np.mean(rho_err * rho_err)) if rho_err.size else 0.0),
    }
    row.update({k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in case_specific.items()})
    if base_case_id == "channel_poiseuille_rect":
        accuracy_score, accuracy_pass, _accuracy_detail = _accuracy_score(row)
        row["physical_converged"] = int(accuracy_pass)
        row["eligible_for_score"] = int(accuracy_pass)
    return row


def annotate_run_metadata(
    row,
    *,
    run_mode,
    method_tol,
    reference_tol,
    reference_converged,
    reference_final_residual,
    cache_label_picard,
    cache_label_method,
):
    row["run_mode"] = str(run_mode)
    row["method_tol"] = float(method_tol)
    row["reference_tol"] = float(reference_tol)
    row["reference_converged"] = int(reference_converged)
    row["reference_final_residual"] = float(reference_final_residual)
    row["cache_label_picard"] = str(cache_label_picard)
    row["cache_label_method"] = str(cache_label_method)
    row["cache_hash_picard"] = _cache_key("picard_lbm", str(cache_label_picard))
    row["cache_hash_method"] = _cache_key(str(row["method"]), str(cache_label_method))
    return row


def _score_case_rows(rows):
    by_case = {}
    for row in rows:
        if row.get("base_case_id") not in SCORING_CASE_IDS:
            continue
        by_case.setdefault(row["case_id"], []).append(row)
    return by_case


def _log_normalized_score(value: float, best: float, worst: float) -> float:
    if not (np.isfinite(value) and value > 0.0):
        return 0.0
    if not (np.isfinite(best) and np.isfinite(worst) and best > 0.0 and worst > 0.0) or abs(math.log(worst) - math.log(best)) < 1.0e-15:
        return 0.5
    lo = math.log(best)
    hi = math.log(worst)
    x = (math.log(value) - lo) / max(hi - lo, 1.0e-30)
    return float(np.clip(1.0 - x, 0.0, 1.0))


def _simplicity_score(method: str) -> float:
    return {
        "picard_lbm": 1.00,
        "proposed": 0.75,
        "preconditioned_lbm": 0.60,
        "anderson_lbm": 0.55,
        "dual_time_mg_lbm": 0.45,
        "inexact_newton_lbe": 0.40,
    }.get(method, 0.50)


def _accuracy_score(row: dict) -> tuple[float, bool, list[dict]]:
    components = _accuracy_components(row["base_case_id"], row)
    if not components:
        return 0.5, False, []
    zs = []
    detail = []
    for name, err, tol in components:
        tol = max(float(tol), 1.0e-30)
        z = float(abs(err) / tol)
        zs.append(z)
        detail.append({"name": name, "error": float(err), "tolerance": float(tol), "z": z})
    E = float(np.sqrt(np.mean(np.square(zs)))) if zs else float("inf")
    score = float(max(0.0, 1.0 - min(E, 2.0) / 2.0))
    return score, bool(E <= 1.0), detail


def _case_hard_gates(row: dict) -> dict:
    finite_args = [row.get("lbe_calls"), row.get("wall_seconds"), row.get("final_residual"), row.get("rel_l2_vs_ref"), row.get("linf_vs_ref"), row.get("rms_vs_ref")]
    if row.get("base_case_id") == "channel_poiseuille_rect":
        finite_args.extend([
            row.get("channel_core_rel_l2_analytic"),
            row.get("channel_full_rel_l2_analytic"),
            row.get("channel_rel_l2_vs_tight_picard"),
            row.get("channel_core_flux_cv"),
            row.get("channel_boundary_flux_imbalance", row.get("flux_imbalance")),
        ])
    if row.get("base_case_id") in {"cavity_re100_n33", "cavity_re400_n49", "cavity_re1000_n129"}:
        finite_args.extend([
            row.get("ghia_u_centerline_rms"),
            row.get("ghia_v_centerline_rms"),
            row.get("cavity_centerline_delta_u_rms"),
            row.get("cavity_centerline_delta_v_rms"),
            row.get("cavity_field_rel_l2_vs_tight_ref"),
        ])
    finite_fields = bool(row.get("finite_fields", 0)) and _finite_fields(*finite_args)
    cache_hash_valid = (
        str(row.get("cache_hash_picard", "")) == _cache_key("picard_lbm", str(row.get("cache_label_picard", "")))
        and str(row.get("cache_hash_method", "")) == _cache_key(str(row.get("method", "")), str(row.get("cache_label_method", "")))
    )
    residual_pass = bool(row.get("converged", 0)) and np.isfinite(float(row.get("final_residual", float("inf"))))
    accuracy_score, accuracy_pass, accuracy_detail = _accuracy_score(row)
    history_monotone = bool(row.get("history_wall_monotone", 0))
    history_consistent = bool(row.get("history_final_consistent", 0))
    physical_converged = bool(row.get("physical_converged", 1))
    hard_pass = bool(
        finite_fields
        and cache_hash_valid
        and residual_pass
        and accuracy_pass
        and physical_converged
        and history_monotone
        and history_consistent
    )
    return {
        "finite_fields": int(finite_fields),
        "cache_hash_valid": int(cache_hash_valid),
        "residual_pass": int(residual_pass),
        "accuracy_pass": int(accuracy_pass),
        "history_wall_monotone": int(history_monotone),
        "history_final_consistent": int(history_consistent),
        "physical_converged": int(physical_converged),
        "hard_pass": int(hard_pass),
        "accuracy_score": float(accuracy_score),
        "accuracy_detail": accuracy_detail,
    }


def score_strict(rows):
    by_case = _score_case_rows(rows)

    case_results = []
    for case_id, case_rows in by_case.items():
        prop = next((r for r in case_rows if r["method"] == "proposed"), None)
        if prop is None:
            continue
        fixed = [r for r in case_rows if r["method"] != "proposed"]
        eligible = [
            r
            for r in fixed
            if r["converged"]
            and np.isfinite(r["final_residual"])
            and r["lbe_calls"] > 0
            and r["wall_seconds"] > 0
            and bool(r.get("finite_fields", 1))
            and bool(_case_hard_gates(r)["hard_pass"])
        ]
        acc_eligible = [r for r in fixed if np.isfinite(r["rel_l2_vs_ref"]) and bool(r.get("finite_fields", 1))]
        if not acc_eligible:
            acc_eligible = fixed
        best_lbe = min((r["lbe_calls"] for r in eligible if r["lbe_calls"] > 0), default=10**18)
        best_wall = min((r["wall_seconds"] for r in eligible if r["wall_seconds"] > 0), default=float("inf"))
        best_acc = min((r["rel_l2_vs_ref"] for r in acc_eligible), default=float("inf"))

        lbe_win = bool(prop["lbe_calls"] <= best_lbe)
        wall_win = bool(prop["wall_seconds"] <= best_wall)
        acc_win = bool(prop["rel_l2_vs_ref"] <= best_acc * 1.001 + 1e-12)
        gates = _case_hard_gates(prop)
        conv = bool(prop["converged"])
        case_results.append(
            {
                "case_id": case_id,
                "base_case_id": prop["base_case_id"],
                "scaling_level": prop["scaling_level"],
                "case_pass": int(conv and lbe_win and wall_win and acc_win and bool(gates["hard_pass"])),
                "converged": int(conv),
                "lbe_win": int(lbe_win),
                "wall_win": int(wall_win),
                "acc_win": int(acc_win),
                "hard_pass": int(gates["hard_pass"]),
                "proposed_lbe": prop["lbe_calls"],
                "best_fixed_lbe": int(best_lbe) if best_lbe < 10**18 else None,
                "proposed_wall": prop["wall_seconds"],
                "best_fixed_wall": best_wall,
                "proposed_rel_l2": prop["rel_l2_vs_ref"],
                "best_fixed_rel_l2": best_acc,
            }
        )
    pass_count = sum(c["case_pass"] for c in case_results)
    lbe_wins = sum(c["lbe_win"] for c in case_results)
    wall_wins = sum(c["wall_win"] for c in case_results)
    acc_wins = sum(c["acc_win"] for c in case_results)
    convs = sum(c["converged"] for c in case_results)
    speedups = [
        c["best_fixed_lbe"] / c["proposed_lbe"] for c in case_results if c["best_fixed_lbe"] is not None and c["proposed_lbe"] > 0
    ]
    return {
        "score_mode": "strict_best_of_reference",
        "all_pass": int(bool(case_results and pass_count == len(case_results))),
        "case_count": len(case_results),
        "pass_count": int(pass_count),
        "converged_count": int(convs),
        "lbe_win_count": int(lbe_wins),
        "wall_win_count": int(wall_wins),
        "accuracy_win_count": int(acc_wins),
        "mean_lbe_speedup_vs_best_fixed": float(np.mean(speedups) if speedups else 0.0),
        "case_results": case_results,
    }


def _reference_class_and_floor(row):
    source = str(row.get("reference_source", ""))
    ref_speed = float(row.get("ref_velocity_l2", 0.0))
    if source == "analytic_zero_flow":
        return "degenerate_zero_flow", 0.0, True
    if source in {"tight_ref", "tight_picard", "picard"} and ref_speed < 1.0e-4:
        return "degenerate_tight_reference", 0.0, True
    if source.startswith("analytic_"):
        return "analytic", 1.0e-4, False
    if source == "ghia_centerline":
        return "ghia_centerline", 2.0e-2, False
    return "tight_reference", 1.0e-4, False


def score_paper(rows):
    by_case = _score_case_rows(rows)

    case_results = []
    for case_id, case_rows in by_case.items():
        prop = next((r for r in case_rows if r["method"] == "proposed"), None)
        if prop is None:
            continue
        fixed = [r for r in case_rows if r["method"] != "proposed"]
        eligible_candidates = [
            r for r in fixed
            if r["converged"] and np.isfinite(r["final_residual"]) and r["lbe_calls"] > 0 and r["wall_seconds"] > 0 and np.isfinite(r["rel_l2_vs_ref"]) and bool(r.get("finite_fields", 1))
        ]
        reference_pool = list(eligible_candidates)
        eligible = [r for r in reference_pool if _case_hard_gates(r)["hard_pass"]]

        hard = _case_hard_gates(prop)
        speed_pool = list(reference_pool)
        if (
            bool(hard["hard_pass"])
            and prop["converged"]
            and np.isfinite(prop["final_residual"])
            and prop["lbe_calls"] > 0
            and prop["wall_seconds"] > 0
            and np.isfinite(prop["rel_l2_vs_ref"])
            and bool(prop.get("finite_fields", 1))
        ):
            speed_pool.append(prop)
        if not speed_pool:
            speed_pool = list(reference_pool)

        best_lbe = min((r["lbe_calls"] for r in reference_pool if r["lbe_calls"] > 0), default=10**18)
        best_wall = min((r["wall_seconds"] for r in reference_pool if r["wall_seconds"] > 0), default=float("inf"))
        speed_best_lbe = min((r["lbe_calls"] for r in speed_pool if r["lbe_calls"] > 0), default=10**18)
        speed_worst_lbe = max((float(r["lbe_calls"]) for r in speed_pool if r["lbe_calls"] > 0), default=float("nan"))
        speed_best_wall = min((r["wall_seconds"] for r in speed_pool if r["wall_seconds"] > 0), default=float("inf"))
        speed_worst_wall = max((float(r["wall_seconds"]) for r in speed_pool if r["wall_seconds"] > 0), default=float("nan"))
        best_acc = min((r["rel_l2_vs_ref"] for r in reference_pool if np.isfinite(r["rel_l2_vs_ref"])), default=float("inf"))
        ref_scores = []
        for row in reference_pool:
            acc_score, acc_pass, acc_detail = _accuracy_score(row)
            wall_score = _log_normalized_score(float(row["wall_seconds"]), speed_best_wall, speed_worst_wall)
            lbe_score = _log_normalized_score(float(row["lbe_calls"]), speed_best_lbe, speed_worst_lbe)
            speed_score = 0.70 * wall_score + 0.30 * lbe_score
            simp_score = _simplicity_score(str(row["method"]))
            total_score = 0.50 * speed_score + 0.40 * acc_score + 0.10 * simp_score
            ref_scores.append({
                "method": row["method"],
                "wall_score": float(wall_score),
                "lbe_score": float(lbe_score),
                "speed_score": float(speed_score),
                "accuracy_pass": int(acc_pass),
                "accuracy_score": float(acc_score),
                "simplicity_score": float(simp_score),
                "total_score": float(total_score),
                "accuracy_detail": acc_detail,
            })
        best_ref_total = max((r["total_score"] for r in ref_scores if r["method"] != "proposed"), default=float("nan"))
        prop_acc_score, prop_acc_pass, prop_acc_detail = _accuracy_score(prop)
        if speed_pool:
            prop_wall_score = _log_normalized_score(float(prop["wall_seconds"]), speed_best_wall, speed_worst_wall)
            prop_lbe_score = _log_normalized_score(float(prop["lbe_calls"]), speed_best_lbe, speed_worst_lbe)
        else:
            prop_wall_score = 0.0
            prop_lbe_score = 0.0
        prop_speed_score = 0.70 * prop_wall_score + 0.30 * prop_lbe_score
        prop_simplicity = _simplicity_score(str(prop["method"]))
        prop_total = 0.50 * prop_speed_score + 0.40 * prop_acc_score + 0.10 * prop_simplicity
        conv = bool(prop["converged"])
        reference_pool_status = "fixed_reference_pool" if reference_pool else "no_finite_fixed_reference"
        reference_pool_fallback = 0
        if (
            not reference_pool
            and bool(hard["hard_pass"])
            and conv
            and prop_acc_pass
            and str(prop.get("reference_source", "")) == "tight_ref"
            and np.isfinite(float(prop.get("rel_l2_vs_ref", float("nan"))))
            and np.isfinite(float(prop.get("ref_velocity_l2", float("nan"))))
            and float(prop.get("ref_velocity_l2", 0.0)) > 0.0
        ):
            best_ref_total = 0.0
            reference_pool_status = "no_finite_fixed_reference_tight_ref_fallback"
            reference_pool_fallback = 1
        total_margin = float(prop_total - best_ref_total) if np.isfinite(best_ref_total) else float("nan")
        proposed_pass = int(bool(hard["hard_pass"]) and conv and prop_acc_pass and np.isfinite(best_ref_total) and total_margin > 0.02)
        case_results.append(
            {
                "case_id": case_id,
                "base_case_id": prop["base_case_id"],
                "scaling_level": prop["scaling_level"],
                "case_pass": proposed_pass,
                "hard_pass": int(hard["hard_pass"]),
                "converged": int(conv),
                "accuracy_pass": int(prop_acc_pass),
                "finite_fields": int(hard["finite_fields"]),
                "cache_hash_valid": int(hard["cache_hash_valid"]),
                "residual_pass": int(hard["residual_pass"]),
                "history_wall_monotone": int(hard["history_wall_monotone"]),
                "history_final_consistent": int(hard["history_final_consistent"]),
                "physical_converged": int(hard.get("physical_converged", 1)),
                "proposed_lbe": prop["lbe_calls"],
                "best_fixed_lbe": int(best_lbe) if best_lbe < 10**18 else None,
                "proposed_wall": prop["wall_seconds"],
                "best_fixed_wall": best_wall,
                "reference_pool_status": reference_pool_status,
                "reference_pool_size": int(len(reference_pool)),
                "reference_pool_fallback": int(reference_pool_fallback),
                "speed_pool_size": int(len(speed_pool)),
                "speed_best_lbe": int(speed_best_lbe) if speed_best_lbe < 10**18 else None,
                "speed_worst_lbe": float(speed_worst_lbe),
                "speed_best_wall": float(speed_best_wall),
                "speed_worst_wall": float(speed_worst_wall),
                "proposed_rel_l2": prop["rel_l2_vs_ref"],
                "best_fixed_rel_l2": best_acc,
                "wall_score": float(prop_wall_score),
                "lbe_score": float(prop_lbe_score),
                "speed_score": float(prop_speed_score),
                "accuracy_score": float(prop_acc_score),
                "simplicity_score": float(prop_simplicity),
                "total_score": float(prop_total),
                "best_ref_total": float(best_ref_total),
                "total_margin": float(total_margin),
                "accuracy_detail": prop_acc_detail,
            }
        )
    evaluated = case_results
    pass_count = sum(c["case_pass"] for c in evaluated)
    return {
        "score_mode": "paper_weighted_total_score",
        "all_pass": int(bool(evaluated and pass_count == len(evaluated))),
        "case_count": len(evaluated),
        "pass_count": int(pass_count),
        "best_ref_total_mean": float(np.mean([c["best_ref_total"] for c in case_results]) if case_results else float("nan")),
        "mean_total_score": float(np.mean([c["total_score"] for c in case_results]) if case_results else float("nan")),
        "case_results": case_results,
    }

def load_existing_rows(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            rows.append({
                "base_case_id": row["base_case_id"],
                "scaling_level": int(row["scaling_level"]),
                "case_id": row["case_id"],
                "case_label": row["case_label"],
                "method": row["method"],
                "tol": float(row["tol"]),
                "N": int(row["N"]),
                "lbe_calls": int(float(row["lbe_calls"])),
                "wall_seconds": _read_float(row, "wall_seconds"),
                "final_residual": _read_float(row, "final_residual", float("inf")),
                "final_residual_kind": row.get("final_residual_kind", ""),
                "final_macro_l2_residual": _read_float(row, "final_macro_l2_residual", _read_float(row, "final_residual")),
                "final_macro_l2_pressure": _read_float(row, "final_macro_l2_pressure"),
                "final_macro_l2_ux": _read_float(row, "final_macro_l2_ux"),
                "final_macro_l2_uy": _read_float(row, "final_macro_l2_uy"),
                "final_macro_l2_uz": _read_float(row, "final_macro_l2_uz"),
                "final_f_rms_residual": _read_float(row, "final_f_rms_residual"),
                "initial_macro_l2_residual": _read_float(row, "initial_macro_l2_residual"),
                "relative_macro_l2_residual": _read_float(row, "relative_macro_l2_residual"),
                "relative_plateau_window": int(float(row.get("relative_plateau_window", 0) or 0)),
                "relative_plateau_eps": _read_float(row, "relative_plateau_eps"),
                "macro_change_eps": _read_float(row, "macro_change_eps"),
                "relative_floor_eps": _read_float(row, "relative_floor_eps"),
                "min_lbe_calls": int(float(row.get("min_lbe_calls", 0) or 0)),
                "max_lbe_calls": int(float(row.get("max_lbe_calls", 0) or 0)),
                "plateau_improvement": _read_float(row, "plateau_improvement"),
                "macro_change": _read_float(row, "macro_change"),
                "relative_plateau": int(float(row.get("relative_plateau", 0) or 0)),
                "macro_change_pass": int(float(row.get("macro_change_pass", 0) or 0)),
                "relative_floor_pass": int(float(row.get("relative_floor_pass", 0) or 0)),
                "min_lbe_pass": int(float(row.get("min_lbe_pass", 0) or 0)),
                "converged": int(row["converged"]),
                "residual_converged": int(float(row.get("residual_converged", row.get("converged", 0)) or 0)),
                "plateau_converged": int(float(row.get("plateau_converged", 0) or 0)),
                "convergence_mode": row.get("convergence_mode", "residual_threshold"),
                "rel_l2_vs_picard": _read_float(row, "rel_l2_vs_picard"),
                "linf_vs_picard": _read_float(row, "linf_vs_picard"),
                "rms_vs_picard": _read_float(row, "rms_vs_picard"),
                "vel_abs_l2_vs_picard": _read_float(row, "vel_abs_l2_vs_picard"),
                "vel_abs_linf_vs_picard": _read_float(row, "vel_abs_linf_vs_picard"),
                "vel_abs_rms_vs_picard": _read_float(row, "vel_abs_rms_vs_picard"),
                "rel_l2_vs_ref": _read_float(row, "rel_l2_vs_ref"),
                "linf_vs_ref": _read_float(row, "linf_vs_ref"),
                "rms_vs_ref": _read_float(row, "rms_vs_ref"),
                "vel_abs_l2_vs_ref": _read_float(row, "vel_abs_l2_vs_ref"),
                "vel_abs_linf_vs_ref": _read_float(row, "vel_abs_linf_vs_ref"),
                "vel_abs_rms_vs_ref": _read_float(row, "vel_abs_rms_vs_ref"),
                "reference_source": row.get("reference_source", ""),
                "ref_velocity_l2": _read_float(row, "ref_velocity_l2"),
                "rho_abs_l2_vs_picard": _read_float(row, "rho_abs_l2_vs_picard"),
                "rho_abs_linf_vs_picard": _read_float(row, "rho_abs_linf_vs_picard"),
                "rho_abs_rms_vs_picard": _read_float(row, "rho_abs_rms_vs_picard"),
                "run_mode": row.get("run_mode", "legacy"),
                "method_tol": _read_float(row, "method_tol", _read_float(row, "tol")),
                "reference_tol": _read_float(row, "reference_tol", _read_float(row, "tol")),
                "reference_converged": int(float(row.get("reference_converged", row.get("converged", 0)) or 0)),
                "reference_final_residual": _read_float(row, "reference_final_residual", _read_float(row, "final_residual")),
                "cache_label_picard": row.get("cache_label_picard", ""),
                "cache_label_method": row.get("cache_label_method", ""),
                "cache_hash_picard": row.get("cache_hash_picard", ""),
                "cache_hash_method": row.get("cache_hash_method", ""),
                "finite_fields": int(float(row.get("finite_fields", 1) or 1)),
                "history_wall_monotone": int(float(row.get("history_wall_monotone", 1) or 1)),
                "history_final_consistent": int(float(row.get("history_final_consistent", 1) or 1)),
                "flux_imbalance": _read_float(row, "flux_imbalance"),
                "velocity_span_rel": _read_float(row, "velocity_span_rel"),
                "max_jump_rel": _read_float(row, "max_jump_rel"),
                "mass_imbalance": _read_float(row, "mass_imbalance"),
                "rho_min": _read_float(row, "rho_min"),
                "rho_max": _read_float(row, "rho_max"),
                "ghia_u_centerline_rms": _read_float(row, "ghia_u_centerline_rms"),
                "ghia_v_centerline_rms": _read_float(row, "ghia_v_centerline_rms"),
                "ghia_u_centerline_linf": _read_float(row, "ghia_u_centerline_linf"),
                "ghia_v_centerline_linf": _read_float(row, "ghia_v_centerline_linf"),
                "cavity_centerline_delta_u_rms": _read_float(row, "cavity_centerline_delta_u_rms"),
                "cavity_centerline_delta_v_rms": _read_float(row, "cavity_centerline_delta_v_rms"),
                "cavity_field_rel_l2_vs_tight_ref": _read_float(row, "cavity_field_rel_l2_vs_tight_ref"),
                "ghia_literature_gate_pass": int(float(row.get("ghia_literature_gate_pass", 0) or 0)),
                "ghia_method_gate_pass": int(float(row.get("ghia_method_gate_pass", 0) or 0)),
                "tight_ref_gate_pass": int(float(row.get("tight_ref_gate_pass", 0) or 0)),
                "physical_converged": int(float(row.get("physical_converged", 0) or 0)),
                "eligible_for_score": int(float(row.get("eligible_for_score", 0) or 0)),
                "centerline_profile_rms": _read_float(row, "centerline_profile_rms"),
                "pressure_drop": _read_float(row, "pressure_drop"),
                "reattachment_x": _read_float(row, "reattachment_x"),
            })
    return rows


def write_outputs(rows, metrics):
    OUT.mkdir(parents=True, exist_ok=True)
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "base_case_id",
        "scaling_level",
        "case_id",
        "case_label",
        "method",
        "tol",
        "N",
        "lbe_calls",
                "wall_seconds",
                "final_residual",
                "final_residual_kind",
                "final_macro_l2_residual",
                "final_macro_l2_pressure",
                "final_macro_l2_ux",
                "final_macro_l2_uy",
                "final_macro_l2_uz",
                "final_f_rms_residual",
                "initial_macro_l2_residual",
                "relative_macro_l2_residual",
                "relative_plateau_window",
                "relative_plateau_eps",
                "macro_change_eps",
                "relative_floor_eps",
                "min_lbe_calls",
                "max_lbe_calls",
                "plateau_improvement",
                "macro_change",
                "relative_plateau",
                "macro_change_pass",
                "relative_floor_pass",
                "min_lbe_pass",
                "converged",
        "residual_converged",
        "plateau_converged",
        "convergence_mode",
        "finite_fields",
        "history_wall_monotone",
        "history_final_consistent",
        "cache_label_picard",
        "cache_label_method",
        "cache_hash_picard",
        "cache_hash_method",
        "channel_core_rel_l2_analytic",
        "channel_full_rel_l2_analytic",
        "channel_rel_l2_vs_tight_picard",
        "channel_core_flux_cv",
        "channel_boundary_flux_imbalance",
        "rel_l2_vs_picard",
        "linf_vs_picard",
        "rms_vs_picard",
        "vel_abs_l2_vs_picard",
        "vel_abs_linf_vs_picard",
        "vel_abs_rms_vs_picard",
        "rel_l2_vs_ref",
        "linf_vs_ref",
        "rms_vs_ref",
        "vel_abs_l2_vs_ref",
        "vel_abs_linf_vs_ref",
        "vel_abs_rms_vs_ref",
        "reference_source",
        "ref_velocity_l2",
        "rho_abs_l2_vs_picard",
        "rho_abs_linf_vs_picard",
        "rho_abs_rms_vs_picard",
        "flux_imbalance",
        "velocity_span_rel",
        "max_jump_rel",
        "mass_imbalance",
        "rho_min",
        "rho_max",
        "ghia_u_centerline_rms",
        "ghia_v_centerline_rms",
        "ghia_u_centerline_linf",
        "ghia_v_centerline_linf",
        "cavity_centerline_delta_u_rms",
        "cavity_centerline_delta_v_rms",
        "cavity_field_rel_l2_vs_tight_ref",
        "ghia_literature_gate_pass",
        "ghia_method_gate_pass",
        "tight_ref_gate_pass",
        "physical_converged",
        "eligible_for_score",
        "centerline_profile_rms",
        "pressure_drop",
        "reattachment_x",
        "run_mode",
        "method_tol",
        "reference_tol",
        "reference_converged",
        "reference_final_residual",
    ]
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row.get(k, float("nan")) for k in fields})

    (OUT / "summary.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    if isinstance(metrics, dict) and ("strict" in metrics or "paper" in metrics):
        strict = metrics.get("strict", {})
        paper = metrics.get("paper", {})
        (OUT / "metrics_strict.json").write_text(json.dumps(strict, indent=2), encoding="utf-8")
        (OUT / "metrics_paper.json").write_text(json.dumps(paper, indent=2), encoding="utf-8")
        (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    else:
        (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def parse_csv_arg(value: str, allowed):
    if value == "all":
        return list(allowed)
    parsed = [x.strip() for x in value.split(",") if x.strip()]
    bad = [x for x in parsed if x not in allowed]
    if bad:
        raise ValueError(f"unknown values: {bad}")
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default="2,3", help="comma-separated scaling levels, or all")
    parser.add_argument("--base-cases", default="all", help="comma-separated base case ids, or all")
    parser.add_argument("--methods", default=",".join(METHODS), help="comma-separated method ids")
    parser.add_argument("--picard-ref-tol-scale", default=None, type=float, help="scale factor applied to case tol when computing Picard reference; default uses case-specific audit scale")
    parser.add_argument("--method-tol-scale", default="1.0", type=float, help="scale factor applied to case tol for non-Picard methods")
    parser.add_argument("--run-mode", default="auto", choices=["auto", "strict", "relaxed"], help="label used to keep strict and relaxed rows separate")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-vtk", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--out-dir", default=None, help="override benchmark output directory")
    parser.add_argument("--cache-dir", default=None, help="override benchmark cache directory")
    parser.add_argument("--exact-methods", action="store_true", help="run exactly the requested methods without adding Picard/proposed")
    parser.add_argument("--suppress-reference-row", action="store_true", help="compute/load the Picard reference but do not write its summary/history row")
    parser.add_argument("--recompute-methods", default="", help="comma-separated methods to recompute while preserving other cached methods")
    args = parser.parse_args()

    if args.out_dir:
        global OUT, HIST_DIR, DIAG_DIR, VTK_DIR, CACHE_DIR
        OUT = Path(args.out_dir)
        HIST_DIR = OUT / "histories"
        DIAG_DIR = OUT / "diagnostics"
        VTK_DIR = OUT / "vtk"
        CACHE_DIR = OUT / "npz_cache"
    if args.cache_dir:
        CACHE_DIR = Path(args.cache_dir)

    levels = [int(x) for x in args.levels.split(",") if x.strip()] if args.levels != "all" else [1, 2, 3]
    for level in levels:
        if level not in {1, 2, 3}:
            raise ValueError("levels must be 1, 2, or 3")
    base_ids = parse_csv_arg(args.base_cases, CASE_IDS)
    methods = parse_csv_arg(args.methods, METHODS)
    if not args.exact_methods and "picard_lbm" not in methods:
        methods = ["picard_lbm"] + methods
    if not args.exact_methods and "proposed" not in methods:
        methods = methods + ["proposed"]
    recompute_methods = {x.strip() for x in str(args.recompute_methods).split(",") if x.strip()}
    bad_recompute = [x for x in recompute_methods if x not in METHODS]
    if bad_recompute:
        raise ValueError(f"unknown recompute methods: {bad_recompute}")
    run_mode = args.run_mode
    if run_mode == "auto":
        run_mode = "strict" if args.picard_ref_tol_scale is None and abs(float(args.method_tol_scale) - 1.0) < 1.0e-15 else "relaxed"

    summary_metadata = {
        "output_dir": str(OUT),
        "run_mode": run_mode,
        "canonical_base_cases": list(TARGET_CASE_IDS),
        "auxiliary_base_cases": list(AUXILIARY_CASE_IDS),
        "physics_scaling": {
            "wall_driven_velocity": "U_scale(level) = U_scale(1) / level",
            "tolerance": "tol_scale(level) = tol_scale(1) / level",
            "cavity_refinement": "N_level = 1 + level * (N_1x - 1)",
            "max_steps": "scaled by level^2 with safety caps",
        },
    }

    rows = [] if args.no_resume else load_existing_rows(OUT / "summary.csv")
    existing = {(r["case_id"], r["method"], r.get("run_mode", "legacy"), float(r.get("tol", float("nan")))) for r in rows}

    start = time.perf_counter()
    for base_id in base_ids:
        for level in levels:
            case_id, label, tol, factory = case_factory_scaled(base_id, level)
            ref_tol_scale = reference_tol_scale_for_case(base_id, level) if args.picard_ref_tol_scale is None else float(args.picard_ref_tol_scale)
            picard_tol = float(tol) * float(ref_tol_scale)
            method_tol = float(tol) * float(args.method_tol_scale)
            cache_label_picard = f"tol{picard_tol:.2e}"
            cache_label_method = f"tol{method_tol:.2e}"
            max_steps = max_steps_for_scaled(base_id, level)
            ref_max_steps = reference_max_steps_for_scaled(base_id, level, max_steps)
            if args.picard_ref_tol_scale is None and abs(float(args.method_tol_scale) - 1.0) < 1.0e-15:
                method_tol = float(tol) * method_tol_scale_for_case(base_id, level)
                cache_label_method = f"tol{method_tol:.2e}"

            methods_for_case = [m for m in methods]
            completed = all((case_id, m, run_mode, method_tol) in existing and m not in recompute_methods for m in methods_for_case)
            if completed and not args.no_resume:
                print(f"[skip] {case_id}: already completed")
                continue

            print(f"[case] {case_id}: {label}", flush=True)

            # always ensure reference baseline for this case
            ref_case = factory()
            eval_case, eval_f, eval_path = _load_tight_reference(base_id, factory)
            ref_f = None
            cache_item = None
            picard_existing_key = (case_id, "picard_lbm", run_mode, picard_tol)
            if picard_existing_key in existing and not args.no_cache and not args.no_resume:
                cache_item = _load_cached(case_id, "picard_lbm", cache_label_picard)
                if cache_item is not None:
                    ref_f, ref_hist, ref_wall = cache_item

            if ref_f is None:
                ref_f, ref_hist, ref_wall = cache_item if cache_item is not None else run_method_with_wall(
                    "picard_lbm", ref_case, tol=picard_tol, max_steps=ref_max_steps, verbose=False
                )
                ref_hist = _append_macro_l2_final_history_row(ref_case, ref_f, ref_hist, ref_wall)
                _save_cached(case_id, "picard_lbm", ref_f, _hist_to_list(ref_hist), ref_wall, cache_label_picard)
            else:
                ref_hist = _append_macro_l2_final_history_row(ref_case, ref_f, ref_hist, ref_wall)

            ref_final_residual = float(_macro_l2_residual_value(ref_case, ref_f)) if ref_f is not None else float("inf")
            ref_converged = int(np.isfinite(ref_final_residual) and ref_final_residual < 5.0 * float(picard_tol))
            if not args.suppress_reference_row:
                method_existing_key = (case_id, "picard_lbm", run_mode, method_tol)
                if abs(float(method_tol) - float(picard_tol)) <= 1.0e-30:
                    picard_case = ref_case
                    picard_f, picard_hist, picard_wall = ref_f, ref_hist, ref_wall
                else:
                    picard_case = factory()
                    method_cache = None if args.no_cache else _load_cached(case_id, "picard_lbm", cache_label_method)
                    if method_cache is None:
                        picard_f, picard_hist, picard_wall = run_method_with_wall(
                            "picard_lbm", picard_case, tol=method_tol, max_steps=max_steps, verbose=False
                        )
                        picard_hist = _append_macro_l2_final_history_row(picard_case, picard_f, picard_hist, picard_wall)
                        _save_cached(case_id, "picard_lbm", picard_f, _hist_to_list(picard_hist), picard_wall, cache_label_method)
                    else:
                        picard_f, picard_hist, picard_wall = method_cache
                        picard_hist = _append_macro_l2_final_history_row(picard_case, picard_f, picard_hist, picard_wall)
                picard_row = row_for(
                    base_id, case_id, label, method_tol, ref_case, ref_f,
                    "picard_lbm", picard_case, picard_f, picard_hist, picard_wall,
                    eval_case=eval_case, eval_f=eval_f,
                    cache_label_picard=cache_label_picard,
                    cache_label_method=cache_label_method,
                )
                picard_row = annotate_run_metadata(
                    picard_row,
                    run_mode=run_mode,
                    method_tol=method_tol,
                    reference_tol=picard_tol,
                    reference_converged=ref_converged,
                    reference_final_residual=ref_final_residual,
                    cache_label_picard=cache_label_picard,
                    cache_label_method=cache_label_method,
                )
                if method_existing_key in existing and not args.no_resume:
                    rows = [r for r in rows if not (r["case_id"] == case_id and r["method"] == "picard_lbm" and r.get("run_mode", "legacy") == run_mode and float(r.get("tol", float("nan"))) == method_tol)]
                rows.append(picard_row)
                write_history_csv(HIST_DIR / f"{case_id}__picard_lbm.csv", picard_hist)
                if not args.no_vtk:
                    try:
                        write_vtk(VTK_DIR / f"{case_id}__picard_lbm.vtk", picard_case, picard_f)
                    except Exception:
                        pass
                existing.add(method_existing_key)

            for method in methods_for_case:
                if method == "picard_lbm":
                    continue
                method_existing_key = (case_id, method, run_mode, method_tol)
                if method_existing_key in existing and method not in recompute_methods and not args.no_resume and not args.no_cache:
                    print(f"  [skip] {method} already exists", flush=True)
                    continue

                case = factory()
                cached = None if args.no_cache or method in recompute_methods else _load_cached(case_id, method, cache_label_method)
                if cached is None:
                    try:
                        f, hist, wall = run_method_with_wall(method, case, tol=method_tol, max_steps=max_steps, verbose=False)
                        hist = _append_macro_l2_final_history_row(case, f, hist, wall)
                        _save_cached(case_id, method, f, _hist_to_list(hist), wall, cache_label_method)
                    except Exception as exc:
                        print(f"  {method} crashed: {exc}", flush=True)
                        f = case.initial_field()
                        hist = [(0, float("inf"), 0, 0.0)]
                        wall = 0.0
                else:
                    f, hist, wall = cached
                    hist = _append_macro_l2_final_history_row(case, f, hist, wall)

                rows = [r for r in rows if not (r["case_id"] == case_id and r["method"] == method and r.get("run_mode", "legacy") == run_mode and float(r.get("tol", float("nan"))) == method_tol)]
                row = row_for(
                    base_id,
                    case_id,
                    label,
                    method_tol,
                    ref_case,
                    ref_f,
                    method,
                    case,
                    f,
                    hist,
                    wall,
                    eval_case=eval_case,
                    eval_f=eval_f,
                    cache_label_picard=cache_label_picard,
                    cache_label_method=cache_label_method,
                )
                row = annotate_run_metadata(
                    row,
                    run_mode=run_mode,
                    method_tol=method_tol,
                    reference_tol=picard_tol,
                    reference_converged=ref_converged,
                    reference_final_residual=ref_final_residual,
                    cache_label_picard=cache_label_picard,
                    cache_label_method=cache_label_method,
                )
                rows.append(row)
                write_history_csv(HIST_DIR / f"{case_id}__{method}.csv", hist)
                write_diagnostic_csv(DIAG_DIR / f"{case_id}__{method}__diagnostics.csv", hist)
                if not args.no_vtk:
                    try:
                        write_vtk(VTK_DIR / f"{case_id}__{method}.vtk", case, f)
                    except Exception:
                        pass

                print(
                    f"  {method:22s} lbe={row['lbe_calls']:8d} wall={row['wall_seconds']:8.3f} "
                    f"res={row['final_residual']:.3e} rel={row.get('rel_l2_vs_ref', row['rel_l2_vs_picard']):.3e} conv={row['converged']}",
                    flush=True,
                )
                existing.add(method_existing_key)

                write_outputs(rows, {
                    **summary_metadata,
                    "elapsed_wall_seconds": float(time.perf_counter() - start),
                    "case_count": len(set((r["case_id"] for r in rows))),
                    "method_count": len(methods),
                    "methods": methods,
                    "levels": levels,
                    "base_cases": base_ids,
                })

    metric_rows = [r for r in rows if r.get("run_mode", "legacy") == run_mode and r.get("base_case_id") in SCORING_CASE_IDS]
    strict_metrics = score_strict(metric_rows)
    paper_metrics = score_paper(metric_rows)
    for metric_obj in (strict_metrics, paper_metrics):
        metric_obj.update(
            {
                "elapsed_wall_seconds": float(time.perf_counter() - start),
                "case_count": len(set((r["case_id"] for r in metric_rows))),
                "method_count": len(methods),
                "methods": methods,
                "levels": levels,
                "base_cases": base_ids,
                **summary_metadata,
            }
        )
    metrics = {
        "strict": strict_metrics,
        "paper": paper_metrics,
    }
    metrics.update(
        {
            "elapsed_wall_seconds": float(time.perf_counter() - start),
            "case_count": len(set((r["case_id"] for r in metric_rows))),
            "method_count": len(methods),
            "methods": methods,
            "levels": levels,
            "base_cases": base_ids,
            **summary_metadata,
        }
    )
    write_outputs(rows, metrics)
    print(f"[saved] {OUT / 'summary.csv'}")
    print(f"[saved] {OUT / 'metrics_strict.json'}")
    print(f"[saved] {OUT / 'metrics_paper.json'}")
    print(f"[saved] {OUT / 'metrics.json'}")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
