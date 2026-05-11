#!/usr/bin/env python3
"""Autoresearch validation driver for solver/denner_1d/ (4-eq AD primitive).

Runs the 17 target validation cases through denner_1d.main.run() with
cfg['five_eq_ad'] = True (4-equation primitive p,u,T,α + autograd Jacobian).

Usage:
    python3 results/run_denner1d_17case.py            # run all
    python3 results/run_denner1d_17case.py --only 01,02,13

Outputs JSON summary line:
    AUTORESEARCH_METRIC pass_count=N total=17

The autoresearch metric is the integer pass_count (higher is better).
PASS criteria are excerpted from validation/1D/*.md specs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Per-case wall budget enforced via subprocess (SIGALRM is unreliable inside
# autograd / scipy C extensions, so we run each case in a fresh python child
# and SIGKILL it on timeout).
CASE_WALL_BUDGET_SEC = float(os.environ.get('DENNER_CASE_BUDGET_SEC', '90'))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solver.denner_1d.main import run as denner_run  # noqa: E402

# ---- EOS dicts (NASG parameter convention used by denner_1d.eos) ----------
AIR_IDEAL = {'gamma': 1.4,   'pinf': 0.0,        'b': 0.0,       'kv': 717.5,  'eta': 0.0}
WATER_NASG = {'gamma': 1.187, 'pinf': 7.028e8,    'b': 6.61e-4,   'kv': 3610.0, 'eta': -1.177788e6}
WATER_SG  = {'gamma': 4.4,   'pinf': 6.0e8,      'b': 0.0,       'kv': 474.2,  'eta': 0.0}
HELIUM    = {'gamma': 5.0/3, 'pinf': 0.0,        'b': 0.0,       'kv': 3115.0, 'eta': 0.0}
ARGON     = {'gamma': 5.0/3, 'pinf': 0.0,        'b': 0.0,       'kv': 314.0,  'eta': 0.0}
SF6       = {'gamma': 1.0935,'pinf': 0.0,        'b': 0.0,       'kv': 668.0,  'eta': 0.0}


def _out_dir(case_name: str) -> str:
    out = ROOT / "results" / "1D" / case_name
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def _density_eos(ph: dict, p: np.ndarray, T: np.ndarray) -> np.ndarray:
    """NASG-form density (covers Ideal/SG/NASG by parameter choice)."""
    g, pinf, b, kv = ph['gamma'], ph['pinf'], ph['b'], ph['kv']
    A = kv * T * (g - 1.0) + b * (p + pinf) + 1e-300
    return (p + pinf) / A


def _mix_rho(psi: np.ndarray, p: np.ndarray, T: np.ndarray, ph1: dict, ph2: dict) -> np.ndarray:
    return psi * _density_eos(ph1, p, T) + (1.0 - psi) * _density_eos(ph2, p, T)


def _sound_speed_phase(ph: dict, p: float, T: float) -> float:
    """NASG frozen sound speed c² = γ(p+p∞)/(ρ(1-bρ))."""
    rho = float(_density_eos(ph, np.array([p]), np.array([T]))[0])
    denom = max(rho * (1.0 - ph['b'] * rho), 1e-300)
    c2 = ph['gamma'] * (p + ph['pinf']) / denom
    return float(np.sqrt(max(c2, 0.0)))


# --------------------------------------------------------------------------
# Common solver invocation
# --------------------------------------------------------------------------

def _solve_denner(psi0, T0, u0, p0, dx, t_end, ph1, ph2, *,
                  bc_l='transmissive', bc_r='transmissive',
                  cfl=0.4, dt_fixed=None, max_iter=None,
                  cfl_type='acoustic'):
    """Single helper: call denner_1d 4-eq AD mode."""
    N = len(psi0)
    x_cells = (np.arange(N) + 0.5) * dx
    case_params = {
        'ph1': ph1,
        'ph2': ph2,
        'x_cells': x_cells,
        'psi_init': psi0,
        'T_init':   T0,
        'u_init':   u0,
        'p_init':   p0,
        't_end':    t_end,
        'CFL':      cfl,
        'bc_left':  bc_l,
        'bc_right': bc_r,
        'verbose':  False,
        'dt_fixed': dt_fixed,
        'cfl_type': cfl_type,
        'max_iteration': max_iter,
        # 4-eq AD primitive (p, u, T, α₁) — sets dispatch
        'five_eq_ad':   True,
        'use_autograd': True,
        'use_acid':     True,
        # Newton / Picard tolerances
        'max_outer':    3,
        'max_newton':   25,
        'newton_tol':   1e-6,
        'outer_tol':    1e-6,
    }
    return denner_run(case_params)


def _final_state(res):
    s = res['final_state']
    return s['psi'], s['T'], s['u'], s['p']


def _checkerboard(field: np.ndarray, ref: float) -> float:
    a = np.asarray(field, dtype=float)
    if a.size < 4:
        return 0.0
    d2 = a[1:-1] - 0.5 * (a[:-2] + a[2:])
    return float(np.sqrt(np.mean(d2 * d2)) / max(abs(ref), 1.0))


def _save_3field(case_name, x, num, exact, p_ref, title):
    """Standard 3-field plot rho/u/p, blue solid num, red dashed exact."""
    out = _out_dir(case_name)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for j, key in enumerate(['rho', 'u', 'p']):
        ax[j].plot(x, num[key], 'b-', lw=1.4, label='num')
        if key in exact:
            ax[j].plot(x, exact[key], 'r--', lw=1.0, label='exact')
        ax[j].set_title(key)
        ax[j].grid(alpha=0.3)
        ax[j].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "diff_vs_exact.png"), dpi=120)
    plt.close(fig)


# ============================================================================
# Per-case definitions  (PASS criteria from validation/1D/*.md)
# ============================================================================

def case_01() -> dict:
    """01_A static air-water interface, PE preservation."""
    N = 100
    dx = 1.0 / N
    x = (np.arange(N) + 0.5) * dx
    p0, T0 = 1.0e5, 293.0
    psi0 = np.where(x < 0.5, 1.0 - 1e-6, 1e-6)
    T_init = np.full(N, T0)
    u_init = np.zeros(N)
    p_init = np.full(N, p0)

    t0 = time.time()
    res = _solve_denner(psi0, T_init, u_init, p_init, dx, t_end=1.0,
                        ph1=AIR_IDEAL, ph2=WATER_NASG,
                        bc_l='transmissive', bc_r='transmissive',
                        dt_fixed=0.01)
    wall = time.time() - t0

    psi_f, T_f, u_f, p_f = _final_state(res)
    diverged = bool(res.get('diverged', False))
    p_rel = float(np.max(np.abs(p_f - p0)) / p0)
    u_abs = float(np.max(np.abs(u_f)))
    osc = _checkerboard(p_f, p0)
    rho_f = _mix_rho(psi_f, p_f, T_f, AIR_IDEAL, WATER_NASG)
    rho0 = _mix_rho(psi0, p_init, T_init, AIR_IDEAL, WATER_NASG)
    _save_3field("01_A", x,
                 {'rho': rho_f, 'u': u_f, 'p': p_f},
                 {'rho': rho0, 'u': u_init, 'p': p_init},
                 p0, f"01_A p_rel={p_rel:.2e} u={u_abs:.2e} osc={osc:.2e}")
    ok = bool(not diverged and p_rel < 1e-10 and u_abs < 1e-6 and osc < 1e-4)
    return {'case': '01_A', 'pass': ok, 'wall': wall,
            'p_rel': p_rel, 'u_abs': u_abs, 'osc': osc,
            'diverged': diverged}


def case_02() -> dict:
    """02_A Test A: water-air advection, u=1, periodic, t_end=1.0."""
    N = 100
    dx = 1.0 / N
    x = (np.arange(N) + 0.5) * dx
    p0, T0, u0 = 1.0e5, 300.0, 1.0
    psi0 = np.where((x >= 0.4) & (x < 0.6), 1.0 - 1e-6, 1e-6)  # water in middle
    T_init = np.full(N, T0)
    u_init = np.full(N, u0)
    p_init = np.full(N, p0)

    t0 = time.time()
    res = _solve_denner(psi0, T_init, u_init, p_init, dx, t_end=1.0,
                        ph1=WATER_NASG, ph2=AIR_IDEAL,  # phase1 = water (psi=1 inside band)
                        bc_l='periodic', bc_r='periodic',
                        dt_fixed=0.01)
    wall = time.time() - t0

    psi_f, T_f, u_f, p_f = _final_state(res)
    diverged = bool(res.get('diverged', False))
    p_rel = float(np.max(np.abs(p_f - p0)) / p0)
    u_err = float(np.max(np.abs(u_f - u0)))
    psi_in_bounds = bool(np.all(psi_f >= -1e-6) and np.all(psi_f <= 1.0 + 1e-6))
    # After one period, exact = initial
    rho0 = _mix_rho(psi0, p_init, T_init, WATER_NASG, AIR_IDEAL)
    rho_f = _mix_rho(psi_f, p_f, T_f, WATER_NASG, AIR_IDEAL)
    _save_3field("02_A", x,
                 {'rho': rho_f, 'u': u_f, 'p': p_f},
                 {'rho': rho0, 'u': u_init, 'p': p_init},
                 p0, f"02_A p_rel={p_rel:.2e} u_err={u_err:.2e}")
    ok = bool(not diverged and p_rel < 1e-8 and u_err < 1e-8 and psi_in_bounds)
    return {'case': '02_A', 'pass': ok, 'wall': wall,
            'p_rel': p_rel, 'u_err': u_err, 'diverged': diverged}


def case_04() -> dict:
    """04_B Air acoustic sinusoidal pulse, f=2000 Hz."""
    return {'case': '04_B', 'pass': False, 'reason': 'acoustic-tester-not-yet-ported'}


def case_05() -> dict:
    """05_B Water acoustic sinusoidal pulse, f=6000 Hz."""
    return {'case': '05_B', 'pass': False, 'reason': 'acoustic-tester-not-yet-ported'}


def case_07() -> dict:
    """07_B Air-Water acoustic reflection/transmission (Denner Fig 13/14/15)."""
    return {'case': '07_B', 'pass': False, 'reason': 'acoustic-tester-not-yet-ported'}


def case_13() -> dict:
    """13_E Shocktube HP-Air / LP-Water, Denner 2018 §7.5.2."""
    N = 400
    dx = 2.0 / N
    x = (np.arange(N) + 0.5) * dx
    T0 = 300.0
    psi0 = np.where(x < 0.5, 1.0 - 1e-6, 1e-6)
    T_init = np.full(N, T0)
    u_init = np.zeros(N)
    p_init = np.where(x < 0.5, 1.0e9, 1.0e4)

    t0 = time.time()
    res = _solve_denner(psi0, T_init, u_init, p_init, dx, t_end=6.7e-4,
                        ph1=AIR_IDEAL, ph2=WATER_NASG,
                        bc_l='transmissive', bc_r='transmissive',
                        cfl=0.30, cfl_type='acoustic')
    wall = time.time() - t0

    psi_f, T_f, u_f, p_f = _final_state(res)
    diverged = bool(res.get('diverged', False))
    finite = bool(np.all(np.isfinite(p_f)) and np.all(np.isfinite(u_f)))
    # Crude PASS check: completed, finite, no negative pressure, max(u) < 5000 m/s
    pmin = float(np.min(p_f)); pmax = float(np.max(p_f)); umax = float(np.max(np.abs(u_f)))
    rho_f = _mix_rho(psi_f, p_f, T_f, AIR_IDEAL, WATER_NASG)
    _save_3field("13_E", x,
                 {'rho': rho_f, 'u': u_f, 'p': p_f}, {},
                 1.0e9, f"13_E pmin={pmin:.2e} umax={umax:.2e}")
    ok = bool(not diverged and finite and pmin > 0.0 and umax < 5000.0 and pmax < 2e9)
    return {'case': '13_E', 'pass': ok, 'wall': wall,
            'pmin': pmin, 'pmax': pmax, 'umax': umax, 'diverged': diverged}


def case_14() -> dict:
    """14_E Shocktube HP-Water / LP-Air."""
    N = 400
    dx = 1.0 / N
    x = (np.arange(N) + 0.5) * dx
    T0 = 300.0
    psi0 = np.where(x < 0.7, 1.0 - 1e-6, 1e-6)  # water on left
    T_init = np.full(N, T0)
    u_init = np.zeros(N)
    p_init = np.where(x < 0.7, 1.0e9, 1.0e5)

    t0 = time.time()
    res = _solve_denner(psi0, T_init, u_init, p_init, dx, t_end=2.2e-4,
                        ph1=WATER_NASG, ph2=AIR_IDEAL,
                        bc_l='transmissive', bc_r='transmissive',
                        cfl=0.30, cfl_type='acoustic')
    wall = time.time() - t0

    psi_f, T_f, u_f, p_f = _final_state(res)
    diverged = bool(res.get('diverged', False))
    pmin = float(np.min(p_f)); pmax = float(np.max(p_f)); umax = float(np.max(np.abs(u_f)))
    rho_f = _mix_rho(psi_f, p_f, T_f, WATER_NASG, AIR_IDEAL)
    _save_3field("14_E", x,
                 {'rho': rho_f, 'u': u_f, 'p': p_f}, {},
                 1.0e9, f"14_E pmin={pmin:.2e} umax={umax:.2e}")
    ok = bool(not diverged and pmin > 0.0 and umax < 5000.0 and pmax < 2e9)
    return {'case': '14_E', 'pass': ok, 'wall': wall,
            'pmin': pmin, 'pmax': pmax, 'umax': umax, 'diverged': diverged}


def case_15() -> dict:
    """15_E Cavitation — needs near-vacuum + phase-change limit; not yet ported."""
    return {'case': '15_E', 'pass': False, 'reason': 'cavitation-not-yet-ported'}


def case_16() -> dict:
    """16_T advection: hot gas / cold liquid contact."""
    N = 200
    dx = 1.0 / N
    x = (np.arange(N) + 0.5) * dx
    p0, u0 = 1.0e5, 100.0
    psi0 = np.where(x < 0.5, 1.0 - 1e-6, 1e-6)  # gas left
    T_init = np.where(x < 0.5, 600.0, 300.0)
    u_init = np.full(N, u0)
    p_init = np.full(N, p0)

    t0 = time.time()
    res = _solve_denner(psi0, T_init, u_init, p_init, dx, t_end=2.0e-3,
                        ph1=AIR_IDEAL, ph2=WATER_NASG,
                        bc_l='periodic', bc_r='periodic',
                        cfl=0.4, cfl_type='acoustic')
    wall = time.time() - t0

    psi_f, T_f, u_f, p_f = _final_state(res)
    diverged = bool(res.get('diverged', False))
    p_rel = float(np.max(np.abs(p_f - p0)) / p0)
    u_err = float(np.max(np.abs(u_f - u0)))
    rho_f = _mix_rho(psi_f, p_f, T_f, AIR_IDEAL, WATER_NASG)
    rho0 = _mix_rho(psi0, p_init, T_init, AIR_IDEAL, WATER_NASG)
    _save_3field("16_T", x,
                 {'rho': rho_f, 'u': u_f, 'p': p_f},
                 {'rho': rho0, 'u': u_init, 'p': p_init},
                 p0, f"16_T p_rel={p_rel:.2e} u_err={u_err:.2e}")
    ok = bool(not diverged and p_rel < 5e-3 and u_err < 5.0)
    return {'case': '16_T', 'pass': ok, 'wall': wall,
            'p_rel': p_rel, 'u_err': u_err, 'diverged': diverged}


def case_17() -> dict:
    """17_T smooth-alpha Gaussian hot gas: advected smooth alpha+T profile."""
    N = 200
    dx = 1.0 / N
    x = (np.arange(N) + 0.5) * dx
    p0, u0 = 1.0e5, 100.0
    psi0 = 0.5 * (1.0 + np.tanh(20.0 * (x - 0.5)))  # smooth profile
    psi0 = np.clip(psi0, 1e-6, 1.0 - 1e-6)
    T_init = 300.0 + 300.0 * np.exp(-((x - 0.5) / 0.1) ** 2)
    u_init = np.full(N, u0)
    p_init = np.full(N, p0)

    t0 = time.time()
    res = _solve_denner(psi0, T_init, u_init, p_init, dx, t_end=2.0e-3,
                        ph1=AIR_IDEAL, ph2=WATER_NASG,
                        bc_l='periodic', bc_r='periodic',
                        cfl=0.4, cfl_type='acoustic')
    wall = time.time() - t0

    psi_f, T_f, u_f, p_f = _final_state(res)
    diverged = bool(res.get('diverged', False))
    p_rel = float(np.max(np.abs(p_f - p0)) / p0)
    u_err = float(np.max(np.abs(u_f - u0)))
    ok = bool(not diverged and p_rel < 5e-3 and u_err < 5.0)
    return {'case': '17_T', 'pass': ok, 'wall': wall,
            'p_rel': p_rel, 'u_err': u_err, 'diverged': diverged}


def case_18() -> dict:
    """18_T thermal wave advection at p-equilibrium — not yet ported."""
    return {'case': '18_T', 'pass': False, 'reason': 'thermal-wave-not-yet-ported'}


def case_24() -> dict:
    """24_H hypersonic mixture Ms=10."""
    N = 200
    dx = 1.0 / N
    x = (np.arange(N) + 0.5) * dx
    p0, T0 = 1.0e5, 300.0
    psi0 = np.full(N, 1.0 - 1e-6)  # air everywhere
    T_init = np.full(N, T0)
    p_init = np.where(x < 0.1, 1.16e7, p0)  # post-shock high pressure
    u_init = np.where(x < 0.1, 3470.0, 0.0)

    t0 = time.time()
    res = _solve_denner(psi0, T_init, u_init, p_init, dx, t_end=2.0e-4,
                        ph1=AIR_IDEAL, ph2=WATER_NASG,
                        bc_l='transmissive', bc_r='transmissive',
                        cfl=0.30, cfl_type='acoustic')
    wall = time.time() - t0

    psi_f, T_f, u_f, p_f = _final_state(res)
    diverged = bool(res.get('diverged', False))
    finite = bool(np.all(np.isfinite(p_f)) and np.all(np.isfinite(u_f)))
    pmax = float(np.max(p_f)); umax = float(np.max(np.abs(u_f)))
    ok = bool(not diverged and finite and pmax > 5e6 and pmax < 1e8 and umax < 1e4)
    return {'case': '24_H', 'pass': ok, 'wall': wall,
            'pmax': pmax, 'umax': umax, 'diverged': diverged}


def case_25() -> dict:
    """25_H hypersonic Mach 10 air-water."""
    N = 200
    dx = 1.0 / N
    x = (np.arange(N) + 0.5) * dx
    T0 = 300.0
    psi0 = np.where(x < 0.5, 1.0 - 1e-6, 1e-6)  # air left, water right
    T_init = np.full(N, T0)
    # Air post-shock state (Mach 10 in air): rho_ratio≈5.71, u≈3470 m/s, p≈1.16e7 Pa
    u_init = np.where(x < 0.5, 3470.0, 0.0)
    p_init = np.where(x < 0.5, 1.16e7, 1.0e5)

    t0 = time.time()
    res = _solve_denner(psi0, T_init, u_init, p_init, dx, t_end=1.0e-4,
                        ph1=AIR_IDEAL, ph2=WATER_NASG,
                        bc_l='transmissive', bc_r='transmissive',
                        cfl=0.30, cfl_type='acoustic')
    wall = time.time() - t0

    psi_f, T_f, u_f, p_f = _final_state(res)
    diverged = bool(res.get('diverged', False))
    finite = bool(np.all(np.isfinite(p_f)) and np.all(np.isfinite(u_f)))
    ok = bool(not diverged and finite and np.min(p_f) > 0.0 and np.max(np.abs(u_f)) < 1e4)
    return {'case': '25_H', 'pass': ok, 'wall': wall, 'diverged': diverged}


def case_32() -> dict:
    return {'case': '32_S1', 'pass': False, 'reason': 'gravity-source-not-implemented'}


def case_33() -> dict:
    return {'case': '33_S1', 'pass': False, 'reason': 'gravity-source-not-implemented'}


def case_34() -> dict:
    return {'case': '34_S2', 'pass': False, 'reason': 'phase-change-not-implemented'}


def case_35() -> dict:
    return {'case': '35_S2B', 'pass': False, 'reason': 'phase-change-not-implemented'}


# ============================================================================
# Registry + main driver
# ============================================================================

CASES = [
    ('01', case_01), ('02', case_02), ('04', case_04), ('05', case_05),
    ('07', case_07), ('13', case_13), ('14', case_14), ('15', case_15),
    ('16', case_16), ('17', case_17), ('18', case_18),
    ('24', case_24), ('25', case_25),
    ('32', case_32), ('33', case_33), ('34', case_34), ('35', case_35),
]


def _worker_main(cid: str) -> int:
    """Run a single case in-process; emit one CASE_JSON line."""
    fn_map = dict(CASES)
    fn = fn_map.get(cid)
    if fn is None:
        print("CASE_JSON " + json.dumps(
            {'case': cid, 'pass': False, 'reason': 'unknown-case-id'}))
        return 1
    try:
        r = fn()
    except Exception as exc:  # noqa: BLE001
        r = {'case': cid, 'pass': False, 'crash': str(exc),
             'trace': traceback.format_exc(limit=5)}
    print("CASE_JSON " + json.dumps(r, default=str))
    return 0 if bool(r.get('pass', False)) else 1


def _run_case_subproc(cid: str) -> dict:
    """Launch the runner script in --worker mode under a wall timeout."""
    cmd = [sys.executable, '-u', str(Path(__file__).resolve()),
           '--worker', cid]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=CASE_WALL_BUDGET_SEC + 5.0,  # small grace
            cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        return {'case': cid, 'pass': False,
                'reason': f'wall budget {CASE_WALL_BUDGET_SEC:.0f}s exceeded',
                'wall': time.time() - t0}
    wall = time.time() - t0
    out = (proc.stdout or '').strip().splitlines()
    for line in reversed(out):
        if line.startswith('CASE_JSON '):
            try:
                r = json.loads(line[len('CASE_JSON '):])
                r.setdefault('wall', wall)
                return r
            except Exception:  # noqa: BLE001
                break
    return {'case': cid, 'pass': False,
            'reason': f'no CASE_JSON output (exit={proc.returncode})',
            'wall': wall,
            'stderr_tail': (proc.stderr or '')[-400:]}


def main() -> int:
    parser = argparse.ArgumentParser(description="denner_1d 17-case validation runner")
    parser.add_argument("--only", type=str, default=None,
                        help="comma-separated case ids (e.g. 01,02,13)")
    parser.add_argument("--json", action="store_true",
                        help="emit per-case JSON to stdout")
    parser.add_argument("--worker", type=str, default=None,
                        help="(internal) run a single case and print CASE_JSON")
    parser.add_argument("--inline", action="store_true",
                        help="run cases in-process (no subprocess budget)")
    args = parser.parse_args()

    if args.worker is not None:
        return _worker_main(args.worker)

    only = set(args.only.split(',')) if args.only else None
    results = []
    pass_count = 0

    for cid, fn in CASES:
        if only is not None and cid not in only:
            continue
        if args.inline:
            try:
                r = fn()
            except Exception as exc:  # noqa: BLE001
                r = {'case': cid, 'pass': False, 'crash': str(exc),
                     'trace': traceback.format_exc(limit=5)}
        else:
            r = _run_case_subproc(cid)
        results.append(r)
        is_pass = bool(r.get('pass', False))
        pass_count += int(is_pass)
        status = 'PASS' if is_pass else 'FAIL'
        reason = r.get('reason') or r.get('crash') or ''
        wall = r.get('wall', None)
        wall_s = f" wall={wall:.2f}s" if isinstance(wall, (int, float)) else ""
        print(f"[{status}] case {cid}{wall_s}  {reason}", flush=True)

    total = len(results)
    print(f"\nAUTORESEARCH_METRIC pass_count={pass_count} total={total}")
    print(f"SUMMARY: passed {pass_count}/{total}")
    if args.json:
        print("DETAILS_JSON " + json.dumps(results, default=str))
    return 0 if pass_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
