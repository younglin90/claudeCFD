# solver/denner_1d/main.py
# Ref: DENNER_SCHEME.md — Mode A, Phase 1a
#
# Entry point for the denner_1d implicit CFD solver.

import numpy as np

from .eos.base import compute_mixture_props
from .boundary import apply_ghost, apply_ghost_velocity
from .timestepping import compute_dt
from .eos.base import compute_specific_total_enthalpy
from .solver_a import step as mode_a_step
from .io_utils import make_snapshot, save_snapshots_npz, print_step_info


def run(case_params):
    """
    Run the denner_1d Mode A solver for a given case.

    Parameters
    ----------
    case_params : dict with keys:

        ph1, ph2         : dict   NASG EOS parameters for phase 1 and 2
        x_cells          : ndarray (N,)   cell-centre positions [m]
        psi_init         : ndarray (N,)   initial VOF field (alpha_1)
        T_init           : ndarray (N,)   initial temperature [K]
        u_init           : ndarray (N,)   initial velocity [m/s]
        p_init           : ndarray (N,)   initial pressure [Pa]
        t_end            : float          end time [s]
        CFL              : float          CFL number (default 0.5)
        bc_left          : str            left BC ('transmissive'|'periodic'|'wall')
        bc_right         : str            right BC
        output_times     : list or None   simulation times at which to save snapshots
        verbose          : bool           print per-step info (default True)
        dt_fixed         : float or None  override CFL-based dt
        max_picard_iter  : int            (default 20)
        picard_tol       : float          (default 1e-6)
        output_path      : str or None    path for .npz output (None = no file)

    Returns
    -------
    result : dict
        final_state  : dict   final solver state
        snapshots    : list   list of snapshot dicts (at output_times + final)
        t_history    : list   times at each step
        dt_history   : list   time steps used
    """
    # -----------------------------------------------------------------
    # Unpack parameters
    # -----------------------------------------------------------------
    ph1 = case_params['ph1']
    ph2 = case_params['ph2']

    x_cells  = np.asarray(case_params['x_cells'], dtype=float)
    psi_init = np.asarray(case_params['psi_init'], dtype=float)
    T_init   = np.asarray(case_params['T_init'],   dtype=float)
    u_init   = np.asarray(case_params['u_init'],   dtype=float)
    p_init   = np.asarray(case_params['p_init'],   dtype=float)

    N  = len(x_cells)
    dx = float(x_cells[1] - x_cells[0]) if N > 1 else 1.0

    t_end    = float(case_params['t_end'])
    CFL      = float(case_params.get('CFL', 0.5))
    bc_l     = case_params.get('bc_left',  'transmissive')
    bc_r     = case_params.get('bc_right', 'transmissive')
    verbose  = bool(case_params.get('verbose', True))
    dt_fixed = case_params.get('dt_fixed', None)
    if dt_fixed is not None:
        dt_fixed = float(dt_fixed)

    output_times = case_params.get('output_times', [])
    if output_times is None:
        output_times = []
    output_times = sorted(set(float(t) for t in output_times))
    output_path  = case_params.get('output_path', None)

    cfg = {
        'max_outer':  case_params.get('max_outer',  case_params.get('max_picard_iter', 5)),
        'max_inner':  case_params.get('max_inner',  10),
        'inner_tol':  case_params.get('inner_tol',  case_params.get('picard_tol', 1e-6)),
        'outer_tol':  case_params.get('outer_tol',  1e-6),
        # legacy aliases kept for backward compat
        'max_picard_iter': case_params.get('max_picard_iter', 5),
        'picard_tol':      case_params.get('picard_tol',      1e-6),
    }

    max_iteration = case_params.get('max_iteration', None)

    # -----------------------------------------------------------------
    # Initialise state
    # -----------------------------------------------------------------
    psi = np.clip(psi_init, 0.0, 1.0)
    psi_reg0 = np.clip(psi, 0.01, 0.99)
    props0 = compute_mixture_props(p_init, u_init, T_init, psi_reg0, ph1, ph2)

    # Initial face velocity (arithmetic mean), n_ghost=2
    u_ext0 = apply_ghost_velocity(u_init, bc_l, bc_r, n_ghost=2)
    ng0 = 2
    u_face0 = np.array([0.5 * (u_ext0[ng0 + f - 1] + u_ext0[ng0 + f])
                         for f in range(N + 1)])

    h_init = compute_specific_total_enthalpy(p_init, u_init, T_init, psi_reg0, ph1, ph2)

    state = {
        'p':       p_init.copy(),
        'u':       u_init.copy(),
        'T':       T_init.copy(),
        'h':       h_init.copy(),
        'psi':     psi.copy(),
        'rho':     props0['rho'].copy(),
        'E_total': props0['E_total'].copy(),
        'u_face':  u_face0.copy(),
    }

    aux = {
        'is_first_step': True,
        'bdf_order':     1,
        'rho_nm1':       None,
        'rhoU_nm1':      None,
        'E_nm1':         None,
        'rho_face_acid': None,
    }

    # -----------------------------------------------------------------
    # Main time loop
    # -----------------------------------------------------------------
    t           = 0.0
    step_num    = 0
    snapshots   = []
    t_history   = [t]
    dt_history  = []

    # Save initial snapshot
    snapshots.append(make_snapshot(t, state, x_cells))
    next_output_idx = 0  # index into output_times list

    if verbose:
        print(f"[denner_1d] Starting simulation: N={N}, t_end={t_end}, "
              f"CFL={CFL}, bc=({bc_l},{bc_r})")

    # -----------------------------------------------------------------
    # Divergence thresholds
    # -----------------------------------------------------------------
    p0_ref    = float(np.mean(np.abs(case_params.get('p_init', [1e5]))))
    T0_ref    = float(np.mean(np.abs(case_params.get('T_init', [300.0]))))
    u0_ref    = float(np.mean(np.abs(case_params.get('u_init', [1.0])))) + 1e-6
    dt_init   = None          # set after first step
    diverged  = False
    diverge_reason = ''

    # Sound speed reference: c_ref = sqrt(gamma_eff * p0 / rho0)
    _gamma_eff = max(ph1.get('gamma', 1.4), ph2.get('gamma', 1.4))
    _rho0_ref  = float(np.mean(state['rho']))
    c0_ref     = float(np.sqrt(_gamma_eff * max(p0_ref, 1.0) / max(_rho0_ref, 1e-10)))

    while t < t_end - 1e-14 * t_end and (max_iteration is None or step_num < max_iteration):
        # Compute dt
        if dt_fixed is not None:
            dt = dt_fixed
        else:
            dt = compute_dt(state['u'], dx, CFL)

        # Don't overshoot t_end
        dt = min(dt, t_end - t)

        # Clamp to hit output_times, avoiding tiny steps from floating-point drift.
        # If remaining < 0.01*dt, skip the clamp (snapshot will be saved at the
        # next regular step by the t >= output_time - 1e-10 condition).
        if next_output_idx < len(output_times):
            remaining = output_times[next_output_idx] - t
            if remaining > 0.01 * dt:
                dt = min(dt, remaining)
        dt = max(dt, 1e-20)  # safety floor

        # Time step
        new_state, new_aux, info = mode_a_step(
            state, ph1, ph2, dx, dt, bc_l, bc_r, aux, cfg)

        state = new_state
        aux   = new_aux
        t    += dt
        step_num += 1

        if dt_init is None:
            dt_init = dt   # record first step size

        t_history.append(t)
        dt_history.append(dt)

        if verbose and (step_num % 1 == 0):
            print_step_info(step_num, t, dt, info, state)

        # ------------------------------------------------------------------
        # Divergence detection — stop immediately on any trigger
        # ------------------------------------------------------------------
        p_now   = state['p']
        T_now   = state['T']
        u_now   = state['u']
        rho_now = state['rho']

        # 1. NaN / Inf in any field
        if not (np.all(np.isfinite(p_now)) and np.all(np.isfinite(u_now))
                and np.all(np.isfinite(T_now))):
            diverged = True
            diverge_reason = 'NaN or Inf in state'

        # 2. Pressure too large or negative
        elif np.max(p_now) > 1e3 * max(p0_ref, 1.0) or np.min(p_now) < 0.0:
            diverged = True
            diverge_reason = (f'Pressure out of range: min={np.min(p_now):.3e}'
                              f'  max={np.max(p_now):.3e}  (ref={p0_ref:.3e})')

        # 3. Temperature non-physical (near-zero or > 1000× initial)
        elif np.min(T_now) <= 1e-3 or np.max(T_now) > 1e3 * max(T0_ref, 1.0):
            diverged = True
            diverge_reason = (f'Temperature non-physical: min={np.min(T_now):.3e}'
                              f'  max={np.max(T_now):.3e}  (ref={T0_ref:.3e})')

        # 4. Velocity explosion: |u| > 1e4 × initial reference velocity
        elif np.max(np.abs(u_now)) > 1e4 * u0_ref:
            diverged = True
            diverge_reason = (f'Velocity explosion: max|u|={np.max(np.abs(u_now)):.3e}'
                              f'  (ref={u0_ref:.3e})')

        # 5. Sound speed explosion: c_mix = sqrt(gamma_eff * p / rho) > 1e4 × c0_ref
        else:
            rho_safe = np.maximum(rho_now, 1e-10)
            c_now    = np.sqrt(_gamma_eff * np.abs(p_now) / rho_safe)
            if np.max(c_now) > 1e4 * c0_ref:
                diverged = True
                diverge_reason = (f'Sound speed explosion: max(c)={np.max(c_now):.3e}'
                                  f'  (c0_ref={c0_ref:.3e})')

            # 6. dt collapsed (CFL runaway — acoustic or otherwise)
            elif (dt_init is not None and dt_fixed is None
                  and dt < 1e-6 * dt_init):
                diverged = True
                diverge_reason = (f'dt collapsed: dt={dt:.3e}  dt_init={dt_init:.3e}  '
                                  f'(ratio={dt/dt_init:.2e})')

        if diverged:
            if verbose:
                print(f'[denner_1d] *** DIVERGENCE at t={t:.4e}s step={step_num}: '
                      f'{diverge_reason} ***')
            break

        # Save snapshots at requested times: save when t first reaches or
        # passes the output time (handles floating-point drift without tiny steps).
        while next_output_idx < len(output_times):
            if t >= output_times[next_output_idx] - 1e-10:
                snapshots.append(make_snapshot(t, state, x_cells))
                next_output_idx += 1
            else:
                break

    # Save final state
    final_snap = make_snapshot(t, state, x_cells)
    if len(snapshots) == 0 or abs(snapshots[-1]['t'] - t) > 1e-12:
        snapshots.append(final_snap)

    # Save to file
    if output_path is not None:
        save_snapshots_npz(output_path, snapshots)
        if verbose:
            print(f"[denner_1d] Saved {len(snapshots)} snapshots to {output_path}")

    if verbose:
        status = 'DIVERGED' if diverged else 'Done'
        print(f"[denner_1d] {status}. t={t:.6e}s, steps={step_num}")

    return {
        'final_state':    state,
        'snapshots':      snapshots,
        't_history':      t_history,
        'dt_history':     dt_history,
        'diverged':       diverged,
        'diverge_reason': diverge_reason,
    }
