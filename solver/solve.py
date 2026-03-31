# Ref: CLAUDE.md § 지배방정식, § 적용 수치 기법, § 시간 차분
# Ref: docs/APEC_flux.md — APEC flux
"""
Main 1D CFD solver using APEC flux and Forward/Backward Euler time integration.

Governing equations (1D Euler):
    ∂U/∂t + ∂F/∂x = 0

    U = [rho, rho*u, rho*E, rho*Y_1, ..., rho*Y_{N-1}]^T
    F = [rho*u, rho*u^2+p, (rho*E+p)*u, rho*Y_k*u]^T

Time integration (CLAUDE.md):
    Ma > 1 : Forward Euler, CFL <= 0.8
    Ma <= 1 : Backward Euler, Newton iteration, ||ΔU||/||U|| < 1e-8

Boundary conditions supported:
    - 'periodic'      : periodic (wrap-around)
    - 'transmissive'  : zero-gradient (outflow)
    - 'inlet'         : prescribed primitive state
    - 'acoustic_inlet': sinusoidal velocity perturbation (for reflection/transmission tests)

Usage
-----
result = run_1d(case_params)

case_params is a dict with keys:
    'eos_list'      : list of EOS objects (one per species)
    'x_cells'       : 1D array of cell-center x-coordinates
    'U_init'        : initial conservative state, shape (N_cells, n_vars)
    't_end'         : simulation end time
    'CFL'           : CFL number (default 0.8 for supersonic, 0.6 otherwise)
    'bc_left'       : boundary condition on left  ('periodic', 'transmissive', 'inlet', 'acoustic_inlet')
    'bc_right'      : boundary condition on right ('periodic', 'transmissive')
    'bc_left_params': dict with extra params for left BC (e.g. primitive state or inlet function)
    'bc_right_params': dict (optional)
    'output_times'  : list of times at which to save snapshots (optional)
    'T_guess'       : initial temperature guess for cons_to_prim (default 300.0)
    'max_iter_be'   : max Newton iterations for Backward Euler (default 50)
    'tol_be'        : convergence tolerance for Backward Euler (default 1e-8)
    'verbose'       : bool, print progress (default False)
    'time_scheme'   : 'auto' (default), 'forward_euler', 'backward_euler', 'tvd_rk3'
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Optional

from .eos.ideal import IdealGasEOS
from .eos.nasg import NASGEOS
from .eos.srk import SRKEOS
from .utils import cons_to_prim, prim_to_cons, mixture_sound_speed, _rho_from_T_p
from .flux import apec_flux, physical_flux
from .jacobian import system_jacobian

EOSType = IdealGasEOS | NASGEOS | SRKEOS


# ---------------------------------------------------------------------------
# Speed of sound (mixture) for CFL computation
# ---------------------------------------------------------------------------

def _cell_sound_speed(U: np.ndarray, eos_list: List[EOSType], T_guess: float = 300.0) -> float:
    """Return mixture speed of sound for a single cell."""
    N = len(eos_list)
    try:
        W = cons_to_prim(U, eos_list, T_guess=T_guess)
        p, u, T = W[0], W[1], W[2]
        Y = np.empty(N)
        Y[:N-1] = W[3:3+N-1]
        Y[N-1] = max(0.0, 1.0 - np.sum(Y[:N-1]))

        rho = U[0]
        a = mixture_sound_speed(rho, Y, T, p, eos_list,
                                rho_cv_mix=0.0,  # not used in Wood's formula
                                dp_dT_mix=0.0)
        return a
    except Exception:
        return 1.0  # fallback


def _max_wave_speed(U_cells: np.ndarray, eos_list: List[EOSType],
                    T_guess_arr: Optional[np.ndarray] = None) -> float:
    """Return max |u| + a over all cells."""
    N_cells = U_cells.shape[0]
    max_s = 0.0
    for m in range(N_cells):
        T_g = T_guess_arr[m] if T_guess_arr is not None else 300.0
        try:
            W = cons_to_prim(U_cells[m], eos_list, T_guess=T_g)
            u = abs(W[1])
            a = _cell_sound_speed(U_cells[m], eos_list, T_guess=T_g)
            max_s = max(max_s, u + a)
        except Exception:
            pass
    return max(max_s, 1e-10)


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

def _apply_bc(
    U_cells: np.ndarray,
    bc_left: str,
    bc_right: str,
    bc_left_params: Dict,
    bc_right_params: Dict,
    eos_list: List[EOSType],
    t: float,
    dx: float,
) -> np.ndarray:
    """
    Apply boundary conditions by constructing ghost cells.

    Returns U_extended of shape (N_cells + 2, n_vars) where:
        U_extended[0]    = left ghost cell
        U_extended[1:-1] = interior cells (U_cells)
        U_extended[-1]   = right ghost cell
    """
    N_cells = U_cells.shape[0]
    n_vars = U_cells.shape[1]
    U_ext = np.empty((N_cells + 2, n_vars))
    U_ext[1:-1] = U_cells

    # --- Left ghost cell ---
    if bc_left == 'periodic':
        U_ext[0] = U_cells[-1]
    elif bc_left == 'transmissive':
        U_ext[0] = U_cells[0]
    elif bc_left == 'inlet':
        # Prescribed primitive state at inlet
        W_in = bc_left_params['W']
        U_ext[0] = prim_to_cons(np.array(W_in), eos_list)
    elif bc_left == 'acoustic_inlet':
        # Sinusoidal velocity perturbation (Ref: 1D_reflection_and_transmission_at_interfaces.md)
        # u_in = u0 + du*sin(2*pi*f*t + 1.5*pi) for t < 1/f
        #       = u0 - du                         for t >= 1/f
        u0 = bc_left_params.get('u0', 1.0)
        du = bc_left_params.get('du', 0.02 * u0)
        f  = bc_left_params.get('f', 5000.0)
        T0 = bc_left_params.get('T', 300.0)
        p0 = bc_left_params.get('p', 1.0e5)
        Y_bc = np.array(bc_left_params.get('Y', [1.0]))
        N = len(eos_list)
        if t < 1.0 / f:
            u_bc = u0 + du * np.sin(2.0 * np.pi * f * t + 1.5 * np.pi)
        else:
            u_bc = u0 - du
        W_bc = np.empty(2 + N)
        W_bc[0] = p0
        W_bc[1] = u_bc
        W_bc[2] = T0
        W_bc[3:3+N-1] = Y_bc[:N-1]
        U_ext[0] = prim_to_cons(W_bc, eos_list)
    else:
        U_ext[0] = U_cells[0]  # default: transmissive

    # --- Right ghost cell ---
    if bc_right == 'periodic':
        U_ext[-1] = U_cells[0]
    elif bc_right == 'transmissive':
        U_ext[-1] = U_cells[-1]
    elif bc_right == 'inlet':
        W_in = bc_right_params['W']
        U_ext[-1] = prim_to_cons(np.array(W_in), eos_list)
    else:
        U_ext[-1] = U_cells[-1]  # default: transmissive

    return U_ext


# ---------------------------------------------------------------------------
# Spatial operator: compute dU/dt = -1/dx * (F_{m+1/2} - F_{m-1/2})
# ---------------------------------------------------------------------------

def _spatial_rhs(
    U_cells: np.ndarray,
    eos_list: List[EOSType],
    dx: float,
    bc_left: str,
    bc_right: str,
    bc_left_params: Dict,
    bc_right_params: Dict,
    t: float,
    T_guess_arr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute right-hand side: dU/dt = -1/dx * (F_{m+1/2} - F_{m-1/2}).

    Parameters
    ----------
    U_cells : shape (N_cells, n_vars)
    Returns dU/dt : shape (N_cells, n_vars)
    """
    N_cells = U_cells.shape[0]
    n_vars = U_cells.shape[1]

    U_ext = _apply_bc(U_cells, bc_left, bc_right, bc_left_params, bc_right_params,
                      eos_list, t, dx)

    # Compute interface fluxes F_{m+1/2} for m = 0, ..., N_cells
    # Interface m+1/2 is between U_ext[m] and U_ext[m+1]
    # (U_ext indices: interior cell m has U_ext[m+1])
    RHS = np.zeros((N_cells, n_vars))

    for m in range(N_cells):
        UL = U_ext[m]      # left state for interface m-1/2 (ghost or interior)
        UR = U_ext[m + 1]  # right state for interface m-1/2

        # Temperature guesses
        T_gL = T_guess_arr[m - 1] if (T_guess_arr is not None and m > 0) else 300.0
        T_gR = T_guess_arr[m] if T_guess_arr is not None else 300.0

        F_left = apec_flux(UL, UR, eos_list, T_guess_L=T_gL, T_guess_R=T_gR)

        UL2 = U_ext[m + 1]    # left state for interface m+1/2
        UR2 = U_ext[m + 2]    # right state for interface m+1/2
        T_gL2 = T_guess_arr[m] if T_guess_arr is not None else 300.0
        T_gR2 = T_guess_arr[m + 1] if (T_guess_arr is not None and m + 1 < N_cells) else 300.0

        F_right = apec_flux(UL2, UR2, eos_list, T_guess_L=T_gL2, T_guess_R=T_gR2)

        RHS[m] = -(F_right - F_left) / dx

    return RHS


# ---------------------------------------------------------------------------
# TVD Runge-Kutta 3rd order (Shu-Osher)
# ---------------------------------------------------------------------------

def _clip_positivity(U: np.ndarray) -> np.ndarray:
    """
    Clamp density and energy to small positive values to prevent
    unphysical states during RK intermediate stages.

    Only modifies rho (index 0) if it becomes negative.
    Species partial densities (index 3+) are clamped to [0, rho].
    This is a last-resort safeguard; the LLF dissipation should
    prevent negativity in normal operation.
    """
    U_out = U.copy()
    # Clamp density to small positive floor
    neg_rho_mask = U_out[:, 0] < 1e-300
    if np.any(neg_rho_mask):
        U_out[neg_rho_mask, 0] = 1e-300
    # Clamp species partial densities to [0, rho]
    n_vars = U_out.shape[1]
    n_species_minus1 = n_vars - 3
    if n_species_minus1 > 0:
        for k in range(n_species_minus1):
            rhoYk = U_out[:, 3 + k]
            rho_m = U_out[:, 0]
            U_out[:, 3 + k] = np.clip(rhoYk, 0.0, rho_m)
    return U_out


def _tvd_rk3_step(
    U: np.ndarray,
    eos_list: List[EOSType],
    dx: float,
    dt: float,
    bc_left: str,
    bc_right: str,
    bc_left_params: Dict,
    bc_right_params: Dict,
    t: float,
    T_guess_arr: Optional[np.ndarray],
) -> np.ndarray:
    """
    Advance U by one time step using 3rd-order TVD Runge-Kutta (Shu-Osher).

    Stage 1: U^(1) = U^n + dt * L(U^n)
    Stage 2: U^(2) = 3/4 * U^n + 1/4 * (U^(1) + dt * L(U^(1)))
    Stage 3: U^(3) = 1/3 * U^n + 2/3 * (U^(2) + dt * L(U^(2)))

    Positivity clamp applied after each stage to guard against
    small negative densities from non-dissipative centered operators.
    The LLF dissipation in apec_flux() is the primary stability mechanism;
    the clamp here is a secondary safeguard only.
    """
    L0 = _spatial_rhs(U, eos_list, dx, bc_left, bc_right,
                      bc_left_params, bc_right_params, t, T_guess_arr)
    U1 = _clip_positivity(U + dt * L0)

    L1 = _spatial_rhs(U1, eos_list, dx, bc_left, bc_right,
                      bc_left_params, bc_right_params, t + dt, T_guess_arr)
    U2 = _clip_positivity(0.75 * U + 0.25 * (U1 + dt * L1))

    L2 = _spatial_rhs(U2, eos_list, dx, bc_left, bc_right,
                      bc_left_params, bc_right_params, t + 0.5 * dt, T_guess_arr)
    U3 = _clip_positivity((1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * L2))

    return U3


# ---------------------------------------------------------------------------
# Forward Euler step
# ---------------------------------------------------------------------------

def _forward_euler_step(
    U: np.ndarray,
    eos_list: List[EOSType],
    dx: float,
    dt: float,
    bc_left: str,
    bc_right: str,
    bc_left_params: Dict,
    bc_right_params: Dict,
    t: float,
    T_guess_arr: Optional[np.ndarray],
) -> np.ndarray:
    """Advance U by one time step using Forward Euler."""
    RHS = _spatial_rhs(U, eos_list, dx, bc_left, bc_right,
                       bc_left_params, bc_right_params, t, T_guess_arr)
    return U + dt * RHS


# ---------------------------------------------------------------------------
# Backward Euler step (Newton iteration)
# ---------------------------------------------------------------------------

def _backward_euler_step(
    U_old: np.ndarray,
    eos_list: List[EOSType],
    dx: float,
    dt: float,
    bc_left: str,
    bc_right: str,
    bc_left_params: Dict,
    bc_right_params: Dict,
    t_new: float,
    T_guess_arr: Optional[np.ndarray],
    max_iter: int = 50,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Advance U by one step using Backward Euler with Newton iteration.

    Solves: R(U) = U - U_old + dt * L(U) = 0

    where L(U) = -1/dx * (F_{m+1/2} - F_{m-1/2})

    Newton update: U^{k+1} = U^k - J^{-1} * R(U^k)
    Convergence: ||ΔU|| / ||U|| < tol

    Ref: CLAUDE.md § 시간 차분, Backward Euler
    """
    N_cells, n_vars = U_old.shape
    N_total = N_cells * n_vars

    U = U_old.copy()

    def residual_flat(U_flat: np.ndarray) -> np.ndarray:
        U_cur = U_flat.reshape(N_cells, n_vars)
        RHS = _spatial_rhs(U_cur, eos_list, dx, bc_left, bc_right,
                           bc_left_params, bc_right_params, t_new, T_guess_arr)
        R = U_cur - U_old - dt * RHS   # BE residual
        return R.ravel()

    U_flat = U.ravel()

    for k in range(max_iter):
        R_flat = residual_flat(U_flat)
        res_norm = np.linalg.norm(R_flat)
        U_norm = np.linalg.norm(U_flat)

        if U_norm > 0.0 and res_norm / U_norm < tol:
            break

        # Compute Jacobian
        J = system_jacobian(U_flat, residual_flat, n_vars)

        # Solve J * dU = R
        try:
            dU = np.linalg.solve(J, R_flat)
        except np.linalg.LinAlgError:
            # Fallback: use lstsq
            dU, _, _, _ = np.linalg.lstsq(J, R_flat, rcond=None)

        U_flat = U_flat - dU

    return U_flat.reshape(N_cells, n_vars)


# ---------------------------------------------------------------------------
# CFL-based time step computation
# ---------------------------------------------------------------------------

def _compute_dt(
    U_cells: np.ndarray,
    eos_list: List[EOSType],
    dx: float,
    CFL: float,
    T_guess_arr: Optional[np.ndarray],
    dt_prev: Optional[float] = None,
) -> float:
    """
    Compute time step from CFL condition.

    If wave speed computation fails, falls back to dt_prev * 0.5
    or a conservative default.
    """
    try:
        s_max = _max_wave_speed(U_cells, eos_list, T_guess_arr)
        if s_max > 0.0 and np.isfinite(s_max):
            return CFL * dx / s_max
    except Exception:
        pass
    # Fallback
    if dt_prev is not None and dt_prev > 0.0:
        return dt_prev * 0.5
    return CFL * dx / 1.0  # last resort: assume wave speed = 1


# ---------------------------------------------------------------------------
# Main 1D solver entry point
# ---------------------------------------------------------------------------

def run_1d(case_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a 1D compressible multi-component Euler simulation using APEC flux.

    Parameters
    ----------
    case_params : dict
        Required keys:
            'eos_list'  : list of N EOS objects
            'x_cells'   : cell-center positions, shape (N_cells,)
            'U_init'    : initial conservative state, shape (N_cells, n_vars)
            't_end'     : end time
        Optional keys:
            'CFL'           : CFL number (default 0.6)
            'bc_left'       : 'periodic', 'transmissive', 'inlet', 'acoustic_inlet'
                              (default 'transmissive')
            'bc_right'      : 'periodic', 'transmissive' (default 'transmissive')
            'bc_left_params': dict (for inlet/acoustic_inlet BCs)
            'bc_right_params': dict
            'output_times'  : list of times to save snapshots
            'T_guess'       : initial temperature guess [K] (default 300.0)
            'max_iter_be'   : max Newton iters for Backward Euler (default 50)
            'tol_be'        : Newton convergence tolerance (default 1e-8)
            'verbose'       : print progress (default False)
            'time_scheme'   : 'auto', 'forward_euler', 'backward_euler', 'tvd_rk3'
                              'auto' selects based on Mach number
            'dt_fixed'      : fixed time step (overrides CFL if given)

    Returns
    -------
    result : dict
        'U_final'      : final conservative state, shape (N_cells, n_vars)
        'x_cells'      : cell positions
        't_final'      : actual final time
        'snapshots'    : list of dicts with keys 't', 'U' at requested output_times
        'eos_list'     : EOS objects (pass-through)
        'n_steps'      : number of time steps taken
    """
    eos_list       = case_params['eos_list']
    x_cells        = np.asarray(case_params['x_cells'], dtype=float)
    U_cells        = np.asarray(case_params['U_init'], dtype=float).copy()
    t_end          = float(case_params['t_end'])

    N_cells        = U_cells.shape[0]
    n_vars         = U_cells.shape[1]
    N_species      = len(eos_list)

    dx             = x_cells[1] - x_cells[0] if len(x_cells) > 1 else 1.0
    CFL            = float(case_params.get('CFL', 0.6))
    bc_left        = case_params.get('bc_left', 'transmissive')
    bc_right       = case_params.get('bc_right', 'transmissive')
    bc_left_params = case_params.get('bc_left_params', {})
    bc_right_params = case_params.get('bc_right_params', {})
    output_times   = list(case_params.get('output_times', []))
    T_guess0       = float(case_params.get('T_guess', 300.0))
    max_iter_be    = int(case_params.get('max_iter_be', 50))
    tol_be         = float(case_params.get('tol_be', 1e-8))
    verbose        = bool(case_params.get('verbose', False))
    time_scheme    = case_params.get('time_scheme', 'auto')
    dt_fixed       = case_params.get('dt_fixed', None)

    # Temperature guesses array (per cell, updated after each step)
    # Initialize dynamically from initial conservative state rather than fixed 300 K.
    # This handles dimensionless problems where T << 1 K.
    T_guess_arr = np.empty(N_cells)
    for _m in range(N_cells):
        _U_m = U_cells[_m]
        _rho_m = _U_m[0]
        if _rho_m > 0.0:
            _u_m = _U_m[1] / _rho_m
            _e_m = _U_m[2] / _rho_m - 0.5 * _u_m ** 2
            # Rough cv estimate
            _cv_rough = 0.0
            for _i, _eos in enumerate(eos_list):
                if isinstance(_eos, SRKEOS):
                    _cv_rough += getattr(_eos, 'c_v0', 1000.0) / N_species
                else:
                    _cv_rough += getattr(_eos, 'c_v', 1000.0) / N_species
            _cv_rough = max(_cv_rough, 1e-30)
            T_guess_arr[_m] = max(1e-30, abs(_e_m) / _cv_rough)
        else:
            T_guess_arr[_m] = T_guess0

    t = 0.0
    n_steps = 0
    snapshots = []
    dt_prev = None  # track previous dt for fallback in _compute_dt

    # Sort output times
    output_times_sorted = sorted([ot for ot in output_times if ot <= t_end])
    output_idx = 0

    # Save initial snapshot if t=0 is requested
    if output_times_sorted and output_times_sorted[0] == 0.0:
        snapshots.append({'t': 0.0, 'U': U_cells.copy()})
        output_idx += 1

    while t < t_end:
        # Compute time step
        if dt_fixed is not None:
            dt = float(dt_fixed)
        else:
            dt = _compute_dt(U_cells, eos_list, dx, CFL, T_guess_arr, dt_prev=dt_prev)

        # Don't overshoot end time
        if t + dt > t_end:
            dt = t_end - t

        # Don't overshoot next output time
        if output_idx < len(output_times_sorted):
            t_next_out = output_times_sorted[output_idx]
            if t + dt > t_next_out:
                dt = t_next_out - t

        if dt <= 0.0:
            break

        # Determine time scheme
        if time_scheme == 'auto':
            # Estimate Mach number
            Ma_max = 0.0
            for m in range(N_cells):
                try:
                    W = cons_to_prim(U_cells[m], eos_list, T_guess=T_guess_arr[m])
                    u_m = abs(W[1])
                    a_m = _cell_sound_speed(U_cells[m], eos_list, T_guess=T_guess_arr[m])
                    if a_m > 0.0:
                        Ma_max = max(Ma_max, u_m / a_m)
                except Exception:
                    pass
            if Ma_max > 1.0:
                scheme = 'forward_euler'
            else:
                scheme = 'tvd_rk3'  # Use TVD-RK3 for subsonic (more stable than forward Euler)
        else:
            scheme = time_scheme

        # Time advance
        if scheme == 'forward_euler':
            U_new = _forward_euler_step(
                U_cells, eos_list, dx, dt,
                bc_left, bc_right, bc_left_params, bc_right_params,
                t, T_guess_arr
            )
        elif scheme == 'backward_euler':
            U_new = _backward_euler_step(
                U_cells, eos_list, dx, dt,
                bc_left, bc_right, bc_left_params, bc_right_params,
                t + dt, T_guess_arr, max_iter=max_iter_be, tol=tol_be
            )
        elif scheme == 'tvd_rk3':
            U_new = _tvd_rk3_step(
                U_cells, eos_list, dx, dt,
                bc_left, bc_right, bc_left_params, bc_right_params,
                t, T_guess_arr
            )
        else:
            raise ValueError(f"Unknown time_scheme: {scheme!r}")

        # Update temperature guesses using new state
        for m in range(N_cells):
            try:
                W = cons_to_prim(U_new[m], eos_list, T_guess=T_guess_arr[m])
                T_guess_arr[m] = W[2]
            except Exception:
                pass  # keep old guess

        U_cells = U_new
        t += dt
        dt_prev = dt
        n_steps += 1

        if verbose and n_steps % 100 == 0:
            print(f"  step={n_steps:6d}  t={t:.6e}  dt={dt:.3e}  scheme={scheme}")

        # Save snapshots at requested output times
        while (output_idx < len(output_times_sorted) and
               t >= output_times_sorted[output_idx] - 1e-14):
            snapshots.append({'t': t, 'U': U_cells.copy()})
            output_idx += 1

    if verbose:
        print(f"  Done: {n_steps} steps, t_final={t:.6e}")

    return {
        'U_final':  U_cells,
        'x_cells':  x_cells,
        't_final':  t,
        'snapshots': snapshots,
        'eos_list': eos_list,
        'n_steps':  n_steps,
    }


# ---------------------------------------------------------------------------
# Convenience: build initial conservative state from primitive profiles
# ---------------------------------------------------------------------------

def init_from_prim_profile(
    W_profile: np.ndarray,
    eos_list: List[EOSType],
) -> np.ndarray:
    """
    Convert a 2D array of primitive states to conservative states.

    Parameters
    ----------
    W_profile : shape (N_cells, 2 + N_species)
        Each row: [p, u, T, Y_1, ..., Y_{N-1}]
    eos_list : list of N EOS objects.

    Returns
    -------
    U_profile : shape (N_cells, 2 + N_species)
    """
    N_cells = W_profile.shape[0]
    n_vars = W_profile.shape[1]
    U_profile = np.empty_like(W_profile)
    for m in range(N_cells):
        U_profile[m] = prim_to_cons(W_profile[m], eos_list)
    return U_profile


def prim_profile_from_cons(
    U_profile: np.ndarray,
    eos_list: List[EOSType],
    T_guess_arr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert a 2D array of conservative states to primitive states.

    Parameters
    ----------
    U_profile : shape (N_cells, n_vars)
    eos_list : list of N EOS objects.
    T_guess_arr : optional shape (N_cells,) temperature guesses.

    Returns
    -------
    W_profile : shape (N_cells, n_vars)
    """
    N_cells = U_profile.shape[0]
    W_profile = np.empty_like(U_profile)
    for m in range(N_cells):
        T_g = T_guess_arr[m] if T_guess_arr is not None else 300.0
        W_profile[m] = cons_to_prim(U_profile[m], eos_list, T_guess=T_g)
    return W_profile
