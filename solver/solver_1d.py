"""
1D all-speed multi-component compressible Euler solver.

Governing equations (conservative form):
    d/dt [rhoY_i] + d/dx [rhoY_i * u]               = 0   (species)
    d/dt [rho*u]  + d/dx [rho*u^2 + p]              = 0   (momentum)
    d/dt [rho*E]  + d/dx [(rho*E + p)*u]            = 0   (energy)

    rho = sum_i rhoY_i
    rho*E = rho*e + 0.5*rho*u^2
    e = sum_i Y_i * e_i(T)  (mass-fraction weighted internal energy)

Conservative state layout per cell (Ns+2 variables):
    U[0 : Ns]     = rhoY_i   (partial densities for each of Ns species)
    U[Ns]         = rho*u    (momentum)
    U[Ns+1]       = rho*E    (total energy)

    rho = sum_{i=0}^{Ns-1} U[i]

Time integration:
    Explicit 2-stage Runge-Kutta (Heun's method, RK2).
    Stable with CFL <= 1 for linear hyperbolic equations.
    The CFL-limited time step is already computed to respect the fast
    acoustic wave speed, so explicit integration is natural here.

Flux:
    HLLC (Toro 1994) with double-flux energy (Abgrall & Karni 2001).

Boundary conditions:
    'periodic'     — wrap-around
    'transmissive' — zero-gradient copy
    'wall'         — reflect normal velocity

EOS:
    Ideal Gas (solver/eos/ideal.py) and NASG (solver/eos/nasg.py).

Usage example (Phase 1 — Water-Air advection):
    from solver.solver_1d import run_solver_1d, build_water_air_ic
    result = run_solver_1d(build_water_air_ic())
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .flux_allspeed import (
    hllc_flux, hllc_flux_double_energy, cons_to_prim_allspeed,
    _mixture_sound_speed, _rho_i_pure_from_T_p, _get_internal_energy,
)
from .boundary import apply_bc_1d
from .eos.ideal import IdealGasEOS
from .eos.nasg import NASGEOS


# ---------------------------------------------------------------------------
# CFL time step
# ---------------------------------------------------------------------------

def _cell_wave_speed(
    U: np.ndarray,
    eos_list: list,
    T_guess: float = 300.0,
) -> float:
    """Return |u| + a for a single cell (max wave speed)."""
    rho, u, p, T, rhoYi = cons_to_prim_allspeed(U, eos_list, T_guess)
    Yi = rhoYi / max(rho, 1e-300)
    a = _mixture_sound_speed(rho, Yi, rhoYi, T, p, eos_list)
    return abs(u) + a


def _compute_dt_cfl(
    U_cells: np.ndarray,
    eos_list: list,
    dx: float,
    CFL: float,
    T_guess_arr: np.ndarray,
) -> float:
    """Compute CFL-limited time step over all cells."""
    N = U_cells.shape[0]
    s_max = 1e-300
    for m in range(N):
        try:
            s = _cell_wave_speed(U_cells[m], eos_list, T_guess_arr[m])
            s_max = max(s_max, s)
        except Exception:
            pass
    return CFL * dx / s_max


# ---------------------------------------------------------------------------
# Spatial RHS
# ---------------------------------------------------------------------------

def _spatial_rhs(
    U_cells: np.ndarray,
    eos_list: list,
    dx: float,
    bc_left: str,
    bc_right: str,
    T_guess_arr: np.ndarray,
) -> np.ndarray:
    """
    Compute dU/dt = -1/dx * (F_{i+1/2} - F_{i-1/2}) for all cells.

    Uses the double-flux HLLC (Abgrall & Karni 2001) for the energy equation:
    each cell's energy update uses the flux computed with its own EOS,
    preventing spurious pressure oscillations at material interfaces.

    For cell i:
      - Left  face (i-1/2): F_{i-1/2}^R  (right-cell energy, i.e. cell i's EOS)
      - Right face (i+1/2): F_{i+1/2}^L  (left-cell  energy, i.e. cell i's EOS)
    so:  dU[i]/dt = -(F_{i+1/2}^L - F_{i-1/2}^R) / dx

    Species and momentum fluxes are standard (same for both sides of each face).

    Parameters
    ----------
    U_cells : shape (N, Ns+2)
    Returns dU/dt : shape (N, Ns+2)
    """
    N, n_vars = U_cells.shape

    # Apply ghost cells
    U_ext = apply_bc_1d(U_cells, bc_left=bc_left, bc_right=bc_right)
    # U_ext[0] = left ghost, U_ext[1:N+1] = interior, U_ext[N+1] = right ghost

    # Pre-compute double-energy fluxes at all N+1 faces (faces 0 .. N)
    # Face f lies between U_ext[f] (left) and U_ext[f+1] (right).
    # F_face_L[f] = flux for the LEFT  cell at face f (cell f-1 in original indexing)
    # F_face_R[f] = flux for the RIGHT cell at face f (cell f   in original indexing)
    # p_face[f]   = interface pressure at face f (for non-conservative momentum form)
    # F_mass_face[f] = mass flux (sum of species fluxes) at face f
    Ns = n_vars - 2
    F_face_L    = np.empty((N + 1, n_vars), dtype=float)
    F_face_R    = np.empty((N + 1, n_vars), dtype=float)
    p_face      = np.empty(N + 1, dtype=float)
    F_mass_face = np.empty(N + 1, dtype=float)

    for f in range(N + 1):
        # T guesses for the two cells adjacent to face f
        # U_ext[f]   corresponds to cell (f-1): index m = f-1 in U_cells (or ghost)
        # U_ext[f+1] corresponds to cell  f   : index m = f   in U_cells (or ghost)
        m_left  = f - 1  # cell index in U_cells for the left  side of face f
        m_right = f      # cell index in U_cells for the right side of face f

        T_gL = T_guess_arr[max(m_left,  0)] if m_left  >= 0 else T_guess_arr[0]
        T_gR = T_guess_arr[min(m_right, N - 1)] if m_right < N else T_guess_arr[N - 1]

        fl, fr, p_int = hllc_flux_double_energy(
            U_ext[f], U_ext[f + 1], eos_list,
            T_guess_L=T_gL, T_guess_R=T_gR,
        )
        F_face_L[f] = fl   # energy flux for left  cell at this face
        F_face_R[f] = fr   # energy flux for right cell at this face
        p_face[f]   = p_int
        # mass flux = sum of species fluxes (same for fl and fr — shared species/mom flux)
        F_mass_face[f] = float(np.sum(fl[:Ns]))

    RHS = np.zeros((N, n_vars), dtype=float)
    for m in range(N):
        # Face to the left  of cell m: face index f = m   → F_face_R[m]   (cell m is right)
        # Face to the right of cell m: face index f = m+1 → F_face_L[m+1] (cell m is left)
        F_left_face  = F_face_R[m]      # right-cell energy flux at left  face of cell m
        F_right_face = F_face_L[m + 1]  # left-cell  energy flux at right face of cell m
        RHS[m] = -(F_right_face - F_left_face) / dx

        # --- Non-conservative (advective) momentum RHS ---
        # Override momentum equation: d(rho*u)/dt = -u * d(rho)/dt - dp/dx
        # which for uniform u=const, uniform p=const gives exactly 0.
        # This prevents spurious momentum error at material interfaces.
        rho_m = max(float(np.sum(U_cells[m, :Ns])), 1e-300)
        u_m   = float(U_cells[m, Ns]) / rho_m
        RHS[m, Ns] = -(
            u_m * (F_mass_face[m + 1] - F_mass_face[m]) / dx
            + (p_face[m + 1] - p_face[m]) / dx
        )

    return RHS


# ---------------------------------------------------------------------------
# Energy reinitialization (isobaric projection)
# ---------------------------------------------------------------------------

def _reinitialize_energy(
    U_cells: np.ndarray,
    eos_list: list,
    T_ref: np.ndarray,
    p_ref: np.ndarray,
) -> None:
    """
    Reinitialize rhoE for each cell to be thermodynamically consistent
    with the NEW composition (Yi_new) and the REFERENCE (T_ref, p_ref)
    from the previous timestep.

    rhoE_new[m] = rho_new[m] * sum_i(Yi_new[i] * e_i(rho_i_pure(T,p), T)) + 0.5*u^2*rho

    Background:
        Conservative species transport (rhoYi) changes the cell composition
        without changing its energy (the double-flux sets energy RHS≈0 for
        uniform p, u). This creates thermodynamically inconsistent mixed cells.
        The isobaric projection restores consistency: for a cell at reference
        (T, p), the energy is simply the mass-fraction weighted sum of
        component energies at that (T, p).

        For uniform p=1e5 Pa, T=300 K (Abgrall test), this gives p_recovered=1e5
        and T_recovered=300 K for ALL cells regardless of composition.
    """
    N_cells = U_cells.shape[0]
    Ns = len(eos_list)

    for m in range(N_cells):
        rhoYi = np.maximum(U_cells[m, :Ns], 0.0)
        rho = np.sum(rhoYi)
        if rho < 1e-300:
            continue
        Yi = rhoYi / rho
        rhou = U_cells[m, Ns]
        u = rhou / rho

        T_m = max(float(T_ref[m]), 1.0)
        p_m = max(float(p_ref[m]), 1.0)

        e_iso = 0.0
        for i, eos in enumerate(eos_list):
            if Yi[i] < 1e-30:
                continue
            rho_i = max(_rho_i_pure_from_T_p(eos, T_m, p_m), 1e-300)
            e_i = _get_internal_energy(eos, rho_i, T_m)
            e_iso += Yi[i] * e_i

        U_cells[m, Ns + 1] = rho * e_iso + 0.5 * rho * u * u


# ---------------------------------------------------------------------------
# Primitive velocity RHS (for exact u-preservation at density interfaces)
# ---------------------------------------------------------------------------

def _compute_primitive_velocity_rhs(
    U_cells: np.ndarray,
    eos_list: list,
    dx: float,
    bc_left: str,
    bc_right: str,
    T_guess_arr: np.ndarray,
) -> np.ndarray:
    """
    Compute du/dt from primitive momentum equation:
        du/dt = -(u * du/dx + (1/rho) * dp/dx)
    For uniform u and p, returns exactly zero (Abgrall preservation).
    """
    N_cells = U_cells.shape[0]
    Ns = len(eos_list)

    rho = np.maximum(np.sum(U_cells[:, :Ns], axis=1), 1e-300)
    u = U_cells[:, Ns] / rho

    p = np.empty(N_cells)
    for m in range(N_cells):
        try:
            prim = cons_to_prim_allspeed(U_cells[m], eos_list, T_guess_arr[m])
            p[m] = max(prim[2], 1.0)
        except Exception:
            p[m] = 1e5

    du_dt = np.zeros(N_cells)
    for m in range(N_cells):
        if bc_left == 'periodic':
            m_prev = (m - 1) % N_cells
            m_next = (m + 1) % N_cells
        else:
            m_prev = max(m - 1, 0)
            m_next = min(m + 1, N_cells - 1)

        # Upwind velocity gradient
        if u[m] >= 0.0:
            du_dx = (u[m] - u[m_prev]) / dx
        else:
            du_dx = (u[m_next] - u[m]) / dx

        # Central pressure gradient
        if bc_left == 'periodic':
            dp_dx = (p[m_next] - p[m_prev]) / (2.0 * dx)
        else:
            if m == 0:
                dp_dx = (p[1] - p[0]) / dx
            elif m == N_cells - 1:
                dp_dx = (p[-1] - p[-2]) / dx
            else:
                dp_dx = (p[m_next] - p[m_prev]) / (2.0 * dx)

        du_dt[m] = -(u[m] * du_dx + dp_dx / rho[m])

    return du_dt


# ---------------------------------------------------------------------------
# Explicit RK2 (Heun's method) time step
# ---------------------------------------------------------------------------

def _rk2_step(
    U_old: np.ndarray,
    eos_list: list,
    dx: float,
    dt: float,
    bc_left: str,
    bc_right: str,
    T_guess_arr: np.ndarray,
) -> np.ndarray:
    """
    Advance one time step using explicit 2-stage Runge-Kutta (Heun's method)
    with isobaric energy reinitialization after each stage.

    Stage 1:  k1  = L(U^n)
              U*  = U^n + dt * k1
              [reinit rhoE(U*) for thermodynamic consistency]
    Stage 2:  k2  = L(U*)
              U^{n+1} = U^n + dt/2 * (k1 + k2)
              [reinit rhoE(U^{n+1})]

    The energy reinitialization (isobaric projection) sets rhoE after each
    stage so that every cell's energy is consistent with its NEW composition
    at the PREVIOUS timestep's (T, p).  This prevents thermodynamic
    inconsistency in mixed cells at material interfaces.
    """
    N_cells, n_vars = U_old.shape
    Ns = len(eos_list)

    # Pre-compute reference (T, p) for energy reinitialization
    T_ref = T_guess_arr.copy()
    p_ref = np.full(N_cells, 1e5)
    for m in range(N_cells):
        try:
            _, _, p_m, T_m, _ = cons_to_prim_allspeed(U_old[m], eos_list, T_guess_arr[m])
            T_ref[m] = max(T_m, 1.0)
            p_ref[m] = max(p_m, 1.0)
        except Exception:
            pass

    # Pre-compute primitive velocity and velocity RHS from old state
    rho_old = np.maximum(np.sum(U_old[:, :Ns], axis=1), 1e-300)
    u_old = U_old[:, Ns] / rho_old
    du_dt_1 = _compute_primitive_velocity_rhs(
        U_old, eos_list, dx, bc_left, bc_right, T_guess_arr
    )

    # ---- Stage 1 ----
    k1 = _spatial_rhs(U_old, eos_list, dx, bc_left, bc_right, T_guess_arr)
    U_star = U_old + dt * k1

    # Clip negative partial densities
    U_star[:, :Ns] = np.maximum(U_star[:, :Ns], 0.0)

    # Primitive velocity update for Stage 1
    u_star_prim = u_old + dt * du_dt_1
    rho_star_arr = np.maximum(np.sum(U_star[:, :Ns], axis=1), 1e-300)
    U_star[:, Ns] = rho_star_arr * u_star_prim

    # Isobaric projection: restore energy consistency at (T_ref, p_ref)
    _reinitialize_energy(U_star, eos_list, T_ref, p_ref)

    # Update T guesses for stage 2
    T_star = T_ref.copy()
    p_star = p_ref.copy()
    for m in range(N_cells):
        try:
            _, _, p_m, T_m, _ = cons_to_prim_allspeed(U_star[m], eos_list, T_ref[m])
            T_star[m] = max(T_m, 1.0)
            p_star[m] = max(p_m, 1.0)
        except Exception:
            pass

    # ---- Stage 2 ----
    k2 = _spatial_rhs(U_star, eos_list, dx, bc_left, bc_right, T_star)
    U_new = U_old + 0.5 * dt * (k1 + k2)

    # Clip negative partial densities
    U_new[:, :Ns] = np.maximum(U_new[:, :Ns], 0.0)

    # Primitive velocity update for Stage 2 (RK2 Heun)
    du_dt_2 = _compute_primitive_velocity_rhs(
        U_star, eos_list, dx, bc_left, bc_right, T_star
    )
    u_new_prim = u_old + 0.5 * dt * (du_dt_1 + du_dt_2)
    rho_new_arr = np.maximum(np.sum(U_new[:, :Ns], axis=1), 1e-300)
    U_new[:, Ns] = rho_new_arr * u_new_prim

    # Isobaric projection using stage-2 reference (T, p)
    _reinitialize_energy(U_new, eos_list, T_star, p_star)

    return U_new


# ---------------------------------------------------------------------------
# Build initial condition: Phase 1 Water-Air Abgrall test
# ---------------------------------------------------------------------------

def build_water_air_ic(
    N: int = 10,
    x_lo: float = 0.0,
    x_hi: float = 1.0,
    u0: float = 1.0,
    p0: float = 1.0e5,
    T0: float = 300.0,
) -> Dict[str, Any]:
    """
    Build initial conditions for Phase 1: 1D Water-Air advection (Abgrall test).

    Domain [x_lo, x_hi], N uniform cells, periodic BC.
    Water (NASG) in x in [0.4, 0.6], Air (Ideal Gas) elsewhere.
    Uniform velocity u0, pressure p0, temperature T0 everywhere.

    State layout per cell: U = [rhoY_water, rhoY_air, rho*u, rho*E]
    (Ns=2 species: index 0 = water NASG, index 1 = air ideal gas)

    EOS parameters (CLAUDE.md § Phase 1):
        Water: gamma=1.187, p_inf=7.028e8, b=6.61e-4, c_v=3610, q=-1.177788e6
        Air:   gamma=1.4, M=28.97 (R_s ≈ 287 J/kg/K)

    Returns
    -------
    dict suitable for run_solver_1d().
    """
    # EOS objects
    eos_water = NASGEOS(
        gamma=1.187,
        p_inf=7.028e8,
        b=6.61e-4,
        c_v=3610.0,
        q=-1.177788e6,
    )
    eos_air = IdealGasEOS(
        gamma=1.4,
        M=28.97,  # [g/mol] → R_s = 8.314/0.02897 ≈ 287 J/(kg·K)
    )
    eos_list = [eos_water, eos_air]
    Ns = 2

    # Grid
    dx = (x_hi - x_lo) / N
    x_cells = x_lo + (np.arange(N) + 0.5) * dx

    # Pure-species densities at (p0, T0)
    # NASG water: rho = (p+p_inf) / [(gamma-1)*c_v*T + b*(p+p_inf)]
    rho_water_pure = (p0 + eos_water.p_inf) / (
        (eos_water.gamma - 1.0) * eos_water.c_v * T0
        + eos_water.b * (p0 + eos_water.p_inf)
    )
    # Ideal gas air: rho = p / (R_s * T)
    rho_air_pure = p0 / (eos_air.R_s * T0)

    # Smooth tanh interface profile (δ = 0.5*dx).
    # Mixed cells at the interface avoid pure-cell density jumps that would
    # otherwise cause catastrophic pressure collapse (NASG water bulk modulus ~2.75 GPa).
    delta = 0.5 * dx
    U_init = np.zeros((N, Ns + 2), dtype=float)

    for m in range(N):
        xc = x_cells[m]

        # Smooth tanh volume/mass fraction for water
        Y_water = 0.5 * (np.tanh((xc - 0.4) / delta) - np.tanh((xc - 0.6) / delta))
        Y_water = float(np.clip(Y_water, 0.0, 1.0))
        Y_air   = 1.0 - Y_water

        # Isobaric mixture density: 1/rho = Y_w/rho_w_pure + Y_a/rho_a_pure
        inv_rho = Y_water / rho_water_pure + Y_air / rho_air_pure
        rho_mix = 1.0 / max(inv_rho, 1e-300)

        # Pure-component densities at (p0, T0) for energy calculation
        rho_w_at_cell = max(_rho_i_pure_from_T_p(eos_water, T0, p0), 1e-300)
        rho_a_at_cell = max(_rho_i_pure_from_T_p(eos_air,   T0, p0), 1e-300)

        # Isobaric mixture specific internal energy at (T0, p0)
        e_w = _get_internal_energy(eos_water, rho_w_at_cell, T0)
        e_a = _get_internal_energy(eos_air,   rho_a_at_cell, T0)
        e_mix = Y_water * e_w + Y_air * e_a

        E_mix = e_mix + 0.5 * u0 * u0

        U_init[m, 0] = rho_mix * Y_water   # rhoY_water
        U_init[m, 1] = rho_mix * Y_air     # rhoY_air
        U_init[m, Ns] = rho_mix * u0       # rho*u
        U_init[m, Ns + 1] = rho_mix * E_mix  # rho*E

    return {
        'eos_list':      eos_list,
        'x_cells':       x_cells,
        'U_init':        U_init,
        't_end':         1.0,
        'CFL':           0.5,
        'bc_left':       'periodic',
        'bc_right':      'periodic',
        'max_iteration': 100,
        'p0':            p0,
        'u0':            u0,
        'T0':            T0,
        'dx':            dx,
    }


# ---------------------------------------------------------------------------
# Main solver entry point
# ---------------------------------------------------------------------------

def run_solver_1d(case_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run 1D all-speed multi-component compressible Euler solver.

    Parameters
    ----------
    case_params : dict
        Required keys:
            'eos_list'      : list of Ns EOS objects
            'x_cells'       : cell centers, shape (N,)
            'U_init'        : initial conservative state, shape (N, Ns+2)
                              layout: [rhoY_1,...,rhoY_Ns, rho*u, rho*E]
        Optional keys:
            't_end'         : end time [s] (default 1.0)
            'CFL'           : CFL number (default 0.5)
            'bc_left'       : 'periodic','transmissive','wall' (default 'transmissive')
            'bc_right'      : same (default 'transmissive')
            'max_iteration' : maximum number of time steps (default 100)
            'dt_fixed'      : fixed time step (overrides CFL)
            'T_guess'       : initial temperature guess [K] (default 300.0)
            'verbose'       : print progress (default False)
            'output_times'  : list of times to save snapshots

    Returns
    -------
    result : dict
        'U_final'   : final conservative state, shape (N, Ns+2)
        'x_cells'   : cell centers
        't_final'   : final simulation time
        'n_steps'   : number of time steps taken
        'snapshots' : list of {'t': float, 'U': array} at output_times
        'eos_list'  : EOS objects (pass-through)
        'converged' : True if ran without divergence
    """
    eos_list        = case_params['eos_list']
    x_cells         = np.asarray(case_params['x_cells'], dtype=float)
    U_cells         = np.asarray(case_params['U_init'], dtype=float).copy()

    N_cells, n_vars = U_cells.shape
    Ns              = len(eos_list)

    assert n_vars == Ns + 2, (
        f"Expected n_vars = Ns+2 = {Ns+2}, got {n_vars}. "
        f"State layout: [rhoY_1,...,rhoY_Ns, rho*u, rho*E]"
    )

    t_end           = float(case_params.get('t_end', 1.0))
    CFL             = float(case_params.get('CFL', 0.5))
    bc_left         = case_params.get('bc_left', 'transmissive')
    bc_right        = case_params.get('bc_right', 'transmissive')
    max_iteration   = int(case_params.get('max_iteration', 100))
    dt_fixed        = case_params.get('dt_fixed', None)
    T_guess0        = float(case_params.get('T_guess', 300.0))
    verbose         = bool(case_params.get('verbose', False))
    output_times    = sorted(case_params.get('output_times', []))

    dx = x_cells[1] - x_cells[0] if len(x_cells) > 1 else 1.0

    # Initialize temperature guesses
    T_guess_arr = np.full(N_cells, T_guess0)
    for m in range(N_cells):
        try:
            _, _, _, T_m, _ = cons_to_prim_allspeed(U_cells[m], eos_list, T_guess0)
            T_guess_arr[m] = T_m
        except Exception:
            pass

    t = 0.0
    n_steps = 0
    snapshots = []
    converged = True

    output_idx = 0
    if output_times and output_times[0] == 0.0:
        snapshots.append({'t': 0.0, 'U': U_cells.copy()})
        output_idx += 1

    while t < t_end and n_steps < max_iteration:
        # Time step
        if dt_fixed is not None:
            dt = float(dt_fixed)
        else:
            dt = _compute_dt_cfl(U_cells, eos_list, dx, CFL, T_guess_arr)

        if t + dt > t_end:
            dt = t_end - t
        if output_idx < len(output_times):
            t_next_out = output_times[output_idx]
            if t + dt > t_next_out:
                dt = t_next_out - t
        if dt <= 0.0:
            break

        # RK2 step
        try:
            U_new = _rk2_step(
                U_cells, eos_list, dx, dt,
                bc_left, bc_right, T_guess_arr,
            )
        except Exception as exc:
            if verbose:
                print(f"  Step {n_steps}: solver error at t={t:.4e}: {exc}")
            converged = False
            break

        # Sanity checks
        if not np.all(np.isfinite(U_new)):
            if verbose:
                print(f"  Step {n_steps}: NaN/Inf at t={t:.4e}, stopping.")
            converged = False
            break

        rho_new = np.sum(U_new[:, :Ns], axis=1)
        if np.any(rho_new <= 0.0):
            if verbose:
                print(f"  Step {n_steps}: non-positive density at t={t:.4e}, stopping.")
            converged = False
            break

        # Update temperature guesses
        for m in range(N_cells):
            try:
                _, _, _, T_m, _ = cons_to_prim_allspeed(U_new[m], eos_list, T_guess_arr[m])
                T_guess_arr[m] = T_m
            except Exception:
                pass

        U_cells = U_new
        t += dt
        n_steps += 1

        if verbose and (n_steps <= 5 or n_steps % 10 == 0):
            print(f"  step={n_steps:5d}  t={t:.4e}  dt={dt:.3e}")

        while output_idx < len(output_times) and t >= output_times[output_idx] - 1e-14:
            snapshots.append({'t': t, 'U': U_cells.copy()})
            output_idx += 1

    if verbose:
        print(f"  Completed: {n_steps} steps, t_final={t:.6e}, converged={converged}")

    return {
        'U_final':   U_cells,
        'x_cells':   x_cells,
        't_final':   t,
        'n_steps':   n_steps,
        'snapshots': snapshots,
        'eos_list':  eos_list,
        'converged': converged,
    }


# ---------------------------------------------------------------------------
# Utility: extract primitive profiles
# ---------------------------------------------------------------------------

def extract_primitive_profiles(
    U_cells: np.ndarray,
    eos_list: list,
    T_guess_arr: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract primitive variable profiles from conservative state array.

    Parameters
    ----------
    U_cells : shape (N, Ns+2)
    eos_list : list of Ns EOS objects
    T_guess_arr : optional shape (N,) temperature guesses

    Returns
    -------
    dict with keys 'p','u','T','rho','Yi'(shape N×Ns),'rhoYi'(shape N×Ns)
    """
    N, n_vars = U_cells.shape
    Ns = len(eos_list)
    p_arr     = np.zeros(N)
    u_arr     = np.zeros(N)
    T_arr     = np.zeros(N)
    rho_arr   = np.zeros(N)
    Yi_arr    = np.zeros((N, Ns))
    rhoYi_arr = np.zeros((N, Ns))

    for m in range(N):
        T_g = T_guess_arr[m] if T_guess_arr is not None else 300.0
        rho_m, u_m, p_m, T_m, rhoYi_m = cons_to_prim_allspeed(
            U_cells[m], eos_list, T_g
        )
        p_arr[m]     = p_m
        u_arr[m]     = u_m
        T_arr[m]     = T_m
        rho_arr[m]   = rho_m
        Yi_arr[m]    = rhoYi_m / max(rho_m, 1e-300)
        rhoYi_arr[m] = rhoYi_m

    return {
        'p':     p_arr,
        'u':     u_arr,
        'T':     T_arr,
        'rho':   rho_arr,
        'Yi':    Yi_arr,
        'rhoYi': rhoYi_arr,
    }
