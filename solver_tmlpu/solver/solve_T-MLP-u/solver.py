"""Main FVM driver — equation-, grid-, reconstruction-, flux-,
integrator-agnostic.

API:

    solve(mesh, eq, U0, *,
          reconstruction='first_order',
          flux='llf',
          integrator='ssp_rk2',
          bc=None,
          cfl=0.5, dt_fixed=None,
          t_end, max_steps=200_000,
          history_every=0)

Returns a dict with 'U_final', 't', 'n_steps', 'history'.

The driver works for any combination of:
  - mesh.dim ∈ {1, 2}
  - equation ∈ {Advection, Euler1D, Euler2D}
  - reconstruction ∈ {first_order, minmod_tvd_1d, mlp_u, t_mlp_u, …}
  - flux ∈ {upwind, llf, hllc_1d, …}
  - integrator ∈ {forward_euler, ssp_rk2, ssp_rk3}

The same skeleton will accommodate the user's T-MLP-u once the
reconstruction implementation lands.

Free parameters: 0 (CFL number is the only knob; reconstruction /
flux / integrator are *named* methods with fixed formulae).
"""
from __future__ import annotations
import os
import time
import numpy as np

from reconstruction import get_reconstruction, Reconstruction
from flux import get_flux
from time_integrator import get_integrator
from boundary import apply_patch_bcs


__all__ = ['solve']


_EPS = 1.0e-30


def _format_duration(seconds):
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _progress_bar(fraction, width):
    width = max(0, int(width))
    if width <= 0:
        return ""
    fraction = min(1.0, max(0.0, float(fraction)))
    filled = int(round(fraction * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"

try:
    from numba import njit, prange, set_num_threads, get_num_threads
    _NUMBA_AVAILABLE = True
    _thread_env = (
        os.environ.get('TMLPU_SOLVER_THREADS')
        or os.environ.get('NUMBA_NUM_THREADS'))
    _default_threads = os.cpu_count() or 1
    _configured_threads = max(1, min(_default_threads, int(
        _thread_env or _default_threads)))
    set_num_threads(_configured_threads)
except Exception:  # pragma: no cover - numba is optional.
    njit = None
    prange = range
    get_num_threads = None
    _NUMBA_AVAILABLE = False
    _configured_threads = 1


if _NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def _accumulate_cell_residual_kernel(AF, cell_faces, cell_signs,
                                         inv_vol, out):
        nvar = AF.shape[0]
        n_cells = cell_faces.shape[0]
        n_face_per_cell = cell_faces.shape[1]
        for c in prange(n_cells):
            for v in range(nvar):
                acc = 0.0
                for j in range(n_face_per_cell):
                    f = cell_faces[c, j]
                    if f >= 0:
                        acc += cell_signs[c, j] * AF[v, f]
                out[v, c] = acc * inv_vol[c]


    @njit(parallel=True, cache=True)
    def _euler2d_global_max_wave_speed_kernel(U, gamma):
        n_cells = U.shape[1]
        wmax = 0.0
        for c in prange(n_cells):
            rho = U[0, c]
            if rho < _EPS:
                rho = _EPS
            u = U[1, c] / rho
            v = U[2, c] / rho
            E = U[3, c] / rho
            p = (gamma - 1.0) * rho * (E - 0.5 * (u * u + v * v))
            csq = gamma * p / rho
            if csq < _EPS:
                csq = _EPS
            wave = np.sqrt(u * u + v * v) + np.sqrt(csq)
            wmax = max(wmax, wave)
        return wmax


def _resolve(name_or_obj, getter):
    if isinstance(name_or_obj, str):
        return getter(name_or_obj)
    return name_or_obj


def _resolve_solver_threads():
    requested = (
        os.environ.get('TMLPU_SOLVER_THREADS')
        or os.environ.get('NUMBA_NUM_THREADS'))
    default_threads = os.cpu_count() or 1
    return max(1, min(default_threads, int(requested or default_threads)))


def solve(mesh, eq, U0, *,
          reconstruction='first_order',
          flux='llf',
          integrator='ssp_rk2',
          bc=None,
          cfl: float = 0.5,
          dt_fixed=None,
          n_face_quad: int = 1,
          face_velocity_mode: str = 'analytic',
          t_end: float,
          max_steps: int = 200_000,
          history_every: int = 0):
    """Time-march U from U0 to t_end and return the final state.

    `n_face_quad` controls the face-integration order:
        1 — midpoint rule (default; O(h²) face quadrature).
        2 — two-point Gauss-Legendre quadrature on the edge (O(h⁴)
            face quadrature, required to maintain ≥3rd-order overall when
            paired with k=2 polynomial reconstruction and SSP-RK3 in time).
    """

    if _NUMBA_AVAILABLE:
        requested_threads = _resolve_solver_threads()
        if get_num_threads() != requested_threads:
            set_num_threads(requested_threads)

    # Resolve plug-ins
    if isinstance(reconstruction, str):
        recon = get_reconstruction(reconstruction)
    else:
        recon: Reconstruction = reconstruction
    flux_fn = _resolve(flux, get_flux)
    flux_hcorr_mode = getattr(flux_fn, 'h_correction_mode', '')
    flux_needs_hcorr = (
        getattr(flux_fn, 'h_correction', False)
        and isinstance(flux_hcorr_mode, str))
    hll_flux_fn = get_flux('hlle') if flux_needs_hcorr else None
    step    = _resolve(integrator, get_integrator)
    bc_spec = bc or {}
    exact_reflective_wall_flux = (
        os.environ.get('TMLPU_EULER_EXACT_REFLECTIVE_WALL_FLUX', '0')
        .lower() in ('1', 'true', 'yes', 'on')
        and getattr(eq, '__class__', type(eq)).__name__ == 'Euler2D'
        and eq.nvar == 4)
    exact_reflective_wall_pressure = os.environ.get(
        'TMLPU_EULER_EXACT_REFLECTIVE_WALL_PRESSURE', 'face'
    ).strip().lower()

    U = np.array(U0, dtype=float, copy=True)
    t = 0.0
    history = []
    finite_check_every = max(
        1, int(os.environ.get('TMLPU_FINITE_CHECK_EVERY', '1')))
    progress_every = max(
        0, int(os.environ.get('TMLPU_PROGRESS_EVERY', '0')))
    progress_bar_width = max(
        0, int(os.environ.get('TMLPU_PROGRESS_BAR_WIDTH', '24')))
    progress_label = os.environ.get('TMLPU_PROGRESS_LABEL', '').strip()
    max_wall_seconds = float(os.environ.get('TMLPU_MAX_WALL_SECONDS', '0.0'))
    wall_t0 = time.time()
    stopped_by_wall = False
    abort_on_dt_drop = bool(int(os.environ.get(
        'TMLPU_ABORT_ON_DT_DROP', '1')))
    abort_on_nonphysical = bool(int(os.environ.get(
        'TMLPU_ABORT_ON_NONPHYSICAL', '1')))
    rho_floor_abort = float(os.environ.get('TMLPU_ABORT_RHO_MIN', '0.0'))
    p_floor_abort = float(os.environ.get('TMLPU_ABORT_P_MIN', '0.0'))
    dt_drop_factor = float(os.environ.get(
        'TMLPU_DT_DROP_FACTOR', '0.15'))
    dt_drop_factor = min(0.99, max(0.01, dt_drop_factor))
    dt_abort_min_steps = max(0, int(os.environ.get(
        'TMLPU_DT_ABORT_MIN_STEPS', '100')))
    if history_every > 0:
        history.append((t, U.copy()))

    inv_vol = 1.0 / mesh.cell_volumes        # (Ncells,)
    n_faces = mesh.n_faces
    owner = mesh.face_owner
    nei   = mesh.face_neighbour
    areas = mesh.face_areas
    normals = mesh.face_normals

    reflective_wall_faces = None
    if exact_reflective_wall_flux:
        boundary_faces = nei < 0
        reflective_wall_faces = np.zeros(mesh.n_faces, dtype=bool)
        face_bc_tag = getattr(mesh, 'face_bc_tag', None)
        bc_patches = tuple(getattr(mesh, 'bc_patches', ()) or ())
        if face_bc_tag is not None and bc_patches:
            for patch_id, patch_name in enumerate(bc_patches, start=1):
                bc_obj = bc_spec.get(patch_name)
                if getattr(bc_obj, 'kind', None) != 'reflective':
                    continue
                reflective_wall_faces |= (
                    boundary_faces & (face_bc_tag == patch_id))
        else:
            for bc_obj in bc_spec.values():
                if getattr(bc_obj, 'kind', None) == 'reflective':
                    reflective_wall_faces |= boundary_faces
                    break
        if not np.any(reflective_wall_faces):
            reflective_wall_faces = None

    # Pre-compute scatter masks once (mesh-only) — used every rhs call.
    nei_mask = nei >= 0
    nei_int = nei[nei_mask]                                # (Nint,)
    if not hasattr(mesh, '_cell_face_residual_cache'):
        mesh._cell_face_residual_cache = _cell_face_residual_arrays(mesh)
    cell_faces_residual, cell_signs_residual = (
        mesh._cell_face_residual_cache)

    # Pre-compute per-cell characteristic length once (mesh-only).
    # Stored on mesh so repeated solve() calls share the same array.
    if not hasattr(mesh, '_cell_length_scale_cache'):
        mesh._cell_length_scale_cache = _cell_length_scale(mesh)
    h_cell_min = float(np.min(mesh._cell_length_scale_cache))

    # Pre-compute Gauss-quadrature points on each face (mesh-only).
    if n_face_quad == 1:
        gqs = mesh.face_centers[:, None, :]               # (Nf, 1, 2)
        gw = np.array([1.0])
    elif n_face_quad == 2:
        gqs, gw = _gauss_2pt_face(mesh)                   # (Nf, 2, 2), (2,)
    elif n_face_quad == 3:
        gqs, gw = _gauss_3pt_face(mesh)                   # (Nf, 3, 2), (3,)
    else:
        raise NotImplementedError(
            f"n_face_quad={n_face_quad}: only 1, 2, or 3 supported.")

    # Pre-compute central-averaged face velocity u_f = ½(a(x_o)+a(x_n)) once.
    # Sampled at owner / neighbour cell centres; boundary faces fall back to
    # owner cell centre.  Used uniformly across all Gauss-quadrature points
    # of each face (single value per face), in contrast with the default
    # `analytic` mode which samples a(x_GP) exactly at every GP.
    face_velocity_central = None
    if face_velocity_mode == 'central_avg':
        if not getattr(eq, 'is_variable_velocity', False):
            face_velocity_central = None  # constant velocity — no benefit
        else:
            cc = mesh.cell_centers
            owner_cc = cc[owner]                          # (Nf, 2)
            nei_safe = np.where(nei >= 0, nei, owner)
            nei_cc = cc[nei_safe]                         # (Nf, 2)
            v_o = eq.velocity_at(owner_cc)                # (Nf, 2)
            v_n = eq.velocity_at(nei_cc)                  # (Nf, 2)
            face_velocity_central = 0.5 * (v_o + v_n)     # (Nf, 2)
    elif face_velocity_mode != 'analytic':
        raise ValueError(
            f"face_velocity_mode must be 'analytic' or 'central_avg', "
            f"got {face_velocity_mode!r}")

    # Pre-allocated reusable buffers (closure-shared across rhs calls).
    _F_face_buf = np.empty((eq.nvar, mesh.n_faces), dtype=float)
    _AF_buf = np.empty((eq.nvar, mesh.n_faces), dtype=float)
    _dUdt_buf = np.empty_like(U)
    _W_buf = np.empty_like(U)
    _use_euler2d_fast_prim = (
        eq.__class__.__name__ == 'Euler2D' and eq.nvar == 4)
    current_time = [0.0]

    def rhs(U_state):
        """∂t U = − (1/V) Σ_f (∫ F·n dl) — Gauss-quadrature on each face."""
        if _use_euler2d_fast_prim:
            W = _fill_euler2d_primitive(eq, U_state, _W_buf)
        else:
            W = eq.cons_to_prim(U_state)
        F_face = _F_face_buf
        F_face.fill(0.0)
        for k in range(n_face_quad):
            GP_k = gqs[:, k, :]                            # (Nf, 2)
            W_L, W_R = recon.reconstruct(mesh, W, eq, eval_points=GP_k)
            W_L, W_R = apply_patch_bcs(
                mesh, eq, W_L, W_R, bc_spec,
                points=GP_k, time=current_time[0])
            if face_velocity_central is not None:
                F_k = flux_fn(eq, W_L, W_R, normals, points=GP_k,
                              face_velocity=face_velocity_central)
            else:
                F_k = flux_fn(eq, W_L, W_R, normals, points=GP_k)
            # Reflective wall faces: the configured upwind `flux_fn` is used
            # directly on the mirrored ghost state — no HLLE wall override.
            if flux_needs_hcorr and eq.__class__.__name__ == 'Euler2D':
                F_hll = hll_flux_fn(eq, W_L, W_R, normals, points=GP_k)
                shock_weight = _neighbor_hll_blend_weight(
                    mesh, eq, W_L, W_R, normals, flux_hcorr_mode)
                F_k = _apply_solver_h_correction(
                    F_k, F_hll, normals, shock_weight, flux_hcorr_mode)
            if reflective_wall_faces is not None:
                faces = reflective_wall_faces
                if exact_reflective_wall_pressure in ('cell', 'owner',
                                                      'cell_average'):
                    p_wall = np.maximum(W[3, owner[faces]], _EPS)
                elif exact_reflective_wall_pressure in ('riemann', 'star',
                                                        'shock'):
                    # Slip-wall flux with a 1-D reflective Riemann pressure.
                    # For an owner state mirrored across the wall, the normal
                    # velocity at the wall is zero.  If the reconstructed state
                    # is moving into the wall, the wall pressure follows the
                    # two-shock Rankine-Hugoniot relation
                    #   q = (p* - p) sqrt(A / (p* + B)),
                    # with q = max(u_n,out, 0).  For outward/parallel motion
                    # this reduces to the local pressure.  This is a boundary
                    # condition consistency correction, not a local shock patch.
                    rho_f = np.maximum(W_L[0, faces], _EPS)
                    p_f = np.maximum(W_L[3, faces], _EPS)
                    un_out = (
                        W_L[1, faces] * normals[faces, 0]
                        + W_L[2, faces] * normals[faces, 1]
                    )
                    q_wall = np.maximum(un_out, 0.0)
                    gamma = float(getattr(eq, 'gamma', 1.4))
                    A = 2.0 / ((gamma + 1.0) * rho_f)
                    B = ((gamma - 1.0) / (gamma + 1.0)) * p_f
                    bq = 2.0 * A * p_f + q_wall * q_wall
                    cq = A * p_f * p_f - q_wall * q_wall * B
                    disc = np.maximum(bq * bq - 4.0 * A * cq, 0.0)
                    p_star = (bq + np.sqrt(disc)) / np.maximum(
                        2.0 * A, _EPS)
                    p_wall = np.where(q_wall > 0.0,
                                      np.maximum(p_star, p_f),
                                      p_f)
                else:
                    p_wall = np.maximum(W_L[3, faces], _EPS)
                F_k[0, faces] = 0.0
                F_k[1, faces] = p_wall * normals[faces, 0]
                F_k[2, faces] = p_wall * normals[faces, 1]
                F_k[3, faces] = 0.0
            if n_face_quad == 1:
                F_face[:] = F_k
            else:
                F_face += gw[k] * F_k

        np.multiply(F_face, areas, out=_AF_buf)             # (nvar, Nf)
        if _NUMBA_AVAILABLE:
            _accumulate_cell_residual_kernel(
                _AF_buf, cell_faces_residual, cell_signs_residual,
                inv_vol, _dUdt_buf)
        else:
            _dUdt_buf.fill(0.0)
            for v in range(eq.nvar):
                np.add.at(_dUdt_buf[v], owner, -_AF_buf[v])
                np.add.at(_dUdt_buf[v], nei_int, _AF_buf[v, nei_mask])
            np.multiply(_dUdt_buf, inv_vol, out=_dUdt_buf)
        return _dUdt_buf

    n_completed = 0
    info_last = {}
    dt_best = 0.0
    kernel_step_times = [] if os.environ.get(
        'TMLPU_KERNEL_TIMING', '0').lower() in (
            '1', 'true', 'yes', 'on') else None
    for n in range(max_steps):
        if t >= t_end:
            break
        if max_wall_seconds > 0.0 and n_completed > 0:
            if time.time() - wall_t0 >= max_wall_seconds:
                stopped_by_wall = True
                break

        # Time-step
        wmax = None
        if dt_fixed is not None:
            dt = float(dt_fixed)
        else:
            # Global CFL with max wave speed
            wmax = float(_global_max_wave_speed(mesh, eq, U))
            if not np.isfinite(wmax) or wmax <= 0.0:
                raise FloatingPointError(f"non-positive max wave speed at t={t}")
            # Use the smallest "characteristic length" of any cell as a
            # geometric scale (cached above — mesh-only quantity).
            dt = cfl * h_cell_min / wmax
            if (abort_on_dt_drop and dt_best > 0.0
                    and n_completed >= dt_abort_min_steps
                    and dt < dt_drop_factor * dt_best):
                diag = _euler2d_runtime_diagnostics(eq, U, wmax, mesh=mesh)
                raise FloatingPointError(
                    "CFL dt collapse detected at "
                    f"step={n_completed}, t={t:.8g}: "
                    f"dt={dt:.6e} < {dt_drop_factor:.3g}*"
                    f"best_dt={dt_best:.6e}{diag}")
            if dt > dt_best:
                dt_best = dt
        if t + dt > t_end:
            dt = t_end - t
        if dt <= 0:
            break

        current_time[0] = t
        if kernel_step_times is None:
            U = step(U, dt, rhs)
        else:
            _kernel_t0 = time.perf_counter()
            U = step(U, dt, rhs)
            kernel_step_times.append(time.perf_counter() - _kernel_t0)
        t += dt
        n_completed = n + 1
        if history_every > 0 and (n_completed % history_every == 0):
            history.append((t, U.copy()))
        if n_completed % finite_check_every == 0 and not np.isfinite(U).all():
            raise FloatingPointError(f"NaN at step {n_completed}, t={t}")
        if (abort_on_nonphysical and _use_euler2d_fast_prim
                and n_completed % finite_check_every == 0):
            rho_min, p_min = _euler2d_min_density_pressure(eq, U)
            if rho_min <= rho_floor_abort or p_min <= p_floor_abort:
                raise FloatingPointError(
                    "nonphysical Euler state at "
                    f"step={n_completed}, t={t:.8g}: "
                    f"rho_min={rho_min:.6g}, p_min={p_min:.6g}")
        if progress_every and n_completed % progress_every == 0:
            diag = _euler2d_runtime_diagnostics(eq, U, wmax, mesh=mesh)
            frac = 1.0 if t_end <= 0.0 else min(1.0, max(0.0, t / t_end))
            elapsed = time.time() - wall_t0
            eta = 0.0
            if 0.0 < frac < 1.0:
                eta = elapsed * (1.0 - frac) / frac
            label = f" {progress_label}" if progress_label else ""
            bar = _progress_bar(frac, progress_bar_width)
            if bar:
                bar = " " + bar
            print(
                f"[solve{label}]{bar} {100.0 * frac:5.1f}% "
                f"step={n_completed} t={t:.8g}/{t_end:.8g} "
                f"dt={dt:.3e} elapsed={_format_duration(elapsed)} "
                f"eta={_format_duration(eta)}{diag}",
                flush=True)
    if not np.isfinite(U).all():
        raise FloatingPointError(f"NaN at final step {n_completed}, t={t}")

    return dict(U_final=U,
                t=t,
                n_steps=n_completed,
                history=history,
                info_last=info_last,
                stopped_by_wall=stopped_by_wall,
                kernel_step_times=kernel_step_times)


def _neighbor_hll_blend_weight(mesh, eq, W_L, W_R, normals, mode):
    """Multidimensional shock-band HLL blend strength.

    A pressure-compression sensor is first evaluated on every face, then
    expanded through the cells adjacent to that face.  This follows the
    multidimensional dissipation idea used to suppress odd-even shock
    decoupling: a face inside a shock band inherits the strongest shock
    sensor in either neighboring cell instead of relying only on its own
    one-dimensional jump.
    """
    p_L = np.maximum(W_L[3], _EPS)
    p_R = np.maximum(W_R[3], _EPS)
    rho_L = np.maximum(W_L[0], _EPS)
    rho_R = np.maximum(W_R[0], _EPS)
    gamma = float(getattr(eq, 'gamma', 1.4))
    c_L = np.sqrt(np.maximum(gamma * p_L / rho_L, _EPS))
    c_R = np.sqrt(np.maximum(gamma * p_R / rho_R, _EPS))
    un_L = W_L[1] * normals[:, 0] + W_L[2] * normals[:, 1]
    un_R = W_R[1] * normals[:, 0] + W_R[2] * normals[:, 1]
    pressure_jump = np.abs(p_R - p_L) / np.maximum(p_R + p_L, _EPS)
    compression = np.maximum(0.0, un_L - un_R) / np.maximum(c_L + c_R, _EPS)
    jump_sensor = np.clip((pressure_jump - 0.05) / 0.35, 0.0, 1.0)
    compression_sensor = np.clip(4.0 * compression, 0.0, 1.0)
    if 'pressureonly' in mode:
        face_sensor = jump_sensor
    else:
        face_sensor = np.sqrt(jump_sensor * compression_sensor)

    if 'pressure_guard' in mode:
        # Preserve the carbuncle damping on genuine acoustic shocks while
        # reducing it on nearly isobaric shear/contact faces.  The pressure
        # jump is converted to an equivalent acoustic velocity jump through
        # the local impedance, so this remains scale-free and face-local.
        tx = -normals[:, 1]
        ty = normals[:, 0]
        ut_L = W_L[1] * tx + W_L[2] * ty
        ut_R = W_R[1] * tx + W_R[2] * ty
        dn = un_R - un_L
        dtan = ut_R - ut_L
        rho_bar = 0.5 * (rho_L + rho_R)
        c_bar = 0.5 * (c_L + c_R)
        dp_vel = np.abs(p_R - p_L) / np.maximum(rho_bar * c_bar, _EPS)
        normality = (dn * dn) / np.maximum(dn * dn + dtan * dtan, _EPS)
        acousticity = ((dp_vel * dp_vel)
                       / np.maximum(dp_vel * dp_vel + dtan * dtan, _EPS))
        face_sensor *= np.maximum(normality, acousticity)
    elif 'shear_guard' in mode:
        # Carbuncle/odd-even damping should target shock-normal decoupling, not
        # the slip-line shear that creates the physical post-shock roll-up.
        # This invariant jump-ratio keeps the same global formula on every
        # face: the H-correction is strong only when the velocity jump is
        # predominantly normal to the face.
        tx = -normals[:, 1]
        ty = normals[:, 0]
        ut_L = W_L[1] * tx + W_L[2] * ty
        ut_R = W_R[1] * tx + W_R[2] * ty
        dn = un_R - un_L
        dtan = ut_R - ut_L
        normality = (dn * dn) / np.maximum(dn * dn + dtan * dtan, _EPS)
        face_sensor *= normality

    if 'face_local' in mode:
        if 'ultrasoft' in mode:
            return face_sensor * face_sensor * face_sensor
        if 'geomean' in mode:
            return face_sensor * np.sqrt(np.maximum(face_sensor, 0.0))
        if 'soft' in mode:
            return face_sensor * face_sensor
        return face_sensor

    owner = mesh.face_owner
    nei = mesh.face_neighbour
    if 'directional' in mode:
        cell_sensor = np.zeros(mesh.n_cells, dtype=float)
        cell_gx = np.zeros(mesh.n_cells, dtype=float)
        cell_gy = np.zeros(mesh.n_cells, dtype=float)
        grad_weight = face_sensor * pressure_jump
        np.maximum.at(cell_sensor, owner, face_sensor)
        np.add.at(cell_gx, owner, grad_weight * normals[:, 0])
        np.add.at(cell_gy, owner, grad_weight * normals[:, 1])
        int_mask = nei >= 0
        if np.any(int_mask):
            nei_int = nei[int_mask]
            np.maximum.at(cell_sensor, nei_int, face_sensor[int_mask])
            np.add.at(cell_gx, nei_int,
                      -grad_weight[int_mask] * normals[int_mask, 0])
            np.add.at(cell_gy, nei_int,
                      -grad_weight[int_mask] * normals[int_mask, 1])
        gnorm = np.sqrt(cell_gx * cell_gx + cell_gy * cell_gy)
        align_o = np.abs(normals[:, 0] * cell_gx[owner]
                         + normals[:, 1] * cell_gy[owner])
        align_o /= np.maximum(gnorm[owner], _EPS)
        nei_safe = np.where(int_mask, nei, owner)
        align_n = np.abs(normals[:, 0] * cell_gx[nei_safe]
                         + normals[:, 1] * cell_gy[nei_safe])
        align_n /= np.maximum(gnorm[nei_safe], _EPS)
        band_sensor = np.maximum(
            face_sensor,
            np.maximum(cell_sensor[owner] * align_o,
                       cell_sensor[nei_safe] * align_n))
        if 'ultrasoft' in mode:
            return band_sensor * band_sensor * band_sensor
        if 'geomean' in mode:
            return band_sensor * np.sqrt(np.maximum(band_sensor, 0.0))
        if 'soft' in mode:
            return band_sensor * band_sensor
        return band_sensor

    cell_sensor = np.zeros(mesh.n_cells, dtype=float)
    np.maximum.at(cell_sensor, owner, face_sensor)
    int_mask = nei >= 0
    np.maximum.at(cell_sensor, nei[int_mask], face_sensor[int_mask])
    nei_safe = np.where(int_mask, nei, owner)
    band_sensor = np.maximum(
        face_sensor,
        np.maximum(cell_sensor[owner], cell_sensor[nei_safe]))

    if 'ultrasoft' in mode:
        return band_sensor * band_sensor * band_sensor
    if 'geomean' in mode:
        return band_sensor * np.sqrt(np.maximum(band_sensor, 0.0))
    if 'soft' in mode:
        return band_sensor * band_sensor
    return band_sensor


def _apply_solver_h_correction(F, F_hll, normals, weight, mode):
    """Apply a channel-selective HLLE shock-band correction."""
    if 'hll_blend' in mode:
        return (1.0 - weight)[None, :] * F + weight[None, :] * F_hll

    nx = normals[:, 0]
    ny = normals[:, 1]
    tx = -ny
    ty = nx
    out = F.copy()
    normal_f = F[1] * nx + F[2] * ny
    tangent_f = F[1] * tx + F[2] * ty
    normal_h = F_hll[1] * nx + F_hll[2] * ny
    tangent_h = F_hll[1] * tx + F_hll[2] * ty

    if mode.startswith('mass_normal_energy'):
        out[0] = (1.0 - weight) * F[0] + weight * F_hll[0]
        normal_blend = (1.0 - weight) * normal_f + weight * normal_h
        out[1] = normal_blend * nx + tangent_f * tx
        out[2] = normal_blend * ny + tangent_f * ty
        out[3] = (1.0 - weight) * F[3] + weight * F_hll[3]
        return out

    if (mode.startswith('normal_energy')
            or mode in ('pressure_normal_momentum',
                        'pressureonly_normal_momentum',
                        'pressureonly_soft_normal_momentum')):
        normal_blend = (1.0 - weight) * normal_f + weight * normal_h
        out[1] = normal_blend * nx + tangent_f * tx
        out[2] = normal_blend * ny + tangent_f * ty
        if mode.startswith('normal_energy'):
            out[3] = (1.0 - weight) * F[3] + weight * F_hll[3]
        return out

    if mode in ('pressure_momentum',):
        out[1] = (1.0 - weight) * F[1] + weight * F_hll[1]
        out[2] = (1.0 - weight) * F[2] + weight * F_hll[2]
        return out

    if mode.startswith('mass_transverse'):
        out[0] = (1.0 - weight) * F[0] + weight * F_hll[0]
        tangent_blend = (1.0 - weight) * tangent_f + weight * tangent_h
        out[1] = normal_f * nx + tangent_blend * tx
        out[2] = normal_f * ny + tangent_blend * ty
        if 'energy' in mode:
            out[3] = (1.0 - weight) * F[3] + weight * F_hll[3]
        return out

    if mode.startswith('mass_energy'):
        out[0] = (1.0 - weight) * F[0] + weight * F_hll[0]
        out[3] = (1.0 - weight) * F[3] + weight * F_hll[3]
        return out

    if mode.startswith('mass_'):
        out[0] = (1.0 - weight) * F[0] + weight * F_hll[0]
        return out

    return out


def _fill_euler2d_primitive(eq, U, out):
    """Fill primitive variables for Euler2D without per-RHS stack allocation."""
    rho = np.maximum(U[0], _EPS)
    out[0] = rho
    np.divide(U[1], rho, out=out[1])
    np.divide(U[2], rho, out=out[2])
    np.divide(U[3], rho, out=out[3])
    out[3] = ((float(eq.gamma) - 1.0) * rho
              * (out[3] - 0.5 * (out[1] * out[1] + out[2] * out[2])))
    return out


def _euler2d_runtime_diagnostics(eq, U, wmax=None, mesh=None):
    if eq.__class__.__name__ != 'Euler2D' or U.shape[0] != 4:
        return ''
    rho_d = np.maximum(U[0], _EPS)
    u_d = U[1] / rho_d
    v_d = U[2] / rho_d
    E_d = U[3] / rho_d
    p_d = ((float(eq.gamma) - 1.0) * rho_d
           * (E_d - 0.5 * (u_d * u_d + v_d * v_d)))
    vel_d = np.sqrt(u_d * u_d + v_d * v_d)
    vel_i = int(np.argmax(vel_d))
    prefix = '' if wmax is None else f" wmax={float(wmax):.6g}"
    loc = ''
    if mesh is not None:
        try:
            cx = float(mesh.cell_centers[vel_i, 0])
            cy = float(mesh.cell_centers[vel_i, 1])
            loc = f"@({cx:.3f},{cy:.3f})"
        except Exception:
            loc = ''
    return (f"{prefix} rho_min={float(np.min(rho_d)):.6g}"
            f" p_min={float(np.min(p_d)):.6g}"
            f" vel_max={float(vel_d[vel_i]):.6g}"
            f" vel_max_cell={vel_i}{loc}")


def _euler2d_min_density_pressure(eq, U):
    rho = U[0]
    rho_safe = np.maximum(rho, _EPS)
    u = U[1] / rho_safe
    v = U[2] / rho_safe
    E = U[3] / rho_safe
    p = ((float(eq.gamma) - 1.0) * rho_safe
         * (E - 0.5 * (u * u + v * v)))
    return float(np.min(rho)), float(np.min(p))


def _cell_face_residual_arrays(mesh):
    """Build cell-local face lists for race-free threaded residual assembly."""
    cf = mesh.cell_faces
    n_cells = mesh.n_cells
    if not isinstance(cf, np.ndarray):
        max_f = max((len(faces) for faces in cf), default=1)
        max_f = max(max_f, 1)
        cell_faces = np.full((n_cells, max_f), -1, dtype=np.int64)
        for c, faces in enumerate(cf):
            cell_faces[c, :len(faces)] = faces
    elif cf.ndim == 1:
        cell_faces = cf.reshape((n_cells, 1)).astype(np.int64, copy=False)
    else:
        cell_faces = cf.astype(np.int64, copy=False)

    owner = mesh.face_owner
    nei = mesh.face_neighbour
    cell_signs = np.zeros(cell_faces.shape, dtype=np.float64)
    for c in range(n_cells):
        for j in range(cell_faces.shape[1]):
            f = int(cell_faces[c, j])
            if f < 0:
                continue
            if owner[f] == c:
                cell_signs[c, j] = -1.0
            elif nei[f] == c:
                cell_signs[c, j] = 1.0
    return cell_faces, cell_signs


def _cell_length_scale(mesh):
    """A characteristic length per cell — used for CFL.

    For 1D structured: equals dx.
    For 2D Cartesian:  V/max(face_area) = dx·dy / max(dx, dy) = min(dx, dy).
    For unstructured:  V / max_face_area is a reasonable inradius proxy.

    Vectorised path: build a padded (N, max_faces_per_cell) array and
    take a row-wise max over face_areas, then divide.  Falls back to V
    where max_area ≤ 0.  Returns identical values to the previous
    Python-loop implementation.
    """
    cf = mesh.cell_faces
    n_cells = mesh.n_cells
    if not isinstance(cf, np.ndarray):
        # cell_faces is typically a list-of-lists.  Pad and vectorise.
        max_f = max((len(faces) for faces in cf), default=1)
        max_f = max(max_f, 1)
        cf_padded = np.full((n_cells, max_f), -1, dtype=int)
        for i, faces in enumerate(cf):
            cf_padded[i, :len(faces)] = faces
    else:
        cf_padded = cf
        max_f = cf_padded.shape[1] if cf_padded.ndim > 1 else 1

    valid = cf_padded >= 0
    safe_idx = np.where(valid, cf_padded, 0)
    face_a = mesh.face_areas[safe_idx]                       # (N, max_f)
    face_a = np.where(valid, face_a, -np.inf)
    max_area = face_a.max(axis=1)
    vols = mesh.cell_volumes
    # Where no positive face: fall back to vol itself (matches old code).
    h = np.where(max_area > 0.0, vols / np.where(max_area > 0.0, max_area, 1.0),
                 vols)
    return h


def _gauss_2pt_face(mesh):
    """Two-point Gauss–Legendre quadrature on each face.

    For 2D meshes, the face is a line segment with length L, normal n,
    and tangent t = ⟂n.  The Gauss points are::

        GP_k = face_centre + ξ_k · (L/2) · t,    ξ_k = ∓1/√3
        weights w_k = 1/2 (sum to 1; absolute scaling carried by face_areas)

    For 1D meshes the face is a point — degenerate to midpoint rule.

    Returns
    -------
    GPs : (n_faces, 2, dim) array
    weights : (2,) array  (each entry = 0.5)
    """
    if mesh.dim == 1:
        return mesh.face_centers[:, None, :], np.array([1.0])
    nx = mesh.face_normals[:, 0]
    ny = mesh.face_normals[:, 1]
    tx = -ny;  ty = nx                          # 90°-rotated normal = tangent
    L = mesh.face_areas
    shift = L / (2.0 * np.sqrt(3.0))
    GPs = np.empty((mesh.n_faces, 2, 2), dtype=float)
    fc0 = mesh.face_centers[:, 0]
    fc1 = mesh.face_centers[:, 1]
    GPs[:, 0, 0] = fc0 - shift * tx
    GPs[:, 0, 1] = fc1 - shift * ty
    GPs[:, 1, 0] = fc0 + shift * tx
    GPs[:, 1, 1] = fc1 + shift * ty
    return GPs, np.array([0.5, 0.5])


def _gauss_3pt_face(mesh):
    """Three-point Gauss-Legendre quadrature on a 2D edge.
    Reference points ξ ∈ {-√(3/5), 0, +√(3/5)}, weights {5/9, 8/9, 5/9}.
    """
    if mesh.dim == 1:
        return mesh.face_centers[:, None, :], np.array([1.0])
    nx = mesh.face_normals[:, 0]
    ny = mesh.face_normals[:, 1]
    tx = -ny;  ty = nx
    L = mesh.face_areas
    half = L * 0.5
    xi = np.sqrt(3.0 / 5.0)
    GPs = np.empty((mesh.n_faces, 3, 2), dtype=float)
    fc0 = mesh.face_centers[:, 0]
    fc1 = mesh.face_centers[:, 1]
    GPs[:, 0, 0] = fc0 - xi * half * tx
    GPs[:, 0, 1] = fc1 - xi * half * ty
    GPs[:, 1, 0] = fc0
    GPs[:, 1, 1] = fc1
    GPs[:, 2, 0] = fc0 + xi * half * tx
    GPs[:, 2, 1] = fc1 + xi * half * ty
    weights = np.array([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0])
    # 5/9·½ = 5/18, 8/9·½ = 8/18, sum = 18/18 = 1
    return GPs, weights


def _global_max_wave_speed(mesh, eq, U):
    """Largest |λ_max| over all cells (per-cell over the equation)."""
    # In 1D the normal is a single direction; for 2D we evaluate against
    # both axes and take the max.
    if mesh.dim == 1:
        n_dummy = np.array([[1.0]])
        try:
            return float(np.max(eq.max_wave_speed(U, n_dummy,
                                                  points=mesh.cell_centers)))
        except TypeError:
            return float(np.max(eq.max_wave_speed(U, n_dummy)))
    if eq.__class__.__name__ == 'Euler2D' and U.shape[0] == 4:
        if _NUMBA_AVAILABLE:
            return float(_euler2d_global_max_wave_speed_kernel(
                U, float(eq.gamma)))
        rho = np.maximum(U[0], _EPS)
        u = U[1] / rho
        v = U[2] / rho
        E = U[3] / rho
        p = ((float(eq.gamma) - 1.0) * rho
             * (E - 0.5 * (u * u + v * v)))
        c = np.sqrt(np.maximum(float(eq.gamma) * p / rho, _EPS))
        return float(np.max(np.sqrt(u * u + v * v) + c))
    # 2D
    n_x = np.array([[1.0, 0.0]])
    n_y = np.array([[0.0, 1.0]])
    pts = mesh.cell_centers
    try:
        lam_x = eq.max_wave_speed(U, n_x, points=pts)
        lam_y = eq.max_wave_speed(U, n_y, points=pts)
    except TypeError:
        lam_x = eq.max_wave_speed(U, n_x)
        lam_y = eq.max_wave_speed(U, n_y)
    return float(np.max(np.maximum(lam_x, lam_y)))
