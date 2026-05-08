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
import numpy as np

from reconstruction import get_reconstruction, Reconstruction
from flux import get_flux
from time_integrator import get_integrator
from boundary import apply_patch_bcs


__all__ = ['solve']


def _resolve(name_or_obj, getter):
    if isinstance(name_or_obj, str):
        return getter(name_or_obj)
    return name_or_obj


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

    # Resolve plug-ins
    if isinstance(reconstruction, str):
        recon = get_reconstruction(reconstruction)
    else:
        recon: Reconstruction = reconstruction
    flux_fn = _resolve(flux, get_flux)
    step    = _resolve(integrator, get_integrator)
    bc_spec = bc or {}

    U = np.array(U0, dtype=float, copy=True)
    t = 0.0
    history = []
    if history_every > 0:
        history.append((t, U.copy()))

    inv_vol = 1.0 / mesh.cell_volumes        # (Ncells,)
    n_faces = mesh.n_faces
    owner = mesh.face_owner
    nei   = mesh.face_neighbour
    areas = mesh.face_areas
    normals = mesh.face_normals

    # Pre-compute scatter masks once (mesh-only) — used every rhs call.
    nei_mask = nei >= 0
    nei_int = nei[nei_mask]                                # (Nint,)

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

    def rhs(U_state):
        """∂t U = − (1/V) Σ_f (∫ F·n dl) — Gauss-quadrature on each face."""
        W = eq.cons_to_prim(U_state)
        F_face = _F_face_buf
        F_face.fill(0.0)
        for k in range(n_face_quad):
            GP_k = gqs[:, k, :]                            # (Nf, 2)
            W_L, W_R = recon.reconstruct(mesh, W, eq, eval_points=GP_k)
            W_L, W_R = apply_patch_bcs(mesh, eq, W_L, W_R, bc_spec)
            if face_velocity_central is not None:
                F_k = flux_fn(eq, W_L, W_R, normals, points=GP_k,
                              face_velocity=face_velocity_central)
            else:
                F_k = flux_fn(eq, W_L, W_R, normals, points=GP_k)
            if n_face_quad == 1:
                F_face[:] = F_k
            else:
                F_face += gw[k] * F_k

        dUdt = np.zeros_like(U_state)
        AF = F_face * areas                                # (nvar, Nf)
        for v in range(eq.nvar):
            np.add.at(dUdt[v], owner, -AF[v])
            np.add.at(dUdt[v], nei_int, AF[v, nei_mask])
        return dUdt * inv_vol

    n_completed = 0
    info_last = {}
    for n in range(max_steps):
        if t >= t_end:
            break

        # Time-step
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
        if t + dt > t_end:
            dt = t_end - t
        if dt <= 0:
            break

        U = step(U, dt, rhs)
        t += dt
        n_completed = n + 1
        if history_every > 0 and (n_completed % history_every == 0):
            history.append((t, U.copy()))
        if not np.isfinite(U).all():
            raise FloatingPointError(f"NaN at step {n_completed}, t={t}")

    return dict(U_final=U,
                t=t,
                n_steps=n_completed,
                history=history,
                info_last=info_last)


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
