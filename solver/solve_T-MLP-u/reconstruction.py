"""Face-state reconstruction for FVM — primitive-variable based.

A `Reconstruction` builds, for every interior face f, the primitive
state W on the *owner* and *neighbour* sides of the face:

    reconstruct(mesh, W_cell, eq) → (W_face_owner, W_face_neighbour)

with both arrays shaped (nvar, n_faces).  Boundary faces are filled by
`boundary.apply_bc` *before* reconstruction (they live as ghost cells).

T-MLP-u (the user's method) is the headline reconstruction that this
package is intended to validate.  We expose:

    FirstOrder            — piecewise constant (no reconstruction)
    MinmodTVD1D           — classical 1D structured TVD (any limiter from limiters.py)
    MLPU                  — Park-Yoon-Kim 2010 baseline (PLACEHOLDER for now)
    TMLPU                 — user's T-MLP-u (PLACEHOLDER — to be filled by the user)

A Reconstruction object is *grid-aware* (Mesh) and *equation-aware*
(Equation) — same interface for 1D / 2D / structured / unstructured,
which matches the user's stated scope.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

from limiters import (minmod, minmod2, superbee, t_mlp_u_face_value,
                      TVD_LIMITERS)


# ─── Base interface ────────────────────────────────────────────────────────
class Reconstruction:
    name: str = 'base'

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        """Return (W_owner_side, W_neighbour_side) at face evaluation points.

        Parameters
        ----------
        mesh, W_cell, eq : as usual.
        eval_points : (n_faces, 2) array, optional
            Face-side evaluation point per face.  When None (default) the
            face *centres* are used — this matches midpoint quadrature.
            The high-order solver path passes Gauss-quadrature points
            here to maintain spatial order ≥ 3.

        The two returned arrays have shape (nvar, n_faces) and represent
        W reconstructed from the owner side and neighbour side at the
        chosen evaluation points.
        """
        raise NotImplementedError


# ─── 1st-order (piecewise constant) ────────────────────────────────────────
@dataclass
class FirstOrder(Reconstruction):
    name: str = 'first_order'

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        # First-order reconstruction is constant in each cell, so the
        # `eval_points` argument has no effect.
        n_faces = mesh.n_faces
        nvar = W_cell.shape[0]
        W_L = np.empty((nvar, n_faces), dtype=float)
        W_R = np.empty((nvar, n_faces), dtype=float)
        owner = mesh.face_owner
        nei   = mesh.face_neighbour
        for v in range(nvar):
            W_L[v] = W_cell[v, owner]
            W_R[v] = np.where(nei >= 0, W_cell[v, np.maximum(nei, 0)], W_cell[v, owner])
        return W_L, W_R


# ─── 1D structured TVD (any limiter) ───────────────────────────────────────
@dataclass
class MinmodTVD1D(Reconstruction):
    """Classical 1D MUSCL-Hancock TVD reconstruction with a swappable
    slope limiter.  Works on `mesh.dim == 1`.

    For each cell C with left neighbour L and right neighbour R, define:
        Δ_L = W_C − W_L,   Δ_R = W_R − W_C
        Δ   = limiter2(Δ_L, Δ_R)              # symmetric minmod2 form
    Face values:
        W_at_face_to_left_of_C  = W_C − ½ Δ
        W_at_face_to_right_of_C = W_C + ½ Δ

    `limiter2(a, b)` defaults to the symmetric minmod (`limiters.minmod2`)
    which is universally TVD with **zero** free parameters.
    """
    limiter2: Callable = minmod2
    name: str = 'minmod_tvd_1d'

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        # 1D structured uses face centres ⇒ eval_points is ignored.
        assert mesh.dim == 1, "MinmodTVD1D requires a 1D mesh."
        nvar, N = W_cell.shape
        # Cell-centred slopes (TVD-limited)
        slopes = np.zeros_like(W_cell)
        # Use cell_neighbours to find L/R; for periodic 1D mesh both exist.
        # For non-periodic 1D, end cells have only one neighbour — keep slope = 0
        # there (degenerates to first-order at the boundary, standard practice).
        for i in range(N):
            nbrs = mesh.cell_neighbours[i]
            valid = [n for n in nbrs if n >= 0]
            if len(valid) < 2:
                continue
            # Sort by x to identify left vs right
            xs = mesh.cell_centers[valid, 0]
            order = np.argsort(xs)
            left  = valid[order[0]]
            right = valid[order[-1]]
            dL = W_cell[:, i]    - W_cell[:, left]
            dR = W_cell[:, right] - W_cell[:, i]
            slopes[:, i] = self.limiter2(dL, dR)

        owner = mesh.face_owner
        nei   = mesh.face_neighbour
        n_faces = mesh.n_faces
        W_L = np.empty((nvar, n_faces), dtype=float)
        W_R = np.empty((nvar, n_faces), dtype=float)

        # Decide sign of half-slope based on which side of the owner the face is.
        # For a 1D structured mesh face_normals is +1 (pointing owner→neighbour).
        # owner contributes +½·slope on its outgoing-normal side.
        sign_owner = mesh.face_normals[:, 0]      # +1 always for our 1D builder
        for v in range(nvar):
            W_L[v] = W_cell[v, owner] + 0.5 * sign_owner * slopes[v, owner]
            # Neighbour-side reconstruction: face_normal points owner→neighbour,
            # so neighbour reconstructs *backward* by −½ slope.
            n_idx = np.maximum(nei, 0)
            W_R[v] = np.where(
                nei >= 0,
                W_cell[v, n_idx] - 0.5 * sign_owner * slopes[v, n_idx],
                W_cell[v, owner],   # boundary placeholder; solver overwrites
            )
        return W_L, W_R


# ─── MLP-u (Park-Yoon-Kim 2010) — placeholder ──────────────────────────────
@dataclass
class MLPU(Reconstruction):
    """MLP-u Multi-dimensional Limiter for unstructured grids
    (Park, Yoon, Kim, J. Comput. Phys. 2010).

    Steps (for each cell):
      1. Compute least-squares unlimited gradient ∇W in the cell.
      2. For each *vertex* of the cell, compute the projected value
         W_vertex = W_cell + ∇W · (x_vertex − x_cell).
      3. Vertex-based MLP slope limiter (Hubbard 1999 + Park-Yoon-Kim 2010
         u-correction) constrains W_vertex into [W_min^vtx, W_max^vtx]
         where the bounds are taken over all cells sharing that vertex.

    Implementation deferred to a follow-up commit — included here as a slot
    so the test infrastructure can switch reconstruction by name.
    """
    name: str = 'mlp_u'

    def reconstruct(self, mesh, W_cell, eq):
        raise NotImplementedError(
            "MLPU.reconstruct: not yet implemented. "
            "Park-Yoon-Kim 2010 vertex-based MLP-u — pending."
        )


# ─── T-MLP-u (user's method) ───────────────────────────────────────────────
@dataclass
class TMLPU(Reconstruction):
    """T-MLP-u — primitive-variable high-order reconstruction wrapping any
    classical TVD limiter (minmod / van Leer / superbee / MC / van Albada /
    UMIST) with a Local Maximum Principle (LMP) bound on top.

    Parameters
    ----------
    tvd : str | callable, default 'superbee'
        Base TVD limiter ψ_TVD wrapped by T-MLP-u.  String keys come
        from `limiters.TVD_LIMITERS`.  Callable must take r → ψ.
    hancock_courant : float, default 0.0
        Hancock factor C_f (0 ⇒ plain MUSCL).
    stencil : {'face', 'vertex'}, default 'face'
        Local stencil used for the LSQ gradient/Hessian *and* for the
        LMP φ_min / φ_max bounds (only relevant on 2D unstructured grids):
          'face'   — 1-ring face neighbours (current default; tightest bound)
          'vertex' — Park-Yoon-Kim 2010 MLP-u stencil: every cell sharing
                     any vertex with C; wider, more accurate gradient and
                     a less-restrictive bound on smooth regions.
    order : {1, 2}, default 1
        Polynomial order of the reconstruction inside each cell.
          1 — linear LSQ  (∇W only, 2nd-order accurate face value)
          2 — quadratic LSQ  (∇W + Hessian, 3rd-order on smooth data)
        order=2 implies stencil='vertex' (face-only triangle stencil has
        too few neighbours for the quadratic system); the wrapper enforces
        this automatically.

    Implementation status:
      - 1D structured: vectorised, working.
      - 2D Cartesian (axis-aligned): linear path only — `stencil`/`order`
        flags are honoured silently.
      - 2D unstructured: both stencils, both orders, vectorised.
    """
    tvd: object = 'superbee'
    hancock_courant: float = 0.0
    stencil: str = 'face'
    order: int = 1
    mlp_bound: bool = True   # False ⇒ pure TVD limiter (no LMP wrapper)
    extremum_relax: bool = False   # smooth-region LMP relaxation
    vertex_mlp: bool = False    # PYG2010 vertex-projected polynomial bound
    tvb_M: float = 0.0   # Cockburn-Shu TVB modulus (M·h² LMP tolerance)
    virtual_uu_gradient: bool = False
    # When True, the slope ratio r = (φ_U − φ_UU)/(φ_D − φ_U) uses a
    # *virtual* far-upwind value derived from the LSQ gradient at the
    # upwind cell — Darwish-Moukalled (2003), Jasak (1996).  Avoids the
    # geometric face-neighbour search and works on any unstructured mesh.
    #     φ_UU_virt = φ_D − 2·∇φ_U · (x_D − x_U)
    #     ⇒ φ_U − φ_UU_virt = −Δ⁺ + 2·∇φ_U · d_UD
    cicsam_full: bool = False
    # Full CICSAM (Ubbink 1997) in NVD framework:
    #   • Hyper-C arm (sharp): φ̃_f^HC = min(1, φ̃_C/Co)
    #   • Ultimate-QUICKEST arm (smooth):
    #       φ̃_f^UQ = (8·Co·φ̃_C + (1−Co)·(6·φ̃_C+3))/8
    #     clipped to [φ̃_C, φ̃_f^HC]
    #   • Blend: φ̃_f = γ·HC + (1−γ)·UQ
    #     γ = cos²(2θ), θ = ∠(∇φ_C, d_UD).  γ→1 sharp interface aligned
    #     with face → maximum compression; γ→0 gradient tangential →
    #     gentler UQ.  Outside monotone region [φ̃_C ∈ 0..1] falls back
    #     to first-order.  Bypasses the standard ψ_TVD path; uses the
    #     virtual-UU formulation for φ̃_C natively.
    cicsam_courant: float = 0.4   # Co for both HC and UQ formulae
    tvd_smooth: object = None
    # Optional secondary TVD limiter used ONLY at smooth cells (when
    # extremum_relax=True and is_smooth_cell[c]=True).  Sharp cells keep
    # the primary `tvd` limiter clipped by LMP.  Mesh-independent
    # criterion (LSQ residual) replaces CICSAM's cos²(2θ) blend.
    #     Sharp cell:  ψ = min(ψ_TVD, ψ_LMP)            ← compressive
    #     Smooth cell: ψ = clip(ψ_TVD_smooth, 0, 2)     ← gentler
    name: str = 't_mlp_u'

    def __post_init__(self):
        if isinstance(self.tvd, str):
            key = self.tvd.lower()
            if key not in TVD_LIMITERS:
                raise ValueError(
                    f"unknown TVD limiter '{self.tvd}'; "
                    f"available: {list(TVD_LIMITERS)}")
            self._psi_tvd = TVD_LIMITERS[key]
            self._tvd_name = key
        elif callable(self.tvd):
            self._psi_tvd = self.tvd
            self._tvd_name = getattr(self.tvd, '__name__', 'custom')
        else:
            raise TypeError("`tvd` must be a string or a callable.")
        # Resolve secondary smooth-cell limiter (optional).
        self._psi_tvd_smooth = None
        if self.tvd_smooth is not None:
            if isinstance(self.tvd_smooth, str):
                key2 = self.tvd_smooth.lower()
                if key2 not in TVD_LIMITERS:
                    raise ValueError(
                        f"unknown tvd_smooth '{self.tvd_smooth}'")
                self._psi_tvd_smooth = TVD_LIMITERS[key2]
            elif callable(self.tvd_smooth):
                self._psi_tvd_smooth = self.tvd_smooth
            else:
                raise TypeError("`tvd_smooth` must be a string or callable.")
        if self.stencil not in ('face', 'vertex', 'vertex2'):
            raise ValueError(
                f"stencil must be 'face' / 'vertex' / 'vertex2', got {self.stencil!r}")
        if self.order not in (1, 2, 3):
            raise ValueError(f"order must be 1, 2, or 3, got {self.order!r}")
        if self.order == 2 and self.stencil == 'face':
            # Quadratic needs 5 unknowns — face stencil (3 nbrs on triangles)
            # is under-determined.  Promote to vertex stencil silently.
            self.stencil = 'vertex'
        if self.order == 3 and self.stencil != 'vertex2':
            # Cubic LSQ needs 9 unknowns; vertex (≈ 7–10 cells) is borderline,
            # vertex2 (≈ 25 cells) is comfortably over-determined.
            self.stencil = 'vertex2'

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        if mesh.dim == 1:
            return self._reconstruct_1d(mesh, W_cell, eq, eval_points)
        if mesh.kind == 'structured_2d':
            return self._reconstruct_2d_axis_aligned(mesh, W_cell, eq, eval_points)
        if mesh.kind == 'unstructured_2d':
            return self._reconstruct_unstructured_2d(mesh, W_cell, eq, eval_points)
        raise NotImplementedError(
            f"TMLPU.reconstruct: unsupported mesh "
            f"(dim={mesh.dim}, kind={mesh.kind})."
        )

    # --- 1D structured implementation ---------------------------------------
    def _reconstruct_1d(self, mesh, W_cell, eq, eval_points=None):
        nvar, N = W_cell.shape
        n_faces = mesh.n_faces
        owner = mesh.face_owner
        nei   = mesh.face_neighbour

        # Build "left / right" neighbour tables for every cell using x-coords.
        left = np.full(N, -1, dtype=int)
        right = np.full(N, -1, dtype=int)
        xs = mesh.cell_centers[:, 0]
        for i in range(N):
            for nb in mesh.cell_neighbours[i]:
                if nb < 0:
                    continue
                if xs[nb] < xs[i]:
                    left[i] = nb
                else:
                    right[i] = nb

        W_L = np.empty((nvar, n_faces), dtype=float)
        W_R = np.empty((nvar, n_faces), dtype=float)

        # First-order default (overridden for interior faces below;
        # boundary slots are then patched by `boundary.apply_patch_bcs`).
        n_idx = np.maximum(nei, 0)
        for v in range(nvar):
            W_L[v] = W_cell[v, owner]
            W_R[v] = np.where(nei >= 0, W_cell[v, n_idx], W_cell[v, owner])

        # Interior faces: full T-MLP-u
        interior = np.where(nei >= 0)[0]
        if interior.size == 0:
            return W_L, W_R

        # For each interior face, work out the UU on each side.
        # Rule: nei is to the right of owner  ⇒ UU_owner = left[owner],
        #                                       UU_nei   = right[nei].
        #       nei is to the left  of owner  ⇒ UU_owner = right[owner],
        #                                       UU_nei   = left[nei].
        n_face_int = interior.size
        UU_o = np.empty(n_face_int, dtype=int)
        UU_n = np.empty(n_face_int, dtype=int)
        for k, f in enumerate(interior):
            o = int(owner[f]); n = int(nei[f])
            if xs[n] > xs[o]:
                UU_o[k] = left[o]
                UU_n[k] = right[n]
            else:
                UU_o[k] = right[o]
                UU_n[k] = left[n]

        # Treat missing UU (at-boundary cell) by falling back to U itself
        # — equivalent to first-order on that face side.  W_L/W_R already
        # holds first-order values, so we only overwrite with the T-MLP-u
        # value when both sides have a valid UU.
        valid_o = UU_o >= 0
        valid_n = UU_n >= 0

        o_idx = owner[interior]
        n_idx_int = nei[interior]
        UU_o_safe = np.where(valid_o, UU_o, o_idx)
        UU_n_safe = np.where(valid_n, UU_n, n_idx_int)

        # Vectorised gather over variables and faces.
        for v in range(nvar):
            phi_U_o = W_cell[v, o_idx]
            phi_D_o = W_cell[v, n_idx_int]
            phi_UU_o = W_cell[v, UU_o_safe]
            recon_owner = t_mlp_u_face_value(phi_UU_o, phi_U_o, phi_D_o,
                                             self._psi_tvd,
                                             hancock_courant=self.hancock_courant)
            W_L[v, interior] = np.where(valid_o, recon_owner, phi_U_o)

            phi_U_n = W_cell[v, n_idx_int]
            phi_D_n = W_cell[v, o_idx]
            phi_UU_n = W_cell[v, UU_n_safe]
            recon_nei = t_mlp_u_face_value(phi_UU_n, phi_U_n, phi_D_n,
                                           self._psi_tvd,
                                           hancock_courant=self.hancock_courant)
            W_R[v, interior] = np.where(valid_n, recon_nei, phi_U_n)

        return W_L, W_R

    # --- 2D structured Cartesian implementation -----------------------------
    def _reconstruct_2d_axis_aligned(self, mesh, W_cell, eq, eval_points=None):
        """T-MLP-u for axis-aligned Cartesian grids.

        For every face the dominant axis is read off the face normal; the
        upstream cell UU is then the same-axis neighbour of U on the side
        opposite to D.  The per-face formula is identical to the 1D path.
        """
        nvar, N = W_cell.shape
        n_faces = mesh.n_faces
        owner = mesh.face_owner
        nei = mesh.face_neighbour

        xs = mesh.cell_centers[:, 0]
        ys = mesh.cell_centers[:, 1]

        # Per-cell axis neighbour tables.
        xneg = np.full(N, -1, dtype=int)
        xpos = np.full(N, -1, dtype=int)
        yneg = np.full(N, -1, dtype=int)
        ypos = np.full(N, -1, dtype=int)
        for i in range(N):
            for nb in mesh.cell_neighbours[i]:
                if nb < 0:
                    continue
                ddx = xs[nb] - xs[i]
                ddy = ys[nb] - ys[i]
                if abs(ddx) >= abs(ddy):
                    if ddx < 0:
                        xneg[i] = nb
                    else:
                        xpos[i] = nb
                else:
                    if ddy < 0:
                        yneg[i] = nb
                    else:
                        ypos[i] = nb

        # First-order default
        W_L = np.empty((nvar, n_faces), dtype=float)
        W_R = np.empty((nvar, n_faces), dtype=float)
        n_idx_def = np.maximum(nei, 0)
        for v in range(nvar):
            W_L[v] = W_cell[v, owner]
            W_R[v] = np.where(nei >= 0, W_cell[v, n_idx_def], W_cell[v, owner])

        # Per-face UU lookup (axis decided by face normal sign)
        interior = np.where(nei >= 0)[0]
        if interior.size == 0:
            return W_L, W_R
        n_face_int = interior.size
        UU_o = np.empty(n_face_int, dtype=int)
        UU_n = np.empty(n_face_int, dtype=int)
        nx = mesh.face_normals[interior, 0]
        ny = mesh.face_normals[interior, 1]
        for k in range(n_face_int):
            f = interior[k]; o = int(owner[f]); nb = int(nei[f])
            if abs(nx[k]) >= abs(ny[k]):
                # x-axis face — nei is on +x or -x of owner depending on sign(nx).
                if nx[k] >= 0:
                    UU_o[k] = xneg[o]
                    UU_n[k] = xpos[nb]
                else:
                    UU_o[k] = xpos[o]
                    UU_n[k] = xneg[nb]
            else:
                if ny[k] >= 0:
                    UU_o[k] = yneg[o]
                    UU_n[k] = ypos[nb]
                else:
                    UU_o[k] = ypos[o]
                    UU_n[k] = yneg[nb]

        valid_o = UU_o >= 0
        valid_n = UU_n >= 0
        o_idx = owner[interior]
        n_idx_int = nei[interior]
        UU_o_safe = np.where(valid_o, UU_o, o_idx)
        UU_n_safe = np.where(valid_n, UU_n, n_idx_int)

        for v in range(nvar):
            phi_U_o = W_cell[v, o_idx]
            phi_D_o = W_cell[v, n_idx_int]
            phi_UU_o = W_cell[v, UU_o_safe]
            recon_owner = t_mlp_u_face_value(phi_UU_o, phi_U_o, phi_D_o,
                                             self._psi_tvd,
                                             hancock_courant=self.hancock_courant)
            W_L[v, interior] = np.where(valid_o, recon_owner, phi_U_o)

            phi_U_n = W_cell[v, n_idx_int]
            phi_D_n = W_cell[v, o_idx]
            phi_UU_n = W_cell[v, UU_n_safe]
            recon_nei = t_mlp_u_face_value(phi_UU_n, phi_U_n, phi_D_n,
                                           self._psi_tvd,
                                           hancock_courant=self.hancock_courant)
            W_R[v, interior] = np.where(valid_n, recon_nei, phi_U_n)

        return W_L, W_R

    # --- Unstructured 2D implementation (vectorised) ------------------------
    def _reconstruct_unstructured_2d(self, mesh, W_cell, eq, eval_points=None):
        """T-MLP-u extended to unstructured 2D grids — fully vectorised.

        Per cell C, an unlimited least-squares gradient ∇W_C is computed
        from the face-neighbour data.  At face f owned by C the candidate
        reconstruction is W̃_face = W_C + ∇W_C · (x_f − x_C).  We pair
        this geometric δ with the user's TVD ratio r and the MLP bound:

            δ        = (1 − C_f) · ∇W_C · (x_f − x_C)
            Δ_+      = W_N − W_C            (cell-to-cell along U → D)
            Δ_-      = W_C − W_UU
            r        = Δ_- / Δ_+
            ψ_TVD    = self._psi_tvd(r)
            φ_min    = min over face-neighbours of C (incl. C itself)
            φ_max    = max over the same set
            ψ_MLP    = (φ_max − W_C)/δ  if δ>0,  (φ_min − W_C)/δ  if δ<0
            ψ_final  = max(0, min(2, ψ_TVD, ψ_MLP))
            W_face   = W_C + ψ_final · δ

        Reduces to the 1D / 2D-Cartesian formulae on uniform structured
        grids (since ∇W_C · (x_face − x_C) = ½ Δ_+ there).

        UU is chosen per face as the face-neighbour of C whose centroid
        offset is most *opposite* the downstream direction (x_D − x_C).
        For unstructured grids without such a "opposite" neighbour
        (boundary cells, acute fans) UU defaults to C itself, giving
        Δ_- = 0 and falling back to the LMP-only limiter.

        Mesh-dependent quantities (UU per face, neighbour padding, A⁻¹
        for the gradient, face-displacement vectors) are cached on the
        mesh so they are computed only once.
        """
        nvar, N = W_cell.shape
        n_faces = mesh.n_faces
        owner = mesh.face_owner
        nei = mesh.face_neighbour

        ctx = self._unstructured_cache(mesh)

        nb_padded = ctx['nb_padded']    # (N, max_nb)
        nb_safe   = ctx['nb_safe']
        valid_nb  = ctx['valid_nb']
        A_basis   = ctx['A']            # (N, max_nb, nbasis)  ← already √W·A
        ATA_inv   = ctx['ATA_inv']      # (N, nbasis, nbasis)
        nbasis    = ctx['nbasis']
        sqrt_w    = ctx['sqrt_w']       # (N, max_nb) — same √W weighting
        UU_o_int  = ctx['UU_o_int']     # interior faces only
        UU_n_int  = ctx['UU_n_int']
        d_o_int   = ctx['d_o_int']      # (Nint, 2)  x_neighbour − x_owner
        interior  = ctx['interior']

        # Per-call evaluation points (face centres by default; high-order
        # solver passes Gauss-quadrature points here to maintain ≥3rd-order
        # face quadrature).
        if eval_points is None:
            eval_points = mesh.face_centers
        dx_fo = eval_points[interior] - mesh.cell_centers[mesh.face_owner[interior]]
        dx_fn = eval_points[interior] - mesh.cell_centers[mesh.face_neighbour[interior]]

        # 1) phi_min / phi_max per cell over (self ∪ chosen stencil).
        # Avoid np.nanmin/nanmax by replacing invalid neighbour slots with
        # the centre cell's own value (a no-op for min/max).  Much faster
        # than NaN-propagating reductions and gives identical results.
        W_self = W_cell[:, :, None]                          # (nvar, N, 1)
        W_nb_filled = np.where(valid_nb[None, :, :],
                               W_cell[:, nb_safe], W_self)   # (nvar, N, max_nb)
        W_with_self = np.concatenate([W_self, W_nb_filled], axis=2)
        phi_min_cell = W_with_self.min(axis=2)
        phi_max_cell = W_with_self.max(axis=2)

        # 2) LSQ polynomial coefficients per cell, per variable.
        #    coeffs[v, c, :] = ATA_inv[c] · (Aᵀ · ΔW)[c]
        # Fused einsum: combine the two separate calls into one
        # (ATA_inv · Aᵀ) · ΔW_w — saves one tensor traversal per variable.
        coeffs = np.empty((nvar, N, nbasis), dtype=float)
        is_smooth_cell = np.zeros((nvar, N), dtype=bool)
        for v in range(nvar):
            delta_W = (W_cell[v, nb_safe] - W_cell[v, :, None]) * valid_nb
            # Weighted RHS: A_basis is already √W·A, multiply ΔW by √W too.
            delta_W_w = delta_W * sqrt_w
            # rhs[c, i] = Σ_k A_basis[c, k, i] · delta_W_w[c, k]
            # coeffs[c, i] = Σ_j ATA_inv[c, i, j] · rhs[c, j]
            # Fused: coeffs[c, i] = Σ_jk ATA_inv[c, i, j] · A_basis[c, k, j] · delta_W_w[c, k]
            coeffs[v] = np.einsum('cij,ckj,ck->ci',
                                  ATA_inv, A_basis, delta_W_w,
                                  optimize=True)
            if self.extremum_relax:
                # Smoothness indicator: relative LSQ residual norm.  On a
                # smooth function the k-th-order LSQ polynomial fits
                # neighbours to O(h^{k+1}); discontinuities give residual
                # ≈ jump.  We additionally restrict relaxation to cells
                # that are LOCAL EXTREMA (the only place LMP is binding).
                # Predicted ΔW (un-weighted): A · p, not √W · A · p.
                # `A_basis` here is √W·A, so divide by sqrt_w (safe because
                # sqrt_w > 0 on valid neighbours).
                delta_W_pred_w = np.einsum('ckb,cb->ck', A_basis, coeffs[v])
                delta_W_pred = delta_W_pred_w / np.maximum(sqrt_w, 1e-30)
                delta_W_pred = delta_W_pred * valid_nb
                resid = (delta_W - delta_W_pred) * valid_nb
                num = np.sqrt(np.sum(resid * resid, axis=1))
                den = np.sqrt(np.sum(delta_W * delta_W, axis=1))
                smoothness = num / np.maximum(den, 1e-30)
                is_smooth_cell[v] = smoothness < 0.1

        # Helper — evaluate the LSQ polynomial at a face displacement vector.
        def _poly_at(coef_per_face, dxs):
            """coef_per_face: (Nf, nbasis), dxs: (Nf, 2). Returns (Nf,)."""
            δx = dxs[:, 0]; δy = dxs[:, 1]
            if nbasis == 2:
                return coef_per_face[:, 0] * δx + coef_per_face[:, 1] * δy
            quad = (coef_per_face[:, 0] * δx +
                    coef_per_face[:, 1] * δy +
                    0.5 * coef_per_face[:, 2] * δx * δx +
                    coef_per_face[:, 3] * δx * δy +
                    0.5 * coef_per_face[:, 4] * δy * δy)
            if nbasis == 5:
                return quad
            # nbasis == 9 (cubic)
            return (quad +
                    coef_per_face[:, 5] * δx * δx * δx / 6.0 +
                    0.5 * coef_per_face[:, 6] * δx * δx * δy +
                    0.5 * coef_per_face[:, 7] * δx * δy * δy +
                    coef_per_face[:, 8] * δy * δy * δy / 6.0)

        # Helper — evaluate the LSQ polynomial of cell C at displacement
        # vectors arranged as (N, V, 2), returning (N, V).
        def _poly_at_cell_offsets(coeffs_v, offsets_NV2):
            """coeffs_v: (N, nbasis), offsets: (N, V, 2). Returns (N, V)."""
            δx = offsets_NV2[:, :, 0]
            δy = offsets_NV2[:, :, 1]
            if nbasis == 2:
                return (coeffs_v[:, None, 0] * δx +
                        coeffs_v[:, None, 1] * δy)
            quad = (coeffs_v[:, None, 0] * δx +
                    coeffs_v[:, None, 1] * δy +
                    0.5 * coeffs_v[:, None, 2] * δx * δx +
                    coeffs_v[:, None, 3] * δx * δy +
                    0.5 * coeffs_v[:, None, 4] * δy * δy)
            if nbasis == 5:
                return quad
            return (quad +
                    coeffs_v[:, None, 5] * δx * δx * δx / 6.0 +
                    0.5 * coeffs_v[:, None, 6] * δx * δx * δy +
                    0.5 * coeffs_v[:, None, 7] * δx * δy * δy +
                    coeffs_v[:, None, 8] * δy * δy * δy / 6.0)

        # TVB tolerance — needed by both vertex-MLP and per-face MLP paths.
        if self.mlp_bound and self.tvb_M > 0.0:
            h_sq = getattr(mesh, '_tvb_h_sq_cache', None)
            if h_sq is None:
                h_sq = float(np.median(mesh.cell_volumes))
                mesh._tvb_h_sq_cache = h_sq
            tvb_eps = self.tvb_M * h_sq
        else:
            tvb_eps = 0.0

        # Vertex-MLP — Park-Yoon-Kim 2010 cell-wise ψ.
        # CANONICAL T-MLP-u: linear gradient projection at vertices only,
        # not the full polynomial.  The LSQ poly's higher-order terms
        # (½∂xx, ∂xy, ½∂yy, …) overshoot at vertices for cubic k=3 LSQ
        # and over-shrink ψ; PYG2010 uses ∇W_C · (x_v − x_C) only.
        # Augmented with TVB (Cockburn-Shu) tolerance M·h² so smooth
        # extrema do not get pinned by the LMP.
        psi_vertex_cell = None
        if self.vertex_mlp and ctx['vertex_offsets'] is not None:
            v2c_safe = ctx['v2c_safe']
            v2c_valid = ctx['v2c_valid']
            cell_node_safe = np.where(ctx['cell_node_valid'],
                                      ctx['cell_node_arr'], 0)
            cell_node_valid = ctx['cell_node_valid']
            vertex_offsets = ctx['vertex_offsets']  # (N, V, 2)

            psi_vertex_cell = np.ones((nvar, N), dtype=float)
            for v in range(nvar):
                W_at_vc = W_cell[v, v2c_safe]                  # (Nnodes, max_v2c)
                # Use np.where + min/max instead of nanmin/max (slow).
                W_self_v = W_cell[v, v2c_safe[:, :1]]
                W_at_vc_filled = np.where(v2c_valid, W_at_vc, W_self_v)
                phi_min_v = W_at_vc_filled.min(axis=1)         # (Nnodes,)
                phi_max_v = W_at_vc_filled.max(axis=1)
                # ─── Fix 1: linear gradient only (canonical PYG2010) ────
                # Per-cell gradient = (coeffs[0], coeffs[1])
                grad_x = coeffs[v, :, 0]                       # (N,)
                grad_y = coeffs[v, :, 1]
                proj = (grad_x[:, None] * vertex_offsets[..., 0]
                        + grad_y[:, None] * vertex_offsets[..., 1])
                W_C = W_cell[v]
                phi_min_at_node = phi_min_v[cell_node_safe]    # (N, V)
                phi_max_at_node = phi_max_v[cell_node_safe]
                # ─── Fix 2: TVB tolerance M·h² added to bounds ──────────
                allowed_max = phi_max_at_node - W_C[:, None] + tvb_eps
                allowed_min = phi_min_at_node - W_C[:, None] - tvb_eps
                # Field-relative epsilon (not absolute 1e-30) — guards
                # divide when ∇φ·δ is tiny vs the local field magnitude.
                W_scale = max(float(np.max(np.abs(W_C))), 1e-30)
                eps = 1e-12 * W_scale
                psi_v_each = np.ones_like(proj)
                pos = proj > eps
                neg = proj < -eps
                psi_v_each = np.where(
                    pos,
                    np.minimum(1.0, allowed_max / np.maximum(proj, eps)),
                    psi_v_each)
                psi_v_each = np.where(
                    neg,
                    np.minimum(1.0, allowed_min / np.minimum(proj, -eps)),
                    psi_v_each)
                psi_v_each = np.where(cell_node_valid, psi_v_each, 1.0)
                psi_v_each = np.clip(psi_v_each, 0.0, 1.0)
                psi_vertex_cell[v] = np.min(psi_v_each, axis=1)

        # 3) Default first-order (overridden for interior below).
        W_L = np.empty((nvar, n_faces), dtype=float)
        W_R = np.empty((nvar, n_faces), dtype=float)
        n_idx_def = np.maximum(nei, 0)
        for v in range(nvar):
            W_L[v] = W_cell[v, owner]
            W_R[v] = np.where(nei >= 0, W_cell[v, n_idx_def], W_cell[v, owner])

        if interior.size == 0:
            return W_L, W_R

        # tvb_eps already computed above before vertex-MLP path.

        o_idx = owner[interior]
        n_idx = nei[interior]
        valid_o = UU_o_int >= 0
        valid_n = UU_n_int >= 0
        UU_o_safe = np.where(valid_o, UU_o_int, o_idx)
        UU_n_safe = np.where(valid_n, UU_n_int, n_idx)

        one_minus_C = (1.0 - self.hancock_courant)
        _EPS = 1e-30

        # Pre-compute side-independent fallback flag once.
        all_valid_when_virt = self.virtual_uu_gradient

        Co_cic = self.cicsam_courant if self.cicsam_full else 0.0

        def _cicsam_face_value(grad_x, grad_y, d_vec, phi_self, delta_p,
                                phi_min_b, phi_max_b, smooth_mask):
            """One-side CICSAM (Ubbink) NVD blend.

            Inputs are aligned to "owner-as-U" or "neighbour-as-U" view of
            the face, with d_vec = x_D − x_U (for the local U).  Returns
            the limited face value (1-D array of length n_int).
            """
            gdotd = grad_x * d_vec[:, 0] + grad_y * d_vec[:, 1]
            denom = 2.0 * gdotd
            safe_denom = np.where(np.abs(denom) > _EPS,
                                   denom, np.copysign(_EPS, denom))
            delta_minus_v = -delta_p + 2.0 * gdotd
            phi_C_t = delta_minus_v / safe_denom
            # HC + UQ arms
            phi_f_HC = np.minimum(1.0, phi_C_t / max(Co_cic, 1e-10))
            phi_f_UQ = ((8.0 * Co_cic) * phi_C_t
                        + (1.0 - Co_cic) * (6.0 * phi_C_t + 3.0)) / 8.0
            phi_f_UQ = np.maximum(phi_f_UQ, phi_C_t)
            phi_f_UQ = np.minimum(phi_f_UQ, phi_f_HC)
            # γ_f = cos²(2θ) ; θ = ∠(∇φ, d_UD)
            grad_sq = grad_x * grad_x + grad_y * grad_y
            d_sq = d_vec[:, 0] ** 2 + d_vec[:, 1] ** 2
            cos2_th = (gdotd * gdotd) / np.maximum(grad_sq * d_sq, _EPS)
            cos_2th = 2.0 * cos2_th - 1.0
            gamma_f = np.minimum(cos_2th * cos_2th, 1.0)
            phi_f_t = gamma_f * phi_f_HC + (1.0 - gamma_f) * phi_f_UQ
            is_mono = (phi_C_t >= 0.0) & (phi_C_t <= 1.0)
            phi_f_t = np.where(is_mono, phi_f_t, phi_C_t)
            recon_cic = phi_self + (phi_f_t - phi_C_t) * denom
            if self.mlp_bound:
                clip_v = np.clip(recon_cic, phi_min_b, phi_max_b)
                if self.extremum_relax:
                    return np.where(smooth_mask, recon_cic, clip_v)
                return clip_v
            return recon_cic

        for v in range(nvar):
            # ---------- Owner side ----------
            phi_U  = W_cell[v, o_idx]
            phi_D  = W_cell[v, n_idx]
            phi_UU = W_cell[v, UU_o_safe]
            delta_unl = _poly_at(coeffs[v, o_idx], dx_fo)
            delta = one_minus_C * delta_unl

            delta_plus = phi_D - phi_U
            abs_dp = np.abs(delta_plus)
            is_zero_dp = abs_dp <= _EPS
            safe_dp = np.where(is_zero_dp,
                               np.copysign(_EPS, delta_plus), delta_plus)

            # ─── Full CICSAM path (bypasses ψ_TVD machinery) ───────────
            if self.cicsam_full:
                phi_min_b = phi_min_cell[v, o_idx] - tvb_eps
                phi_max_b = phi_max_cell[v, o_idx] + tvb_eps
                recon_o = _cicsam_face_value(
                    coeffs[v, o_idx, 0], coeffs[v, o_idx, 1],
                    d_o_int, phi_U, delta_plus,
                    phi_min_b, phi_max_b,
                    is_smooth_cell[v, o_idx] if self.extremum_relax else None,
                )
                if all_valid_when_virt:
                    W_L[v, interior] = recon_o
                else:
                    W_L[v, interior] = np.where(valid_o, recon_o, phi_U)
                # ----- Neighbour side ---------------------------------
                phi_Un = W_cell[v, n_idx]
                phi_Dn = W_cell[v, o_idx]
                delta_p_n = phi_Dn - phi_Un
                phi_min_bn = phi_min_cell[v, n_idx] - tvb_eps
                phi_max_bn = phi_max_cell[v, n_idx] + tvb_eps
                # Neighbour's d_UD = x_o − x_n = −d_o_int
                d_n_int = -d_o_int
                recon_n = _cicsam_face_value(
                    coeffs[v, n_idx, 0], coeffs[v, n_idx, 1],
                    d_n_int, phi_Un, delta_p_n,
                    phi_min_bn, phi_max_bn,
                    is_smooth_cell[v, n_idx] if self.extremum_relax else None,
                )
                if all_valid_when_virt:
                    W_R[v, interior] = recon_n
                else:
                    W_R[v, interior] = np.where(valid_n, recon_n, phi_Un)
                continue   # next variable — skip standard ψ path

            if self.virtual_uu_gradient:
                grad_x_U = coeffs[v, o_idx, 0]
                grad_y_U = coeffs[v, o_idx, 1]
                gdotd = grad_x_U * d_o_int[:, 0] + grad_y_U * d_o_int[:, 1]
                delta_minus_virt = -delta_plus + 2.0 * gdotd
                delta_minus = np.where(valid_o,
                                       phi_U - phi_UU, delta_minus_virt)
            else:
                delta_minus = np.where(valid_o, phi_U - phi_UU, 0.0)
            r = delta_minus / safe_dp
            psi_tvd = self._psi_tvd(r)
            psi_tvd = np.where(is_zero_dp, 2.0, psi_tvd)

            if self.mlp_bound:
                if psi_vertex_cell is not None:
                    psi_v_o = psi_vertex_cell[v, o_idx]
                    psi_lmp = np.minimum(psi_tvd, psi_v_o)
                else:
                    phi_min = phi_min_cell[v, o_idx] - tvb_eps
                    phi_max = phi_max_cell[v, o_idx] + tvb_eps
                    # Single-pass NVD bound: clip δ away from 0 (sign-aware).
                    delta_clip_pos = np.maximum(delta,  _EPS)
                    delta_clip_neg = np.minimum(delta, -_EPS)
                    psi_mlp_pos = (phi_max - phi_U) / delta_clip_pos
                    psi_mlp_neg = (phi_min - phi_U) / delta_clip_neg
                    psi_mlp = np.where(delta >  _EPS, psi_mlp_pos,
                              np.where(delta < -_EPS, psi_mlp_neg, 2.0))
                    psi_lmp = np.minimum(psi_tvd, psi_mlp)
                # Final clip [0, 2] outside the inner branches.
                np.clip(psi_lmp, 0.0, 2.0, out=psi_lmp)
                if self.extremum_relax:
                    if self._psi_tvd_smooth is not None:
                        psi_smooth = np.clip(self._psi_tvd_smooth(r),
                                             0.0, 2.0)
                    else:
                        psi_smooth = np.clip(psi_tvd, 0.0, 2.0)
                    psi_final = np.where(is_smooth_cell[v, o_idx],
                                         psi_smooth, psi_lmp)
                else:
                    psi_final = psi_lmp
            else:
                psi_final = np.clip(psi_tvd, 0.0, 2.0)
            recon = phi_U + psi_final * delta
            if all_valid_when_virt:
                W_L[v, interior] = recon       # virt-UU: every face valid
            else:
                W_L[v, interior] = np.where(valid_o, recon, phi_U)

            # ---------- Neighbour side ----------
            phi_U  = W_cell[v, n_idx]
            phi_D  = W_cell[v, o_idx]
            phi_UU = W_cell[v, UU_n_safe]
            delta_unl = _poly_at(coeffs[v, n_idx], dx_fn)
            delta = one_minus_C * delta_unl

            delta_plus = phi_D - phi_U
            abs_dp = np.abs(delta_plus)
            is_zero_dp = abs_dp <= _EPS
            safe_dp = np.where(is_zero_dp,
                               np.copysign(_EPS, delta_plus), delta_plus)
            if self.virtual_uu_gradient:
                grad_x_U = coeffs[v, n_idx, 0]
                grad_y_U = coeffs[v, n_idx, 1]
                gdotd = -(grad_x_U * d_o_int[:, 0] +
                          grad_y_U * d_o_int[:, 1])
                delta_minus_virt = -delta_plus + 2.0 * gdotd
                delta_minus = np.where(valid_n,
                                       phi_U - phi_UU, delta_minus_virt)
            else:
                delta_minus = np.where(valid_n, phi_U - phi_UU, 0.0)
            r = delta_minus / safe_dp
            psi_tvd = self._psi_tvd(r)
            psi_tvd = np.where(is_zero_dp, 2.0, psi_tvd)

            if self.mlp_bound:
                if psi_vertex_cell is not None:
                    psi_v_n = psi_vertex_cell[v, n_idx]
                    psi_lmp = np.minimum(psi_tvd, psi_v_n)
                else:
                    phi_min = phi_min_cell[v, n_idx] - tvb_eps
                    phi_max = phi_max_cell[v, n_idx] + tvb_eps
                    delta_clip_pos = np.maximum(delta,  _EPS)
                    delta_clip_neg = np.minimum(delta, -_EPS)
                    psi_mlp_pos = (phi_max - phi_U) / delta_clip_pos
                    psi_mlp_neg = (phi_min - phi_U) / delta_clip_neg
                    psi_mlp = np.where(delta >  _EPS, psi_mlp_pos,
                              np.where(delta < -_EPS, psi_mlp_neg, 2.0))
                    psi_lmp = np.minimum(psi_tvd, psi_mlp)
                np.clip(psi_lmp, 0.0, 2.0, out=psi_lmp)
                if self.extremum_relax:
                    if self._psi_tvd_smooth is not None:
                        psi_smooth = np.clip(self._psi_tvd_smooth(r),
                                             0.0, 2.0)
                    else:
                        psi_smooth = np.clip(psi_tvd, 0.0, 2.0)
                    psi_final = np.where(is_smooth_cell[v, n_idx],
                                         psi_smooth, psi_lmp)
                else:
                    psi_final = psi_lmp
            else:
                psi_final = np.clip(psi_tvd, 0.0, 2.0)
            recon = phi_U + psi_final * delta
            if all_valid_when_virt:
                W_R[v, interior] = recon
            else:
                W_R[v, interior] = np.where(valid_n, recon, phi_U)

        return W_L, W_R

    # --- Mesh-dependent cache for the unstructured path ---------------------
    def _unstructured_cache(self, mesh):
        """Cache LSQ operator, LMP stencil, UU lookup, face offsets.
        Same dict is returned on every call (mesh-keyed)."""
        cache_key = f'_tmlpu_cache_{id(self)}'
        if hasattr(mesh, cache_key):
            return getattr(mesh, cache_key)

        N = mesh.n_cells
        n_faces = mesh.n_faces
        n_centers = mesh.cell_centers
        f_centers = mesh.face_centers
        owner = mesh.face_owner
        nei = mesh.face_neighbour

        # ---- 1) Choose stencil for LSQ + LMP bound -----------------------
        if self.stencil in ('vertex', 'vertex2'):
            n_rings = 1 if self.stencil == 'vertex' else 2
            nb_lists = _build_vertex_neighbours(mesh, n_rings=n_rings)
            if nb_lists is None:
                raise ValueError(
                    f"stencil='{self.stencil}' requires mesh.cell_nodes; "
                    "use the unstructured constructor or `criss_cross_box`.")
        else:
            nb_lists = mesh.cell_neighbours

        # Always keep the *face* neighbour list separately — UU pick must
        # use it (the TVD ratio is defined on the face-cell direction).
        face_nb_lists = mesh.cell_neighbours

        # ---- 2) Padded neighbour table (LSQ + LMP) -----------------------
        max_nb = max((len(nbs) for nbs in nb_lists if nbs), default=1)
        max_nb = max(max_nb, 1)
        nb_padded = np.full((N, max_nb), -1, dtype=int)
        for c in range(N):
            nbs = [int(nb) for nb in nb_lists[c] if int(nb) >= 0]
            nb_padded[c, :len(nbs)] = nbs
        valid_nb = nb_padded >= 0
        nb_safe = np.where(valid_nb, nb_padded, 0)
        d_full = n_centers[nb_safe] - n_centers[:, None, :]  # (N, max_nb, 2)
        d_full = d_full * valid_nb[:, :, None]

        # ---- 3) LSQ basis matrix A and (Aᵀ A)⁻¹ --------------------------
        # order=1: A = [dx, dy]                            (nbasis = 2)
        # order=2: A = [dx, dy, ½dx², dx·dy, ½dy²]        (nbasis = 5)
        dx = d_full[:, :, 0]
        dy = d_full[:, :, 1]
        if self.order == 1:
            A = d_full                                         # (N, max_nb, 2)
            nbasis = 2
        elif self.order == 2:
            A = np.stack([dx, dy,
                          0.5 * dx * dx,
                          dx * dy,
                          0.5 * dy * dy], axis=-1)             # (N, max_nb, 5)
            nbasis = 5
        else:  # order == 3
            A = np.stack([dx, dy,
                          0.5 * dx * dx, dx * dy, 0.5 * dy * dy,
                          dx * dx * dx / 6.0,
                          0.5 * dx * dx * dy,
                          0.5 * dx * dy * dy,
                          dy * dy * dy / 6.0], axis=-1)        # (N, max_nb, 9)
            nbasis = 9
        A = A * valid_nb[:, :, None]
        # Inverse-distance LSQ weighting — emphasises closer cells.
        # weight = 1/d^6
        dist_sq = dx * dx + dy * dy + 1e-30
        sqrt_w = (1.0 / dist_sq) ** 1.5 * valid_nb              # √(1/d^6) = 1/d^3
        A = A * sqrt_w[:, :, None]                              # A → √W · A
        ATA = np.einsum('cki,ckj->cij', A, A)                  # (N, nbasis, nbasis)

        if nbasis == 2:
            det = ATA[:, 0, 0] * ATA[:, 1, 1] - ATA[:, 0, 1] * ATA[:, 1, 0]
            ok = np.abs(det) > 1e-30
            det_safe = np.where(ok, det, 1.0)
            ATA_inv = np.empty_like(ATA)
            ATA_inv[:, 0, 0] = ATA[:, 1, 1] / det_safe
            ATA_inv[:, 1, 1] = ATA[:, 0, 0] / det_safe
            ATA_inv[:, 0, 1] = -ATA[:, 0, 1] / det_safe
            ATA_inv[:, 1, 0] = -ATA[:, 1, 0] / det_safe
            ATA_inv = np.where(ok[:, None, None], ATA_inv, 0.0)
        else:
            # 5×5 batched inverse — fall back to per-cell pinv for singular
            # cells (typical: cells with too few valid neighbours).
            ATA_inv = np.zeros_like(ATA)
            sign, logdet = np.linalg.slogdet(ATA)
            ok = np.isfinite(logdet) & (sign != 0)
            ok_idx = np.where(ok)[0]
            if ok_idx.size > 0:
                ATA_inv[ok_idx] = np.linalg.inv(ATA[ok_idx])
            for c in np.where(~ok)[0]:
                try:
                    ATA_inv[c] = np.linalg.pinv(ATA[c])
                except np.linalg.LinAlgError:
                    pass

        # ---- 4) Per-face UU pick (uses FACE-neighbour set) ----------------
        face_max_nb = max((len(nbs) for nbs in face_nb_lists), default=1)
        face_max_nb = max(face_max_nb, 1)
        face_nb_padded = np.full((N, face_max_nb), -1, dtype=int)
        for c in range(N):
            nbs = [int(nb) for nb in face_nb_lists[c] if int(nb) >= 0]
            face_nb_padded[c, :len(nbs)] = nbs
        face_valid = face_nb_padded >= 0
        face_safe = np.where(face_valid, face_nb_padded, 0)
        d_face = (n_centers[face_safe] - n_centers[:, None, :]) * face_valid[:, :, None]

        interior = np.where(nei >= 0)[0]
        if interior.size > 0:
            o_idx = owner[interior]
            n_idx = nei[interior]
            d_o = n_centers[n_idx] - n_centers[o_idx]                # (Nint, 2)
            score_o = -np.einsum('fki,fi->fk', d_face[o_idx], d_o)
            score_o = np.where(face_valid[o_idx], score_o, -np.inf)
            best_k_o = np.argmax(score_o, axis=1)
            best_score_o = score_o[np.arange(interior.size), best_k_o]
            UU_o_int = face_nb_padded[o_idx, best_k_o]
            UU_o_int = np.where(best_score_o > 0.0, UU_o_int, -1)

            d_n = n_centers[o_idx] - n_centers[n_idx]
            score_n = -np.einsum('fki,fi->fk', d_face[n_idx], d_n)
            score_n = np.where(face_valid[n_idx], score_n, -np.inf)
            best_k_n = np.argmax(score_n, axis=1)
            best_score_n = score_n[np.arange(interior.size), best_k_n]
            UU_n_int = face_nb_padded[n_idx, best_k_n]
            UU_n_int = np.where(best_score_n > 0.0, UU_n_int, -1)

            dx_fo = f_centers[interior] - n_centers[o_idx]
            dx_fn = f_centers[interior] - n_centers[n_idx]
            # Owner→neighbour displacement, used by the gradient-based
            # virtual-UU formula (Darwish-Moukalled).  d_n_int = −d_o_int.
            d_o_int = d_o
        else:
            UU_o_int = np.zeros(0, dtype=int)
            UU_n_int = np.zeros(0, dtype=int)
            dx_fo = np.zeros((0, 2), dtype=float)
            dx_fn = np.zeros((0, 2), dtype=float)
            d_o_int = np.zeros((0, 2), dtype=float)

        # ---- 5) Vertex-MLP supporting structures ------------------------
        # Used only when self.vertex_mlp is True; cheap to build always.
        cn = getattr(mesh, 'cell_nodes', None)
        if cn:
            n_v_per_cell = max(len(c) for c in cn)
            cell_node_arr = np.full((N, n_v_per_cell), -1, dtype=int)
            for c, vs in enumerate(cn):
                cell_node_arr[c, :len(vs)] = vs
            cell_node_valid = cell_node_arr >= 0
            cell_node_safe = np.where(cell_node_valid, cell_node_arr, 0)
            # Vertex coordinates (broadcast vs cell centre).
            vertex_xy = mesh.nodes[cell_node_safe]            # (N, V, 2)
            vertex_offsets = vertex_xy - mesh.cell_centers[:, None, :]
            vertex_offsets = vertex_offsets * cell_node_valid[:, :, None]
            # Inverse map vertex → cells.
            n_nodes = mesh.nodes.shape[0]
            v2c_lists = [[] for _ in range(n_nodes)]
            for c, vs in enumerate(cn):
                for v in vs:
                    v2c_lists[int(v)].append(c)
            v2c_max = max(len(L) for L in v2c_lists) if v2c_lists else 1
            v2c_max = max(v2c_max, 1)
            v2c_padded = np.full((n_nodes, v2c_max), -1, dtype=int)
            for vi, cs in enumerate(v2c_lists):
                v2c_padded[vi, :len(cs)] = cs
            v2c_valid = v2c_padded >= 0
            v2c_safe = np.where(v2c_valid, v2c_padded, 0)
        else:
            cell_node_arr = None
            cell_node_valid = None
            vertex_offsets = None
            v2c_padded = v2c_safe = v2c_valid = None

        ctx = dict(
            nb_padded=nb_padded, nb_safe=nb_safe, valid_nb=valid_nb,
            A=A, ATA_inv=ATA_inv, nbasis=nbasis, sqrt_w=sqrt_w,
            interior=interior,
            UU_o_int=UU_o_int, UU_n_int=UU_n_int,
            d_o_int=d_o_int,
            dx_fo=dx_fo, dx_fn=dx_fn,
            order=self.order, stencil=self.stencil,
            cell_node_arr=cell_node_arr,
            cell_node_valid=cell_node_valid,
            vertex_offsets=vertex_offsets,
            v2c_safe=v2c_safe, v2c_valid=v2c_valid,
        )
        setattr(mesh, cache_key, ctx)
        return ctx


def _build_vertex_neighbours(mesh, n_rings: int = 1):
    """Cells sharing any vertex with each cell.

    n_rings = 1 : direct vertex-neighbours (Park-Yoon-Kim 2010 default).
    n_rings = 2 : 2-ring  — vertex-neighbours of the 1-ring (much wider
                  stencil; ~25–30 cells / triangle on criss-cross).
    """
    if not getattr(mesh, 'cell_nodes', None):
        return None
    vertex_cells = {}
    for c, nodes in enumerate(mesh.cell_nodes):
        for v in nodes:
            vertex_cells.setdefault(int(v), []).append(c)
    ring1 = []
    for c, nodes in enumerate(mesh.cell_nodes):
        s = set()
        for v in nodes:
            for c2 in vertex_cells[int(v)]:
                if c2 != c:
                    s.add(c2)
        ring1.append(s)
    if n_rings == 1:
        return [sorted(s) for s in ring1]
    out = []
    for c in range(mesh.n_cells):
        s = set(ring1[c])
        for c2 in ring1[c]:
            s.update(ring1[c2])
        s.discard(c)
        out.append(sorted(s))
    return out


# ─── Registry helper ───────────────────────────────────────────────────────
def get_reconstruction(name: str, **kwargs) -> Reconstruction:
    """Construct a Reconstruction object by name."""
    table = {
        'first_order':  FirstOrder,
        'minmod_tvd_1d': MinmodTVD1D,
        'mlp_u':        MLPU,
        't_mlp_u':      TMLPU,
    }
    name = name.lower()
    if name not in table:
        raise ValueError(f"unknown reconstruction '{name}'; available: {list(table)}")
    return table[name](**kwargs)
