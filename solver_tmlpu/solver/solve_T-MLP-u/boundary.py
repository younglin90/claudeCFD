"""Boundary-condition handling for face-state reconstruction.

Two layers:

  (1) **patch BCs** — applied per boundary patch (mesh.bc_patches).
      A BC is a callable: bc(W_inside, face_normal) → W_face_outside.
      The solver evaluates patch BCs at every boundary face *after*
      reconstruction has produced (W_L, W_R) with W_R as a "default"
      copy of W_L for boundary slots.  We overwrite W_R there with the
      patch-specific value.

  (2) **periodic** — handled inside the Mesh builder by folding faces;
      no BC application needed.

Built-in patch types (all parameter-free):
  'transmissive' — W_R = W_L (zero-gradient)
  'reflective'   — flips the normal component of velocity (advection.velocity
                   not relevant; for Euler, u_n → −u_n)
  'dirichlet'    — W_R = supplied constant W_state
  'dirichlet_func' — W_R = state(face_point, time) for exact moving BCs
  'periodic_pair' — internally treated by Mesh builder, listed here only
                   so users can label patches as such.

The `BoundarySpec` is a dict keyed by patch name (string) → BoundaryCondition.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence
import numpy as np


@dataclass
class BoundaryCondition:
    kind: str                              # 'transmissive' | 'reflective' | 'dirichlet'
    state: Optional[object] = None  # fixed W tuple, or callable for dirichlet_func


# Helper: per-face vector reflection of velocity component(s).
def _reflect_velocity(eq, W_in, normal):
    """Flip the velocity component(s) along the face normal.

    Advection has a constant background velocity (no per-cell velocity to
    reflect — handled at the equation level if the user wishes); for
    Euler1D / Euler2D we mirror the normal velocity.
    """
    eqname = eq.__class__.__name__
    W = np.array(W_in, copy=True)
    if eqname == 'Euler1D':
        # W = (rho, u, p);   u → −u
        W[1] = -W[1]
    elif eqname == 'Euler2D':
        # W = (rho, u, v, p);   (u,v) → (u,v) − 2 (u·n) n
        un = W[1] * normal[..., 0] + W[2] * normal[..., 1]
        W[1] -= 2.0 * un * normal[..., 0]
        W[2] -= 2.0 * un * normal[..., 1]
    elif eqname == 'Advection':
        # nothing to reflect — passive scalar
        pass
    else:
        raise ValueError(f"reflective BC undefined for equation {eqname}")
    return W


def apply_patch_bcs(mesh, eq, W_L, W_R, bc_spec: Dict[str, BoundaryCondition],
                    points=None, time: float = 0.0):
    """Overwrite the W_R column at every boundary face according to the
    patch BC dict.

    `bc_spec` maps `patch_name` → BoundaryCondition.
    """
    if not bc_spec:
        return W_L, W_R

    patch_faces = getattr(mesh, '_bc_patch_face_cache', None)
    if patch_faces is None:
        boundary = mesh.face_neighbour < 0
        patch_faces = {}
        handled = np.zeros(mesh.n_faces, dtype=bool)
        for i, patch in enumerate(mesh.bc_patches, start=1):
            faces = np.where(boundary & (mesh.face_bc_tag == i))[0]
            patch_faces[patch] = faces
            handled[faces] = True
        patch_faces[None] = np.where(boundary & ~handled)[0]
        mesh._bc_patch_face_cache = patch_faces

    untagged = patch_faces.get(None)
    if untagged is not None and untagged.size:
        W_R[:, untagged] = W_L[:, untagged]

    for patch, faces in patch_faces.items():
        if patch is None or faces.size == 0:
            continue
        bc = bc_spec.get(patch)
        if bc is None:
            W_R[:, faces] = W_L[:, faces]
            continue
        if bc.kind == 'transmissive':
            W_R[:, faces] = W_L[:, faces]
        elif bc.kind == 'reflective':
            W_R[:, faces] = _reflect_velocity(
                eq, W_L[:, faces], mesh.face_normals[faces])
        elif bc.kind == 'dirichlet':
            if bc.state is None:
                raise ValueError(f"BC patch '{patch}': dirichlet requires `state=`.")
            W_R[:, faces] = np.asarray(bc.state, dtype=float)[:, None]
        elif bc.kind == 'dirichlet_func':
            if not callable(bc.state):
                raise ValueError(
                    f"BC patch '{patch}': dirichlet_func requires callable state.")
            face_pts = (mesh.face_centers[faces]
                        if points is None else points[faces])
            tried_batch = False
            try:
                vals = bc.state(face_pts, time)
                tried_batch = True
            except TypeError:
                try:
                    vals = bc.state(face_pts)
                    tried_batch = True
                except Exception:
                    tried_batch = False
            except Exception:
                tried_batch = False
            if tried_batch:
                vals = np.asarray(vals, dtype=float)
                if vals.shape == (faces.size, W_R.shape[0]):
                    W_R[:, faces] = vals.T
                elif vals.shape == (W_R.shape[0], faces.size):
                    W_R[:, faces] = vals
                else:
                    tried_batch = False
            if not tried_batch:
                for f in faces:
                    point = (mesh.face_centers[f]
                             if points is None else points[f])
                    try:
                        W_R[:, f] = np.asarray(
                            bc.state(point, time), dtype=float)
                    except TypeError:
                        W_R[:, f] = np.asarray(bc.state(point), dtype=float)
        else:
            raise ValueError(f"unknown BC kind '{bc.kind}' on patch '{patch}'")
    return W_L, W_R
