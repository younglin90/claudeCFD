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
    state: Optional[Sequence[float]] = None  # for dirichlet  (W of the fixed state)


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


def apply_patch_bcs(mesh, eq, W_L, W_R, bc_spec: Dict[str, BoundaryCondition]):
    """Overwrite the W_R column at every boundary face according to the
    patch BC dict.

    `bc_spec` maps `patch_name` → BoundaryCondition.
    """
    if not bc_spec:
        return W_L, W_R
    for f in range(mesh.n_faces):
        if mesh.face_neighbour[f] >= 0:
            continue                              # interior face
        tag = int(mesh.face_bc_tag[f])
        if tag <= 0 or tag - 1 >= len(mesh.bc_patches):
            # Untagged boundary face — fall back to transmissive
            W_R[:, f] = W_L[:, f]
            continue
        patch = mesh.bc_patches[tag - 1]
        bc = bc_spec.get(patch)
        if bc is None:
            W_R[:, f] = W_L[:, f]
            continue
        n = mesh.face_normals[f]
        if bc.kind == 'transmissive':
            W_R[:, f] = W_L[:, f]
        elif bc.kind == 'reflective':
            W_R[:, f] = _reflect_velocity(eq, W_L[:, f], n)
        elif bc.kind == 'dirichlet':
            if bc.state is None:
                raise ValueError(f"BC patch '{patch}': dirichlet requires `state=`.")
            W_R[:, f] = np.asarray(bc.state, dtype=float)
        else:
            raise ValueError(f"unknown BC kind '{bc.kind}' on patch '{patch}'")
    return W_L, W_R
