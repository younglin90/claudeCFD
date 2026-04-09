# solver/denner_1d/interface/cicsam.py
# Ref: DENNER_SCHEME.md § 5, Eq.(18)
#
# CICSAM (Compressive Interface Capturing Scheme for Arbitrary Meshes)
# 1D implementation = Hyper-C scheme.
# Note: compression term is NOT added [Zanutto2022].

import numpy as np


def cicsam_face(psi_ext, u_face, dt, dx, n_ghost=2):
    """
    Compute face values of psi using CICSAM Hyper-C.

    Parameters
    ----------
    psi_ext  : ndarray (N + 2*n_ghost,)
               VOF field with ghost cells already applied.
    u_face   : ndarray (N+1,)
               Face velocities at interior faces i+1/2, i=0..N-1.
               Face f is between cells (f-1) and f in interior indexing
               (i.e., between psi_ext[n_ghost+f-1] and psi_ext[n_ghost+f]).
    dt       : float  time step [s]
    dx       : float  cell size [m]
    n_ghost  : int    number of ghost cells on each side

    Returns
    -------
    psi_face : ndarray (N+1,)
               CICSAM face values, clipped to [0, 1].
    """
    ng = n_ghost
    N = len(u_face) - 1  # number of interior cells
    psi_face = np.zeros(N + 1)

    for f in range(N + 1):
        uf = u_face[f]
        # Face f sits between interior cells f-1 and f.
        # In extended array: left cell = ng + f - 1, right cell = ng + f.

        if uf >= 0.0:
            # Flow goes right: donor = left cell, acceptor = right cell
            # upupwind = donor - 1
            i_D   = ng + f - 1   # donor
            i_A   = ng + f       # acceptor
            i_UU  = ng + f - 2   # upupwind (one cell further upstream)
        else:
            # Flow goes left: donor = right cell, acceptor = left cell
            # upupwind = donor + 1
            i_D   = ng + f       # donor
            i_A   = ng + f - 1   # acceptor
            i_UU  = ng + f + 1   # upupwind

        psi_D  = psi_ext[i_D]
        psi_A  = psi_ext[i_A]
        psi_UU = psi_ext[i_UU]

        # Normalised donor value [Eq. 18]
        denom = psi_A - psi_UU
        if abs(denom) < 1e-10:
            # Uniform region: upwind (donor value)
            psi_face[f] = psi_D
            continue

        psi_tilde_D = (psi_D - psi_UU) / denom

        # Local Courant number
        Co_f = abs(uf) * dt / dx

        # Hyper-C face normalised value [Eq. 18]
        if 0.0 <= psi_tilde_D <= 1.0:
            psi_tilde_f = min(psi_tilde_D / max(Co_f, 1e-10), 1.0)
        else:
            psi_tilde_f = psi_tilde_D

        # Un-normalise
        psi_face[f] = psi_UU + psi_tilde_f * denom

    # Clip to physical range
    psi_face = np.clip(psi_face, 0.0, 1.0)
    return psi_face


def cicsam_face_beta(psi_ext, u_face, dt, dx, n_ghost=2):
    """
    Compute CICSAM blending factor beta at each face.
    phi_face = (1 - beta) * phi_Donor + beta * phi_Acceptor

    Caller determines donor/acceptor from sign of u_face:
      u >= 0: donor=left, acceptor=right
      u <  0: donor=right, acceptor=left

    Returns beta : ndarray (N+1,)
    """
    ng = n_ghost
    N = len(u_face) - 1
    beta = np.zeros(N + 1)

    for f in range(N + 1):
        uf = u_face[f]

        if uf >= 0.0:
            i_D  = ng + f - 1   # donor (left)
            i_A  = ng + f       # acceptor (right)
            i_UU = ng + f - 2   # upupwind
        else:
            i_D  = ng + f       # donor (right)
            i_A  = ng + f - 1   # acceptor (left)
            i_UU = ng + f + 1   # upupwind

        psi_D  = psi_ext[i_D]
        psi_A  = psi_ext[i_A]
        psi_UU = psi_ext[i_UU]

        # If acceptor == donor, purely upwind → beta=0
        if abs(psi_A - psi_D) < 1e-10:
            beta[f] = 0.0
            continue

        # Compute the CICSAM face value using same Hyper-C logic
        denom = psi_A - psi_UU
        if abs(denom) < 1e-10:
            # Uniform region: upwind → phi_face = phi_D → beta=0
            beta[f] = 0.0
            continue

        psi_tilde_D = (psi_D - psi_UU) / denom
        Co_f = abs(uf) * dt / dx

        if 0.0 <= psi_tilde_D <= 1.0:
            psi_tilde_f = min(psi_tilde_D / max(Co_f, 1e-10), 1.0)
        else:
            psi_tilde_f = psi_tilde_D

        psi_face_val = psi_UU + psi_tilde_f * denom
        psi_face_val = float(np.clip(psi_face_val, 0.0, 1.0))

        # beta = (phi_face - phi_D) / (phi_A - phi_D)
        raw_beta = (psi_face_val - psi_D) / (psi_A - psi_D)
        beta[f] = float(np.clip(raw_beta, 0.0, 1.0))

    return beta


def cicsam_face_jacobian(psi_ext, u_face, dt, dx, n_ghost=2):
    """
    Compute CICSAM face values AND their exact Newton Jacobian.

    Returns Y_face and ∂Y_face/∂Y_{D,A,UU} for Newton linearization:
      Y_face(Y) ≈ Y_face(Y_k) + J_D·δY_D + J_A·δY_A + J_UU·δY_UU

    Returns
    -------
    psi_face : (N+1,) CICSAM face values
    jac_D, jac_A, jac_UU : (N+1,) Jacobian w.r.t. Donor, Acceptor, UpUpwind
    idx_D, idx_A, idx_UU : (N+1,) int — interior 0-based cell indices (periodic-wrapped)
    """
    ng = n_ghost
    N = len(u_face) - 1
    psi_face = np.zeros(N + 1)
    jac_D  = np.zeros(N + 1)
    jac_A  = np.zeros(N + 1)
    jac_UU = np.zeros(N + 1)
    idx_D  = np.zeros(N + 1, dtype=int)
    idx_A  = np.zeros(N + 1, dtype=int)
    idx_UU = np.zeros(N + 1, dtype=int)

    for f in range(N + 1):
        uf = u_face[f]
        if uf >= 0.0:
            i_D = ng + f - 1; i_A = ng + f; i_UU = ng + f - 2
            idx_D[f] = f - 1; idx_A[f] = f; idx_UU[f] = f - 2
        else:
            i_D = ng + f; i_A = ng + f - 1; i_UU = ng + f + 1
            idx_D[f] = f; idx_A[f] = f - 1; idx_UU[f] = f + 1

        psi_D  = psi_ext[i_D]
        psi_A  = psi_ext[i_A]
        psi_UU = psi_ext[i_UU]

        denom = psi_A - psi_UU
        if abs(denom) < 1e-10:
            # Uniform: upwind
            psi_face[f] = psi_D; jac_D[f] = 1.0
            continue

        psi_tilde_D = (psi_D - psi_UU) / denom
        Co_f = max(abs(uf) * dt / dx, 1e-10)

        if 0.0 <= psi_tilde_D <= 1.0:
            if psi_tilde_D / Co_f <= 1.0:
                # Interpolated: Y_face = Y_UU + (Y_D - Y_UU) / Co
                inv_Co = 1.0 / Co_f
                psi_face[f] = psi_UU + (psi_D - psi_UU) * inv_Co
                jac_D[f] = inv_Co
                jac_UU[f] = 1.0 - inv_Co
            else:
                # Downwind: Y_face = Y_A
                psi_face[f] = psi_A
                jac_A[f] = 1.0
        else:
            # Upwind: Y_face = Y_D
            psi_face[f] = psi_D
            jac_D[f] = 1.0

    psi_face = np.clip(psi_face, 0.0, 1.0)
    idx_D  = idx_D % N
    idx_A  = idx_A % N
    idx_UU = idx_UU % N

    return psi_face, jac_D, jac_A, jac_UU, idx_D, idx_A, idx_UU
