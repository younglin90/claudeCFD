# solver/denner_1d/eos/invert.py
# Ref: DENNER_SCHEME.md § 2.3, Eq.(10)-(11)
#
# Newton inversion: (psi, rho_target, E_target) -> (p, T)
# Solves the 2x2 nonlinear system cell-by-cell.

import numpy as np
from .base import compute_phase_props


def invert_eos(psi, rho_target, E_target, p_guess, T_guess, ph1, ph2,
               tol=1e-12, max_iter=20):
    """
    Recover (p, T) from conserved mixture state (psi, rho, E_int).

    Parameters
    ----------
    psi       : ndarray (N,)  volume fraction of phase 1
    rho_target: ndarray (N,)  mixture density [kg/m³]
    E_target  : ndarray (N,)  mixture internal energy density [J/m³]
                              (E_total - 0.5*rho*u² for the internal part)
    p_guess   : ndarray (N,)  initial pressure guess [Pa]
    T_guess   : ndarray (N,)  initial temperature guess [K]
    ph1, ph2  : dict          NASG EOS parameters
    tol       : float         convergence tolerance on relative residual
    max_iter  : int           maximum Newton iterations

    Returns
    -------
    p : ndarray (N,)  recovered pressure [Pa]
    T : ndarray (N,)  recovered temperature [K]
    """
    N = len(psi)
    p = p_guess.copy()
    T = T_guess.copy()

    for _ in range(max_iter):
        pr1 = compute_phase_props(p, T, ph1)
        pr2 = compute_phase_props(p, T, ph2)

        rho1, rho2 = pr1['rho'], pr2['rho']
        E1,   E2   = pr1['E'],   pr2['E']

        # Residuals  [Eq. 10]
        f1 = psi * rho1 + (1.0 - psi) * rho2 - rho_target
        f2 = psi * E1   + (1.0 - psi) * E2   - E_target

        # Check convergence
        rel1 = np.abs(f1) / (np.abs(rho_target) + 1e-300)
        rel2 = np.abs(f2) / (np.abs(E_target)   + 1e-300)
        if np.max(rel1) < tol and np.max(rel2) < tol:
            break

        # Jacobian  [Eq. 11]
        J00 = psi * pr1['zeta'] + (1.0 - psi) * pr2['zeta']   # ∂f1/∂p
        J01 = psi * pr1['phi']  + (1.0 - psi) * pr2['phi']    # ∂f1/∂T
        J10 = psi * pr1['dEdp'] + (1.0 - psi) * pr2['dEdp']   # ∂f2/∂p
        J11 = psi * pr1['dEdT'] + (1.0 - psi) * pr2['dEdT']   # ∂f2/∂T

        # 2x2 determinant  (cell-wise, vectorised)
        det = J00 * J11 - J01 * J10 + 1e-300

        # Newton update  [delta = J^{-1} @ f]
        dp = ( J11 * f1 - J01 * f2) / det
        dT = (-J10 * f1 + J00 * f2) / det

        p -= dp
        T -= dT

        # Physical bounds
        p = np.maximum(p, 1.0)
        T = np.maximum(T, 1e-3)

    return p, T
