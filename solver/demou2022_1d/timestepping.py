# solver/demou2022_1d/timestepping.py
"""
CFL-based time step for Mode A (acoustic CFL).

Mode A constraint: dt ≤ CFL * dx / (|u| + c_mix)

c_mix is computed from the current mixture sound speed.
For water/air, c_mix ≈ c_water ≈ 1500 m/s dominates.
"""

import numpy as np


def compute_dt(u: np.ndarray, c_mix: np.ndarray,
               dx: float, CFL: float) -> float:
    """
    Parameters
    ----------
    u      : (N,) cell velocity
    c_mix  : (N,) mixture sound speed (sqrt of c2_mix)
    dx     : cell width
    CFL    : target CFL number (Mode A: ≤ 0.5 for CICSAM stability)

    Returns
    -------
    dt : float
    """
    s_max = float(np.max(np.abs(u) + c_mix))
    if s_max <= 0.0:
        s_max = 1.0
    return CFL * dx / s_max
