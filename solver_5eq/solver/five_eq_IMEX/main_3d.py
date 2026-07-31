"""3-D entry point for the five_eq_IMEX multidimensional extension."""
from __future__ import annotations

from .nd_solver import solve_nd


def solve_3d(eos1, eos2, W0, dx, t_end, **kwargs):
    """Advance a 3-D state W=(alpha1,T1,T2,ux,uy,uz,p)."""
    return solve_nd(eos1, eos2, W0, dx, t_end, dim=3, **kwargs)


solve = solve_3d
