"""2-D entry point for the five_eq_IMEX multidimensional extension."""
from __future__ import annotations

from .nd_solver import solve_nd


def solve_2d(eos1, eos2, W0, dx, t_end, **kwargs):
    """Advance a 2-D state W=(alpha1,T1,T2,ux,uy,p)."""
    return solve_nd(eos1, eos2, W0, dx, t_end, dim=2, **kwargs)


solve = solve_2d
