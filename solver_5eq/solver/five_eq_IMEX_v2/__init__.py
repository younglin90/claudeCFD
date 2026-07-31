"""v2 R1 — explicit FVM solver for the 5-equation Allaire-Massoni model.

Status: R1 baseline (Forward Euler + 1st-order upwind + Allaire D₁=0).
See `docs/five_eq_IMEX_v2_plan.md` for the round-by-round roadmap and
`docs/v2_round_<R>.md` for per-round results.

Public API:
    from solver.five_eq_IMEX_v2 import solve
    from solver.five_eq_IMEX_v2 import EOSPair, make_eos, IdealEOS, SGEOS, NASGEOS
"""
from .main import solve
from .eos_facade import EOSPair, make_eos, IdealEOS, SGEOS, NASGEOS

__all__ = ['solve', 'EOSPair', 'make_eos', 'IdealEOS', 'SGEOS', 'NASGEOS']
