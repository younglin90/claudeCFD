# solver/denner_1d/__init__.py
# Ref: DENNER_SCHEME.md — Package entry point
#
# Implicit 1D two-phase compressible Euler solver.
# Mode A: segregated VOF (Crank-Nicolson) + coupled (p,u,T) BDF1/BDF2.

from .main import run
from .config import SolverConfig, PHASE_AIR, PHASE_WATER_LIQUID, PHASE_WATER_VAPOR
from .grid import make_uniform_grid, smooth_vof_profile

__all__ = [
    'run',
    'SolverConfig',
    'PHASE_AIR',
    'PHASE_WATER_LIQUID',
    'PHASE_WATER_VAPOR',
    'make_uniform_grid',
    'smooth_vof_profile',
]
