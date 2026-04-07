# solver/denner_1d/config.py
# Ref: DENNER_SCHEME.md § 0 (implementation roadmap)
#
# Configuration dataclass for the denner_1d solver.

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SolverConfig:
    """
    Configuration for the Mode A implicit 1D solver.

    Attributes
    ----------
    mode : str
        Solver mode. Currently only 'A' supported (segregated VOF + coupled p,u,T).
    phase : int
        Implementation phase. 1 = inviscid Euler (Phase 1a).
    max_picard_iter : int
        Maximum Picard (outer non-linear) iterations per time step.
    picard_tol : float
        Convergence tolerance for Picard iterations (relative residual).
    CFL : float
        Target convective CFL number for automatic dt selection.
    bdf_order : int
        BDF order for time integration. 1 = BDF1 (first step), 2 = BDF2.
    p_floor : float
        Minimum allowed pressure [Pa].
    T_floor : float
        Minimum allowed temperature [K].
    n_ghost : int
        Number of ghost cell layers. Phase 1: 2, Phase 2: 3 (for BSD).
    verbose : bool
        Print progress information.
    dt_fixed : Optional[float]
        If set, override CFL-based dt with this fixed value [s].
    output_every : int
        Save snapshot every N steps (0 = no intermediate output).
    """
    mode:            str   = 'A'
    phase:           int   = 1
    max_picard_iter: int   = 20
    picard_tol:      float = 1e-6
    CFL:             float = 0.5
    bdf_order:       int   = 2    # applied after first step; first step always BDF1
    p_floor:         float = 1.0
    T_floor:         float = 1e-3
    n_ghost:         int   = 2
    verbose:         bool  = True
    dt_fixed:        Optional[float] = None
    output_every:    int   = 0

    def to_dict(self):
        """Return config as a plain dict (for passing to step functions)."""
        return {
            'mode':            self.mode,
            'phase':           self.phase,
            'max_picard_iter': self.max_picard_iter,
            'picard_tol':      self.picard_tol,
            'CFL':             self.CFL,
            'bdf_order':       self.bdf_order,
            'p_floor':         self.p_floor,
            'T_floor':         self.T_floor,
            'n_ghost':         self.n_ghost,
            'verbose':         self.verbose,
            'dt_fixed':        self.dt_fixed,
            'output_every':    self.output_every,
        }


# Default EOS parameters (standard NASG values from CLAUDE.md)
PHASE_AIR = {
    'gamma': 1.4,
    'pinf':  0.0,
    'b':     0.0,
    'kv':    717.5,
    'eta':   0.0,
}

PHASE_WATER_LIQUID = {
    'gamma': 1.187,
    'pinf':  7.028e8,
    'b':     6.61e-4,
    'kv':    3610.0,
    'eta':  -1.177788e6,
}

PHASE_WATER_VAPOR = {
    'gamma': 1.467,
    'pinf':  0.0,
    'b':     0.0,
    'kv':    955.0,
    'eta':   2.077616e6,
}
