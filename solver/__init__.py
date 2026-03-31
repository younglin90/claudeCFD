# Ref: CLAUDE.md § 구현 파일 구조
"""
CFD Solver package for compressible multi-component flows.
Implements APEC flux (Approximate Pressure-Equilibrium-preserving with Conservation).

Ref: Terashima, Ly, Ihme, J. Comput. Phys. 524 (2025) 113701
"""

from .eos import IdealGasEOS, NASGEOS, SRKEOS
from .utils import cons_to_prim, prim_to_cons, mixture_density, mixture_internal_energy
from .flux import apec_flux
from .jacobian import numerical_jacobian
from .solve import run_1d

__all__ = [
    "IdealGasEOS",
    "NASGEOS",
    "SRKEOS",
    "cons_to_prim",
    "prim_to_cons",
    "mixture_density",
    "mixture_internal_energy",
    "apec_flux",
    "numerical_jacobian",
    "run_1d",
]
