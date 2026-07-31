"""Thin facade over `solver.He2024.eos_general` for the new IMEX solver.

Keeps the new module's public interface narrow: the IMEX solver only consumes
`EOSPair` (a frozen pair of phase EOS objects).  Internally we re-use the
validated EOS classes and analytic (p, T) derivatives produced in Phase 1.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .he2024_compat import load_eos_general

_eos_general = load_eos_general()
IdealEOS = _eos_general.IdealEOS
SGEOS = _eos_general.SGEOS
NASGEOS = _eos_general.NASGEOS
_to_eos = _eos_general.to_eos

__all__ = ['EOSPair', 'make_eos', 'IdealEOS', 'SGEOS', 'NASGEOS']


def make_eos(kind, **kwargs):
    """Construct an EOS instance by string tag (only Ideal/SG/NASG in Phase 3)."""
    kind = kind.lower()
    if kind in ('ideal', 'gas'):
        return IdealEOS(**kwargs)
    if kind in ('sg', 'stiffened'):
        return SGEOS(**kwargs)
    if kind in ('nasg',):
        return NASGEOS(**kwargs)
    raise ValueError(
        f"Phase 3 supports only 'ideal' | 'sg' | 'nasg'; got '{kind}'. "
        "MieGruneisen/JWL/RKPR are scheduled for Phase 10+ "
        "(see docs/five_eq_all_mach_plan.md).")


@dataclass(frozen=True)
class EOSPair:
    """Pair of phase EOS objects used by the 5-equation solver."""
    eos1: object
    eos2: object

    @classmethod
    def from_dicts(cls, ph1, ph2):
        """Build from legacy dict spec (e.g. {'gamma': 1.4, 'pinf': 0, 'kv': 717.5})."""
        return cls(_to_eos(ph1), _to_eos(ph2))

    def names(self):
        return getattr(self.eos1, 'name', '?'), getattr(self.eos2, 'name', '?')

    def assert_admissible(self, rho1, rho2, p, T1, T2):
        """Raise on inadmissible state (used inside Newton + after recovery)."""
        adm1 = self.eos1.is_admissible(rho1, p, T1)
        adm2 = self.eos2.is_admissible(rho2, p, T2)
        if not (np.all(adm1) and np.all(adm2)):
            raise FloatingPointError(
                f"Inadmissible EOS state: any(adm1)={np.any(adm1)}, any(adm2)={np.any(adm2)}; "
                f"min ρ1={np.min(rho1):.3e}, ρ2={np.min(rho2):.3e}, "
                f"p={np.min(p):.3e}, T1={np.min(T1):.3e}, T2={np.min(T2):.3e}")
