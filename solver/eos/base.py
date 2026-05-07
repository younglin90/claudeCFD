"""
Abstract base class for EOS implementations.

All EOS classes must implement this interface for use with the
1D all-speed compressible solver (solver_1d.py).

EOS contract:
    - pressure(rho, T) -> float
    - internal_energy(rho, T) -> float
    - sound_speed(rho, T) -> float
    - temperature_from_rho_e(rho, e) -> float
    - temperature_from_rho_p(rho, p) -> float

The parameter `rho` here is the *species* partial density rho_i = rho * Y_i.
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class EOSBase(ABC):
    """Abstract base class for equation of state implementations."""

    @abstractmethod
    def pressure(self, rho: float, T: float) -> float:
        """Compute pressure given species density and temperature."""
        ...

    @abstractmethod
    def internal_energy(self, rho: float, T: float) -> float:
        """Compute specific internal energy given species density and temperature."""
        ...

    @abstractmethod
    def sound_speed(self, rho: float, T: float) -> float:
        """Compute speed of sound given species density and temperature."""
        ...

    @abstractmethod
    def temperature_from_rho_e(self, rho: float, e: float) -> float:
        """Invert internal energy to recover temperature."""
        ...

    @abstractmethod
    def temperature_from_rho_p(self, rho: float, p: float) -> float:
        """Recover temperature given density and pressure."""
        ...
