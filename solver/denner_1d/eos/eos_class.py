# solver/denner_1d/eos/eos_class.py
# General EOS interface + NasgEOS implementation + factory function.
# New EOS types (RKPR, SRK, etc.) can be added by subclassing EOS.

import numpy as np


class EOS:
    """General EOS interface. All methods accept scalar or ndarray."""

    def rho(self, p, T):
        """Density [kg/m³]."""
        raise NotImplementedError

    def h(self, p, T):
        """Specific static enthalpy [J/kg]."""
        raise NotImplementedError

    def c(self, p, T):
        """Speed of sound [m/s]."""
        raise NotImplementedError

    def cp(self, p, T):
        """∂h/∂T|_p  [J/(kg·K)]."""
        raise NotImplementedError

    def dh_dp(self, p, T):
        """∂h/∂p|_T  [m³/kg]."""
        raise NotImplementedError

    def drho_dp(self, p, T):
        """ζ = ∂ρ/∂p|_T  [kg/(m³·Pa)]."""
        raise NotImplementedError

    def drho_dT(self, p, T):
        """φ = ∂ρ/∂T|_p  [kg/(m³·K)]."""
        raise NotImplementedError

    def e_vol(self, p, T):
        """Volumetric internal energy ρe [J/m³]."""
        raise NotImplementedError

    def de_vol_dp(self, p, T):
        """∂(ρe)/∂p|_T  [J/(m³·Pa)]."""
        raise NotImplementedError

    def de_vol_dT(self, p, T):
        """∂(ρe)/∂T|_p  [J/(m³·K)]."""
        raise NotImplementedError


class NasgEOS(EOS):
    """Noble-Abel Stiffened Gas EOS.

    Equations of state:
        h = γ·κᵥ·T + b·p + η
        ρ = (p + p∞) / [κᵥ·T·(γ-1) + b·(p + p∞)]
        c = sqrt(γ·(p+p∞) / (ρ·(1−b·ρ)))

    For Ideal Gas: set pinf=0, b=0, eta=0.
    """

    def __init__(self, gamma, pinf, b, kv, eta=0.0):
        self.gamma = gamma
        self.pinf = pinf
        self.b = b
        self.kv = kv
        self.eta = eta
        self._gm1 = gamma - 1.0
        self._gkv = gamma * kv

    def _A(self, p, T):
        """A = κᵥ·T·(γ-1) + b·(p+p∞)  [denominator of NASG density]."""
        return self.kv * T * self._gm1 + self.b * (p + self.pinf) + 1e-300

    def rho(self, p, T):
        return (p + self.pinf) / self._A(p, T)

    def h(self, p, T):
        return self._gkv * T + self.b * p + self.eta

    def c(self, p, T):
        rho_val = self.rho(p, T)
        one_minus_b_rho = 1.0 - self.b * rho_val
        return np.sqrt(self.gamma * (p + self.pinf) / (rho_val * one_minus_b_rho + 1e-300))

    def cp(self, p, T):
        """∂h/∂T|_p = γ·κᵥ  (constant for NASG)."""
        return self._gkv

    def dh_dp(self, p, T):
        """∂h/∂p|_T = b  (constant for NASG)."""
        return self.b

    def drho_dp(self, p, T):
        A = self._A(p, T)
        return self.kv * T * self._gm1 / (A * A + 1e-300)

    def drho_dT(self, p, T):
        A = self._A(p, T)
        return -(p + self.pinf) * self.kv * self._gm1 / (A * A + 1e-300)

    def e_vol(self, p, T):
        rho_val = self.rho(p, T)
        one_minus_b_rho = 1.0 - self.b * rho_val
        return rho_val * (self.kv * T + self.eta) + self.pinf * one_minus_b_rho / self._gm1

    def de_vol_dp(self, p, T):
        zeta = self.drho_dp(p, T)
        coef = self.kv * T + self.eta - self.pinf * self.b / self._gm1
        return coef * zeta

    def de_vol_dT(self, p, T):
        rho_val = self.rho(p, T)
        phi_val = self.drho_dT(p, T)
        coef = self.kv * T + self.eta - self.pinf * self.b / self._gm1
        return rho_val * self.kv + coef * phi_val


def create_eos(ph):
    """Factory: phase dict → NasgEOS object.

    If ph is already an EOS instance, return it unchanged (pass-through).
    This preserves backward compatibility with existing dict-based callers.

    Parameters
    ----------
    ph : dict or EOS
        If dict: must contain 'gamma', 'kv'; may contain 'pinf', 'b', 'eta'
                 (defaults to 0.0 if absent).
        If EOS:  returned as-is.

    Returns
    -------
    EOS
    """
    if isinstance(ph, EOS):
        return ph
    return NasgEOS(
        gamma=float(ph['gamma']),
        pinf=float(ph.get('pinf', 0.0)),
        b=float(ph.get('b', 0.0)),
        kv=float(ph['kv']),
        eta=float(ph.get('eta', 0.0)),
    )
