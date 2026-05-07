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
        """Volumetric internal energy ρe [J/m³].
        Default: ρe = ρh − p (thermodynamic identity h ≡ e + p/ρ)."""
        return self.rho(p, T) * self.h(p, T) - p

    def de_vol_dp(self, p, T):
        """∂(ρe)/∂p|_T  [J/(m³·Pa)].
        Default: d(ρh−p)/dp = ρ·∂h/∂p + h·∂ρ/∂p − 1."""
        return self.rho(p, T) * self.dh_dp(p, T) + self.h(p, T) * self.drho_dp(p, T) - 1.0

    def de_vol_dT(self, p, T):
        """∂(ρe)/∂T|_p  [J/(m³·K)].
        Default: d(ρh−p)/dT = ρ·cp + h·∂ρ/∂T."""
        return self.rho(p, T) * self.cp(p, T) + self.h(p, T) * self.drho_dT(p, T)


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

    # e_vol, de_vol_dp, de_vol_dT: use base class defaults (ρh − p identity)


class StiffenedGasEOS(EOS):
    """Stiffened Gas EOS with two conventions.

    convention='standard' (Le Métayer 2004):
        ρ  = (p + P∞) / (R·T)
        h  = cp·T + q
        a² = γ·(p + P∞) / ρ = γ·R·T  (pressure-independent)

    convention='denner' (Denner 2018 Eq.4-7):
        ρ  = (p + γΠ) / (R·T)
        h  = cp₀·T·(p+Π)/(p+γΠ) + q
        a² = γ·(p+Π)/ρ = γ·R·T·(p+Π)/(p+γΠ)  (pressure-dependent)

    When q≠0: d(ρh)/dT ≠ 0 → Newton T-diagonal alive.
    When q=0: d(ρh)/dT = 0 exactly → T-mode ill-conditioned.

    Parameters
    ----------
    gamma : float   heat capacity ratio γ
    pinf  : float   P∞ or Π [Pa]
    cv    : float   specific isochoric heat capacity [J/(kg·K)]
    q     : float   reference energy [J/kg] (default 0)
    convention : str  'standard' or 'denner' (default 'standard')
    """

    def __init__(self, gamma, pinf, cv, q=0.0, convention='standard'):
        self.gamma = gamma
        self.pinf = pinf
        self.cv = cv
        self.q = q
        self.convention = convention
        self._gm1 = gamma - 1.0
        self._cp0 = gamma * cv           # cp₀ = γ·cv
        self._R = self._gm1 * cv         # R = (γ-1)·cv
        self._gPi = gamma * pinf         # γΠ (used in denner convention)

    def _peff(self, p):
        """Effective pressure in density formula: P∞ (standard) or γΠ (denner)."""
        if self.convention == 'denner':
            return p + self._gPi
        return p + self.pinf

    def rho(self, p, T):
        return self._peff(p) / (self._R * T + 1e-300)

    def h(self, p, T):
        if self.convention == 'denner':
            # h = cp₀·T·(p+Π)/(p+γΠ) + q
            return self._cp0 * T * (p + self.pinf) / (self._peff(p) + 1e-300) + self.q
        return self._cp0 * T + self.q

    def c(self, p, T):
        rho_val = self.rho(p, T)
        if self.convention == 'denner':
            return np.sqrt(self.gamma * (p + self.pinf) / (rho_val + 1e-300))
        return np.sqrt(self.gamma * (p + self.pinf) / (rho_val + 1e-300))

    def cp(self, p, T):
        if self.convention == 'denner':
            return self._cp0 * (p + self.pinf) / (self._peff(p) + 1e-300)
        return self._cp0

    def dh_dp(self, p, T):
        if self.convention == 'denner':
            denom = self._peff(p) + 1e-300
            return self._cp0 * T * self._gm1 * self.pinf / (denom * denom)
        return 0.0

    def drho_dp(self, p, T):
        return 1.0 / (self._R * T + 1e-300)

    def drho_dT(self, p, T):
        return -self._peff(p) / (self._R * T * T + 1e-300)

    # e_vol, de_vol_dp, de_vol_dT: use base class defaults (ρh − p identity)


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
    eos_type = ph.get('eos_type', ph.get('type', 'nasg'))
    if eos_type == 'stiffened':
        return StiffenedGasEOS(
            gamma=float(ph['gamma']),
            pinf=float(ph.get('pinf', ph.get('Pi', 0.0))),
            cv=float(ph['cv']),
            q=float(ph.get('q', 0.0)),
            convention=ph.get('convention', 'standard'),
        )
    return NasgEOS(
        gamma=float(ph['gamma']),
        pinf=float(ph.get('pinf', ph.get('p_inf', 0.0))),
        b=float(ph.get('b', ph.get('b_covolume', 0.0))),
        kv=float(ph.get('kappa_v', ph.get('kv', ph.get('cv', ph.get('c_v', 1.0))))),
        eta=float(ph.get('eta', ph.get('q', 0.0))),
    )
