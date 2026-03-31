# Ref: CLAUDE.md § EOS 종류 — NASG EOS
# Ref: docs/APEC_flux.md εᵢ 계산
"""
Noble-Abel Stiffened Gas (NASG) EOS.

State equations:
    p = (gamma-1) * rho * (e - q) / (1 - b*rho) - p_inf
    e = c_v * T + q + p_inf / rho   (approximation used for T recovery)

Internal energy with exact p:
    e = c_v * T + q + p_inf * (1 - b*rho) / ((gamma-1)*rho)

Parameters: gamma, p_inf, b, c_v, q (material constants).

Standard water parameters (Le Metayer et al.):
    gamma = 1.19
    p_inf = 7.028e8  Pa
    b     = 6.61e-4  m^3/kg
    c_v   = 3610.0   J/(kg·K)
    q     = -1.177788e6  J/kg

Standard air parameters (Ideal Gas limit, b=0, p_inf=0):
    gamma = 1.4
    p_inf = 0
    b     = 0
    c_v   = 717.5   J/(kg·K)
    q     = 0
"""

import numpy as np


class NASGEOS:
    """
    Noble-Abel Stiffened Gas (NASG) EOS.

    Parameters
    ----------
    gamma : float
        Polytropic index.
    p_inf : float
        Stiffness pressure parameter [Pa].
    b : float
        Excluded volume (co-volume) [m^3/kg].
    c_v : float
        Specific heat at constant volume [J/(kg·K)].
    q : float
        Reference energy [J/kg].
    """

    def __init__(self, gamma: float, p_inf: float, b: float, c_v: float, q: float):
        self.gamma = float(gamma)
        self.p_inf = float(p_inf)
        self.b = float(b)
        self.c_v = float(c_v)
        self.q = float(q)

    # ------------------------------------------------------------------
    # Basic thermodynamic relations
    # ------------------------------------------------------------------

    def pressure(self, rho: float, T: float) -> float:
        """
        p = (gamma-1) * c_v * rho * T / (1 - b*rho) - p_inf

        Derived from NASG caloric EOS:  e = c_v*T + q + (p+p_inf)*b/(gamma-1)  ... exact form
        Combined with thermal eq:
            p = (gamma-1)*(e-q)*rho/(1-b*rho) - p_inf
        and e = c_v*T + q + ... gives:
            p = (gamma-1)*c_v*rho*T/(1-b*rho) - p_inf
        """
        return (self.gamma - 1.0) * self.c_v * rho * T / (1.0 - self.b * rho) - self.p_inf

    def internal_energy(self, rho: float, T: float) -> float:
        """
        e = c_v * T + q + p_inf*(1 - b*rho) / ((gamma-1)*rho)

        Exact NASG caloric equation of state.
        """
        return self.c_v * T + self.q + self.p_inf * (1.0 - self.b * rho) / ((self.gamma - 1.0) * rho)

    def temperature_from_e(self, rho: float, e: float) -> float:
        """
        T = (e - q - p_inf*(1 - b*rho)/((gamma-1)*rho)) / c_v
        """
        return (e - self.q - self.p_inf * (1.0 - self.b * rho) / ((self.gamma - 1.0) * rho)) / self.c_v

    def temperature_from_rho_p(self, rho: float, p: float) -> float:
        """
        T = (p + p_inf) * (1 - b*rho) / ((gamma-1) * c_v * rho)
        """
        return (p + self.p_inf) * (1.0 - self.b * rho) / ((self.gamma - 1.0) * self.c_v * rho)

    def pressure_from_rho_e(self, rho: float, e: float) -> float:
        """
        p = (gamma-1) * rho * (e - q) / (1 - b*rho) - p_inf
        """
        return (self.gamma - 1.0) * rho * (e - self.q) / (1.0 - self.b * rho) - self.p_inf

    def sound_speed(self, rho: float, T: float) -> float:
        """
        a^2 = gamma * (p + p_inf) / (rho * (1 - b*rho))

        For NASG: a^2 = gamma*(gamma-1)*c_v*T / (1-b*rho)^2
        """
        p = self.pressure(rho, T)
        return np.sqrt(self.gamma * (p + self.p_inf) / (rho * (1.0 - self.b * rho)))

    # ------------------------------------------------------------------
    # Partial derivatives (needed for APEC εᵢ calculation)
    # Ref: docs/APEC_flux.md
    # ------------------------------------------------------------------

    def dp_dT(self, rho: float) -> float:
        """
        (∂p/∂T)_rho = (gamma-1) * c_v * rho / (1 - b*rho)
        """
        return (self.gamma - 1.0) * self.c_v * rho / (1.0 - self.b * rho)

    def dp_drho(self, rho: float, T: float) -> float:
        """
        (∂p/∂rho)_T
        p = (gamma-1)*c_v*rho*T / (1-b*rho) - p_inf
        dp/drho = (gamma-1)*c_v*T * d/drho[rho/(1-b*rho)]
                = (gamma-1)*c_v*T / (1-b*rho)^2
        """
        return (self.gamma - 1.0) * self.c_v * T / (1.0 - self.b * rho) ** 2

    def de_drho_T(self, rho: float) -> float:
        """
        (∂e/∂rho)_T
        e = c_v*T + q + p_inf*(1 - b*rho) / ((gamma-1)*rho)
        de/drho = p_inf * d/drho[(1-b*rho)/rho] / (gamma-1)
                = p_inf * (-b*rho - (1-b*rho)) / ((gamma-1)*rho^2)
                = -p_inf / ((gamma-1)*rho^2)
        """
        return -self.p_inf / ((self.gamma - 1.0) * rho ** 2)

    def drho_e_drho_i_T(self, rho: float, T: float) -> float:
        """
        (∂(rho*e)/∂rho_i)_{rho_j≠i, T}
        = e + rho * (∂e/∂rho)_T
        = (c_v*T + q + p_inf*(1-b*rho)/((gamma-1)*rho))
          + rho * (-p_inf / ((gamma-1)*rho^2))
        = c_v*T + q + p_inf*(1-b*rho)/((gamma-1)*rho) - p_inf/((gamma-1)*rho)
        = c_v*T + q + p_inf*(-b*rho)/((gamma-1)*rho)
        = c_v*T + q - p_inf*b/(gamma-1)
        """
        return self.c_v * T + self.q - self.p_inf * self.b / (self.gamma - 1.0)

    def epsilon_i(self, rho: float, T: float, rho_cv_mix: float, dp_dT_mix: float) -> float:
        """
        Compute εᵢ = (∂ρe/∂ρᵢ)_{ρⱼ≠ᵢ, p}

        Using the chain rule:
            εᵢ = (∂ρe/∂ρᵢ)_T - (ρ Cᵥ_mix / (∂p/∂T)_mix) * (∂p/∂ρᵢ)_T

        Parameters
        ----------
        rho : float
            Mixture density.
        T : float
            Temperature.
        rho_cv_mix : float
            rho * c_v_mix (mixture volumetric heat capacity).
        dp_dT_mix : float
            (∂p/∂T)_rho for the mixture.
        """
        drhoE_drho_i_T = self.drho_e_drho_i_T(rho, T)
        dp_drho_i_T = self.dp_drho(rho, T)
        eps_i = drhoE_drho_i_T - (rho_cv_mix / dp_dT_mix) * dp_drho_i_T
        return eps_i
