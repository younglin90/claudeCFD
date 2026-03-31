# Ref: CLAUDE.md § EOS 종류 — Ideal Gas EOS
# Ref: docs/APEC_flux.md εᵢ 계산
"""
Ideal Gas EOS (Calorically Perfect Gas).

State equations:
    p = rho * R_s * T = rho * (gamma - 1) * e
    e = c_v * T
    c_v = R_s / (gamma - 1)
    R_s = R_u / M   (specific gas constant)

where R_u = 8.314 J/(mol·K) is the universal gas constant.
"""

import numpy as np

R_UNIVERSAL = 8.314462618  # J/(mol·K)


class IdealGasEOS:
    """
    Ideal (Calorically Perfect) Gas EOS.

    Parameters
    ----------
    gamma : float
        Heat capacity ratio c_p/c_v.
    M : float
        Molar mass [g/mol].  Stored internally in kg/mol.
    """

    def __init__(self, gamma: float, M: float):
        self.gamma = float(gamma)
        self.M = float(M) * 1e-3          # g/mol → kg/mol
        self.R_s = R_UNIVERSAL / self.M   # J/(kg·K)
        self.c_v = self.R_s / (self.gamma - 1.0)

    # ------------------------------------------------------------------
    # Basic thermodynamic relations
    # ------------------------------------------------------------------

    def pressure(self, rho: float, T: float) -> float:
        """p = rho * R_s * T"""
        return rho * self.R_s * T

    def pressure_from_rho_e(self, rho: float, e: float) -> float:
        """p = rho * (gamma-1) * e"""
        return rho * (self.gamma - 1.0) * e

    def internal_energy(self, T: float) -> float:
        """e = c_v * T"""
        return self.c_v * T

    def temperature_from_e(self, e: float) -> float:
        """T = e / c_v"""
        return e / self.c_v

    def temperature_from_rho_p(self, rho: float, p: float) -> float:
        """T = p / (rho * R_s)"""
        return p / (rho * self.R_s)

    def sound_speed(self, rho: float, T: float) -> float:
        """a = sqrt(gamma * R_s * T)"""
        return np.sqrt(self.gamma * self.R_s * T)

    def sound_speed_from_rho_p(self, rho: float, p: float) -> float:
        """a = sqrt(gamma * p / rho)"""
        return np.sqrt(self.gamma * p / rho)

    # ------------------------------------------------------------------
    # Partial derivatives (needed for APEC εᵢ calculation)
    # Ref: docs/APEC_flux.md
    # ------------------------------------------------------------------

    def dp_dT(self, rho: float) -> float:
        """(∂p/∂T)_rho = rho * R_s"""
        return rho * self.R_s

    def dp_drho(self, T: float) -> float:
        """(∂p/∂rho)_T = R_s * T"""
        return self.R_s * T

    def de_dT(self) -> float:
        """(∂e/∂T) = c_v"""
        return self.c_v

    def drho_e_drho_i_T(self, rho: float, T: float) -> float:
        """
        (∂(rho*e)/∂rho_i)_{rho_j≠i, T}

        For Ideal Gas with species-specific e_i = c_v_i * T:
            rho * e = sum_i rho_i * e_i(T)
            (∂(rho*e)/∂rho_i)_T = e_i(T) = c_v * T
        """
        return self.c_v * T

    def epsilon_i(self, rho: float, T: float, rho_cv_mix: float, dp_dT_mix: float) -> float:
        """
        Compute εᵢ = (∂ρe/∂ρᵢ)_{ρⱼ≠ᵢ, p}

        Using the chain rule:
            εᵢ = (∂ρe/∂ρᵢ)_T - (ρ Cᵥ_mix / (∂p/∂T)_mix) * (∂p/∂ρᵢ)_T

        Parameters
        ----------
        rho : float
            Mixture density [kg/m^3].
        T : float
            Temperature [K].
        rho_cv_mix : float
            rho * c_v_mix  (mixture volumetric heat capacity) [J/(m^3·K)].
        dp_dT_mix : float
            (∂p/∂T)_rho for the mixture [Pa/K].
        """
        drhoE_drho_i_T = self.drho_e_drho_i_T(rho, T)
        dp_drho_i_T = self.dp_drho(T)
        eps_i = drhoE_drho_i_T - (rho_cv_mix / dp_dT_mix) * dp_drho_i_T
        return eps_i
