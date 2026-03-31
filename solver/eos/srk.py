# Ref: CLAUDE.md § EOS 종류 — SRK EOS
# Ref: docs/APEC_flux.md εᵢ 계산, Eq.(49)~(51)
"""
Soave-Redlich-Kwong (SRK) EOS.

Equation of state:
    p = R_u*T / (v - b)  -  a*alpha(T) / (v*(v + b))

    alpha(T) = [1 + m*(1 - sqrt(T/T_c))]^2
    m        = 0.48 + 1.574*omega - 0.176*omega^2
    a        = 0.42748 * R_u^2 * T_c^2 / p_c
    b        = 0.08664 * R_u * T_c / p_c

where v = 1/rho is specific volume [m^3/mol] ... NOTE: all here in per-mol basis
converted to per-kg via molar mass M.

Applied to: CH4/N2 mixture (transcritical conditions)
Temperature inversion (p,rho) -> T: uses scipy.optimize.brentq

Critical properties:
    CH4: T_c=190.56 K, p_c=4.599e6 Pa, rho_c=162.66 kg/m^3, omega=0.011
    N2:  T_c=126.19 K, p_c=3.396e6 Pa, rho_c=313.3 kg/m^3,  omega=0.037
"""

import numpy as np
from scipy.optimize import brentq

R_UNIVERSAL = 8.314462618  # J/(mol·K)


class SRKEOS:
    """
    Soave-Redlich-Kwong EOS for a pure species.

    Parameters
    ----------
    T_c : float
        Critical temperature [K].
    p_c : float
        Critical pressure [Pa].
    omega : float
        Acentric factor [-].
    M : float
        Molar mass [g/mol].
    c_v0 : float
        Ideal-gas specific heat at constant volume [J/(kg·K)].
        Used for internal energy calculations.
    """

    def __init__(self, T_c: float, p_c: float, omega: float, M: float, c_v0: float):
        self.T_c = float(T_c)
        self.p_c = float(p_c)
        self.omega = float(omega)
        self.M = float(M) * 1e-3          # g/mol → kg/mol
        self.R_s = R_UNIVERSAL / self.M   # J/(kg·K)
        self.c_v0 = float(c_v0)           # J/(kg·K)

        # SRK parameters (molar basis, then converted)
        self._a_mol = 0.42748 * R_UNIVERSAL ** 2 * self.T_c ** 2 / self.p_c  # J·m^3/mol^2
        self._b_mol = 0.08664 * R_UNIVERSAL * self.T_c / self.p_c             # m^3/mol
        self._m = 0.48 + 1.574 * self.omega - 0.176 * self.omega ** 2

        # Convert to per-kg basis
        # v_mol = v_kg * M  =>  a_mol/(v_mol^2) = a_kg/(v_kg^2) with a_kg = a_mol/M^2
        self.a_kg = self._a_mol / self.M ** 2   # J·m^3/kg^2
        self.b_kg = self._b_mol / self.M        # m^3/kg

    # ------------------------------------------------------------------
    # alpha(T) and its derivatives
    # ------------------------------------------------------------------

    def _alpha(self, T: float) -> float:
        """alpha(T) = [1 + m*(1 - sqrt(T/T_c))]^2"""
        sqr = np.sqrt(T / self.T_c)
        return (1.0 + self._m * (1.0 - sqr)) ** 2

    def _dalpha_dT(self, T: float) -> float:
        """d alpha / d T"""
        sqr = np.sqrt(T / self.T_c)
        # dalpha/dT = 2*(1 + m*(1-sqr)) * m * (-1/(2*sqrt(T*T_c)))
        return -self._m * (1.0 + self._m * (1.0 - sqr)) / (np.sqrt(self.T_c * T))

    def _d2alpha_dT2(self, T: float) -> float:
        """d^2 alpha / d T^2"""
        sqr = np.sqrt(T / self.T_c)
        # Let f = 1 + m*(1 - sqr), g = -m/(2*sqrt(T_c*T))
        # alpha = f^2,  dalpha/dT = 2*f*g
        # d2alpha/dT2 = 2*g^2 + 2*f*(dg/dT)
        # dg/dT = m/(4*T_c * (T/T_c)^(3/2)) / T_c = m/(4*T_c^(1/2)*T^(3/2))... re-derive:
        # g = -m/(2*(T_c*T)^(1/2)) = -m/(2*sqrt(T_c)) * T^(-1/2)
        # dg/dT = -m/(2*sqrt(T_c)) * (-1/2) * T^(-3/2) = m/(4*sqrt(T_c)) * T^(-3/2)
        f = 1.0 + self._m * (1.0 - sqr)
        g = -self._m / (2.0 * np.sqrt(self.T_c * T))
        dg_dT = self._m / (4.0 * np.sqrt(self.T_c) * T ** 1.5)
        return 2.0 * g ** 2 + 2.0 * f * dg_dT

    # ------------------------------------------------------------------
    # Basic thermodynamic relations (per-kg basis)
    # ------------------------------------------------------------------

    def pressure(self, rho: float, T: float) -> float:
        """
        p = rho*R_s*T / (1 - b_kg*rho)  -  a_kg*alpha(T)*rho^2 / (1 + b_kg*rho)

        (SRK in specific volume v=1/rho form, converted)
        """
        v = 1.0 / rho
        alp = self._alpha(T)
        # p = R_s*T/(v - b_kg)  -  a_kg*alpha/(v*(v + b_kg))
        return self.R_s * T / (v - self.b_kg) - self.a_kg * alp / (v * (v + self.b_kg))

    def internal_energy(self, rho: float, T: float) -> float:
        """
        Residual + ideal-gas contribution:
            e = e_ig(T) + e_res(rho, T)

        e_ig = c_v0 * T  (reference T=0 is arbitrary, handled by PE cancellation)

        Residual from SRK:
            e_res = integral_{inf}^{rho} [T*(dp/dT)_rho - p] * (-1/rho^2) drho
                  = (a_kg/b_kg) * (alpha - T*dalpha_dT) * ln(1 + b_kg*rho)

        Derivation: de/dv|_T = T*(dp/dT)_v - p
            = T * (R_s/(v-b) - a*dalpha_dT/(v*(v+b))) - (R_s*T/(v-b) - a*alpha/(v*(v+b)))
            = a*(alpha - T*dalpha_dT)/(v*(v+b))
        Integrating from v=inf to v: e_res = (a/b)*( alpha - T*dalpha_dT)*ln((v+b)/v)
                                           = (a/b)*(alpha - T*dalpha_dT)*ln(1 + b/v)
        In per-kg basis with v=1/rho:
            e_res = (a_kg/b_kg)*(alpha - T*dalpha_dT)*ln(1 + b_kg*rho)
        """
        alp = self._alpha(T)
        dalp = self._dalpha_dT(T)
        e_ig = self.c_v0 * T
        e_res = (self.a_kg / self.b_kg) * (alp - T * dalp) * np.log(1.0 + self.b_kg * rho)
        return e_ig + e_res

    def temperature_from_rho_p(self, rho: float, p: float,
                                T_lo: float = 50.0, T_hi: float = 2000.0) -> float:
        """
        Invert p(rho, T) = p  for T using brentq.
        """
        def residual(T):
            return self.pressure(rho, T) - p

        try:
            T = brentq(residual, T_lo, T_hi, xtol=1e-8, maxiter=200)
        except ValueError:
            # Expand bracket if needed
            T_lo2, T_hi2 = 1.0, 5000.0
            T = brentq(residual, T_lo2, T_hi2, xtol=1e-8, maxiter=500)
        return T

    def temperature_from_rho_e(self, rho: float, e: float,
                                T_lo: float = 50.0, T_hi: float = 2000.0) -> float:
        """
        Invert e(rho, T) = e  for T using brentq.
        """
        def residual(T):
            return self.internal_energy(rho, T) - e

        try:
            T = brentq(residual, T_lo, T_hi, xtol=1e-8, maxiter=200)
        except ValueError:
            T_lo2, T_hi2 = 1.0, 5000.0
            T = brentq(residual, T_lo2, T_hi2, xtol=1e-8, maxiter=500)
        return T

    def sound_speed(self, rho: float, T: float) -> float:
        """
        a^2 = -(v^2) * (dp/dv)_s
            = (v^2) * [(dp/dv)_T + T*(dp/dT)^2_v / (rho * cv)]

        Using: a^2 = -v^2*(dp/dv)_T + T*v^2*(dp/dT)^2_v / (c_v)
        """
        v = 1.0 / rho
        alp = self._alpha(T)
        dalp = self._dalpha_dT(T)

        # (dp/dv)_T
        dp_dv_T = (-self.R_s * T / (v - self.b_kg) ** 2
                   + self.a_kg * alp * (2.0 * v + self.b_kg) / (v * (v + self.b_kg)) ** 2)

        # (dp/dT)_v
        dp_dT_v = self.R_s / (v - self.b_kg) - self.a_kg * dalp / (v * (v + self.b_kg))

        # c_v: de/dT at constant v
        d2alp = self._d2alpha_dT2(T)
        c_v = self.c_v0 - (self.a_kg / self.b_kg) * T * d2alp * np.log(1.0 + self.b_kg * rho)

        a2 = -v ** 2 * dp_dv_T + T * v ** 2 * dp_dT_v ** 2 / c_v
        return np.sqrt(max(a2, 0.0))

    # ------------------------------------------------------------------
    # Partial derivatives for APEC εᵢ
    # Ref: docs/APEC_flux.md εᵢ 계산 (Eq.(49)~(51))
    # ------------------------------------------------------------------

    def dp_dT_v(self, rho: float, T: float) -> float:
        """
        (∂p/∂T)_v = R_s/(v - b_kg) - a_kg*dalpha/dT / (v*(v + b_kg))
        """
        v = 1.0 / rho
        dalp = self._dalpha_dT(T)
        return self.R_s / (v - self.b_kg) - self.a_kg * dalp / (v * (v + self.b_kg))

    def dp_drho_T(self, rho: float, T: float) -> float:
        """
        (∂p/∂rho)_T = -(1/rho^2) * (dp/dv)_T converted:

        Direct form from p(rho, T):
        p = rho*R_s*T/(1 - b_kg*rho)  - a_kg*alpha*rho^2/(1 + b_kg*rho)

        dp/drho = R_s*T/(1 - b_kg*rho)^2
                  - a_kg*alpha * (2*rho*(1 + b_kg*rho) - b_kg*rho^2) / (1 + b_kg*rho)^2
        """
        alp = self._alpha(T)
        bk = self.b_kg
        num1 = self.R_s * T / (1.0 - bk * rho) ** 2
        num2 = self.a_kg * alp * (2.0 * rho * (1.0 + bk * rho) - bk * rho ** 2) / (1.0 + bk * rho) ** 2
        return num1 - num2

    def drho_e_drho_i_T(self, rho: float, T: float) -> float:
        """
        (∂(rho*e)/∂rho_i)_{rho_j≠i, T}
        = e + rho * (∂e/∂rho)_T

        (∂e/∂rho)_T from residual:
            e_res = (a_kg/b_kg)*(alpha - T*dalpha)*ln(1 + b_kg*rho)
            de_res/drho = (a_kg/b_kg)*(alpha - T*dalpha) * b_kg/(1 + b_kg*rho)
                        = a_kg*(alpha - T*dalpha) / (1 + b_kg*rho)
            de_ig/drho = 0
        So: (∂e/∂rho)_T = a_kg*(alpha - T*dalpha) / (1 + b_kg*rho)
        """
        alp = self._alpha(T)
        dalp = self._dalpha_dT(T)
        e_val = self.internal_energy(rho, T)
        de_drho_T = self.a_kg * (alp - T * dalp) / (1.0 + self.b_kg * rho)
        return e_val + rho * de_drho_T

    def cv_real(self, rho: float, T: float) -> float:
        """
        c_v = c_v0 - (a_kg/b_kg)*T*d2alpha_dT2 * ln(1 + b_kg*rho)
        """
        d2alp = self._d2alpha_dT2(T)
        return self.c_v0 - (self.a_kg / self.b_kg) * T * d2alp * np.log(1.0 + self.b_kg * rho)

    def epsilon_i(self, rho: float, T: float, rho_cv_mix: float, dp_dT_mix: float) -> float:
        """
        Compute εᵢ = (∂ρe/∂ρᵢ)_{ρⱼ≠ᵢ, p}

        εᵢ = (∂ρe/∂ρᵢ)_T - (ρ Cᵥ_mix / (∂p/∂T)_mix) * (∂p/∂ρᵢ)_T
        """
        drhoE_drho_i_T = self.drho_e_drho_i_T(rho, T)
        dp_drho_i_T = self.dp_drho_T(rho, T)
        eps_i = drhoE_drho_i_T - (rho_cv_mix / dp_dT_mix) * dp_drho_i_T
        return eps_i
