"""General EOS framework with thermodynamic derivatives.

Provides a uniform interface for Stiffened Gas (SG), Noble-Abel Stiffened Gas
(NASG), Ideal Gas, and extensible to Mie-Grüneisen, Cochran-Chan, JWL.

Core API (all functions take cell-wise numpy arrays):
  pressure(rho, e)           -> p
  energy(rho, p)             -> e   (specific internal energy)
  temperature(rho, e)        -> T
  sound_speed_sq(rho, e, p)  -> c²  (for phase k, independent of mixture)
  dpdrho_e(rho, e)           -> (∂p/∂ρ)_e
  dpde_rho(rho, e)           -> (∂p/∂e)_ρ

Thermodynamic identities used:
  Γ_k = (∂p/∂e)_ρ / ρ       (Grüneisen coefficient)
  c²_k = (∂p/∂ρ)_e + Γ_k · p/ρ   (isentropic sound speed squared)

Mixture sound speed (Wood-like) uses individual c²_k from these derivatives,
so the solver is EOS-agnostic: add a new EOS class → everything else works.
"""
import numpy as np
from dataclasses import dataclass, field


# ─── Base protocol (duck-typed via dataclass) ─────────────────────────────
@dataclass
class _EOSBase:
    """Base EOS class (not instantiated directly).

    Note: `name` uses `field(init=False)` so that subclasses' first
    positional arg is `gamma`, not `name`. Previous versions had `name`
    as init field, which caused silent bugs when callers used positional
    args (e.g., `IdealEOS(1.4, 717.5)` set name=1.4, gamma=717.5).
    """
    name: str = field(default='base', init=False)

    def pressure(self, rho, e):
        raise NotImplementedError

    def energy(self, rho, p):
        raise NotImplementedError

    def temperature(self, rho, e):
        raise NotImplementedError

    def sound_speed_sq(self, rho, e, p):
        """c² = (∂p/∂ρ)_e + p/ρ² · (∂p/∂e)_ρ"""
        dpr = self.dpdrho_e(rho, e)
        dpe = self.dpde_rho(rho, e)
        return dpr + p / np.maximum(rho ** 2, 1e-30) * dpe

    def dpdrho_e(self, rho, e):
        raise NotImplementedError

    def dpde_rho(self, rho, e):
        raise NotImplementedError

    # ── Thermodynamic derivatives (optional — override for exact analytic form)
    def cv(self, rho, T):
        """Isochoric heat capacity. Default: self.kv constant."""
        return np.full_like(rho, getattr(self, 'kv', 717.5))

    def dpdT_rho(self, rho, T):
        """(∂p/∂T)_ρ via chain rule: (∂p/∂e)_ρ · cv."""
        p = self.pressure_from_rhoT(rho, T)
        e = self.energy(rho, p)
        return self.dpde_rho(rho, e) * self.cv(rho, T)

    def dpdrho_T(self, rho, T):
        """(∂p/∂ρ)_T via chain rule: (∂p/∂ρ)_e + (∂p/∂e)_ρ · (∂e/∂ρ)_T."""
        p = self.pressure_from_rhoT(rho, T)
        e = self.energy(rho, p)
        dpr_e = self.dpdrho_e(rho, e)
        dpe_r = self.dpde_rho(rho, e)
        dedrho_T = self.dedrho_T(rho, T)
        return dpr_e + dpe_r * dedrho_T

    def dedrho_T(self, rho, T):
        """(∂e/∂ρ)_T. Default numerical FD."""
        eps = 1e-4 * np.maximum(rho, 1.0)
        p1 = self.pressure_from_rhoT(rho + eps, T)
        p2 = self.pressure_from_rhoT(rho - eps, T)
        e1 = self.energy(rho + eps, p1)
        e2 = self.energy(rho - eps, p2)
        return (e1 - e2) / (2.0 * eps)

    def pressure_from_rhoT(self, rho, T):
        """p(ρ, T). Default: Newton from e=cv·T seed."""
        e_seed = self.cv(rho, T) * T
        return self.pressure(rho, e_seed)

    # ── (p, T)-anchored derivatives — sufficient for dU/dW assembly ────────
    # Identities used (constant-T or constant-p partials):
    #   (∂ρ/∂p)_T  = 1 / (∂p/∂ρ)_T
    #   (∂ρ/∂T)_p  = -(∂p/∂T)_ρ / (∂p/∂ρ)_T
    #   (∂e/∂p)_T  = (∂e/∂ρ)_T · (∂ρ/∂p)_T
    #   (∂e/∂T)_p  = cv + (∂e/∂ρ)_T · (∂ρ/∂T)_p
    # Subclasses with closed forms should override for cost / round-off.
    def density(self, p, T):
        """ρ(p, T). Default: 1-D Newton from ideal-gas seed.

        Subclasses with a closed form (Ideal/SG/NASG) override directly.
        Newton on residual r(ρ) = pressure_from_rhoT(ρ, T) − p, with
        derivative dpdrho_T. Always returns ρ > 0.
        """
        rho_arr = np.asarray(np.maximum(p, 1.0)
                             / np.maximum((self.gamma_safe() - 1.0) * self.cv(np.ones_like(p), T) * T, 1e-30),
                             dtype=float)
        for _ in range(20):
            res = self.pressure_from_rhoT(rho_arr, T) - p
            jac = np.maximum(self.dpdrho_T(rho_arr, T), 1e-12)
            step = res / jac
            rho_arr = np.maximum(rho_arr - step, 1e-12)
            if np.max(np.abs(res) / np.maximum(np.abs(p), 1.0)) < 1e-12:
                break
        return rho_arr

    def gamma_safe(self):
        """Effective γ for Newton seed (subclasses can override)."""
        return float(getattr(self, 'gamma', 1.4))

    def drhodp_T(self, rho, T):
        """(∂ρ/∂p)_T = 1 / (∂p/∂ρ)_T."""
        return 1.0 / np.maximum(self.dpdrho_T(rho, T), 1e-30)

    def drhodT_p(self, rho, T):
        """(∂ρ/∂T)_p = -(∂p/∂T)_ρ / (∂p/∂ρ)_T."""
        return -self.dpdT_rho(rho, T) / np.maximum(self.dpdrho_T(rho, T), 1e-30)

    def dedp_T(self, rho, T):
        """(∂e/∂p)_T = (∂e/∂ρ)_T · (∂ρ/∂p)_T."""
        return self.dedrho_T(rho, T) * self.drhodp_T(rho, T)

    def dedT_p(self, rho, T):
        """(∂e/∂T)_p = cv + (∂e/∂ρ)_T · (∂ρ/∂T)_p."""
        return self.cv(rho, T) + self.dedrho_T(rho, T) * self.drhodT_p(rho, T)

    def is_admissible(self, rho, p=None, T=None):
        """Default: ρ > 0. Subclasses can tighten (NASG covolume, RKPR spinodal)."""
        return np.asarray(rho) > 0.0

    # For compatibility with existing `ph['gamma']`, `ph['pinf']` dict access
    def as_dict(self):
        """Return legacy dict form for backward compatibility."""
        return dict(gamma=getattr(self, 'gamma', 1.4),
                    pinf=getattr(self, 'pinf', 0.0),
                    kv=getattr(self, 'kv', 717.5),
                    b=getattr(self, 'b', 0.0),
                    eta=getattr(self, 'eta', 0.0),
                    q=getattr(self, 'q', 0.0))

    def __getitem__(self, key):
        """Dict-like access for backward-compat with ph['gamma'] etc."""
        return getattr(self, key, 0.0)

    def get(self, key, default=None):
        """dict.get() equivalent."""
        return getattr(self, key, default)


# ─── Ideal Gas ────────────────────────────────────────────────────────────
@dataclass
class IdealEOS(_EOSBase):
    """Ideal gas: p = (γ-1) ρ e."""
    gamma: float = 1.4
    kv: float = 717.5  # cv
    name: str = field(default='ideal', init=False)

    # Legacy compat
    pinf: float = 0.0
    b: float = 0.0
    eta: float = 0.0
    q: float = 0.0

    def pressure(self, rho, e):
        return (self.gamma - 1.0) * rho * e

    def energy(self, rho, p):
        return p / ((self.gamma - 1.0) * np.maximum(rho, 1e-30))

    def temperature(self, rho, e):
        return e / self.kv

    def dpdrho_e(self, rho, e):
        return (self.gamma - 1.0) * e

    def dpde_rho(self, rho, e):
        return (self.gamma - 1.0) * rho

    def sound_speed_sq(self, rho, e, p):
        # c² = γ p / ρ (closed-form for ideal)
        return self.gamma * p / np.maximum(rho, 1e-30)

    # Analytic thermodynamic derivatives (Ideal gas)
    def pressure_from_rhoT(self, rho, T):
        return (self.gamma - 1.0) * rho * self.kv * T  # p = (γ-1)ρkvT

    def density(self, p, T):
        """ρ from (p, T) — ideal gas: ρ = p / ((γ-1)·kv·T)"""
        return p / ((self.gamma - 1.0) * self.kv * np.maximum(T, 1.0))

    def dpdT_rho(self, rho, T):
        return (self.gamma - 1.0) * rho * self.kv

    def dpdrho_T(self, rho, T):
        return (self.gamma - 1.0) * self.kv * T

    def dedrho_T(self, rho, T):
        return np.zeros_like(rho)  # e = kv·T independent of ρ for ideal

    # Closed-form (p, T)-derivatives — Ideal: e=kv·T, ρ=p/((γ-1)kv T)
    def drhodp_T(self, rho, T):
        # ∂ρ/∂p|_T = 1 / ((γ-1)·kv·T)
        return 1.0 / ((self.gamma - 1.0) * self.kv * np.maximum(T, 1.0))

    def drhodT_p(self, rho, T):
        # ∂ρ/∂T|_p = -ρ / T  (since ρ ∝ 1/T at fixed p)
        return -rho / np.maximum(T, 1.0)

    def dedp_T(self, rho, T):
        return np.zeros_like(rho)   # e = kv·T independent of p

    def dedT_p(self, rho, T):
        return np.full_like(rho, self.kv)   # e = kv·T

    def is_admissible(self, rho, p=None, T=None):
        """Per-cell boolean array: True where state is physically admissible."""
        rho = np.asarray(rho)
        return rho > 0


# ─── Stiffened Gas (SG) ───────────────────────────────────────────────────
@dataclass
class SGEOS(_EOSBase):
    """Stiffened gas: p = (γ-1) ρ e - γ P∞.

    Equivalent: ρ e = (p + γ P∞)/(γ-1).

    Standard air: γ=1.4, P∞=0 (reduces to ideal).
    Water (Denner 2018): γ=4.1, P∞=4.4e8.
    Water (Yoo & Sung 2018): γ=4.4, P∞=6e8.
    """
    gamma: float = 4.1
    pinf: float = 4.4e8
    kv: float = 474.2
    name: str = field(default='sg', init=False)

    # Legacy compat
    b: float = 0.0
    eta: float = 0.0
    q: float = 0.0

    def pressure(self, rho, e):
        return (self.gamma - 1.0) * rho * e - self.gamma * self.pinf

    def energy(self, rho, p):
        return (p + self.gamma * self.pinf) / ((self.gamma - 1.0) * np.maximum(rho, 1e-30))

    def temperature(self, rho, e):
        # T = (p + P∞) / ((γ-1) ρ kv) = e_thermal / kv
        return (e - self.pinf / np.maximum(rho, 1e-30)) / self.kv

    def dpdrho_e(self, rho, e):
        return (self.gamma - 1.0) * e

    def dpde_rho(self, rho, e):
        return (self.gamma - 1.0) * rho

    def sound_speed_sq(self, rho, e, p):
        # c² = γ (p + P∞) / ρ   (closed-form for SG)
        return self.gamma * (p + self.pinf) / np.maximum(rho, 1e-30)

    # Analytic thermodynamic derivatives (SG)
    def pressure_from_rhoT(self, rho, T):
        # p = (γ-1)ρkvT - P∞
        return (self.gamma - 1.0) * rho * self.kv * T - self.pinf

    def density(self, p, T):
        """ρ from (p, T) — SG: ρ = (p + P∞) / ((γ-1)·kv·T)"""
        return (p + self.pinf) / ((self.gamma - 1.0) * self.kv * np.maximum(T, 1.0))

    def dpdT_rho(self, rho, T):
        return (self.gamma - 1.0) * rho * self.kv

    def dpdrho_T(self, rho, T):
        return (self.gamma - 1.0) * self.kv * T

    def dedrho_T(self, rho, T):
        # e = kv·T + P∞/ρ  →  ∂e/∂ρ|_T = -P∞/ρ²
        return -self.pinf / np.maximum(rho ** 2, 1e-30)

    # Closed-form (p, T)-derivatives — SG: ρ=(p+P∞)/((γ-1)kv T), e=kv T+P∞/ρ
    def drhodp_T(self, rho, T):
        return 1.0 / ((self.gamma - 1.0) * self.kv * np.maximum(T, 1.0))

    def drhodT_p(self, rho, T):
        # ρ = (p+P∞)/((γ-1)kv T) → ∂ρ/∂T|_p = -ρ/T
        return -rho / np.maximum(T, 1.0)

    def dedp_T(self, rho, T):
        # e = kv·T + P∞·(γ-1)kv T/(p+P∞) → ∂e/∂p|_T = -P∞(γ-1)kvT/(p+P∞)²
        # Equivalent: -P∞/ρ² · drhodp_T  (chain rule).
        p = self.pressure_from_rhoT(rho, T)
        pp = np.maximum(p + self.pinf, 1e-30)
        return -self.pinf * (self.gamma - 1.0) * self.kv * T / (pp ** 2)

    def dedT_p(self, rho, T):
        # e = kv·T + P∞/ρ(p,T)
        # ∂e/∂T|_p = kv + P∞·(γ-1)kv/(p+P∞)
        p = self.pressure_from_rhoT(rho, T)
        pp = np.maximum(p + self.pinf, 1e-30)
        return self.kv + self.pinf * (self.gamma - 1.0) * self.kv / pp

    def is_admissible(self, rho, p=None, T=None):
        """Per-cell boolean array: True where state is physically admissible."""
        rho = np.asarray(rho)
        return rho > 0


# ─── Noble-Abel Stiffened Gas (NASG) ──────────────────────────────────────
@dataclass
class NASGEOS(_EOSBase):
    """Noble-Abel stiffened gas: p = (γ-1) ρ (e - q) / (1 - b ρ) - γ P∞.

    Includes covolume b and reference energy q (Le Métayer-Saurel 2016).
    Recovers SG when b=0, q=0.
    """
    gamma: float = 2.35
    pinf: float = 1e9
    kv: float = 943.8
    b: float = 6.61e-4
    eta: float = -1167e3  # aka q_prime
    q: float = 0.0
    name: str = field(default='nasg', init=False)

    def pressure(self, rho, e):
        denom = np.maximum(1.0 - self.b * rho, 1e-10)
        return (self.gamma - 1.0) * rho * (e - self.eta) / denom - self.gamma * self.pinf

    def energy(self, rho, p):
        denom = (self.gamma - 1.0) * np.maximum(rho, 1e-30)
        return (p + self.gamma * self.pinf) * (1.0 - self.b * rho) / denom + self.eta

    def temperature(self, rho, e):
        # T = (e - η - P∞ (1/ρ - b)) / kv  (reversible heat capacity)
        v = 1.0 / np.maximum(rho, 1e-30)
        return (e - self.eta - self.pinf * (v - self.b)) / self.kv

    def dpdrho_e(self, rho, e):
        # ∂p/∂ρ|_e via quotient rule
        num = (self.gamma - 1.0) * (e - self.eta)
        denom = np.maximum(1.0 - self.b * rho, 1e-10)
        # d/dρ [ρ / (1 - bρ)] = 1/(1-bρ) + ρb/(1-bρ)² = 1/(1-bρ)²
        return num / denom ** 2

    def dpde_rho(self, rho, e):
        denom = np.maximum(1.0 - self.b * rho, 1e-10)
        return (self.gamma - 1.0) * rho / denom

    def sound_speed_sq(self, rho, e, p):
        """NASG analytic c²: γ(p + P∞) / (ρ(1 - bρ)).
        Ref: Le Métayer & Saurel 2016 Eq. (A.7).
        Overrides base-class finite-difference formula to avoid round-off
        in the stiff co-volume denominator.
        """
        denom = np.maximum(rho * (1.0 - self.b * rho), 1e-30)
        return self.gamma * (p + self.pinf) / denom

    # Analytic thermodynamic derivatives (NASG)
    def pressure_from_rhoT(self, rho, T):
        # p = (γ-1)ρkvT / (1 - bρ) - P∞
        denom = np.maximum(1.0 - self.b * rho, 1e-10)
        return (self.gamma - 1.0) * rho * self.kv * T / denom - self.pinf

    def density(self, p, T):
        """ρ from (p, T) — NASG: ρ = (p + P∞) / ((γ-1)·kv·T + b·(p + P∞))
        Closed-form (no iteration). Always admissible: bρ ≤ 1 by construction
        since bρ = b(p+P∞)/((γ-1)kv·T + b(p+P∞)) < 1.
        """
        pp = p + self.pinf
        denom = (self.gamma - 1.0) * self.kv * np.maximum(T, 1.0) + self.b * pp
        return pp / np.maximum(denom, 1e-30)

    def dpdT_rho(self, rho, T):
        denom = np.maximum(1.0 - self.b * rho, 1e-10)
        return (self.gamma - 1.0) * rho * self.kv / denom

    def dpdrho_T(self, rho, T):
        # d/dρ [ρ/(1-bρ)] = 1/(1-bρ)² → multiplied by (γ-1)·kv·T
        denom = np.maximum(1.0 - self.b * rho, 1e-10)
        return (self.gamma - 1.0) * self.kv * T / denom ** 2

    def dedrho_T(self, rho, T):
        # e = kv·T + η + P∞(1/ρ - b)  →  ∂e/∂ρ|_T = -P∞/ρ²
        return -self.pinf / np.maximum(rho ** 2, 1e-30)

    # Closed-form (p, T)-derivatives — NASG:
    #   1/ρ = (γ-1)·kv·T / (p+P∞) + b   ⇒ ρ = (p+P∞)/((γ-1)kv T + b(p+P∞))
    #   e   = kv·T + η + P∞·(1/ρ − b)
    def drhodp_T(self, rho, T):
        # d(1/ρ)/dp|_T = -(γ-1)kv·T / (p+P∞)²  ⇒ dρ/dp = ρ²·(γ-1)kv·T/(p+P∞)²
        p = self.pressure_from_rhoT(rho, T)
        pp = np.maximum(p + self.pinf, 1e-30)
        return rho ** 2 * (self.gamma - 1.0) * self.kv * T / (pp ** 2)

    def drhodT_p(self, rho, T):
        # d(1/ρ)/dT|_p = (γ-1)kv/(p+P∞)  ⇒ dρ/dT = -ρ²·(γ-1)kv/(p+P∞)
        p = self.pressure_from_rhoT(rho, T)
        pp = np.maximum(p + self.pinf, 1e-30)
        return -rho ** 2 * (self.gamma - 1.0) * self.kv / pp

    def dedp_T(self, rho, T):
        # e depends on p only through ρ:  e = kv·T + η + P∞·((γ-1)kvT/(p+P∞))
        # ⇒ ∂e/∂p|_T = -P∞·(γ-1)kvT/(p+P∞)²
        p = self.pressure_from_rhoT(rho, T)
        pp = np.maximum(p + self.pinf, 1e-30)
        return -self.pinf * (self.gamma - 1.0) * self.kv * T / (pp ** 2)

    def dedT_p(self, rho, T):
        # e = kv·T + η + P∞·(γ-1)kvT/(p+P∞)  ⇒ ∂e/∂T|_p = kv·(p+γP∞)/(p+P∞)
        p = self.pressure_from_rhoT(rho, T)
        pp = np.maximum(p + self.pinf, 1e-30)
        return self.kv * (p + self.gamma * self.pinf) / pp

    def is_admissible(self, rho, p=None, T=None):
        """Per-cell boolean array: True where state is physically admissible.
        NASG requires ρ > 0 and b·ρ < 0.95 (excludes covolume singularity).
        """
        rho = np.asarray(rho)
        return (rho > 0) & (self.b * rho < 0.95)


# ─── Mie-Grüneisen (generic) ──────────────────────────────────────────────
@dataclass
class MieGruneisenEOS(_EOSBase):
    """Mie-Grüneisen: p = p_ref(ρ) + Γ ρ (e - e_ref(ρ)).

    Simple form with constant Γ and power-law reference curves.
    For validation against SG/NASG, user supplies callable `p_ref_fn`, `e_ref_fn`.
    """
    Gamma_G: float = 0.4  # Grüneisen coefficient
    rho_ref: float = 1000.0
    p_ref_coef: float = 0.0  # p_ref = p_ref_coef · (ρ/ρ_ref)^n
    p_ref_n: float = 7.0
    e_ref_coef: float = 0.0
    kv: float = 474.2
    name: str = field(default='mg', init=False)

    # Legacy compat
    gamma: float = 1.4
    pinf: float = 0.0
    b: float = 0.0
    eta: float = 0.0
    q: float = 0.0

    def _p_ref(self, rho):
        return self.p_ref_coef * (rho / self.rho_ref) ** self.p_ref_n

    def _dp_ref_drho(self, rho):
        return (self.p_ref_coef * self.p_ref_n / self.rho_ref
                * (rho / self.rho_ref) ** (self.p_ref_n - 1.0))

    def _e_ref(self, rho):
        return self.e_ref_coef * (rho / self.rho_ref) ** (self.p_ref_n - 1.0)

    def _de_ref_drho(self, rho):
        return (self.e_ref_coef * (self.p_ref_n - 1.0) / self.rho_ref
                * (rho / self.rho_ref) ** (self.p_ref_n - 2.0))

    def pressure(self, rho, e):
        return self._p_ref(rho) + self.Gamma_G * rho * (e - self._e_ref(rho))

    def energy(self, rho, p):
        return (p - self._p_ref(rho)) / (self.Gamma_G * np.maximum(rho, 1e-30)) + self._e_ref(rho)

    def temperature(self, rho, e):
        return (e - self._e_ref(rho)) / self.kv

    def dpdrho_e(self, rho, e):
        return (self._dp_ref_drho(rho)
                + self.Gamma_G * (e - self._e_ref(rho))
                - self.Gamma_G * rho * self._de_ref_drho(rho))

    def dpde_rho(self, rho, e):
        return self.Gamma_G * rho


# ─── JWL (Jones-Wilkins-Lee) EOS — 폭약 products ────────────────────────
@dataclass
class JWLEOS(_EOSBase):
    """Jones-Wilkins-Lee EOS for detonation products.

    Normalized form (v_norm = ρ₀/ρ):
      p = A(1 - ω/(R₁·V))·e^(-R₁·V) + B(1 - ω/(R₂·V))·e^(-R₂·V) + ω·ρ·e

    Grüneisen coefficient Γ = ω (constant). Linear in e at fixed ρ → inversion
    trivial: e = (p - p_ref(ρ)) / (ω·ρ)

    TNT (Lee-Tarver LLNL 1973):
      A = 3.712e11, B = 3.231e9, R₁ = 4.15, R₂ = 0.95, ω = 0.3, ρ₀ = 1630
      Q (detonation energy) = 7.0e6 J/kg  (used for initialization only)
    """
    A: float = 3.712e11     # Pa
    B: float = 3.231e9      # Pa
    R1: float = 4.15
    R2: float = 0.95
    omega: float = 0.3
    rho0: float = 1630.0    # kg/m³  (initial products density)
    Q: float = 7.0e6        # J/kg   (detonation energy, IC only)
    kv: float = 1000.0      # approximate cv for temperature diag
    name: str = field(default='jwl', init=False)

    # Legacy compat
    gamma: float = 1.3       # γ ≈ ω+1 in high-pressure regime
    pinf: float = 0.0
    b: float = 0.0
    eta: float = 0.0
    q: float = 0.0

    def _V(self, rho):
        """Normalized volume V = ρ₀/ρ."""
        return self.rho0 / np.maximum(rho, 1e-30)

    def _p_ref(self, rho):
        """Reference (isentropic) pressure — ω·ρ·e 제외한 부분."""
        V = self._V(rho)
        return (self.A * (1.0 - self.omega / (self.R1 * V)) * np.exp(-self.R1 * V)
                + self.B * (1.0 - self.omega / (self.R2 * V)) * np.exp(-self.R2 * V))

    def pressure(self, rho, e):
        return self._p_ref(rho) + self.omega * rho * e

    def energy(self, rho, p):
        return (p - self._p_ref(rho)) / (self.omega * np.maximum(rho, 1e-30))

    def temperature(self, rho, e):
        """Crude: T = (e - e_ref) / kv (Gruneisen form)."""
        return e / self.kv  # simplified; actual T requires cold-curve integration

    def dpdrho_e(self, rho, e):
        # ∂p/∂ρ|_e = d/dρ p_ref(ρ) + ω·e
        V = self._V(rho)
        # dV/dρ = -ρ₀/ρ²
        dV_drho = -self.rho0 / np.maximum(rho ** 2, 1e-30)
        # dp_ref/dρ = dp_ref/dV · dV/dρ
        # dp_ref/dV: chain rule through exp and (1 - ω/(R·V))
        term1 = self.A * np.exp(-self.R1 * V) * (
            self.omega / (self.R1 * V ** 2) - self.R1 * (1.0 - self.omega / (self.R1 * V)))
        term2 = self.B * np.exp(-self.R2 * V) * (
            self.omega / (self.R2 * V ** 2) - self.R2 * (1.0 - self.omega / (self.R2 * V)))
        dp_ref_dV = term1 + term2
        return dp_ref_dV * dV_drho + self.omega * e

    def dpde_rho(self, rho, e):
        return self.omega * rho

    def sound_speed_sq(self, rho, e, p):
        return self.dpdrho_e(rho, e) + p / np.maximum(rho ** 2, 1e-30) * self.dpde_rho(rho, e)


# ─── Peng-Robinson / RKPR (Cubic EOS) ─────────────────────────────────────
@dataclass
class RKPREOS(_EOSBase):
    """Redlich-Kwong-Peng-Robinson generalized cubic EOS.

    p = RT/(v-b) - a(T)/[(v+δ₁b)(v+δ₂b)]

    Peng-Robinson special case: δ₁ = 1+√2, δ₂ = 1-√2.
    General RKPR (Cismondi-Mollerup 2005): δ₁, δ₂ chosen from acentric factor ω.

    **현재 구현 범위**:
    - **PR form** (δ₁=1+√2, δ₂=1-√2) 고정
    - Ideal-gas reference enthalpy: h_ideal(T) = cp_ref · T (constant cp approx)
    - a(T) = a_c · α(T)  where α(T) = [1 + κ(1 - √(T/Tc))]²,  κ = 0.37464 + 1.54226·ω - 0.26992·ω²
    - Multi-root: Cardano analytic roots → branch='gas'|'liquid' selection

    **제한**:
    - cp 상수 근사 (실제 cp(T) 변동은 미포함 — 저온에서 오차)
    - spinodal 근처 Newton 실패 시 Brent fallback (`mixture_pressure_solve`에서 처리)
    """
    Tc: float = 304.13        # 임계 온도 [K]
    pc: float = 7.377e6       # 임계 압력 [Pa]
    omega: float = 0.225      # acentric factor (CO₂)
    M: float = 44.01e-3       # 분자량 [kg/mol]
    cp_ref: float = 846.0     # 상수 cp [J/(kg·K)] (CO₂ 표준)
    kv: float = 657.0         # cv ≈ cp - R/M (CO₂)
    branch: str = 'gas'       # 'gas' | 'liquid' | 'auto'
    name: str = field(default='rkpr', init=False)

    # Legacy compat
    gamma: float = 1.28
    pinf: float = 0.0
    b: float = 0.0
    eta: float = 0.0
    q: float = 0.0

    # Derived constants (computed in __post_init__)
    _R_spec: float = 0.0      # specific gas constant R/M
    _a_c: float = 0.0         # Peng-Robinson a_c
    _b_PR: float = 0.0        # Peng-Robinson b
    _kappa: float = 0.0       # α(T) 계수

    def __post_init__(self):
        R_univ = 8.3145
        self._R_spec = R_univ / self.M  # J/(kg·K)
        # Peng-Robinson
        self._a_c = 0.45724 * self._R_spec ** 2 * self.Tc ** 2 / self.pc
        self._b_PR = 0.07780 * self._R_spec * self.Tc / self.pc
        self._kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega ** 2

    def _alpha_T(self, T):
        return (1.0 + self._kappa * (1.0 - np.sqrt(T / self.Tc))) ** 2

    def _a_of_T(self, T):
        return self._a_c * self._alpha_T(T)

    # ── Primary queries (ρ, T) centered ─────────────────────────────
    def pressure_from_rhoT(self, rho, T):
        v = 1.0 / np.maximum(rho, 1e-30)  # specific volume [m³/kg]
        b = self._b_PR
        a = self._a_of_T(T)
        d1, d2 = 1.0 + np.sqrt(2.0), 1.0 - np.sqrt(2.0)
        return self._R_spec * T / np.maximum(v - b, 1e-30) - a / ((v + d1 * b) * (v + d2 * b))

    def temperature(self, rho, e):
        # e = cp_ref·T - R·T + departure (ideal part + cubic correction)
        # Approximation: solve Newton for T such that e = cv·T + e_dep(ρ, T)
        # e_dep(ρ, T) = -(a/(v·(δ₁-δ₂)·b))·ln((v+δ₁b)/(v+δ₂b)) + T·da/dT · similar
        # For simplicity: e_ideal ≈ cv·T (constant cv) + bounded departure term
        # Newton initialize from e/cv guess
        T = np.maximum(e / self.kv, 100.0)
        for _ in range(5):
            e_dep = self._departure_e(rho, T)
            resid = self.kv * T + e_dep - e
            d_resid = self.kv  # approx: ignore d(e_dep)/dT for simplicity
            T = T - resid / d_resid
            T = np.maximum(T, 100.0)
        return T

    def _departure_e(self, rho, T):
        """Departure internal energy (real − ideal): closed form PR."""
        v = 1.0 / np.maximum(rho, 1e-30)
        b = self._b_PR
        d1, d2 = 1.0 + np.sqrt(2.0), 1.0 - np.sqrt(2.0)
        a = self._a_of_T(T)
        # d(alpha)/dT
        kap = self._kappa
        sqrtT_Tc = np.sqrt(T / self.Tc)
        dalpha_dT = 2.0 * (1.0 + kap * (1.0 - sqrtT_Tc)) * (-kap / (2.0 * np.sqrt(T * self.Tc)))
        da_dT = self._a_c * dalpha_dT
        # e_dep = (T·da/dT − a) / (b·(δ₁-δ₂)) · ln((v+δ₁b)/(v+δ₂b))
        log_term = np.log(np.maximum((v + d1 * b) / np.maximum(v + d2 * b, 1e-30), 1e-30))
        return (T * da_dT - a) / (b * (d1 - d2)) * log_term

    def pressure(self, rho, e):
        T = self.temperature(rho, e)
        return self.pressure_from_rhoT(rho, T)

    def energy(self, rho, p):
        """(ρ, p) → e via Newton on T_from_rhoP + e = cv·T + e_dep."""
        T = self._T_from_rhoP(rho, p)
        return self.kv * T + self._departure_e(rho, T)

    def _T_from_rhoP(self, rho, p):
        """Newton solve p(ρ, T) = p_target."""
        T = np.maximum(p / (rho * self._R_spec), 100.0)  # ideal gas guess
        for _ in range(8):
            p_cur = self.pressure_from_rhoT(rho, T)
            # ∂p/∂T|_ρ  (analytic)
            v = 1.0 / np.maximum(rho, 1e-30)
            b = self._b_PR
            d1, d2 = 1.0 + np.sqrt(2.0), 1.0 - np.sqrt(2.0)
            kap = self._kappa
            sqrtT_Tc = np.sqrt(T / self.Tc)
            dalpha_dT = 2.0 * (1.0 + kap * (1.0 - sqrtT_Tc)) * (-kap / (2.0 * np.sqrt(T * self.Tc)))
            da_dT = self._a_c * dalpha_dT
            dp_dT = self._R_spec / np.maximum(v - b, 1e-30) - da_dT / ((v + d1 * b) * (v + d2 * b))
            T = T - (p_cur - p) / np.maximum(np.abs(dp_dT), 1e-10) * np.sign(dp_dT + 1e-10)
            T = np.maximum(T, 100.0)
        return T

    def sound_speed_sq(self, rho, e, p):
        """c² via generic (∂p/∂ρ)_e + p/ρ² · (∂p/∂e)_ρ."""
        return self.dpdrho_e(rho, e) + p / np.maximum(rho ** 2, 1e-30) * self.dpde_rho(rho, e)

    def dpdrho_e(self, rho, e):
        """(∂p/∂ρ)_e via numerical FD (analytic too complex)."""
        eps = np.maximum(rho * 1e-5, 1.0)
        p1 = self.pressure(rho + eps, e)
        p2 = self.pressure(rho - eps, e)
        return (p1 - p2) / (2.0 * eps)

    def dpde_rho(self, rho, e):
        """(∂p/∂e)_ρ via FD."""
        eps = np.maximum(np.abs(e) * 1e-5, 1.0)
        p1 = self.pressure(rho, e + eps)
        p2 = self.pressure(rho, e - eps)
        return (p1 - p2) / (2.0 * eps)


# ─── Convenience factory ──────────────────────────────────────────────────
def make_eos(kind='sg', **kwargs):
    """Factory: make_eos('sg', gamma=4.1, pinf=4.4e8, kv=474.2)."""
    kind = kind.lower()
    if kind in ('ideal', 'gas'):
        return IdealEOS(**kwargs)
    elif kind in ('sg', 'stiffened', 'stiffened_gas'):
        return SGEOS(**kwargs)
    elif kind in ('nasg', 'noble_abel_stiffened_gas'):
        return NASGEOS(**kwargs)
    elif kind in ('mg', 'mie_gruneisen'):
        return MieGruneisenEOS(**kwargs)
    elif kind in ('rkpr', 'pr', 'peng_robinson'):
        return RKPREOS(**kwargs)
    elif kind in ('jwl', 'jones_wilkins_lee'):
        return JWLEOS(**kwargs)
    else:
        raise ValueError(f'Unknown EOS kind: {kind}')


def to_eos(ph):
    """Convert legacy dict `ph` with gamma/pinf/kv/b/eta/q to EOS object.

    Heuristic:
      b > 0 or eta != 0  →  NASG
      pinf > 0           →  SG
      else               →  Ideal
    """
    if isinstance(ph, _EOSBase):
        return ph
    g = ph.get('gamma', 1.4)
    pinf = ph.get('pinf', 0.0)
    kv = ph.get('kv', 717.5)
    b = ph.get('b', 0.0)
    eta = ph.get('eta', 0.0)
    q = ph.get('q', 0.0)
    if b > 0.0 or abs(eta) > 0.0:
        return NASGEOS(gamma=g, pinf=pinf, kv=kv, b=b, eta=eta, q=q)
    elif pinf > 0.0:
        return SGEOS(gamma=g, pinf=pinf, kv=kv)
    else:
        return IdealEOS(gamma=g, kv=kv)


def mixture_sound_speed_sq(a1, rho1, e1, p1, eos1,
                            rho2, e2, p2, eos2):
    """Wood's mixture sound speed squared: 1/(ρ c²) = Σ α_k / (ρ_k c²_k).

    Uses phase-specific c²_k from EOS (general thermodynamic derivatives).
    """
    c1_sq = eos1.sound_speed_sq(rho1, e1, p1)
    c2_sq = eos2.sound_speed_sq(rho2, e2, p2)
    a2 = 1.0 - a1
    rho = a1 * rho1 + a2 * rho2
    # 1/(ρc²) = α₁/(ρ₁c₁²) + α₂/(ρ₂c₂²)
    inv_rhoc2 = a1 / np.maximum(rho1 * c1_sq, 1e-30) + a2 / np.maximum(rho2 * c2_sq, 1e-30)
    return 1.0 / (np.maximum(rho, 1e-30) * inv_rhoc2), c1_sq, c2_sq


# ─── Mixture pressure closure (Kapila p-equilibrium) ──────────────────────
def _is_linear_in_p(eos):
    """True if eos.energy(ρ, p) is linear in p (Ideal, SG, NASG, MG, JWL).

    RKPR/cubic EOS is NOT linear → Newton/Brent fallback in mixture solver.
    """
    return isinstance(eos, (IdealEOS, SGEOS, NASGEOS, MieGruneisenEOS, JWLEOS))


def _linear_mixture_pressure(a1, rho1, rho2, rho_e, eos1, eos2):
    """Direct solve for Ideal/SG/NASG/MG combinations.

    For each phase: e_k = A_k(ρ_k)·p + B_k(ρ_k)
      - Ideal:  e = p/((γ-1)ρ)       → A=1/((γ-1)ρ), B=0
      - SG:     e = (p + γP∞)/((γ-1)ρ) → A=1/((γ-1)ρ), B=γP∞/((γ-1)ρ)
      - NASG:   e = (p + γP∞)(1-bρ)/((γ-1)ρ) + η → A=(1-bρ)/((γ-1)ρ), B=γP∞(1-bρ)/((γ-1)ρ)+η
      - MG:     e = (p - p_ref)/(Γρ) + e_ref → A=1/(Γρ), B=-p_ref/(Γρ)+e_ref

    Mixture: ρe = α₁ρ₁·e₁ + α₂ρ₂·e₂
           = (α₁ρ₁·A₁ + α₂ρ₂·A₂)·p + (α₁ρ₁·B₁ + α₂ρ₂·B₂)
    → p = (ρe - α₁ρ₁B₁ - α₂ρ₂B₂) / (α₁ρ₁A₁ + α₂ρ₂A₂)
    """
    a2 = 1.0 - a1
    A1, B1 = _linear_coeffs(rho1, eos1)
    A2, B2 = _linear_coeffs(rho2, eos2)
    num = rho_e - a1 * rho1 * B1 - a2 * rho2 * B2
    den = a1 * rho1 * A1 + a2 * rho2 * A2
    return num / np.maximum(np.abs(den), 1e-30) * np.sign(den + 1e-30)


def _linear_coeffs(rho, eos):
    """Return (A, B) such that e(ρ, p) = A·p + B for linear-in-p EOS."""
    rho_safe = np.maximum(rho, 1e-30)
    if isinstance(eos, IdealEOS):
        A = 1.0 / ((eos.gamma - 1.0) * rho_safe)
        B = np.zeros_like(rho) if hasattr(rho, 'shape') else 0.0
        return A, B
    elif isinstance(eos, SGEOS):
        A = 1.0 / ((eos.gamma - 1.0) * rho_safe)
        B = eos.gamma * eos.pinf / ((eos.gamma - 1.0) * rho_safe)
        return A, B
    elif isinstance(eos, NASGEOS):
        one_minus_brho = 1.0 - eos.b * rho
        A = one_minus_brho / ((eos.gamma - 1.0) * rho_safe)
        B = eos.gamma * eos.pinf * one_minus_brho / ((eos.gamma - 1.0) * rho_safe) + eos.eta
        return A, B
    elif isinstance(eos, MieGruneisenEOS):
        A = 1.0 / (eos.Gamma_G * rho_safe)
        B = -eos._p_ref(rho) / (eos.Gamma_G * rho_safe) + eos._e_ref(rho)
        return A, B
    elif isinstance(eos, JWLEOS):
        # e = (p - p_ref(ρ)) / (ω·ρ)  →  A = 1/(ω·ρ), B = -p_ref(ρ)/(ω·ρ)
        A = 1.0 / (eos.omega * rho_safe)
        B = -eos._p_ref(rho) / (eos.omega * rho_safe)
        return A, B
    else:
        raise ValueError(f'No linear coeffs for EOS: {type(eos).__name__}')


def _residual(p, a1, rho1, rho2, rho_e, eos1, eos2):
    """R(p) = α₁ρ₁·e₁(ρ₁,p) + α₂ρ₂·e₂(ρ₂,p) - ρe."""
    e1 = eos1.energy(rho1, p)
    e2 = eos2.energy(rho2, p)
    return a1 * rho1 * e1 + (1.0 - a1) * rho2 * e2 - rho_e


def _residual_and_jac(p, a1, rho1, rho2, rho_e, eos1, eos2, dp_rel=1e-6):
    """R(p) and numerical dR/dp (robust for any EOS)."""
    R = _residual(p, a1, rho1, rho2, rho_e, eos1, eos2)
    dp = np.maximum(np.abs(p) * dp_rel, 1.0)
    Rp = _residual(p + dp, a1, rho1, rho2, rho_e, eos1, eos2)
    dR = (Rp - R) / dp
    # Avoid zero Jacobian
    dR = np.where(np.abs(dR) < 1e-20, 1.0, dR)
    return R, dR


def _brent_root(residual_fn, p_lo, p_hi, tol=1e-10, max_iter=50):
    """Brent's method for scalar root-finding (EOS-agnostic robust fallback).

    Works element-wise on numpy arrays via vectorized implementation.
    """
    a = np.asarray(p_lo, dtype=float).copy()
    b = np.asarray(p_hi, dtype=float).copy()
    fa = residual_fn(a)
    fb = residual_fn(b)
    # Ensure bracket
    bad = fa * fb > 0
    if np.any(bad):
        # Widen and retry once
        a = np.where(bad, a * 0.01 - 1e5, a)
        b = np.where(bad, b * 100.0 + 1e10, b)
        fa = residual_fn(a)
        fb = residual_fn(b)
    # Vectorized simple bisection (Brent-like not strictly needed for 1D Kapila)
    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = residual_fn(c)
        sign_ac = fa * fc
        # c replaces a or b based on sign
        a = np.where(sign_ac < 0, a, c)
        fa = np.where(sign_ac < 0, fa, fc)
        b = np.where(sign_ac < 0, c, b)
        fb = np.where(sign_ac < 0, fc, fb)
        if np.max(np.abs(b - a)) < tol:
            break
    return 0.5 * (a + b)


def mixture_pressure_solve(a1, rho1, rho2, rho_e, eos1, eos2,
                            p_guess=None, newton_tol=1e-10, max_newton=3):
    """Kapila p-equilibrium: solve Σ α_k ρ_k e_k(ρ_k, p) = ρe for p.

    Strategy:
      1. If both EOS are linear in p (Ideal/SG/NASG/MG) → direct analytic solve.
      2. Else Newton (max 3 iterations) with SG-like initial guess.
      3. If Newton fails to converge → Brent bisection (globally convergent).

    Maintains bit-exact regression for SG/NASG by using the linear fast path
    identically to previous SG hardcode closure.
    """
    # Linear fast path: both EOS linear in p
    if _is_linear_in_p(eos1) and _is_linear_in_p(eos2):
        return _linear_mixture_pressure(a1, rho1, rho2, rho_e, eos1, eos2)

    # Newton with linear-EOS warm start
    if p_guess is None:
        # Use Ideal-like guess if cubic EOS involved
        a2 = 1.0 - a1
        gm1 = getattr(eos1, 'gamma', 1.4) - 1.0
        gm2 = getattr(eos2, 'gamma', 1.4) - 1.0
        Gamma_inv = a1 / max(gm1, 0.1) + a2 / max(gm2, 0.1)
        pinf1 = getattr(eos1, 'pinf', 0.0)
        pinf2 = getattr(eos2, 'pinf', 0.0)
        Pi = a1 * (gm1 + 1) * pinf1 / max(gm1, 0.1) + a2 * (gm2 + 1) * pinf2 / max(gm2, 0.1)
        p_guess = (rho_e - Pi) / np.maximum(Gamma_inv, 1e-30)

    p = np.asarray(p_guess, dtype=float).copy()
    for it in range(max_newton):
        R, dR = _residual_and_jac(p, a1, rho1, rho2, rho_e, eos1, eos2)
        rel = np.max(np.abs(R) / np.maximum(np.abs(rho_e), 1.0))
        if rel < newton_tol:
            return p
        dp = -R / dR
        # Line search: limit step size to avoid negative-p regions
        p_new = p + dp
        # Simple backtracking if |dp| too large
        over = np.abs(dp) > np.maximum(np.abs(p) * 10.0, 1e6)
        dp = np.where(over, dp * 0.5, dp)
        p = p + dp

    # Newton did not converge — fallback to Brent bracketing
    def _res_fn(p_test):
        return _residual(p_test, a1, rho1, rho2, rho_e, eos1, eos2)

    p_lo = np.minimum(p - np.abs(p), -1e7)  # allow negative tension
    p_hi = np.maximum(p + np.abs(p) * 10.0, 1e10)
    return _brent_root(_res_fn, p_lo, p_hi, tol=newton_tol, max_iter=50)


def pressure_temperature_relaxation(a1, rho1, rho2, rho_e, eos1, eos2,
                                      max_iter=20, tol=1e-10):
    """Saurel 2007 instantaneous p-T relaxation for 2-phase Kapila.

    Input: current (α_k, ρ_k, ρe) possibly out-of-equilibrium (T_k ≠ T_mix,
           phase densities outside EOS admissibility).
    Output: (α_k_relaxed, ρ_k_relaxed, p_relaxed, T_relaxed) such that:
        - p_k_relaxed = p_relaxed ∀ k  (pressure equilibrium — already Kapila)
        - T_k_relaxed = T_relaxed ∀ k  (temperature equilibrium)
        - Σ α_k ρ_k = ρ (total mass conserved)
        - Σ α_k ρ_k e_k(ρ_k, p, T) = ρe (total energy conserved)
        - ρ_k = eos_k.density(p, T) (EOS-admissible)

    Algorithm (Saurel 2007 Sec. 3):
      1. Guess T from current state
      2. For each k, compute ρ_k_eos = eos_k.density(p, T)
      3. New α_k such that α_k·ρ_k_eos conserves α_kρ_k (mass)
        → α_k_new = α_k·ρ_k_old / ρ_k_eos
      4. Check: Σ α_k_new = 1 and Σ α_k_new·ρ_k_eos·e_k(ρ_k_eos, T) = ρe
      5. If not, Newton iterate on (p, T)

    For pure Kapila (p already mixture-consistent), only T equilibrium needed.
    """
    a2 = 1.0 - a1
    rho_total = a1 * rho1 + a2 * rho2

    # Initial T guess: mass-weighted from each phase's current T
    T1_guess = eos1.temperature(rho1, eos1.energy(rho1, np.maximum(
        (rho_e / np.maximum(a1 + a2, _EPS)),  1.0))) if a1.max() > 0 else np.ones_like(rho1) * 300.0
    T_guess = np.maximum(np.where(a1 >= 0.5, T1_guess,
                                   eos2.temperature(rho2, eos2.energy(rho2, 1e5))),
                          1.0)

    p = mixture_pressure_solve(a1, rho1, rho2, rho_e, eos1, eos2)

    # Iterate: find (p, T) such that:
    # Σ α_k ρ_k(p,T) e_k(ρ_k(p,T), p) = ρe  (energy conservation)
    # Σ α_k = 1  (volume closure — automatic)
    # Phase masses: α_k·ρ_k(p,T) = α_k_old·ρ_k_old (each phase mass fixed)
    # → α_k_new = α_k_old·ρ_k_old / ρ_k(p,T)
    m_a = a1 * rho1  # phase 1 partial mass
    m_b = a2 * rho2  # phase 2 partial mass

    p_cur = p.copy()
    T_cur = T_guess.copy()
    for it in range(max_iter):
        # EOS-admissible phase densities
        try:
            rho1_new = eos1.density(p_cur, T_cur)
            rho2_new = eos2.density(p_cur, T_cur)
        except (AttributeError, NotImplementedError):
            return a1, rho1, rho2, p_cur, T_cur

        rho1_new = np.maximum(rho1_new, _EPS)
        rho2_new = np.maximum(rho2_new, _EPS)
        # New volume fractions
        a1_new = m_a / rho1_new
        a2_new = m_b / rho2_new
        a_sum = a1_new + a2_new
        # Force sum = 1: rescale
        a1_new = a1_new / np.maximum(a_sum, _EPS)
        a2_new = a2_new / np.maximum(a_sum, _EPS)
        # Check energy conservation residual
        e1_new = eos1.energy(rho1_new, p_cur)
        e2_new = eos2.energy(rho2_new, p_cur)
        rho_e_new = a1_new * rho1_new * e1_new + a2_new * rho2_new * e2_new
        R = rho_e_new - rho_e
        # T adjustment via (∂ρe/∂T)_{p,m}
        if it < max_iter - 1:
            dT = np.maximum(T_cur * 1e-5, 0.1)
            T2 = T_cur + dT
            r1_2 = eos1.density(p_cur, T2)
            r2_2 = eos2.density(p_cur, T2)
            a1_2 = m_a / np.maximum(r1_2, _EPS)
            a2_2 = m_b / np.maximum(r2_2, _EPS)
            s2 = a1_2 + a2_2
            a1_2 = a1_2 / np.maximum(s2, _EPS)
            a2_2 = a2_2 / np.maximum(s2, _EPS)
            rho_e_2 = (a1_2 * r1_2 * eos1.energy(r1_2, p_cur)
                       + a2_2 * r2_2 * eos2.energy(r2_2, p_cur))
            dR_dT = (rho_e_2 - rho_e_new) / dT
            dR_dT = np.where(np.abs(dR_dT) < 1e-20, 1.0, dR_dT)
            T_cur = T_cur - R / dR_dT
            T_cur = np.maximum(T_cur, 1.0)
        if np.max(np.abs(R) / np.maximum(np.abs(rho_e), 1.0)) < tol:
            break
    return a1_new, rho1_new, rho2_new, p_cur, T_cur


def mixture_temperature(a1, rho1, rho2, T1, T2):
    """Mass-weighted mixture T (Kapila has T_k per phase, mixture T for diagnostics)."""
    rho = a1 * rho1 + (1.0 - a1) * rho2
    mass_w1 = a1 * rho1 / np.maximum(rho, 1e-30)
    return mass_w1 * T1 + (1.0 - mass_w1) * T2


# ─── K-phase generalization (K ≥ 2) ──────────────────────────────────────
def mixture_pressure_solve_K(alphas, rhos, rho_e, eos_list,
                              p_guess=None, newton_tol=1e-10, max_newton=5):
    """K-phase p-equilibrium: solve Σ α_k ρ_k e_k(ρ_k, p) = ρe for p.

    Arguments:
        alphas: list of K numpy arrays (volume fractions, each cell-shape)
        rhos:   list of K numpy arrays (phase densities)
        rho_e:  numpy array (mixture internal energy density)
        eos_list: list of K EOS objects

    Strategy identical to 2-phase version:
      1. All linear-in-p → direct analytic
      2. Else Newton + Brent fallback
    """
    K = len(eos_list)
    assert len(alphas) == K and len(rhos) == K

    # Linear fast path
    if all(_is_linear_in_p(e) for e in eos_list):
        return _linear_mixture_pressure_K(alphas, rhos, rho_e, eos_list)

    # Newton/Brent fallback
    if p_guess is None:
        # Ideal-like initial guess
        Gamma_inv = sum(alphas[k] / max(getattr(eos_list[k], 'gamma', 1.4) - 1.0, 0.1)
                        for k in range(K))
        Pi = sum(alphas[k] * getattr(eos_list[k], 'gamma', 1.4) * getattr(eos_list[k], 'pinf', 0.0)
                 / max(getattr(eos_list[k], 'gamma', 1.4) - 1.0, 0.1) for k in range(K))
        p_guess = (rho_e - Pi) / np.maximum(Gamma_inv, 1e-30)

    p = np.asarray(p_guess, dtype=float).copy()
    for it in range(max_newton):
        R_val = _residual_K(p, alphas, rhos, rho_e, eos_list)
        rel = np.max(np.abs(R_val) / np.maximum(np.abs(rho_e), 1.0))
        if rel < newton_tol:
            return p
        # Numerical Jacobian
        dp = np.maximum(np.abs(p) * 1e-6, 1.0)
        R_p = _residual_K(p + dp, alphas, rhos, rho_e, eos_list)
        dR = (R_p - R_val) / dp
        dR = np.where(np.abs(dR) < 1e-20, 1.0, dR)
        p = p - R_val / dR

    # Brent fallback
    def _fn(p_test):
        return _residual_K(p_test, alphas, rhos, rho_e, eos_list)
    p_lo = np.minimum(p - np.abs(p), -1e7)
    p_hi = np.maximum(p + np.abs(p) * 10.0, 1e10)
    return _brent_root(_fn, p_lo, p_hi, tol=newton_tol, max_iter=50)


def _linear_mixture_pressure_K(alphas, rhos, rho_e, eos_list):
    """Direct linear solve for K phases with linear-in-p EOS."""
    K = len(eos_list)
    num = rho_e.copy() if hasattr(rho_e, 'copy') else np.asarray(rho_e).copy()
    den = np.zeros_like(num)
    for k in range(K):
        Ak, Bk = _linear_coeffs(rhos[k], eos_list[k])
        num = num - alphas[k] * rhos[k] * Bk
        den = den + alphas[k] * rhos[k] * Ak
    return num / np.maximum(np.abs(den), 1e-30) * np.sign(den + 1e-30)


def _residual_K(p, alphas, rhos, rho_e, eos_list):
    K = len(eos_list)
    R = -rho_e.copy() if hasattr(rho_e, 'copy') else -np.asarray(rho_e).copy()
    for k in range(K):
        R = R + alphas[k] * rhos[k] * eos_list[k].energy(rhos[k], p)
    return R


def mixture_sound_speed_K(alphas, rhos, es, p, eos_list):
    """K-phase Wood-like mixture sound speed."""
    K = len(eos_list)
    c_sqs = [eos_list[k].sound_speed_sq(rhos[k], es[k], p) for k in range(K)]
    rho = sum(alphas[k] * rhos[k] for k in range(K))
    inv_rhoc2 = sum(alphas[k] / np.maximum(rhos[k] * c_sqs[k], 1e-30) for k in range(K))
    c_sq_mix = 1.0 / np.maximum(rho * inv_rhoc2, 1e-30)
    return c_sq_mix, c_sqs
