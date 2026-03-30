"""Find the source of PE error: MUSCL reconstruction vs energy flux."""
import sys
sys.path.insert(0, r'D:\work\claude_code\Abgrall_solve_1')
import numpy as np
from pressure_eq import (initial_condition, srk_c2, srk_p, prim, pe_err,
                     epsilon_v, _minmod)

N    = 501
dx   = 1.0 / N
x    = np.linspace(dx/2, 1 - dx/2, N)
CFL  = 0.3

r1, r2, u, rhoE, T, p = initial_condition(x, 5e6)
rho = r1 + r2
rhoU = rho * u

c2   = srk_c2(r1, r2, T)
lam_cell = np.abs(u) + np.sqrt(c2)
lam = float(np.max(lam_cell))
dt   = CFL * dx / lam

def llf_flux_1st_order(r1, r2, u, rhoe, p, lam_cell, scheme='FC', eps_pair=None):
    """First-order LLF (no MUSCL)."""
    rho = r1 + r2
    rhoE = rhoe + 0.5*rho*u**2

    r1p = np.roll(r1, -1); r2p = np.roll(r2, -1)
    up  = np.roll(u, -1);  pp  = np.roll(p, -1)
    rhop = r1p + r2p
    ehoep = np.roll(rhoe, -1)
    rhoEp = np.roll(rhoE, -1)

    lam_intf = np.maximum(lam_cell, np.roll(lam_cell, -1))

    F1 = 0.5*(r1*u + r1p*up) - 0.5*lam_intf*(r1p - r1)
    F2 = 0.5*(r2*u + r2p*up) - 0.5*lam_intf*(r2p - r2)
    FU = 0.5*(rho*u**2 + p + rhop*up**2 + pp) - 0.5*lam_intf*(rhop*up - rho*u)

    rho_h = 0.5*(rho + rhop)
    u_h   = 0.5*(u + up)
    p_h   = 0.5*(p + pp)
    rhoe_h = 0.5*(rhoe + ehoep)
    dr1 = r1p - r1; dr2 = r2p - r2; du = up - u

    if scheme == 'APEC_correct':
        eps0, eps1 = eps_pair
        eps0p = np.roll(eps0, -1); eps1p = np.roll(eps1, -1)
        eps0_h = 0.5*(eps0 + eps0p); eps1_h = 0.5*(eps1 + eps1p)
        # Complete PE-consistent: Σε_s*Δρ_s + ½u_h²*Δρ + ρ_h*u_h*Δu
        drhoE_pep = (eps0_h*dr1 + eps1_h*dr2
                     + 0.5*u_h**2*(dr1+dr2)
                     + rho_h*u_h*du)
        FE_cen = (rhoe_h + 0.5*rho_h*u_h**2 + p_h) * u_h
        FE = FE_cen - 0.5*lam_intf*drhoE_pep
    else:
        FE_cen = (rhoe_h + 0.5*rho_h*u_h**2 + p_h) * u_h
        FE = FE_cen - 0.5*lam_intf*(rhoEp - rhoE)

    return F1, F2, FU, FE

def one_step_1st(r1, r2, u, rhoe, rhoE, T, p, scheme='FC', eps_pair=None):
    """One Euler step with first-order LLF."""
    lam_cell_ = np.abs(u) + np.sqrt(srk_c2(r1, r2, T))
    F1, F2, FU, FE = llf_flux_1st_order(r1, r2, u, rhoe, p, lam_cell_, scheme, eps_pair)
    d = lambda F: -(F - np.roll(F,1)) / dx
    r1_new  = r1  + dt*d(F1)
    r2_new  = r2  + dt*d(F2)
    rhoU_new= rho*u + dt*d(FU)
    rhoE_new= rhoE  + dt*d(FE)
    r1_new = np.maximum(r1_new, 0)
    r2_new = np.maximum(r2_new, 0)
    rho_new = r1_new + r2_new
    u_new   = rhoU_new / np.maximum(rho_new, 1e-30)
    rhoe_new= rhoE_new - 0.5*rho_new*u_new**2
    # Compute p directly
    from pressure_eq import T_from_rhoe
    T_new = T_from_rhoe(r1_new, r2_new, rhoe_new, T)
    p_new = srk_p(r1_new, r2_new, T_new)
    return r1_new, r2_new, u_new, rhoe_new, rhoE_new, T_new, p_new

# Fix the function above
from pressure_eq import T_from_rhoe, srk_p as _srk_p

def step1(r1, r2, u, rhoe, rhoE, T, p, scheme='FC', eps_pair=None):
    lam_cell_ = np.abs(u) + np.sqrt(srk_c2(r1, r2, T))
    F1, F2, FU, FE = llf_flux_1st_order(r1, r2, u, rhoe, p, lam_cell_, scheme, eps_pair)
    def d(F): return -(F - np.roll(F,1)) / dx
    r1n = np.maximum(r1 + dt*d(F1), 0)
    r2n = np.maximum(r2 + dt*d(F2), 0)
    rhon = r1n + r2n
    rhoUn = rho*u + dt*d(FU)
    rhoEn = rhoE + dt*d(FE)
    un = rhoUn / np.maximum(rhon, 1e-30)
    rhoEn_adj = rhoEn
    rhoen = rhoEn_adj - 0.5*rhon*un**2
    Tn = T_from_rhoe(r1n, r2n, rhoen, T)
    pn = _srk_p(r1n, r2n, Tn)
    return r1n, r2n, un, rhoen, rhoEn_adj, Tn, pn

rhoe = rhoE - 0.5*rho*u**2

# 1. First-order FC (no MUSCL)
r1n, r2n, un, rhoen, rhoEn, Tn, pn = step1(r1, r2, u, rhoe, rhoE, T, p, 'FC')
print(f"1st-order FC   step1: PE={pe_err(pn, 5e6):.4e}")

# 2. First-order APEC with CORRECT PE-consistent dissipation (including ½u²Δρ)
eps0 = epsilon_v(r1, r2, T, 0)
eps1 = epsilon_v(r1, r2, T, 1)
r1n, r2n, un, rhoen, rhoEn, Tn, pn = step1(
    r1, r2, u, rhoe, rhoE, T, p, 'APEC_correct', eps_pair=(eps0, eps1))
print(f"1st-order APEC_correct step1: PE={pe_err(pn, 5e6):.4e}")
