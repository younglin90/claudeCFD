// cfd/euler2d.hpp — 2D compressible Euler (gamma-law) + LLF flux.
// Port of Euler2D (equations.py) and llf (flux.py) for nvar=4.
//   U=(rho,rho u,rho v,rho E), W=(rho,u,v,p), e=p/((g-1)rho), E=e+(u^2+v^2)/2.
// Scalar per-cell/per-face leaf functions (acc routine seq) for OpenMP/OpenACC.
#pragma once
#include "cfd/eos.hpp"   // CFD_ROUTINE_SEQ
#include <cmath>

namespace cfd {

constexpr double EU2_EPS = 1e-30;

struct Euler2D {
    double gamma = 1.4;
    static constexpr int nvar = 4;

    CFD_ROUTINE_SEQ void prim_to_cons(const double W[4], double U[4]) const {
        double rho = W[0] > EU2_EPS ? W[0] : EU2_EPS;
        double u = W[1], v = W[2], p = W[3];
        double e = p / ((gamma - 1.0) * rho);
        double E = e + 0.5 * (u*u + v*v);
        U[0] = rho; U[1] = rho*u; U[2] = rho*v; U[3] = rho*E;
    }
    CFD_ROUTINE_SEQ void cons_to_prim(const double U[4], double W[4]) const {
        double rho = U[0] > EU2_EPS ? U[0] : EU2_EPS;
        double u = U[1]/rho, v = U[2]/rho, E = U[3]/rho;
        double p = (gamma - 1.0) * rho * (E - 0.5*(u*u + v*v));
        W[0] = rho; W[1] = u; W[2] = v; W[3] = p;
    }
    CFD_ROUTINE_SEQ void physical_flux(const double U[4], double nx, double ny, double F[4]) const {
        double rho = U[0] > EU2_EPS ? U[0] : EU2_EPS;
        double u = U[1]/rho, v = U[2]/rho, E = U[3]/rho;
        double p = (gamma - 1.0) * rho * (E - 0.5*(u*u + v*v));
        double un = u*nx + v*ny;
        F[0] = rho*un;
        F[1] = rho*u*un + p*nx;
        F[2] = rho*v*un + p*ny;
        F[3] = (U[3] + p)*un;
    }
    CFD_ROUTINE_SEQ double max_wave_speed(const double U[4], double nx, double ny) const {
        double rho = U[0] > EU2_EPS ? U[0] : EU2_EPS;
        double u = U[1]/rho, v = U[2]/rho, E = U[3]/rho;
        double p = (gamma - 1.0) * rho * (E - 0.5*(u*u + v*v));
        double un = u*nx + v*ny;
        double c2 = gamma * p / rho; if (c2 < EU2_EPS) c2 = EU2_EPS;
        return std::fabs(un) + std::sqrt(c2);
    }
};

// Local Lax-Friedrichs (Rusanov) flux from primitive L/R states + normal.
CFD_ROUTINE_SEQ
inline void llf_euler2d(const Euler2D& eq, const double WL[4], const double WR[4],
                        double nx, double ny, double F[4]) {
    double UL[4], UR[4], FL[4], FR[4];
    eq.prim_to_cons(WL, UL); eq.prim_to_cons(WR, UR);
    eq.physical_flux(UL, nx, ny, FL); eq.physical_flux(UR, nx, ny, FR);
    double lamL = eq.max_wave_speed(UL, nx, ny), lamR = eq.max_wave_speed(UR, nx, ny);
    double lam = lamL > lamR ? lamL : lamR;
    for (int k = 0; k < 4; ++k) F[k] = 0.5*(FL[k] + FR[k]) - 0.5*lam*(UR[k] - UL[k]);
}

CFD_ROUTINE_SEQ inline double sgn(double x) { return (x > 0) - (x < 0); }
CFD_ROUTINE_SEQ inline double safe_den(double d) {
    return (std::fabs(d) > EU2_EPS) ? d : (sgn(d) * EU2_EPS + EU2_EPS);
}

// Shock-stabilized HLLC/HLLE-hybrid for 2D Euler (port of hllc_adc_2d). Far less
// diffusive than LLF at contacts/shear; blends in HLLE at strong shocks.
CFD_ROUTINE_SEQ
inline void hllc_euler2d(const Euler2D& eq, const double WL[4], const double WR[4],
                         double nx, double ny, double F[4]) {
    double tx = -ny, ty = nx;
    double rL = WL[0] > EU2_EPS ? WL[0] : EU2_EPS, rR = WR[0] > EU2_EPS ? WR[0] : EU2_EPS;
    double uL = WL[1], vL = WL[2], pL = WL[3] > EU2_EPS ? WL[3] : EU2_EPS;
    double uR = WR[1], vR = WR[2], pR = WR[3] > EU2_EPS ? WR[3] : EU2_EPS;
    double unL = uL*nx + vL*ny, unR = uR*nx + vR*ny;
    double utL = uL*tx + vL*ty, utR = uR*tx + vR*ty;
    double g = eq.gamma;
    double cL = std::sqrt(std::max(g*pL/rL, EU2_EPS)), cR = std::sqrt(std::max(g*pR/rR, EU2_EPS));
    double EL = pL/((g-1.0)*rL) + 0.5*(uL*uL+vL*vL);
    double ER = pR/((g-1.0)*rR) + 0.5*(uR*uR+vR*vR);
    double UL[4], UR[4], FL[4], FR[4];
    eq.prim_to_cons(WL, UL); eq.prim_to_cons(WR, UR);
    eq.physical_flux(UL, nx, ny, FL); eq.physical_flux(UR, nx, ny, FR);

    // Wave speeds. Default = Davis direct min/max (diffusive at contacts). HLLC_PVRS = Toro
    // pressure-based adaptive (Spruce-Speares TSS, Toro 10.59): q_K compresses the bracket in
    // shocks, tightens contact dissipation -> sharper slip line (matches Cheng 2021 ref [78]).
    static const bool PVRS = []{ const char* e = std::getenv("HLLC_PVRS"); return e ? std::atoi(e) != 0 : false; }();
    double SL, SR;
    if (PVRS) {
        double rbar = 0.5*(rL+rR), cbar = 0.5*(cL+cR);
        double ppv = std::max(0.0, 0.5*(pL+pR) - 0.5*(unR-unL)*rbar*cbar);
        double gL = ppv<=pL ? 1.0 : std::sqrt(std::max(1.0 + (g+1.0)/(2.0*g)*(ppv/pL - 1.0), 0.0));
        double gR = ppv<=pR ? 1.0 : std::sqrt(std::max(1.0 + (g+1.0)/(2.0*g)*(ppv/pR - 1.0), 0.0));
        SL = unL - cL*gL; SR = unR + cR*gR;
    } else {
        SL = std::min(unL - cL, unR - cR); SR = std::max(unL + cL, unR + cR);
    }
    double den = safe_den(rL*(SL-unL) - rR*(SR-unR));
    double SM = (pR - pL + rL*unL*(SL-unL) - rR*unR*(SR-unR)) / den;

    auto star = [&](double rho, double un, double ut, double p, double E, double S, double Us[4]) {
        double ds = safe_den(S - SM);
        double fac = rho*(S - un)/ds;
        double mn = fac*SM, mt = fac*ut;
        double wd = safe_den(rho*(S - un));
        Us[0] = fac; Us[1] = mn*nx + mt*tx; Us[2] = mn*ny + mt*ty;
        Us[3] = fac*(E + (SM - un)*(SM + p/wd));
    };
    double UsL[4], UsR[4]; star(rL,unL,utL,pL,EL,SL,UsL); star(rR,unR,utR,pR,ER,SR,UsR);
    double Fh[4], Fhll[4];
    for (int k = 0; k < 4; ++k) {
        if (SL >= 0.0) Fh[k] = FL[k];
        else if (SM >= 0.0) Fh[k] = FL[k] + SL*(UsL[k] - UL[k]);
        else if (SR > 0.0) Fh[k] = FR[k] + SR*(UsR[k] - UR[k]);
        else Fh[k] = FR[k];
        // HLL
        double dh = std::max(SR - SL, EU2_EPS);
        if (SL >= 0.0) Fhll[k] = FL[k];
        else if (SR <= 0.0) Fhll[k] = FR[k];
        else Fhll[k] = (SR*FL[k] - SL*FR[k] + SL*SR*(UR[k]-UL[k]))/dh;
    }
    double pj = std::fabs(pR - pL) / std::max(pR + pL, EU2_EPS);
    double comp = std::max(0.0, unL - unR) / std::max(cL + cR, EU2_EPS);
    double shock = std::min(std::max((pj - 0.05)/0.35, 0.0), 1.0)
                 * std::min(std::max(4.0*comp, 0.0), 1.0);
    // HLLC_HLLBLEND: max fraction of 2-wave HLL blended in at strong shocks (carbuncle cure).
    // Default 0.45. Set 0 for PURE contact-restoring HLLC (Cheng/Toro TSS) -> sharper KH slip-line
    // rolls (the blend contaminates the DM triple-point roll-up region with contactless HLL).
    static const double HLLC_BLEND = []{ const char* e=std::getenv("HLLC_HLLBLEND"); return e?std::atof(e):0.45; }();
    double blend = HLLC_BLEND * shock;
    for (int k = 0; k < 4; ++k) F[k] = (1.0 - blend)*Fh[k] + blend*Fhll[k];
}

// Standard HLL (Harten-Lax-van Leer, 2-wave, NO contact restoration) — the flux used
// by Cheng et al. 2021 (JCP 428:110088) for the single-phase Euler BVD tests incl. the
// Mach-3 forward step. More dissipative than HLLC (smears contacts/shear) but robust;
// the extra dissipation stabilises THINC's discontinuity sharpening. Davis wave speeds.
CFD_ROUTINE_SEQ
inline void hll_euler2d(const Euler2D& eq, const double WL[4], const double WR[4],
                        double nx, double ny, double F[4]) {
    double rL = WL[0] > EU2_EPS ? WL[0] : EU2_EPS, rR = WR[0] > EU2_EPS ? WR[0] : EU2_EPS;
    double uL = WL[1], vL = WL[2], pL = WL[3] > EU2_EPS ? WL[3] : EU2_EPS;
    double uR = WR[1], vR = WR[2], pR = WR[3] > EU2_EPS ? WR[3] : EU2_EPS;
    double unL = uL*nx + vL*ny, unR = uR*nx + vR*ny;
    double g = eq.gamma;
    double cL = std::sqrt(std::max(g*pL/rL, EU2_EPS)), cR = std::sqrt(std::max(g*pR/rR, EU2_EPS));
    double UL[4], UR[4], FL[4], FR[4];
    eq.prim_to_cons(WL, UL); eq.prim_to_cons(WR, UR);
    eq.physical_flux(UL, nx, ny, FL); eq.physical_flux(UR, nx, ny, FR);
    double SL = std::min(unL - cL, unR - cR), SR = std::max(unL + cL, unR + cR);
    double dh = std::max(SR - SL, EU2_EPS);
    for (int k = 0; k < 4; ++k) {
        if (SL >= 0.0) F[k] = FL[k];
        else if (SR <= 0.0) F[k] = FR[k];
        else F[k] = (SR*FL[k] - SL*FR[k] + SL*SR*(UR[k]-UL[k]))/dh;
    }
}

// Rotated-HLLC (Nishikawa-Kitamura): evaluate HLLC in two orthonormal rotated
// directions n1 (aligned with the velocity difference = shock normal) and n2 (perp),
// then F = (n.n1) HLLC(n1) + (n.n2) HLLC(n2). Carbuncle-FREE at grid-aligned strong
// shocks: the n1-Riemann aligns with the actual shock so the upwind dissipation acts
// normal to it, while n2 resolves the tangential/contact with little diffusion.
CFD_ROUTINE_SEQ
inline void rotated_hllc_euler2d(const Euler2D& eq, const double WL[4], const double WR[4],
                                 double nx, double ny, double F[4]) {
    double dvx = WR[1]-WL[1], dvy = WR[2]-WL[2];
    double dv = std::sqrt(dvx*dvx + dvy*dvy);
    double n1x, n1y;
    if (dv > 1e-12) { n1x = dvx/dv; n1y = dvy/dv; } else { n1x = nx; n1y = ny; }
    double a1 = n1x*nx + n1y*ny;
    if (a1 < 0.0) { n1x = -n1x; n1y = -n1y; a1 = -a1; }   // make n1.n >= 0
    double n2x = -n1y, n2y = n1x;
    double a2 = n2x*nx + n2y*ny;
    if (a2 < 0.0) { n2x = -n2x; n2y = -n2y; a2 = -a2; }   // make n2.n >= 0
    double F1[4], F2[4];
    hllc_euler2d(eq, WL, WR, n1x, n1y, F1);
    hllc_euler2d(eq, WL, WR, n2x, n2y, F2);
    for (int k = 0; k < 4; ++k) F[k] = a1*F1[k] + a2*F2[k];
}

// HLLCM — robust HLLC-type for strong shock (Shen-Yan-Yuan JCP 309 (2016) 185-206, ref [48] in
// Tann/Deng/Loubere/Xiao 2020, used there explicitly "to prevent the carbuncle phenomenon").
// Identical HLLC base + HLL blend as hllc_euler2d, but the shock-sensor blend toward HLL is NOT
// capped at 0.45: it ramps to FULL HLL (blend->1) at strong compressive shocks (large pressure
// jump AND compression), fully curing the carbuncle, while staying full HLLC (blend=0) at
// contacts/shear (pj~0 => sharp KH). The sensor (pj*comp) fires only on shocks, never on contacts.
CFD_ROUTINE_SEQ
inline void hllcm_euler2d(const Euler2D& eq, const double WL[4], const double WR[4],
                          double nx, double ny, double F[4]) {
    double tx = -ny, ty = nx;
    double rL = WL[0] > EU2_EPS ? WL[0] : EU2_EPS, rR = WR[0] > EU2_EPS ? WR[0] : EU2_EPS;
    double uL = WL[1], vL = WL[2], pL = WL[3] > EU2_EPS ? WL[3] : EU2_EPS;
    double uR = WR[1], vR = WR[2], pR = WR[3] > EU2_EPS ? WR[3] : EU2_EPS;
    double unL = uL*nx + vL*ny, unR = uR*nx + vR*ny;
    double utL = uL*tx + vL*ty, utR = uR*tx + vR*ty;
    double g = eq.gamma;
    double cL = std::sqrt(std::max(g*pL/rL, EU2_EPS)), cR = std::sqrt(std::max(g*pR/rR, EU2_EPS));
    double EL = pL/((g-1.0)*rL) + 0.5*(uL*uL+vL*vL);
    double ER = pR/((g-1.0)*rR) + 0.5*(uR*uR+vR*vR);
    double UL[4], UR[4], FL[4], FR[4];
    eq.prim_to_cons(WL, UL); eq.prim_to_cons(WR, UR);
    eq.physical_flux(UL, nx, ny, FL); eq.physical_flux(UR, nx, ny, FR);
    double SL = std::min(unL - cL, unR - cR), SR = std::max(unL + cL, unR + cR);
    double den = safe_den(rL*(SL-unL) - rR*(SR-unR));
    double SM = (pR - pL + rL*unL*(SL-unL) - rR*unR*(SR-unR)) / den;
    auto star = [&](double rho, double un, double ut, double p, double E, double S, double Us[4]) {
        double ds = safe_den(S - SM);
        double fac = rho*(S - un)/ds;
        double mn = fac*SM, mt = fac*ut;
        double wd = safe_den(rho*(S - un));
        Us[0] = fac; Us[1] = mn*nx + mt*tx; Us[2] = mn*ny + mt*ty;
        Us[3] = fac*(E + (SM - un)*(SM + p/wd));
    };
    double UsL[4], UsR[4]; star(rL,unL,utL,pL,EL,SL,UsL); star(rR,unR,utR,pR,ER,SR,UsR);
    double Fh[4], Fhll[4];
    for (int k = 0; k < 4; ++k) {
        if (SL >= 0.0) Fh[k] = FL[k];
        else if (SM >= 0.0) Fh[k] = FL[k] + SL*(UsL[k] - UL[k]);
        else if (SR > 0.0) Fh[k] = FR[k] + SR*(UsR[k] - UR[k]);
        else Fh[k] = FR[k];
        double dh = std::max(SR - SL, EU2_EPS);
        if (SL >= 0.0) Fhll[k] = FL[k];
        else if (SR <= 0.0) Fhll[k] = FR[k];
        else Fhll[k] = (SR*FL[k] - SL*FR[k] + SL*SR*(UR[k]-UL[k]))/dh;
    }
    double pj = std::fabs(pR - pL) / std::max(pR + pL, EU2_EPS);
    double comp = std::max(0.0, unL - unR) / std::max(cL + cR, EU2_EPS);
    double shock = std::min(std::max((pj - 0.05)/0.35, 0.0), 1.0)
                 * std::min(std::max(4.0*comp, 0.0), 1.0);
    double blend = shock;   // HLLCM: ramp to FULL HLL at strong shock (vs hllc's 0.45 cap)
    for (int k = 0; k < 4; ++k) F[k] = (1.0 - blend)*Fh[k] + blend*Fhll[k];
}

// Rotated-HLLCM: rotated_hllc but with hllcm in each rotated direction (double carbuncle cure:
// rotation breaks grid-alignment + the M-blend adds full-HLL dissipation at the strong shock).
CFD_ROUTINE_SEQ
inline void rotated_hllcm_euler2d(const Euler2D& eq, const double WL[4], const double WR[4],
                                  double nx, double ny, double F[4]) {
    double dvx = WR[1]-WL[1], dvy = WR[2]-WL[2];
    double dv = std::sqrt(dvx*dvx + dvy*dvy);
    double n1x, n1y;
    if (dv > 1e-12) { n1x = dvx/dv; n1y = dvy/dv; } else { n1x = nx; n1y = ny; }
    double a1 = n1x*nx + n1y*ny;
    if (a1 < 0.0) { n1x = -n1x; n1y = -n1y; a1 = -a1; }
    double n2x = -n1y, n2y = n1x;
    double a2 = n2x*nx + n2y*ny;
    if (a2 < 0.0) { n2x = -n2x; n2y = -n2y; a2 = -a2; }
    double F1[4], F2[4];
    hllcm_euler2d(eq, WL, WR, n1x, n1y, F1);
    hllcm_euler2d(eq, WL, WR, n2x, n2y, F2);
    for (int k = 0; k < 4; ++k) F[k] = a1*F1[k] + a2*F2[k];
}

// SLAU2 all-Mach flux (Kitamura & Shima, JCP 245 (2013) 62-83) — 2D single-species.
// Low-dissipation at low Mach (chi=(1-M_hat)^2 scales the pressure-diffusion in the mass
// flux) + shock-stable SLAU2 pressure flux. Faithful port of the project oracle
// solver/He2024/common.py::slau2_flux_anp (standard piecewise AUSM splits). Normal-
// direction flux; tangential momentum carried by the mass flux (upwinded).
inline void slau2_euler2d(const Euler2D& eq, const double WL[4], const double WR[4],
                          double nx, double ny, double F[4]) {
    const double g = eq.gamma;
    double rhoL=WL[0], uL=WL[1], vL=WL[2], pL=WL[3];
    double rhoR=WR[0], uR=WR[1], vR=WR[2], pR=WR[3];
    double VnL = uL*nx+vL*ny, VnR = uR*nx+vR*ny;
    double cL = std::sqrt(g*pL/rhoL), cR = std::sqrt(g*pR/rhoR);
    double cb = 0.5*(cL+cR); if (cb < 1e-300) cb = 1e-300;
    double ML = VnL/cb, MR = VnR/cb, aML=std::fabs(ML), aMR=std::fabs(MR);
    double Mp = (aML>=1.0)? 0.5*(ML+aML) :  0.25*(ML+1.0)*(ML+1.0);
    double Mm = (aMR>=1.0)? 0.5*(MR-aMR) : -0.25*(MR-1.0)*(MR-1.0);
    double Pp = (aML>=1.0)? (ML>0.0?1.0:0.0) : 0.25*(ML+1.0)*(ML+1.0)*(2.0-ML);
    double Pm = (aMR>=1.0)? (MR>0.0?0.0:1.0) : 0.25*(MR-1.0)*(MR-1.0)*(2.0+MR);
    double Mbar = std::sqrt(0.5*(VnL*VnL+VnR*VnR))/cb;
    double Mhat = Mbar<1.0?Mbar:1.0; double chi=(1.0-Mhat)*(1.0-Mhat);
    double bLm = std::max(std::min(ML,0.0),-1.0), bRp = std::min(std::max(MR,0.0),1.0);
    double gg = -bLm*bRp;                                   // expansion/stagnation detector
    double VpL = (1.0-gg)*cb*Mp + gg*std::fabs(VnL);
    double VmR = (1.0-gg)*cb*Mm - gg*std::fabs(VnR);
    double mdot = 0.5*(rhoL*(VnL+std::fabs(VpL)) + rhoR*(VnR-std::fabs(VmR))) - chi/(2.0*cb)*(pR-pL);
    double ubar = std::sqrt(0.5*(VnL*VnL+VnR*VnR));
    double pface = 0.5*(pL+pR) + 0.5*(Pp-Pm)*(pL-pR) + ubar*(Pp+Pm-1.0)*0.5*(rhoL+rhoR)*cb;
    double rhoEL = pL/(g-1.0)+0.5*rhoL*(uL*uL+vL*vL), rhoER = pR/(g-1.0)+0.5*rhoR*(uR*uR+vR*vR);
    double HL=(rhoEL+pL)/rhoL, HR=(rhoER+pR)/rhoR;
    bool pos = mdot>=0.0;
    double u_up=pos?uL:uR, v_up=pos?vL:vR, H_up=pos?HL:HR;
    F[0]=mdot;
    F[1]=mdot*u_up + pface*nx;
    F[2]=mdot*v_up + pface*ny;
    F[3]=mdot*H_up;
}

// Rotated SLAU2 (Nishikawa-Kitamura rotation; mirror of rotated_hllc_euler2d): SLAU2 in
// n1 (velocity-difference = shock normal) and n2 (perp), F = (n.n1)SLAU2(n1)+(n.n2)SLAU2(n2).
inline void rotated_slau2_euler2d(const Euler2D& eq, const double WL[4], const double WR[4],
                                  double nx, double ny, double F[4]) {
    double dvx = WR[1]-WL[1], dvy = WR[2]-WL[2];
    double dv = std::sqrt(dvx*dvx + dvy*dvy);
    double n1x, n1y;
    if (dv > 1e-12) { n1x = dvx/dv; n1y = dvy/dv; } else { n1x = nx; n1y = ny; }
    double a1 = n1x*nx + n1y*ny;
    if (a1 < 0.0) { n1x = -n1x; n1y = -n1y; a1 = -a1; }
    double n2x = -n1y, n2y = n1x;
    double a2 = n2x*nx + n2y*ny;
    if (a2 < 0.0) { n2x = -n2x; n2y = -n2y; a2 = -a2; }
    double F1[4], F2[4];
    slau2_euler2d(eq, WL, WR, n1x, n1y, F1);
    slau2_euler2d(eq, WL, WR, n2x, n2y, F2);
    for (int k = 0; k < 4; ++k) F[k] = a1*F1[k] + a2*F2[k];
}

// Roe flux (4-wave: acoustic-/entropy/shear/acoustic+, |lambda| no entropy fix) — exact
// port of solver_tmlpu/.../flux.py::_roe_face. Returns global-frame F=(mass,mom_x,mom_y,E).
// Falls back to HLLC if a^2<=0 or non-finite.
CFD_ROUTINE_SEQ
inline void roe_face_euler2d(const Euler2D& eq, const double WL[4], const double WR[4],
                             double nx, double ny, double F[4]) {
    const double g = eq.gamma, eps = 1e-30;
    double rl=WL[0]>eps?WL[0]:eps, rr=WR[0]>eps?WR[0]:eps;
    double pl=WL[3]>eps?WL[3]:eps, pr=WR[3]>eps?WR[3]:eps;
    double ul=WL[1], vl=WL[2], ur=WR[1], vr=WR[2];
    double tx=-ny, ty=nx;
    double unl=ul*nx+vl*ny, unr=ur*nx+vr*ny, utl=ul*tx+vl*ty, utr=ur*tx+vr*ty;
    double ql=ul*ul+vl*vl, qr=ur*ur+vr*vr;
    double el=pl/((g-1.0)*rl)+0.5*ql, er=pr/((g-1.0)*rr)+0.5*qr, El=rl*el, Er=rr*er;
    double Hl=(El+pl)/rl, Hr=(Er+pr)/rr;
    double Fl0=rl*unl, Fl1n=rl*unl*unl+pl, Fl1t=rl*unl*utl, Fl3=(El+pl)*unl;
    double Fr0=rr*unr, Fr1n=rr*unr*unr+pr, Fr1t=rr*unr*utr, Fr3=(Er+pr)*unr;
    double srl=std::sqrt(rl), srr=std::sqrt(rr), inv=1.0/std::max(srl+srr,eps);
    double un=(srl*unl+srr*unr)*inv, ut=(srl*utl+srr*utr)*inv, H=(srl*Hl+srr*Hr)*inv;
    double rho=srl*srr, q=un*un+ut*ut, a2=(g-1.0)*(H-0.5*q);
    if (a2 <= eps) { hllc_euler2d(eq, WL, WR, nx, ny, F); return; }
    double a=std::sqrt(a2);
    double drho=rr-rl, dun=unr-unl, dut=utr-utl, dp=pr-pl, ia2=1.0/std::max(a2,eps);
    double am=0.5*(dp*ia2 - rho*dun/std::max(a,eps)), ap=0.5*(dp*ia2 + rho*dun/std::max(a,eps));
    double aent=drho-dp*ia2, ash=rho*dut;
    double lm=std::fabs(un-a), lc=std::fabs(un), lp=std::fabs(un+a);
    double d0 = lm*am + lc*aent + lp*ap;
    double d1n= lm*am*(un-a) + lc*aent*un + lp*ap*(un+a);
    double d1t= lm*am*ut + lc*aent*ut + lc*ash + lp*ap*ut;
    double d3 = lm*am*(H-un*a) + lc*aent*(0.5*q) + lc*ash*ut + lp*ap*(H+un*a);
    double f0=0.5*(Fl0+Fr0)-0.5*d0, fn=0.5*(Fl1n+Fr1n)-0.5*d1n;
    double ft=0.5*(Fl1t+Fr1t)-0.5*d1t, f3=0.5*(Fl3+Fr3)-0.5*d3;
    F[0]=f0; F[1]=fn*nx+ft*tx; F[2]=fn*ny+ft*ty; F[3]=f3;
    if (!std::isfinite(F[0]+F[1]+F[2]+F[3])) hllc_euler2d(eq, WL, WR, nx, ny, F);
}

// Rotated Roe hybrid (port of roe_rotated_hybrid_2d): HLLC in the velocity-difference
// direction n1 (=shock normal, dissipative & carbuncle-free) + Roe in the perpendicular
// n2 (low-diffusion contact/shear). F = (n.n1) HLLC(n1) + (n.n2) Roe(n2).
CFD_ROUTINE_SEQ
inline void rotated_roe_euler2d(const Euler2D& eq, const double WL[4], const double WR[4],
                                double nx, double ny, double F[4]) {
    double du=WR[1]-WL[1], dv=WR[2]-WL[2], sj=std::sqrt(du*du+dv*dv);
    double scale = 2.220446049250313e-16*(std::fabs(WL[1])+std::fabs(WR[1])+std::fabs(WL[2])+std::fabs(WR[2])+1.0);
    double n1x, n1y;
    if (sj > scale) { n1x=du/std::max(sj,1e-30); n1y=dv/std::max(sj,1e-30); } else { n1x=nx; n1y=ny; }
    double a1=nx*n1x+ny*n1y; if (a1<0.0){n1x=-n1x;n1y=-n1y;a1=-a1;}
    double n2x=-n1y, n2y=n1x; double a2=nx*n2x+ny*n2y; if (a2<0.0){n2x=-n2x;n2y=-n2y;a2=-a2;}
    double Fh[4], Fr[4];
    hllc_euler2d(eq, WL, WR, n1x, n1y, Fh);     // HLLC in shock-normal direction
    roe_face_euler2d(eq, WL, WR, n2x, n2y, Fr); // Roe in the perpendicular direction
    for (int k = 0; k < 4; ++k) F[k] = a1*Fh[k] + a2*Fr[k];
}

// Harten entropy fix: smooth |lambda| near zero -> kills the sonic-point expansion-shock glitch.
CFD_ROUTINE_SEQ
inline double harten_ef_(double lam, double delta) {
    double al = std::fabs(lam);
    return (al >= delta) ? al : 0.5*(lam*lam/(delta>1e-30?delta:1e-30) + delta);
}

// Plain Roe flux WITH the standard Harten entropy fix on the genuinely-nonlinear ACOUSTIC waves
// (un-a, un+a) only; the linearly-degenerate contact/shear (un) is left exact so contacts stay
// sharp. delta = 0.1*(|un|+a). Same wave algebra as roe_face_euler2d.
CFD_ROUTINE_SEQ
inline void roe_ef_face_euler2d(const Euler2D& eq, const double WL[4], const double WR[4],
                                double nx, double ny, double F[4]) {
    const double g = eq.gamma, eps = 1e-30;
    double rl=WL[0]>eps?WL[0]:eps, rr=WR[0]>eps?WR[0]:eps;
    double pl=WL[3]>eps?WL[3]:eps, pr=WR[3]>eps?WR[3]:eps;
    double ul=WL[1], vl=WL[2], ur=WR[1], vr=WR[2];
    double tx=-ny, ty=nx;
    double unl=ul*nx+vl*ny, unr=ur*nx+vr*ny, utl=ul*tx+vl*ty, utr=ur*tx+vr*ty;
    double ql=ul*ul+vl*vl, qr=ur*ur+vr*vr;
    double el=pl/((g-1.0)*rl)+0.5*ql, er=pr/((g-1.0)*rr)+0.5*qr, El=rl*el, Er=rr*er;
    double Hl=(El+pl)/rl, Hr=(Er+pr)/rr;
    double Fl0=rl*unl, Fl1n=rl*unl*unl+pl, Fl1t=rl*unl*utl, Fl3=(El+pl)*unl;
    double Fr0=rr*unr, Fr1n=rr*unr*unr+pr, Fr1t=rr*unr*utr, Fr3=(Er+pr)*unr;
    double srl=std::sqrt(rl), srr=std::sqrt(rr), inv=1.0/std::max(srl+srr,eps);
    double un=(srl*unl+srr*unr)*inv, ut=(srl*utl+srr*utr)*inv, H=(srl*Hl+srr*Hr)*inv;
    double rho=srl*srr, q=un*un+ut*ut, a2=(g-1.0)*(H-0.5*q);
    if (a2 <= eps) { hllc_euler2d(eq, WL, WR, nx, ny, F); return; }
    double a=std::sqrt(a2);
    double drho=rr-rl, dun=unr-unl, dut=utr-utl, dp=pr-pl, ia2=1.0/std::max(a2,eps);
    double am=0.5*(dp*ia2 - rho*dun/std::max(a,eps)), ap=0.5*(dp*ia2 + rho*dun/std::max(a,eps));
    double aent=drho-dp*ia2, ash=rho*dut;
    double delta=0.1*(std::fabs(un)+a);
    double lm=harten_ef_(un-a,delta), lc=std::fabs(un), lp=harten_ef_(un+a,delta);
    double d0 = lm*am + lc*aent + lp*ap;
    double d1n= lm*am*(un-a) + lc*aent*un + lp*ap*(un+a);
    double d1t= lm*am*ut + lc*aent*ut + lc*ash + lp*ap*ut;
    double d3 = lm*am*(H-un*a) + lc*aent*(0.5*q) + lc*ash*ut + lp*ap*(H+un*a);
    double f0=0.5*(Fl0+Fr0)-0.5*d0, fn=0.5*(Fl1n+Fr1n)-0.5*d1n;
    double ft=0.5*(Fl1t+Fr1t)-0.5*d1t, f3=0.5*(Fl3+Fr3)-0.5*d3;
    F[0]=f0; F[1]=fn*nx+ft*tx; F[2]=fn*ny+ft*ty; F[3]=f3;
    if (!std::isfinite(F[0]+F[1]+F[2]+F[3])) hllc_euler2d(eq, WL, WR, nx, ny, F);
}

} // namespace cfd
