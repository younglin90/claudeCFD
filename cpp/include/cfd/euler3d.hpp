// cfd/euler3d.hpp — 3D compressible Euler (gamma-law) + LLF/HLLC/rotated-HLLC.
// 3D extension of euler2d.hpp for nvar=5.
//   U=(rho,rho u,rho v,rho w,rho E), W=(rho,u,v,w,p),
//   e=p/((g-1)rho), E=e+(u^2+v^2+w^2)/2.
// Scalar per-face leaf functions (acc routine seq) for OpenMP/OpenACC.
// HLLC star state carries the FULL tangential velocity u_t = u - (u.n) n
// (no explicit tangent basis needed in 3D); rotated-HLLC solves the Riemann
// problem in the plane spanned by the face normal and the velocity jump
// (n1 = dv/|dv|, n2 = Gram-Schmidt of n against n1) — the natural 3D mirror of
// the 2D Nishikawa-Kitamura rotated flux.
#pragma once
#include "cfd/eos.hpp"   // CFD_ROUTINE_SEQ
#include <cmath>

namespace cfd {

constexpr double EU3_EPS = 1e-30;

struct Euler3D {
    double gamma = 1.4;
    static constexpr int nvar = 5;

    CFD_ROUTINE_SEQ void prim_to_cons(const double W[5], double U[5]) const {
        double rho = W[0] > EU3_EPS ? W[0] : EU3_EPS;
        double u = W[1], v = W[2], w = W[3], p = W[4];
        double e = p / ((gamma - 1.0) * rho);
        double E = e + 0.5 * (u*u + v*v + w*w);
        U[0] = rho; U[1] = rho*u; U[2] = rho*v; U[3] = rho*w; U[4] = rho*E;
    }
    CFD_ROUTINE_SEQ void cons_to_prim(const double U[5], double W[5]) const {
        double rho = U[0] > EU3_EPS ? U[0] : EU3_EPS;
        double u = U[1]/rho, v = U[2]/rho, w = U[3]/rho, E = U[4]/rho;
        double p = (gamma - 1.0) * rho * (E - 0.5*(u*u + v*v + w*w));
        W[0] = rho; W[1] = u; W[2] = v; W[3] = w; W[4] = p;
    }
    CFD_ROUTINE_SEQ void physical_flux(const double U[5], double nx, double ny, double nz, double F[5]) const {
        double rho = U[0] > EU3_EPS ? U[0] : EU3_EPS;
        double u = U[1]/rho, v = U[2]/rho, w = U[3]/rho, E = U[4]/rho;
        double p = (gamma - 1.0) * rho * (E - 0.5*(u*u + v*v + w*w));
        double un = u*nx + v*ny + w*nz;
        F[0] = rho*un;
        F[1] = rho*u*un + p*nx;
        F[2] = rho*v*un + p*ny;
        F[3] = rho*w*un + p*nz;
        F[4] = (U[4] + p)*un;
    }
    CFD_ROUTINE_SEQ double max_wave_speed(const double U[5], double nx, double ny, double nz) const {
        double rho = U[0] > EU3_EPS ? U[0] : EU3_EPS;
        double u = U[1]/rho, v = U[2]/rho, w = U[3]/rho, E = U[4]/rho;
        double p = (gamma - 1.0) * rho * (E - 0.5*(u*u + v*v + w*w));
        double un = u*nx + v*ny + w*nz;
        double c2 = gamma * p / rho; if (c2 < EU3_EPS) c2 = EU3_EPS;
        return std::fabs(un) + std::sqrt(c2);
    }
};

// Local Lax-Friedrichs (Rusanov).
CFD_ROUTINE_SEQ
inline void llf_euler3d(const Euler3D& eq, const double WL[5], const double WR[5],
                        double nx, double ny, double nz, double F[5]) {
    double UL[5], UR[5], FL[5], FR[5];
    eq.prim_to_cons(WL, UL); eq.prim_to_cons(WR, UR);
    eq.physical_flux(UL, nx, ny, nz, FL); eq.physical_flux(UR, nx, ny, nz, FR);
    double lamL = eq.max_wave_speed(UL, nx, ny, nz), lamR = eq.max_wave_speed(UR, nx, ny, nz);
    double lam = lamL > lamR ? lamL : lamR;
    for (int k = 0; k < 5; ++k) F[k] = 0.5*(FL[k] + FR[k]) - 0.5*lam*(UR[k] - UL[k]);
}

CFD_ROUTINE_SEQ inline double sgn3(double x) { return (x > 0) - (x < 0); }
CFD_ROUTINE_SEQ inline double safe_den3(double d) {
    return (std::fabs(d) > EU3_EPS) ? d : (sgn3(d) * EU3_EPS + EU3_EPS);
}

// Shock-stabilized HLLC/HLLE-hybrid for 3D Euler (mirror of hllc_euler2d). The star
// state carries u_t = u - un n (full tangential velocity), so no tangent basis is
// needed. Blends toward HLL at strong shocks (carbuncle cure).
CFD_ROUTINE_SEQ
inline void hllc_euler3d(const Euler3D& eq, const double WL[5], const double WR[5],
                         double nx, double ny, double nz, double F[5]) {
    double rL = WL[0] > EU3_EPS ? WL[0] : EU3_EPS, rR = WR[0] > EU3_EPS ? WR[0] : EU3_EPS;
    double uL = WL[1], vL = WL[2], wL_ = WL[3], pL = WL[4] > EU3_EPS ? WL[4] : EU3_EPS;
    double uR = WR[1], vR = WR[2], wR_ = WR[3], pR = WR[4] > EU3_EPS ? WR[4] : EU3_EPS;
    double unL = uL*nx + vL*ny + wL_*nz, unR = uR*nx + vR*ny + wR_*nz;
    double g = eq.gamma;
    double cL = std::sqrt(std::max(g*pL/rL, EU3_EPS)), cR = std::sqrt(std::max(g*pR/rR, EU3_EPS));
    double EL = pL/((g-1.0)*rL) + 0.5*(uL*uL+vL*vL+wL_*wL_);
    double ER = pR/((g-1.0)*rR) + 0.5*(uR*uR+vR*vR+wR_*wR_);
    double UL[5], UR[5], FL[5], FR[5];
    eq.prim_to_cons(WL, UL); eq.prim_to_cons(WR, UR);
    eq.physical_flux(UL, nx, ny, nz, FL); eq.physical_flux(UR, nx, ny, nz, FR);

    double SL = std::min(unL - cL, unR - cR), SR = std::max(unL + cL, unR + cR);
    double den = safe_den3(rL*(SL-unL) - rR*(SR-unR));
    double SM = (pR - pL + rL*unL*(SL-unL) - rR*unR*(SR-unR)) / den;

    // star state: tangential velocity u_t = u - un n preserved; normal -> SM.
    auto star = [&](double rho, double u, double v, double w, double un, double p, double E,
                    double S, double Us[5]) {
        double ds = safe_den3(S - SM);
        double fac = rho*(S - un)/ds;
        double wd = safe_den3(rho*(S - un));
        double utx = u - un*nx, uty = v - un*ny, utz = w - un*nz;  // tangential (3-vector)
        Us[0] = fac;
        Us[1] = fac*(SM*nx + utx);
        Us[2] = fac*(SM*ny + uty);
        Us[3] = fac*(SM*nz + utz);
        Us[4] = fac*(E + (SM - un)*(SM + p/wd));
    };
    double UsL[5], UsR[5];
    star(rL,uL,vL,wL_,unL,pL,EL,SL,UsL);
    star(rR,uR,vR,wR_,unR,pR,ER,SR,UsR);
    double Fh[5], Fhll[5];
    double dh = std::max(SR - SL, EU3_EPS);
    for (int k = 0; k < 5; ++k) {
        if (SL >= 0.0) Fh[k] = FL[k];
        else if (SM >= 0.0) Fh[k] = FL[k] + SL*(UsL[k] - UL[k]);
        else if (SR > 0.0) Fh[k] = FR[k] + SR*(UsR[k] - UR[k]);
        else Fh[k] = FR[k];
        if (SL >= 0.0) Fhll[k] = FL[k];
        else if (SR <= 0.0) Fhll[k] = FR[k];
        else Fhll[k] = (SR*FL[k] - SL*FR[k] + SL*SR*(UR[k]-UL[k]))/dh;
    }
    double pj = std::fabs(pR - pL) / std::max(pR + pL, EU3_EPS);
    double comp = std::max(0.0, unL - unR) / std::max(cL + cR, EU3_EPS);
    double shock = std::min(std::max((pj - 0.05)/0.35, 0.0), 1.0)
                 * std::min(std::max(4.0*comp, 0.0), 1.0);
    double blend = 0.45 * shock;
    for (int k = 0; k < 5; ++k) F[k] = (1.0 - blend)*Fh[k] + blend*Fhll[k];
}

// Standard HLL (2-wave, no contact) — robust, more dissipative.
CFD_ROUTINE_SEQ
inline void hll_euler3d(const Euler3D& eq, const double WL[5], const double WR[5],
                        double nx, double ny, double nz, double F[5]) {
    double rL = WL[0] > EU3_EPS ? WL[0] : EU3_EPS, rR = WR[0] > EU3_EPS ? WR[0] : EU3_EPS;
    double uL = WL[1], vL = WL[2], wL_ = WL[3], pL = WL[4] > EU3_EPS ? WL[4] : EU3_EPS;
    double uR = WR[1], vR = WR[2], wR_ = WR[3], pR = WR[4] > EU3_EPS ? WR[4] : EU3_EPS;
    double unL = uL*nx + vL*ny + wL_*nz, unR = uR*nx + vR*ny + wR_*nz;
    double g = eq.gamma;
    double cL = std::sqrt(std::max(g*pL/rL, EU3_EPS)), cR = std::sqrt(std::max(g*pR/rR, EU3_EPS));
    double UL[5], UR[5], FL[5], FR[5];
    eq.prim_to_cons(WL, UL); eq.prim_to_cons(WR, UR);
    eq.physical_flux(UL, nx, ny, nz, FL); eq.physical_flux(UR, nx, ny, nz, FR);
    double SL = std::min(unL - cL, unR - cR), SR = std::max(unL + cL, unR + cR);
    double dh = std::max(SR - SL, EU3_EPS);
    for (int k = 0; k < 5; ++k) {
        if (SL >= 0.0) F[k] = FL[k];
        else if (SR <= 0.0) F[k] = FR[k];
        else F[k] = (SR*FL[k] - SL*FR[k] + SL*SR*(UR[k]-UL[k]))/dh;
    }
}

// Rotated-HLLC (Nishikawa-Kitamura, 3D): n1 = velocity-jump direction (shock
// normal), n2 = component of the face normal perpendicular to n1 (Gram-Schmidt).
// n lies in span{n1,n2}, so F = (n.n1) HLLC(n1) + (n.n2) HLLC(n2). Carbuncle-free
// at grid-aligned strong shocks.
CFD_ROUTINE_SEQ
inline void rotated_hllc_euler3d(const Euler3D& eq, const double WL[5], const double WR[5],
                                 double nx, double ny, double nz, double F[5]) {
    double dvx = WR[1]-WL[1], dvy = WR[2]-WL[2], dvz = WR[3]-WL[3];
    double dv = std::sqrt(dvx*dvx + dvy*dvy + dvz*dvz);
    double n1x, n1y, n1z;
    if (dv > 1e-12) { n1x = dvx/dv; n1y = dvy/dv; n1z = dvz/dv; }
    else            { n1x = nx;     n1y = ny;     n1z = nz;     }
    double a1 = n1x*nx + n1y*ny + n1z*nz;
    if (a1 < 0.0) { n1x = -n1x; n1y = -n1y; n1z = -n1z; a1 = -a1; }   // n1.n >= 0
    // n2 = normalize(n - a1 n1); a2 = |n - a1 n1| = sqrt(1-a1^2).
    double tx = nx - a1*n1x, ty = ny - a1*n1y, tz = nz - a1*n1z;
    double a2 = std::sqrt(tx*tx + ty*ty + tz*tz);
    if (a2 > 1e-12) {
        double n2x = tx/a2, n2y = ty/a2, n2z = tz/a2;
        double F1[5], F2[5];
        hllc_euler3d(eq, WL, WR, n1x, n1y, n1z, F1);
        hllc_euler3d(eq, WL, WR, n2x, n2y, n2z, F2);
        for (int k = 0; k < 5; ++k) F[k] = a1*F1[k] + a2*F2[k];
    } else {
        hllc_euler3d(eq, WL, WR, nx, ny, nz, F);   // n parallel to n1: pure normal
    }
}

// ── Harten entropy fix: smooth |lambda| near zero (kills sonic expansion-shock glitch).
CFD_ROUTINE_SEQ inline double harten_ef3_(double lam, double delta) {
    double al = std::fabs(lam);
    return (al >= delta) ? al : 0.5*(lam*lam/(delta > EU3_EPS ? delta : EU3_EPS) + delta);
}

// 3D Roe flux with Harten entropy fix on the genuinely-nonlinear ACOUSTIC waves (un±a) only;
// the linearly-degenerate contact (un) + 2 shear waves are left exact -> contacts/shears stay
// sharp. delta = 0.1*(|un|+a). Tangent basis t1,t2 built from the least-aligned axis. Conserved-
// variable dissipation rotated back to (x,y,z). Falls back to HLLC if a^2<=0 or non-finite.
CFD_ROUTINE_SEQ
inline void roe_euler3d(const Euler3D& eq, const double WL[5], const double WR[5],
                        double nx, double ny, double nz, double F[5]) {
    const double g = eq.gamma, eps = EU3_EPS;
    double rL = WL[0]>eps?WL[0]:eps, rR = WR[0]>eps?WR[0]:eps;
    double pL = WL[4]>eps?WL[4]:eps, pR = WR[4]>eps?WR[4]:eps;
    // tangent basis: t1 = normalize(n x e), e = least-aligned axis; t2 = n x t1
    double ax=std::fabs(nx), ay=std::fabs(ny), az=std::fabs(nz);
    double ex=0,ey=0,ez=0; if(ax<=ay&&ax<=az)ex=1; else if(ay<=az)ey=1; else ez=1;
    double t1x=ny*ez-nz*ey, t1y=nz*ex-nx*ez, t1z=nx*ey-ny*ex;
    double t1n=std::sqrt(t1x*t1x+t1y*t1y+t1z*t1z); if(t1n<eps){t1x=1;t1y=0;t1z=0;t1n=1;}
    t1x/=t1n; t1y/=t1n; t1z/=t1n;
    double t2x=ny*t1z-nz*t1y, t2y=nz*t1x-nx*t1z, t2z=nx*t1y-ny*t1x;
    double uL=WL[1],vL=WL[2],wL_=WL[3], uR=WR[1],vR=WR[2],wR_=WR[3];
    double unL=uL*nx+vL*ny+wL_*nz, unR=uR*nx+vR*ny+wR_*nz;
    double ut1L=uL*t1x+vL*t1y+wL_*t1z, ut1R=uR*t1x+vR*t1y+wR_*t1z;
    double ut2L=uL*t2x+vL*t2y+wL_*t2z, ut2R=uR*t2x+vR*t2y+wR_*t2z;
    double EL=pL/((g-1.0)*rL)+0.5*(uL*uL+vL*vL+wL_*wL_), ER=pR/((g-1.0)*rR)+0.5*(uR*uR+vR*vR+wR_*wR_);
    double HL=EL+pL/rL, HR=ER+pR/rR;
    double srL=std::sqrt(rL), srR=std::sqrt(rR), inv=1.0/(srL+srR);
    double un=(srL*unL+srR*unR)*inv, ut1=(srL*ut1L+srR*ut1R)*inv, ut2=(srL*ut2L+srR*ut2R)*inv;
    double H=(srL*HL+srR*HR)*inv, rho=srL*srR;
    double q=un*un+ut1*ut1+ut2*ut2, a2=(g-1.0)*(H-0.5*q);
    if(a2<=eps){ hllc_euler3d(eq,WL,WR,nx,ny,nz,F); return; }
    double a=std::sqrt(a2), ia2=1.0/a2;
    double drho=rR-rL, dun=unR-unL, dp=pR-pL, dut1=ut1R-ut1L, dut2=ut2R-ut2L;
    double am=0.5*(dp*ia2 - rho*dun/a), ap=0.5*(dp*ia2 + rho*dun/a);   // un-a, un+a strengths
    double aent=drho - dp*ia2;                                          // entropy strength
    double delta=0.1*(std::fabs(un)+a);
    double lm=harten_ef3_(un-a,delta), lc=std::fabs(un), lp=harten_ef3_(un+a,delta);
    double s_ac = lm*am + lc*aent + lp*ap;        // common (mass-like) acoustic+entropy sum
    double D_rho = s_ac;
    double D_mn  = lm*am*(un-a) + lc*aent*un + lp*ap*(un+a);
    double D_mt1 = s_ac*ut1 + lc*rho*dut1;
    double D_mt2 = s_ac*ut2 + lc*rho*dut2;
    double D_E   = lm*am*(H-un*a) + lc*aent*0.5*q + lc*rho*dut1*ut1 + lc*rho*dut2*ut2 + lp*ap*(H+un*a);
    // rotate momentum dissipation back to (x,y,z)
    double D_mx = D_mn*nx + D_mt1*t1x + D_mt2*t2x;
    double D_my = D_mn*ny + D_mt1*t1y + D_mt2*t2y;
    double D_mz = D_mn*nz + D_mt1*t1z + D_mt2*t2z;
    double UL[5],UR[5],FL[5],FR[5];
    eq.prim_to_cons(WL,UL); eq.prim_to_cons(WR,UR);
    eq.physical_flux(UL,nx,ny,nz,FL); eq.physical_flux(UR,nx,ny,nz,FR);
    F[0]=0.5*(FL[0]+FR[0])-0.5*D_rho;
    F[1]=0.5*(FL[1]+FR[1])-0.5*D_mx;
    F[2]=0.5*(FL[2]+FR[2])-0.5*D_my;
    F[3]=0.5*(FL[3]+FR[3])-0.5*D_mz;
    F[4]=0.5*(FL[4]+FR[4])-0.5*D_E;
    if(!std::isfinite(F[0]+F[1]+F[2]+F[3]+F[4])) hllc_euler3d(eq,WL,WR,nx,ny,nz,F);
}

// Rotated-Roe hybrid (3D mirror of rotated_roe_euler2d): HLLC in the velocity-jump direction
// n1 (shock-normal, dissipative & carbuncle-free) + Roe in the in-plane perpendicular n2
// (low-diffusion contact/shear). F = (n.n1) HLLC(n1) + (n.n2) Roe(n2).
CFD_ROUTINE_SEQ
inline void rotated_roe_euler3d(const Euler3D& eq, const double WL[5], const double WR[5],
                                double nx, double ny, double nz, double F[5]) {
    double dvx=WR[1]-WL[1], dvy=WR[2]-WL[2], dvz=WR[3]-WL[3];
    double dv=std::sqrt(dvx*dvx+dvy*dvy+dvz*dvz);
    double n1x,n1y,n1z;
    if(dv>1e-12){ n1x=dvx/dv; n1y=dvy/dv; n1z=dvz/dv; } else { n1x=nx; n1y=ny; n1z=nz; }
    double a1=n1x*nx+n1y*ny+n1z*nz;
    if(a1<0.0){ n1x=-n1x; n1y=-n1y; n1z=-n1z; a1=-a1; }
    double tx=nx-a1*n1x, ty=ny-a1*n1y, tz=nz-a1*n1z;
    double a2=std::sqrt(tx*tx+ty*ty+tz*tz);
    if(a2>1e-12){
        double n2x=tx/a2, n2y=ty/a2, n2z=tz/a2; double F1[5],F2[5];
        hllc_euler3d(eq,WL,WR,n1x,n1y,n1z,F1);    // HLLC in shock-normal
        roe_euler3d (eq,WL,WR,n2x,n2y,n2z,F2);    // Roe in perpendicular (low-diffusion)
        for(int k=0;k<5;++k) F[k]=a1*F1[k]+a2*F2[k];
    } else {
        hllc_euler3d(eq,WL,WR,nx,ny,nz,F);
    }
}

// SLAU2 all-Mach flux (Kitamura & Shima, JCP 245 (2013) 62-83) — 3D single-species. Faithful
// 3D port of slau2_euler2d: low-dissipation at low Mach (chi=(1-Mhat)^2 scales the pressure-
// diffusion in the mass flux) + shock-stable SLAU2 pressure flux. Tangential momentum carried
// by the (upwinded) mass flux; only the normal velocity Vn changes vs 2D.
CFD_ROUTINE_SEQ
inline void slau2_euler3d(const Euler3D& eq, const double WL[5], const double WR[5],
                          double nx, double ny, double nz, double F[5]) {
    const double g = eq.gamma;
    double rhoL=WL[0]>EU3_EPS?WL[0]:EU3_EPS, uL=WL[1], vL=WL[2], wL_=WL[3], pL=WL[4]>EU3_EPS?WL[4]:EU3_EPS;
    double rhoR=WR[0]>EU3_EPS?WR[0]:EU3_EPS, uR=WR[1], vR=WR[2], wR_=WR[3], pR=WR[4]>EU3_EPS?WR[4]:EU3_EPS;
    double VnL=uL*nx+vL*ny+wL_*nz, VnR=uR*nx+vR*ny+wR_*nz;
    double cL=std::sqrt(g*pL/rhoL), cR=std::sqrt(g*pR/rhoR);
    double cb=0.5*(cL+cR); if(cb<1e-300) cb=1e-300;
    double ML=VnL/cb, MR=VnR/cb, aML=std::fabs(ML), aMR=std::fabs(MR);
    double Mp=(aML>=1.0)? 0.5*(ML+aML) :  0.25*(ML+1.0)*(ML+1.0);
    double Mm=(aMR>=1.0)? 0.5*(MR-aMR) : -0.25*(MR-1.0)*(MR-1.0);
    double Pp=(aML>=1.0)? (ML>0.0?1.0:0.0) : 0.25*(ML+1.0)*(ML+1.0)*(2.0-ML);
    double Pm=(aMR>=1.0)? (MR>0.0?0.0:1.0) : 0.25*(MR-1.0)*(MR-1.0)*(2.0+MR);
    double Mbar=std::sqrt(0.5*(VnL*VnL+VnR*VnR))/cb; double Mhat=Mbar<1.0?Mbar:1.0; double chi=(1.0-Mhat)*(1.0-Mhat);
    double bLm=std::max(std::min(ML,0.0),-1.0), bRp=std::min(std::max(MR,0.0),1.0); double gg=-bLm*bRp;
    double VpL=(1.0-gg)*cb*Mp + gg*std::fabs(VnL);
    double VmR=(1.0-gg)*cb*Mm - gg*std::fabs(VnR);
    double mdot=0.5*(rhoL*(VnL+std::fabs(VpL)) + rhoR*(VnR-std::fabs(VmR))) - chi/(2.0*cb)*(pR-pL);
    double ubar=std::sqrt(0.5*(VnL*VnL+VnR*VnR));
    double pface=0.5*(pL+pR) + 0.5*(Pp-Pm)*(pL-pR) + ubar*(Pp+Pm-1.0)*0.5*(rhoL+rhoR)*cb;
    double rhoEL=pL/(g-1.0)+0.5*rhoL*(uL*uL+vL*vL+wL_*wL_), rhoER=pR/(g-1.0)+0.5*rhoR*(uR*uR+vR*vR+wR_*wR_);
    double HL=(rhoEL+pL)/rhoL, HR=(rhoER+pR)/rhoR;
    bool pos=mdot>=0.0;
    double u_up=pos?uL:uR, v_up=pos?vL:vR, w_up=pos?wL_:wR_, H_up=pos?HL:HR;
    F[0]=mdot;
    F[1]=mdot*u_up + pface*nx;
    F[2]=mdot*v_up + pface*ny;
    F[3]=mdot*w_up + pface*nz;
    F[4]=mdot*H_up;
    if(!std::isfinite(F[0]+F[1]+F[2]+F[3]+F[4])) hllc_euler3d(eq,WL,WR,nx,ny,nz,F);
}

// Rotated SLAU2 (3D mirror of rotated_slau2_euler2d): SLAU2 in n1 (velocity-jump = shock
// normal) + SLAU2 in the in-plane perpendicular n2. F = (n.n1)SLAU2(n1)+(n.n2)SLAU2(n2).
CFD_ROUTINE_SEQ
inline void rotated_slau2_euler3d(const Euler3D& eq, const double WL[5], const double WR[5],
                                  double nx, double ny, double nz, double F[5]) {
    double dvx=WR[1]-WL[1], dvy=WR[2]-WL[2], dvz=WR[3]-WL[3];
    double dv=std::sqrt(dvx*dvx+dvy*dvy+dvz*dvz);
    double n1x,n1y,n1z;
    if(dv>1e-12){ n1x=dvx/dv; n1y=dvy/dv; n1z=dvz/dv; } else { n1x=nx; n1y=ny; n1z=nz; }
    double a1=n1x*nx+n1y*ny+n1z*nz; if(a1<0.0){ n1x=-n1x; n1y=-n1y; n1z=-n1z; a1=-a1; }
    double tx=nx-a1*n1x, ty=ny-a1*n1y, tz=nz-a1*n1z; double a2=std::sqrt(tx*tx+ty*ty+tz*tz);
    if(a2>1e-12){
        double n2x=tx/a2, n2y=ty/a2, n2z=tz/a2; double F1[5],F2[5];
        slau2_euler3d(eq,WL,WR,n1x,n1y,n1z,F1);
        slau2_euler3d(eq,WL,WR,n2x,n2y,n2z,F2);
        for(int k=0;k<5;++k) F[k]=a1*F1[k]+a2*F2[k];
    } else {
        slau2_euler3d(eq,WL,WR,nx,ny,nz,F);
    }
}

} // namespace cfd
