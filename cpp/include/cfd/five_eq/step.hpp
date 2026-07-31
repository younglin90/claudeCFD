// cfd/five_eq/step.hpp — M10 production step driver (one imex_ad_step).
//
// Header-only C++ twin of solver/five_eq_IMEX/imex_ad.py::imex_ad_step
// (3902-4090) for the production BASE_ENV configuration.  Stage order:
//   1. regime_auto pressure-closure pick (M8, regime_auto.hpp)
//   2. material_update (M6)  -> q1_new,q2_new,m_adv,rhoE_adv,alpha_new
//   3. clip alpha to [1e-12, 1-1e-12]
//   4. acoustic_solve (M7)   -> u_new,p_new
//   5. closure (pressure_work_consistent / compressive_recovery):
//        recompute Z from the W^n anchor, build acoustic-Riemann faces from
//        (u_new,p_new,Z), rebuild energy
//          rhoE_new = rhoE_adv - dt*(p_f[1:]*u_f[1:] - p_f[:-1]*u_f[:-1])/dx
//        then the PW pure-shock recovery mask (compressive & ~pure) via
//        _recover_pressure_from_total_energy — inactive for 02A/07B acoustic.
//   6. near-vacuum velocity regularisation (inactive away from cavitation)
//   7. primitive LMP/LED filter — 'auto' -> off unless W^n has a pressure jump
//   8. explicit primitive recovery W_new = (alpha, T1, T2, u_new, p_new)
//
// The optional pressure closures are reconstructed from the material state;
// their Python-oracle coverage lives alongside the production step tests.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "cfd/eos.hpp"
#include "cfd/five_eq/config.hpp"
#include "cfd/five_eq/regime_auto.hpp"
#include "cfd/five_eq/material_update.hpp"
#include "cfd/five_eq/acoustic_solve.hpp"
#include "cfd/five_eq/sound_speed.hpp"

namespace cfd {
namespace five_eq {

struct StepResult {
    std::vector<double> alpha, T1, T2, u, p;   // W_new
    PressureClosure closure = PressureClosure::pressure_work_consistent;
    int vacuum_velocity_cells = 0;
};

namespace step_detail {

constexpr double MU_EPS = 1e-30;

// dp/de |_rho, analytic per EOS form (matches He2024 eos.dpde_rho):
//   Ideal/SG: (gamma-1)*rho ;  NASG: (gamma-1)*rho / max(1-b*rho, 1e-10).
inline double dpde_rho(const EOS& e, double rho, double /*ener*/) {
    if (e.kind == EOS::NASG) {
        double denom = EOS::max2(1.0 - e.b * rho, 1e-10);
        return (e.gamma - 1.0) * rho / denom;
    }
    return (e.gamma - 1.0) * rho;
}

inline AcousticBC to_acoustic_bc(BC5 b, bool left_side) {
    switch (b) {
        case BC5::Periodic:     return AcousticBC::periodic;
        case BC5::Reflective:   return AcousticBC::reflective;
        case BC5::Transmissive: return AcousticBC::transmissive;
        case BC5::Inlet:        return AcousticBC::inlet;
        case BC5::Dirichlet:    return left_side ? AcousticBC::inlet : AcousticBC::outlet;
        case BC5::InletAcoustic:return AcousticBC::inlet_acoustic;
        case BC5::Outlet:       return AcousticBC::outlet;
    }
    return AcousticBC::transmissive;
}

// _extend_np: 1-ghost extension.  odd => reflective flips sign; transmissive
// copies; periodic wraps.  (Matches imex_ad._extend_np for the 3 BC kinds.)
inline std::vector<double> extend(const std::vector<double>& a,
                                  BC5 bc_l, BC5 bc_r, bool odd) {
    const int n = (int)a.size();
    std::vector<double> e(n + 2);
    for (int i = 0; i < n; ++i) e[i + 1] = a[i];
    if (bc_l == BC5::Periodic && bc_r == BC5::Periodic) {
        e[0] = a[n - 1]; e[n + 1] = a[0];
        return e;
    }
    e[0]     = (odd && bc_l == BC5::Reflective) ? -a[0]     : a[0];
    e[n + 1] = (odd && bc_r == BC5::Reflective) ? -a[n - 1] : a[n - 1];
    return e;
}

// _acoustic_faces_np (imex_ad.py 464-513): first-order acoustic Z-Riemann faces.
struct Faces { std::vector<double> pf, uf; };  // length n+1
inline Faces acoustic_faces(const std::vector<double>& u,
                            const std::vector<double>& p,
                            const std::vector<double>& Z,
                            BC5 bc_l, BC5 bc_r,
                            std::optional<double> u_inlet_l = {},
                            std::optional<double> p_inlet_l = {},
                            std::optional<double> p_outlet_r = {}) {
    const int n = (int)u.size();
    const double EPS = MU_EPS;
    std::vector<double> ue(n + 2), pe(n + 2), Ze(n + 2);
    const bool periodic = (bc_l == BC5::Periodic && bc_r == BC5::Periodic);
    for (int i = 0; i < n; ++i) { ue[i + 1] = u[i]; pe[i + 1] = p[i]; Ze[i + 1] = Z[i]; }
    if (periodic) {
        ue[0] = u[n - 1]; pe[0] = p[n - 1]; Ze[0] = Z[n - 1];
        ue[n + 1] = u[0]; pe[n + 1] = p[0]; Ze[n + 1] = Z[0];
    } else {
        double ul = (bc_l == BC5::Reflective) ? -u[0]
                  : ((bc_l == BC5::Inlet || bc_l == BC5::InletAcoustic || bc_l == BC5::Dirichlet) && u_inlet_l ? *u_inlet_l : u[0]);
        double pl = ((bc_l == BC5::Inlet || bc_l == BC5::InletAcoustic || bc_l == BC5::Dirichlet) && p_inlet_l) ? *p_inlet_l : p[0];
        double ur = (bc_r == BC5::Reflective) ? -u[n - 1] : u[n - 1];
        double pr = ((bc_r == BC5::Outlet || bc_r == BC5::Dirichlet) && p_outlet_r) ? *p_outlet_r : p[n - 1];
        ue[0] = ul; pe[0] = pl; Ze[0] = Z[0];
        ue[n + 1] = ur; pe[n + 1] = pr; Ze[n + 1] = Z[n - 1];
    }
    Faces fc; fc.pf.resize(n + 1); fc.uf.resize(n + 1);
    for (int f = 0; f < n + 1; ++f) {
        double ZL = Ze[f], ZR = Ze[f + 1];
        double pL = pe[f], pR = pe[f + 1];
        double uL = ue[f], uR = ue[f + 1];
        double den = std::fmax(ZL + ZR, EPS);
        fc.pf[f] = (ZR * pL + ZL * pR + ZL * ZR * (uL - uR)) / den;
        fc.uf[f] = (pL - pR + ZL * uL + ZR * uR) / den;
    }
    if (bc_l == BC5::Reflective) { fc.pf[0] = p[0]; fc.uf[0] = 0.0; }
    if (bc_r == BC5::Reflective) { fc.pf[n] = p[n - 1]; fc.uf[n] = 0.0; }
    return fc;
}

// _recover_pressure_from_total_energy (imex_ad.py ~50-90): scalar Newton on the
// mixture internal-energy constraint, 16 iterations, per cell.
inline std::vector<double> recover_pressure_from_total_energy(
        const std::vector<double>& q1, const std::vector<double>& q2,
        const std::vector<double>& rhoE, const std::vector<double>& alpha,
        const std::vector<double>& u, const std::vector<double>& p_seed,
        const EOS& eos1, const EOS& eos2) {
    const int n = (int)q1.size();
    const double EPS = MU_EPS;
    std::vector<double> p_out(n);
    for (int i = 0; i < n; ++i) {
        double ac = std::fmin(std::fmax(alpha[i], 1.0e-12), 1.0 - 1.0e-12);
        double rho = std::fmax(q1[i] + q2[i], EPS);
        double rho1 = std::fmax(q1[i] / ac, EPS);
        double rho2 = std::fmax(q2[i] / std::fmax(1.0 - ac, 1.0e-12), EPS);
        double target = rhoE[i] - 0.5 * rho * u[i] * u[i];
        double p = std::fmax(p_seed[i], 1.0e-12);
        for (int it = 0; it < 16; ++it) {
            double e1 = eos1.energy(rho1, p);
            double e2 = eos2.energy(rho2, p);
            double f = ac * rho1 * e1 + (1.0 - ac) * rho2 * e2 - target;
            double dpde1 = std::fmax(dpde_rho(eos1, rho1, e1), EPS);
            double dpde2 = std::fmax(dpde_rho(eos2, rho2, e2), EPS);
            double df = ac * rho1 / dpde1 + (1.0 - ac) * rho2 / dpde2;
            double stepv = f / std::fmax(df, EPS);
            double p_next = std::fmax(p - stepv, 1.0e-12);
            if (std::fabs(p_next - p) / std::fmax(p_next, 1.0) < 1.0e-10) {
                p = p_next; break;
            }
            p = p_next;
        }
        p_out[i] = p;
    }
    return p_out;
}

// _regularize_near_vacuum_velocity (imex_ad.py 272-318): smooth velocity only in
// expanding near-vacuum (density AND pressure collapsed, or low-pressure vacuum).
inline std::vector<double> regularize_near_vacuum_velocity(
        const std::vector<double>& alpha_n, const std::vector<double>& T1_n,
        const std::vector<double>& T2_n, const std::vector<double>& p_n,
        const std::vector<double>& q1_new, const std::vector<double>& q2_new,
        std::vector<double> u_new, const std::vector<double>& p_new,
        const EOS& eos1, const EOS& eos2, double alpha_pure_tol,
        BC5 bc_l, BC5 bc_r, int& mask_count, int passes = 6,
        MixtureSoundSpeedKind mixture_kind = MixtureSoundSpeedKind::Kapila) {
    const int n = (int)u_new.size();
    const double EPS = MU_EPS;
    std::vector<double> rho(n), rho_anchor(n), p_anchor(n);
    double rho_domain = 1.0, p_domain = 1.0;
    for (int i = 0; i < n; ++i) {
        rho[i] = std::fmax(q1_new[i] + q2_new[i], EPS);
        PhaseAcoustic pa = phase_acoustic(eos1, eos2, alpha_n[i], T1_n[i], T2_n[i],
                                          p_n[i], alpha_pure_tol, mixture_kind);
        rho_anchor[i] = std::fmax(pa.rho, EPS);
        p_anchor[i]   = std::fmax(std::fabs(p_n[i]), 1.0);
        rho_domain = std::fmax(rho_domain, rho_anchor[i]);
        p_domain   = std::fmax(p_domain, p_anchor[i]);
    }
    std::vector<double> ue = extend(u_new, bc_l, bc_r, /*odd=*/true);
    std::vector<char> mask(n, 0);
    int cnt = 0;
    for (int i = 0; i < n; ++i) {
        bool expanding = (ue[i + 1] - ue[i] > 0.0) || (ue[i + 2] - ue[i + 1] > 0.0);
        bool dens_col = (rho[i] < 1.0e-2 * rho_anchor[i]) || (rho[i] < 1.0e-3 * rho_domain);
        bool pres_col = (p_new[i] < 1.0e-2 * p_anchor[i]) || (p_new[i] < 1.0e-3 * p_domain);
        bool low_vac  = p_new[i] < std::fmin(5.0e-2 * p_domain, 5.0e3);
        if (expanding && ((dens_col && pres_col) || low_vac)) { mask[i] = 1; ++cnt; }
    }
    mask_count = cnt;
    if (cnt == 0) return u_new;
    std::vector<double> u_reg = u_new;
    for (int pass = 0; pass < std::max(passes, 1); ++pass) {
        std::vector<double> ee = extend(u_reg, bc_l, bc_r, true);
        std::vector<double> u_next = u_reg;
        for (int i = 0; i < n; ++i) {
            if (mask[i]) u_next[i] = 0.25 * ee[i] + 0.5 * ee[i + 1] + 0.25 * ee[i + 2];
        }
        u_reg.swap(u_next);
    }
    return u_reg;
}

inline void apply_primitive_filter(const std::vector<double>& u_old,
                                   const std::vector<double>& p_old,
                                   std::vector<double>& u_new, std::vector<double>& p_new,
                                   BC5 bc_l, BC5 bc_r, PrimitiveFilter mode) {
    if (mode == PrimitiveFilter::Off) return;
    bool pressure_jump = false;
    for (int i = 0; i + 1 < (int)p_old.size(); ++i) {
        const double scale = std::fmax(std::fmax(std::fabs(p_old[i]), std::fabs(p_old[i + 1])), 1.0);
        if (std::fabs(p_old[i + 1] - p_old[i]) / scale > regime_detail::eps025()) { pressure_jump = true; break; }
    }
    if (mode == PrimitiveFilter::Auto && !pressure_jump) return;
    if (mode == PrimitiveFilter::GlobalPressure) {
        double lo = p_old[0], hi = p_old[0];
        for (double v : p_old) { lo = std::fmin(lo, v); hi = std::fmax(hi, v); }
        for (double& v : p_new) v = std::fmax(std::fmin(std::fmax(v, lo), hi), 1.0e-12);
        return;
    }
    const std::vector<double> ue = extend(u_old, bc_l, bc_r, true);
    const std::vector<double> pe = extend(p_old, bc_l, bc_r, false);
    for (int i = 0; i < (int)u_new.size(); ++i) {
        if (mode == PrimitiveFilter::Stencil) {
            const double ulo = std::fmin(ue[i], std::fmin(ue[i + 1], ue[i + 2]));
            const double uhi = std::fmax(ue[i], std::fmax(ue[i + 1], ue[i + 2]));
            const double plo = std::fmin(pe[i], std::fmin(pe[i + 1], pe[i + 2]));
            const double phi = std::fmax(pe[i], std::fmax(pe[i + 1], pe[i + 2]));
            u_new[i] = std::fmin(std::fmax(u_new[i], ulo), uhi);
            p_new[i] = std::fmax(std::fmin(std::fmax(p_new[i], plo), phi), 1.0e-12);
        }
    }
    if (mode == PrimitiveFilter::Stencil) return;
    const std::vector<double> uc = extend(u_new, bc_l, bc_r, true);
    const std::vector<double> pc = extend(p_new, bc_l, bc_r, false);
    for (int i = 0; i < (int)u_new.size(); ++i) {
        if (mode != PrimitiveFilter::LedPressure)
            u_new[i] = std::fmin(std::fmax(uc[i + 1], std::fmin(uc[i], uc[i + 2])), std::fmax(uc[i], uc[i + 2]));
        if (mode != PrimitiveFilter::LedVelocity)
            p_new[i] = std::fmax(std::fmin(std::fmax(pc[i + 1], std::fmin(pc[i], pc[i + 2])), std::fmax(pc[i], pc[i + 2])), 1.0e-12);
    }
}

// Conservative pure-Euler shortcut from imex_ad.py::_single_phase_euler_rusanov_step.
inline StepResult single_phase_euler_step(const std::vector<double>& alpha,
                                          const std::vector<double>& T1,
                                          const std::vector<double>& T2,
                                          const std::vector<double>& u,
                                          const std::vector<double>& p,
                                          double dt, double dx, const EOS& eos,
                                          const StepConfig& cfg) {
    const int n=(int)alpha.size(), ne=n+2;
    const auto Te=cfd::detail::extend1(T1,cfg.bc_l,cfg.bc_r,false,cfg.T1_inlet_l);
    const auto ue=cfd::detail::extend1(u,cfg.bc_l,cfg.bc_r,true,cfg.u_inlet_l);
    const auto pe=cfd::detail::extend1(p,cfg.bc_l,cfg.bc_r,false,cfg.p_inlet_l,cfg.p_outlet_r);
    std::vector<double> rho(ne), c2(ne), dr(ne,0.), du(ne,0.), dp(ne,0.);
    for(int k=0;k<ne;++k) { rho[k]=std::fmax(eos.density(pe[k],Te[k]),MU_EPS); c2[k]=std::fmax(eos.sound_speed_sq(rho[k],eos.energy(rho[k],pe[k]),pe[k]),MU_EPS); }
    const auto limit = [&](double left, double right) {
        const double product = left * right;
        switch (cfg.pure_tvd_limiter) {
            case PureTvdLimiter::Minmod: return product <= 0. ? 0. : (std::fabs(left) < std::fabs(right) ? left : right);
            case PureTvdLimiter::Mc: {
                if (product <= 0.) return 0.;
                const double centered = .5 * (left + right);
                const double bound = 2. * (std::fabs(left) < std::fabs(right) ? left : right);
                return std::fabs(centered) < std::fabs(bound) ? centered : bound;
            }
            case PureTvdLimiter::Superbee: {
                if (product <= 0.) return 0.;
                const double a = std::copysign(std::fmin(2. * std::fabs(left), std::fabs(right)), left);
                const double b = std::copysign(std::fmin(std::fabs(left), 2. * std::fabs(right)), left);
                return std::fabs(a) > std::fabs(b) ? a : b;
            }
            case PureTvdLimiter::VanAlbada:
                return product <= 0. ? 0. : left * right * (left + right) /
                    std::fmax(left * left + right * right, MU_EPS);
            case PureTvdLimiter::Umist: {
                if (product <= 0.) return 0.;
                const double sign = std::copysign(1., left);
                const double al = std::fabs(left), ar = std::fabs(right);
                return sign * std::fmax(0., std::min({2. * al, 2. * ar, (al + 3. * ar) / 4., (3. * al + ar) / 4.}));
            }
            case PureTvdLimiter::VanLeer:
                return cfd::detail::vanleer_pair(left, right);
        }
        return 0.;
    };
    for(int k=1;k+1<ne;++k) {
        const double dlr=rho[k]-rho[k-1], dlv=ue[k]-ue[k-1], dlp=pe[k]-pe[k-1];
        const double drr=rho[k+1]-rho[k], drv=ue[k+1]-ue[k], drp=pe[k+1]-pe[k];
        if (cfg.pure_euler_characteristic_reconstruction) {
            const double rc=std::fmax(rho[k],MU_EPS), cc=std::sqrt(c2[k]), inv_c2=1./c2[k];
            const double am=limit(.5*(dlp*inv_c2-rc*dlv/cc), .5*(drp*inv_c2-rc*drv/cc));
            const double ap=limit(.5*(dlp*inv_c2+rc*dlv/cc), .5*(drp*inv_c2+rc*drv/cc));
            const double az=limit(dlr-dlp*inv_c2, drr-drp*inv_c2);
            dr[k]=az+am+ap; du[k]=cc/rc*(ap-am); dp[k]=c2[k]*(am+ap);
        } else {
            dr[k]=limit(dlr,drr); du[k]=limit(dlv,drv); dp[k]=limit(dlp,drp);
        }
    }
    std::vector<double> F0(n+1),F1(n+1),F2(n+1), R0(n+1),R1(n+1),R2(n+1);
    for(int f=0;f<=n;++f) {
        const int l=f,r=f+1;
        const double hcoef = cfg.pure_euler_hancock ? .5 : 0.;
        const double rlx=rho[l]+.5*dr[l]+hcoef*dt*(-ue[l]*dr[l]/dx-rho[l]*du[l]/dx);
        const double rrx=rho[r]-.5*dr[r]+hcoef*dt*(-ue[r]*dr[r]/dx-rho[r]*du[r]/dx);
        const double ulx=ue[l]+.5*du[l]+hcoef*dt*(-ue[l]*du[l]/dx-dp[l]/dx/rho[l]);
        const double urx=ue[r]-.5*du[r]+hcoef*dt*(-ue[r]*du[r]/dx-dp[r]/dx/rho[r]);
        const double plx=pe[l]+.5*dp[l]+hcoef*dt*(-ue[l]*dp[l]/dx-rho[l]*c2[l]*du[l]/dx);
        const double prx=pe[r]-.5*dp[r]+hcoef*dt*(-ue[r]*dp[r]/dx-rho[r]*c2[r]*du[r]/dx);
        double rL=rlx,uL=ulx,pL=plx,rR=rrx,uR=urx,pR=prx;
        if(!(std::isfinite(rL)&&std::isfinite(rR)&&std::isfinite(uL)&&std::isfinite(uR)&&std::isfinite(pL)&&std::isfinite(pR)&&rL>MU_EPS&&rR>MU_EPS&&pL>MU_EPS&&pR>MU_EPS)) {rL=rho[l];rR=rho[r];uL=ue[l];uR=ue[r];pL=pe[l];pR=pe[r];}
        if (!(cfg.bc_l==BC5::Periodic && cfg.bc_r==BC5::Periodic)) { if(f==0){rL=rho[0];uL=ue[0];pL=pe[0];} if(f==n){rR=rho[ne-1];uR=ue[ne-1];pR=pe[ne-1];} }
        const double eL=eos.energy(rL,pL),eR=eos.energy(rR,pR), EL=rL*eL+.5*rL*uL*uL, ER=rR*eR+.5*rR*uR*uR;
        const double cL=std::sqrt(std::fmax(eos.sound_speed_sq(rL,eL,pL),MU_EPS)),cR=std::sqrt(std::fmax(eos.sound_speed_sq(rR,eR,pR),MU_EPS));
        const double sL=std::fmin(uL-cL,uR-cR),sR=std::fmax(uL+cL,uR+cR);
        const double fl0=rL*uL,fl1=rL*uL*uL+pL,fl2=(EL+pL)*uL, fr0=rR*uR,fr1=rR*uR*uR+pR,fr2=(ER+pR)*uR;
        const double sr = std::fmax(std::fabs(uL) + cL, std::fabs(uR) + cR);
        R0[f] = .5 * (fl0 + fr0) - .5 * sr * (rR - rL);
        R1[f] = .5 * (fl1 + fr1) - .5 * sr * (rR * uR - rL * uL);
        R2[f] = .5 * (fl2 + fr2) - .5 * sr * (ER - EL);
        if (cfg.pure_euler_flux == PureEulerFlux::Hllc) {
            const double den=rL*(sL-uL)-rR*(sR-uR), ds=std::fabs(den)>MU_EPS?den:(den>=0.?MU_EPS:-MU_EPS);
            const double sm=(pR-pL+rL*uL*(sL-uL)-rR*uR*(sR-uR))/ds;
            const double pm=.5*(pL+rL*(sL-uL)*(sm-uL)+pR+rR*(sR-uR)*(sm-uR));
            if (!std::isfinite(sm)||!std::isfinite(pm)) { const double d=std::fmax(sR-sL,MU_EPS); F0[f]=(sR*fl0-sL*fr0+sL*sR*(rR-rL))/d;F1[f]=(sR*fl1-sL*fr1+sL*sR*(rR*uR-rL*uL))/d;F2[f]=(sR*fl2-sL*fr2+sL*sR*(ER-EL))/d; }
            else if(sL>=0.) {F0[f]=fl0;F1[f]=fl1;F2[f]=fl2;} else if(sm>=0.) {const double rs=rL*(sL-uL)/(sL-sm), es=((sL-uL)*EL-pL*uL+pm*sm)/(sL-sm);F0[f]=fl0+sL*(rs-rL);F1[f]=fl1+sL*(rs*sm-rL*uL);F2[f]=fl2+sL*(es-EL);} else if(sR>0.) {const double rs=rR*(sR-uR)/(sR-sm), es=((sR-uR)*ER-pR*uR+pm*sm)/(sR-sm);F0[f]=fr0+sR*(rs-rR);F1[f]=fr1+sR*(rs*sm-rR*uR);F2[f]=fr2+sR*(es-ER);} else {F0[f]=fr0;F1[f]=fr1;F2[f]=fr2;}
        } else if(sL>=0.) {F0[f]=fl0;F1[f]=fl1;F2[f]=fl2;} else if(sR<=0.) {F0[f]=fr0;F1[f]=fr1;F2[f]=fr2;} else { const double d=std::fmax(sR-sL,MU_EPS); F0[f]=(sR*fl0-sL*fr0+sL*sR*(rR-rL))/d;F1[f]=(sR*fl1-sL*fr1+sL*sR*(rR*uR-rL*uL))/d;F2[f]=(sR*fl2-sL*fr2+sL*sR*(ER-EL))/d; }
        if (cfg.pure_euler_rusanov_fallback &&
            (!std::isfinite(F0[f]) || !std::isfinite(F1[f]) || !std::isfinite(F2[f]))) {
            F0[f] = R0[f]; F1[f] = R1[f]; F2[f] = R2[f];
        }
    }
    if(cfg.bc_l==BC5::Reflective){F0[0]=F2[0]=0.;F1[0]=p[0];}
    if(cfg.bc_r==BC5::Reflective){F0[n]=F2[n]=0.;F1[n]=p[n-1];}
    StepResult out; out.alpha=alpha;out.T1.resize(n);out.T2.resize(n);out.u.resize(n);out.p.resize(n); std::vector<double> rho_new(n), mom_new(n), rho_old(n), energy_old(n);
    for(int i=0;i<n;++i) { rho_old[i]=alpha[i]*eos.density(p[i],T1[i])+(1.-alpha[i])*eos.density(p[i],T2[i]); mom_new[i]=rho_old[i]*u[i]-dt*(F1[i+1]-F1[i])/dx; rho_new[i]=std::fmax(rho_old[i]-dt*(F0[i+1]-F0[i])/dx,MU_EPS); energy_old[i]=alpha[i]*eos.density(p[i],T1[i])*eos.energy(eos.density(p[i],T1[i]),p[i])+(1.-alpha[i])*eos.density(p[i],T2[i])*eos.energy(eos.density(p[i],T2[i]),p[i])+.5*rho_old[i]*u[i]*u[i]; const double rE=energy_old[i]-dt*(F2[i+1]-F2[i])/dx; out.u[i]=mom_new[i]/rho_new[i]; const double ei=rE/rho_new[i]-.5*out.u[i]*out.u[i];out.p[i]=eos.pressure(rho_new[i],ei);out.T1[i]=out.T2[i]=eos.temperature(rho_new[i],ei); }
    bool invalid_update=false; for(int i=0;i<n;++i) invalid_update|=!std::isfinite(out.p[i])||!std::isfinite(out.T1[i])||out.p[i]<=0.||out.T1[i]<=0.;
    if (cfg.pure_euler_rusanov_fallback && invalid_update) {
        for(int i=0;i<n;++i) { rho_new[i]=std::fmax(rho_old[i]-dt*(R0[i+1]-R0[i])/dx,MU_EPS); mom_new[i]=rho_old[i]*u[i]-dt*(R1[i+1]-R1[i])/dx; const double rE=energy_old[i]-dt*(R2[i+1]-R2[i])/dx; out.u[i]=mom_new[i]/rho_new[i]; const double ei=rE/rho_new[i]-.5*out.u[i]*out.u[i]; out.p[i]=eos.pressure(rho_new[i],ei); out.T1[i]=out.T2[i]=eos.temperature(rho_new[i],ei); }
    }
    bool pjump=false; for(int i=0;i+1<n;++i) if(std::fabs(p[i+1]-p[i])/std::fmax(std::fmax(std::fabs(p[i]),std::fabs(p[i+1])),1.)>regime_detail::eps025()){pjump=true;break;}
    const bool led=(cfg.primitive_filter==PrimitiveFilter::Led || cfg.primitive_filter==PrimitiveFilter::LedVelocity || (cfg.primitive_filter==PrimitiveFilter::Auto && pjump));
    if(led){ const auto re=extend(rho_new,cfg.bc_l,cfg.bc_r,false); for(int i=0;i<n;++i) rho_new[i]=std::fmax(std::fmin(std::fmax(re[i+1],std::fmin(re[i],re[i+2])),std::fmax(re[i],re[i+2])),MU_EPS); for(int i=0;i<n;++i)out.u[i]=mom_new[i]/rho_new[i]; }
    apply_primitive_filter(u,p,out.u,out.p,cfg.bc_l,cfg.bc_r,cfg.primitive_filter);
    for(int i=0;i<n;++i) out.T1[i]=out.T2[i]=eos.temperature(rho_new[i],eos.energy(rho_new[i],out.p[i]));
    return out;
}

} // namespace step_detail

// One production imex_ad_step (single-stage).  Note: the pure-domain single-
// phase Euler-Rusanov shortcut (imex_ad.py 3929-3953) is intentionally NOT
// ported — it dispatches to a separate module and never triggers for the
// immiscible-interface acceptance cases (02A/07B), which always take the two-
// phase acoustic path selected by regime_auto.
inline StepResult imex_ad_step(
        const std::vector<double>& alpha_n, const std::vector<double>& T1_n,
        const std::vector<double>& T2_n, const std::vector<double>& u_n,
        const std::vector<double>& p_n, double dt, double dx,
        const EOS& eos1, const EOS& eos2, const StepConfig& cfg) {
    using namespace step_detail;
    const int n = (int)alpha_n.size();
    const double apt = cfg.alpha_pure_tol;

    double amin=alpha_n[0], amax=alpha_n[0], tgap=0.0;
    for (int i=0;i<n;++i) { amin=std::fmin(amin,alpha_n[i]); amax=std::fmax(amax,alpha_n[i]); tgap=std::fmax(tgap,std::fabs(T1_n[i]-T2_n[i])); }
    const bool same_eos = eos1.kind==eos2.kind && std::fabs(eos1.gamma-eos2.gamma)<=1.e-14 &&
        std::fabs(eos1.kv-eos2.kv)<=1.e-14 && std::fabs(eos1.pinf-eos2.pinf)<=1.e-14 && std::fabs(eos1.b-eos2.b)<=1.e-14 && std::fabs(eos1.eta-eos2.eta)<=1.e-14 && std::fabs(eos1.q-eos2.q)<=1.e-14;
    if (apt>0.0 && amax<=apt) return single_phase_euler_step(alpha_n,T1_n,T2_n,u_n,p_n,dt,dx,eos2,cfg);
    if (apt>0.0 && amin>=1.0-apt) return single_phase_euler_step(alpha_n,T1_n,T2_n,u_n,p_n,dt,dx,eos1,cfg);
    if (same_eos && amax-amin<=1.e-14 && tgap<=1.e-12) return single_phase_euler_step(alpha_n,T1_n,T2_n,u_n,p_n,dt,dx,eos1,cfg);

    // ── 1. regime_auto pressure-closure pick (M8) ────────────────────────────
    PressureClosure closure = cfg.pressure_closure ? *cfg.pressure_closure
        : select_regime(alpha_n, T1_n, T2_n, p_n, eos1, eos2, apt,
                        cfg.mixture_sound_speed_kind);

    // ── 2. material update (M6) ──────────────────────────────────────────────
    MaterialConfig material_cfg = cfg.material_config();
    if (closure == PressureClosure::apec_pe)
        material_cfg.energy_form = EnergyForm::Secant;
    MaterialResult mat = material_update(alpha_n, T1_n, T2_n, u_n, p_n, dt, dx,
                                         eos1, eos2, material_cfg);

    // ── 3. clip alpha ────────────────────────────────────────────────────────
    std::vector<double> alpha_new = mat.alpha_new;
    for (double& a : alpha_new) a = std::fmin(std::fmax(a, 1.0e-12), 1.0 - 1.0e-12);

    // ── 4. acoustic (u,p) solve (M7) ─────────────────────────────────────────
    AcousticSolveResult ac = acoustic_solve(
        n, dx, dt, eos1, eos2,
        alpha_n.data(), T1_n.data(), T2_n.data(), u_n.data(), p_n.data(),
        mat.q1_new.data(), mat.q2_new.data(), mat.m_adv.data(),
        to_acoustic_bc(cfg.bc_l, true), to_acoustic_bc(cfg.bc_r, false), apt,
        0.5, 1e-8, cfg.u_inlet_l, cfg.p_inlet_l, cfg.p_outlet_r,
        cfg.acoustic_interface_be, cfg.acoustic_pure_tol_consistent, cfg.acoustic_acid,
        nullptr, false, cfg.acoustic_trbdf2, cfg.acoustic_muscl, cfg.acoustic_stencil_clean,
        cfg.acoustic_waf, static_cast<int>(cfg.acoustic_waf_sigma), static_cast<int>(cfg.acoustic_reconstruction), cfg.acoustic_diss_consistent, cfg.acoustic_interface_centered,
        cfg.mixture_sound_speed_kind);
    std::vector<double> u_new = ac.u_new;
    std::vector<double> p_new = ac.p_new;
    std::vector<double> rhoE_new = mat.rhoE_new;

    // ── 5. energy closure ────────────────────────────────────────────────────
    if (closure == PressureClosure::implicit_energy ||
        closure == PressureClosure::implicit_energy_momentum ||
        closure == PressureClosure::apec_pe) {
        std::vector<double> Z(n);
        for (int i = 0; i < n; ++i)
            Z[i] = phase_acoustic(eos1, eos2, alpha_n[i], T1_n[i], T2_n[i], p_n[i], apt,
                                  cfg.mixture_sound_speed_kind).Z;
        const std::vector<char> mask = compressive_pressure_mask(u_n, p_n);
        for (int i = 0; i < n; ++i) if (mask[i]) {
            const double a = std::fmin(std::fmax(alpha_new[i], 1.0e-12), 1.0 - 1.0e-12);
            const double q1 = std::fmax(mat.q1_new[i], MU_EPS), q2 = std::fmax(mat.q2_new[i], MU_EPS);
            const double rho = std::fmax(q1 + q2, MU_EPS), r1 = std::fmax(q1 / a, MU_EPS), r2 = std::fmax(q2 / (1.0 - a), MU_EPS);
            double pi = std::fmax(p_new[i], 1.0e-12);
            auto residual = [&](double trial) {
                std::vector<double> pt = p_new; pt[i] = std::fmax(trial, 1.0e-12);
                const Faces f = acoustic_faces(u_new, pt, Z, cfg.bc_l, cfg.bc_r,
                                               cfg.u_inlet_l, cfg.p_inlet_l, cfg.p_outlet_r);
                const double e = q1 * eos1.energy(r1, pt[i]) + q2 * eos2.energy(r2, pt[i])
                    + .5 * rho * u_new[i] * u_new[i];
                return e - mat.rhoE_adv[i] + dt * (f.pf[i + 1] * f.uf[i + 1] - f.pf[i] * f.uf[i]) / dx;
            };
            for (int it = 0; it < 8; ++it) {
                const double r0 = residual(pi), h = 1.0e-7 * std::fmax(std::fabs(pi), 1.0);
                const double dr = (residual(pi + h) - r0) / h;
                if (!std::isfinite(dr) || std::fabs(dr) < MU_EPS) break;
                const double next = std::fmax(pi - r0 / dr, 1.0e-12);
                if (std::fabs(next - pi) / std::fmax(std::fabs(next), 1.0) < 1.0e-9) { pi = next; break; }
                pi = next;
            }
            p_new[i] = pi;
        }
        if (closure == PressureClosure::implicit_energy_momentum) {
            const Faces f = acoustic_faces(u_new, p_new, Z, cfg.bc_l, cfg.bc_r,
                                           cfg.u_inlet_l, cfg.p_inlet_l, cfg.p_outlet_r);
            for (int i = 0; i < n; ++i)
                u_new[i] = (mat.m_adv[i] - dt * (f.pf[i + 1] - f.pf[i]) / dx)
                    / std::fmax(mat.q1_new[i] + mat.q2_new[i], MU_EPS);
        }
        const Faces f = acoustic_faces(u_new, p_new, Z, cfg.bc_l, cfg.bc_r,
                                       cfg.u_inlet_l, cfg.p_inlet_l, cfg.p_outlet_r);
        for (int i = 0; i < n; ++i)
            rhoE_new[i] = mat.rhoE_adv[i] - dt * (f.pf[i + 1] * f.uf[i + 1] - f.pf[i] * f.uf[i]) / dx;
    } else if (closure == PressureClosure::compressive_recovery) {
        std::vector<char> rmask = compressive_pressure_mask(u_n, p_n);
        if (apt > 0.0) {
            double pure_tol = std::fmax(apt, regime_detail::eps025());
            std::vector<char> pm = pure_material_cell_mask(alpha_n, pure_tol);
            for (int i = 0; i < n; ++i) rmask[i] = rmask[i] && !pm[i];
        }
        bool any = false; for (char m : rmask) any |= m;
        if (any) {
            std::vector<double> prec = recover_pressure_from_total_energy(
                mat.q1_new, mat.q2_new, rhoE_new, alpha_new, u_new, p_new,
                eos1, eos2);
            for (int i = 0; i < n; ++i) if (rmask[i]) p_new[i] = prec[i];
        }
    } else if (closure == PressureClosure::pressure_work_consistent) {
        // Recompute Z from the W^n anchor, build acoustic faces, rebuild energy.
        std::vector<double> Z(n);
        for (int i = 0; i < n; ++i)
            Z[i] = phase_acoustic(eos1, eos2, alpha_n[i], T1_n[i], T2_n[i],
                                  p_n[i], apt, cfg.mixture_sound_speed_kind).Z;
        Faces fc = acoustic_faces(u_new, p_new, Z, cfg.bc_l, cfg.bc_r,
                                  cfg.u_inlet_l, cfg.p_inlet_l, cfg.p_outlet_r);
        for (int i = 0; i < n; ++i) {
            rhoE_new[i] = mat.rhoE_adv[i]
                - dt * (fc.pf[i + 1] * fc.uf[i + 1] - fc.pf[i] * fc.uf[i]) / dx;
        }
        // PW pure-shock recovery: compressive & ~pure (FIVE_EQ_IMEX_PW_PURE_
        // SHOCK_RECOVERY=1 default).  Inactive for 02A/07B acoustic.
        std::vector<char> rmask = compressive_pressure_mask(u_n, p_n);
        double pure_tol = std::fmax(apt, regime_detail::eps025());
        std::vector<char> pm = pure_material_cell_mask(alpha_n, pure_tol);
        for (int i = 0; i < n; ++i) rmask[i] = rmask[i] && !pm[i];
        bool any = false; for (char m : rmask) any |= m;
        if (any) {
            std::vector<double> prec = recover_pressure_from_total_energy(
                mat.q1_new, mat.q2_new, rhoE_new, alpha_new, u_new, p_new,
                eos1, eos2);
            for (int i = 0; i < n; ++i) if (rmask[i]) p_new[i] = prec[i];
        }
    } else if (closure == PressureClosure::dual_entropy) {
        const std::vector<char> mask = compressive_pressure_mask(u_n, p_n);
        for (int i = 0; i < n; ++i) if (mask[i]) {
            const double a = std::fmin(std::fmax(alpha_new[i], 1.0e-12), 1.0 - 1.0e-12);
            const double r10 = std::fmax(eos1.density(p_n[i], T1_n[i]), MU_EPS);
            const double r20 = std::fmax(eos2.density(p_n[i], T2_n[i]), MU_EPS);
            const double r1 = std::fmax(mat.q1_new[i] / a, MU_EPS);
            const double r2 = std::fmax(mat.q2_new[i] / (1.0 - a), MU_EPS);
            const double p1 = (p_n[i] + eos1.pinf) * std::pow(r1 / r10, eos1.gamma) - eos1.pinf;
            const double p2 = (p_n[i] + eos2.pinf) * std::pow(r2 / r20, eos2.gamma) - eos2.pinf;
            const double pe = a * p1 + (1.0 - a) * p2;
            if (std::isfinite(pe)) p_new[i] = std::fmax(pe, 1.0e-12);
        }
    }
    // implicit_energy: not on the 02A/07B path — would use _solve_acoustic_energy_ad.

    // ── 6. near-vacuum velocity regularisation ───────────────────────────────
    int vac = 0;
    u_new = regularize_near_vacuum_velocity(
        alpha_n, T1_n, T2_n, p_n, mat.q1_new, mat.q2_new, u_new, p_new,
        eos1, eos2, apt, cfg.bc_l, cfg.bc_r, vac, 6, cfg.mixture_sound_speed_kind);

    // ── 7. primitive LMP/LED filter ──────────────────────────────────────────
    apply_primitive_filter(u_n, p_n, u_new, p_new, cfg.bc_l, cfg.bc_r,
                           cfg.primitive_filter);

    // ── 8. explicit primitive recovery (M9 simple path) ──────────────────────
    StepResult R;
    R.alpha = alpha_new;
    R.T1.resize(n); R.T2.resize(n);
    R.u = u_new; R.p = p_new;
    R.closure = closure;
    R.vacuum_velocity_cells = vac;
    for (int i = 0; i < n; ++i) {
        double rho1 = mat.q1_new[i] / std::fmax(alpha_new[i], 1.0e-12);
        double rho2 = mat.q2_new[i] / std::fmax(1.0 - alpha_new[i], 1.0e-12);
        double e1 = eos1.energy(rho1, p_new[i]);
        double e2 = eos2.energy(rho2, p_new[i]);
        R.T1[i] = eos1.temperature(rho1, e1);
        R.T2[i] = eos2.temperature(rho2, e2);
    }
    return R;
}

} // namespace five_eq
} // namespace cfd
