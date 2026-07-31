// EOS-consistent 1-D face state used by the ARS explicit residual.
#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "cfd/five_eq/energy_flux.hpp"
#include "cfd/five_eq/face_options.hpp"
#include "cfd/five_eq/positivity.hpp"
#include "cfd/five_eq/step.hpp"

namespace cfd::five_eq {

struct FaceState {
    FaceEnergy energy;
    std::vector<double> c1_sq, c2_sq;
    std::vector<double> u_L, u_R, p_L, p_R;
    std::vector<double> rho_L, rho_R;
};

inline FaceState acid_face_state(const StepResult& W, const EOS& e1, const EOS& e2,
                                 BC5 l, BC5 r,
                                 std::optional<double> u_inlet_l = {},
                                 std::optional<double> p_inlet_l = {},
                                 std::optional<double> alpha_inlet_l = {},
                                 std::optional<double> T1_inlet_l = {},
                                 std::optional<double> T2_inlet_l = {},
                                 const FaceStateOptions& options = {}) {
    const int n = static_cast<int>(W.alpha.size());
    const auto ae = ::cfd::detail::extend1(W.alpha, l, r, false, alpha_inlet_l);
    const auto t1e = ::cfd::detail::extend1(W.T1, l, r, false, T1_inlet_l);
    const auto t2e = ::cfd::detail::extend1(W.T2, l, r, false, T2_inlet_l);
    const auto ue = ::cfd::detail::extend1(W.u, l, r, true, u_inlet_l);
    const auto pe = ::cfd::detail::extend1(W.p, l, r, false, p_inlet_l);
    FaceState out;
    auto& f = out.energy;
    f.alpha.resize(n+1); f.p.resize(n+1); f.u.resize(n+1);
    f.a_L.resize(n+1); f.a_R.resize(n+1);
    f.rho1.resize(n+1); f.rho2.resize(n+1);
    f.rho1_L.resize(n+1); f.rho1_R.resize(n+1);
    f.rho2_L.resize(n+1); f.rho2_R.resize(n+1);
    f.T1.resize(n+1); f.T2.resize(n+1); f.e1.resize(n+1); f.e2.resize(n+1);
    out.c1_sq.resize(n+1); out.c2_sq.resize(n+1);
    out.u_L.resize(n+1); out.u_R.resize(n+1); out.p_L.resize(n+1); out.p_R.resize(n+1);
    out.rho_L.resize(n+1); out.rho_R.resize(n+1);
    for (int k=0; k<=n; ++k) {
        const double al=std::clamp(ae[k], 1.e-12, 1.-1.e-12);
        const double ar=std::clamp(ae[k+1], 1.e-12, 1.-1.e-12);
        const double t1l=std::fmax(t1e[k],1.0), t1r=std::fmax(t1e[k+1],1.0);
        const double t2l=std::fmax(t2e[k],1.0), t2r=std::fmax(t2e[k+1],1.0);
        const double pl=std::fmax(pe[k],1.0), pr=std::fmax(pe[k+1],1.0);
        const double ul=ue[k], ur=ue[k+1];
        const double r1l=std::fmax(e1.density(pl,t1l),1.e-30), r1r=std::fmax(e1.density(pr,t1r),1.e-30);
        const double r2l=std::fmax(e2.density(pl,t2l),1.e-30), r2r=std::fmax(e2.density(pr,t2r),1.e-30);
        const bool pre_up=.5*(ul+ur)>=0.0;
        const double uf=options.up_scheme==UPFaceScheme::Central ? .5*(ul+ur) : (pre_up?ul:ur);
        const double pf=std::fmax(options.up_scheme==UPFaceScheme::Central ? .5*(pl+pr) : (pre_up?pl:pr),1.0);
        const bool up=uf>=0.0;
        f.a_L[k]=al; f.a_R[k]=ar; f.p[k]=pf; f.u[k]=uf;
        f.alpha[k]=up?al:ar; f.T1[k]=up?t1l:t1r; f.T2[k]=up?t2l:t2r;
        f.rho1_L[k]=r1l; f.rho1_R[k]=r1r; f.rho2_L[k]=r2l; f.rho2_R[k]=r2r;
        out.u_L[k]=ul; out.u_R[k]=ur; out.p_L[k]=pl; out.p_R[k]=pr;
        out.rho_L[k]=al*r1l+(1.-al)*r2l; out.rho_R[k]=ar*r1r+(1.-ar)*r2r;
    }

    const auto alpha_slope = [&](int j) {
        if (j <= 0 || j+1 >= static_cast<int>(ae.size())) return 0.0;
        const double dl=ae[j]-ae[j-1], dr=ae[j+1]-ae[j];
        return dl*dr>0.0 ? std::copysign(std::fmin(std::fabs(dl),std::fabs(dr)),dl) : 0.0;
    };
    if (options.alpha_scheme==AlphaFaceScheme::Muscl) {
        for (int k=0;k<=n;++k) {
            const double al=f.a_L[k], ar=f.a_R[k], lo=std::fmin(al,ar), hi=std::fmax(al,ar);
            const double left=std::clamp(al+.5*alpha_slope(k),lo,hi);
            const double right=std::clamp(ar-.5*alpha_slope(k+1),lo,hi);
            f.alpha[k]=(f.u[k]>=0.0)?left:right;
        }
    } else if (options.alpha_scheme==AlphaFaceScheme::Central) {
        for (int k=0;k<=n;++k) f.alpha[k]=.5*(f.a_L[k]+f.a_R[k]);
    } else if (options.alpha_scheme==AlphaFaceScheme::Cicsam || options.alpha_scheme==AlphaFaceScheme::AdaptiveBvd ||
               options.alpha_scheme==AlphaFaceScheme::Mstacs || options.alpha_scheme==AlphaFaceScheme::ThincBvd) {
        if (!options.has_dt_dx || options.dx<=0.0)
            throw std::invalid_argument("high-order alpha face scheme requires dt and dx");
        std::vector<double> alpha(n+1);
        if (options.alpha_scheme==AlphaFaceScheme::Cicsam)
            cicsam_alpha_face(ae.data(),static_cast<int>(ae.size()),f.u.data(),n+1,options.dt,options.dx,alpha.data());
        else if (options.alpha_scheme==AlphaFaceScheme::AdaptiveBvd)
            adaptive_bvd_alpha_face(ae.data(),static_cast<int>(ae.size()),f.u.data(),n+1,options.dt,options.dx,
                                    options.alpha_limiter,options.alpha_pure_tol,alpha.data());
        else if (options.alpha_scheme==AlphaFaceScheme::Mstacs)
            mstacs_alpha_face(ae.data(),static_cast<int>(ae.size()),f.u.data(),n+1,options.dt,options.dx,alpha.data());
        else
            thinc_bvd_alpha_face(ae.data(),static_cast<int>(ae.size()),f.u.data(),n+1,options.dt,options.dx,options.alpha_limiter,alpha.data());
        f.alpha=std::move(alpha);
    } else if (options.alpha_scheme==AlphaFaceScheme::Stacs || options.alpha_scheme==AlphaFaceScheme::VanLeer || options.alpha_scheme==AlphaFaceScheme::Thinc) {
        std::vector<double> alpha(n+1);
        if(options.alpha_scheme==AlphaFaceScheme::Stacs) stacs_alpha_face(ae.data(),static_cast<int>(ae.size()),f.u.data(),n+1,alpha.data());
        else if(options.alpha_scheme==AlphaFaceScheme::VanLeer) vanleer_alpha_face(ae.data(),static_cast<int>(ae.size()),f.u.data(),n+1,alpha.data());
        else thinc_alpha_face(ae.data(),static_cast<int>(ae.size()),f.u.data(),n+1,alpha.data());
        f.alpha=std::move(alpha);
    }
    for (double& a:f.alpha) a=std::clamp(a,1.e-12,1.-1.e-12);

    TvdLimiter primitive_limiter=options.primitive_limiter;
    bool high_primitive=false, weno_primitive=false;
    switch (options.primitive_scheme) {
        case PrimitiveFaceScheme::Tmlpu: high_primitive=true; break;
        case PrimitiveFaceScheme::Weno3: weno_primitive=true; break;
        case PrimitiveFaceScheme::Superbee: high_primitive=true; primitive_limiter=TvdLimiter::Superbee; break;
        case PrimitiveFaceScheme::VanLeer: high_primitive=true; primitive_limiter=TvdLimiter::VanLeer; break;
        case PrimitiveFaceScheme::Minmod: high_primitive=true; primitive_limiter=TvdLimiter::Minmod; break;
        case PrimitiveFaceScheme::MC: high_primitive=true; primitive_limiter=TvdLimiter::MC; break;
        case PrimitiveFaceScheme::VanAlbada: high_primitive=true; primitive_limiter=TvdLimiter::VanAlbada; break;
        case PrimitiveFaceScheme::Umist: high_primitive=true; primitive_limiter=TvdLimiter::Umist; break;
        default: break;
    }
    if (options.primitive_scheme==PrimitiveFaceScheme::Central) {
        for (int k=0;k<=n;++k) { f.T1[k]=.5*(t1e[k]+t1e[k+1]); f.T2[k]=.5*(t2e[k]+t2e[k+1]); }
    } else if (weno_primitive) {
        reconstruct_weno3_upwind_faces(t1e.data(),static_cast<int>(t1e.size()),f.u.data(),1.0,f.T1.data());
        reconstruct_weno3_upwind_faces(t2e.data(),static_cast<int>(t2e.size()),f.u.data(),1.0,f.T2.data());
    } else if (high_primitive) {
        reconstruct_upwind_faces(t1e.data(),static_cast<int>(t1e.size()),f.u.data(),primitive_limiter,
                                 0.0,1.0,false,1.0,f.T1.data());
        reconstruct_upwind_faces(t2e.data(),static_cast<int>(t2e.size()),f.u.data(),primitive_limiter,
                                 0.0,1.0,false,1.0,f.T2.data());
    }

    for (int k=0;k<=n;++k) {
        const bool up=f.u[k]>=0.0;
        if (options.thermo_scheme==FaceThermoScheme::Acid) {
            f.rho1[k]=std::fmax(e1.density(f.p[k],f.T1[k]),1.e-30);
            f.rho2[k]=std::fmax(e2.density(f.p[k],f.T2[k]),1.e-30);
            f.e1[k]=e1.energy(f.rho1[k],f.p[k]); f.e2[k]=e2.energy(f.rho2[k],f.p[k]);
            out.c1_sq[k]=phase_sound_speed_sq(e1,f.rho1[k],f.T1[k]);
            out.c2_sq[k]=phase_sound_speed_sq(e2,f.rho2[k],f.T2[k]);
        } else {
            f.rho1[k]=up?f.rho1_L[k]:f.rho1_R[k]; f.rho2[k]=up?f.rho2_L[k]:f.rho2_R[k];
            const double p_cell=up?out.p_L[k]:out.p_R[k];
            f.e1[k]=e1.energy(f.rho1[k],p_cell); f.e2[k]=e2.energy(f.rho2[k],p_cell);
            out.c1_sq[k]=phase_sound_speed_sq(e1,f.rho1[k],up?t1e[k]:t1e[k+1]);
            out.c2_sq[k]=phase_sound_speed_sq(e2,f.rho2[k],up?t2e[k]:t2e[k+1]);
        }
    }
    return out;
}

struct ExplicitResidual { std::vector<double> m1,m2,mom,rhoE,alpha; FaceState face; };

// Python limiters.py low-order fluxes, using the L/R ACID cache retained in
// FaceState.  Pressure remains in the implicit operator.
inline AdvectiveFlux5 rusanov_advective_fluxes(const FaceState& s,
                                                const EOS& e1, const EOS& e2) {
    const int nf=static_cast<int>(s.energy.alpha.size());
    AdvectiveFlux5 out; out.m1.resize(nf);out.m2.resize(nf);out.mom.resize(nf);out.rhoE.resize(nf);out.alpha.resize(nf);
    for(int f=0;f<nf;++f) {
        const auto& q=s.energy; const double al=q.a_L[f], ar=q.a_R[f];
        const double ul=s.u_L[f], ur=s.u_R[f], pl=s.p_L[f], pr=s.p_R[f];
        const double r1l=q.rho1_L[f],r2l=q.rho2_L[f],r1r=q.rho1_R[f],r2r=q.rho2_R[f];
        const double e1l=e1.energy(r1l,pl),e2l=e2.energy(r2l,pl);
        const double e1r=e1.energy(r1r,pr),e2r=e2.energy(r2r,pr);
        const double rl=al*r1l+(1.-al)*r2l, rr=ar*r1r+(1.-ar)*r2r;
        const double a=std::fmax(std::fabs(ul),std::fabs(ur))+.001;
        const double m1l=al*r1l, m1r=ar*r1r, m2l=(1.-al)*r2l,m2r=(1.-ar)*r2r;
        const double moml=rl*ul,momr=rr*ur;
        const double El=al*r1l*e1l+(1.-al)*r2l*e2l+.5*rl*ul*ul;
        const double Er=ar*r1r*e1r+(1.-ar)*r2r*e2r+.5*rr*ur*ur;
        out.m1[f]=.5*(m1l*ul+m1r*ur)-.5*a*(m1r-m1l);
        out.m2[f]=.5*(m2l*ul+m2r*ur)-.5*a*(m2r-m2l);
        out.mom[f]=.5*(rl*ul*ul+rr*ur*ur)-.5*a*(momr-moml);
        out.rhoE[f]=.5*(El*ul+Er*ur)-.5*a*(Er-El);
        out.alpha[f]=.5*(al*ul+ar*ur)-.5*a*(ar-al);
    }
    return out;
}

inline AdvectiveFlux5 pe_preserving_low_flux(const FaceState& s) {
    const int nf=static_cast<int>(s.energy.alpha.size());
    AdvectiveFlux5 out; out.m1.resize(nf);out.m2.resize(nf);out.mom.resize(nf);out.rhoE.resize(nf);out.alpha.resize(nf);
    const auto& q=s.energy;
    for(int f=0;f<nf;++f) { const double a=q.alpha[f],u=q.u[f],r1=q.rho1[f],r2=q.rho2[f];
        const double rho=a*r1+(1.-a)*r2; out.m1[f]=a*r1*u;out.m2[f]=(1.-a)*r2*u;out.alpha[f]=a*u;
        out.mom[f]=rho*u*u;out.rhoE[f]=(a*r1*q.e1[f]+(1.-a)*r2*q.e2[f]) *u+.5*u*u*(out.m1[f]+out.m2[f]); }
    return out;
}

// Port of residual.py::explicit_residual default ACID/APEC face branch.  The
// high-order reconstruction and theta limiter are deliberately separate from
// this leaf; ARS uses this first-order PE-consistent anchor by default.
inline ExplicitResidual explicit_residual(const StepResult& W, const EOS& e1, const EOS& e2,
                                          double dx, BC5 l, BC5 r,
                                          bool kapila_closure=false,
                                          bool kapila_source_in_implicit=false,
                                          EnergyForm energy_form=EnergyForm::Differential,
                                          std::optional<double> u_inlet_l = {},
                                          std::optional<double> p_inlet_l = {},
                                          std::optional<double> alpha_inlet_l = {},
                                          std::optional<double> T1_inlet_l = {},
                                          std::optional<double> T2_inlet_l = {},
                                          bool positivity=false, bool force_low=false,
                                          bool rusanov_low=false,
                                          double positivity_dt=0.0,
                                          const FaceStateOptions& face_options = {}) {
    ExplicitResidual out; out.face=acid_face_state(W,e1,e2,l,r,u_inlet_l,p_inlet_l,
                                                    alpha_inlet_l,T1_inlet_l,T2_inlet_l,face_options);
    const int n=static_cast<int>(W.alpha.size()); const auto& s=out.face; const auto& f=s.energy;
    std::vector<double> fq1(n+1),fq2(n+1),fa(n+1),frho(n+1),fmom(n+1),fE;
    for(int k=0;k<=n;++k) {
        fq1[k]=f.alpha[k]*f.rho1[k]*f.u[k]; fq2[k]=(1.-f.alpha[k])*f.rho2[k]*f.u[k];
        fa[k]=f.alpha[k]*f.u[k]; frho[k]=fq1[k]+fq2[k];
        const double rho=f.alpha[k]*f.rho1[k]+(1.-f.alpha[k])*f.rho2[k];
        fmom[k]=rho*f.u[k]*f.u[k];
    }
    total_energy_flux(f,e1,e2,fq1,fq2,fa,frho,energy_form,
                      face_options.energy_alpha_pure_tol,fE);
    // Python only enters the blended-flux path when an explicit-stage dt is
    // supplied.  A bare positivity/force_lo flag retains the high-order flux.
    if ((positivity || force_low) && positivity_dt > 0.0) {
        AdvectiveFlux5 high{fq1,fq2,fmom,fE,fa};
        const AdvectiveFlux5 low=rusanov_low
            ? rusanov_advective_fluxes(out.face,e1,e2)
            : pe_preserving_low_flux(out.face);
        std::vector<ConsU> U(n);
        for (int i=0;i<n;++i)
            U[i]=prim_to_cons_W({W.alpha[i],W.T1[i],W.T2[i],W.u[i],W.p[i]},e1,e2);
        double alpha_margin=.5;
        for (double a:W.alpha) alpha_margin=std::fmin(alpha_margin,std::fmin(a,1.-a));
        const double alpha_floor=std::fmin(1.e-6,std::fmax(1.e-12,.1*alpha_margin));
        std::vector<double> theta=force_low ? std::vector<double>(n+1,0.)
            : positivity_blend_theta(high,low,U,dx,positivity_dt,1.e-10,alpha_floor);
        const auto blended=blend_advective_fluxes(high,low,theta);
        fq1=blended.m1; fq2=blended.m2; fmom=blended.mom; fE=blended.rhoE; fa=blended.alpha;
        for(int k=0;k<=n;++k) frho[k]=fq1[k]+fq2[k];
    }
    out.m1.resize(n); out.m2.resize(n); out.mom.resize(n); out.rhoE.resize(n); out.alpha.resize(n);
    for(int i=0;i<n;++i) {
        const double inv=1./dx;
        const double divu=(f.u[i+1]-f.u[i])*inv;
        double B=W.alpha[i];
        if(kapila_closure && !kapila_source_in_implicit) {
            const auto dk = [&](int k) {
                const double a=f.alpha[k], b=1.-a;
                const double x=std::fmax(f.rho1[k]*s.c1_sq[k],1.e-30);
                const double y=std::fmax(f.rho2[k]*s.c2_sq[k],1.e-30);
                return a*b>1.e-12 ? a*b*(y-x)/std::fmax(b*x+a*y,1.e-30) : 0.;
            };
            B=.5*((f.alpha[i]+dk(i))+(f.alpha[i+1]+dk(i+1)));
        }
        out.m1[i]=(fq1[i+1]-fq1[i])*inv; out.m2[i]=(fq2[i+1]-fq2[i])*inv;
        out.mom[i]=(fmom[i+1]-fmom[i])*inv; out.rhoE[i]=(fE[i+1]-fE[i])*inv;
        out.alpha[i]=(fa[i+1]-fa[i])*inv-B*divu;
    }
    return out;
}
} // namespace cfd::five_eq
