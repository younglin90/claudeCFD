// cfd/five_eq/implicit_faces.hpp -- ARS implicit acoustic face/divergence leaf.
// Port of residual.py::implicit_face_pu / implicit_divergences (central,
// interface-Riemann, biharmonic and upwind dissipation paths).
#pragma once

#include <cmath>
#include <vector>

#include "cfd/five_eq/step.hpp"

namespace cfd::five_eq {
struct ImplicitFaces { std::vector<double> p, u; };
struct ImplicitDivergences { std::vector<double> grad_p, div_pu, div_u; ImplicitFaces face; };

inline std::vector<double> extend_two(const std::vector<double>& q, BC5 l, BC5 r, bool odd) {
    const int n=(int)q.size(); std::vector<double> x(n+4);
    for(int i=0;i<n;++i) x[i+2]=q[i];
    const double sl=(l==BC5::Reflective&&odd)?-1.0:1.0;
    const double sr=(r==BC5::Reflective&&odd)?-1.0:1.0;
    if(l==BC5::Periodic) { x[0]=q[n>1?n-2:0]; x[1]=q[n-1]; }
    else if(l==BC5::Reflective) { x[0]=sl*q[n>1?1:0]; x[1]=sl*q[0]; }
    else { x[0]=x[1]=q[0]; }
    if(r==BC5::Periodic) { x[n+2]=q[0]; x[n+3]=q[n>1?1:0]; }
    else if(r==BC5::Reflective) { x[n+2]=sr*q[n-1]; x[n+3]=sr*q[n>1?n-2:n-1]; }
    else { x[n+2]=x[n+3]=q[n-1]; }
    return x;
}

inline ImplicitFaces implicit_face_pu(const StepResult& W, const EOS& e1, const EOS& e2,
                                      BC5 l, BC5 r, double dissipation=0.0,
                                      bool biharmonic=true, bool rhie_chow=false,
                                      double gamma_dt=0.0, double dx=1.0,
                                      double alpha_jump_tol=1.e-8,
                                      bool acoustic_riemann=false,
                                      bool upwind_dissipation=false) {
    const int n=(int)W.p.size(); const bool bih=biharmonic&&dissipation>0.0&&
        !acoustic_riemann&&!upwind_dissipation;
    const bool use_rc=rhie_chow && !bih && l==BC5::Periodic && r==BC5::Periodic && gamma_dt>0.0;
    const auto pe=bih?extend_two(W.p,l,r,false):step_detail::extend(W.p,l,r,false);
    const auto ue=bih?extend_two(W.u,l,r,true):step_detail::extend(W.u,l,r,true);
    const auto ae=bih?extend_two(W.alpha,l,r,false):step_detail::extend(W.alpha,l,r,false);
    const auto t1e=bih?extend_two(W.T1,l,r,false):step_detail::extend(W.T1,l,r,false);
    const auto t2e=bih?extend_two(W.T2,l,r,false):step_detail::extend(W.T2,l,r,false);
    ImplicitFaces out; out.p.resize(n+1); out.u.resize(n+1);
    if (use_rc) {
        std::vector<double> rho(n);
        for (int i=0; i<n; ++i) {
            const double rho1=e1.density(W.p[i],W.T1[i]);
            const double rho2=e2.density(W.p[i],W.T2[i]);
            rho[i]=W.alpha[i]*rho1+(1.0-W.alpha[i])*rho2;
        }
        const auto p2=extend_two(W.p,l,r,false);
        const auto u2=extend_two(W.u,l,r,true);
        const auto rho2=extend_two(rho,l,r,false);
        for (int f=0; f<=n; ++f) {
            const double pim1=p2[f], pi=p2[f+1], pip1=p2[f+2], pip2=p2[f+3];
            const double ui=u2[f+1], uip1=u2[f+2];
            const double grad_face=(pip1-pi)/dx;
            const double grad_i=.5*(pip1-pim1)/dx;
            const double grad_ip1=.5*(pip2-pi)/dx;
            const double rho_face=std::fmax(.5*(rho2[f+1]+rho2[f+2]),1.e-30);
            out.p[f]=.5*(pi+pip1);
            out.u[f]=.5*(ui+uip1)-gamma_dt/rho_face*(grad_face-.5*(grad_i+grad_ip1));
        }
        return out;
    }
    if (acoustic_riemann) {
        for (int f=0; f<=n; ++f) {
            const int j=bih?f+1:f;
            const double Zl=phase_acoustic(e1,e2,ae[j],t1e[j],t2e[j],pe[j],0.0).Z;
            const double Zr=phase_acoustic(e1,e2,ae[j+1],t1e[j+1],t2e[j+1],pe[j+1],0.0).Z;
            const double den=std::fmax(Zl+Zr,1.e-30);
            out.p[f]=(Zr*pe[j]+Zl*pe[j+1]+Zl*Zr*(ue[j]-ue[j+1]))/den;
            out.u[f]=(pe[j]-pe[j+1]+Zl*ue[j]+Zr*ue[j+1])/den;
        }
        if (dissipation>0.0 && n>1) {
            const double w=std::fmin(std::fmax(dissipation,0.0),0.49);
            const auto p0=out.p, u0=out.u;
            for (int f=1; f<n; ++f) {
                const double pc=p0[f-1]-2.0*p0[f]+p0[f+1];
                const double uc=u0[f-1]-2.0*u0[f]+u0[f+1];
                const double pa=std::fabs(p0[f-1])+2.0*std::fabs(p0[f])+std::fabs(p0[f+1])+1.e-30;
                const double ua=std::fabs(u0[f-1])+2.0*std::fabs(u0[f])+std::fabs(u0[f+1])+1.e-30;
                const double ps=std::fmin(std::fmax((std::fabs(pc)/pa-.15)/.35,0.0),1.0);
                const double us=std::fmin(std::fmax((std::fabs(uc)/ua-.15)/.35,0.0),1.0);
                out.p[f]=p0[f]+w*ps*pc;
                out.u[f]=u0[f]+w*us*uc;
            }
        }
        return out;
    }
    for(int f=0;f<=n;++f) {
        const int j=bih?f+1:f;
        double pf=.5*(pe[j]+pe[j+1]), uf=.5*(ue[j]+ue[j+1]);
        if(bih) { pf-=dissipation*(-pe[j-1]+3*pe[j]-3*pe[j+1]+pe[j+2])/8.; uf-=dissipation*(-ue[j-1]+3*ue[j]-3*ue[j+1]+ue[j+2])/8.; }
        // Python only applies this Riemann override at material jumps.
        const double al=ae[j], ar=ae[j+1];
        if(!upwind_dissipation && std::fabs(ar-al)>alpha_jump_tol) {
            const double pl=pe[j], pr=pe[j+1], ul=ue[j], ur=ue[j+1];
            const double Zl=phase_acoustic(e1,e2,al,t1e[j],t2e[j],pl,0.0).Z;
            const double Zr=phase_acoustic(e1,e2,ar,t1e[j+1],t2e[j+1],pr,0.0).Z;
            const double d=std::fmax(Zl+Zr,1.e-30);
            pf=(Zr*pl+Zl*pr+Zl*Zr*(ul-ur))/d; uf=(pl-pr+Zl*ul+Zr*ur)/d;
        }
        out.p[f]=pf; out.u[f]=uf;
    }
    if (upwind_dissipation && dissipation>0.0) {
        for (int f=0; f<=n; ++f) {
            const int j=bih?f+1:f;
            const double sign=out.u[f]>=0.0 ? 1.0 : -1.0;
            out.p[f]-=dissipation*.5*sign*(pe[j+1]-pe[j]);
            out.u[f]-=dissipation*.5*sign*(ue[j+1]-ue[j]);
        }
        for (int f=0; f<=n; ++f) {
            const int j=bih?f+1:f;
            const double al=ae[j], ar=ae[j+1];
            if (std::fabs(ar-al)<=alpha_jump_tol) continue;
            const double pl=pe[j], pr=pe[j+1], ul=ue[j], ur=ue[j+1];
            const double Zl=phase_acoustic(e1,e2,al,t1e[j],t2e[j],pl,0.0).Z;
            const double Zr=phase_acoustic(e1,e2,ar,t1e[j+1],t2e[j+1],pr,0.0).Z;
            const double den=std::fmax(Zl+Zr,1.e-30);
            out.p[f]=(Zr*pl+Zl*pr+Zl*Zr*(ul-ur))/den;
            out.u[f]=(pl-pr+Zl*ul+Zr*ur)/den;
        }
    }
    return out;
}

inline ImplicitDivergences implicit_divergences(const StepResult& W, const EOS& e1, const EOS& e2,
                                                double dx, BC5 l, BC5 r, double dissipation=0.0,
                                                bool biharmonic=true, bool rhie_chow=false,
                                                double gamma_dt=0.0,
                                                bool acoustic_riemann=false,
                                                bool upwind_dissipation=false,
                                                double compact_lap_coeff=0.0) {
    ImplicitDivergences out; out.face=implicit_face_pu(
        W,e1,e2,l,r,dissipation,biharmonic,rhie_chow,gamma_dt,dx,1.e-8,
        acoustic_riemann,upwind_dissipation);
    const int n=(int)W.p.size(); out.grad_p.resize(n);out.div_u.resize(n);out.div_pu.resize(n);
    for(int i=0;i<n;++i) { out.grad_p[i]=(out.face.p[i+1]-out.face.p[i])/dx; out.div_u[i]=(out.face.u[i+1]-out.face.u[i])/dx; out.div_pu[i]=(out.face.p[i+1]*out.face.u[i+1]-out.face.p[i]*out.face.u[i])/dx; }
    // residual.py::implicit_divergences: compact periodic Laplacian correction.
    if (compact_lap_coeff != 0.0 && l == BC5::Periodic && r == BC5::Periodic) {
        for (int i = 0; i < n; ++i) {
            const int im1 = (i + n - 1) % n, ip1 = (i + 1) % n;
            out.grad_p[i] += compact_lap_coeff * (W.p[ip1] - 2.0 * W.p[i] + W.p[im1]) / dx;
        }
    }
    return out;
}
} // namespace cfd::five_eq
