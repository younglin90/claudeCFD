// cfd/five_eq/primitive_nd.hpp -- scalar 2-D/3-D primitive/conservative transforms.
// Port of five_eq_IMEX/nd_primitive.py; field solvers store one of these per cell.
#pragma once

#include <array>
#include <cmath>

#include "cfd/eos.hpp"

namespace cfd::five_eq {

template <int Dim> struct PrimND {
    static_assert(Dim == 2 || Dim == 3, "five-equation ND supports 2D or 3D");
    double alpha, T1, T2;
    std::array<double, Dim> velocity;
    double p;
};

template <int Dim> struct ConsND {
    double m1, m2;
    std::array<double, Dim> momentum;
    double rhoE, alpha;
};

template <int Dim>
inline ConsND<Dim> prim_to_cons_nd(const PrimND<Dim>& W, const EOS& e1, const EOS& e2,
                                   double alpha_eps = 1.e-12) {
    const double a = std::fmin(std::fmax(W.alpha, alpha_eps), 1.0-alpha_eps), a2=1.0-a;
    const double r1=e1.density(W.p,W.T1), r2=e2.density(W.p,W.T2);
    const double rho=a*r1+a2*r2;
    double speed2=0.0; for (double v:W.velocity) speed2+=v*v;
    ConsND<Dim> U{}; U.m1=a*r1; U.m2=a2*r2; U.alpha=W.alpha;
    for (int d=0;d<Dim;++d) U.momentum[d]=rho*W.velocity[d];
    U.rhoE=a*r1*e1.energy(r1,W.p)+a2*r2*e2.energy(r2,W.p)+.5*rho*speed2;
    return U;
}

template <int Dim>
inline PrimND<Dim> cons_to_prim_nd(const ConsND<Dim>& U, const EOS& e1, const EOS& e2,
                                   const PrimND<Dim>* seed = nullptr,
                                   double alpha_eps=1.e-12, double p_floor=1.e-8,
                                   int p_max_iter=24) {
    constexpr double eps=1.e-30;
    const double a=std::fmin(std::fmax(U.alpha,alpha_eps),1.0-alpha_eps), a2=1.0-a;
    const double q1=std::fmax(U.m1,eps), q2=std::fmax(U.m2,eps), rho=std::fmax(q1+q2,eps);
    const double r1=q1/a, r2=q2/a2;
    PrimND<Dim> W{}; W.alpha=a;
    double kinetic=0.0;
    for (int d=0;d<Dim;++d) { W.velocity[d]=U.momentum[d]/rho; kinetic+=W.velocity[d]*W.velocity[d]; }
    const double rhoe=std::fmax(U.rhoE-.5*rho*kinetic,eps);
    double p=seed ? seed->p : std::fmax(.4*rhoe,1.0);
    p=std::fmax(p,p_floor);
    for (int it=0;it<p_max_iter;++it) {
        const double en1=e1.energy(r1,p), en2=e2.energy(r2,p);
        const double f=a*r1*en1+a2*r2*en2-rhoe;
        const double de1=1.0/std::fmax((e1.gamma-1.0)*r1,eps);
        const double de2=1.0/std::fmax((e2.gamma-1.0)*r2/(e2.kind==EOS::NASG ? std::fmax(1.0-e2.b*r2,1.e-10) : 1.0),eps);
        const double dp=f/std::fmax(a*r1*de1+a2*r2*de2,eps);
        const double pn=std::fmax(p-dp,p_floor);
        p=pn;
        if (std::fabs(dp)/std::fmax(std::fabs(p),1.0)<1.e-11) break;
    }
    W.p=p; W.T1=e1.temperature(r1,e1.energy(r1,p)); W.T2=e2.temperature(r2,e2.energy(r2,p));
    return W;
}

template <int Dim>
inline ConsND<Dim> clip_cons_alpha_nd(ConsND<Dim> U, double alpha_eps=1.e-12) {
    U.alpha=std::fmin(std::fmax(U.alpha,alpha_eps),1.0-alpha_eps); return U;
}

template <int Dim>
inline std::array<double,3> mixture_density_from_W(
        const PrimND<Dim>& W, const EOS& e1, const EOS& e2, double alpha_eps=1.e-12) {
    const double a=std::fmin(std::fmax(W.alpha,alpha_eps),1.0-alpha_eps);
    const double rho1=e1.density(W.p,W.T1), rho2=e2.density(W.p,W.T2);
    return {a*rho1+(1.0-a)*rho2,rho1,rho2};
}

template <int Dim>
inline ConsND<Dim> clip_cons_alpha(ConsND<Dim> U, double alpha_eps=1.e-12) {
    return clip_cons_alpha_nd(U,alpha_eps);
}

} // namespace cfd::five_eq
