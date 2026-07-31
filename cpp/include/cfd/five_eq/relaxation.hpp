// cfd/five_eq/relaxation.hpp -- pressure / pressure-temperature equilibrium projections.
// Scalar-cell port of five_eq_IMEX/relaxation.py.
#pragma once

#include <cmath>
#include <vector>

#include "cfd/eos.hpp"
#include "cfd/five_eq/step.hpp"

namespace cfd::five_eq {
namespace relaxation_detail {
constexpr double eps = 1.e-30;
inline double positive(double x) { return std::fmax(x, eps); }
}

inline StepResult relax_pressure(const StepResult& W, const EOS& e1, const EOS& e2,
                                 int max_iter = 10, double rtol = 1.e-12) {
    using namespace relaxation_detail;
    StepResult out = W;
    const std::size_t n = W.alpha.size();
    for (std::size_t i = 0; i < n; ++i) {
        const double a = W.alpha[i], a2 = 1.0 - a, p0 = W.p[i];
        const double r1 = positive(e1.density(p0, W.T1[i]));
        const double r2 = positive(e2.density(p0, W.T2[i]));
        const double rhoe = a*r1*e1.energy(r1,p0) + a2*r2*e2.energy(r2,p0);
        const double plo = std::fmax(.5*p0, 1.0), phi = 2.0*p0;
        const double e1lo = e1.energy(r1,plo), e1hi = e1.energy(r1,phi);
        const double e2lo = e2.energy(r2,plo), e2hi = e2.energy(r2,phi);
        const double A1 = (e1hi-e1lo)/(phi-plo), B1 = e1lo-A1*plo;
        const double A2 = (e2hi-e2lo)/(phi-plo), B2 = e2lo-A2*plo;
        double p = p0;
        const double linerr1 = std::fabs((A1*p0+B1)-e1.energy(r1,p0))/std::fmax(std::fabs(e1.energy(r1,p0)),1.0);
        const double linerr2 = std::fabs((A2*p0+B2)-e2.energy(r2,p0))/std::fmax(std::fabs(e2.energy(r2,p0)),1.0);
        if (linerr1 < 1.e-10 && linerr2 < 1.e-10) {
            p = std::fmax((rhoe - (a*r1*B1+a2*r2*B2))/positive(a*r1*A1+a2*r2*A2), 1.0);
        } else {
            for (int it=0; it<max_iter; ++it) {
                const double F = a*r1*e1.energy(r1,p) + a2*r2*e2.energy(r2,p) - rhoe;
                const double dp = std::fmax(std::fabs(p)*1.e-7,1.0);
                const double Fp = a*r1*e1.energy(r1,p+dp) + a2*r2*e2.energy(r2,p+dp) - rhoe;
                const double Fm = a*r1*e1.energy(r1,p-dp) + a2*r2*e2.energy(r2,p-dp) - rhoe;
                const double step = F/positive((Fp-Fm)/(2.0*dp));
                const double pn = std::fmax(p-step,1.0);
                p = pn;
                if (std::fabs(step)/std::fmax(std::fabs(p),1.0)<rtol) break;
            }
        }
        out.p[i]=p;
        out.T1[i]=std::fmax(e1.temperature(r1,e1.energy(r1,p)),1.0);
        out.T2[i]=std::fmax(e2.temperature(r2,e2.energy(r2,p)),1.0);
    }
    return out;
}

inline StepResult relax_pT(const StepResult& W, const EOS& e1, const EOS& e2,
                           int max_iter = 15, double rtol = 1.e-12) {
    using namespace relaxation_detail;
    StepResult out = W;
    for (std::size_t i=0;i<W.alpha.size();++i) {
        const double a=W.alpha[i], a2=1.0-a;
        const double r1old=positive(e1.density(W.p[i],W.T1[i]));
        const double r2old=positive(e2.density(W.p[i],W.T2[i]));
        const double m1=a*r1old, m2=a2*r2old;
        const double rhoe=a*r1old*e1.energy(r1old,W.p[i])+a2*r2old*e2.energy(r2old,W.p[i]);
        double p=W.p[i], T=.5*(W.T1[i]+W.T2[i]);
        for (int it=0;it<max_iter;++it) {
            const double r1=positive(e1.density(p,T)), r2=positive(e2.density(p,T));
            const double F1=a*r1-m1;
            const double F3=a*r1*e1.energy(r1,p)+a2*r2*e2.energy(r2,p)-rhoe;
            const double dp=std::fmax(std::fabs(p)*1.e-7,1.0), dT=std::fmax(std::fabs(T)*1.e-7,1.0);
            const double r1p=positive(e1.density(p+dp,T)), r2p=positive(e2.density(p+dp,T));
            const double r1t=positive(e1.density(p,T+dT)), r2t=positive(e2.density(p,T+dT));
            const double F1p=a*r1p-m1, F1t=a*r1t-m1;
            const double F3p=a*r1p*e1.energy(r1p,p+dp)+a2*r2p*e2.energy(r2p,p+dp)-rhoe;
            const double F3t=a*r1t*e1.energy(r1t,p)+a2*r2t*e2.energy(r2t,p)-rhoe;
            const double J11=(F1p-F1)/dp, J12=(F1t-F1)/dT;
            const double J21=(F3p-F3)/dp, J22=(F3t-F3)/dT;
            double det=J11*J22-J12*J21;
            if (std::fabs(det)<eps) det=eps;
            const double dP=(-F1*J22+F3*J12)/det;
            const double dTemp=(-F3*J11+F1*J21)/det;
            p=std::fmax(p+dP,1.0); T=std::fmax(T+dTemp,1.0);
            if (std::fabs(dP)/std::fmax(std::fabs(p),1.0)<rtol && std::fabs(dTemp)/std::fmax(std::fabs(T),1.0)<rtol) break;
        }
        out.p[i]=p; out.T1[i]=T; out.T2[i]=T;
    }
    return out;
}

} // namespace cfd::five_eq
