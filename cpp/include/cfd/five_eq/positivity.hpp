// Phase-8 conservative high/low flux blending limiter.
#pragma once

#include <algorithm>
#include <vector>

#include "cfd/primitive.hpp"

namespace cfd::five_eq {

struct AdvectiveFlux5 {
    std::vector<double> m1, m2, mom, rhoE, alpha;
};

inline std::vector<double> positivity_blend_theta(
        const AdvectiveFlux5& high, const AdvectiveFlux5& low,
        const std::vector<ConsU>& state, double dx, double dt,
        double phase_mass_floor=1.e-10, double alpha_floor=1.e-6,
        int max_iter=30) {
    const int n=static_cast<int>(state.size());
    std::vector<double> theta(n+1,1.0), next(n+1);
    for (int it=0; it<max_iter; ++it) {
        std::vector<char> bad(n,0); bool any=false;
        for (int i=0; i<n; ++i) {
            const auto flux=[&](const std::vector<double>& h,const std::vector<double>& l,int f) {
                return theta[f]*h[f]+(1.0-theta[f])*l[f]; };
            const double q1=state[i].m1-dt*(flux(high.m1,low.m1,i+1)-flux(high.m1,low.m1,i))/dx;
            const double q2=state[i].m2-dt*(flux(high.m2,low.m2,i+1)-flux(high.m2,low.m2,i))/dx;
            const double a=state[i].a1-dt*(flux(high.alpha,low.alpha,i+1)-flux(high.alpha,low.alpha,i))/dx;
            bad[i]=q1<=phase_mass_floor || q2<=phase_mass_floor || a<=alpha_floor || a>=1.0-alpha_floor;
            any=any||bad[i];
        }
        if (!any) return theta;
        next=theta;
        for (int i=0; i<n; ++i) if (bad[i]) { next[i]*=.5; next[i+1]*=.5; }
        if (next==theta) break;
        theta.swap(next);
    }
    return theta;
}

inline AdvectiveFlux5 blend_advective_fluxes(const AdvectiveFlux5& high,
                                              const AdvectiveFlux5& low,
                                              const std::vector<double>& theta) {
    AdvectiveFlux5 out; const int nf=static_cast<int>(theta.size());
    auto blend=[&](const std::vector<double>& h,const std::vector<double>& l,std::vector<double>& o) {
        o.resize(nf); for(int f=0;f<nf;++f)o[f]=theta[f]*h[f]+(1.-theta[f])*l[f]; };
    blend(high.m1,low.m1,out.m1); blend(high.m2,low.m2,out.m2);
    blend(high.mom,low.mom,out.mom); blend(high.rhoE,low.rhoE,out.rhoE);
    blend(high.alpha,low.alpha,out.alpha); return out;
}
} // namespace cfd::five_eq
