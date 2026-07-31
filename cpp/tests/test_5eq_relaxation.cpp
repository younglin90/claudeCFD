#include "cfd/five_eq/relaxation.hpp"
#include <cmath>
#include <cstdio>

int main() {
    const cfd::EOS e1=cfd::EOS::ideal(1.4,717.5);
    const cfd::EOS e2=cfd::EOS::nasg(1.187,7.028e8,3610.,6.61e-4,-1.177788e6);
    cfd::five_eq::StepResult W;
    W.alpha={.2,.7}; W.T1={290.,330.}; W.T2={310.,295.}; W.u={2.,-1.}; W.p={1.e5,1.2e5};
    const auto R=cfd::five_eq::relax_pressure(W,e1,e2);
    for (size_t i=0;i<W.alpha.size();++i) {
        const double r1=e1.density(W.p[i],W.T1[i]), r2=e2.density(W.p[i],W.T2[i]);
        const double E=W.alpha[i]*r1*e1.energy(r1,W.p[i])+(1-W.alpha[i])*r2*e2.energy(r2,W.p[i]);
        const double Enew=W.alpha[i]*r1*e1.energy(r1,R.p[i])+(1-W.alpha[i])*r2*e2.energy(r2,R.p[i]);
        if (R.alpha[i]!=W.alpha[i] || R.u[i]!=W.u[i] || !std::isfinite(R.p[i]) ||
            std::fabs(Enew-E)/std::fmax(std::fabs(E),1.0)>1.e-10) return 1;
    }
    const auto RT=cfd::five_eq::relax_pT(W,e1,e2);
    for (size_t i=0;i<W.alpha.size();++i)
        if (!std::isfinite(RT.p[i]) || !std::isfinite(RT.T1[i]) || std::fabs(RT.T1[i]-RT.T2[i])>1.e-12) return 2;
    std::puts("relaxation passed");
    return 0;
}
