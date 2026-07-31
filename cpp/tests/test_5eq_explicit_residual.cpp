#include "cfd/five_eq/face_state.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

int main() {
    using namespace cfd;
    using namespace cfd::five_eq;
    const auto air=EOS::ideal(1.4,717.5);
    const auto water=EOS::nasg(1.187,7.028e8,3610.,6.61e-4,-1.177788e6);
    StepResult W;
    W.alpha={.3,.3,.3,.3}; W.T1={300,300,300,300}; W.T2={300,300,300,300};
    W.u={2,2,2,2}; W.p={1.e5,1.e5,1.e5,1.e5};
    const auto r0=explicit_residual(W,air,water,.1,BC5::Periodic,BC5::Periodic,true);
    for (const auto* v : {&r0.m1,&r0.m2,&r0.mom,&r0.rhoE,&r0.alpha})
        for (double x : *v) if (!std::isfinite(x) || std::fabs(x)>1.e-8) return 1;
    W.alpha[2]=.7; W.T1[2]=320; W.u[2]=-1.;
    const auto r1=explicit_residual(W,air,water,.1,BC5::Reflective,BC5::Transmissive,true);
    for (const auto* v : {&r1.m1,&r1.m2,&r1.mom,&r1.rhoE,&r1.alpha})
        for (double x : *v) if (!std::isfinite(x)) return 2;

    std::ifstream in(EXPLICIT_POSITIVITY_REF);
    if (!in) return 3;
    const double T1=air.temperature(1.157,air.energy(1.157,1.e5));
    const double T2=water.temperature(998.,water.energy(998.,1.e5));
    StepResult P{{.2,.5,.8},{T1,T1+.2,T1-.1},{T2-.2,T2+.3,T2},
                 {.03,-.01,.02},{100030.,99980.,100010.}};
    const auto blended=explicit_residual(P,air,water,.1,BC5::Periodic,BC5::Periodic,
                                         false,false,EnergyForm::Differential,
                                         {},{},{},{},{},true,false,false,.2);
    const auto forced=explicit_residual(P,air,water,.1,BC5::Periodic,BC5::Periodic,
                                        false,false,EnergyForm::Differential,
                                        {},{},{},{},{},false,true,false,0.);
    const auto forced_dt=explicit_residual(P,air,water,.1,BC5::Periodic,BC5::Periodic,
                                           false,false,EnergyForm::Differential,
                                           {},{},{},{},{},false,true,false,.2);
    std::string line; double worst=0.;
    while (std::getline(in,line)) {
        if (line.empty() || line[0]=='#') continue;
        std::istringstream s(line); int kind,i; double ref[5];
        s >> kind >> i >> ref[0] >> ref[1] >> ref[2] >> ref[3] >> ref[4];
        const auto& got=kind==0 ? blended : (kind==1 ? forced : forced_dt);
        const double val[5]={got.m1[i],got.m2[i],got.mom[i],got.rhoE[i],got.alpha[i]};
        for (int k=0;k<5;++k)
            worst=std::fmax(worst,std::fabs(val[k]-ref[k])/std::fmax(std::fabs(ref[k]),1.));
    }
    std::printf("explicit positivity residual max %.3e\n",worst);
    if (worst>2.e-12) return 4;
    std::puts("explicit residual passed");
    return 0;
}
