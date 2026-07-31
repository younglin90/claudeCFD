#include "cfd/five_eq/face_state.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace cfd; using namespace cfd::five_eq;

int main() {
    std::ifstream in(FACE_OPTIONS_REF); if (!in) return 1;
    const auto air=EOS::ideal(1.4,717.5);
    const auto water=EOS::nasg(1.187,7.028e8,3610.,6.61e-4,-1.177788e6);
    const double t1=air.temperature(1.157,air.energy(1.157,1.e5));
    const double t2=water.temperature(998.,water.energy(998.,1.e5));
    StepResult W{{.08,.36,.72,.91,.46},{t1-.4,t1+.6,t1-.2,t1+.8,t1+.1},
                 {t2+.3,t2-.5,t2+.2,t2-.1,t2+.6},{.04,-.03,.01,.05,-.02},
                 {100040.,99970.,100015.,100070.,99975.}};
    FaceStateOptions o0; o0.alpha_scheme=AlphaFaceScheme::Muscl;
    o0.primitive_scheme=PrimitiveFaceScheme::Superbee; o0.up_scheme=UPFaceScheme::Upwind;
    o0.thermo_scheme=FaceThermoScheme::Cell; o0.has_dt_dx=true; o0.dt=.02; o0.dx=.1;
    FaceStateOptions o1; o1.alpha_scheme=AlphaFaceScheme::Cicsam;
    o1.primitive_scheme=PrimitiveFaceScheme::Superbee; o1.has_dt_dx=true; o1.dt=.02; o1.dx=.1;
    FaceStateOptions o2=o1; o2.alpha_scheme=AlphaFaceScheme::AdaptiveBvd;
    FaceStateOptions o3; o3.primitive_scheme=PrimitiveFaceScheme::Weno3;
    FaceStateOptions o4; o4.alpha_scheme=AlphaFaceScheme::Stacs;
    FaceStateOptions o5=o1; o5.alpha_scheme=AlphaFaceScheme::Mstacs; o5.primitive_scheme=PrimitiveFaceScheme::Upwind;
    FaceStateOptions o6; o6.alpha_scheme=AlphaFaceScheme::VanLeer;
    FaceStateOptions o7; o7.alpha_scheme=AlphaFaceScheme::Thinc;
    FaceStateOptions o8=o1; o8.alpha_scheme=AlphaFaceScheme::ThincBvd; o8.primitive_scheme=PrimitiveFaceScheme::Upwind;
    std::vector<FaceState> states;
    for(const auto& o : {o0,o1,o2,o3,o4,o5,o6,o7,o8})
        states.push_back(acid_face_state(W,air,water,BC5::Periodic,BC5::Periodic,{},{},{},{},{},o));
    std::string line; double worst=0.;
    while(std::getline(in,line)) {
        if(line.empty() || line[0]=='#') continue;
        std::istringstream s(line); int kind,i; double ref[11];
        s>>kind>>i; for(double& x:ref) s>>x;
        const auto& q=states.at(kind);
        const double got[11]={q.energy.alpha[i],q.energy.T1[i],q.energy.T2[i],q.energy.u[i],q.energy.p[i],
                              q.energy.rho1[i],q.energy.rho2[i],q.energy.e1[i],q.energy.e2[i],q.c1_sq[i],q.c2_sq[i]};
        for(int k=0;k<11;++k)
            worst=std::fmax(worst,std::fabs(got[k]-ref[k])/std::fmax(std::fabs(ref[k]),1.));
    }
    std::printf("face option max %.3e\n",worst);
    return worst<=3e-12 ? 0 : 2;
}
