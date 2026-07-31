#include "cfd/five_eq/implicit_faces.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace cfd;
using namespace cfd::five_eq;

int main() {
    std::ifstream input(ACOUSTIC_RIEMANN_REF);
    if (!input) return 1;
    std::vector<double> p_ref, u_ref;
    std::string line;
    while (std::getline(input,line)) {
        if (line.empty() || line[0]=='#') continue;
        std::istringstream row(line); int i; double p,u; row>>i>>p>>u;
        p_ref.push_back(p); u_ref.push_back(u);
    }
    const auto eos1=EOS::ideal(1.4,717.5);
    const auto eos2=EOS::nasg(1.187,7.028e8,3610.,6.61e-4,-1.177788e6);
    const double T1=eos1.temperature(1.157,eos1.energy(1.157,1.e5));
    const double T2=eos2.temperature(998.,eos2.energy(998.,1.e5));
    StepResult W{{.15,.42,.70,.90},{T1,T1+.3,T1-.2,T1+.5},
                 {T2-.4,T2+.2,T2+.5,T2-.1},{.03,-.02,.01,-.01},
                 {100050.,99980.,100035.,99960.}};
    const auto face=implicit_face_pu(W,eos1,eos2,BC5::Periodic,BC5::Periodic,
                                     .2,true,false,0.,.1,1.e-8,true);
    double worst=0.;
    for (std::size_t i=0;i<p_ref.size();++i) {
        worst=std::fmax(worst,std::fabs(face.p[i]-p_ref[i])/std::fmax(std::fabs(p_ref[i]),1.));
        worst=std::fmax(worst,std::fabs(face.u[i]-u_ref[i])/std::fmax(std::fabs(u_ref[i]),1.));
    }
    std::printf("acoustic-Riemann faces oracle max %.3e\\n",worst);
    return worst<=2.e-12?0:2;
}
