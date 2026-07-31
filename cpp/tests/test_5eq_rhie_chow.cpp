#include "cfd/five_eq/implicit_faces.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace cfd;
using namespace cfd::five_eq;

int main() {
    std::ifstream input(RHIE_CHOW_REF);
    if (!input) return 1;
    std::vector<std::array<double, 5>> expected;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        int i; std::array<double, 5> values{};
        row >> i >> values[0] >> values[1] >> values[2] >> values[3] >> values[4];
        expected.push_back(values);
    }
    const auto eos1=EOS::ideal(1.4,717.5);
    const auto eos2=EOS::nasg(1.187,7.028e8,3610.,6.61e-4,-1.177788e6);
    const double T1=eos1.temperature(1.157,eos1.energy(1.157,1.e5));
    const double T2=eos2.temperature(998.,eos2.energy(998.,1.e5));
    StepResult W{{.25,.45,.65,.80},{T1,T1+.3,T1-.2,T1+.5},
                 {T2-.4,T2+.2,T2+.5,T2-.1},{.03,-.02,.01,-.01},
                 {100050.,99980.,100035.,99960.}};
    const auto d=implicit_divergences(W,eos1,eos2,.1,BC5::Periodic,BC5::Periodic,
                                      0.,true,true,1.e-8);
    double worst=0.;
    for (std::size_t f=0; f<expected.size(); ++f) {
        const int i=static_cast<int>(f % W.p.size());
        const double actual[5]={d.face.p[f],d.face.u[f],d.grad_p[i],d.div_pu[i],d.div_u[i]};
        for (int k=0; k<5; ++k) worst=std::fmax(worst,std::fabs(actual[k]-expected[f][k])/
            std::fmax(std::fabs(expected[f][k]),1.));
    }
    std::printf("Rhie-Chow oracle max %.3e\\n",worst);
    return worst<=2.e-12?0:2;
}
