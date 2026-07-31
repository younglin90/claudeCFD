#include "cfd/five_eq/implicit_faces.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#ifndef COMPACT_IMPLICIT_REF
#  define COMPACT_IMPLICIT_REF "compact_implicit_ref.txt"
#endif

int main() {
    using namespace cfd;
    using namespace cfd::five_eq;
    const EOS e1 = EOS::ideal(1.4, 717.5);
    const EOS e2 = EOS::nasg(1.187, 7.028e8, 3610., 6.61e-4, -1.177788e6);
    const double T1=e1.temperature(1.157,e1.energy(1.157,1.e5));
    const double T2=e2.temperature(998.,e2.energy(998.,1.e5));
    const StepResult W{{.15,.42,.70,.90},{T1,T1+.3,T1-.2,T1+.5},
                       {T2-.4,T2+.2,T2+.5,T2-.1},{.03,-.02,.01,-.01},
                       {100050.,99980.,100035.,99960.}};
    const auto got=implicit_divergences(W,e1,e2,.1,BC5::Periodic,BC5::Periodic,
                                        .2,true,false,0.,false,false,.125);
    std::ifstream in(COMPACT_IMPLICIT_REF); if (!in) return 1;
    std::string line; int fail=0, rows=0; double worst=0.;
    while (std::getline(in,line)) {
        const auto first=line.find_first_not_of(" \t\r\n");
        if (first==std::string::npos || line[first]=='#') continue;
        std::istringstream s(line); int i; double gp,pu,du;
        if (!(s>>i>>gp>>pu>>du)) continue;
        const double values[]={got.grad_p[i],got.div_pu[i],got.div_u[i]};
        const double refs[]={gp,pu,du};
        for (int k=0;k<3;++k) {
            const double rel=std::fabs(values[k]-refs[k])/std::fmax(std::fabs(refs[k]),1.);
            worst=std::fmax(worst,rel); if(rel>2.e-12) ++fail;
        }
        ++rows;
    }
    std::printf("compact implicit oracle rows=%d max_rel=%.3e %s\n",rows,worst,fail?"FAIL":"PASS");
    return fail?2:0;
}
