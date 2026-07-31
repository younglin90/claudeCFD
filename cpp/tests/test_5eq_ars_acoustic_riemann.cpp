#include "cfd/five_eq/ars_solver.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace cfd;
using namespace cfd::five_eq;

int main() {
    std::ifstream input(ARS_ACOUSTIC_RIEMANN_REF);
    if (!input) return 1;
    StepResult initial;
    std::vector<PrimW> expected;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        int i; PrimW current{}, result{};
        row >> i >> current.alpha1 >> current.T1 >> current.T2 >> current.u >> current.p
            >> result.alpha1 >> result.T1 >> result.T2 >> result.u >> result.p;
        initial.alpha.push_back(current.alpha1); initial.T1.push_back(current.T1);
        initial.T2.push_back(current.T2); initial.u.push_back(current.u); initial.p.push_back(current.p);
        expected.push_back(result);
    }
    const auto eos1=EOS::ideal(1.4,717.5);
    const auto eos2=EOS::nasg(1.187,7.028e8,3610.,6.61e-4,-1.177788e6);
    StepConfig config; config.bc_l=BC5::Periodic; config.bc_r=BC5::Periodic;
    config.ars_implicit_dissipation_form=ImplicitDissipationForm::AcousticRiemann;
    config.ars_implicit_dissipation=.2;
    NewtonInfo stage2, stage3;
    const auto got=ars222_step(initial,1.e-8,.1,eos1,eos2,config,&stage2,&stage3);
    double worst=0.;
    for (std::size_t i=0; i<expected.size(); ++i) {
        const double actual[5]={got.alpha[i],got.T1[i],got.T2[i],got.u[i],got.p[i]};
        const double reference[5]={expected[i].alpha1,expected[i].T1,expected[i].T2,
                                   expected[i].u,expected[i].p};
        for (int k=0; k<5; ++k) worst=std::fmax(worst,std::fabs(actual[k]-reference[k])/
            std::fmax(std::fabs(reference[k]),1.));
    }
    std::printf("ARS222 acoustic-Riemann oracle max %.3e; Newton=(%d,%d) (%d,%d)\\n",worst,
                stage2.converged?1:0,stage2.iterations,stage3.converged?1:0,stage3.iterations);
    return stage2.converged && stage3.converged && worst<=2.e-5 ? 0 : 2;
}
