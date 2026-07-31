#include "cfd/five_eq/ars_schur.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace cfd;
using namespace cfd::five_eq;

int main() {
    std::ifstream input(SCHUR_ITERATION_REF);
    if (!input) return 1;
    StepResult initial;
    std::vector<PrimW> target;
    std::vector<PrimW> expected;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        int i;
        PrimW current{}, anchor{}, result{};
        row >> i >> current.alpha1 >> current.T1 >> current.T2 >> current.u >> current.p
            >> anchor.alpha1 >> anchor.T1 >> anchor.T2 >> anchor.u >> anchor.p
            >> result.alpha1 >> result.T1 >> result.T2 >> result.u >> result.p;
        initial.alpha.push_back(current.alpha1);
        initial.T1.push_back(current.T1);
        initial.T2.push_back(current.T2);
        initial.u.push_back(current.u);
        initial.p.push_back(current.p);
        expected.push_back(result);
        target.push_back(anchor); // Converted after EOS construction.
    }

    const auto eos1 = EOS::ideal(1.4, 717.5);
    const auto eos2 = EOS::nasg(1.187, 7.028e8, 3610., 6.61e-4, -1.177788e6);
    std::vector<ConsU> target_u;
    for (const auto& w : target) target_u.push_back(prim_to_cons_W(w, eos1, eos2));
    const auto got = ars_schur_iteration(initial, target_u, 1.e-8, eos1, eos2, .1);

    double worst = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double actual[5] = {got.alpha[i], got.T1[i], got.T2[i], got.u[i], got.p[i]};
        const double reference[5] = {expected[i].alpha1, expected[i].T1, expected[i].T2,
                                     expected[i].u, expected[i].p};
        for (int k = 0; k < 5; ++k)
            worst = std::fmax(worst, std::fabs(actual[k] - reference[k]) /
                std::fmax(std::fabs(reference[k]), 1.0));
    }
    std::printf("schur iteration oracle max %.3e\n", worst);
    return worst <= 2.e-7 ? 0 : 2;
}
