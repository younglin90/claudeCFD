#include "cfd/five_eq/nd_solver.hpp"

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
    std::ifstream in(ND3_REF);
    if (!in) return 1;
    std::vector<PrimND<3>> input(8), reference(8);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream stream(line);
        int i, j, k;
        PrimND<3> initial{}, expected{};
        stream >> i >> j >> k
               >> initial.alpha >> initial.T1 >> initial.T2
               >> initial.velocity[0] >> initial.velocity[1] >> initial.velocity[2] >> initial.p
               >> expected.alpha >> expected.T1 >> expected.T2
               >> expected.velocity[0] >> expected.velocity[1] >> expected.velocity[2] >> expected.p;
        input[(i * 2 + j) * 2 + k] = initial;
        reference[(i * 2 + j) * 2 + k] = expected;
    }

    const auto e1 = EOS::ideal(1.4, 717.5);
    const auto e2 = EOS::nasg(1.187, 7.028e8, 3610., 6.61e-4, -1.177788e6);
    NDOptions<3> options{};
    options.dx = {.1, .1, .1};
    options.boundary.fill(NDBoundary::Periodic);
    const auto result = solve_3d({2, 2, 2}, input, 1.e-8, e1, e2, options, .35, 1.e-8);

    double worst = 0.0;
    for (std::size_t q = 0; q < result.W.size(); ++q) {
        const std::array<std::pair<double, double>, 7> values{{
            {result.W[q].alpha, reference[q].alpha}, {result.W[q].T1, reference[q].T1},
            {result.W[q].T2, reference[q].T2}, {result.W[q].velocity[0], reference[q].velocity[0]},
            {result.W[q].velocity[1], reference[q].velocity[1]},
            {result.W[q].velocity[2], reference[q].velocity[2]}, {result.W[q].p, reference[q].p}}};
        for (const auto& value : values) {
            worst = std::fmax(worst, std::fabs(value.first - value.second) /
                std::fmax(std::fabs(value.second), 1.0));
        }
    }
    std::printf("nd3 tmlpu oracle max %.3e\n", worst);
    return worst <= 1.e-10 ? 0 : 2;
}
