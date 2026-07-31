#include "cfd/five_eq/solver.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

int main() {
    using namespace cfd;
    using namespace cfd::five_eq;
    const EOS air = EOS::ideal(1.4, 717.5);
    const EOS water = EOS::nasg(1.187, 7.028e8, 3610., 6.61e-4, -1.177788e6);
    const double T1 = air.temperature(1.157, air.energy(1.157, 1.e5));
    const double T2 = water.temperature(998., water.energy(998., 1.e5));
    const std::vector<double> alpha{.35, .50, .65, .45};
    const std::vector<double> u{.03, -.01, .02, 0.};
    const std::vector<double> p{1.e5, 100100., 99940., 100030.};
    const std::vector<double> T1v(4, T1), T2v(4, T2);
    RunConfig config = RunConfig::python_solve_defaults();
    config.step_config.bc_l = BC5::Periodic;
    config.step_config.bc_r = BC5::Periodic;
    config.dt_fixed = 1.e-8;
    const RunResult got = solve(alpha, T1v, T2v, u, p, .1, 1.e-8, air, water, config);
    if (got.termination != RunTermination::completed || got.steps != 1) return 1;
    std::ifstream in(PYTHON_SOLVE_DEFAULTS_REF);
    if (!in) return 1;
    std::string line;
    int fail = 0, rows = 0;
    double worst = 0.;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream s(line);
        int i;
        double initial[5], ref[5];
        s >> i;
        for (double& x : initial) s >> x;
        for (double& x : ref) s >> x;
        const double value[] = {got.W.alpha[i], got.W.T1[i], got.W.T2[i], got.W.u[i], got.W.p[i]};
        for (int k = 0; k < 5; ++k) {
            const double rel = std::fabs(value[k] - ref[k]) / std::fmax(std::fabs(ref[k]), 1.);
            worst = std::fmax(worst, rel);
            if (rel > 3.e-5) ++fail;
        }
        ++rows;
    }
    std::printf("python solve defaults: rows=%d max_rel=%.3e %s\n", rows, worst,
                fail ? "FAIL" : "PASS");
    return fail ? 2 : 0;
}
