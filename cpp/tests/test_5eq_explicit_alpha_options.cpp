#include "cfd/five_eq/explicit.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main() {
    using namespace cfd;
    using namespace cfd::five_eq;
    const EOS air = EOS::ideal(1.4, 717.5);
    const EOS water = EOS::nasg(1.187, 7.028e8, 3610., 6.61e-4, -1.177788e6);
    const double T1 = air.temperature(1.157, air.energy(1.157, 1.e5));
    const double T2 = water.temperature(998., water.energy(998., 1.e5));
    const std::vector<double> a{.02,.08,.20,.35,.50,.65,.80,.95};
    const std::vector<double> u{.20,.12,.04,-.03,-.08,-.02,.09,.16};
    const std::vector<double> p{100180.,100080.,99970.,99890.,99840.,99960.,100070.,100150.};
    const std::vector<double> T1v(a.size(), T1), T2v(a.size(), T2);
    const std::array<AlphaFaceScheme, 7> schemes{{
        AlphaFaceScheme::Cicsam, AlphaFaceScheme::Stacs, AlphaFaceScheme::Mstacs,
        AlphaFaceScheme::VanLeer, AlphaFaceScheme::AdaptiveBvd, AlphaFaceScheme::Thinc,
        AlphaFaceScheme::ThincBvd}};
    std::array<StepResult, 7> got;
    for (int kind = 0; kind < 7; ++kind) {
        StepConfig cfg;
        cfg.bc_l = BC5::Periodic; cfg.bc_r = BC5::Periodic;
        cfg.kapila_closure = false;
        cfg.explicit_alpha_scheme = schemes[kind];
        got[kind] = explicit_rusanov_step(a, T1v, T2v, u, p, 1.e-7, .125, air, water, cfg);
    }
    std::ifstream in(EXPLICIT_ALPHA_OPTIONS_REF);
    if (!in) return 1;
    std::string line;
    int fail = 0, rows = 0;
    double worst = 0.;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        int kind, i;
        double ref[5];
        std::istringstream stream(line);
        stream >> kind >> i;
        for (double& value : ref) stream >> value;
        const double value[] = {got[kind].alpha[i], got[kind].T1[i], got[kind].T2[i],
                                got[kind].u[i], got[kind].p[i]};
        for (int k = 0; k < 5; ++k) {
            const double rel = std::fabs(value[k] - ref[k]) / std::fmax(std::fabs(ref[k]), 1.);
            worst = std::fmax(worst, rel);
            if (rel > 3.e-10) {
                ++fail;
                std::printf("kind=%d cell=%d field=%d got=%.17g ref=%.17g rel=%.3e\n",
                            kind, i, k, value[k], ref[k], rel);
            }
        }
        ++rows;
    }
    std::printf("explicit alpha options: rows=%d max_rel=%.3e %s\n", rows, worst,
                fail ? "FAIL" : "PASS");
    return fail ? 2 : 0;
}
