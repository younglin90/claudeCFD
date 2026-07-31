#include "cfd/five_eq/pe_correction.hpp"

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
    std::ifstream in(PE_REF);
    if (!in) return 1;
    StepResult W;
    W.alpha.resize(3); W.T1.resize(3); W.T2.resize(3); W.u.resize(3); W.p.resize(3);
    Residual5 raw;
    raw.m1.resize(3); raw.m2.resize(3); raw.mom.resize(3); raw.rhoE.resize(3); raw.alpha.resize(3);
    std::array<std::array<double, 5>, 3> gradient{}, tangent{};
    std::array<double, 3> pi{}, energy{};
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream stream(line);
        int i;
        stream >> i >> W.alpha[i] >> W.T1[i] >> W.T2[i] >> W.u[i] >> W.p[i]
               >> raw.m1[i] >> raw.m2[i] >> raw.mom[i] >> raw.rhoE[i] >> raw.alpha[i];
        for (double& value : gradient[i]) stream >> value;
        stream >> pi[i] >> energy[i];
        for (double& value : tangent[i]) stream >> value;
    }

    const auto e1 = EOS::ideal(1.4, 717.5);
    const auto e2 = EOS::nasg(1.187, 7.028e8, 3610., 6.61e-4, -1.177788e6);
    double worst = 0.0, gradient_worst = 0.0, energy_worst = 0.0, tangent_worst = 0.0;
    for (int i = 0; i < 3; ++i) {
        std::array<double, 5> actual{};
        if (!dpdU({W.alpha[i], W.T1[i], W.T2[i], W.u[i], W.p[i]}, e1, e2, actual)) return 2;
        for (int k = 0; k < 5; ++k) {
            const double error = std::fabs(actual[k] - gradient[i][k]) /
                std::fmax(std::fabs(gradient[i][k]), 1.0);
            gradient_worst = std::fmax(gradient_worst, error);
            worst = std::fmax(worst, error);
        }
    }
    Residual5 energy_corrected = raw;
    std::vector<double> actual_pi;
    apply_pe_energy_correction(energy_corrected, W, e1, e2, &actual_pi);
    Residual5 tangent_corrected = raw;
    apply_pe_tangent_projection(tangent_corrected, W, e1, e2);
    for (int i = 0; i < 3; ++i) {
        const double pi_error = std::fabs(actual_pi[i] - pi[i]) / std::fmax(std::fabs(pi[i]), 1.0);
        const double energy_error = std::fabs(energy_corrected.rhoE[i] - energy[i]) /
            std::fmax(std::fabs(energy[i]), 1.0);
        energy_worst = std::fmax(energy_worst, std::fmax(pi_error, energy_error));
        worst = std::fmax(worst, std::fmax(pi_error, energy_error));
        const std::array<double, 5> actual{{tangent_corrected.m1[i], tangent_corrected.m2[i],
            tangent_corrected.mom[i], tangent_corrected.rhoE[i], tangent_corrected.alpha[i]}};
        for (int k = 0; k < 5; ++k) {
            const double error = std::fabs(actual[k] - tangent[i][k]) /
                std::fmax(std::fabs(tangent[i][k]), 1.0);
            tangent_worst = std::fmax(tangent_worst, error);
            worst = std::fmax(worst, error);
        }
    }
    std::printf("pe correction oracle max %.3e [dpdU %.3e energy %.3e tangent %.3e]\n",
        worst, gradient_worst, energy_worst, tangent_worst);
    return worst <= 3.e-7 ? 0 : 3;
}
