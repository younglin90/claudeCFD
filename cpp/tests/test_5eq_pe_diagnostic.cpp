#include "cfd/five_eq/pe_diagnostic.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

using namespace cfd;
using namespace cfd::five_eq;

int main() {
    std::ifstream in(PE_DIAG_REF);
    if (!in) return 1;
    std::string line;
    double rho1, rho2, u, alpha, Fq1, Fq2, Falpha, expected_q1, expected_q2;
    PrimW Wn[2]{}, We[2]{};
    double expected_update[2]{};
    int data_row = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream stream(line);
        if (data_row++ == 0) {
            stream >> rho1 >> rho2 >> u >> alpha >> Fq1 >> Fq2 >> Falpha >> expected_q1 >> expected_q2;
        } else {
            int i = 0;
            stream >> i >> Wn[i].alpha1 >> Wn[i].T1 >> Wn[i].T2 >> Wn[i].u >> Wn[i].p
                   >> We[i].alpha1 >> We[i].T1 >> We[i].T2 >> We[i].u >> We[i].p >> expected_update[i];
        }
    }
    const auto face = pe_face_consistency(rho1, rho2, u, Fq1, Fq2, Falpha);
    const auto e1 = EOS::ideal(1.4, 717.5);
    const auto e2 = EOS::nasg(1.187, 7.028e8, 3610., 6.61e-4, -1.177788e6);
    double worst = std::fmax(std::fabs(face.phase1 - expected_q1), std::fabs(face.phase2 - expected_q2));
    for (int i = 0; i < 2; ++i) {
        const double actual = pe_update_residual(Wn[i], We[i], e1, e2);
        worst = std::fmax(worst, std::fabs(actual - expected_update[i]) /
            std::fmax(std::fabs(expected_update[i]), 1.0));
    }
    std::printf("pe diagnostic oracle max %.3e\n", worst);
    // The update identity subtracts O(1e9) phase energies to yield O(10).
    // Python/NumPy and scalar C++ differ by a few ulps in that cancellation.
    return worst <= 3.e-8 ? 0 : 2;
}
