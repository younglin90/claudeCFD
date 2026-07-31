#include "cfd/five_eq/eos_facade.hpp"

#include <cmath>
#include <stdexcept>

using namespace cfd;
using namespace cfd::five_eq;

int main() {
    const EOS ideal = make_eos("gas", {1.4, 0.0, 717.5});
    const EOS sg = make_eos("stiffened", {1.6, 2.e6, 1100.});
    const EOS nasg = make_eos("nasg", {1.187, 7.028e8, 3610., 6.61e-4, -1.177788e6});
    if (ideal.kind != EOS::Ideal || sg.kind != EOS::SG || nasg.kind != EOS::NASG) return 1;
    if (std::fabs(ideal.density(1.e5, 300.) - 1.e5 / (.4 * 717.5 * 300.)) > 1.e-12) return 2;
    const EOSPair pair{ideal, nasg};
    if (pair.names().first != "ideal" || pair.names().second != "nasg") return 3;
    if (!pair.admissible(1., 1000.) || pair.admissible(-1., 1000.) || pair.admissible(1., 1600.)) return 4;
    try {
        make_eos("jwl");
        return 5;
    } catch (const std::invalid_argument&) {
    }
    try {
        pair.assert_admissible(1., 1600.);
        return 6;
    } catch (const std::domain_error&) {
    }
    return 0;
}
