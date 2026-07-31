#include "denner1d/cases.hpp"
#include "denner1d/eos.hpp"
#include "denner1d/numerics.hpp"
#include "denner1d/solver.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* msg) {
    if (!ok) {
        std::cerr << "denner1d_unit FAIL: " << msg << "\n";
        ++failures;
    }
}
}  // namespace

int main() {
    // --- EOS sanity + conservative recovery round-trip -----------------------
    const auto air = denner1d::air_phase();
    const auto props = denner1d::phase_props(100000.0, 300.0, air);
    check(std::isfinite(props.rho) && props.rho > 0.0 && props.c > 0.0, "air phase_props");
    check(std::abs(denner1d::van_leer_phi(1.0) - 1.0) < 1.0e-12, "van_leer_phi(1)=1");

    const auto water = denner1d::water_liquid_phase();
    for (double alpha : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        const double p0 = 8.0e6;
        const double T0 = 360.0;
        const double rho0 = denner1d::mixture_density(p0, T0, alpha, air, water);
        const double e0 = denner1d::mixture_internal_energy_density(p0, T0, alpha, air, water);
        double p = 1.1 * p0;
        double T = 0.9 * T0;
        const bool ok = denner1d::recover_pressure_temperature_from_density_energy(
            rho0, e0, alpha, air, water, p, T);
        check(ok, "recover converged");
        check(std::abs(p - p0) / p0 < 1.0e-9, "recover p");
        check(std::abs(T - T0) / T0 < 1.0e-9, "recover T");
    }

    std::vector<double> a{1.0, 2.0, 3.0};
    auto g = denner1d::apply_ghost(a, "transmissive", "wall", 2, false);
    check(g.size() == 7, "ghost size");
    check(g[0] == 1.0 && g[1] == 1.0 && g[5] == 3.0 && g[6] == 2.0, "ghost values");

    // --- conservative operator invariant: static air-water interface stays static
    // (interface-equilibrium / Collis IEC property). This is the invariant the
    // double-flux recovery broke; the production scheme must hold it to roundoff.
    {
        const auto c01 = denner1d::find_case("01");
        const auto s = denner1d::solve_case(c01);
        double p0 = s.p.empty() ? 0.0 : s.p.front();
        double max_dp = 0.0, max_u = 0.0;
        bool finite = true;
        for (std::size_t i = 0; i < s.x.size(); ++i) {
            max_dp = std::max(max_dp, std::abs(s.p[i] - p0));
            max_u = std::max(max_u, std::abs(s.u[i]));
            finite = finite && std::isfinite(s.p[i]) && std::isfinite(s.u[i]);
        }
        check(finite, "case01 finite");
        check(max_dp < 1.0, "case01 interface-equilibrium pressure (|dp| < 1 Pa)");
        check(max_u < 1.0e-4, "case01 interface-equilibrium velocity (|u| < 1e-4)");
    }

    if (failures == 0) {
        std::cout << "denner1d_unit ok\n";
        return 0;
    }
    std::cerr << "denner1d_unit: " << failures << " failure(s)\n";
    return 1;
}
